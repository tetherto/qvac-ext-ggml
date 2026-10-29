#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "energy-ops.h"

_Static_assert(HTP_ENERGY_F32 == HTP_TYPE_F32, "Energy F32 protocol type mismatch");

struct energy_context {
    const uint8_t * source;
    float * destination;
    struct htp_energy_layout layout;
    float reciprocal;
};

static inline HVX_Vector square_f32(HVX_Vector input) {
    // SQR materializes an F32 tensor in the unfused graph. Retain that rounding
    // barrier instead of accumulating the unrounded QF32 product.
    HVX_Vector rounded = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(input, input));
    // XQF generation can fold this conversion into the next ADD. Keep the
    // materialized SQR boundary in a register, as the F32 residual path does.
    __asm__ volatile("" : "+v"(rounded));
    return rounded;
}

static inline HVX_Vector load_partial(const float * source, uint32_t valid) {
    HVX_Vector value = Q6_V_vzero();
    volatile float * lanes = (volatile float *)&value;
    for (uint32_t i = 0; i < valid; ++i) lanes[i] = source[i];
    return value;
}

static inline float square_sum_row(const uint8_t * source, uint32_t width, bool direct_tail) {
    HVX_Vector sum = Q6_V_vzero();
    const uint32_t vectors = width / HTP_ENERGY_LANES;
    const uint32_t tail = width % HTP_ENERGY_LANES;
    _Pragma("unroll(4)")
    for (uint32_t i = 0; i < vectors; ++i) {
        const HVX_Vector input = ((const HVX_UVector *)source)[i];
        sum = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(sum), square_f32(input));
    }
    if (tail) {
        const float * pointer = (const float *)(source + vectors * 128);
        const HVX_Vector input = direct_tail ? *(const HVX_UVector *)pointer : load_partial(pointer, tail);
        const HVX_Vector masked = Q6_V_valign_VVR(input, Q6_V_vzero(), tail * sizeof(float));
        sum = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(sum), square_f32(masked));
    }
    return hvx_vec_get_f32(Q6_Vsf_equals_Vqf32(hvx_vec_reduce_sum_qf32(sum)));
}

static inline void store_rows(float * destination, uint32_t valid, HVX_Vector value) {
    if (valid == HTP_ENERGY_LANES) {
        *(HVX_UVector *)destination = value;
    } else {
        float lanes[HTP_ENERGY_LANES] __attribute__((aligned(128)));
        *(HVX_Vector *)lanes = value;
        volatile float * output = destination;
        for (uint32_t i = 0; i < valid; ++i) output[i] = lanes[i];
    }
}

static void energy_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct energy_context * context = data;
    const struct htp_energy_layout * layout = &context->layout;
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector reciprocal = hvx_vec_splat_f32(context->reciprocal);
    for (uint32_t job = ith; job < layout->jobs; job += nth) {
        HVX_Vector sums;
        uint32_t output_row, valid;
        if (layout->mode == HTP_ENERGY_ROWS) {
            output_row = job * HTP_ENERGY_LANES;
            valid = MIN(HTP_ENERGY_LANES, layout->rows - output_row);
            float rows[HTP_ENERGY_LANES] __attribute__((aligned(128)));
            *(HVX_Vector *)rows = zero;
            for (uint32_t lane = 0; lane < valid; ++lane) {
                const uint32_t row = output_row + lane;
                rows[lane] = square_sum_row(context->source + (size_t)row * layout->row_stride,
                                             layout->width, htp_energy_direct_row_tail(layout, row));
            }
            sums = *(const HVX_Vector *)rows;
        } else {
            const uint32_t tiles = (layout->spatial - 1) / HTP_ENERGY_LANES + 1;
            const uint32_t group = job / tiles;
            const uint32_t spatial = (job % tiles) * HTP_ENERGY_LANES;
            output_row = group * layout->spatial + spatial;
            valid = MIN(HTP_ENERGY_LANES, layout->spatial - spatial);
            sums = zero;
            const uint8_t * base = context->source + (size_t)group * layout->group_stride + spatial * sizeof(float);
            for (uint32_t channel = 0; channel < layout->width; ++channel) {
                const float * pointer = (const float *)(base + (size_t)channel * layout->channel_stride);
                const HVX_Vector input = valid == HTP_ENERGY_LANES ? *(const HVX_UVector *)pointer : load_partial(pointer, valid);
                sums = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(sums), square_f32(input));
            }
            sums = Q6_Vsf_equals_Vqf32(sums);
        }
        const HVX_Vector product = Q6_Vqf32_vmpy_VsfVsf(sums, reciprocal);
        const HVX_Vector result = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(product, zero));
        store_rows(context->destination + output_row, valid, result);
    }
}

int op_energy_mean_f32(struct htp_ops_context * octx) {
    if (!octx || !octx->ctx || !octx->src[0] || !octx->dst || !octx->src[0]->data || !octx->dst->data) return HTP_STATUS_INVAL_PARAMS;
    for (int i = 1; i < HTP_OP_MAX_INPUTS; ++i) if (octx->src[i]) return HTP_STATUS_INVAL_PARAMS;
    struct htp_energy_tensor tensors[2] = {0};
    for (int i = 0; i < 2; ++i) {
        const struct htp_tensor * t = i ? octx->dst : octx->src[0];
        tensors[i].type = t->type;
        tensors[i].size = t->size;
        tensors[i].address = t->data;
        for (int d = 0; d < 4; ++d) { tensors[i].ne[d] = t->ne[d]; tensors[i].nb[d] = t->nb[d]; }
    }
    struct energy_context context = {0};
    if (!htp_energy_validate(&tensors[0], &tensors[1], octx->n_threads, &context.layout)) return HTP_STATUS_INVAL_PARAMS;
    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) return HTP_STATUS_OK;
    context.source = (const uint8_t *)octx->src[0]->data;
    context.destination = (float *)octx->dst->data;
    context.reciprocal = 1.0f / (float)context.layout.width;
    worker_pool_run_func(octx->ctx->worker_pool, energy_thread, &context, context.layout.threads);
    return HTP_STATUS_OK;
}
