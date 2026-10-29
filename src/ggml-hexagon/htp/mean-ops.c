#include "hvx-utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "mean-ops.h"

_Static_assert(HTP_MEAN_TYPE_F32 == HTP_TYPE_F32, "MEAN F32 protocol type mismatch");

struct mean_context {
    const uint8_t * source;
    float * destination;
    struct htp_mean_layout layout;
    uint32_t rows_per_thread;
    float reciprocal;
};

// Preserve SUM_ROWS' vector accumulation/reduction tree. The one unsafe final
// tail is copied into private storage instead of reading outside the tensor.
static inline float mean_sum_bounded(const float * source, uint32_t width) {
    HVX_Vector acc = Q6_V_vzero();
    const uint32_t vectors = width / HTP_MEAN_LANES;
    const uint32_t tail = width % HTP_MEAN_LANES;
    _Pragma("unroll(4)")
    for (uint32_t i = 0; i < vectors; ++i) {
        acc = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(acc), ((const HVX_UVector *)source)[i]);
    }
    if (tail) {
        HVX_Vector padded = Q6_V_vzero();
        volatile float * lanes = (volatile float *)&padded;
        for (uint32_t i = 0; i < tail; ++i) lanes[i] = source[vectors * HTP_MEAN_LANES + i];
        const HVX_Vector input = Q6_V_valign_VVR(padded, Q6_V_vzero(), tail * sizeof(float));
        acc = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(acc), input);
    }
    return hvx_vec_get_f32(Q6_Vsf_equals_Vqf32(hvx_vec_reduce_sum_qf32(acc)));
}

static void mean_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct mean_context * context = data;
    const uint32_t begin = ith * context->rows_per_thread;
    const uint32_t end = MIN(begin + context->rows_per_thread, context->layout.rows);
    const HVX_Vector reciprocal = hvx_vec_splat_f32(context->reciprocal);
    const HVX_Vector zero = Q6_V_vzero();
    for (uint32_t row = begin; row < end;) {
        const uint32_t valid = htp_mean_block_rows(row, end);
        float sums[HTP_MEAN_LANES] __attribute__((aligned(128)));
        *(HVX_Vector *)sums = zero;
        for (uint32_t lane = 0; lane < valid; ++lane) {
            const uint32_t current = row + lane;
            const uint8_t * source = context->source + (size_t)current * context->layout.stride;
            if (htp_mean_direct_tail(&context->layout, current)) {
                sums[lane] = (uintptr_t)source % 128 == 0 ? hvx_reduce_sum_f32_a(source, context->layout.width) :
                                                           hvx_reduce_sum_f32_u(source, context->layout.width);
            } else {
                sums[lane] = mean_sum_bounded((const float *)source, context->layout.width);
            }
        }
        // Every sum is converted to F32 before packing. Preserve the previous
        // SCALE(scale, 0) expression, now for up to 32 independent row results.
        const HVX_Vector product = Q6_Vqf32_vmpy_VsfVsf(*(const HVX_Vector *)sums, reciprocal);
        const HVX_Vector scaled = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(product, zero));
        if (valid == HTP_MEAN_LANES) {
            *(HVX_UVector *)(context->destination + row) = scaled;
        } else {
            *(HVX_Vector *)sums = scaled;
            volatile float * destination = context->destination + row;
            for (uint32_t lane = 0; lane < valid; ++lane) destination[lane] = sums[lane];
        }
        row += valid;
    }
    (void)nth;
}

int op_mean(struct htp_ops_context * octx) {
    if (!octx || !octx->ctx || !octx->src[0] || !octx->dst || !octx->src[0]->data || !octx->dst->data) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    for (int i = 1; i < HTP_OP_MAX_INPUTS; ++i) {
        if (octx->src[i]) return HTP_STATUS_INVAL_PARAMS;
    }
    struct htp_mean_tensor tensors[2] = {0};
    for (int i = 0; i < 2; ++i) {
        const struct htp_tensor * tensor = i ? octx->dst : octx->src[0];
        tensors[i].type = tensor->type;
        tensors[i].size = tensor->size;
        tensors[i].address = tensor->data;
        for (int d = 0; d < 4; ++d) {
            tensors[i].ne[d] = tensor->ne[d];
            tensors[i].nb[d] = tensor->nb[d];
        }
    }
    struct mean_context context = {0};
    if (!htp_mean_validate(&tensors[0], &tensors[1], octx->n_threads, &context.layout)) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) return HTP_STATUS_OK;
    context.source = (const uint8_t *)octx->src[0]->data;
    context.destination = (float *)octx->dst->data;
    context.rows_per_thread = (context.layout.rows - 1) / context.layout.threads + 1;
    context.reciprocal = 1.0f / (float)context.layout.width;
    worker_pool_run_func(octx->ctx->worker_pool, mean_thread, &context, context.layout.threads);
    return HTP_STATUS_OK;
}
