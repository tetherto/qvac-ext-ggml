#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "gru-ops.h"
#include "hvx-utils.h"
#include "hvx-sigmoid.h"

_Static_assert(HTP_GRU_TYPE_F32 == HTP_TYPE_F32, "GRU F32 type must match the DSP protocol");

struct htp_gru_context {
    struct htp_ops_context * octx;
    struct htp_gru_layout layout;
    const float * weights;
    const float * bias;
    uint8_t * scratch;
    bool reverse;
};

static struct htp_gru_tensor gru_tensor(const struct htp_tensor * t) {
    struct htp_gru_tensor result = {.size = t->size, .address = t->data, .type = t->type};
    for (int i = 0; i < 4; ++i) {
        result.ne[i] = t->ne[i];
        result.nb[i] = t->nb[i];
    }
    return result;
}

static void gru_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_gru_context * g = (const struct htp_gru_context *)data;
    const uint32_t h = g->layout.h, hp = g->layout.padded_h, bcount = g->layout.b;
    const float * gi = (const float *)(uintptr_t)g->octx->src[1]->data;
    float * dst = (float *)(uintptr_t)g->octx->dst->data;
    float * state = (float *)(g->scratch + ith * g->layout.worker_bytes);
    float * gh = state + hp;
    float * input = gh + 3 * hp;
    const HVX_Vector one = hvx_vec_splat_f32(1.0f);
    const HVX_Vector high = hvx_vec_splat_f32(87.0f), low = hvx_vec_splat_f32(-87.0f);
    for (uint32_t b = ith; b < bcount; b += nth) {
        memset(state, 0, g->layout.worker_bytes);
        for (uint32_t s = 0; s < g->layout.l; ++s) {
            const uint32_t t = g->reverse ? g->layout.l - 1 - s : s;
            const float * git = gi + (size_t)3 * h * (b + (size_t)bcount * t);
            // Stage each short/odd gate separately: vector tails never cross
            // tensor bounds or borrow lanes from the next gate/batch.
            for (uint32_t gate = 0; gate < 3; ++gate) {
                memcpy(input + gate * hp, git + gate * h, h * sizeof(float));
                for (uint32_t j = 0; j < hp; j += VLEN_FP32) {
                    HVX_Vector acc = Q6_V_vzero();
                    for (uint32_t k = 0; k < h; ++k) {
                        HVX_Vector w = hvx_vmem(g->weights + ((size_t)gate * h + k) * hp + j);
                        acc = hvx_vec_add_f32_f32(acc, hvx_vec_mul_f32_f32(w, hvx_vec_splat_f32(state[k])));
                    }
                    hvx_vmem(gh + gate * hp + j) = hvx_vec_add_f32_f32(acc, hvx_vmem(g->bias + gate * hp + j));
                }
            }
            // All three recurrent transforms above read the OLD state. Only
            // now update h, with PyTorch's reset-after candidate convention.
            for (uint32_t j = 0; j < hp; j += VLEN_FP32) {
                HVX_Vector r = hvx_vec_fast_sigmoid_f32_guard(
                    hvx_vec_add_f32_f32(hvx_vmem(input + j), hvx_vmem(gh + j)), one, high, low);
                HVX_Vector z = hvx_vec_fast_sigmoid_f32_guard(
                    hvx_vec_add_f32_f32(hvx_vmem(input + hp + j), hvx_vmem(gh + hp + j)), one, high, low);
                HVX_Vector n = hvx_vec_tanh_f32(hvx_vec_add_f32_f32(hvx_vmem(input + 2 * hp + j),
                                                                    hvx_vec_mul_f32_f32(r, hvx_vmem(gh + 2 * hp + j))));
                HVX_Vector hn =
                    hvx_vec_add_f32_f32(n, hvx_vec_mul_f32_f32(z, hvx_vec_sub_f32_f32(hvx_vmem(state + j), n)));
                hvx_vmem(state + j) = hn;
            }
            memcpy(dst + (size_t)h * (b + (size_t)bcount * t), state, h * sizeof(float));
        }
    }
}

static void gru_packed_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_gru_context * g = (const struct htp_gru_context *)data;
    const uint32_t h = g->layout.h, bcount = g->layout.b;
    const uint32_t packed = htp_gru_packed_batches(h, bcount);
    const float * gi = (const float *)(uintptr_t)g->octx->src[1]->data;
    float * dst = (float *)(uintptr_t)g->octx->dst->data;
    float * state = (float *)(g->scratch + ith * g->layout.worker_bytes);
    HVX_Vector * gh = (HVX_Vector *)(state + HTP_GRU_LANES);
    HVX_Vector * input = gh + 3;
    float * staged = (float *)(input + 3);
    uint8_t repl_bytes[VLEN] __attribute__((aligned(VLEN)));
    uint32_t offsets[3][HTP_GRU_LANES] __attribute__((aligned(VLEN)));
    for (uint32_t i = 0; i < VLEN; ++i) {
        repl_bytes[i] = htp_gru_packed_repl_control(h, i);
    }
    for (uint32_t gate = 0; gate < 3; ++gate) {
        for (uint32_t lane = 0; lane < HTP_GRU_LANES; ++lane) {
            offsets[gate][lane] = htp_gru_packed_input_word(h, gate, lane) * sizeof(float);
        }
    }
    const HVX_Vector repl = hvx_vmem(repl_bytes);
    const HVX_Vector one = hvx_vec_splat_f32(1.0f);
    const HVX_Vector high = hvx_vec_splat_f32(87.0f), low = hvx_vec_splat_f32(-87.0f);
    const uint32_t tiles = (bcount + packed - 1) / packed;
    for (uint32_t tile = ith; tile < tiles; tile += nth) {
        const uint32_t b = tile * packed;
        const uint32_t valid = bcount - b < packed ? bcount - b : packed;
        memset(state, 0, g->layout.worker_bytes);
        for (uint32_t s = 0; s < g->layout.l; ++s) {
            const uint32_t t = g->reverse ? g->layout.l - 1 - s : s;
            const float * git = gi + (size_t)3 * h * (b + (size_t)bcount * t);
            // Staging is exactly three vectors for a full tile. A partial
            // tile retains zero-filled unused sequences throughout the run.
            memcpy(staged, git, (size_t)3 * h * valid * sizeof(float));
            for (uint32_t gate = 0; gate < 3; ++gate) {
                Q6_vgather_ARMVw(&input[gate], (size_t)staged, 3 * VLEN - 1, hvx_vmem(offsets[gate]));
            }
            const HVX_Vector old = hvx_vmem(state);
            HVX_Vector acc_r = Q6_V_vzero(), acc_z = Q6_V_vzero(), acc_n = Q6_V_vzero();
            for (uint32_t k = 0; k < h; ++k) {
                // Rotate h[k] to each subgroup's first word, then replicate
                // only inside that subgroup. No lane reads another batch.
                const HVX_Vector hk = Q6_V_vdelta_VV(Q6_V_vror_VR(old, k * sizeof(float)), repl);
                acc_r = hvx_vec_add_f32_f32(acc_r, hvx_vec_mul_f32_f32(hvx_vmem(g->weights + k * HTP_GRU_LANES), hk));
                acc_z =
                    hvx_vec_add_f32_f32(acc_z, hvx_vec_mul_f32_f32(hvx_vmem(g->weights + (h + k) * HTP_GRU_LANES), hk));
                acc_n = hvx_vec_add_f32_f32(
                    acc_n, hvx_vec_mul_f32_f32(hvx_vmem(g->weights + (2 * h + k) * HTP_GRU_LANES), hk));
            }
            gh[0] = hvx_vec_add_f32_f32(acc_r, hvx_vmem(g->bias));
            gh[1] = hvx_vec_add_f32_f32(acc_z, hvx_vmem(g->bias + HTP_GRU_LANES));
            gh[2] = hvx_vec_add_f32_f32(acc_n, hvx_vmem(g->bias + 2 * HTP_GRU_LANES));
            const HVX_Vector r = hvx_vec_fast_sigmoid_f32_guard(hvx_vec_add_f32_f32(input[0], gh[0]), one, high, low);
            const HVX_Vector z = hvx_vec_fast_sigmoid_f32_guard(hvx_vec_add_f32_f32(input[1], gh[1]), one, high, low);
            const HVX_Vector n = hvx_vec_tanh_f32(hvx_vec_add_f32_f32(input[2], hvx_vec_mul_f32_f32(r, gh[2])));
            hvx_vmem(state) = hvx_vec_add_f32_f32(n, hvx_vec_mul_f32_f32(z, hvx_vec_sub_f32_f32(old, n)));
            memcpy(dst + (size_t)h * (b + (size_t)bcount * t), state, (size_t)h * valid * sizeof(float));
        }
    }
}

int op_gru(struct htp_ops_context * octx) {
    if (!octx->src[0] || !octx->src[1] || !octx->src[2] || !octx->dst || !octx->src[0]->data || !octx->src[1]->data ||
        !octx->src[2]->data || !octx->dst->data || !octx->ctx->vtcm_base || (uintptr_t)octx->ctx->vtcm_base % 128)
        return HTP_STATUS_INVAL_PARAMS;
    struct htp_gru_tensor w = gru_tensor(octx->src[0]), gi = gru_tensor(octx->src[1]);
    struct htp_gru_tensor bias = gru_tensor(octx->src[2]), dst = gru_tensor(octx->dst);
    struct htp_gru_layout layout;
    if (!htp_gru_validate(&w, &gi, &bias, &dst, octx->op_params[0], octx->n_threads, octx->ctx->vtcm_size, &layout))
        return HTP_STATUS_INVAL_PARAMS;
    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)
        return HTP_STATUS_OK;
    float * weights = (float *)octx->ctx->vtcm_base;
    float * biases = (float *)(octx->ctx->vtcm_base + layout.weights_bytes);
    memset(weights, 0, layout.weights_bytes + layout.bias_bytes);
    const float * original = (const float *)(uintptr_t)octx->src[0]->data;
    const float * original_bias = (const float *)(uintptr_t)octx->src[2]->data;
    const bool packed = htp_gru_packed_batches(layout.h, layout.b) > 1;
    for (uint32_t gate = 0; gate < 3; ++gate) {
        if (packed) {
            for (uint32_t j = 0; j < HTP_GRU_LANES; ++j) {
                biases[gate * HTP_GRU_LANES + j] = original_bias[gate * layout.h + j % layout.h];
            }
        } else {
            memcpy(biases + gate * layout.padded_h, original_bias + gate * layout.h, layout.h * sizeof(float));
        }
        for (uint32_t k = 0; k < layout.h; ++k)
            for (uint32_t j = 0; j < (packed ? HTP_GRU_LANES : layout.h); ++j)
                weights[((size_t)gate * layout.h + k) * layout.padded_h + j] =
                    original[k + (size_t)layout.h * (gate * layout.h + j % layout.h)];
    }
    struct htp_gru_context context = {.octx = octx,
                                      .layout = layout,
                                      .weights = weights,
                                      .bias = biases,
                                      .scratch = octx->ctx->vtcm_base + layout.weights_bytes + layout.bias_bytes,
                                      .reverse = octx->op_params[0] != 0};
    worker_pool_run_func(octx->ctx->worker_pool, packed ? gru_packed_thread : gru_thread, &context, layout.threads);
    return HTP_STATUS_OK;
}
