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
    for (uint32_t gate = 0; gate < 3; ++gate) {
        memcpy(biases + gate * layout.padded_h, original_bias + gate * layout.h, layout.h * sizeof(float));
        for (uint32_t k = 0; k < layout.h; ++k)
            for (uint32_t j = 0; j < layout.h; ++j)
                weights[((size_t)gate * layout.h + k) * layout.padded_h + j] =
                    original[k + (size_t)layout.h * (gate * layout.h + j)];
    }
    struct htp_gru_context context = {.octx = octx,
                                      .layout = layout,
                                      .weights = weights,
                                      .bias = biases,
                                      .scratch = octx->ctx->vtcm_base + layout.weights_bytes + layout.bias_bytes,
                                      .reverse = octx->op_params[0] != 0};
    worker_pool_run_func(octx->ctx->worker_pool, gru_thread, &context, layout.threads);
    return HTP_STATUS_OK;
}
