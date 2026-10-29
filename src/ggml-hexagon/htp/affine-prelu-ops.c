#include <string.h>

#include "htp-ctx.h"
#include "affine-prelu-ops.h"
#include "hvx-utils.h"

_Static_assert(HTP_AFFINE_F32_TYPE == HTP_TYPE_F32, "affine/PReLU F32 protocol type mismatch");

struct htp_affine_f32_context {
    struct htp_ops_context * octx;
    struct htp_affine_f32_layout layout;
};

static HVX_Vector load_tile(const float * source, uint32_t valid, float * scratch) {
    if (valid == HTP_AFFINE_F32_LANES) {
        return hvx_vmemu(source);
    }
    memset(scratch, 0, 128);
    memcpy(scratch, source, valid * sizeof(float));
    return hvx_vmem(scratch);
}

static void affine_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_affine_f32_context * context = (const struct htp_affine_f32_context *)data;
    const struct htp_affine_f32_layout * l = &context->layout;
    struct htp_ops_context * octx = context->octx;
    const float * x = (const float *)(uintptr_t)octx->src[0]->data;
    const float * aw = (const float *)(uintptr_t)octx->src[1]->data;
    const float * ab = (const float *)(uintptr_t)octx->src[2]->data;
    const float * slopes = (const float *)(uintptr_t)octx->src[3]->data;
    float * output = (float *)(uintptr_t)octx->dst->data;
    float * scratch = (float *)(octx->ctx->vtcm_base + (size_t)ith * HTP_AFFINE_F32_WORKER_BYTES);
    const HVX_Vector zero = Q6_V_vzero();
    for (uint32_t job = ith; job < l->jobs; job += nth) {
        const uint32_t start = job % l->tiles * HTP_AFFINE_F32_LANES;
        const uint32_t plane = job / l->tiles;
        const uint32_t channel = plane % l->c;
        const uint32_t valid = l->f - start < HTP_AFFINE_F32_LANES ? l->f - start : HTP_AFFINE_F32_LANES;
        const HVX_Vector weight = load_tile(aw + (size_t)channel * l->f + start, valid, scratch);
        const HVX_Vector bias = load_tile(ab + (size_t)channel * l->f + start, valid, scratch + 32);
        const HVX_Vector slope = hvx_vec_splat_f32(slopes[channel]);
        for (uint32_t t = 0; t < l->t; ++t) {
            const size_t offset = ((size_t)plane * l->t + t) * l->f + start;
            const HVX_Vector value = load_tile(x + offset, valid, scratch + 64);
            // Match the existing CPU and seven-op F32 graph rounding order.
            const HVX_Vector relu = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(value, zero), value, zero);
            const HVX_Vector negative = hvx_vec_sub_f32_f32(value, relu);
            const HVX_Vector prelu = hvx_vec_add_f32_f32(relu, hvx_vec_mul_f32_f32(slope, negative));
            const HVX_Vector affine = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(value, weight), bias);
            const HVX_Vector result = hvx_vec_add_f32_f32(affine, prelu);
            if (valid == HTP_AFFINE_F32_LANES) {
                hvx_vmemu(output + offset) = result;
            } else {
                hvx_vmem(scratch + 96) = result;
                memcpy(output + offset, scratch + 96, valid * sizeof(float));
            }
        }
    }
}

int op_affine_prelu(struct htp_ops_context * octx) {
    if (!octx || !octx->ctx || !octx->dst || !octx->dst->data || !octx->ctx->vtcm_base ||
        (uintptr_t)octx->ctx->vtcm_base % 128) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    for (int i = 0; i < HTP_OP_MAX_INPUTS; ++i) {
        if (i < 4 ? (!octx->src[i] || !octx->src[i]->data) : octx->src[i] != NULL) {
            return HTP_STATUS_INVAL_PARAMS;
        }
    }
    struct htp_affine_f32_tensor tensors[5] = {0};
    for (int i = 0; i < 5; ++i) {
        const struct htp_tensor * tensor = i < 4 ? octx->src[i] : octx->dst;
        tensors[i].size = tensor->size;
        tensors[i].address = tensor->data;
        tensors[i].type = tensor->type;
        for (int d = 0; d < 4; ++d) {
            tensors[i].ne[d] = tensor->ne[d];
            tensors[i].nb[d] = tensor->nb[d];
        }
    }
    struct htp_affine_f32_context context = {.octx = octx};
    if (!htp_affine_f32_validate(tensors, octx->n_threads, octx->ctx->vtcm_size, &context.layout)) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }
    worker_pool_run_func(octx->ctx->worker_pool, affine_thread, &context, context.layout.threads);
    return HTP_STATUS_OK;
}
