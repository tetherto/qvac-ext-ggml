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

static void enqueue_copy(dma_queue * queue, void * destination, const void * source, uint32_t bytes) {
    while (!dma_queue_push_single_1d(queue, dma_make_ptr(destination, source), bytes)) {
        dma_queue_flush(queue);
    }
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
    float * staged_weight = (float *)(octx->ctx->vtcm_base + (size_t)ith * l->worker_bytes);
    float * staged_bias = staged_weight + l->coefficient_floats;
    float * staged_input = staged_bias + l->coefficient_floats;
    float * staged_output = staged_input + l->plane_floats;
    float * scratch = staged_output + l->plane_floats;
    dma_queue * queue = octx->ctx->dma[ith];
    dma_queue_flush(queue);
    const HVX_Vector zero = Q6_V_vzero();
    for (uint32_t plane = ith; plane < l->jobs; plane += nth) {
        const uint32_t channel = plane % l->c;
        const size_t plane_offset = (size_t)plane * l->f * l->t;
        // Stage only valid source bytes. Zero the padded coefficient lanes so
        // every weight/bias vector can be loaded directly from aligned VTCM.
        memset(staged_weight, 0, 2 * l->coefficient_floats * sizeof(float));
        enqueue_copy(queue, staged_weight, aw + (size_t)channel * l->f, l->f * sizeof(float));
        enqueue_copy(queue, staged_bias, ab + (size_t)channel * l->f, l->f * sizeof(float));
        enqueue_copy(queue, staged_input, x + plane_offset, l->plane_bytes);
        enqueue_copy(queue, scratch, slopes + channel, sizeof(float));
        dma_queue_flush(queue);
        const HVX_Vector slope = hvx_vec_splat_f32(scratch[0]);
        for (uint32_t tile = 0; tile < l->tiles; ++tile) {
            const uint32_t start = tile * HTP_AFFINE_F32_LANES;
            const uint32_t valid = l->f - start < HTP_AFFINE_F32_LANES ? l->f - start : HTP_AFFINE_F32_LANES;
            const HVX_Vector weight = hvx_vmem(staged_weight + start);
            const HVX_Vector bias = hvx_vmem(staged_bias + start);
            for (uint32_t t = 0; t < l->t; ++t) {
                const size_t offset = (size_t)t * l->f + start;
                const HVX_Vector value = load_tile(staged_input + offset, valid, scratch);
                // Match the existing CPU and seven-op F32 graph rounding order.
                const HVX_Vector relu = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(value, zero), value, zero);
                const HVX_Vector negative = hvx_vec_sub_f32_f32(value, relu);
                const HVX_Vector prelu = hvx_vec_add_f32_f32(relu, hvx_vec_mul_f32_f32(slope, negative));
                const HVX_Vector affine = hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(value, weight), bias);
                const HVX_Vector result = hvx_vec_add_f32_f32(affine, prelu);
                if (valid == HTP_AFFINE_F32_LANES) {
                    hvx_vmemu(staged_output + offset) = result;
                } else {
                    hvx_vmem(scratch + 32) = result;
                    memcpy(staged_output + offset, scratch + 32, valid * sizeof(float));
                }
            }
        }
        enqueue_copy(queue, output + plane_offset, staged_output, l->plane_bytes);
        dma_queue_flush(queue);
    }
    dma_queue_flush(queue);
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
    if (context.layout.total_bytes > UINTPTR_MAX - (uintptr_t)octx->ctx->vtcm_base) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    for (uint32_t i = 0; i < context.layout.threads; ++i) {
        if (!octx->ctx->dma[i]) {
            return HTP_STATUS_INVAL_PARAMS;
        }
    }
    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }
    worker_pool_run_func(octx->ctx->worker_pool, affine_thread, &context, context.layout.threads);
    return HTP_STATUS_OK;
}
