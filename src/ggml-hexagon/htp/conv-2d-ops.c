#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "conv-2d-ops.h"
#include "hvx-utils.h"

_Static_assert(HTP_CONV_F32_TYPE == HTP_TYPE_F32, "direct convolution F32 protocol type mismatch");

struct htp_conv_f32_context {
    struct htp_ops_context * octx;
    struct htp_conv_f32_layout layout;
    const float * weights;
    uint8_t * scratch;
    struct htp_conv_f32_pointwise_layout pointwise;
    struct htp_conv_f32_spatial_layout spatial;
};

static struct htp_conv_f32_tensor conv_tensor(const struct htp_tensor * t) {
    struct htp_conv_f32_tensor result = {.size = t->size, .address = t->data, .type = t->type};
    for (int i = 0; i < 4; ++i) {
        result.ne[i] = t->ne[i];
        result.nb[i] = t->nb[i];
    }
    return result;
}

static void stage_row(float * stage, const float * source, const struct htp_conv_f32_layout * l, uint32_t n,
                      uint32_t ci, int64_t iy, int64_t ix, uint32_t span) {
    memset(stage, 0, HTP_CONV_F32_STAGE_FLOATS * sizeof(float));
    if (iy < 0 || iy >= l->h) {
        return;
    }
    uint32_t first, target, count;
    htp_conv_f32_row_window(l->w, ix, span, &first, &target, &count);
    if (count) {
        const float * row = source + (size_t)l->plane * (ci + (size_t)l->ci * n) + (size_t)iy * l->w;
        memcpy(stage + target, row + first, count * sizeof(float));
    }
}

// CHANNELS is compile-time 1 or 4: real OC=1 convolutions must not pay for
// three dummy output channels. Fixed small loops also keep sums in registers.
#define DEFINE_CONV_TILE(NAME, CHANNELS)                                                                               \
    static void NAME(const struct htp_conv_f32_context * g, uint32_t n, uint32_t oc, uint32_t oy, uint32_t start,      \
                     uint32_t valid, uint32_t valid_oc, float * stage, HVX_Vector * tmp) {                             \
        const struct htp_conv_f32_layout * l = &g->layout;                                                             \
        const float * source = (const float *)(uintptr_t)g->octx->src[1]->data;                                        \
        float * output = (float *)(uintptr_t)g->octx->dst->data;                                                       \
        HVX_Vector sums[CHANNELS];                                                                                     \
        _Pragma("unroll") for (uint32_t c = 0; c < CHANNELS; ++c) sums[c] = Q6_V_vzero();                              \
        if (l->flat_pointwise) {                                                                                       \
            for (uint32_t ci = 0; ci < l->ci; ++ci) {                                                                  \
                const float * input = source + (size_t)l->plane * (ci + (size_t)l->ci * n) + start;                    \
                HVX_Vector values;                                                                                     \
                if (valid == HTP_CONV_F32_LANES) {                                                                     \
                    values = hvx_vmemu(input);                                                                         \
                } else {                                                                                               \
                    memset(stage, 0, HTP_CONV_F32_LANES * sizeof(float));                                              \
                    memcpy(stage, input, valid * sizeof(float));                                                       \
                    values = hvx_vmem(stage);                                                                          \
                }                                                                                                      \
                _Pragma("unroll") for (uint32_t c = 0; c < CHANNELS; ++c) {                                            \
                    HVX_Vector weight = hvx_vec_splat_f32(g->weights[ci + (size_t)l->k_elements * (oc + c)]);          \
                    sums[c] = hvx_vec_add_f32_f32(sums[c], hvx_vec_mul_f32_f32(values, weight));                       \
                }                                                                                                      \
            }                                                                                                          \
        } else {                                                                                                       \
            uint32_t offsets[HTP_CONV_F32_LANES] __attribute__((aligned(128)));                                        \
            for (uint32_t lane = 0; lane < HTP_CONV_F32_LANES; ++lane)                                                 \
                offsets[lane] = lane * l->sx * sizeof(float);                                                          \
            const HVX_Vector gather_offsets = hvx_vmem(offsets);                                                       \
            const uint32_t region = ((HTP_CONV_F32_LANES - 1) * l->sx + 1) * sizeof(float) - 1;                        \
            const uint32_t span = (valid - 1) * l->sx + (l->kw - 1) * l->dx + 1;                                       \
            const int64_t ix = (int64_t)start * l->sx - l->px;                                                         \
            for (uint32_t ci = 0; ci < l->ci; ++ci) {                                                                  \
                for (uint32_t ky = 0; ky < l->kh; ++ky) {                                                              \
                    const int64_t iy = (int64_t)oy * l->sy + ky * l->dy - l->py;                                       \
                    stage_row(stage, source, l, n, ci, iy, ix, span);                                                  \
                    for (uint32_t kx = 0; kx < l->kw; ++kx) {                                                          \
                        HVX_Vector values;                                                                             \
                        if (l->sx == 1) {                                                                              \
                            values = hvx_vmemu(stage + kx * l->dx);                                                    \
                        } else {                                                                                       \
                            Q6_vgather_ARMVw(tmp, (size_t)(stage + kx * l->dx), region, gather_offsets);               \
                            values = *tmp;                                                                             \
                        }                                                                                              \
                        const uint32_t k = (ci * l->kh + ky) * l->kw + kx;                                             \
                        _Pragma("unroll") for (uint32_t c = 0; c < CHANNELS; ++c) {                                    \
                            HVX_Vector weight = hvx_vec_splat_f32(g->weights[k + (size_t)l->k_elements * (oc + c)]);   \
                            sums[c] = hvx_vec_add_f32_f32(sums[c], hvx_vec_mul_f32_f32(values, weight));               \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        for (uint32_t c = 0; c < valid_oc; ++c) {                                                                      \
            float * dst = output + (size_t)l->out_plane * (oc + c + (size_t)l->co * n) + (size_t)oy * l->ow + start;   \
            if (valid == HTP_CONV_F32_LANES) {                                                                         \
                hvx_vmemu(dst) = sums[c];                                                                              \
            } else {                                                                                                   \
                *tmp = sums[c];                                                                                        \
                memcpy(dst, tmp, valid * sizeof(float));                                                               \
            }                                                                                                          \
        }                                                                                                              \
    }

DEFINE_CONV_TILE(conv_tile_one, 1)
DEFINE_CONV_TILE(conv_tile_four, 4)
#undef DEFINE_CONV_TILE

static void conv_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_conv_f32_context * g = (const struct htp_conv_f32_context *)data;
    const struct htp_conv_f32_layout * l = &g->layout;
    float * stage = (float *)(g->scratch + (size_t)ith * l->worker_bytes);
    HVX_Vector * tmp = (HVX_Vector *)(stage + HTP_CONV_F32_STAGE_FLOATS);
    for (uint32_t job = ith; job < l->jobs; job += nth) {
        const uint32_t start = job % l->tiles * HTP_CONV_F32_LANES;
        uint32_t rest = job / l->tiles;
        const uint32_t oy = l->flat_pointwise ? 0 : rest % l->oh;
        if (!l->flat_pointwise)
            rest /= l->oh;
        const uint32_t oc = rest % l->oc_blocks * HTP_CONV_F32_OC_BLOCK;
        const uint32_t n = rest / l->oc_blocks;
        const uint32_t remaining = (l->flat_pointwise ? l->out_plane : l->ow) - start;
        const uint32_t valid = remaining < HTP_CONV_F32_LANES ? remaining : HTP_CONV_F32_LANES;
        const uint32_t valid_oc = l->co - oc < HTP_CONV_F32_OC_BLOCK ? l->co - oc : HTP_CONV_F32_OC_BLOCK;
        if (valid_oc == 1) {
            conv_tile_one(g, n, oc, oy, start, valid, valid_oc, stage, tmp);
        } else {
            conv_tile_four(g, n, oc, oy, start, valid, valid_oc, stage, tmp);
        }
    }
}

// Keep the same one/four-channel specializations as the original kernel.
#define DEFINE_POINTWISE_TILE(NAME, CHANNELS)                                                                          \
    static void NAME(const struct htp_conv_f32_context * g, uint32_t oc, const float * input, float * output) {         \
        const struct htp_conv_f32_layout * l = &g->layout;                                                             \
        HVX_Vector sums[CHANNELS];                                                                                     \
        _Pragma("unroll") for (uint32_t c = 0; c < CHANNELS; ++c) sums[c] = Q6_V_vzero();                              \
        for (uint32_t ci = 0; ci < l->ci; ++ci) {                                                                      \
            const HVX_Vector values = hvx_vmem(input + (size_t)ci * HTP_CONV_F32_LANES);                               \
            _Pragma("unroll") for (uint32_t c = 0; c < CHANNELS; ++c) {                                                \
                const HVX_Vector weight = hvx_vec_splat_f32(g->weights[ci + (size_t)l->k_elements * (oc + c)]);      \
                sums[c] = hvx_vec_add_f32_f32(sums[c], hvx_vec_mul_f32_f32(values, weight));                           \
            }                                                                                                          \
        }                                                                                                              \
        _Pragma("unroll") for (uint32_t c = 0; c < CHANNELS; ++c)                                                     \
            hvx_vmem(output + c * HTP_CONV_F32_LANES) = sums[c];                                                       \
    }

DEFINE_POINTWISE_TILE(pointwise_tile_one, 1)
DEFINE_POINTWISE_TILE(pointwise_tile_four, 4)
DEFINE_POINTWISE_TILE(pointwise_tile_eight, 8)
#undef DEFINE_POINTWISE_TILE

// Reuse a CI-by-32 spatial tile across every output-channel block. DDR is
// accessed by bounded DMA rows; all vector loads and stores use private VTCM.
static void conv_pointwise_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_conv_f32_context * g = (const struct htp_conv_f32_context *)data;
    const struct htp_conv_f32_layout * l = &g->layout;
    const struct htp_conv_f32_pointwise_layout * p = &g->pointwise;
    const float * source = (const float *)(uintptr_t)g->octx->src[1]->data;
    float * output = (float *)(uintptr_t)g->octx->dst->data;
    float * staged_input = (float *)(g->scratch + (size_t)ith * p->worker_bytes);
    float * staged_output = staged_input + p->input_bytes / sizeof(float);
    dma_queue * queue = g->octx->ctx->dma[ith];
    dma_queue_flush(queue);
    for (uint32_t job = ith; job < p->jobs; job += nth) {
        const uint32_t start = job % l->tiles * HTP_CONV_F32_LANES;
        const uint32_t n = job / l->tiles;
        const uint32_t valid = l->out_plane - start < HTP_CONV_F32_LANES ?
            l->out_plane - start : HTP_CONV_F32_LANES;
        if (valid < HTP_CONV_F32_LANES) memset(staged_input, 0, p->input_bytes);
        const float * input = source + (size_t)n * l->ci * l->plane + start;
        dma_queue_copy_rows(queue, dma_make_ptr(staged_input, input), HTP_CONV_F32_LANES * sizeof(float),
                            (size_t)l->plane * sizeof(float), valid * sizeof(float), l->ci);
        for (uint32_t oc = 0; oc < l->co; oc += HTP_CONV_F32_OC_BLOCK) {
            const uint32_t valid_oc = l->co - oc < HTP_CONV_F32_OC_BLOCK ? l->co - oc : HTP_CONV_F32_OC_BLOCK;
            if (valid_oc == 1) {
                pointwise_tile_one(g, oc, staged_input, staged_output);
            } else {
                pointwise_tile_four(g, oc, staged_input, staged_output);
            }
            float * destination = output + ((size_t)n * l->co + oc) * l->out_plane + start;
            dma_queue_copy_rows(queue, dma_make_ptr(destination, staged_output), (size_t)l->out_plane * sizeof(float),
                                HTP_CONV_F32_LANES * sizeof(float), valid * sizeof(float), valid_oc);
        }
    }
    dma_queue_flush(queue);
}

// Use eight accumulators only for eight real channels. Remainders retain the
// original one/four kernels, including their padded-four weight bounds.
static void conv_pointwise_eight_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_conv_f32_context * g = (const struct htp_conv_f32_context *)data;
    const struct htp_conv_f32_layout * l = &g->layout;
    const struct htp_conv_f32_pointwise_layout * p = &g->pointwise;
    const float * source = (const float *)(uintptr_t)g->octx->src[1]->data;
    float * output = (float *)(uintptr_t)g->octx->dst->data;
    float * staged_input = (float *)(g->scratch + (size_t)ith * p->worker_bytes);
    float * staged_output = staged_input + p->input_bytes / sizeof(float);
    dma_queue * queue = g->octx->ctx->dma[ith];
    dma_queue_flush(queue);
    for (uint32_t job = ith; job < p->jobs; job += nth) {
        const uint32_t start = job % l->tiles * HTP_CONV_F32_LANES;
        const uint32_t n = job / l->tiles;
        const uint32_t valid = l->out_plane - start < HTP_CONV_F32_LANES ?
            l->out_plane - start : HTP_CONV_F32_LANES;
        if (valid < HTP_CONV_F32_LANES) {
            for (uint32_t offset = 0; offset < p->input_bytes; offset += 128)
                hvx_vmem((uint8_t *)staged_input + offset) = Q6_V_vzero();
        }
        const float * input = source + (size_t)n * l->ci * l->plane + start;
        dma_queue_copy_rows(queue, dma_make_ptr(staged_input, input), HTP_CONV_F32_LANES * sizeof(float),
                            (size_t)l->plane * sizeof(float), valid * sizeof(float), l->ci);
        for (uint32_t oc = 0; oc < l->co;) {
            const uint32_t remaining = l->co - oc;
            const uint32_t block = remaining >= 8 ? 8 : HTP_CONV_F32_OC_BLOCK;
            const uint32_t valid_oc = remaining < block ? remaining : block;
            if (block == 8) pointwise_tile_eight(g, oc, staged_input, staged_output);
            else if (valid_oc == 1) pointwise_tile_one(g, oc, staged_input, staged_output);
            else pointwise_tile_four(g, oc, staged_input, staged_output);
            float * destination = output + ((size_t)n * l->co + oc) * l->out_plane + start;
            dma_queue_copy_rows(queue, dma_make_ptr(destination, staged_output), (size_t)l->out_plane * sizeof(float),
                                HTP_CONV_F32_LANES * sizeof(float), valid * sizeof(float), valid_oc);
            oc += valid_oc;
        }
    }
    dma_queue_flush(queue);
}

// Exact source-row intersections only. Whole vectors never cross a source-row
// boundary; volatile F32 stores keep the short remainder free of memcpy calls.
static inline __attribute__((always_inline)) void stage_spatial_row(float * stage, const float * source,
    const struct htp_conv_f32_layout * l, uint32_t row_floats, uint32_t n, uint32_t ci,
    int64_t iy, int64_t ix, uint32_t span) {
    for (uint32_t offset = 0; offset < row_floats; offset += HTP_CONV_F32_LANES)
        hvx_vmem(stage + offset) = Q6_V_vzero();
    if (iy < 0 || iy >= l->h) return;
    uint32_t first, target, count;
    htp_conv_f32_row_window(l->w, ix, span, &first, &target, &count);
    const float * input = source + (size_t)l->plane * (ci + (size_t)l->ci * n) + (size_t)iy * l->w + first;
    uint32_t copied = 0;
    for (; count - copied >= HTP_CONV_F32_LANES; copied += HTP_CONV_F32_LANES)
        hvx_vmemu(stage + target + copied) = hvx_vmemu(input + copied);
    volatile float * tail = stage + target + copied;
    for (uint32_t i = 0; i < count - copied; ++i) tail[i] = input[copied + i];
}

#define DEFINE_SPATIAL_TILE(NAME, CHANNELS)                                                                            \
    static inline __attribute__((always_inline)) void NAME(const struct htp_conv_f32_context * g,                      \
        uint32_t n, uint32_t oc, uint32_t oy, uint32_t start, uint32_t valid, uint32_t valid_oc,                         \
        const float * staged_input, HVX_Vector offsets, HVX_Vector * tmp) {                                            \
        const struct htp_conv_f32_layout * l = &g->layout;                                                             \
        HVX_Vector sums[CHANNELS];                                                                                     \
        _Pragma("unroll") for (uint32_t c = 0; c < CHANNELS; ++c) sums[c] = Q6_V_vzero();                              \
        const uint32_t region = ((HTP_CONV_F32_LANES - 1) * l->sx + 1) * sizeof(float) - 1;                            \
        for (uint32_t ci = 0; ci < l->ci; ++ci) {                                                                      \
            for (uint32_t ky = 0; ky < l->kh; ++ky) {                                                                  \
                const float * row = staged_input + ((size_t)ci * l->kh + ky) * g->spatial.row_floats;                \
                for (uint32_t kx = 0; kx < l->kw; ++kx) {                                                              \
                    HVX_Vector values;                                                                                 \
                    if (l->sx == 1) values = hvx_vmemu(row + kx * l->dx);                                              \
                    else {                                                                                             \
                        Q6_vgather_ARMVw(tmp, (size_t)(row + kx * l->dx), region, offsets);                            \
                        values = *tmp;                                                                                 \
                    }                                                                                                  \
                    const uint32_t k = (ci * l->kh + ky) * l->kw + kx;                                                 \
                    _Pragma("unroll") for (uint32_t c = 0; c < CHANNELS; ++c) {                                        \
                        const HVX_Vector weight = hvx_vec_splat_f32(g->weights[k + (size_t)l->k_elements * (oc + c)]);\
                        sums[c] = hvx_vec_add_f32_f32(sums[c], hvx_vec_mul_f32_f32(values, weight));                   \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        float * output = (float *)(uintptr_t)g->octx->dst->data;                                                       \
        _Pragma("unroll") for (uint32_t c = 0; c < CHANNELS; ++c) if (c < valid_oc) {                                  \
            float * dst = output + (size_t)l->out_plane * (oc + c + (size_t)l->co * n) + (size_t)oy * l->ow + start;   \
            if (valid == HTP_CONV_F32_LANES) hvx_vmemu(dst) = sums[c];                                                  \
            else hvx_vec_store_u(dst, valid * sizeof(float), sums[c]);                                                 \
        }                                                                                                              \
    }

DEFINE_SPATIAL_TILE(spatial_tile_one, 1)
DEFINE_SPATIAL_TILE(spatial_tile_four, 4)
DEFINE_SPATIAL_TILE(spatial_tile_eight, 8)
#undef DEFINE_SPATIAL_TILE

static void conv_spatial_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_conv_f32_context * g = (const struct htp_conv_f32_context *)data;
    const struct htp_conv_f32_layout * l = &g->layout;
    const struct htp_conv_f32_spatial_layout * p = &g->spatial;
    const float * source = (const float *)(uintptr_t)g->octx->src[1]->data;
    float * input = (float *)(g->scratch + (size_t)ith * p->worker_bytes);
    HVX_Vector * tmp = (HVX_Vector *)((uint8_t *)input + p->input_bytes);
    uint32_t indices[HTP_CONV_F32_LANES] __attribute__((aligned(128)));
    for (uint32_t lane = 0; lane < HTP_CONV_F32_LANES; ++lane) indices[lane] = lane * l->sx * sizeof(float);
    const HVX_Vector offsets = hvx_vmem(indices);
    for (uint32_t job = ith; job < p->jobs; job += nth) {
        const uint32_t start = job % l->tiles * HTP_CONV_F32_LANES;
        const uint32_t rest = job / l->tiles;
        const uint32_t oy = rest % l->oh;
        const uint32_t n = rest / l->oh;
        const uint32_t valid = l->ow - start < HTP_CONV_F32_LANES ? l->ow - start : HTP_CONV_F32_LANES;
        const uint32_t span = (valid - 1) * l->sx + (l->kw - 1) * l->dx + 1;
        const int64_t ix = (int64_t)start * l->sx - l->px;
        for (uint32_t ci = 0; ci < l->ci; ++ci) for (uint32_t ky = 0; ky < l->kh; ++ky) {
            const int64_t iy = (int64_t)oy * l->sy + ky * l->dy - l->py;
            stage_spatial_row(input + ((size_t)ci * l->kh + ky) * p->row_floats, source, l, p->row_floats,
                               n, ci, iy, ix, span);
        }
        for (uint32_t oc = 0; oc < l->co;) {
            const uint32_t remaining = l->co - oc;
            const uint32_t block = remaining >= 8 ? 8 : HTP_CONV_F32_OC_BLOCK;
            const uint32_t valid_oc = remaining < block ? remaining : block;
            if (block == 8) spatial_tile_eight(g, n, oc, oy, start, valid, valid_oc, input, offsets, tmp);
            else if (valid_oc == 1) spatial_tile_one(g, n, oc, oy, start, valid, valid_oc, input, offsets, tmp);
            else spatial_tile_four(g, n, oc, oy, start, valid, valid_oc, input, offsets, tmp);
            oc += valid_oc;
        }
    }
}

int op_conv_2d(struct htp_ops_context * octx) {
    if (!octx || !octx->ctx || !octx->src[0] || !octx->src[1] || octx->src[2] || !octx->dst || !octx->src[0]->data ||
        !octx->src[1]->data || !octx->dst->data || !octx->ctx->vtcm_base || (uintptr_t)octx->ctx->vtcm_base % 128) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    struct htp_conv_f32_tensor k = conv_tensor(octx->src[0]), x = conv_tensor(octx->src[1]), y = conv_tensor(octx->dst);
    struct htp_conv_f32_layout layout;
    if (!htp_conv_f32_validate(&k, &x, &y, octx->op_params, octx->n_threads, octx->ctx->vtcm_size, &layout)) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)
        return HTP_STATUS_OK;
    float * weights = (float *)octx->ctx->vtcm_base;
    memset(weights, 0, layout.weights_bytes);
    memcpy(weights, (const void *)(uintptr_t)octx->src[0]->data, layout.kernel_bytes);
    struct htp_conv_f32_context context = {
        .octx = octx, .layout = layout, .weights = weights, .scratch = octx->ctx->vtcm_base + layout.weights_bytes};
    bool staged = htp_conv_f32_plan_pointwise(&layout, octx->ctx->vtcm_size, &context.pointwise);
    bool eight = false;
    struct htp_conv_f32_pointwise_layout pointwise_eight;
    if (staged && htp_conv_f32_plan_pointwise_eight(&layout, octx->ctx->vtcm_size, &pointwise_eight) &&
        pointwise_eight.total_bytes <= UINTPTR_MAX - (uintptr_t)octx->ctx->vtcm_base) {
        context.pointwise = pointwise_eight;
        eight = true;
    }
    if (staged) {
        staged = context.pointwise.total_bytes <= UINTPTR_MAX - (uintptr_t)octx->ctx->vtcm_base;
        for (uint32_t i = 0; staged && i < context.pointwise.threads; ++i) staged = octx->ctx->dma[i] != NULL;
    }
    if (staged) {
        worker_pool_run_func(octx->ctx->worker_pool, eight ? conv_pointwise_eight_thread : conv_pointwise_thread,
                             &context, context.pointwise.threads);
    } else if (htp_conv_f32_plan_spatial(&layout, octx->ctx->vtcm_size, &context.spatial) &&
               context.spatial.total_bytes <= UINTPTR_MAX - (uintptr_t)octx->ctx->vtcm_base) {
        worker_pool_run_func(octx->ctx->worker_pool, conv_spatial_thread, &context, context.spatial.threads);
    } else {
        worker_pool_run_func(octx->ctx->worker_pool, conv_thread, &context, layout.threads);
    }
    return HTP_STATUS_OK;
}
