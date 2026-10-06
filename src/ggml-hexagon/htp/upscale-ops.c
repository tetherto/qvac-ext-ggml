#include <HAP_farf.h>

#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "hex-dma.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-utils.h"

#define UPSCALE_CHUNK 8192

struct htp_upscale_context {
    struct htp_ops_context * octx;
    uint32_t                 factor[HTP_OP_MAX_DIMS];
    uint32_t                 chunks_per_row;
    uint32_t                 n_items;
    uint32_t                 items_per_thread;
    uint32_t                 in_bytes;
    uint32_t                 plan_bytes;
    uint32_t                 bytes_per_thread;
    HVX_Vector *             phase_offsets;
};

static uint32_t upscale_in_bytes(uint32_t factor) {
    return hex_round_up(((UPSCALE_CHUNK - 1) / factor + 2) * sizeof(float) + VLEN, VLEN);
}

static void upscale_plan_phase(struct htp_upscale_context * u, uint32_t phase) {
    int32_t lanes[VLEN_FP32] __attribute__((aligned(VLEN)));
    for (uint32_t j = 0; j < VLEN_FP32; j++) {
        lanes[j] = (int32_t) (((phase + j) / u->factor[0]) * sizeof(float));
    }
    u->phase_offsets[phase] = *(const HVX_Vector *) lanes;
}

static void upscale_plan(struct htp_upscale_context * u) {
    u->phase_offsets = (HVX_Vector *) u->octx->ctx->vtcm_base;
    for (uint32_t phase = 0; phase < u->factor[0]; phase++) {
        upscale_plan_phase(u, phase);
    }
}

static const uint8_t * upscale_src_row(const struct htp_upscale_context * u, uint32_t row) {
    const struct htp_tensor * src = u->octx->src[0];
    const struct htp_tensor * dst = u->octx->dst;
    const uint32_t            i1  = row % dst->ne[1];
    const uint32_t            i2  = (row / dst->ne[1]) % dst->ne[2];
    const uint32_t            i3  = row / (dst->ne[1] * dst->ne[2]);
    return (const uint8_t *) src->data + (i1 / u->factor[1]) * src->nb[1] + (i2 / u->factor[2]) * src->nb[2] +
           (i3 / u->factor[3]) * src->nb[3];
}

static uint8_t * upscale_dst_row(const struct htp_upscale_context * u, uint32_t row) {
    const struct htp_tensor * dst = u->octx->dst;
    const uint32_t            i1  = row % dst->ne[1];
    const uint32_t            i2  = (row / dst->ne[1]) % dst->ne[2];
    const uint32_t            i3  = row / (dst->ne[1] * dst->ne[2]);
    return (uint8_t *) dst->data + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3];
}

// Output a = q * factor + phase repeats input q, so thirty-two consecutive
// outputs are one word gather at the input of their first element.
static void upscale_gather(const struct htp_upscale_context * u, HVX_Vector * tmp, const uint8_t * in, float * out,
                           uint32_t c0, uint32_t n_out) {
    const uint32_t factor   = u->factor[0];
    const uint32_t first    = c0 / factor;
    const uint32_t step_q   = VLEN_FP32 / factor;
    const uint32_t step_rem = VLEN_FP32 % factor;
    uint32_t       q        = first;
    uint32_t       phase    = c0 - first * factor;
    for (uint32_t i = 0; i < n_out; i += VLEN_FP32) {
        const uint8_t * base   = in + (q - first) * sizeof(float);
        const uint32_t  region = (uint32_t) (in + u->in_bytes - base) - 1;
        Q6_vgather_ARMVw(tmp, (size_t) base, region, u->phase_offsets[phase]);
        *(HVX_Vector *) (out + i) = *tmp;
        q += step_q;
        phase += step_rem;
        if (phase >= factor) {
            phase -= factor;
            q++;
        }
    }
}

static void upscale_item(const struct htp_upscale_context * u, unsigned int ith, uint32_t item) {
    const struct htp_ops_context * octx = u->octx;
    const uint32_t                 row  = item / u->chunks_per_row;
    const uint32_t                 c0   = (item - row * u->chunks_per_row) * UPSCALE_CHUNK;
    const uint32_t                 n    = MIN(UPSCALE_CHUNK, octx->dst->ne[0] - c0);
    const uint32_t                 q0   = c0 / u->factor[0];
    const uint32_t                 n_in = (c0 + n - 1) / u->factor[0] - q0 + 1;

    uint8_t *    base = octx->ctx->vtcm_base + u->plan_bytes + (size_t) ith * u->bytes_per_thread;
    uint8_t *    in   = base;
    HVX_Vector * tmp  = (HVX_Vector *) (base + u->in_bytes);
    float *      out  = (float *) (base + u->in_bytes + VLEN);
    dma_queue *  q    = octx->ctx->dma[ith];

    const size_t in_size = (size_t) n_in * sizeof(float);
    dma_queue_copy_rows(q, dma_make_ptr(in, upscale_src_row(u, row) + (size_t) q0 * sizeof(float)), in_size, in_size,
                        in_size, 1);
    upscale_gather(u, tmp, in, out, c0, n);
    const size_t out_size = (size_t) n * sizeof(float);
    dma_queue_copy_rows(q, dma_make_ptr(upscale_dst_row(u, row) + (size_t) c0 * sizeof(float), out), out_size,
                        out_size, out_size, 1);
}

static void upscale_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_upscale_context * u = (const struct htp_upscale_context *) data;

    const uint32_t start = u->items_per_thread * ith;
    const uint32_t end   = MIN(start + u->items_per_thread, u->n_items);
    for (uint32_t item = start; item < end; item++) {
        upscale_item(u, ith, item);
    }
}

static bool upscale_factors(const struct htp_tensor * src, const struct htp_tensor * dst, uint32_t * factor) {
    for (int d = 0; d < HTP_OP_MAX_DIMS; d++) {
        if (src->ne[d] == 0 || dst->ne[d] % src->ne[d] != 0) {
            return false;
        }
        factor[d] = dst->ne[d] / src->ne[d];
    }
    return factor[0] <= HTP_UPSCALE_MAX_FACTOR;
}

int op_upscale(struct htp_ops_context * octx) {
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;

    struct htp_upscale_context u = { .octx = octx };
    if (src->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 || octx->op_params[0] != HTP_UPSCALE_MODE_NEAREST ||
        src->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) || !upscale_factors(src, dst, u.factor)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    u.in_bytes         = upscale_in_bytes(u.factor[0]);
    u.plan_bytes       = u.factor[0] * VLEN;
    u.bytes_per_thread = u.in_bytes + VLEN + UPSCALE_CHUNK * sizeof(float);
    u.chunks_per_row   = (dst->ne[0] + UPSCALE_CHUNK - 1) / UPSCALE_CHUNK;
    u.n_items          = u.chunks_per_row * dst->ne[1] * dst->ne[2] * dst->ne[3];

    const uint32_t threads = MIN(octx->n_threads, u.n_items);
    if ((octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) || threads == 0) {
        return HTP_STATUS_OK;
    }
    if (u.plan_bytes + (size_t) u.bytes_per_thread * threads > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    upscale_plan(&u);
    u.items_per_thread = (u.n_items + threads - 1) / threads;
    worker_pool_run_func(octx->ctx->worker_pool, upscale_thread, &u, threads);
    return HTP_STATUS_OK;
}
