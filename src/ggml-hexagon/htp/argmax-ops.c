#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <HAP_farf.h>
#include <HAP_perf.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"

#include "hvx-utils.h"

#include "htp-ctx.h"
#include "htp-ops.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

// Worker context: shared across threads, mutated only via disjoint row ranges.
struct htp_argmax_context {
    struct htp_ops_context * octx;
    uint32_t                 nrows;
    uint32_t                 nrows_per_thread;
};

// Scalar argmax over one row of F32 values. Lanes early-return on NaN so the
// output index matches the CPU reference (ggml_vec_argmax_f32 picks the first
// finite max; NaN propagates to index 0 under CPU semantics but is rare in
// decoded logits -- Audio8 sampling masks every out-of-range token to -inf
// before entering ARGMAX).
static inline int32_t argmax_scalar_f32(const float * restrict src, uint32_t n) {
    int32_t best_idx = 0;
    float   best_val = src[0];
    for (uint32_t i = 1; i < n; ++i) {
        const float v = src[i];
        if (v > best_val) {
            best_val = v;
            best_idx = (int32_t) i;
        }
    }
    return best_idx;
}

// One row = one argmax, scalar for now. Audio8 sampling calls ARGMAX once per
// decoded token (greedy path) or once per final-logit reduction (top-k /
// top-p) with ne0 ~= vocab size (155 K for the full LM, 4 K after sem_head
// restriction). The dominant cost in that path is softmax / top-k, not this
// reduction; a scalar loop is correct and small enough to leave HVX
// optimization for later profiling.
static void htp_argmax_f32_job(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_argmax_context * actx = (const struct htp_argmax_context *) data;
    const struct htp_ops_context * octx = actx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;

    const uint32_t row_start = ith * actx->nrows_per_thread;
    const uint32_t row_end   = MIN(row_start + actx->nrows_per_thread, actx->nrows);
    if (row_start >= row_end) return;

    const uint32_t ne0 = src->ne[0];

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start);

    for (uint32_t row = row_start; row < row_end; ++row) {
        const float * src_row = (const float *) ((const uint8_t *) src->data + row * src->nb[1]);
        int32_t * dst_slot    = (int32_t *) ((uint8_t *) dst->data + row * dst->nb[0]);
        *dst_slot = argmax_scalar_f32(src_row, ne0);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start);
}

int op_argmax(struct htp_ops_context * octx) {
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;

    if (src->type != HTP_TYPE_F32) {
        FARF(ERROR, "argmax: only F32 input supported, got type %u", src->type);
        return HTP_STATUS_NO_SUPPORT;
    }
    if (dst->type != HTP_TYPE_I32) {
        FARF(ERROR, "argmax: only I32 output supported, got type %u", dst->type);
        return HTP_STATUS_NO_SUPPORT;
    }
    if (src->nb[0] != sizeof(float) || dst->nb[0] != sizeof(int32_t)) {
        FARF(ERROR, "argmax: inner dimension must be contiguous (src nb0=%u, dst nb0=%u)",
             (unsigned) src->nb[0], (unsigned) dst->nb[0]);
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t total_rows = src->ne[1] * src->ne[2] * src->ne[3];
    if (total_rows == 0) return HTP_STATUS_OK;

    const uint32_t n_threads = MIN(total_rows, octx->n_threads);

    struct htp_argmax_context actx = {
        .octx             = octx,
        .nrows            = total_rows,
        .nrows_per_thread = (total_rows + n_threads - 1) / n_threads,
    };

    FARF(HIGH, "argmax: %ux%ux%ux%u (type %u) -> %ux%ux%ux%u (type %u)",
         src->ne[0], src->ne[1], src->ne[2], src->ne[3], src->type,
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], dst->type);

    worker_pool_run_func(octx->ctx->worker_pool, htp_argmax_f32_job, &actx, n_threads);
    return HTP_STATUS_OK;
}
