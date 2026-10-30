#ifndef HTP_MEAN_OPS_H
#define HTP_MEAN_OPS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// GGML_TYPE_F32 / HTP_TYPE_F32; checked at the host and DSP integration sites.
#define HTP_MEAN_TYPE_F32 0u
#define HTP_MEAN_LANES 32u

struct htp_mean_tensor {
    uint64_t ne[4], nb[4], size, address;
    uint32_t type;
};

struct htp_mean_layout {
    uint32_t width, rows, stride, source_bytes, threads;
};

static inline bool htp_mean_span_fits(uint64_t address, uint64_t bytes, uint64_t limit) {
    return address <= limit && bytes <= limit - address;
}

// Rows may contain padding, but the higher dimensions must pack those rows.
static inline bool htp_mean_tensor_bytes(const struct htp_mean_tensor * t, bool padded, uint64_t * bytes) {
    if (!t || !bytes || t->type != HTP_MEAN_TYPE_F32 || t->nb[0] != sizeof(float) ||
        (t->address && t->address % sizeof(float))) {
        return false;
    }
    for (int d = 0; d < 4; ++d) {
        if (!t->ne[d] || t->ne[d] > UINT32_MAX || t->nb[d] > UINT32_MAX) return false;
    }
    if (t->ne[0] > UINT32_MAX / sizeof(float)) return false;
    const uint64_t row_bytes = t->ne[0] * sizeof(float);
    if (t->nb[1] < row_bytes || t->nb[1] % sizeof(float) || (!padded && t->nb[1] != row_bytes)) {
        return false;
    }
    uint64_t stride = t->nb[1];
    for (int d = 1; d < 4; ++d) {
        if (stride > UINT32_MAX / t->ne[d]) return false;
        stride *= t->ne[d];
        if (d < 3 && t->nb[d + 1] != stride) return false;
    }
    const uint64_t span = stride - t->nb[1] + row_bytes;
    if (span > t->size || (t->address && !htp_mean_span_fits(t->address, span, UINTPTR_MAX))) return false;
    *bytes = span;
    return true;
}

static inline bool htp_mean_validate(const struct htp_mean_tensor * src, const struct htp_mean_tensor * dst,
                                     uint32_t max_threads, struct htp_mean_layout * out) {
    uint64_t source_bytes, destination_bytes;
    if (!out || !max_threads || max_threads > 10 ||
        !htp_mean_tensor_bytes(src, true, &source_bytes) || !htp_mean_tensor_bytes(dst, false, &destination_bytes) ||
        dst->ne[0] != 1) {
        return false;
    }
    for (int d = 1; d < 4; ++d) {
        if (src->ne[d] != dst->ne[d]) return false;
    }
    if (src->address && dst->address && src->address < dst->address + destination_bytes &&
        dst->address < src->address + source_bytes) {
        return false;
    }
    const uint64_t rows = destination_bytes / sizeof(float);
    const struct htp_mean_layout layout = {(uint32_t)src->ne[0], (uint32_t)rows, (uint32_t)src->nb[1],
                                           (uint32_t)source_bytes, rows < max_threads ? (uint32_t)rows : max_threads};
    *out = layout;
    return true;
}

// A packed store never crosses the current worker's exclusive row boundary.
static inline uint32_t htp_mean_block_rows(uint32_t row, uint32_t end) {
    if (row >= end) return 0;
    const uint32_t remaining = end - row;
    return remaining < HTP_MEAN_LANES ? remaining : HTP_MEAN_LANES;
}

// The existing reducer loads a full vector for its last partial vector. Only
// reuse that load when all 128 bytes lie inside the validated source span.
static inline bool htp_mean_direct_tail(const struct htp_mean_layout * layout, uint32_t row) {
    if (!layout || row >= layout->rows) return false;
    const uint64_t end = (uint64_t)row * layout->stride +
                         ((uint64_t)layout->width + HTP_MEAN_LANES - 1) / HTP_MEAN_LANES * 128;
    return end <= layout->source_bytes;
}

#endif
