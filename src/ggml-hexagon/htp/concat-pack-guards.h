#ifndef HTP_CONCAT_PACK_GUARDS_H
#define HTP_CONCAT_PACK_GUARDS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define HTP_CONCAT_PACK_F32_TYPE 0u

struct htp_concat_pack_tensor {
    uint64_t ne[4], nb[4], size, address;
    uint32_t type;
};

struct htp_concat_pack_layout {
    uint32_t row_elements, rows, rows_per_vector, threads;
};

// This is an optional fast path, not a replacement for general CONCAT.
// Bound every read/store by a checked, densely packed F32 tensor span.
static inline bool htp_concat_pack_span(const struct htp_concat_pack_tensor * t, uint64_t * bytes) {
    if (!t || !bytes || t->type != HTP_CONCAT_PACK_F32_TYPE || !t->address || (t->address & 3)) return false;
    uint64_t span = sizeof(float);
    for (int i = 0; i < 4; ++i) {
        if (!t->ne[i] || t->ne[i] > UINT32_MAX || t->nb[i] != span || t->ne[i] > UINT32_MAX / span) return false;
        span *= t->ne[i];
    }
    if (t->size < span || t->address > UINTPTR_MAX || span > UINTPTR_MAX - t->address) return false;
    *bytes = span;
    return true;
}

static inline bool htp_concat_pack_plan(const struct htp_concat_pack_tensor * a,
                                        const struct htp_concat_pack_tensor * b,
                                        const struct htp_concat_pack_tensor * d, int dim, uint32_t threads,
                                        struct htp_concat_pack_layout * out) {
    if (!a || !b || !d || !out || dim != 0 || !threads || threads > 10) return false;
    const uint64_t width = a->ne[0];
    if ((width != 4 && width != 8 && width != 16) || b->ne[0] != width || d->ne[0] != 2 * width) return false;
    for (int i = 1; i < 4; ++i) if (a->ne[i] != b->ne[i] || a->ne[i] != d->ne[i]) return false;
    uint64_t a_bytes, b_bytes, d_bytes;
    if (!htp_concat_pack_span(a, &a_bytes) || !htp_concat_pack_span(b, &b_bytes) ||
        !htp_concat_pack_span(d, &d_bytes)) return false;
    if ((d->address < a->address + a_bytes && a->address < d->address + d_bytes) ||
        (d->address < b->address + b_bytes && b->address < d->address + d_bytes)) return false;
    const uint64_t rows = a_bytes / (width * sizeof(float));
    if (!rows || rows > UINT32_MAX || b_bytes != a_bytes || d_bytes != 2 * a_bytes) return false;
    const struct htp_concat_pack_layout result = {
        (uint32_t)width, (uint32_t)rows, (uint32_t)(128 / (width * sizeof(float))),
        rows < threads ? (uint32_t)rows : threads};
    *out = result;
    return true;
}

static inline void htp_concat_pack_rows(const struct htp_concat_pack_layout * l, uint32_t worker,
                                        uint32_t * begin, uint32_t * end) {
    const uint32_t per_worker = l->rows / l->threads + (l->rows % l->threads != 0);
    const uint64_t first = (uint64_t)worker * per_worker;
    *begin = first < l->rows ? (uint32_t)first : l->rows;
    *end = l->rows - *begin < per_worker ? l->rows : *begin + per_worker;
}

#endif
