#ifndef HTP_ENERGY_OPS_H
#define HTP_ENERGY_OPS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define HTP_ENERGY_F32 0u
#define HTP_ENERGY_LANES 32u
#define HTP_ENERGY_MAX_CHANNELS 128u

enum htp_energy_mode { HTP_ENERGY_ROWS = 0, HTP_ENERGY_CHANNELS = 1 };

struct htp_energy_tensor {
    uint64_t ne[4], nb[4], size, address;
    uint32_t type;
};

struct htp_energy_layout {
    uint32_t mode, width, rows, spatial, groups;
    uint32_t row_stride, channel_stride, group_stride;
    uint32_t source_bytes, destination_bytes, jobs, threads;
};

static inline bool htp_energy_span_fits(uint64_t address, uint64_t bytes, uint64_t limit) {
    return address <= limit && bytes <= limit - address;
}

// Accept only the two source layouts used by the graph patterns: ordinary
// contiguous F32 rows, or [channels, spatial, groups] views of channel planes.
// The second layout is read directly, without executing CONT or writing SQR.
static inline bool htp_energy_validate(const struct htp_energy_tensor * x,
                                       const struct htp_energy_tensor * y,
                                       uint32_t max_threads,
                                       struct htp_energy_layout * out) {
    if (!x || !y || !out || !max_threads || max_threads > 10 ||
        x->type != HTP_ENERGY_F32 || y->type != HTP_ENERGY_F32 ||
        (x->address && x->address % sizeof(float)) ||
        (y->address && y->address % sizeof(float)) || y->ne[0] != 1) return false;
    for (int d = 0; d < 4; ++d) {
        if (!x->ne[d] || x->ne[d] > UINT32_MAX || x->nb[d] > UINT32_MAX ||
            !y->ne[d] || y->ne[d] > UINT32_MAX || y->nb[d] > UINT32_MAX ||
            (d && x->ne[d] != y->ne[d])) return false;
    }
    uint64_t destination_bytes = sizeof(float);
    for (int d = 0; d < 4; ++d) {
        if (y->nb[d] != destination_bytes || destination_bytes > UINT32_MAX / y->ne[d]) return false;
        destination_bytes *= y->ne[d];
    }
    const uint64_t rows = destination_bytes / sizeof(float);
    struct htp_energy_layout layout = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    layout.width = (uint32_t)x->ne[0];
    layout.rows = (uint32_t)rows;
    uint64_t source_bytes = 0;
    if (x->nb[0] == sizeof(float) && x->ne[0] <= UINT32_MAX / sizeof(float) &&
        x->nb[1] >= x->ne[0] * sizeof(float) && x->nb[1] % sizeof(float) == 0) {
        uint64_t stride = x->nb[1];
        for (int d = 1; d < 4; ++d) {
            if (stride > UINT32_MAX / x->ne[d]) return false;
            stride *= x->ne[d];
            if (d < 3 && x->nb[d + 1] != stride) return false;
        }
        source_bytes = stride - x->nb[1] + x->ne[0] * sizeof(float);
        layout.mode = HTP_ENERGY_ROWS;
        layout.row_stride = (uint32_t)x->nb[1];
        layout.jobs = (layout.rows - 1) / HTP_ENERGY_LANES + 1;
    } else {
        if (x->ne[0] > HTP_ENERGY_MAX_CHANNELS || x->ne[1] > UINT32_MAX / sizeof(float) ||
            x->nb[1] != sizeof(float) || x->nb[0] != x->ne[1] * sizeof(float)) return false;
        uint64_t stride = x->nb[0];
        if (stride > UINT32_MAX / x->ne[0]) return false;
        stride *= x->ne[0];
        if (x->nb[2] != stride || stride > UINT32_MAX / x->ne[2]) return false;
        stride *= x->ne[2];
        if (x->nb[3] != stride || stride > UINT32_MAX / x->ne[3]) return false;
        source_bytes = stride * x->ne[3];
        layout.mode = HTP_ENERGY_CHANNELS;
        layout.spatial = (uint32_t)x->ne[1];
        layout.groups = (uint32_t)(x->ne[2] * x->ne[3]);
        layout.channel_stride = (uint32_t)x->nb[0];
        layout.group_stride = (uint32_t)x->nb[2];
        const uint64_t tiles = (x->ne[1] - 1) / HTP_ENERGY_LANES + 1;
        const uint64_t jobs = tiles * layout.groups;
        if (jobs > UINT32_MAX) return false;
        layout.jobs = (uint32_t)jobs;
    }
    if (source_bytes > x->size || destination_bytes > y->size ||
        (x->address && !htp_energy_span_fits(x->address, source_bytes, UINTPTR_MAX)) ||
        (y->address && !htp_energy_span_fits(y->address, destination_bytes, UINTPTR_MAX))) return false;
    if (x->address && y->address && x->address < y->address + destination_bytes &&
        y->address < x->address + source_bytes) return false;
    layout.source_bytes = (uint32_t)source_bytes;
    layout.destination_bytes = (uint32_t)destination_bytes;
    layout.threads = layout.jobs < max_threads ? layout.jobs : max_threads;
    *out = layout;
    return true;
}

static inline bool htp_energy_direct_row_tail(const struct htp_energy_layout * layout, uint32_t row) {
    if (!layout || layout->mode != HTP_ENERGY_ROWS || row >= layout->rows) return false;
    const uint64_t end = (uint64_t)row * layout->row_stride +
        ((uint64_t)layout->width + HTP_ENERGY_LANES - 1) / HTP_ENERGY_LANES * 128;
    return end <= layout->source_bytes;
}

#endif
