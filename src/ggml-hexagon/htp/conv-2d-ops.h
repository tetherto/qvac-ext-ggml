#ifndef HTP_CONV_2D_OPS_H
#define HTP_CONV_2D_OPS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define HTP_CONV_F32_TYPE 0u
#define HTP_CONV_F32_LANES 32u
#define HTP_CONV_F32_OC_BLOCK 4u
#define HTP_CONV_F32_STAGE_FLOATS 96u
#define HTP_CONV_F32_WORKER_BYTES ((HTP_CONV_F32_STAGE_FLOATS + HTP_CONV_F32_LANES) * sizeof(float))

struct htp_conv_f32_tensor {
    uint64_t ne[4], nb[4], size, address;
    uint32_t type;
};

struct htp_conv_f32_layout {
    uint32_t w, h, ci, co, n, kw, kh, ow, oh;
    uint32_t sx, sy, px, py, dx, dy;
    uint32_t k_elements, plane, out_plane, tiles, oc_blocks, jobs, threads;
    uint32_t kernel_bytes, weights_bytes, worker_bytes, total_bytes;
    bool flat_pointwise;
};

// Optional spatial staging plan; the original host eligibility is unchanged.
struct htp_conv_f32_spatial_layout {
    uint32_t jobs, threads, row_floats, input_bytes, worker_bytes, total_bytes;
};

struct htp_conv_f32_pointwise_layout {
    uint32_t jobs, threads, input_bytes, worker_bytes, total_bytes;
};

static inline bool htp_conv_f32_span_fits(uint64_t address, uint64_t bytes, uint64_t limit) {
    return address <= limit && bytes <= limit - address;
}

// Coordinates for a zero-padded row segment, independent of pointer math.
static inline void htp_conv_f32_row_window(uint32_t width, int64_t first_x, uint32_t span, uint32_t * source,
                                           uint32_t * target, uint32_t * count) {
    *source = *target = *count = 0;
    if (!width || !span || span > HTP_CONV_F32_STAGE_FLOATS || first_x >= (int64_t)width || first_x <= -(int64_t)span) {
        return;
    }
    const int64_t first = first_x < 0 ? -first_x : 0;
    const int64_t available = (int64_t)width - first_x;
    const int64_t end = available < span ? available : span;
    if (first < end) {
        *source = (uint32_t)(first_x + first);
        *target = (uint32_t)first;
        *count = (uint32_t)(end - first);
    }
}

static inline bool htp_conv_f32_tensor_bytes(const struct htp_conv_f32_tensor * t, uint64_t * bytes) {
    if (!t || !bytes || t->type != HTP_CONV_F32_TYPE || (t->address && t->address % sizeof(float))) {
        return false;
    }
    uint64_t stride = sizeof(float);
    for (int i = 0; i < 4; ++i) {
        if (!t->ne[i] || t->ne[i] > UINT32_MAX || t->nb[i] != stride || stride > UINT32_MAX / t->ne[i]) {
            return false;
        }
        stride *= t->ne[i];
    }
    if (stride > t->size || (t->address && !htp_conv_f32_span_fits(t->address, stride, UINTPTR_MAX))) {
        return false;
    }
    *bytes = stride;
    return true;
}

static inline bool htp_conv_f32_validate(const struct htp_conv_f32_tensor * k, const struct htp_conv_f32_tensor * x,
                                         const struct htp_conv_f32_tensor * y, const int32_t * p, uint32_t max_threads,
                                         uint64_t vtcm_bytes, struct htp_conv_f32_layout * out) {
    uint64_t kb, xb, yb;
    if (!p || !out || !max_threads || max_threads > 10 || !htp_conv_f32_tensor_bytes(k, &kb) ||
        !htp_conv_f32_tensor_bytes(x, &xb) || !htp_conv_f32_tensor_bytes(y, &yb)) {
        return false;
    }
    if (!((k->ne[0] == 1 && k->ne[1] == 1) || (k->ne[0] == 3 && k->ne[1] == 3)) || k->ne[2] != x->ne[2] ||
        y->ne[2] != k->ne[3] || y->ne[3] != x->ne[3]) {
        return false;
    }
    if (p[0] < 1 || p[0] > 2 || p[1] < 1 || p[1] > 2 || p[2] < 0 || p[3] < 0 || p[4] < 1 || p[4] > 2 || p[5] < 1 ||
        p[5] > 2) {
        return false;
    }
    const int64_t width = (int64_t)x->ne[0] + 2 * (int64_t)p[2] - (int64_t)p[4] * (int64_t)(k->ne[0] - 1);
    const int64_t height = (int64_t)x->ne[1] + 2 * (int64_t)p[3] - (int64_t)p[5] * (int64_t)(k->ne[1] - 1);
    if (width <= 0 || height <= 0 || (uint64_t)((width - 1) / p[0] + 1) != y->ne[0] ||
        (uint64_t)((height - 1) / p[1] + 1) != y->ne[1]) {
        return false;
    }
    const struct htp_conv_f32_tensor * inputs[2] = {k, x};
    const uint64_t sizes[2] = {kb, xb};
    for (int i = 0; i < 2; ++i) {
        if (y->address && inputs[i]->address && y->address < inputs[i]->address + sizes[i] &&
            inputs[i]->address < y->address + yb) {
            return false;
        }
    }
    const bool flat = k->ne[0] == 1 && p[0] == 1 && p[1] == 1 && p[2] == 0 && p[3] == 0;
    const uint64_t plane = x->ne[0] * x->ne[1], out_plane = y->ne[0] * y->ne[1];
    const uint64_t tile_elements = flat ? out_plane : y->ne[0];
    const uint64_t tiles = (tile_elements + HTP_CONV_F32_LANES - 1) / HTP_CONV_F32_LANES;
    const uint64_t oc_blocks = (k->ne[3] + HTP_CONV_F32_OC_BLOCK - 1) / HTP_CONV_F32_OC_BLOCK;
    const uint64_t jobs = tiles * (flat ? 1 : y->ne[1]) * oc_blocks * x->ne[3];
    const uint64_t threads = jobs < max_threads ? jobs : max_threads;
    const uint64_t k_elements = k->ne[0] * k->ne[1] * k->ne[2];
    // Padding OC to a block makes all four broadcasts valid, including tails.
    const uint64_t padded_weights = k_elements * oc_blocks * HTP_CONV_F32_OC_BLOCK * sizeof(float);
    const uint64_t weights = (padded_weights + 127) & ~(uint64_t)127;
    const uint64_t total = weights + threads * HTP_CONV_F32_WORKER_BYTES;
    if (!jobs || jobs > UINT32_MAX || total > UINT32_MAX || total > vtcm_bytes) {
        return false;
    }
    const struct htp_conv_f32_layout layout = {(uint32_t)x->ne[0],
                                               (uint32_t)x->ne[1],
                                               (uint32_t)x->ne[2],
                                               (uint32_t)k->ne[3],
                                               (uint32_t)x->ne[3],
                                               (uint32_t)k->ne[0],
                                               (uint32_t)k->ne[1],
                                               (uint32_t)y->ne[0],
                                               (uint32_t)y->ne[1],
                                               (uint32_t)p[0],
                                               (uint32_t)p[1],
                                               (uint32_t)p[2],
                                               (uint32_t)p[3],
                                               (uint32_t)p[4],
                                               (uint32_t)p[5],
                                               (uint32_t)k_elements,
                                               (uint32_t)plane,
                                               (uint32_t)out_plane,
                                               (uint32_t)tiles,
                                               (uint32_t)oc_blocks,
                                               (uint32_t)jobs,
                                               (uint32_t)threads,
                                               (uint32_t)kb,
                                               (uint32_t)weights,
                                               (uint32_t)HTP_CONV_F32_WORKER_BYTES,
                                               (uint32_t)total,
                                               flat};
    *out = layout;
    return true;
}

// Optional acceleration plan. A false result retains the existing convolution
// kernel and does not change the host's supported-operation predicate.
static inline bool htp_conv_f32_plan_pointwise(const struct htp_conv_f32_layout * l, uint64_t vtcm_bytes,
                                               struct htp_conv_f32_pointwise_layout * out) {
    if (!l || !out || !l->flat_pointwise || !l->ci || !l->co || !l->n || !l->tiles ||
        !l->threads || l->threads > 10) {
        return false;
    }
    const uint64_t jobs = (uint64_t)l->tiles * l->n;
    const uint64_t threads = jobs < l->threads ? jobs : l->threads;
    const uint64_t input_bytes = (uint64_t)l->ci * HTP_CONV_F32_LANES * sizeof(float);
    const uint64_t worker_bytes = input_bytes + HTP_CONV_F32_OC_BLOCK * HTP_CONV_F32_LANES * sizeof(float);
    const uint64_t total_bytes = l->weights_bytes + threads * worker_bytes;
    if (!jobs || jobs > UINT32_MAX || input_bytes > UINT32_MAX || worker_bytes > UINT32_MAX ||
        total_bytes > UINT32_MAX || total_bytes > vtcm_bytes) {
        return false;
    }
    const struct htp_conv_f32_pointwise_layout result = {(uint32_t)jobs, (uint32_t)threads, (uint32_t)input_bytes,
                                                         (uint32_t)worker_bytes, (uint32_t)total_bytes};
    *out = result;
    return true;
}

// Eight OC accumulators reuse one 32-position input vector. Never trade away
// workers to fit this optional plan; the original one/four path remains valid.
static inline bool htp_conv_f32_plan_pointwise_eight(const struct htp_conv_f32_layout * l, uint64_t vtcm_bytes,
                                                     struct htp_conv_f32_pointwise_layout * out) {
    struct htp_conv_f32_pointwise_layout base;
    if (!out || !htp_conv_f32_plan_pointwise(l, vtcm_bytes, &base) || l->co < 8) return false;
    const uint64_t worker = (uint64_t)base.input_bytes + 8 * HTP_CONV_F32_LANES * sizeof(float);
    const uint64_t total = l->weights_bytes + base.threads * worker;
    if (worker > UINT32_MAX || total > UINT32_MAX || total > vtcm_bytes) return false;
    base.worker_bytes = (uint32_t)worker;
    base.total_bytes = (uint32_t)total;
    *out = base;
    return true;
}

// Stage all CI/KY row windows once per spatial job, then reuse them across
// output-channel blocks. SX=1 needs64floats/window; SX=2 needs96. Keep the
// previous generic kernel when its worker count or VTCM budget cannot be kept.
static inline bool htp_conv_f32_plan_spatial(const struct htp_conv_f32_layout * l, uint64_t vtcm_bytes,
                                             struct htp_conv_f32_spatial_layout * out) {
    if (!l || !out || l->flat_pointwise || l->kw != 3 || l->kh != 3 || !l->ci || !l->co || !l->n ||
        !l->w || !l->h || !l->ow || !l->oh || !l->tiles || !l->threads || l->threads > 10 ||
        l->sx < 1 || l->sx > 2 || l->sy < 1 || l->sy > 2 || l->dx < 1 || l->dx > 2 || l->dy < 1 || l->dy > 2 ||
        (uint64_t)l->w * l->h != l->plane || (uint64_t)l->ow * l->oh != l->out_plane ||
        ((uint64_t)l->ow + HTP_CONV_F32_LANES - 1) / HTP_CONV_F32_LANES != l->tiles) {
        return false;
    }
    uint64_t jobs = l->tiles;
    if (jobs > UINT32_MAX / l->oh) return false;
    jobs *= l->oh;
    if (jobs > UINT32_MAX / l->n) return false;
    jobs *= l->n;
    if (jobs < l->threads) return false;
    const uint64_t span = (HTP_CONV_F32_LANES - 1) * l->sx + (l->kw - 1) * l->dx + 1;
    const uint64_t row_floats = (span + HTP_CONV_F32_LANES - 1) / HTP_CONV_F32_LANES * HTP_CONV_F32_LANES;
    const uint64_t row_group = (uint64_t)l->kh * row_floats * sizeof(float);
    if (row_floats > HTP_CONV_F32_STAGE_FLOATS || l->ci > UINT32_MAX / row_group) return false;
    const uint64_t input = l->ci * row_group;
    const uint64_t worker = input + HTP_CONV_F32_LANES * sizeof(float);
    const uint64_t total = l->weights_bytes + l->threads * worker;
    if (worker > UINT32_MAX || total > UINT32_MAX || total > vtcm_bytes) return false;
    const struct htp_conv_f32_spatial_layout result = {(uint32_t)jobs, l->threads, (uint32_t)row_floats,
                                                       (uint32_t)input, (uint32_t)worker, (uint32_t)total};
    *out = result;
    return true;
}

#endif
