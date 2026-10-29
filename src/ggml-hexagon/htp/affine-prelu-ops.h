#ifndef HTP_AFFINE_PRELU_OPS_H
#define HTP_AFFINE_PRELU_OPS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define HTP_AFFINE_F32_TYPE 0u
#define HTP_AFFINE_F32_LANES 32u
#define HTP_AFFINE_F32_WORKER_BYTES (4u * 128u)

struct htp_affine_f32_tensor {
    uint64_t ne[4], nb[4], size, address;
    uint32_t type;
};

struct htp_affine_f32_layout {
    uint32_t f, t, c, b, tiles, jobs, threads;
};

static inline bool htp_affine_f32_tensor_bytes(const struct htp_affine_f32_tensor * tensor, uint64_t * bytes) {
    if (!tensor || !bytes || tensor->type != HTP_AFFINE_F32_TYPE || tensor->address % sizeof(float)) {
        return false;
    }
    uint64_t stride = sizeof(float);
    for (int d = 0; d < 4; ++d) {
        if (!tensor->ne[d] || tensor->ne[d] > UINT32_MAX || tensor->nb[d] != stride ||
            stride > UINT32_MAX / tensor->ne[d]) {
            return false;
        }
        stride *= tensor->ne[d];
    }
    if (stride > tensor->size || tensor->address > UINTPTR_MAX || stride > UINTPTR_MAX - tensor->address) {
        return false;
    }
    *bytes = stride;
    return true;
}

// The portable predicate is shared by the host preflight, DSP and guard tests.
// Metadata-only queries use address 0; actual execution additionally requires
// non-null mapped pointers. Every product is bounded before it is narrowed.
static inline bool htp_affine_f32_validate(const struct htp_affine_f32_tensor tensors[5], uint32_t max_threads,
                                           uint64_t vtcm_bytes, struct htp_affine_f32_layout * layout) {
    if (!tensors || !layout || !max_threads || max_threads > 10) {
        return false;
    }
    uint64_t bytes[5];
    for (int i = 0; i < 5; ++i) {
        if (!htp_affine_f32_tensor_bytes(&tensors[i], &bytes[i])) {
            return false;
        }
    }
    const uint64_t * x = tensors[0].ne;
    const uint64_t * aw = tensors[1].ne;
    const uint64_t * ab = tensors[2].ne;
    const uint64_t * slope = tensors[3].ne;
    if (aw[0] != x[0] || aw[1] != x[2] || aw[2] != 1 || aw[3] != 1 ||
        ab[0] != x[0] || ab[1] != x[2] || ab[2] != 1 || ab[3] != 1 ||
        slope[0] != x[2] || slope[1] != 1 || slope[2] != 1 || slope[3] != 1) {
        return false;
    }
    for (int d = 0; d < 4; ++d) {
        if (tensors[4].ne[d] != x[d]) {
            return false;
        }
    }
    for (int i = 0; i < 4; ++i) {
        if (tensors[4].address && tensors[i].address &&
            tensors[4].address < tensors[i].address + bytes[i] &&
            tensors[i].address < tensors[4].address + bytes[4]) {
            return false;
        }
    }
    const uint64_t tiles = (x[0] + HTP_AFFINE_F32_LANES - 1) / HTP_AFFINE_F32_LANES;
    const uint64_t jobs = tiles * x[2] * x[3]; // <= elements, already bounded by tensor_bytes
    const uint64_t threads = jobs < max_threads ? jobs : max_threads;
    if (!jobs || jobs > UINT32_MAX || threads * HTP_AFFINE_F32_WORKER_BYTES > vtcm_bytes) {
        return false;
    }
    const struct htp_affine_f32_layout result = {(uint32_t)x[0], (uint32_t)x[1], (uint32_t)x[2], (uint32_t)x[3],
                                                 (uint32_t)tiles, (uint32_t)jobs, (uint32_t)threads};
    *layout = result;
    return true;
}

#endif
