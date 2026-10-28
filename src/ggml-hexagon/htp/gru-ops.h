#ifndef HTP_GRU_OPS_H
#define HTP_GRU_OPS_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#define HTP_GRU_MAX_H 128
#define HTP_GRU_LANES 32
// Shared with HTP_TYPE_F32 / GGML_TYPE_F32; asserted at both integration sites.
#define HTP_GRU_TYPE_F32 0u

// Portable validation shared by host support checks, DSP execution and tests.
struct htp_gru_tensor {
    uint64_t ne[4], nb[4], size, address;
    uint32_t type;
};

struct htp_gru_layout {
    uint32_t h, b, l, padded_h, threads;
    uint32_t weights_bytes, bias_bytes, worker_bytes, total_bytes;
};

static inline bool htp_gru_span_fits(uint64_t address, uint64_t bytes, uint64_t address_limit) {
    return address <= address_limit && bytes <= address_limit - address;
}

static inline bool htp_gru_tensor_bytes(const struct htp_gru_tensor * t, uint64_t * bytes) {
    if (!t || t->type != HTP_GRU_TYPE_F32 || (t->address && t->address % sizeof(float)))
        return false;
    uint64_t stride = sizeof(float);
    for (int d = 0; d < 4; ++d) {
        if (!t->ne[d] || t->ne[d] > UINT32_MAX || t->nb[d] != stride || stride > UINT32_MAX / t->ne[d])
            return false;
        stride *= t->ne[d];
    }
    if (stride > t->size || (t->address && !htp_gru_span_fits(t->address, stride, UINTPTR_MAX)))
        return false;
    *bytes = stride;
    return true;
}

static inline bool htp_gru_validate(const struct htp_gru_tensor * w, const struct htp_gru_tensor * gi,
                                    const struct htp_gru_tensor * bias, const struct htp_gru_tensor * dst, int reverse,
                                    uint32_t max_threads, uint64_t vtcm_bytes, struct htp_gru_layout * out) {
    uint64_t bytes[4];
    const struct htp_gru_tensor * tensors[4] = {w, gi, bias, dst};
    if (!out || (reverse != 0 && reverse != 1) || !max_threads || max_threads > 10)
        return false;
    for (int i = 0; i < 4; ++i)
        if (!htp_gru_tensor_bytes(tensors[i], &bytes[i]))
            return false;
    const uint64_t h = w->ne[0], b = gi->ne[1], l = gi->ne[2];
    if (h > HTP_GRU_MAX_H || w->ne[1] != 3 * h || w->ne[2] != 1 || w->ne[3] != 1 || gi->ne[0] != 3 * h ||
        gi->ne[3] != 1 || bias->ne[0] != 3 * h || bias->ne[1] != 1 || bias->ne[2] != 1 || bias->ne[3] != 1 ||
        dst->ne[0] != h || dst->ne[1] != b || dst->ne[2] != l || dst->ne[3] != 1)
        return false;
    for (int i = 0; i < 3; ++i) {
        if (dst->address && tensors[i]->address && dst->address < tensors[i]->address + bytes[i] &&
            tensors[i]->address < dst->address + bytes[3])
            return false;
    }
    const uint64_t padded = (h + HTP_GRU_LANES - 1) / HTP_GRU_LANES * HTP_GRU_LANES;
    const uint64_t threads = b < max_threads ? b : max_threads;
    const uint64_t weights = 3 * h * padded * sizeof(float), biases = 3 * padded * sizeof(float);
    const uint64_t worker = 7 * padded * sizeof(float), total = weights + biases + threads * worker;
    if (total > vtcm_bytes || total > UINT32_MAX)
        return false;
    const struct htp_gru_layout layout = {(uint32_t)h,      (uint32_t)b,       (uint32_t)l,
                                          (uint32_t)padded, (uint32_t)threads, (uint32_t)weights,
                                          (uint32_t)biases, (uint32_t)worker,  (uint32_t)total};
    *out = layout;
    return true;
}

#endif
