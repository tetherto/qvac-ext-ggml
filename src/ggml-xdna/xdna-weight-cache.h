#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <array>
#include <cstdint>
#include <cstring>

static inline bool xdna_weight_is_cacheable(const ggml_tensor * tensor) {
    while (tensor && tensor->view_src) {
        tensor = tensor->view_src;
    }
    return tensor && tensor->op == GGML_OP_NONE && tensor->buffer &&
           ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

static inline uint64_t xdna_weight_fingerprint(const void * data, size_t size) {
    constexpr uint64_t PRIME = 1099511628211ull;
    std::array<uint64_t, 4> lanes{ 1469598103934665603ull, 1099511628211ull,
                                   9650029242287828579ull, 2870177450012600261ull };
    const auto * bytes = static_cast<const unsigned char *>(data);
    size_t offset = 0;
    while (size - offset >= sizeof(uint64_t) * lanes.size()) {
        for (size_t lane = 0; lane < lanes.size(); lane++) {
            uint64_t word;
            std::memcpy(&word, bytes + offset, sizeof(word));
            lanes[lane] = (lanes[lane] ^ word) * PRIME;
            offset += sizeof(word);
        }
    }
    uint64_t result = size;
    for (const uint64_t lane : lanes) {
        result = (result ^ lane) * PRIME;
    }
    while (offset < size) {
        result = (result ^ bytes[offset++]) * PRIME;
    }
    return result;
}

struct xdna_weight_signature {
    const void * data = nullptr;
    ggml_type type = GGML_TYPE_F32;
    std::array<int64_t, GGML_MAX_DIMS> ne{};
    std::array<size_t, GGML_MAX_DIMS> nb{};
    uint64_t fingerprint = 0;
    bool valid = false;

    explicit xdna_weight_signature(const ggml_tensor * tensor = nullptr) {
        if (!xdna_weight_is_cacheable(tensor) || !tensor->data) {
            return;
        }
        data = tensor->data;
        type = tensor->type;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            ne[i] = tensor->ne[i];
            nb[i] = tensor->nb[i];
        }
        fingerprint = xdna_weight_fingerprint(data, ggml_nbytes(tensor));
        valid = true;
    }

    bool operator==(const xdna_weight_signature & other) const {
        return valid && other.valid && data == other.data && type == other.type &&
               ne == other.ne && nb == other.nb && fingerprint == other.fingerprint;
    }
};
