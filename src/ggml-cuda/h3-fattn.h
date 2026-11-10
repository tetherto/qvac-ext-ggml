#ifndef GGML_CUDA_H3_FATTN_H
#define GGML_CUDA_H3_FATTN_H

#include "ggml-impl.h"

#include <cmath>
#include <cstring>

namespace ggml_cuda_h3 {

static inline bool fattn_tuning_supported(const ggml_tensor * output) {
    if (!output || output->op != GGML_OP_FLASH_ATTN_EXT || output->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(output) || output->view_src || output->src[3] || output->src[4] ||
        ggml_flash_attn_ext_get_prec(output) != GGML_PREC_F32) {
        return false;
    }
    const ggml_tensor * q = output->src[0];
    const ggml_tensor * k = output->src[1];
    const ggml_tensor * v = output->src[2];
    if (!q || !k || !v || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 ||
        !ggml_is_contiguous(q) || !ggml_is_contiguous(k) || !ggml_is_contiguous(v) ||
        !ggml_are_same_shape(q, k) || !ggml_are_same_shape(q, v) ||
        q->ne[0] != 128 || q->ne[1] < 1024 || q->ne[1] > 32768 || q->ne[2] != 56 || q->ne[3] != 1) {
        return false;
    }
    if (output->ne[0] != 128 || output->ne[1] != 56 || output->ne[2] != q->ne[1] || output->ne[3] != 1) {
        return false;
    }
    float params[3];
    std::memcpy(params, output->op_params, sizeof(params));
    return std::isfinite(params[0]) && params[0] > 0.f && params[1] == 0.f && params[2] == 0.f;
}

} // namespace ggml_cuda_h3

#endif
