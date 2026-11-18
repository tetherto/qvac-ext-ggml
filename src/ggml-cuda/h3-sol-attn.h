#ifndef GGML_CUDA_H3_SOL_ATTN_H
#define GGML_CUDA_H3_SOL_ATTN_H

#include "h3-fattn.h"

namespace ggml_cuda_h3 {

// Private, versioned engine/backend experiment. The first four FA op parameters
// retain their standard meaning. No tensor or CUDA context ABI is changed.
struct sol_metadata {
    int32_t magic;
    int32_t version;
    int32_t mode;  // 6 = BF16 Tensor Core centroid correction; 7 = all-keep control.
                   // Earlier experiment modes 1-5 are intentionally rejected.
    int32_t video_start;
    int32_t video_end;
    int32_t frame_tokens;
    float beta;
    int32_t dense_prefix_layers = 2;
    int32_t dense_suffix_layers = 2;
};
static_assert(sizeof(sol_metadata) <= GGML_MAX_OP_PARAMS - 4 * sizeof(int32_t), "FA metadata space");
static constexpr int32_t sol_magic = 0x48335343;

static inline bool sol_metadata_supported(const ggml_tensor * dst, sol_metadata & meta) {
    if (!fattn_tuning_supported(dst, 65536)) {
        return false;
    }
    std::memcpy(&meta, dst->op_params + 4, sizeof(meta));
    return meta.magic == sol_magic && meta.version == 3 &&
        (meta.mode == 6 || meta.mode == 7) && meta.video_start >= 0 &&
        meta.video_end <= dst->src[0]->ne[1] && meta.video_end > meta.video_start &&
        meta.frame_tokens > 0 &&
        (meta.video_end - meta.video_start) % meta.frame_tokens == 0 &&
        std::isfinite(meta.beta) && meta.beta >= -10.f && meta.beta <= 2.f &&
        meta.dense_prefix_layers >= 0 && meta.dense_prefix_layers <= 200 &&
        meta.dense_suffix_layers >= 0 && meta.dense_suffix_layers <= 200;
}

} // namespace ggml_cuda_h3
#endif
