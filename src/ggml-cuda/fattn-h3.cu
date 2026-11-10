#include "fattn.cuh"
#include <cinttypes>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "fattn-mma-f16.cuh"
#include "h3-fattn.h"

bool ggml_cuda_flash_attn_ext_h3(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    return false;
#else
    static const bool enabled = [] {
        const char * value = getenv("GGML_CUDA_H3_FATTN_PREFETCH");
        return !value || strcmp(value, "1") == 0;
    }();
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!enabled || cc != GGML_CUDA_CC_BLACKWELL || !ampere_mma_available(cc) ||
        ggml_cuda_highest_compiled_arch(cc) != GGML_CUDA_CC_BLACKWELL ||
        !ggml_cuda_h3::fattn_tuning_supported(dst)) {
        return false;
    }
    for (const ggml_tensor * tensor : {dst, dst->src[0], dst->src[1], dst->src[2]}) {
        if (!tensor->data || (uintptr_t) tensor->data % 16 != 0) {
            return false;
        }
    }
    ggml_cuda_flash_attn_ext_mma_f16_launch<128, 128, 64, 1, true>(ctx, dst);
    static const bool trace = getenv("GGML_CUDA_H3_FATTN_TRACE") &&
        strcmp(getenv("GGML_CUDA_H3_FATTN_TRACE"), "1") == 0;
    if (trace) {
        GGML_LOG_INFO("h3_fattn_prefetch stages=1 tokens=%" PRId64 " heads=56\n", dst->src[0]->ne[1]);
    }
    return true;
#endif
}
#endif
