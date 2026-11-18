#include "fattn.cuh"
#include <cinttypes>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "fattn-mma-f16.cuh"
#include "h3-fattn.h"
#include "h3-sol-attn.cuh"

#ifdef FLASH_ATTN_AVAILABLE
static size_t h3_prefetch_shared(int columns, int cc) {
    const auto c = ggml_cuda_fattn_mma_get_config(128, 128, columns, cc);
    const size_t q = columns * (128/2 + 4) * sizeof(half2);
    const size_t kv = c.nbatch_fa * std::max(c.nbatch_K2 + 4, c.nbatch_V2 + 4) * sizeof(half2);
    const size_t mask = columns * (c.nbatch_fa/2 + 4) * sizeof(half2);
    const size_t combine = (c.nthreads/WARP_SIZE) * std::min(columns, get_cols_per_warp(cc)) *
        (c.nbatch_combine + 4) * sizeof(half2);
    return std::max(combine, c.Q_in_reg ? std::max(q, kv + mask) : q + kv + mask);
}

static bool h3_wide_attention(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int nq = dst->src[0]->ne[1];
#ifdef USE_CUDA_GRAPH
    const ggml_cuda_graph graph_policy;
    if (graph_policy.is_enabled()) {
        return false;
    }
#endif
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const auto narrow = ggml_cuda_fattn_mma_get_config(128, 128, 64, cc);
    const auto wide = ggml_cuda_fattn_mma_get_config(128, 128, 128, cc);
    const auto compatible = [](const fattn_mma_config & c, int threads) {
        return c.nthreads == threads && c.nbatch_fa == 64 && c.nbatch_K2 == 64 &&
            c.nbatch_V2 == 64 && c.nbatch_combine == 64 && c.Q_in_reg;
    };
    if (!compatible(narrow, 128) || !compatible(wide, 256)) {
        return false;
    }
    int narrow_occupancy = 0, wide_occupancy = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&narrow_occupancy,
        flash_attn_ext_f16<128, 128, 64, 1, false, false, true>, 128, h3_prefetch_shared(64, cc)));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&wide_occupancy,
        flash_attn_ext_f16<128, 128, 128, 1, false, false, true>, 256, h3_prefetch_shared(128, cc)));
    const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
    // Repair uses the narrow reduction partitions; the wide grid must have exactly half as many blocks.
    if (narrow_occupancy != 2 || wide_occupancy != 1 ||
        ((nq + 63)/64)*56 <= 2*nsm || ((nq + 127)/128)*56 <= nsm) {
        return false;
    }
    ggml_cuda_flash_attn_ext_mma_f16_launch<128, 128, 128, 1, true>(ctx, dst);
    ggml_cuda_flash_attn_ext_mma_f16_launch<128, 128, 64, 1, true, false, false, true>(ctx, dst);
    static const bool logged = [] {
        GGML_LOG_INFO("h3_fattn_wide boundary_repair=1\n");
        return true;
    }();
    GGML_UNUSED(logged);
    return true;
}

#endif

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
        ggml_cuda_highest_compiled_arch(cc) != GGML_CUDA_CC_BLACKWELL) {
        return false;
    }
    if (ggml_cuda_h3_sol_attention(ctx, dst)) {
        return true;
    }
    static const bool long_context = getenv("GGML_CUDA_H3_FATTN_LONG_CONTEXT") &&
        strcmp(getenv("GGML_CUDA_H3_FATTN_LONG_CONTEXT"), "1") == 0;
    if (!ggml_cuda_h3::fattn_tuning_supported(dst, long_context ? 65536 : 32768)) {
        return false;
    }
    for (const ggml_tensor * tensor : {dst, dst->src[0], dst->src[1], dst->src[2]}) {
        if (!tensor->data || (uintptr_t) tensor->data % 16 != 0) {
            return false;
        }
    }
    static const bool wide = [] {
        const char * value = getenv("GGML_CUDA_H3_FATTN_WIDE");
        return value && strcmp(value, "1") == 0;
    }();
    const bool used_wide = wide && h3_wide_attention(ctx, dst);
    if (!used_wide) {
        ggml_cuda_flash_attn_ext_mma_f16_launch<128, 128, 64, 1, true>(ctx, dst);
    }
    static const bool trace = getenv("GGML_CUDA_H3_FATTN_TRACE") &&
        strcmp(getenv("GGML_CUDA_H3_FATTN_TRACE"), "1") == 0;
    if (trace) {
        GGML_LOG_INFO("h3_fattn_prefetch stages=1 tokens=%" PRId64 " heads=56 wide=%d\n",
            dst->src[0]->ne[1], used_wide);
    }
    return true;
#endif
}
#endif
