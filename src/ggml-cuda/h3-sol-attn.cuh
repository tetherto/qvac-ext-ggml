#ifndef GGML_CUDA_H3_SOL_ATTN_CUH
#define GGML_CUDA_H3_SOL_ATTN_CUH

#include "h3-sol-attn.h"

#include "h3-sol-routing.cuh"

static void ggml_cuda_h3_sol_fused_attention(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                           const ggml_cuda_h3::sol_metadata & meta) {
    const int tokens = dst->src[0]->ne[1];
    const int blocks = (tokens + 63)/64;
    int max_ctas_per_sm;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_ctas_per_sm, cudaDevAttrMaxBlocksPerMultiprocessor, ctx.device));
    const size_t scratch_tiles = size_t(ggml_cuda_info().devices[ctx.device].nsm)*max_ctas_per_sm;
    const size_t centroids = size_t(56)*blocks*128;
    const size_t words = 16 + 3*centroids + 2*56*128 + scratch_tiles*64*h3_sol_correction_stride;
    ggml_cuda_pool_alloc<float> storage(ctx.pool(), words);
    const h3_sol_fused_header header{meta, blocks, {0,0}};
    CUDA_CHECK(cudaMemcpyAsync(storage.ptr, &header, sizeof(header), cudaMemcpyHostToDevice, ctx.stream()));
    auto * stats = reinterpret_cast<unsigned int *>(storage.ptr + 12);
    CUDA_CHECK(cudaMemsetAsync(stats, 0, 4*sizeof(unsigned int), ctx.stream()));
    float * qc = storage.ptr + 16;
    float * kc = qc + centroids;
    float * vc = kc + centroids;
    float * mean = vc + centroids;
    float * variance = mean + 56*128;
    float scale;
    std::memcpy(&scale, dst->op_params, sizeof(scale));
    h3_sol_centroids<true><<<dim3(blocks,56),128,0,ctx.stream()>>>(
        static_cast<const float *>(dst->src[0]->data), static_cast<const half *>(dst->src[1]->data),
        static_cast<const half *>(dst->src[2]->data), qc, kc, vc, tokens, blocks, scale);
    h3_sol_kc_stats<<<56,128,0,ctx.stream()>>>(kc, mean, variance, blocks);
    CUDA_CHECK(cudaGetLastError());
    ggml_cuda_flash_attn_ext_mma_f16_launch<128,128,64,1,true,true,true>(
        ctx, dst, reinterpret_cast<const int *>(storage.ptr));
    if (getenv("GGML_CUDA_H3_SOL_ATTN_TRACE") && strcmp(getenv("GGML_CUDA_H3_SOL_ATTN_TRACE"), "1") == 0) {
        unsigned int counts[4];
        CUDA_CHECK(cudaMemcpyAsync(counts, stats, sizeof(counts), cudaMemcpyDeviceToHost, ctx.stream()));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        GGML_LOG_INFO("h3_sol mode=%d routing=fused correction=bf16_tensor_mass tokens=%d beta=%.4g video=[%d,%d) frame_tokens=%d eligible=%u selected_out=%u skipped=%u total=%u scratch_bytes=%zu\n",
            meta.mode, tokens, meta.beta, meta.video_start, meta.video_end, meta.frame_tokens,
            counts[0], counts[1], counts[2], counts[3], words*sizeof(float));
    }
}

static bool ggml_cuda_h3_sol_attention(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_h3::sol_metadata meta;
    if (!ggml_cuda_h3::sol_metadata_supported(dst, meta)) {
        return false;
    }
#ifdef USE_CUDA_GRAPH
    // Only use request-owned scratch when the backend actually executes eagerly.
    // This shares the backend's process-cached environment policy.
    const ggml_cuda_graph graph_policy;
    if (graph_policy.is_enabled()) return false;
#endif
    for (const ggml_tensor * tensor : {dst, dst->src[0], dst->src[1], dst->src[2]}) {
        if (!tensor->data || (uintptr_t) tensor->data % 16 != 0) return false;
    }
    if (getenv("GGML_CUDA_H3_SOL_FUSED_ROUTING") && strcmp(getenv("GGML_CUDA_H3_SOL_FUSED_ROUTING"), "1") == 0) {
        ggml_cuda_h3_sol_fused_attention(ctx, dst, meta);
        return true;
    }
    const int tokens = dst->src[0]->ne[1];
    const int blocks = (tokens + 63)/64;
    GGML_ASSERT(ggml_cuda_fattn_mma_get_nbatch_fa(128, 128, 64, GGML_CUDA_CC_BLACKWELL) == 64);
    const size_t q_elements = size_t(56)*blocks*128;
    ggml_cuda_pool_alloc<float> pooled_q(ctx.pool(), q_elements + 2*56*128 + size_t(56)*blocks*blocks);
    float * kc_mean = pooled_q.ptr + q_elements;
    float * kc_variance = kc_mean + 56*128;
    float * proxy_scores = kc_variance + 56*128;
    int max_ctas_per_sm;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_ctas_per_sm, cudaDevAttrMaxBlocksPerMultiprocessor, ctx.device));
    const size_t scratch_tiles = size_t(ggml_cuda_info().devices[ctx.device].nsm)*max_ctas_per_sm;
    const size_t words = size_t(56)*blocks*blocks + 4;
    const size_t centroids = size_t(56)*blocks*128;
    const size_t scratch_words = scratch_tiles*64*h3_sol_correction_stride;
    ggml_cuda_pool_alloc<int> mask(ctx.pool(), words + 2*centroids + scratch_words);
    float * pooled_k = reinterpret_cast<float *>(mask.ptr + words);
    float * summed_v = pooled_k + centroids;
    auto * stats = reinterpret_cast<unsigned int *>(mask.ptr + words - 4);
    CUDA_CHECK(cudaMemsetAsync(stats, 0, 4*sizeof(unsigned int), ctx.stream()));
    float scale;
    std::memcpy(&scale, dst->op_params, sizeof(scale));
    h3_sol_centroids<<<dim3(blocks, 56), 128, 0, ctx.stream()>>>(
        static_cast<const float *>(dst->src[0]->data), static_cast<const half *>(dst->src[1]->data),
        static_cast<const half *>(dst->src[2]->data), pooled_q.ptr, pooled_k, summed_v, tokens, blocks, scale);
    h3_sol_kc_stats<<<56,128,0,ctx.stream()>>>(pooled_k,kc_mean,kc_variance,blocks);
    if (!h3_sol_proxy_scores(ctx.cublas_handle(), ctx.stream(), pooled_q.ptr, pooled_k, proxy_scores, blocks)) {
        GGML_LOG_WARN("H3 sparse proxy GEMM unavailable; retaining dense attention\n");
        return false;
    }
    h3_sol_routes<<<dim3(blocks, 56), 256, 0, ctx.stream()>>>(
        pooled_q.ptr, pooled_k, mask.ptr, blocks, scale, kc_mean, kc_variance, meta, proxy_scores);
    CUDA_CHECK(cudaGetLastError());
    ggml_cuda_flash_attn_ext_mma_f16_launch<128, 128, 64, 1, true, true>(ctx, dst, mask.ptr);
    if (getenv("GGML_CUDA_H3_SOL_ATTN_TRACE") && strcmp(getenv("GGML_CUDA_H3_SOL_ATTN_TRACE"), "1") == 0) {
        unsigned int counts[4];
        CUDA_CHECK(cudaMemcpyAsync(counts, stats, sizeof(counts), cudaMemcpyDeviceToHost, ctx.stream()));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        GGML_LOG_INFO("h3_sol mode=%d correction=%s tokens=%d beta=%.4g video=[%d,%d) frame_tokens=%d eligible=%u selected_out=%u skipped=%u total=%u\n",
            meta.mode, "bf16_tensor_mass", tokens, meta.beta, meta.video_start, meta.video_end, meta.frame_tokens,
            counts[0], counts[1], counts[2], counts[3]);
    }
    return true;
}

#endif
