#ifndef GGML_CUDA_H3_SOL_ROUTING_CUH
#define GGML_CUDA_H3_SOL_ROUTING_CUH

#include "h3-sol-attn.h"

// Proxy-score correction follows NVIDIA's released Sol-Attn SM120 mainloop:
// K block means and V block sums retain both numerator and denominator mass.
template<bool fused = false>
static __global__ void h3_sol_centroids(const float * q, const half * k, const half * v,
                                      float * pooled_q, float * pooled_k, float * summed_v,
                                      int tokens, int blocks, float scale) {
    const int c = threadIdx.x, b = blockIdx.x, h = blockIdx.y;
    const int count = min(64, tokens - b*64);
    float sq = 0.f, sk = 0.f, sv = 0.f;
    const half scale_h = __float2half(scale);
    for (int t = 0; t < count; ++t) {
        const int64_t index = (int64_t(h)*tokens + b*64 + t)*128 + c;
        const float effective = __half2float(__hmul(__float2half(q[index]), scale_h));
        if constexpr (fused) {
            sq += __bfloat162float(__float2bfloat16(effective));
        } else {
            sq += effective;
        }
        sk += __half2float(k[index]);
        sv += __half2float(v[index]);
    }
    pooled_q[(h*blocks + b)*128 + c] = sq / count;
    pooled_k[(h*blocks + b)*128 + c] = fused ? __bfloat162float(__float2bfloat16(sk / count)) : sk / count;
    summed_v[(h*blocks + b)*128 + c] = fused ? __bfloat162float(__float2bfloat16(sv)) : sv;
}

static __global__ void h3_sol_kc_stats(const float * kc, float * mean, float * variance, int blocks) {
    const int c = threadIdx.x, h = blockIdx.x;
    float total = 0.f, squared = 0.f;
    for (int b = 0; b < blocks; ++b) {
        const float x = kc[(h*blocks + b)*128 + c];
        total += x;squared = fmaf(x,x,squared);
    }
    const float mu = total / blocks;
    mean[h*128+c] = mu;
    variance[h*128+c] = fmaxf(squared/blocks-mu*mu,0.f);
}

static bool h3_sol_proxy_scores(cublasHandle_t handle, cudaStream_t stream,
        const float * q, const float * k, float * scores, int blocks) {
    const float one = 1.f, zero = 0.f;
    cublasPointerMode_t saved;
    CUBLAS_CHECK(cublasGetPointerMode(handle, &saved));
    if (saved != CUBLAS_POINTER_MODE_HOST) CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    const cublasStatus_t status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        blocks, blocks, 128, &one, k, CUDA_R_32F, 128, int64_t(blocks)*128,
        q, CUDA_R_32F, 128, int64_t(blocks)*128, &zero, scores, CUDA_R_32F,
        blocks, int64_t(blocks)*blocks, 56, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
    if (saved != CUBLAS_POINTER_MODE_HOST) CUBLAS_CHECK(cublasSetPointerMode(handle, saved));
    return status == CUBLAS_STATUS_SUCCESS;
}

static __global__ void h3_sol_routes(const float * pooled_q, const float * pooled_k,
        int * mask, int blocks, float scale, const float * kc_mean, const float * kc_variance,
        ggml_cuda_h3::sol_metadata meta, const float * proxy_scores = nullptr) {
    __shared__ float scores[1024];
    __shared__ float diag_mean[128], diag_variance[128];
    __shared__ unsigned int warp_counts[32];
    __shared__ float threshold;
    __shared__ bool finite;
    const int qb = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const bool video_q = qb*64 >= meta.video_start && (qb+1)*64 <= meta.video_end;
    if (!video_q) {
        for (int kb = tid; kb < blocks; kb += blockDim.x) mask[(h*blocks+qb)*blocks+kb] = 1;
        return;
    }
    if (tid < 128) {
        const float qc = pooled_q[(h*blocks+qb)*128+tid];
        diag_mean[tid] = qc*kc_mean[h*128+tid];
        diag_variance[tid] = qc*qc*kc_variance[h*128+tid];
    }
    for (int kb = tid; kb < blocks; kb += blockDim.x) {
        float score = 0.f;
        if (proxy_scores) score = proxy_scores[(h*blocks+qb)*blocks+kb];
        else for (int c = 0; c < 128; ++c) score += pooled_q[(h*blocks+qb)*128+c]*pooled_k[(h*blocks+kb)*128+c];
        scores[kb] = score;
    }
    GGML_UNUSED(scale);
    __syncthreads();
    if (tid == 0) {
        float mean = 0.f, variance = 0.f;
        finite = true;
        for (int b = 0; b < blocks; ++b) finite = finite && isfinite(scores[b]);
        for (int c = 0; c < 128; ++c) { mean += diag_mean[c]; variance += diag_variance[c]; }
        constexpr float log2e = 1.4426950408889634f;
        threshold = mean + meta.beta*sqrtf(variance + 1e-6f/(log2e*log2e));
        finite = finite && isfinite(threshold);
    }
    __syncthreads();
    unsigned int eligible_count = 0, pruned_count = 0;
    for (int kb = tid; kb < blocks; kb += blockDim.x) {
        const bool video_k = kb*64 >= meta.video_start && (kb+1)*64 <= meta.video_end;
        const int q0 = (qb*64-meta.video_start)/meta.frame_tokens;
        const int q1 = ((qb+1)*64-1-meta.video_start)/meta.frame_tokens;
        const int k0 = (kb*64-meta.video_start)/meta.frame_tokens;
        const int k1 = ((kb+1)*64-1-meta.video_start)/meta.frame_tokens;
        const bool nearby = k0 <= q1+1 && k1 >= q0-1;
        const bool eligible = video_k && !nearby;
        const bool prune = meta.mode == 6 && finite && eligible && scores[kb] < threshold;
        mask[(h*blocks+qb)*blocks+kb] = prune ? 0 : 1;
        eligible_count += eligible;
        pruned_count += prune;
    }
    for (int offset = 16; offset; offset >>= 1) {
        eligible_count += __shfl_down_sync(0xffffffff, eligible_count, offset);
        pruned_count += __shfl_down_sync(0xffffffff, pruned_count, offset);
    }
    if (tid%32 == 0) { warp_counts[2*(tid/32)] = eligible_count; warp_counts[2*(tid/32)+1] = pruned_count; }
    __syncthreads();
    if (tid == 0) {
        eligible_count = 0; pruned_count = 0;
        for (int warp = 0; warp < blockDim.x/32; ++warp) { eligible_count += warp_counts[2*warp]; pruned_count += warp_counts[2*warp+1]; }
        auto * stats = reinterpret_cast<unsigned int *>(mask + 56*blocks*blocks);
        atomicAdd(stats, eligible_count); atomicAdd(stats+1, pruned_count);
    }
}

#endif
