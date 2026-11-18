#ifndef GGML_CUDA_H3_SOL_FUSED_CUH
#define GGML_CUDA_H3_SOL_FUSED_CUH

#include "h3-sol-attn.h"
#include "h3-sol-bf16.cuh"

// Adapted from NVIDIA Sol-Attn's SM120 routing and centroid-mass design:
// NVlabs/Sana, cf67a3acad7511acb85053e56a05e5dd24c82ba0 (Apache-2.0).
// The retained ggml FP16 exact path and Stream-K reduction remain unchanged.
struct alignas(16) h3_sol_fused_header {
    ggml_cuda_h3::sol_metadata meta;
    int blocks;
    int reserved[2];
};
static_assert(sizeof(h3_sol_fused_header) == 48, "Private fused routing layout");

struct h3_sol_tile_data {
    const int * route = nullptr;
    unsigned int * stats = nullptr;
    const float * kc = nullptr;
    const float * vc = nullptr;
    float * correction = nullptr;
    const h3_sol_fused_header * header = nullptr;
};

static __device__ __forceinline__ h3_sol_tile_data h3_sol_fused_tile(const int * data, int head, int cta) {
    const auto * header = reinterpret_cast<const h3_sol_fused_header *>(data);
    const int centroids = 56 * header->blocks * 128;
    const float * base = reinterpret_cast<const float *>(data + 16);
    h3_sol_tile_data result;
    result.stats = reinterpret_cast<unsigned int *>(const_cast<int *>(data + 12));
    result.kc = base + centroids + head * header->blocks * 128;
    result.vc = base + 2 * centroids + head * header->blocks * 128;
    result.correction = const_cast<float *>(base + 3 * centroids + 2 * 56 * 128) + cta * 64 * h3_sol_correction_stride;
    result.header = header;
    return result;
}

struct alignas(16) h3_sol_fused_workspace {
    h3_sol_bf16_workspace math;
    float partial[256];
    float threshold;
    int finite;
    int skipped;
    int eligible;
};

// The route is CTA-local; the same Q*KC scores drive selection and correction.
static __device__ __forceinline__ int h3_sol_fused_accumulate(const float2 * q, int q_stride, int q_start, int tokens,
                                                              const float * kc, const float * vc, int * route,
                                                              int first, int stop, float scale, float * correction,
                                                              void * shared_memory, const h3_sol_fused_header * header,
                                                              int head) {
#if defined(AMPERE_MMA_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    using A = ggml_cuda_mma::tile<16, 8, nv_bfloat162>;
    using B = ggml_cuda_mma::tile<8, 8, nv_bfloat162>;
    using C = ggml_cuda_mma::tile<16, 8, float>;
    auto * scratch = static_cast<h3_sol_fused_workspace *>(shared_memory);
    auto * workspace = &scratch->math;
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int row_start = threadIdx.y * 16;
    const int blocks = header->blocks;
    const auto meta = header->meta;
    for (int block = tid; block < blocks; block += 128)
        route[block] = 1;
    __syncthreads();
    if (meta.mode != 6 || q_start < meta.video_start || q_start + 64 > meta.video_end)
        return 0;

    const float * base = reinterpret_cast<const float *>(header) + 16;
    const float * qc = base + (head * blocks + q_start / 64) * 128;
    const float * mean = base + 3 * 56 * blocks * 128 + head * 128;
    const float * variance = mean + 56 * 128;
    scratch->partial[tid] = qc[tid] * mean[tid];
    scratch->partial[128 + tid] = qc[tid] * qc[tid] * variance[tid];
    __syncthreads();
    if (tid == 0) {
        float mu = 0.f, var = 0.f;
        for (int c = 0; c < 128; ++c) {
            mu += scratch->partial[c];
            var += scratch->partial[128 + c];
        }
        constexpr float log2e = 1.4426950408889634f;
        scratch->threshold = mu + meta.beta * sqrtf(var + 1e-6f / (log2e * log2e));
        scratch->finite = isfinite(scratch->threshold);
        scratch->skipped = scratch->eligible = 0;
    }
    __syncthreads();
    if (!scratch->finite)
        return 0;
    const half scale_h = __float2half(scale);
    for (int index = tid; index < 64 * 128; index += 128) {
        const int row = index / 128, c = index % 128;
        const float input = reinterpret_cast<const float *>(q + (q_start + row) * q_stride)[c];
        workspace->q[index] = h3_sol_bf16_round(__half2float(__hmul(__float2half(input), scale_h)));
    }
    if (tid < 64) {
        workspace->maximum[tid] = -FLT_MAX / 2.f;
        workspace->mass[tid] = 0.f;
    }
    C output[16];
#pragma unroll
    for (int tile = 0; tile < 16; ++tile) {
#pragma unroll
        for (int l = 0; l < C::ne; ++l)
            output[tile].x[l] = 0.f;
    }
    __syncthreads();
    for (int start = first + 1; start < stop - 1; start += 16) {
        for (int index = tid; index < 16 * 128; index += 128) {
            const int block = start + index / 128, c = index % 128;
            workspace->kv[index] = h3_sol_bf16_round(block < stop - 1 ? kc[block * 128 + c] : 0.f);
        }
        __syncthreads();
        C scores[2];
#pragma unroll
        for (int group = 0; group < 2; ++group) {
#pragma unroll
            for (int l = 0; l < C::ne; ++l)
                scores[group].x[l] = 0.f;
        }
#pragma unroll
        for (int channel = 0; channel < 128; channel += 16) {
            A query;
            ggml_cuda_mma::load_generic(
                query, reinterpret_cast<const nv_bfloat162 *>(workspace->q) + row_start * 64 + channel / 2, 64);
#pragma unroll
            for (int group = 0; group < 2; ++group) {
                B key;
                ggml_cuda_mma::load_generic(
                    key, reinterpret_cast<const nv_bfloat162 *>(workspace->kv) + group * 8 * 64 + channel / 2, 64);
                ggml_cuda_mma::mma(scores[group], query, key);
            }
        }
#pragma unroll
        for (int group = 0; group < 2; ++group) {
#pragma unroll
            for (int l = 0; l < C::ne; ++l) {
                workspace->score[(row_start + C::get_i(l)) * 16 + group * 8 + C::get_j(l)] = scores[group].x[l];
            }
        }
        __syncthreads();
        float sum = 0.f;
        for (int row = tid / 16; row < 64; row += 8)
            sum += workspace->score[row * 16 + tid % 16];
        scratch->partial[tid] = sum;
        __syncthreads();
        if (tid < 16 && start + tid < stop - 1) {
            const int block = start + tid;
            float total = 0.f;
            for (int part = 0; part < 8; ++part)
                total += scratch->partial[part * 16 + tid];
            const float score = total / 64.f;
            const bool video = block * 64 >= meta.video_start && (block + 1) * 64 <= meta.video_end;
            const int q0 = (q_start - meta.video_start) / meta.frame_tokens;
            const int q1 = (q_start + 63 - meta.video_start) / meta.frame_tokens;
            const int k0 = (block * 64 - meta.video_start) / meta.frame_tokens;
            const int k1 = (block * 64 + 63 - meta.video_start) / meta.frame_tokens;
            const bool eligible = video && !(k0 <= q1 + 1 && k1 >= q0 - 1);
            const bool prune = eligible && isfinite(score) && score < scratch->threshold;
            route[block] = prune ? 0 : 1;
            if (eligible)
                atomicAdd(&scratch->eligible, 1);
            if (prune)
                atomicAdd(&scratch->skipped, 1);
        }
        __syncthreads();
        const bool selected = tid < 16 && start + tid < stop - 1 && route[start + tid] == 0;
        if (!__syncthreads_or(selected))
            continue;
        if (tid < 64) {
            float maximum = workspace->maximum[tid];
            for (int col = 0; col < 16; ++col)
                if (start + col < stop - 1 && route[start + col] == 0)
                    maximum = fmaxf(maximum, workspace->score[tid * 16 + col] + FATTN_KQ_MAX_OFFSET);
            const float alpha = expf(workspace->maximum[tid] - maximum);
            float mass = workspace->mass[tid] * alpha;
            for (int col = 0; col < 16; ++col) {
                const int block = start + col;
                const bool skipped = block < stop - 1 && route[block] == 0;
                const float probability = skipped ? expf(workspace->score[tid * 16 + col] - maximum) : 0.f;
                if (skipped)
                    mass += probability * min(64, tokens - 64 * block);
                workspace->probability[tid * 16 + col] = h3_sol_bf16_round(probability);
            }
            workspace->maximum[tid] = maximum;
            workspace->mass[tid] = mass;
            workspace->alpha[tid] = alpha;
        }
        for (int index = tid; index < 16 * 128; index += 128) {
            const int block = start + index / 128, c = index % 128;
            workspace->kv[c * 16 + index / 128] =
                h3_sol_bf16_round(block < stop - 1 && route[block] == 0 ? vc[block * 128 + c] : 0.f);
        }
        __syncthreads();
        A probability;
        ggml_cuda_mma::load_generic(probability,
                                    reinterpret_cast<const nv_bfloat162 *>(workspace->probability) + row_start * 8, 8);
#pragma unroll
        for (int tile = 0; tile < 16; ++tile) {
#pragma unroll
            for (int l = 0; l < C::ne; ++l)
                output[tile].x[l] *= workspace->alpha[row_start + C::get_i(l)];
            B value;
            ggml_cuda_mma::load_generic(value, reinterpret_cast<const nv_bfloat162 *>(workspace->kv) + tile * 8 * 8, 8);
            ggml_cuda_mma::mma(output[tile], probability, value);
        }
        __syncthreads();
    }
    if (tid < 64) {
        correction[tid * h3_sol_correction_stride] = workspace->maximum[tid];
        correction[tid * h3_sol_correction_stride + 1] = workspace->mass[tid];
    }
#pragma unroll
    for (int tile = 0; tile < 16; ++tile) {
#pragma unroll
        for (int l = 0; l < C::ne; ++l)
            correction[(row_start + C::get_i(l)) * h3_sol_correction_stride + 4 + tile * 8 + C::get_j(l)] =
                output[tile].x[l];
    }
    __syncthreads();
    if (tid == 0) {
        auto * stats = reinterpret_cast<unsigned int *>(const_cast<h3_sol_fused_header *>(header) + 1);
        atomicAdd(stats, static_cast<unsigned int>(scratch->eligible));
        atomicAdd(stats + 1, static_cast<unsigned int>(scratch->skipped));
    }
    return scratch->skipped;
#else
    GGML_UNUSED_VARS(q, q_stride, q_start, tokens, kc, vc, route, first, stop, scale, correction, shared_memory, header,
                     head);
    NO_DEVICE_CODE;
    return 0;
#endif
}

#endif
