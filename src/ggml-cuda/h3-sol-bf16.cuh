#ifndef GGML_CUDA_H3_SOL_BF16_CUH
#define GGML_CUDA_H3_SOL_BF16_CUH

#include "mma.cuh"
#include "fattn-common.cuh"
#include "h3-sol-math.cuh"

struct alignas(16) h3_sol_bf16_workspace {
    uint16_t q[64*128];
    uint16_t kv[16*128];
    uint16_t probability[64*16];
    float score[64*16];
    float maximum[64];
    float mass[64];
    float alpha[64];
};

static_assert(sizeof(h3_sol_bf16_workspace) == 27392, "H3 proxy shared-memory layout");

static __device__ __forceinline__ uint16_t h3_sol_bf16_round(float value) {
    const uint32_t bits = __float_as_uint(value);
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

static __device__ __forceinline__ void h3_sol_accumulate_bf16(
        const float2 * q, int q_stride, int q_start, int tokens,
        const float * kc, const float * vc, const int * route,
        int first, int stop, float scale, float * correction, void * shared_memory) {
#if defined(AMPERE_MMA_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    using A = ggml_cuda_mma::tile<16,8,nv_bfloat162>;
    using B = ggml_cuda_mma::tile<8,8,nv_bfloat162>;
    using C = ggml_cuda_mma::tile<16,8,float>;
    auto * workspace = static_cast<h3_sol_bf16_workspace *>(shared_memory);
    const int tid = threadIdx.y*32 + threadIdx.x;
    const int row_start = threadIdx.y*16;
    const half scale_h = __float2half(scale);
    for (int index = tid; index < 64*128; index += 128) {
        const int row = index/128, c = index%128, token = q_start+row;
        const float input = token < tokens ? reinterpret_cast<const float *>(q + token*q_stride)[c] : 0.f;
        const float effective = __half2float(__hmul(__float2half(input),scale_h));
        workspace->q[index] = h3_sol_bf16_round(effective);
    }
    if (tid < 64) {
        workspace->maximum[tid] = -FLT_MAX/2.f;
        workspace->mass[tid] = 0.f;
    }
    C output[16];
#pragma unroll
    for (int tile = 0; tile < 16; ++tile) {
#pragma unroll
        for (int l = 0; l < C::ne; ++l) output[tile].x[l] = 0.f;
    }
    __syncthreads();
    for (int start = first+1; start < stop-1; start += 16) {
        const int candidate = start+tid;
        const bool selected = tid < 16 && candidate < stop-1 && route[candidate] == 0;
        if (!__syncthreads_or(selected)) continue;
        for (int index = tid; index < 16*128; index += 128) {
            const int block = start+index/128, c = index%128;
            const float key = block < stop-1 && route[block] == 0 ? kc[block*128+c] : 0.f;
            workspace->kv[index] = h3_sol_bf16_round(key);
        }
        __syncthreads();
        C scores[2];
#pragma unroll
        for (int group = 0; group < 2; ++group) {
#pragma unroll
            for (int l = 0; l < C::ne; ++l) scores[group].x[l] = 0.f;
        }
#pragma unroll
        for (int channel = 0; channel < 128; channel += 16) {
            A query;
            ggml_cuda_mma::load_generic(query,
                reinterpret_cast<const nv_bfloat162 *>(workspace->q) + row_start*64 + channel/2,64);
#pragma unroll
            for (int group = 0; group < 2; ++group) {
                B key;
                ggml_cuda_mma::load_generic(key,
                    reinterpret_cast<const nv_bfloat162 *>(workspace->kv) + group*8*64 + channel/2,64);
                ggml_cuda_mma::mma(scores[group],query,key);
            }
        }
#pragma unroll
        for (int group = 0; group < 2; ++group) {
#pragma unroll
            for (int l = 0; l < C::ne; ++l) {
                const int row = row_start+C::get_i(l), col = group*8+C::get_j(l);
                workspace->score[row*16+col] = scores[group].x[l];
            }
        }
        __syncthreads();
        if (tid < 64) {
            float maximum = workspace->maximum[tid];
            for (int col = 0; col < 16; ++col) {
                const int block = start+col;
                if (block < stop-1 && route[block] == 0) maximum = fmaxf(maximum,workspace->score[tid*16+col]+FATTN_KQ_MAX_OFFSET);
            }
            const float alpha = expf(workspace->maximum[tid]-maximum);
            float mass = workspace->mass[tid]*alpha;
            for (int col = 0; col < 16; ++col) {
                const int block = start+col;
                const bool skipped = block < stop-1 && route[block] == 0;
                const float probability = skipped ? expf(workspace->score[tid*16+col]-maximum) : 0.f;
                if (skipped) mass += probability*min(64,tokens-64*block);
                workspace->probability[tid*16+col] = h3_sol_bf16_round(probability);
            }
            workspace->maximum[tid] = maximum;
            workspace->mass[tid] = mass;
            workspace->alpha[tid] = alpha;
        }
        for (int index = tid; index < 16*128; index += 128) {
            const int block = start+index/128, c = index%128;
            const float value = block < stop-1 && route[block] == 0 ? vc[block*128+c] : 0.f;
            workspace->kv[c*16+index/128] = h3_sol_bf16_round(value);
        }
        __syncthreads();
        A probability;
        ggml_cuda_mma::load_generic(probability,
            reinterpret_cast<const nv_bfloat162 *>(workspace->probability) + row_start*8,8);
#pragma unroll
        for (int tile = 0; tile < 16; ++tile) {
#pragma unroll
            for (int l = 0; l < C::ne; ++l) output[tile].x[l] *= workspace->alpha[row_start+C::get_i(l)];
            B value;
            ggml_cuda_mma::load_generic(value,
                reinterpret_cast<const nv_bfloat162 *>(workspace->kv) + tile*8*8,8);
            ggml_cuda_mma::mma(output[tile],probability,value);
        }
        __syncthreads();
    }
    if (tid < 64) {
        correction[tid*h3_sol_correction_stride] = workspace->maximum[tid];
        correction[tid*h3_sol_correction_stride+1] = workspace->mass[tid];
    }
#pragma unroll
    for (int tile = 0; tile < 16; ++tile) {
#pragma unroll
        for (int l = 0; l < C::ne; ++l) {
            const int row = row_start+C::get_i(l), c = tile*8+C::get_j(l);
            correction[row*h3_sol_correction_stride+4+c] = output[tile].x[l];
        }
    }
    __syncthreads();
#else
    GGML_UNUSED_VARS(q,q_stride,q_start,tokens,kc,vc,route,first,stop,scale,correction,shared_memory);
    NO_DEVICE_CODE;
#endif
}

#endif
