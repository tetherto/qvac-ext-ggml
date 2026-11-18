#ifndef TEST_CUDA_H3_SOL_REFERENCE_CUH
#define TEST_CUDA_H3_SOL_REFERENCE_CUH

#include "ggml-cuda/h3-sol-math.cuh"

// Scalar oracle for the GPU primitive tests; never dispatched by the runtime.
// Each Stream-K partial contributes its own centroid numerator and mass.
// The first/last key blocks of a partial remain exact, including short tails.
template<bool exact_terms = false>
static __device__ __forceinline__ void h3_sol_accumulate(
        const float2 * q, int q_stride, int q_start, int tokens,
        const float * kc, const float * vc, const int * route,
        int first, int stop, float scale, float * correction,
        const half * raw_k = nullptr, int stride_k = 0, const half * raw_v = nullptr, int stride_v = 0) {
    const int lane = threadIdx.x;
    const half scale_h = __float2half(scale);
    for (int row = threadIdx.y; row < 64; row += blockDim.y) {
        const int token = q_start + row;
        float query[4], numerator[4] = {};
        for (int i = 0; i < 4; ++i) {
            const int c = lane + 32*i;
            const float x = token < tokens ? reinterpret_cast<const float *>(q + token*q_stride)[c] : 0.f;
            query[i] = __half2float(__hmul(__float2half(x), scale_h));
        }
        float maximum = -FLT_MAX/2.f, mass = 0.f;
        for (int block = first + 1; block < stop - 1; ++block) {
            if (route[block] != 0 || token >= tokens) continue;
            const int length = min(64, tokens - 64*block);
            for (int t = 0; t < (exact_terms ? length : 1); ++t) {
            float score = 0.f;
            for (int i = 0; i < 4; ++i) {
                const int c = lane + 32*i;
                const float key = exact_terms ? __half2float(raw_k[(block*64+t)*stride_k+c]) : kc[block*128+c];
                score = fmaf(query[i], key, score);
            }
            for (int offset = 16; offset; offset >>= 1) score += __shfl_down_sync(0xffffffff, score, offset);
            score = __shfl_sync(0xffffffff, score, 0);
            const float next = fmaxf(maximum, score + FATTN_KQ_MAX_OFFSET);
            const float alpha = expf(maximum - next), probability = expf(score - next);
            for (int i = 0; i < 4; ++i) {
                const int c = lane + 32*i;
                const float value = exact_terms ? __half2float(raw_v[(block*64+t)*stride_v+c]) : vc[block*128+c];
                numerator[i] = numerator[i]*alpha + probability*value;
            }
            mass = mass*alpha + probability*(exact_terms ? 1 : length);
            maximum = next;
            }
        }
        float * out = correction + row*h3_sol_correction_stride;
        if (lane == 0) { out[0] = maximum; out[1] = mass; }
        for (int i = 0; i < 4; ++i) out[4 + lane + 32*i] = numerator[i];
    }
    __syncthreads();
}


#endif
