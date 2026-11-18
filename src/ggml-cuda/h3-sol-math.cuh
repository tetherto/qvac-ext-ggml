#ifndef GGML_CUDA_H3_SOL_MATH_CUH
#define GGML_CUDA_H3_SOL_MATH_CUH

#include "common.cuh"

static constexpr int h3_sol_correction_stride = 132;

static __device__ __forceinline__ float2 h3_sol_merge_metadata(float2 exact, float * correction) {
    const float maximum = fmaxf(exact.x, correction[0]);
    correction[2] = expf(exact.x - maximum);
    correction[3] = expf(correction[0] - maximum);
    return make_float2(maximum, exact.y*correction[2] + correction[1]*correction[3]);
}

#endif
