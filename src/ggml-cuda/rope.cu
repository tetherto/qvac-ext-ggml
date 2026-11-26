#include "convert.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "rope.cuh"

struct rope_flux_strides {
    size_t x[4];
    size_t pe[4];
};

static __device__ __forceinline__ float rope_flux_dot(float x0, float p0, float x1, float p1) {
#if defined(GGML_USE_HIP)
    float a = x0 * p0;
    float b = x1 * p1;
    // Keep HIP fast-math from contracting products across separate graph operations.
    asm volatile("" : "+v"(a), "+v"(b));
    return a + b;
#else
    return __fadd_rn(__fmul_rn(x0, p0), __fmul_rn(x1, p1));
#endif
}

template <bool has_pe, bool split_half = false>
static __global__ void rope_flux_f32(
        const char * x, const char * pe, float * dst,
        const int d_head, const int n_head, const int length,
        const int pairs, const rope_flux_strides strides) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pairs) {
        return;
    }

    const int pair = i % (d_head / 2);
    const int token = (i / (d_head / 2)) % length;
    const int batch_head = i / ((d_head / 2) * length);
    const int d0 = split_half ? pair : 2 * pair;
    const int d1 = split_half ? pair + d_head / 2 : 2 * pair + 1;
    const size_t offset = (batch_head / n_head) * strides.x[3] +
        token * strides.x[2] + (batch_head % n_head) * strides.x[1] +
        d0 * strides.x[0];
    const float even = *reinterpret_cast<const float *>(x + offset);
    const float odd = *reinterpret_cast<const float *>(x + offset + (d1 - d0) * strides.x[0]);
    const int out0 = split_half ? (i / (d_head / 2)) * d_head + d0 : 2 * i;
    const int out1 = split_half ? out0 + d_head / 2 : 2 * i + 1;

    if (has_pe) {
        const char * p = pe + token * strides.pe[3] + pair * strides.pe[2];
        const float p00 = *reinterpret_cast<const float *>(p);
        const float p01 = *reinterpret_cast<const float *>(p + strides.pe[0]);
        const float p10 = *reinterpret_cast<const float *>(p + strides.pe[1]);
        const float p11 = *reinterpret_cast<const float *>(p + strides.pe[0] + strides.pe[1]);
        // Preserve the separate products and sum used by the unfused graph.
        dst[out0] = rope_flux_dot(even, p00, odd, p01);
        dst[out1] = rope_flux_dot(even, p10, odd, p11);
    } else {
        dst[out0] = even;
        dst[out1] = odd;
    }
}

void ggml_cuda_op_rope_flux(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * pe = dst->src[1];
    GGML_ASSERT(x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_nelements(dst) <= INT32_MAX);

    rope_flux_strides strides = {};
    for (int d = 0; d < 4; ++d) {
        strides.x[d] = x->nb[d];
        strides.pe[d] = pe ? pe->nb[d] : 0;
    }
    const int pairs = ggml_nelements(dst) / 2;
    const dim3 blocks((pairs + CUDA_ROPE_BLOCK_SIZE - 1) / CUDA_ROPE_BLOCK_SIZE);
    const char * x_data = static_cast<const char *>(x->data);
    if (dst->op_params[0] != 0) {
        if (pe) {
            rope_flux_f32<true, true><<<blocks, CUDA_ROPE_BLOCK_SIZE, 0, ctx.stream()>>>(
                    x_data, static_cast<const char *>(pe->data), static_cast<float *>(dst->data),
                    x->ne[0], x->ne[1], x->ne[2], pairs, strides);
        } else {
            rope_flux_f32<false, true><<<blocks, CUDA_ROPE_BLOCK_SIZE, 0, ctx.stream()>>>(
                    x_data, nullptr, static_cast<float *>(dst->data),
                    x->ne[0], x->ne[1], x->ne[2], pairs, strides);
        }
    } else if (pe) {
        GGML_ASSERT(pe->type == GGML_TYPE_F32);
        rope_flux_f32<true><<<blocks, CUDA_ROPE_BLOCK_SIZE, 0, ctx.stream()>>>(
            x_data, static_cast<const char *>(pe->data), static_cast<float *>(dst->data),
            x->ne[0], x->ne[1], x->ne[2], pairs, strides);
    } else {
        rope_flux_f32<false><<<blocks, CUDA_ROPE_BLOCK_SIZE, 0, ctx.stream()>>>(
            x_data, nullptr, static_cast<float *>(dst->data),
            x->ne[0], x->ne[1], x->ne[2], pairs, strides);
    }
}

struct rope_corr_dims {
    float v[2];
};


struct mrope_sections {
    int v[4];
};

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
template<bool forward>
static __device__ void rope_yarn(
        const float theta_extrap, const float freq_scale, const rope_corr_dims corr_dims, const int64_t i0, const float ext_factor,
        float mscale, float & cos_theta, float & sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_norm(const T *            x,
                                 D *                  dst,
                                 const int            ne00,
                                 const int            ne01,
                                 const int            ne02,
                                 const int            s01,
                                 const int            s02,
                                 const int            s03,
                                 const int            s1,
                                 const int            s2,
                                 const int            s3,
                                 const int            n_dims,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 + i1 * s01 + i2 * s02 + i3 * s03;
    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0;
        idst += row_indices[i2] * set_rows_stride;
    }

    const auto & store_coaelsced = [&](float x0, float x1) {
        if constexpr (std::is_same_v<float, D>) {
            float2 v = make_float2(x0, x1);
            ggml_cuda_memcpy_1<8>(dst + idst, &v);
        } else if constexpr (std::is_same_v<half, D>) {
            half2 v = make_half2(x0, x1);
            ggml_cuda_memcpy_1<4>(dst + idst, &v);
        }
    };
    if (i0 >= n_dims) {
        store_coaelsced(x[ix + 0], x[ix + 1]);
        return;
    }

    const float theta_base = pos[i2]*powf(theta_scale, i0/2.0f);

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta, x0 * sin_theta + x1 * cos_theta);
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_neox(const T *            x,
                                 D *                  dst,
                                 const int            ne00,
                                 const int            ne01,
                                 const int            ne02,
                                 const int            s01,
                                 const int            s02,
                                 const int            s03,
                                 const int            s1,
                                 const int            s2,
                                 const int            s3,
                                 const int            n_dims,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride) {
    ggml_cuda_pdl_lc();
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;
    ggml_cuda_pdl_sync();

    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    if (i0 >= n_dims) {
        dst[idst + i0 / 2 + 0] = ggml_cuda_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_cuda_cast<D>(x[ix + i0 / 2 + 1]);

        return;
    }

    const float theta_base = pos[i2]*powf(theta_scale, i0/2.0f);

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims/2];

    dst[idst + 0]          = ggml_cuda_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_dims / 2] = ggml_cuda_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

template <bool forward, bool has_ff, typename T>
static __global__ void rope_multi(const T *            x,
                                  T *                  dst,
                                  const int            ne00,
                                  const int            ne01,
                                  const int            ne02,
                                  const int            s01,
                                  const int            s02,
                                  const int            s03,
                                  const int            s1,
                                  const int            s2,
                                  const int            s3,
                                  const int            n_dims,
                                  const int32_t *      pos,
                                  const float          freq_scale,
                                  const float          ext_factor,
                                  const float          attn_factor,
                                  const rope_corr_dims corr_dims,
                                  const float          theta_scale,
                                  const float *        freq_factors,
                                  const mrope_sections sections,
                                  const bool           is_imrope) {
    const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ggml_cuda_pdl_sync();
    if (i0 >= n_dims) {
        dst[idst + i0/2 + 0] = x[ix + i0/2 + 0];
        dst[idst + i0/2 + 1] = x[ix + i0/2 + 1];

        return;
    }

    const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (is_imrope) {
        if (sector % 3 == 1 && sector < 3 * sections.v[1]) {         // h
            theta_base = pos[i2 + ne02 * 1] * powf(theta_scale, i0 / 2.0f);
        } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {  // w
            theta_base = pos[i2 + ne02 * 2] * powf(theta_scale, i0 / 2.0f);
        } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {  // t
            theta_base = pos[i2] * powf(theta_scale, i0 / 2.0f);
        } else {
            theta_base = pos[i2 + ne02 * 3] * powf(theta_scale, i0 / 2.0f);
        }
    } else {
        if (sector < sections.v[0]) {
            theta_base = pos[i2] * powf(theta_scale, i0 / 2.0f);
        } else if (sector >= sections.v[0] && sector < sec_w) {
            theta_base = pos[i2 + ne02 * 1] * powf(theta_scale, i0 / 2.0f);
        } else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 2] * powf(theta_scale, i0 / 2.0f);
        } else if (sector >= sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 3] * powf(theta_scale, i0 / 2.0f);
        }
    }

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims/2];

    dst[idst + 0]        = x0*cos_theta - x1*sin_theta;
    dst[idst + n_dims/2] = x0*sin_theta + x1*cos_theta;
}

template <bool forward, bool has_ff, typename T>
static __global__ void rope_vision(const T *            x,
                                   T *                  dst,
                                   const int            ne00,
                                   const int            ne01,
                                   const int            ne02,
                                   const int            s01,
                                   const int            s02,
                                   const int            s03,
                                   const int            s1,
                                   const int            s2,
                                   const int            s3,
                                   const int            n_dims,
                                   const int32_t *      pos,
                                   const float          freq_scale,
                                   const float          ext_factor,
                                   const float          attn_factor,
                                   const rope_corr_dims corr_dims,
                                   const float          theta_scale,
                                   const float *        freq_factors,
                                   const mrope_sections sections) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ggml_cuda_pdl_sync();
    const int sect_dims = sections.v[0] + sections.v[1];
    const int sec_w     = sections.v[1] + sections.v[0];
    const int sector    = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (sector < sections.v[0]) {
        const int p = sector;
        theta_base  = pos[i2] * powf(theta_scale, p);
    } else if (sector >= sections.v[0] && sector < sec_w) {
        const int p = sector - sections.v[0];
        theta_base  = pos[i2 + ne02] * powf(theta_scale, p);
    }

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims];

    dst[idst + 0]      = x0*cos_theta - x1*sin_theta;
    dst[idst + n_dims] = x0*sin_theta + x1*cos_theta;
}

template <bool forward, typename T, typename D>
static void rope_norm_cuda(const T *            x,
                           D *                  dst,
                           const int            ne00,
                           const int            ne01,
                           const int            ne02,
                           const int            s01,
                           const int            s02,
                           const int            s03,
                           const int            s1,
                           const int            s2,
                           const int            s3,
                           const int            n_dims,
                           const int            nr,
                           const int32_t *      pos,
                           const float          freq_scale,
                           const float          freq_base,
                           const float          ext_factor,
                           const float          attn_factor,
                           const rope_corr_dims corr_dims,
                           const float *        freq_factors,
                           const int64_t *      row_indices,
                           const int            set_rows_stride,
                           cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        rope_norm<forward, false><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride);
    } else {
        rope_norm<forward, true><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride);
    }
}

template <bool forward, typename T, typename D>
static void rope_neox_cuda(const T *            x,
                           D *                  dst,
                           const int            ne00,
                           const int            ne01,
                           const int            ne02,
                           const int            s01,
                           const int            s02,
                           const int            s03,
                           const int            s1,
                           const int            s2,
                           const int            s3,
                           const int            n_dims,
                           const int            nr,
                           const int32_t *      pos,
                           const float          freq_scale,
                           const float          freq_base,
                           const float          ext_factor,
                           const float          attn_factor,
                           const rope_corr_dims corr_dims,
                           const float *        freq_factors,
                           const int64_t *      row_indices,
                           const int            set_rows_stride,
                           cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};

    if (freq_factors == nullptr) {
        ggml_cuda_kernel_launch(rope_neox<forward, false, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride);
    } else {
        ggml_cuda_kernel_launch(rope_neox<forward, true, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride);
    }
}

template <bool forward, typename T>
static void rope_multi_cuda(const T *            x,
                            T *                  dst,
                            const int            ne00,
                            const int            ne01,
                            const int            ne02,
                            const int            s01,
                            const int            s02,
                            const int            s03,
                            const int            s1,
                            const int            s2,
                            const int            s3,
                            const int            n_dims,
                            const int            nr,
                            const int32_t *      pos,
                            const float          freq_scale,
                            const float          freq_base,
                            const float          ext_factor,
                            const float          attn_factor,
                            const rope_corr_dims corr_dims,
                            const float *        freq_factors,
                            const mrope_sections sections,
                            const bool           is_imrope,
                            cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(rope_multi<forward, false, T>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope);
    } else {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(rope_multi<forward, true, T>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope);
    }
}

template <bool forward, typename T>
static void rope_vision_cuda(const T *            x,
                             T *                  dst,
                             const int            ne00,
                             const int            ne01,
                             const int            ne02,
                             const int            s01,
                             const int            s02,
                             const int            s03,
                             const int            s1,
                             const int            s2,
                             const int            s3,
                             const int            n_dims,
                             const int            nr,
                             const int32_t *      pos,
                             const float          freq_scale,
                             const float          freq_base,
                             const float          ext_factor,
                             const float          attn_factor,
                             const rope_corr_dims corr_dims,
                             const float *        freq_factors,
                             const mrope_sections sections,
                             cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);
    // break down (head_dim, heads, seq) into (CUDA_ROPE_BLOCK_SIZE, x, heads * seq)
    // where x ~= ceil(head_dim / CUDA_ROPE_BLOCK_SIZE);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_vision<forward, false, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    } else {
        rope_vision<forward, true, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    }
}

template <bool forward>
void ggml_cuda_op_rope_impl(ggml_backend_cuda_context & ctx,
                            ggml_tensor *               dst,
                            const ggml_tensor *         set_rows = nullptr) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;

    void *          dst_d           = dst->data;
    const int64_t * row_indices     = nullptr;
    ggml_type       dst_type        = dst->type;
    int             set_rows_stride = 0;

    if (set_rows != nullptr) {
        GGML_ASSERT(forward);
        dst_d           = set_rows->data;
        row_indices     = (const int64_t *) set_rows->src[1]->data;
        dst_type        = set_rows->type;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    // When not fused, src0 and dst types must match
    // When fused (ROPE+VIEW+SET_ROWS), src0 may be F32 and dst may be F16
    GGML_ASSERT(src0->type == dst->type || (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    const int64_t ne00 = src0->ne[0]; // head dims
    const int64_t ne01 = src0->ne[1]; // num heads
    const int64_t ne02 = src0->ne[2]; // num heads
    const int64_t nr = ggml_nrows(src0);

    const size_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    const size_t s02 = src0->nb[2] / ggml_type_size(src0->type);
    const size_t s03 = src0->nb[3] / ggml_type_size(src0->type);

    const size_t s1 = dst->nb[1] / ggml_type_size(dst->type);
    const size_t s2 = dst->nb[2] / ggml_type_size(dst->type);
    const size_t s3 = dst->nb[3] / ggml_type_size(dst->type);

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    mrope_sections sections;

    // RoPE alteration for extended context
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections.v,  (int32_t *) dst->op_params + 11, sizeof(int)*4);

    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 || sections.v[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne00/2);
    }

    const int32_t * pos = (const int32_t *) src1_d;

    const float * freq_factors = nullptr;
    if (src2 != nullptr) {
        freq_factors = (const float *) src2->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    // compute
    if (is_neox) {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                  set_rows_stride, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                 set_rows_stride, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                                                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                set_rows_stride, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_mrope && !is_vision) {
        if (src0->type == GGML_TYPE_F32) {
            rope_multi_cuda<forward>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                     s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                     corr_dims, freq_factors, sections, is_imrope, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_multi_cuda<forward>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                     s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                     corr_dims, freq_factors, sections, is_imrope, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_vision) {
        if (src0->type == GGML_TYPE_F32) {
            rope_vision_cuda<forward>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_vision_cuda<forward>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                  set_rows_stride, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                 set_rows_stride, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                s03, s1, s2, s3, n_dims, nr, pos, freq_scale, freq_base,
                                                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                set_rows_stride, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    }
}

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_rope_impl<true>(ctx, dst);
}

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_rope_impl<false>(ctx, dst);
}

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rope, ggml_tensor * set_rows) {
    ggml_cuda_op_rope_impl<true>(ctx, rope, set_rows);
}
