#include "common.cuh"
#include "convrot.cuh"
#include "convert.cuh"
#include <climits>
#include <cstdlib>
#include <cstring>

// H256 is symmetric: dot(H256 * (scale * w), x) ==
// scale * dot(w, H256 * x). Transform activations once instead of transforming
// every weight row. The scratch allocation is K * columns floats, never K*N.
template <typename T>
__global__ void convrot_rotate_input(const T * x, float * rotated) {
    __shared__ float v[256];
    const int tid = threadIdx.x;
    ggml_cuda_pdl_sync();
    const size_t offset = size_t(blockIdx.x) * 256;
    v[tid] = float(x[offset + tid]);
    __syncthreads();
    #pragma unroll
    for (int stride = 1; stride < 256; stride *= 4) {
        const int base = (tid / (4 * stride)) * (4 * stride) + tid % stride;
        const int lane = tid / stride % 4;
        const float a = v[base], b = v[base + stride];
        const float c = v[base + 2 * stride], d = v[base + 3 * stride];
        const float value = lane == 0 ? (a+b+c-d)*0.5f : lane == 1 ? (a+b-c+d)*0.5f :
                            lane == 2 ? (a-b+c+d)*0.5f : (-a+b+c+d)*0.5f;
        __syncthreads();
        v[tid] = value;
        __syncthreads();
    }
    rotated[offset + tid] = v[tid];
}

// Reconstruct the same F16 matrix produced by stable-diffusion.cpp's
// compatibility loader, one 256-element ConvRot tile at a time.  This is
// deliberately separate from the compact fused kernel: the caller below then
// invokes the standard F16 cuBLAS path, preserving its F16 input, accumulation
// and output semantics without keeping an F16 copy of every model weight.
__global__ void convrot_reconstruct_f16(const int8_t * weights, const float * scales,
                                        half * reconstructed, int k) {
    __shared__ float values[256];
    __shared__ float transformed[256];

    const int tid = threadIdx.x;
    ggml_cuda_pdl_sync();
    const int row = blockIdx.x;
    const int tile = blockIdx.y;
    const size_t offset = size_t(row) * k + size_t(tile) * 256;
    values[tid] = float(weights[offset + tid]) * scales[row];
    __syncthreads();
    for (int stride = 1; stride < 256; stride *= 4) {
        const int block = (tid / (4 * stride)) * (4 * stride);
        const int lane = (tid / stride) % 4;
        const int index = tid % stride;
        const float a = values[block + 0 * stride + index];
        const float b = values[block + 1 * stride + index];
        const float c = values[block + 2 * stride + index];
        const float d = values[block + 3 * stride + index];
        transformed[tid] = lane == 0 ? ( a + b + c - d) * 0.5f :
                           lane == 1 ? ( a + b - c + d) * 0.5f :
                           lane == 2 ? ( a - b + c + d) * 0.5f :
                                       (-a + b + c + d) * 0.5f;
        __syncthreads();
        values[tid] = transformed[tid];
        __syncthreads();
    }
    reconstructed[offset + tid] = __float2half_rn(values[tid]);
}

static bool ggml_cuda_op_mul_mat_convrot_f16_compat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * activations = dst->src[0];
    const ggml_tensor * weights = dst->src[1];
    const ggml_tensor * scales = dst->src[2];
    if (!ggml_is_contiguous(activations) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(scales) ||
        !ggml_is_contiguous(dst) || activations->ne[2] != 1 || activations->ne[3] != 1 ||
        weights->ne[0] > INT_MAX || weights->ne[1] > INT_MAX || activations->ne[1] > INT_MAX) {
        return false;
    }

    const int k = int(weights->ne[0]);
    const int n = int(weights->ne[1]);
    const int columns = int(activations->ne[1]);
    ggml_cuda_pool_alloc<half> reconstructed(ctx.pool(), size_t(k) * n);
    ggml_cuda_pool_alloc<half> activation_f16(ctx.pool(), ggml_nelements(activations));
    ggml_cuda_pool_alloc<half> output_f16(ctx.pool(), ggml_nelements(dst));

    const auto reconstruct_launch = ggml_cuda_kernel_launch_params(dim3(n, k / 256), dim3(256), 0, ctx.stream());
    ggml_cuda_kernel_launch(convrot_reconstruct_f16, reconstruct_launch,
                            (const int8_t *) weights->data, (const float *) scales->data,
                            reconstructed.get(), k);

    const auto to_f16 = ggml_get_to_fp16_cuda(activations->type);
    if (to_f16 == nullptr) {
        return false;
    }
    to_f16(activations->data, activation_f16.get(), ggml_nelements(activations), ctx.stream());

    static const half alpha = 1.0;
    static const half beta = 0.0;
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), ctx.stream()));
    CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                               n, columns, k,
                               &alpha, reconstructed.get(), CUDA_R_16F, k,
                                       activation_f16.get(), CUDA_R_16F, k,
                               &beta, output_f16.get(), CUDA_R_16F, n,
                               CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    ggml_get_to_fp32_cuda(GGML_TYPE_F16)(output_f16.get(), (float *) dst->data, ggml_nelements(dst), ctx.stream());
    return true;
}

// Four bytes per lane keep global weight reads coalesced. Each warp computes
// one output row for up to four columns, reusing each weight load. All dot
// products use FP32 FMA and shuffle reduction (no FP16 conversion).
template <int Columns>
__global__ void convrot_i8_matvec(const float * x, const int8_t * w, const float * scales,
                                float * dst, int k, int n, int columns) {
    const int lane = threadIdx.x;
    ggml_cuda_pdl_sync();
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n) return;
    const int first_column = blockIdx.y * Columns;
    x += size_t(first_column) * k;
    const int * packed = reinterpret_cast<const int *>(w + size_t(row) * k);
    float sums[Columns][4] = {};
    for (int i = lane; i < k / 4; i += 32) {
        const int q = packed[i];
        const float w0 = float(int8_t(q)), w1 = float(int8_t(q >> 8));
        const float w2 = float(int8_t(q >> 16)), w3 = float(int8_t(q >> 24));
        #pragma unroll
        for (int c = 0; c < Columns; ++c) {
            if (first_column + c < columns) {
                const float4 a = reinterpret_cast<const float4 *>(x + size_t(c)*k)[i];
                sums[c][0] = fmaf(w0, a.x, sums[c][0]);
                sums[c][1] = fmaf(w1, a.y, sums[c][1]);
                sums[c][2] = fmaf(w2, a.z, sums[c][2]);
                sums[c][3] = fmaf(w3, a.w, sums[c][3]);
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < Columns; ++c) {
        float sum = (sums[c][0] + sums[c][1]) + (sums[c][2] + sums[c][3]);
        for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0 && first_column + c < columns) dst[row + size_t(first_column+c) * n] = sum * scales[row];
    }
}

template <typename T>
__global__ void mul_mat_convrot_cuda(
        const char * activations, const char * weights, const char * scales, char * dst,
        int k, int ne02,
        size_t nb00, size_t nb01, size_t nb02, size_t nb03,
        size_t nb10, size_t nb11, size_t nb20,
        size_t nb0, size_t nb1, size_t nb2, size_t nb3) {
    __shared__ float values[256];
    __shared__ float transformed[256];

    const int tid = threadIdx.x;
    ggml_cuda_pdl_sync();
    const int row = blockIdx.x;
    const int i1 = blockIdx.y;
    const int i2 = blockIdx.z % ne02;
    const int i3 = blockIdx.z / ne02;
    const float scale = *reinterpret_cast<const float *>(scales + row*nb20);
    const char * activation = activations + i1*nb01 + i2*nb02 + i3*nb03;
    const char * weight_row = weights + row*nb11;

    float sum = 0.0f;
    for (int k0 = 0; k0 < k; k0 += 256) {
        values[tid] = float(*reinterpret_cast<const int8_t *>(weight_row + (k0 + tid)*nb10)) * scale;
        __syncthreads();
        for (int stride = 1; stride < 256; stride *= 4) {
            const int block = (tid / (4*stride)) * (4*stride);
            const int lane = (tid / stride) % 4;
            const int offset = tid % stride;
            const float a = values[block + 0*stride + offset];
            const float b = values[block + 1*stride + offset];
            const float c = values[block + 2*stride + offset];
            const float d = values[block + 3*stride + offset];
            transformed[tid] = lane == 0 ? ( a + b + c - d)*0.5f :
                               lane == 1 ? ( a + b - c + d)*0.5f :
                               lane == 2 ? ( a - b + c + d)*0.5f :
                                           (-a + b + c + d)*0.5f;
            __syncthreads();
            values[tid] = transformed[tid];
            __syncthreads();
        }
        sum += values[tid] * float(*reinterpret_cast<const T *>(activation + (k0 + tid)*nb00));
        __syncthreads();
    }
    values[tid] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride /= 2) {
        if (tid < stride) values[tid] += values[tid + stride];
        __syncthreads();
    }
    if (tid == 0) *reinterpret_cast<float *>(dst + row*nb0 + i1*nb1 + i2*nb2 + i3*nb3) = values[0];
}

void ggml_cuda_op_mul_mat_convrot(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * activations = dst->src[0];
    const ggml_tensor * weights = dst->src[1];
    const ggml_tensor * scales = dst->src[2];
    GGML_ASSERT(ggml_get_op_params_i32(dst, 0) == 256);
    GGML_ASSERT(activations->ne[0] == weights->ne[0] && weights->ne[0] % 256 == 0);
    const char * f16_compat_mode = std::getenv("GGML_CUDA_CONVROT_F16_MATMUL");
    const bool f16_compat_requested = ggml_get_op_params_i32(dst, 1) != 0 ||
                                      (f16_compat_mode != nullptr && std::strcmp(f16_compat_mode, "0") != 0);
    if (f16_compat_requested &&
        ggml_cuda_op_mul_mat_convrot_f16_compat(ctx, dst)) {
        return;
    }
    if (ggml_is_contiguous(activations) && ggml_is_contiguous(weights) &&
        ggml_is_contiguous(scales) && ggml_is_contiguous(dst) &&
        uintptr_t(weights->data) % alignof(int) == 0 &&
        weights->ne[0] <= INT_MAX && weights->ne[1] <= INT_MAX - 3 &&
        ggml_nelements(activations) / 256 <= INT_MAX &&
        ggml_nrows(activations) <= 65535 && ggml_cuda_info().devices[ctx.device].warp_size == 32) {
        const int k = int(weights->ne[0]);
        const int n = int(weights->ne[1]);
        const int columns = int(ggml_nrows(activations));
        ggml_cuda_pool_alloc<float> rotated(ctx.pool(), ggml_nelements(activations));
        const auto rotate_launch = ggml_cuda_kernel_launch_params(dim3(ggml_nelements(activations) / 256), dim3(256), 0, ctx.stream());
        if (activations->type == GGML_TYPE_F32) {
            ggml_cuda_kernel_launch(convrot_rotate_input<float>, rotate_launch, (const float *) activations->data, rotated.get());
        } else {
            ggml_cuda_kernel_launch(convrot_rotate_input<half>, rotate_launch, (const half *) activations->data, rotated.get());
        }
        const int columns_per_block = columns > 1 ? 4 : 1;
        const auto dot_launch = ggml_cuda_kernel_launch_params(
            dim3((n + 3) / 4, (columns + columns_per_block - 1) / columns_per_block),
            dim3(32, 4), 0, ctx.stream());
        if (columns > 1) {
            ggml_cuda_kernel_launch(convrot_i8_matvec<4>, dot_launch, rotated.get(), (const int8_t *) weights->data,
                                   (const float *) scales->data, (float *) dst->data, k, n, columns);
        } else {
            ggml_cuda_kernel_launch(convrot_i8_matvec<1>, dot_launch, rotated.get(), (const int8_t *) weights->data,
                                   (const float *) scales->data, (float *) dst->data, k, n, columns);
        }
        return;
    }

    const dim3 grid(weights->ne[1], activations->ne[1], activations->ne[2] * activations->ne[3]);
    const auto launch = ggml_cuda_kernel_launch_params(grid, dim3(256), 0, ctx.stream());
    if (activations->type == GGML_TYPE_F32) {
        ggml_cuda_kernel_launch(mul_mat_convrot_cuda<float>, launch,
            (const char *) activations->data, (const char *) weights->data, (const char *) scales->data, (char *) dst->data,
            (int) activations->ne[0], (int) activations->ne[2], activations->nb[0], activations->nb[1], activations->nb[2], activations->nb[3],
            weights->nb[0], weights->nb[1], scales->nb[0], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
    } else {
        ggml_cuda_kernel_launch(mul_mat_convrot_cuda<half>, launch,
            (const char *) activations->data, (const char *) weights->data, (const char *) scales->data, (char *) dst->data,
            (int) activations->ne[0], (int) activations->ne[2], activations->nb[0], activations->nb[1], activations->nb[2], activations->nb[3],
            weights->nb[0], weights->nb[1], scales->nb[0], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
    }
}
