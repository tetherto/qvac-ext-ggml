#include "common.cuh"
#include "convrot-h256.cuh"

// This is the exact radix-4 convention used by ComfyUI ConvRot, not the
// binary Walsh ordering used by the upstream FWHT helper.
__global__ void convrot_h256_cuda(const float * src, float * dst, int64_t rows) {
    __shared__ float values[256];
    __shared__ float transformed[256];

    const int tid = threadIdx.x;
    const int64_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    values[tid] = src[row * 256 + tid];
    __syncthreads();
    for (int stride = 1; stride < 256; stride *= 4) {
        const int block = (tid / (4 * stride)) * (4 * stride);
        const int lane = (tid / stride) % 4;
        const int offset = tid % stride;
        const float a = values[block + 0 * stride + offset];
        const float b = values[block + 1 * stride + offset];
        const float c = values[block + 2 * stride + offset];
        const float d = values[block + 3 * stride + offset];
        transformed[tid] = lane == 0 ? ( a + b + c - d) * 0.5f :
                           lane == 1 ? ( a + b - c + d) * 0.5f :
                           lane == 2 ? ( a - b + c + d) * 0.5f :
                                       (-a + b + c + d) * 0.5f;
        __syncthreads();
        values[tid] = transformed[tid];
        __syncthreads();
    }
    dst[row * 256 + tid] = values[tid];
}

bool ggml_cuda_op_convrot_h256(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src->ne[0] != 256 ||
        !ggml_are_same_shape(src, dst) || !ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t rows = ggml_nrows(src);
    const auto launch = ggml_cuda_kernel_launch_params(dim3(rows), dim3(256), 0, ctx.stream());
    ggml_cuda_kernel_launch(convrot_h256_cuda, launch, (const float *) src->data, (float *) dst->data, rows);
    return true;
}
