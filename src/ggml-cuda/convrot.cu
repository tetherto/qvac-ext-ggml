#include "common.cuh"
#include "convrot.cuh"

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
