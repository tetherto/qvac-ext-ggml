#include "common.cuh"

// Native compact I8/F32 ConvRot mat-vec. Weights remain compact in device
// memory and each 256-value transform block lives only in shared memory.
void ggml_cuda_op_mul_mat_convrot(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
