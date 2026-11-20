#include "common.cuh"

void ggml_cuda_op_repeat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_sub(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_div(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_repeat_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_fused_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse);
void ggml_cuda_op_fused_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse);

#if defined(GGML_USE_HIP)
// Keep Strix H3 fusion within the hidden size and row range validated for this path.
constexpr int GGML_CUDA_H3_HIDDEN_SIZE = 5376;
constexpr int GGML_CUDA_H3_MAX_VALIDATED_ROWS = 15429;

void ggml_cuda_op_scale_residual_gfx1151(ggml_backend_cuda_context & ctx,
                                      const ggml_tensor * value,
                                      const ggml_tensor * scale,
                                      ggml_tensor * dst);
#endif
