#include "common.cuh"

#ifdef GGML_CUDA_CUDNN_SDPA
bool ggml_cuda_h3_cudnn_sdpa_enabled(int device);
bool ggml_cuda_try_cudnn_sdpa(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#endif

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);
