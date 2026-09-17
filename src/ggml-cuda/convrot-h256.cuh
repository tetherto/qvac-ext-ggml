#include "common.cuh"

// Returns whether the native ComfyUI regular Hadamard H256 transform was used.
bool ggml_cuda_op_convrot_h256(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);
