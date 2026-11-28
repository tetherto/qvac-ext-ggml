# Spark cuDNN attention

The optional SDPA path targets NVIDIA GB10 (SM 12.1), batch one, 56 heads and 128 channels per head. It accepts unmasked F32 queries with F16 keys and values and returns F32. The graph packs queries into F16, accumulates in F32 and converts the F16 output back to F32. This precision change can alter generated video and audio. Qualify the complete model at the intended resolution and sampling policy.

Configure with `GGML_CUDA=ON`, `GGML_CUDA_FA=ON`, `GGML_CUDA_CUDNN_SDPA=ON` and `GGML_CUDA_GRAPHS=OFF`. Supply cuDNN 9 via `CUDNN_ROOT` and the cuDNN frontend headers via `CUDNN_FRONTEND_ROOT`. The tested SDK versions are cuDNN 9.19 and cudnn-frontend 1.19. This build selects the path automatically for eligible attention on GB10. Set `GGML_CUDA_H3_CUDNN_SDPA=0` to use native attention. Builds with `GGML_CUDA_CUDNN_SDPA=OFF` require only the existing CUDA dependencies.

The same Spark build also selects BF16 cuBLAS with FP32 accumulation for the
validated MiniMax-H3 Q4_K QKV, attention output, and two MLP projections with
at least 1024 tokens. The weights remain Q4_K in storage. The projection path
follows the attention enable flag and supports an independent
`GGML_CUDA_H3_BF16_PROJECTIONS=0` override.

The CUDA backend keeps one plan per context on its primary stream. Capture, other streams, unsupported shapes and cuDNN plan failures use native attention. The plan's private scratch holds packed Q, packed output and the cuDNN workspace. Its limit is 512 MiB per context, including the workspace. For a sequence of S tokens, the two F16 buffers occupy `4 × 56 × 128 × S` bytes. The previous plan releases its scratch before a replacement allocation. The cap covers the explicit allocation. CUDA and cuDNN manage their own context and plan metadata. A failed scratch allocation or unsupported execution uses native attention. A CUDA device fault propagates through the backend.

With `GGML_BUILD_TESTS=ON`, run `ctest -V -R '^test-cuda-cudnn-sdpa-(off|on|default)$'` on a GB10. The tests compare changing inputs against a double-precision softmax reference at the production attention scale and check dispatch, plan reuse and masked fallback.

At 5,912 and 15,435 tokens the cuDNN test allows post-scale maximum absolute error of 0.0001 and relative L2 error of 0.001. In a separate comparison against the pristine CUDA backend, all 12 native output fingerprints matched across both builds. The long-sequence native reference reached 0.001084 maximum absolute error and 0.004853 relative L2 error, so its respective regression limits are 0.0012 and 0.0055. The 256-token native and masked cases retain the tighter limits.
