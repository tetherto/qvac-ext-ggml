# Parakeet affine Metal follow-up

The opt-in affine matmul now dispatches separate INT4 and INT8 pipelines, hoists row/block metadata and uses aligned paired-byte loads plus two independent FP32 affine lanes. Effective FP16 rounding and FP32 accumulation remain unchanged; no dense weight cache. No CPU support or global enablement is added.

Research evidence: parakeet-research1878ba9, reports/plan_v3.36primitive cases across three rounds passed exact weight restoration and bitwise output comparisons against the previous kernel. Fresh matched whole-call comparisons preserve every transcript and reduce median paired latency about12.5–12.7% on32-row clean/noisy development scopes and5.6% on one642-second user clip. These are small development measurements, not hardware-independent guarantees.

Production Mosaic is a 486,693,152-byte GGUF using ordinary Q4_0/Q8_0 blocks and stock GGML kernels. Eight decoder matrices use Q8_0; the retained encoder recipe is unchanged. Its SHA256 is `9e17114b5773d1b1ed9b6a4fafb0f46427a3bb9a80b77e645c24576bda9188ef`. This custom affine operation serves the older 610.6 MB affine research artifact; it is not required for Mosaic on CPU, Metal, CUDA or HIP. Do not attribute Mosaic's results to this operation. The 16-step decoder probe was not useful overall and is not included.

The supplied Mosaic Compact GGUF is a separate 339,564,832-byte artifact,
SHA256 `8229311ec51911e7f305413f8d9430b1e0d357101ffb3b75d32feed6b8520780`.
Its Q4_0/Q8_0/Q3_K/IQ3_S tensors also use stock GGML kernels, without the
affine opt-in. The companion speechcpp PR prepares a separate matching Core ML
encoder sidecar for each model by decoding that model's exact quantized weights.
Core ML export and runtime routing belong to speechcpp; this GGML affine
operation is not used by either model's sidecar. The shared Compact format
does not imply inclusion of the research runtime's additional performance fusions.

This feature PR targets speech. Existing opt-in/error behavior and contract limits remain. Engine recipe/export documentation is in the companion speechcpp PR337.
