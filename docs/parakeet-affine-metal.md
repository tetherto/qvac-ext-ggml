# Parakeet affine Metal follow-up

The opt-in affine matmul now dispatches separate INT4 and INT8 pipelines, hoists row/block metadata and uses aligned paired-byte loads plus two independent FP32 affine lanes. Effective FP16 rounding and FP32 accumulation remain unchanged; no dense weight cache. No CPU support or global enablement is added.

Research evidence: parakeet-research1878ba9, reports/plan_v3.36primitive cases across three rounds passed exact weight restoration and bitwise output comparisons against the previous kernel. Fresh matched whole-call comparisons preserve every transcript and reduce median paired latency about12.5–12.7% on32-row clean/noisy development scopes and5.6% on one642-second user clip. These are small development measurements, not hardware-independent guarantees.

The newer503.7MB Mosaic Compact recipe uses ordinary Q4_0/Q8_0 blocks and stock GGML kernels; this custom affine operation serves the older610.6MB affine recipe. Do not attribute Compact's results to this operation. The16-step decoder probe was not useful overall and is not included.

This feature PR targets speech. Existing opt-in/error behavior and contract limits remain. Engine recipe/export documentation is in the companion speechcpp PR337.
