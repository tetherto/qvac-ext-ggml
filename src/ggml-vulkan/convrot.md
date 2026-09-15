# ConvRot Vulkan paths

The compact I8 + FP32-scale operation remains available on supported Vulkan
devices. Single-column native evaluation uses the fused shader. General native
evaluation reconstructs FP32 weights and uses FP32 GEMM; the F16 compatibility
hint retains its separate F16 operand/output-rounding path.

On RADV GFX1151 (Strix), many-column native evaluation additionally uses a
validated hybrid. H256 rotates activations once in FP32; high and scaled residual
F16 parts then feed FP32-accumulate GEMMs. K=5376 uses eight partial high sums;
K>=7168 uses explicit K=512 products and FP32 accumulation. Output columns are
tiled to bound scratch. I8 weights are exactly representable as F16. Arithmetic
is tested against FP32 reconstruction, not claimed bitwise-identical to another
backend's reduction order.

A GPU range check rejects nonfinite values, half overflow, or nonzero rotated
values disappearing in both parts. Such inputs use the original FP32 path.
Other Vulkan devices, other K sizes, single-column operations, and the F16
compatibility hint retain their existing path. Persistent scratch is shared
between layers, with barriers before reuse; it is not a weight cache.

Diagnostics (presence enables either option):

- `GGML_VULKAN_CONVROT_TRACE=1`: log device/path selection and range fallback.
- `GGML_VULKAN_CONVROT_DISABLE_HYBRID=1`: select the original path for comparison.

`test-mul-mat-convrot-vulkan` covers native/F16 compatibility, repeated graphs,
scratch reuse, exact conversion probes, and hybrid range fallback. Production
benchmarks additionally accept `GGML_CONVROT_BENCH=1`, `GGML_CONVROT_BENCH_K`,
`GGML_CONVROT_BENCH_N`, and `GGML_CONVROT_BENCH_COLS`. These sample the full grid
and every 1024-column tile boundary. `GGML_CONVROT_BENCH_NATIVE_ONLY=1` omits the
unchanged compatibility cases from a native timing comparison.
