# Compact native CUDA ConvRot optimization

Measured 2026-09-14 on nv5090, CUDA device 0 (RTX 5090), CUDA toolkit 13.3.

The native `GGML_OP_MUL_MAT_CONVROT` implementation rotates each activation
group once, then multiplies compact I8 weights with FP32 accumulation and
per-output FP32 scales. The identity is
`dot(H * (s*w), x) = s * dot(w, H*x)` because the Comfy H256 matrix is symmetric.
Floating-point evaluation order changes; bitwise equality is not promised.

The fast path uses two internal CUDA launches within one native graph operation.
It is not a single fused CUDA kernel. No expanded weight matrix, hint API, or
new graph operation is required by this optimization. Scratch is `4*K*columns`
bytes. The dot kernel packs four I8 values per lane and reuses weights across up
to four columns. Noncontiguous layouts, unaligned weight views, unsupported
launch sizes and non-32-lane devices retain the existing kernel.

## Timings

Medians of seven rounds, 30 synchronized graph executions per round, after five
warmups per path. Variant order rotates between rounds. Transfers, allocations,
CPU reference calculation and model loading are excluded. These are warm,
repeated-weight microbenchmarks, not full-model video timings or cold-cache
bandwidth measurements. Compact weights can benefit from cache residency.

| K | N | Columns | Input | Native optimized (ms) | Hint FP16 default (ms) | Hint FP32 (ms) |
|---:|---:|---:|---|---:|---:|---:|
| 5376 | 14336 | 1 | F32 | 0.024577 | 0.099586 | 0.190311 |
| 5376 | 14336 | 1 | F16 | 0.024645 | 0.099893 | 0.190530 |
| 5376 | 14336 | 7 | F16 | 0.071041 | 0.102166 | 0.193952 |
| 5376 | 14336 | 16 | F32 | 0.147453 | 0.102171 | 0.196401 |
| 14336 | 5376 | 1 | F32 | 0.025572 | 0.100196 | 0.190896 |
| 512 | 17 | 3 | F32 | 0.007148 | 0.012788 | 0.006757 |

The original kernel from `f747005b`, rebuilt under the same harness on the same
host, measured **0.335661 ms** for K=5376, N=14336, columns=1, F32. The optimized
implementation is **13.66x faster than that original**, **4.05x faster than the
FP16 hint**, and **7.74x faster than the FP32 hint** on this case. At 16 columns
the FP16 hint remains faster; the native path does not universally beat GEMM.

## Accuracy and safety

The benchmark uses deterministic random I8 weights, non-power-of-two FP32 row
scales, and F32/F16 inputs. Its independent CPU double reference rotates the
weights rather than the activations. It checks 64 spread-out output rows per
column (all rows when fewer than 64), rejecting nonfinite values and enforcing
the fixed native bound `abs(error) <= 2e-6*sum(abs(products)) + 1e-5`.
The native pass/fail bound was not relaxed after measurement.

All six native cases passed (1,715 checked outputs). Native relative L2 errors
were 1.13e-7 to 3.38e-7. For the main single-column F32 case, maximum absolute
error was 2.82e-5 and relative L2 error was 1.47e-7. The original kernel's relative
L2 error on the same inputs was 1.27e-7: both are close to the double reference.

Hint precision diagnostics are reported separately and do not gate native
correctness. The batched FP32 hint exceeded the strict reference bound despite
requesting `GGML_PREC_F32`; this experiment does not establish why that backend
path loses precision. Do not describe all hint-FP32 shapes as FP32-accurate.

The old experiment's claim that its FP16 drift came entirely from materializing
weights was not established: that dataset's weights were exactly representable
as FP16. The revised test includes both default and explicitly FP32-accumulating
FP16 matmuls; both representation and backend arithmetic must be considered.

The CUDA unit test passes for F32/F16 contiguous and strided activation layouts.
Compute Sanitizer memcheck reports 0 errors; racecheck reports 0 hazards,
0 errors, and 0 warnings on those tests.

## Memory

For K=5376, N=14336, one column, logical per-path weight storage is:

| Path | Weight storage | Additional native scratch |
|---|---:|---:|
| Native compact I8 + FP32 scales | 73.554688 MiB | 21 KiB |
| Hint with FP16 weights | 147 MiB | N/A |
| Hint with FP32 weights | 294 MiB | N/A |

Each hint also needs the 256 KiB dense H matrix and its rotated activation
output. Values are tensor payload sizes, excluding allocator alignment/pooling,
driver allocations and other model tensors. The benchmark keeps all variants'
weights resident simultaneously; these figures are not total process VRAM.
Earlier reports labeled decimal MB figures as MiB; the values above correct
that unit error.

## Reproduce

Build `test-mul-mat-convrot-cuda` and `test-mul-mat-convrot-h256-bench` with
`GGML_CUDA=ON`, `GGML_BUILD_TESTS=ON`. Invoke the benchmark as:

```sh
./build/bin/test-mul-mat-convrot-cuda
./build/bin/test-mul-mat-convrot-h256-bench 5376 14336 1 0
./build/bin/test-mul-mat-convrot-h256-bench 5376 14336 16 0
compute-sanitizer --tool memcheck --error-exitcode 9 ./build/bin/test-mul-mat-convrot-cuda
compute-sanitizer --tool racecheck --error-exitcode 9 ./build/bin/test-mul-mat-convrot-cuda
```

Arguments are K, N, columns, and whether to use F16 activations (0 or 1).
The comparison branch includes the existing experimental hint implementation.
The native optimization itself is confined to `src/ggml-cuda/convrot.cu`.
Metal/Vulkan and end-to-end H3 generation have not been validated for this CUDA
optimization.
