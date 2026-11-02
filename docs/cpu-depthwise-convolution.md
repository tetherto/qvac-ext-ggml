# Planar CPU depthwise convolution experiment

The F32 planar `CONV_2D_DW` path has a SIMD specialization for one-dimensional,
unit-stride, unit-dilation convolution. It evaluates independent time samples in
parallel and checks padding only at row edges. Other shapes and F16 kernels keep
the generic implementation. The specialization uses existing GGML SIMD mappings
and requires no model option or machine-specific dispatch rule.

## Measurements

Measured on 2026-11-02 with Windows 11, Ryzen 7 7700, GCC 14.2.0,
Release `-O3 -march=native`, OpenMP enabled, and tinyBLAS disabled. Baseline:
`2d14e3017eaa8a46aa615a4ec30c6179f115d0c7`. Both variants use identical build
options; only the CPU kernel differs. No power, firmware, or memory settings were
changed. Timings include graph execution overhead.

The kernel benchmark uses three warmups and the median of eleven timed runs,
batch one, nine taps, and symmetric padding of four samples. Across widths
137/750/3750 and channel counts 640/1024, speedups were 11.4-18.9x with one thread
and 1.8-15.0x with eight threads. Selected eight-thread results:

| Width | Channels | Baseline ms | SIMD ms | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 137 | 640 | 0.2117 | 0.1190 | 1.78x |
| 750 | 640 | 0.4883 | 0.1162 | 4.20x |
| 3750 | 640 | 3.0270 | 0.2021 | 14.98x |
| 3750 | 1024 | 3.6890 | 0.3126 | 11.80x |

Model measurements use TranscribePsy-250M-EU-sub Q8_0, the Parakeet CLI,
`test/samples/sample-16k.wav` (20.13 seconds), eight CPU threads, the default
window policy, three warmups and fifteen timed runs per process. The process
order is baseline, SIMD, SIMD, baseline. Model loading is excluded.

| Pair | Baseline encoder ms | SIMD encoder ms | Baseline inference ms | SIMD inference ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 403.821 | 404.826 | 481.642 | 478.223 |
| 2 | 428.516 | 432.012 | 505.300 | 511.873 |

There is no repeatable model-level improvement in these measurements. The kernel
speedup alone does not establish an inference speedup; other encoder operations
and run-to-run variation dominate this workload.

The model file SHA256 is
`a93ef324d38e410b5e49389db95d06364eddbb890b74a69f311e482d2fda6f4b`.

## Additional ASR checkpoints

The same baseline and optimized binaries were compared on three more Q8_0
checkpoints from the QVAC registry. Each model used the same 20.13-second
`sample-16k.wav`, eight CPU threads, default offline window policy, three warmups
and fifteen timed runs in each process. Process order was baseline, SIMD, SIMD,
baseline. Nemotron used `--language en-US`. The table pools the thirty timed
samples per variant and reports their medians; positive reductions mean lower
latency. Loading is excluded.

| Model | Baseline inference ms | SIMD inference ms | Inference reduction | Encoder reduction |
| --- | ---: | ---: | ---: | ---: |
| Parakeet-TDT 0.6B-v3 | 841.422 | 841.163 | 0.03% | -0.65% |
| Parakeet-Unified-en 0.6B | 1029.840 | 1015.216 | 1.42% | 1.35% |
| Nemotron 3.5 ASR Streaming 0.6B | 1140.236 | 1122.441 | 1.56% | 2.78% |

TDT has no meaningful change. Unified improves in both paired comparisons:
total latency reductions of 3.27% and 1.39%, with encoder reductions of 2.08%
and 1.12%. Nemotron's encoder improves in both comparisons (2.30% and 3.03%),
but total inference is 3.17% faster in one pair and 0.99% slower in the other.
Its decoder timings vary substantially, so the pooled total does not establish
a consistent end-to-end improvement. These are small offline fixture gains;
cache-aware streaming was not benchmarked.

Checkpoint provenance:

- TDT: `qvac_models_compiled/ggml/parakeet/2026-05-11/parakeet-tdt-0.6b-v3.q8_0.gguf`,
  SHA256 `0f26bd6cfe8ba53624ea67ed2df7ddaa70179efc82160264561a0f05b70ba23d`.
- Unified: `qvac_models_compiled/ggml/parakeet/2026-08-13/parakeet-unified-en-0.6b.q8_0.gguf`,
  SHA256 `48102e55d61aaa827b89af03d772106e37de6ee12e8746f278f6315e71d980ad`.
- Nemotron: `qvac_models_compiled/ggml/parakeet/2026-09-30/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf`,
  SHA256 `7bd5394dbf10a5f7537dd1817bffa9cefd97d200e14e45754328b90a4f8fe4b6`.

TDT and Unified hashes match the consumer's model manifest. All timed transcripts
match between variants. Separate runs on the same audio also produce identical
token ID sequences, but encoder tensors differ:

| Model | Encoder relative L2 difference | Maximum absolute difference |
| --- | ---: | ---: |
| TDT | 0.017529 | 0.006113 |
| Unified | 0.005746 | 0.008039 |
| Nemotron | 0.009567 | 0.023340 |

All three exceed a `0.005` encoder-difference threshold against the old CPU
kernel. This does not measure their error against NeMo; it reinforces the need
for broader numerical validation before promoting this experiment.

## Validation and limits

The new scalar-reference regression cases cover short rows, SIMD tails, kernels
of 1/9/31 taps, padding, batches, uneven thread partitions, F16 fallback, and
non-unit stride/dilation fallback. They and the existing 2D depthwise tests pass
with both native SIMD and portable AVX2 builds.

Baseline and optimized token IDs match on `jfk.wav`, `sample-16k.wav`, and
`fr-multipiece-16k.wav`. This is limited fixture coverage, not a WER evaluation.

The implementation prevents multiply-add contraction in the SIMD loop. It is
still not bitwise equivalent to the compiler-generated generic kernel: a
137-by-640 diagnostic differed by at most `4.77e-7`, and the Q8_0 model amplifies
rounding differences. On JFK the encoder relative L2 difference is `0.009366`
and maximum absolute difference is `0.003736`. A repeated baseline run is bitwise
stable. This encoder difference exceeds the `0.005` threshold used by some
existing encoder reference tests; no NeMo reference-test pass is claimed here.

Treat this as an experimental contribution. Model-level benefit and broader
numerical acceptance must be established before updating consumer dependency
pins. ARM SIMD and MSVC execution have not been measured.

## Reproduction

Build `test-conv1d-dw-planar` and `test-conv2d-dw` in a Release CPU build, keeping
compiler/backend options identical for the baseline and optimized revisions.

```sh
cmake --build build --config Release --target test-conv1d-dw-planar test-conv2d-dw
ctest --test-dir build -C Release -R 'test-conv(1d-dw-planar|2d-dw)$' --output-on-failure
build/bin/test-conv1d-dw-planar --bench
parakeet --model transcribepsy-250m-eu-sub.gguf --wav sample-16k.wav \
  --backend cpu --threads 8 --bench --bench-warmup 3 --bench-runs 15 \
  --bench-json result.json
```

Copy the new test source and its CMake registration into the baseline build to
benchmark the same harness without the specialization. Preserve separate runtime
library directories when comparing shared-library builds.
