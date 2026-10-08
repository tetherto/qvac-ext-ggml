# Audio8 matmul profiling and correctness

QVAC-26715 originally proposed routing Audio8's large fused `MUL_MAT+ADD` to
HMX. An HMX fused implementation exists, but **current Audio8 codec matmuls
request F32 precision and stay on HVX**. Historical HMX dispatch does not
establish the route taken by the current application.

## Experimental F32 codec panels

`GGML_HEXAGON_F16_F32_PANEL=1` opts into a 2x2 HVX DDR output panel for
unbatched F16-weight/F32-activation matmuls. It widens the weights to F32 and
reuses each weight and activation vector across two outputs. The existing
`vec_dot_f16_f32_uu_1x1` narrows activations to F16 internally, so this candidate
changes arithmetic as well as data reuse. F32 precision requested by a graph
does not by itself prove that the old DDR helper retains F32 activations.

Eligibility requires at least two weight and activation rows, K at least 64
and divisible by 32, scalar-contiguous inner axes, and no fused ADD. Padded row strides and
unaligned row bases are supported; K=96 and odd output dimensions have explicit
remainders. Batched, fused, and other layouts retain the existing routes. HMX
selection keeps its existing precedence; set `GGML_HEXAGON_MM_SELECT=2` when
testing the panel explicitly. The option applies only when the normal VTCM
path cannot be used. Eligible small shapes retain the existing VTCM kernel;
focused DDR tests use activation matrices larger than the device VTCM budget.

This option defaults off. Evaluate with `GGML_HEXAGON_OPSTAGE=3`, the exact-F32
oracle, fixed-code codec/PCM comparisons, and end-to-end Audio8 timing against
the same build with the option unset. Local compilation alone establishes no
device correctness or speedup. The measured codec shapes motivating the
candidate are `[192,192] x [192,71680]`, `[96,96] x [96,143360]`, and
`[384,384] x [384,17920]` in ggml dimension order.

October 8 isolated-kernel checks on the renewed QDC SM8750/v79 device
(`57dd7911`) passed all 14 initial independent-oracle cases. Three repeated
comparisons after an initial trial gave these medians (each test executable
internally ran 30 iterations for K=96 and 15 for K=192/384):

| K / output channels | Activation rows | Panel off (ms) | Panel on (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 96 | 184320 | 211.786 | 93.626 | 2.262x |
| 192 | 92160 | 243.832 | 118.364 | 2.060x |
| 384 | 23040 | 154.834 | 104.358 | 1.484x |

These large-shape probes use the same channel counts as the current profile
and the longer activation sequences from the historical fixture. They are
isolated matmuls, not a measurement of full Audio8 inference. Both variants
used the same candidate build, `OPPOLL=1`, `OPSTAGE=3`, `OPFUSION=1`,
`HOSTBUF=1`, and default matmul selection; only `F16_F32_PANEL` changed.
The repeated order was panel on, then off. Run with:

```sh
GGML_HEXAGON_F16_F32_PANEL=1 ./test-backend-ops perf -b HTP0 -p audio8_codec_f32=1
```

Raw evidence is in `hexagon-codec-build/results/device-oracle.log`,
`device-perf-repeat.log`, and `device-perf-summary.json` beside the checkout.
The DSP artifact SHA-256 was
`49b8e8d93f12b19494da69666c47b9f4b766bd08d62aba6dcc5c0b7485fbb041`.
The initial broad selector passed fixed-code CPU/OpenCL parity, but repeated
full S1 inference medians were 19.8494 s off versus 19.8996 s on. Paired profiles
matched all 67,539 leaf operations: 76 existing DDR matmuls saved 5,408.814 ms,
while replacing 41 existing VTCM matmuls added 5,341.744 ms. The selector now
preserves every normal VTCM choice and only replaces DDR fallbacks. The results below validate this narrower selector.

## Validated DDR-only panel result, October 8

On the renewed QDC SM8750/v79 device (`57dd7911`), the opt-in panel reduced
median S1 inference from **20.346 s to 14.603 s**: **1.3933x faster**, or
**28.23% less inference time**. The codec synthesis median fell from 12.4843 s
to 7.0899 s. These are same-device comparisons of the same binary, changing
only `GGML_HEXAGON_F16_F32_PANEL` between 0 and 1.

| Setting | Median inference (s) | Median codec synthesis (s) | Inference RTF |
| --- | ---: | ---: | ---: |
| Panel off | 20.346 | 12.4843 | 6.6381 |
| DDR-only panel on | 14.603 | 7.0899 | 4.7644 |

Each setting had three warmups and five timed runs, interleaved off/on.
The prompt was "The quick brown fox jumps over the lazy dog.", with greedy
sampling, default seed 42, four threads, and a 70-frame cap. Every run produced
the same 66-frame code sequence and 3.065034 s of audio. WAV hashes repeat
within each setting; cross-setting WAVs differ slightly and pass the numeric
gates (cosine `0.9999999404`, NMSE `1.19266e-7`). Inference uses the CLI's
`StageTimings.total_ms`, excluding model loading, not process wall time.

Both 3-frame and 66-frame fixed-code checks pass all five codec/PCM boundaries
against CPU reference. The 3-frame PCM is identical between panel settings.
For 66 frames, candidate PCM versus CPU has cosine `0.9999966593` and NMSE
`6.72147e-6`. Direct candidate versus OpenCL PCM has cosine `0.9999900834` and
NMSE `1.98341e-5`; all compared PCM is finite. Gates are cosine at least
0.9999 and NMSE at most 2e-4. This establishes bounded numerical agreement,
not bit-identical waveforms, universal prompt coverage, or subjective audio
quality. OpenCL comparison uses identical codes; differing free-running
CPU/OpenCL/Hexagon trajectories remain outside this optimization's claim.

Validation also passed 23 independent-oracle device cases, host reference
checks, host/Android/DSP v79 builds, and independent review. The cases cover
natural DDR routing, VTCM preservation, varied and precision-sensitive inputs,
odd dimensions, strides, offsets, bias, and excluded shapes. The existing
small VTCM arithmetic is intentionally retained. The option remains off by
default.

Use the corrected ggml host-buffer and speech key-cache fixes, then enable:

```sh
export GGML_HEXAGON_HOSTBUF=1
export GGML_HEXAGON_OPPOLL=1
export GGML_HEXAGON_OPSTAGE=3
export GGML_HEXAGON_OPFUSION=1
export GGML_HEXAGON_F16_F32_PANEL=1
```

Do not use `OPSTAGE=1`: it omits computation. Performance runs used
`GGML_HEXAGON_PROFILE=0` and default matmul selection. The model identities are
those listed in the corrected baseline below. Runtime routing commit was
`f4ab5509`, tests `e4e6ac15`, and speech runtime `3778e63d` (later documented at `0f75ca40`). The DSP is unchanged
from the isolated-kernel checks. The staged stripped host library SHA-256 is
`bc39c3acdb013644fec7adc08b93a7ee2a9c25d3aa7c1a000082b777954aa768`.

Evidence beside the checkout is under `hexagon-codec-build/`:

- `results/model-ddr/`: all raw logs, code/WAV outputs, and fixed-code PCM exports.
- `results/model-ddr-summary.json` and `summarize-model.py`: timings, hashes,
  waveform metrics and completeness checks for all ten timed runs.
- `stage/validate-audio-ddr.sh` and `results/model-validation-ddr.log`: exact
  run order, model hash verification, flags and commands.
- `results/device-oracle-ddr.log` and `results/ci-validation.txt`: test/build evidence.
- `results/profiles/`: paired profiles explaining why VTCM routes must be retained.

The OpenCL reference run uses the earlier OpenCL-enabled registry with the
same CPU/base libraries; the candidate registry is built for CPU/Hexagon.
This registry setup does not change the timed Hexagon library set. The earlier
polling-only baseline below came from a different QDC device serial; do not
combine the two speedup ratios as a same-device measurement.

## Corrected S1 baseline, October 8

On the same QDC Snapdragon 8 Elite, enabling polling reduced median inference
time from **24.5906 s to 19.4765 s**, a **1.2626x speedup** (20.80% less time).
The ten timed Hexagon runs produced identical code files and WAV files across
both settings. This is evidence for polling preserving the measured S1 output.

| Variant | Median inference (s) | Median process wall time (s) | Frames | Audio duration (s) | Inference RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hexagon baseline, `OPPOLL=0` | 24.5906 | 24.9704 | 66 | 3.065034 | 8.022945 |
| Hexagon polling, `OPPOLL=1` | 19.4765 | 19.8224 | 66 | 3.065034 | 6.354416 |
| OpenCL | 3.3727 | 4.7182 | 70 | 3.250794 | 1.037500 |

Inference time is Audio8's `StageTimings.total_ms`, which covers generation
after model loading. Process wall time additionally includes CLI startup,
loading, output writing, and teardown. RTF is inference seconds divided by the
actual WAV duration. Each column reports its own median of five timed runs.

Protocol: S1 text was "The quick brown fox jumps over the lazy dog." on QDC
`sa880573`, device serial `f3b4a4c5`, with Q8_0 LM/decoder models, greedy
decoding, four CPU threads, default seed 42, and a 70-frame cap. Each variant
had three warmups followed by five timed CLI invocations. Variants ran in
interleaved order `hex-base`, `hex-poll`, `opencl` on every repetition.
`GGML_HEXAGON_OPSTAGE=3`, `HOSTBUF=1`, `OPFUSION=1`, and `PROFILE=0` were fixed;
only `OPPOLL` changed between the two Hexagon variants. No diagnostic tensor
callback was enabled.

Runtime source was ggml `9ef658d3` (later documentation commit `73b087b1`) and
speech `3778e63d`. All variants used the same staged CLI and device library set.
That set included the corrected Hexagon host library from the standalone
Hexagon build alongside the OpenCL libraries. Model SHA-256 values were:

| Model | SHA-256 |
| --- | --- |
| `audio8-lm-q8_0.gguf` | `f33c58b2fd46320c01544eec961112b4b9778106ce883e86914dd38854072d40` |
| `audio8-codec-decoder-q8_0.gguf` | `84d4b88fc5b2274c2cbfd95c4ce1bdae7820eb1e2464b7bd715d3fe370895432` |

All five runs within each variant had repeatable code and WAV hashes. However,
Hexagon and OpenCL generated different codes and lengths: OpenCL reached the
70-frame cap, while Hexagon produced 66 frames. These measurements therefore
do not establish CPU/OpenCL/Hexagon parity or equal generated workloads. This
is a single-prompt baseline, not the earlier five-prompt suite. The completed
scope is the buffer correctness fix and this baseline; broader parity work
and further kernel/routing optimization remain follow-ups.

Local evidence resides under
`/home/alokr/code/speech-qvac/hexagon-26715-build/`:

- `results/baseline-S1.json`: per-run timings, medians, output hashes.
- `results/benchmark-S1.log`: run order and process timestamps, including warmups.
- `results/benchmark-device/{hex-base,hex-poll,opencl}-S1-{warmup1..3,timed1..5}.{log,codes,wav}`:
  raw CLI logs and generated outputs.
- `results/model-sha256.txt`: checked model identities.
- `device-stage/benchmark.sh`: device runner; `single` reproduces the S1 protocol.
- `summarize-benchmark.py`: reconstructs the summary from timestamps, logs, and WAVs.

After staging the matching build and models in `/data/local/tmp/qvac-26715`,
run `sh /data/local/tmp/qvac-26715/benchmark.sh single` on the device and capture
its output plus the generated `results/` files. Recompute the existing local
summary from the ggml checkout with:

```sh
python3 ../hexagon-26715-build/summarize-benchmark.py \
  ../hexagon-26715-build/results/benchmark-S1.log \
  ../hexagon-26715-build/results/benchmark-device
```

## Current application routing

In the speech checkout, `engines/tts/src/audio8/codec_ops.cpp` sets
`GGML_PREC_F32` on both `sum_taps` and `causal_conv_transpose` matmuls. These
requests were present in the original Audio8 implementation (`09566de3`).
The current ggml HMX eligibility check rejects that precision because HMX
multiplies in F16 (`be3edbd6`). Fused parameter selection also preserves the
original MUL_MAT node's precision (`ccb67cb8`); older code passed the ADD node,
which could bypass that check. Both ggml fixes are in the merged `8c85fb0f` base.

Fusion requires either an HMX kernel or a single activation row. Consequently,
the large current codec products use separate HVX `MUL_MAT` and ADD operations.
The October 8 three-frame `hex-base-profile.log` confirms zero `MUL_MAT+ADD`
calls; its largest codec products (`192:192 x 192:3072`,
`96:96 x 96:6144`, and `384:384 x 384:768`) are labelled `hvx-tiled`.
This trace used `OPSTAGE=3`, but preceded the quantized host-buffer correction
below and produced invalid outputs, so it establishes dispatch, not valid
end-to-end performance.

Preserve the explicit F32 requirement. Relaxing the HMX row threshold alone
does not make these products eligible. Any future precision change would need
separate application-level correctness evidence and must not silently override
`GGML_PREC_F32` in the backend.

A fresh S3 trace after both correctness fixes, with `OPSTAGE=3`, `OPPOLL=1`,
and `PROFILE=1`, also records zero `MUL_MAT+ADD` calls. It generated 70 frames.
The three largest groups were:

| Weight shape K:N | Activation shape K:M | Kernel | Share of leaf cycles |
| --- | --- | --- | ---: |
| 192:192 | 192:71680 | `hvx-flat` | 34.22% |
| 96:96 | 96:143360 | `hvx-flat` | 28.00% |
| 384:384 | 384:17920 | `hvx-flat` | 19.77% |

These groups account for about 82% of the 29,765,246,194 leaf cycles. Inclusive
batch cycles were 30,285,894,686 and must not be added to leaf cycles. Codec
synthesis took 13,132.9 ms of the reported 21,078.9 ms inference time. Logging
affects these timings; use the unprofiled S1 measurements above for the baseline.
The trace identifies current F32 codec matmuls as follow-up profiling targets;
it does not establish cross-backend parity or justify changing precision.
Local evidence is `results/hex-corrected-S3-profile.log` and its parsed
`results/hex-corrected-S3-profile.json` under the build directory above.

## Historical October 7 profile

The older profile labels every `MUL_MAT+ADD` call `hmx-tiled`. This shows that
the fused implementation was exercised in that run. Its exact repository
commits are not recorded, so the discrepancy with current routing cannot be
attributed to a specific historical build.

The source log is `audio8-raw/hex-profile.log` (not checked into this repository).
It contains two labelled runs, including the warmup despite its label saying
"not captured". It does not record `OPSTAGE`; therefore it establishes the
logged dispatch choices, not a validated full-compute performance baseline
(see the `OPSTAGE` requirement below). The captured run contains:

| Measurement | Captured run |
| --- | ---: |
| Fused calls | 98 |
| Fused cycles | 829,590,819 |
| All individual operation cycles | 2,687,987,851 |
| Inclusive OPBATCH cycles | 3,701,900,582 |
| Fused share of individual operation cycles | 30.86% |
| Fused share of inclusive batch cycles | 22.41% |

`OPBATCH` times the entire DSP batch, including its operations and batch setup
and cleanup. Adding it to individual operation times double-counts work. The
original 196 calls also combined the warmup and captured run. Neither DSP
percentage represents the share of end-to-end inference latency.

The dominant historical captured fused shapes are F16 weights, F32 activations, and a
full-size F32 partial convolution as the ADD operand:

| Weight shape K:N | Activation shape K:M | Calls | Cycles |
| --- | --- | ---: | ---: |
| 96:96 | 96:184320 | 18 | 291,346,079 |
| 192:192 | 192:92160 | 19 | 271,181,453 |
| 384:384 | 384:23040 | 18 | 133,180,485 |
| 768:768 | 768:2880 | 18 | 41,459,353 |

These are vocoder products with many activation rows. They are distinct from
the AR batch-one projection. The latter appears as an unfused F32 `MUL_MAT`
(`896:4096 x 896:1`) in this log. Relaxing a batch-one HMX predicate would not
change the fused operations above.

## Reproduce the analysis

Run from the ggml checkout:

```sh
python3 tests/test-hexagon-profile.py
python3 scripts/hexagon-profile.py /path/to/hex-profile.log --section 'profile run (captured)'
python3 scripts/hexagon-profile.py /path/to/hex-profile.log --json > profile.json
```

The analyzer preserves sections, repeated run labels, and HTP sessions
separately. Unlabelled input remains unlabelled; it cannot infer warmups from
the records. Groups retain operation, shape, types, and the selected kernel.
Malformed profile records fail rather than silently dropping timed work.

## Quantized weight buffer correctness

Before timing Audio8, verify the Hexagon weight-buffer contract:

```sh
./test-hexagon-buffer
GGML_HEXAGON_HOSTBUF=0 ./test-hexagon-buffer
```

The default Hexagon buffer advertises host-accessible storage when `HOSTBUF=1`
(the default). CPU fallback and tensor copies may read its data pointer without
calling the backend's `get_tensor`. Such storage must retain canonical ggml
quantized blocks. Uploading tiled DSP blocks there corrupts CPU fallback inputs
even if backend `set_tensor`/`get_tensor` roundtrips appear correct. Audio8 loads
weights into this default buffer, making this contract relevant to its Q8
projections.

Host buffers now retain canonical bytes; explicit `HTP0-REPACK` buffers and the
legacy non-host `HOSTBUF=0` mode retain tiled storage with conversion on upload
and readback. The test covers Q4_0, Q4_1, Q8_0, IQ4_NL, and MXFP4, checking API
roundtrips, host-visible bytes, copies to CPU, and partial transfers on host
buffers. It also checks explicit repack buffers when available. Run this test
on the device; the Hexagon portion is skipped on hosts without that backend.

On the October 8 QDC Snapdragon 8 Elite run, the pre-fix default buffer passed
API roundtrips but failed direct host-byte and CPU-copy checks for all five
quantized types. After the fix those checks pass with both `HOSTBUF=1` and
`HOSTBUF=0`, including explicit repack buffers in the default mode. The default
Audio8 three-frame smoke now produces nonzero codes, but codes still diverge
from CPU/OpenCL in the first frame. This establishes the buffer correction,
not end-to-end parity. The matched S1 timings above establish polling's
performance and output stability for that prompt; waveform finiteness and
perceptual quality remain separate validation requirements.

## Build the focused tests

Host correctness (CPU optimized path compared to the CPU reference):

```sh
cmake -S . -B ../hexagon-26715-build/host/ggml -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_BUILD_TESTS=ON \
  -DGGML_BUILD_EXAMPLES=OFF -DGGML_OPENMP=OFF
cmake --build ../hexagon-26715-build/host/ggml --target test-backend-ops -j 6
../hexagon-26715-build/host/ggml/bin/test-backend-ops test -b CPU -o AUDIO8_MUL_MAT_ADD
```

Android and the Snapdragon 8 Elite v79 DSP skeleton (adjust installation paths
for another machine):

```sh
cmake -S . -B ../hexagon-26715-build/android/ggml -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/home/alokr/android-sdk/android-ndk-r28b/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 \
  -DGGML_HEXAGON=ON -DGGML_OPENMP=OFF -DGGML_BUILD_TESTS=ON \
  -DGGML_BUILD_EXAMPLES=OFF -DPREBUILT_LIB_DIR=android_aarch64 \
  -DHEXAGON_SDK_ROOT=/home/alokr/code/speech-qvac/poc/toolchains/Hexagon_SDK/6.6.0.0 \
  -DHEXAGON_TOOLS_ROOT=/home/alokr/code/speech-qvac/poc/toolchains/Hexagon_SDK/6.6.0.0/tools/HEXAGON_Tools/19.0.07
cmake --build ../hexagon-26715-build/android/ggml --target test-backend-ops test-hexagon-buffer htp-v79 -j 6
```

`PREBUILT_LIB_DIR` is required by this SDK's CMake helpers even for the host
stub. The backend test binary, `src/libqvac-speech-ggml*.so`,
`src/ggml-hexagon/libqvac-speech-ggml-hexagon.so`,
`src/ggml-hexagon/libggml-htp-v79.so`, and the NDK's `libc++_shared.so` must be
staged together on the device. Preserve the matching ggml/backend/skel build.

## Device verification and trace

Run these commands in the staged directory on the device. Set `LD_LIBRARY_PATH`
and `ADSP_LIBRARY_PATH` to that directory using the device's existing FastRPC
setup. Record the device, model hashes, both repository commits, thread count,
thermal state, and environment alongside the logs.

`GGML_HEXAGON_OPSTAGE` is a bitmask: queueing is bit 0 (`1`), and computation
is bit 1 (`2`). Use `3` (the default) to execute both. The previously reported
"tuned" setting `OPSTAGE=1` sets `HTP_OPFLAGS_SKIP_COMPUTE`; it skips arithmetic
in kernels that honor that flag. Fused and unfused paths do not all handle it
identically. Timings obtained with `OPSTAGE=1` cannot establish a valid inference
speedup or correctness baseline. Repeat those measurements with `OPSTAGE=3`.

```sh
GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_OPSTAGE=3 GGML_HEXAGON_OPFUSION=1 \
  ./test-backend-ops test -b HTP0 -o AUDIO8_MUL_MAT_ADD

GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_OPSTAGE=3 GGML_HEXAGON_OPFUSION=0 \
  ./test-backend-ops test -b HTP0 -o AUDIO8_MUL_MAT_ADD

GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_OPSTAGE=3 GGML_HEXAGON_OPFUSION=1 \
  GGML_HEXAGON_PROFILE=1 \
  ./test-backend-ops perf -b HTP0 -o AUDIO8_MUL_MAT_ADD > audio8-matmul-profile.log 2>&1

GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_OPSTAGE=3 GGML_HEXAGON_OPFUSION=1 \
  GGML_HEXAGON_PROFILE=3 GGML_HEXAGON_OPTRACE=262144 \
  ./test-backend-ops perf -b HTP0 -o AUDIO8_MUL_MAT_ADD \
  -p 'm=96,n=184320,k=96,residual=1' > audio8-matmul-trace.log 2>&1

GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_OPSTAGE=3 GGML_HEXAGON_OPFUSION=1 \
  ./test-backend-ops perf -b HTP0 -o AUDIO8_MUL_MAT_ADD > audio8-matmul-perf.log 2>&1
```

The correctness cases cover full residual and broadcast bias epilogues,
activation row counts 1/4/5 across the current HVX/HMX threshold, incomplete
activation tiles, and partial output-column tiles. They use the existing
matmul NMSE tolerance of `5e-4`; they do not certify end-to-end speech parity.
Perf mode uses the four exact shapes above with each ADD operand type.
These synthetic graphs use default matmul precision to exercise the existing
fused implementation; they do not reproduce the current codec's explicit F32
precision requirement. Passing them does not authorize routing the real codec
through HMX.

`PROFILE=3` exposes `HVX_A_PREP`, `HVX_W_PREP`, `HVX_O_PROC`, `HMX_COMP`, and
DMA events. These distinguish conversion/output work from HMX multiplication.
Check each OPBATCH `evt-cnt` against the per-thread `OPTRACE` capacity before
using a trace: the logger clips events at that capacity. Pair events within
each batch and thread; timestamps are 32-bit cycle values and can wrap.
Concurrent thread intervals overlap and must not be summed as elapsed time.
The analyzer above summarizes operation timings, not these trace intervals.

Use the unprofiled run for timing claims; profiler logging changes the workload.
For complete Audio8 validation, compare matched Hexagon/OpenCL runs on the same
device, separate model loading from inference, use three warmups and five timed
runs, and check per-stage cosine >=0.9999 plus generated tokens, EOS, frame
count, and audio quality. Do not enable a routing change based only on these
synthetic backend tests.
