# Audio8 fused matmul profiling

QVAC-26715 originally proposed routing Audio8's large fused `MUL_MAT+ADD` to
HMX. The October 7 profile used for that proposal already labels **every fused
call `hmx-tiled`**. Verify the kernel and workload before changing dispatch.

The source log is `audio8-raw/hex-profile.log` (not checked into this repository).
It contains two labelled runs, including the warmup despite its label saying
"not captured". The captured run contains:

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

The dominant captured fused shapes are F16 weights, F32 activations, and a
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
cmake --build ../hexagon-26715-build/android/ggml --target test-backend-ops htp-v79 -j 6
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

```sh
GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_OPSTAGE=1 GGML_HEXAGON_OPFUSION=1 \
  ./test-backend-ops test -b HTP0 -o AUDIO8_MUL_MAT_ADD

GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_OPSTAGE=1 GGML_HEXAGON_OPFUSION=0 \
  ./test-backend-ops test -b HTP0 -o AUDIO8_MUL_MAT_ADD

GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_OPSTAGE=1 GGML_HEXAGON_OPFUSION=1 \
  GGML_HEXAGON_PROFILE=1 \
  ./test-backend-ops perf -b HTP0 -o AUDIO8_MUL_MAT_ADD > audio8-matmul-profile.log 2>&1

GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_OPSTAGE=1 GGML_HEXAGON_OPFUSION=1 \
  GGML_HEXAGON_PROFILE=3 GGML_HEXAGON_OPTRACE=262144 \
  ./test-backend-ops perf -b HTP0 -o AUDIO8_MUL_MAT_ADD \
  -p 'm=96,n=184320,k=96,residual=1' > audio8-matmul-trace.log 2>&1

GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_OPSTAGE=1 GGML_HEXAGON_OPFUSION=1 \
  ./test-backend-ops perf -b HTP0 -o AUDIO8_MUL_MAT_ADD > audio8-matmul-perf.log 2>&1
```

The correctness cases cover full residual and broadcast bias epilogues,
activation row counts 1/4/5 across the current HVX/HMX threshold, incomplete
activation tiles, and partial output-column tiles. They use the existing
matmul NMSE tolerance of `5e-4`; they do not certify end-to-end speech parity.
Perf mode uses the four exact shapes above with each ADD operand type.

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
