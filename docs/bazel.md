# Bazel build and tests

Bazel builds the ggml core library and the CPU backend, and runs the C++ tests in `tests/` as `cc_test` targets.
CMake stays the build that the `ggml` port in `tetherto/qvac-registry-vcpkg` uses. The Bazel build mirrors that
port's configuration, so the tests exercise the code that ships.

## Run the tests

Install [Bazelisk](https://github.com/bazelbuild/bazelisk). It reads `.bazelversion` and fetches the pinned Bazel.

```bash
bazel test //...
```

Bazel caches test results: a second run executes only the tests whose inputs changed. To run one test, name it:
`bazel test //tests:test-convrot`.

| Command | What it does |
|---|---|
| `bazel test --config=asan //...` | AddressSanitizer |
| `bazel test --config=ubsan //...` | UndefinedBehaviorSanitizer, failing on the first report |
| `bazel test --config=tsan //...` | ThreadSanitizer |
| `bazel test --config=native //...` | `-march=native` or `-mcpu=native` instead of the portable baseline |
| `bazel test -c dbg //...` | Debug build without optimization |
| `bazel test --runs_per_test=20 //tests:test-diffusion-batch-ops` | Repeat a test to catch flakes |

Personal flags, such as `--disk_cache=~/.cache/bazel-disk`, go in an untracked `user.bazelrc`.

## Targets

| Target | CMake equivalent |
|---|---|
| `//:ggml-base` | `ggml-base` |
| `//:ggml-cpu` | `ggml-cpu`, static as with `GGML_CPU_STATIC` |
| `//:ggml` | `ggml`: the backend registry, with the CPU backend registered |
| `//tests:*` | the tests in `tests/CMakeLists.txt` that need no GPU |

## How the configuration matches the port

- Linux and Android: `GGML_BACKEND_DL` with a static CPU backend, as the port builds them. `test-backend-loader`
  runs only on these platforms.
- macOS: a static build, as the port builds it. The CPU backend uses Accelerate.
- Every platform: `GGML_MAX_NAME=160`, no OpenMP, no llamafile, `GGML_USE_CPU_REPACK`, and `-O3` in the default
  `opt` mode.
- x86-64 uses the port's fixed AVX2 set (`-msse4.2 -mf16c -mfma -mbmi2 -mavx -mavx2`), not `-march=native`. Bazel
  cache keys hold the flags, not the CPU, so an object built natively on an AVX-512 machine could be restored on a
  machine without AVX-512.
- arm64 uses the compiler's default target, as the port does.

`bazel/ggml_options.bzl` holds these settings. When the port changes a CMake option, change it there too.

## Adding a test

Add the source to `tests/CMakeLists.txt` as before, and a `cc_test` with the same name to `tests/BUILD.bazel`.
A test passes when it exits 0. A test that needs a GPU belongs next to that backend's Bazel target, with
`target_compatible_with` set so that `bazel test //...` skips it where the backend is not built.

## Not covered yet

- GPU backends: CUDA, Vulkan, Metal and OpenCL build only with CMake, so `test-mul-mat-convrot-gpu` and the CUDA
  tests have no Bazel target yet.
- Windows: the MSVC flags are not mapped yet, so every target is marked incompatible there.

## CI

`.github/workflows/bazel.yml` runs `bazel test //...` on every pull request and on pushes to the dated branches. It
covers Linux x86-64 and arm64, macOS arm64, and ASan, UBSan and TSan on Linux x86-64. `setup-bazel` keeps the
Bazelisk download, the repository cache and the disk cache in the GitHub Actions cache. Pull requests restore those
caches, and pushes to a dated branch save them.
