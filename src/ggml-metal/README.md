# Embedded precompiled Metal shaders on macOS

`GGML_METAL_EMBED_PRECOMPILED=ON` adds a precompiled shader library to the
existing embedded source. It requires `GGML_METAL_EMBED_LIBRARY=ON`, a macOS
target, and Xcode's `metal` and `metallib` tools through `xcrun`. The option is
off by default. No external shader resource needs to be installed.

If a newer Xcode installation reports a missing Metal Toolchain, install Apple's
component with `xcodebuild -downloadComponent MetalToolchain` before building.

The compiled library targets Metal 3.1 and macOS 14 with BF16 kernels. Devices
with BF16 support and without native Metal tensor operations try this library
first. Other devices retain runtime source compilation, including devices using
native tensor operations. A rejected precompiled library also falls back to the
embedded source.

This avoids source compilation at backend initialization on supported devices;
it does not eliminate compute pipeline specialization or model loading. Measure
backend initialization and complete process time separately from generation time.
