GGML_VERSION = "0.17.0"
GGML_COMMIT = "unknown"
GGML_MAX_NAME = 160
GGML_SCHED_MAX_COPIES = 4

C_STANDARD_OPTS = ["-std=gnu11"]
CXX_STANDARD_OPTS = ["-std=gnu++17"]
RELEASE_OPTIMIZATION_OPTS = ["-O3"]

SHARED_WARNING_OPTS = [
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-Wcast-qual",
    "-Wno-unused-function",
]

C_WARNING_OPTS = SHARED_WARNING_OPTS + [
    "-Wshadow",
    "-Wstrict-prototypes",
    "-Wpointer-arith",
    "-Wmissing-prototypes",
    "-Werror=implicit-int",
    "-Werror=implicit-function-declaration",
    "-Wdouble-promotion",
]

CXX_WARNING_OPTS = SHARED_WARNING_OPTS + [
    "-Wmissing-declarations",
    "-Wmissing-noreturn",
    "-Wextra-semi",
]

X86_64_PORTABLE_ISA_OPTS = [
    "-msse4.2",
    "-mf16c",
    "-mfma",
    "-mbmi2",
    "-mavx",
    "-mavx2",
]

X86_64_PORTABLE_ISA_DEFINES = [
    "GGML_SSE42",
    "GGML_F16C",
    "GGML_FMA",
    "GGML_BMI2",
    "GGML_AVX",
    "GGML_AVX2",
]

X86_64_NATIVE_ISA_OPTS = ["-march=native"]
AARCH64_NATIVE_ISA_OPTS = ["-mcpu=native"]

X86_64_CPU_SRCS = [
    "src/ggml-cpu/arch/x86/quants.c",
    "src/ggml-cpu/arch/x86/repack.cpp",
]

AARCH64_CPU_SRCS = [
    "src/ggml-cpu/arch/arm/quants.c",
    "src/ggml-cpu/arch/arm/repack.cpp",
]

ACCELERATE_DEFINES = [
    "GGML_USE_ACCELERATE",
    "ACCELERATE_NEW_LAPACK",
    "ACCELERATE_LAPACK_ILP64",
]

ACCELERATE_LINKOPTS = [
    "-framework",
    "Accelerate",
]

def _quoted_define(name, value):
    return '%s=\\"%s\\"' % (name, value)

def _on_dynamic_backends(values):
    return select({
        Label("//bazel:dynamic_backends"): values,
        "//conditions:default": [],
    })

def ggml_optimization_opts():
    return select({
        Label("//bazel:optimized"): RELEASE_OPTIMIZATION_OPTS,
        "//conditions:default": [],
    })

def ggml_c_opts():
    return C_STANDARD_OPTS + C_WARNING_OPTS

def ggml_cxx_opts():
    return CXX_STANDARD_OPTS + CXX_WARNING_OPTS

def supported_platforms():
    return select({
        Label("//bazel:windows"): ["@platforms//:incompatible"],
        "//conditions:default": [],
    })

def dynamic_backends_only():
    return select({
        Label("//bazel:dynamic_backends"): [],
        "//conditions:default": ["@platforms//:incompatible"],
    })

def ggml_source_defines():
    return [
        "GGML_SCHED_MAX_COPIES=%d" % GGML_SCHED_MAX_COPIES,
        "_XOPEN_SOURCE=600",
    ] + select({
        Label("//bazel:linux_or_android"): ["_GNU_SOURCE"],
        Label("//bazel:apple"): ["_DARWIN_C_SOURCE"],
        "//conditions:default": [],
    })

def ggml_base_public_defines():
    return ["GGML_MAX_NAME=%d" % GGML_MAX_NAME] + _on_dynamic_backends([
        "GGML_BACKEND_DL",
        "GGML_SHARED",
    ])

def ggml_base_private_defines():
    return ggml_source_defines() + [
        _quoted_define("GGML_VERSION", GGML_VERSION),
        _quoted_define("GGML_COMMIT", GGML_COMMIT),
    ] + _on_dynamic_backends(["GGML_BUILD"])

def ggml_base_linkopts():
    return select({
        Label("//bazel:linux"): ["-pthread", "-lm"],
        Label("//bazel:android"): ["-lm", "-ldl"],
        "//conditions:default": [],
    })

def ggml_cpu_arch_srcs():
    return select({
        Label("//bazel:x86_64"): X86_64_CPU_SRCS,
        Label("//bazel:aarch64"): AARCH64_CPU_SRCS,
        "//conditions:default": [],
    })

def ggml_cpu_isa_opts():
    return select({
        Label("//bazel:x86_64_native"): X86_64_NATIVE_ISA_OPTS,
        Label("//bazel:aarch64_native"): AARCH64_NATIVE_ISA_OPTS,
        Label("//bazel:x86_64"): X86_64_PORTABLE_ISA_OPTS,
        "//conditions:default": [],
    })

def ggml_cpu_arch_defines():
    return select({
        Label("//bazel:x86_64_native"): [],
        Label("//bazel:x86_64"): X86_64_PORTABLE_ISA_DEFINES,
        Label("//bazel:aarch64"): [],
        "//conditions:default": ["GGML_CPU_GENERIC"],
    })

def ggml_cpu_public_defines():
    return _on_dynamic_backends(["GGML_BACKEND_SHARED"])

def ggml_cpu_private_defines():
    return ggml_source_defines() + ggml_cpu_arch_defines() + [
        "GGML_USE_CPU_REPACK",
    ] + _on_dynamic_backends(["GGML_BACKEND_BUILD"]) + select({
        Label("//bazel:macos"): ACCELERATE_DEFINES,
        "//conditions:default": [],
    })

def ggml_cpu_linkopts():
    return select({
        Label("//bazel:macos"): ACCELERATE_LINKOPTS,
        "//conditions:default": [],
    })

def ggml_registry_private_defines():
    return ggml_source_defines() + _on_dynamic_backends(["GGML_BUILD"])

def ggml_registry_linkopts():
    return select({
        Label("//bazel:linux"): ["-ldl"],
        "//conditions:default": [],
    })
