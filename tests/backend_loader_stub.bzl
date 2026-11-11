load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("//bazel:ggml_options.bzl", "CXX_STANDARD_OPTS", "dynamic_backends_only")

def backend_loader_stub(name, stub_name, score, fail_init = False):
    cc_binary(
        name = name,
        srcs = ["test-backend-loader-stub.cpp"],
        cxxopts = CXX_STANDARD_OPTS,
        linkshared = True,
        local_defines = [
            'STUB_NAME=\\"%s\\"' % stub_name,
            "STUB_SCORE=%d" % score,
        ] + (["STUB_FAIL_INIT"] if fail_init else []),
        target_compatible_with = dynamic_backends_only(),
        deps = [
            "//:ggml-base",
            "//:ggml-internal-headers",
        ],
    )
