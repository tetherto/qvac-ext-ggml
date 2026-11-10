#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda/h3-fattn.h"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static void log_tuning(ggml_log_level, const char * message, void * opaque) {
    if (std::strstr(message, "h3_fattn_prefetch ")) ++*static_cast<std::atomic<int> *>(opaque);
    std::fputs(message, stderr);
}

static float value(uint32_t & state) {
    state = state * 1664525u + 1013904223u;
    return (int(state >> 16) - 32768) / 32768.f;
}

static void unchanged(ggml_tensor * tensor, const void * expected) {
    std::vector<unsigned char> actual(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
    require(std::memcmp(actual.data(), expected, actual.size()) == 0, "retained input or padding was modified");
}

static void run_case(ggml_backend_t backend, int tokens, bool enabled) {
    auto * ctx = ggml_init({ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false), nullptr, true});
    require(ctx != nullptr, "context allocation failed");
    auto * graph = ggml_new_graph_custom(ctx, 64, false);
    auto * raw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 136, tokens, 56);
    auto * q = ggml_view_3d(ctx, raw, 128, tokens, 56, raw->nb[1], raw->nb[2], 8 * sizeof(float));
    auto * packed = ggml_cont(ctx, q);
    auto * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, tokens, 56);
    auto * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 128, tokens, 56);
    for (auto * tensor : {raw, k, v}) { ggml_set_input(tensor); ggml_set_output(tensor); }
    auto * reference = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 128.f / std::sqrt(128.f), 0.f, 0.f);
    auto * candidate = ggml_flash_attn_ext(ctx, packed, k, v, nullptr, 128.f / std::sqrt(128.f), 0.f, 0.f);
    for (auto * output : {reference, candidate}) {
        ggml_flash_attn_ext_set_prec(output, GGML_PREC_F32);
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
    }
    require(!ggml_cuda_h3::fattn_tuning_supported(reference), "padded baseline must retain the original kernel");
    require(ggml_cuda_h3::fattn_tuning_supported(candidate), "packed H3 candidate must be eligible");
    auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    require(ggml_gallocr_alloc_graph(allocator, graph), "graph allocation failed");
    std::vector<float> first_reference, first_candidate;
    for (int run = 0; run < 3; ++run) {
        uint32_t state = run == 1 ? 67 : 17;
        std::vector<float> q_values(ggml_nelements(raw), 7.25f);
        for (int h = 0; h < 56; ++h)
            for (int t = 0; t < tokens; ++t)
                for (int c = 0; c < 128; ++c) q_values[(h * tokens + t) * 136 + c + 8] = value(state);
        std::vector<ggml_fp16_t> k_values(ggml_nelements(k)), v_values(ggml_nelements(v));
        for (size_t i = 0; i < k_values.size(); ++i) {
            k_values[i] = ggml_fp32_to_fp16(value(state) / 128.f);
            v_values[i] = ggml_fp32_to_fp16(value(state) / 128.f);
        }
        ggml_backend_tensor_set(raw, q_values.data(), 0, q_values.size() * sizeof(float));
        ggml_backend_tensor_set(k, k_values.data(), 0, k_values.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(v, v_values.data(), 0, v_values.size() * sizeof(ggml_fp16_t));
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "attention graph failed");
        ggml_backend_synchronize(backend);
        std::vector<float> expected(ggml_nelements(reference)), actual(ggml_nelements(candidate));
        ggml_backend_tensor_get(reference, expected.data(), 0, expected.size() * sizeof(float));
        ggml_backend_tensor_get(candidate, actual.data(), 0, actual.size() * sizeof(float));
        double error = 0., energy = 0.;
        float maximum = 0.f;
        for (size_t i = 0; i < actual.size(); ++i) {
            require(std::isfinite(expected[i]) && std::isfinite(actual[i]), "nonfinite attention output");
            const double delta = double(actual[i]) - expected[i];
            maximum = std::max(maximum, float(std::abs(delta)));
            error += delta * delta; energy += double(expected[i]) * expected[i];
        }
        const double relative_rms = std::sqrt(error / std::max(energy, 1e-300));
        require(std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)) == 0,
            "prefetch/default/disabled attention output differs from the original kernel");
        if (run == 0) { first_reference = expected; first_candidate = actual; }
        if (run == 2) require(first_reference == expected && first_candidate == actual, "A/B/A outputs changed");
        unchanged(raw, q_values.data()); unchanged(k, k_values.data()); unchanged(v, v_values.data());
        std::printf("PASS tokens=%d run=%d enabled=%d max_abs=%.9g relative_rms=%.9g\n",
            tokens, run, enabled, maximum, relative_rms);
    }
    ggml_gallocr_free(allocator); ggml_free(ctx);
}

int main(int argc, char ** argv) {
    try {
        const bool enabled = argc == 2 && std::strcmp(argv[1], "--expect-prefetch") == 0;
        if (!getenv("CUDA_VISIBLE_DEVICES") || std::strcmp(getenv("CUDA_VISIBLE_DEVICES"), "0") != 0) return 77;
        int count = 0; cudaDeviceProp prop;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count != 1 ||
            cudaGetDeviceProperties(&prop, 0) != cudaSuccess || prop.major != 12 || prop.minor != 0) return 77;
        std::atomic<int> launches{0}; ggml_log_set(log_tuning, &launches);
        ggml_backend_load_all();
        auto backend = ggml_backend_init_by_name("CUDA0", nullptr);
        require(backend != nullptr, "CUDA0 backend unavailable");
        for (int tokens : {1024, 1025, 3073}) run_case(backend, tokens, enabled);
        require(launches == (enabled ? 9 : 0), "unexpected number of tuned attention launches");
        ggml_backend_free(backend); ggml_log_set(nullptr, nullptr);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
