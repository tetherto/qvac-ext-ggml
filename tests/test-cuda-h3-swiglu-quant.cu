#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda/quantize.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

static float fixture(int kind, size_t i, int seed) {
    uint32_t x = uint32_t(i) * 1664525u + 1013904223u + uint32_t(seed) * 12345u;
    x ^= x >> 13; x *= 2246822519u; x ^= x >> 16;
    if (kind == 0) return (int(x >> 8) - 8388608) / 1048576.f;
    const float values[] = {0.f, -0.f, std::numeric_limits<float>::denorm_min(),
        -std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::min(),
        -std::numeric_limits<float>::min(), 0x1p-20f, -0x1p-20f, 20.f, -20.f, 100.f, -100.f,
        65504.f, -65504.f, 0x1p30f, -0x1p30f};
    return values[(x >> 16) % (sizeof(values) / sizeof(values[0]))];
}

static void run_case(ggml_backend_t backend, int64_t width, int64_t rows, int kind, bool capture, bool upstream) {
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    require(ctx != nullptr, "context allocation failed");
    auto * packed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * width, rows);
    ggml_set_input(packed); ggml_set_output(packed);
    auto * scaled_packed = upstream ? ggml_scale(ctx, packed, 128.f) : packed;
    auto * activation = ggml_swiglu(ctx, scaled_packed);
    auto * scaled = ggml_scale(ctx, activation, 1.f / 128.f);
    ggml_set_output(scaled);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, scaled);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "tensor allocation failed");
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const int64_t padded = GGML_PAD(width, MATRIX_ROW_PADDING);
    constexpr size_t guard = 256;
    const size_t payload = padded / QK8_1_MMQ * rows * sizeof(block_q8_1_mmq);
    const size_t size = payload + 2 * guard;
    unsigned char * expected_d, * actual_d;
    check(cudaMalloc(reinterpret_cast<void **>(&expected_d), size));
    check(cudaMalloc(reinterpret_cast<void **>(&actual_d), size));
    auto launch = [&] {
        quantize_mmq_q8_1_swiglu_cuda(static_cast<float *>(packed->data), actual_d + guard,
            width, 2 * width, 2 * width * rows, 2 * width * rows, padded, rows, 1, 1, 1.f / 128.f, 0.f, stream, upstream);
    };
    cudaGraph_t captured = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (capture) {
        ggml_backend_synchronize(backend);
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        launch();
        check(cudaStreamEndCapture(stream, &captured));
        check(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
    }
    std::vector<unsigned char> first;
    for (int run = 0; run < 3; ++run) {
        const int seed = run == 1 ? 41 : 17;
        std::vector<float> input(ggml_nelements(packed));
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = fixture(kind, i, seed);
            if (kind >= 2) {
                const bool gate = i % (2 * width) >= (size_t) width;
                const float tiny = std::numeric_limits<float>::denorm_min();
                const float negative[] = {-tiny, -2.f * tiny, -32.f * tiny, -std::numeric_limits<float>::min()};
                const float x = kind == 2 ? negative[(i / 32 + seed) % 4] : (i / 32 + seed) % 2 ? -0.f : 0.f;
                input[i] = gate ? (upstream ? 1.f / 128.f : 1.f) : x;
            }
        }
        ggml_backend_tensor_set(packed, input.data(), 0, input.size() * sizeof(float));
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "SwiGLU + SCALE failed");
        ggml_backend_synchronize(backend);
        check(cudaMemsetAsync(expected_d, 0xa5, size, stream));
        check(cudaMemsetAsync(actual_d, 0xa5, size, stream));
        quantize_mmq_q8_1_cuda(static_cast<float *>(scaled->data), nullptr, expected_d + guard, GGML_TYPE_Q4_K,
            width, width, width * rows, width * rows, padded, rows, 1, 1, stream);
        if (capture) check(cudaGraphLaunch(executable, stream)); else launch();
        check(cudaGetLastError()); check(cudaStreamSynchronize(stream));
        std::vector<unsigned char> expected(size), actual(size);
        check(cudaMemcpy(expected.data(), expected_d, size, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(actual.data(), actual_d, size, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < guard; ++i) {
            require(expected[i] == 0xa5 && actual[i] == 0xa5 &&
                expected[guard + payload + i] == 0xa5 && actual[guard + payload + i] == 0xa5, "guard overwritten");
        }
        if (expected != actual) {
            size_t offset = std::mismatch(expected.begin(), expected.end(), actual.begin()).first - expected.begin();
            throw std::runtime_error("SwiGLU DS4 differs at byte " + std::to_string(offset) +
                " width=" + std::to_string(width) + " fixture=" + std::to_string(kind));
        }
        if (run == 0) first = actual;
        if (run == 2) require(first == actual, "A/B/A payload changed");
        std::vector<float> after(input.size());
        ggml_backend_tensor_get(packed, after.data(), 0, after.size() * sizeof(float));
        require(std::memcmp(input.data(), after.data(), input.size() * sizeof(float)) == 0, "input modified");
    }
    std::printf("SWIGLU_DS4_EXACT width=%lld rows=%lld fixture=%d capture=%d upstream=%d payload=%zu A/B/A guards readonly\n",
        (long long) width, (long long) rows, kind, capture, upstream, payload);
    if (executable) check(cudaGraphExecDestroy(executable));
    if (captured) check(cudaGraphDestroy(captured));
    check(cudaFree(expected_d)); check(cudaFree(actual_d)); check(cudaStreamDestroy(stream));
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
}

static std::vector<float> read_f32(ggml_tensor * tensor) {
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}

static void exact(const std::vector<float> & a, const std::vector<float> & b, const char * message) {
    require(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0, message);
}

static void run_projection(ggml_backend_t backend, const std::string & guard, bool upstream) {
    constexpr int64_t width = 14336, rows = 512, out_width = 5376;
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    auto * packed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * width, rows);
    auto * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, width, out_width);
    ggml_set_input(packed); ggml_set_output(packed);
    auto * scaled_packed = upstream ? ggml_scale(ctx, packed, guard == "other-upstream-scale" ? 64.f : 128.f) : packed;
    if (guard == "observed-upstream") ggml_set_output(scaled_packed);
    auto * activation = ggml_swiglu(ctx, scaled_packed);
    if (guard == "swapped") {
        const int32_t swapped = 1;
        std::memcpy(reinterpret_cast<char *>(activation->op_params) + sizeof(int32_t), &swapped, sizeof(swapped));
    }
    auto * scale = ggml_scale(ctx, activation, 1.f / 128.f);
    auto * mm = ggml_mul_mat(ctx, weight, scale);
    ggml_mul_mat_set_prec(mm, GGML_PREC_F32);
    auto * output = ggml_scale(ctx, mm, 128.f);
    auto * observed = guard == "multi-consumer" ? ggml_scale(ctx, activation, 2.f) : nullptr;
    ggml_set_output(output);
    if (observed) ggml_set_output(observed);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    if (observed) ggml_build_forward_expand(graph, observed);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "projection allocation failed");
    std::vector<float> weight_row(width);
    for (size_t i = 0; i < weight_row.size(); ++i) weight_row[i] = fixture(0, i, 3) / 128.f;
    std::vector<unsigned char> encoded(ggml_row_size(GGML_TYPE_Q4_K, width));
    require(ggml_quantize_chunk(GGML_TYPE_Q4_K, weight_row.data(), encoded.data(), 0, 1, width, nullptr) == encoded.size(),
        "weight fixture quantization failed");
    for (int row = 0; row < out_width; ++row) {
        ggml_backend_tensor_set(weight, encoded.data(), row * encoded.size(), encoded.size());
    }
    std::vector<float> input(ggml_nelements(packed));
    std::vector<uint32_t> poison(ggml_nelements(activation), 0x7fc12345u);
    std::vector<float> first;
    for (int run = 0; run < 3; ++run) {
        for (size_t i = 0; i < input.size(); ++i) input[i] = fixture(0, i, run == 1 ? 41 : 17) / (upstream ? 128.f : 1.f);
        ggml_backend_tensor_set(packed, input.data(), 0, input.size() * sizeof(float));
        ggml_set_output(activation); ggml_set_output(scale);
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "reference projection failed");
        auto expected = read_f32(output);
        auto expected_activation = read_f32(activation);
        auto expected_packed = read_f32(scaled_packed);
        auto expected_observed = observed ? read_f32(observed) : std::vector<float>();
        scale->flags &= ~GGML_TENSOR_FLAG_OUTPUT;
        if (guard != "observed") activation->flags &= ~GGML_TENSOR_FLAG_OUTPUT;
        ggml_backend_tensor_set(activation, poison.data(), 0, poison.size() * sizeof(uint32_t));
        if (upstream) {
            std::vector<uint32_t> packed_poison(ggml_nelements(scaled_packed), 0x7fc12345u);
            ggml_backend_tensor_set(scaled_packed, packed_poison.data(), 0, packed_poison.size() * sizeof(uint32_t));
        }
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "candidate projection failed");
        auto actual = read_f32(output);
        exact(expected, actual, "full Q4 projection differs");
        const bool enabled = std::getenv("GGML_CUDA_H3_SWIGLU_QUANT_FUSION") &&
            std::strcmp(std::getenv("GGML_CUDA_H3_SWIGLU_QUANT_FUSION"), "1") == 0 &&
            !(std::getenv("GGML_CUDA_DISABLE_FUSION") && std::atoi(std::getenv("GGML_CUDA_DISABLE_FUSION")));
        if ((guard == "eligible" || guard == "observed-upstream" || guard == "other-upstream-scale") && enabled) {
            const auto skipped = read_f32(activation);
            require(std::memcmp(skipped.data(), poison.data(), poison.size() * sizeof(uint32_t)) == 0,
                "eligible activation was materialized");
        } else {
            exact(expected_activation, read_f32(activation), "fallback activation not preserved");
        }
        if (upstream) {
            if (guard == "eligible" && enabled) {
                auto skipped = read_f32(scaled_packed);
                for (float value : skipped) {
                    uint32_t bits;
                    std::memcpy(&bits, &value, sizeof(bits));
                    require(bits == 0x7fc12345u, "eligible upstream scale was materialized");
                }
            } else {
                exact(expected_packed, read_f32(scaled_packed), "fallback upstream scale changed");
            }
        }
        if (observed) exact(expected_observed, read_f32(observed), "observed consumer changed");
        exact(input, read_f32(packed), "projection changed input");
        if (run == 0) first = actual;
        if (run == 2) exact(first, actual, "projection A/B/A changed");
    }
    std::printf("SWIGLU_PROJECTION_EXACT guard=%s upstream=%d A/B/A readonly\n", guard.c_str(), upstream);
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
}

int main() {
#ifdef GGML_TEST_H3_SWIGLU_FORCE_CUBLAS
    return 77;
#endif
    try {
        ggml_backend_load_all();
        auto backend = ggml_backend_init_by_name("CUDA0", nullptr);
        if (!backend) return 77;
        cudaDeviceProp properties;
        check(cudaGetDeviceProperties(&properties, 0));
        if (properties.major != 12 || properties.minor != 0) {
            ggml_backend_free(backend);
            return 77;
        }
        for (auto width : {128, 132, 516, 14336}) {
            for (int kind = 0; kind < 4; ++kind) {
                for (bool upstream : {false, true}) {
                    run_case(backend, width, width == 14336 ? 512 : 17, kind, false, upstream);
                    run_case(backend, width, 3, kind, true, upstream);
                }
            }
        }
        for (const auto & guard : {"eligible", "observed", "multi-consumer", "swapped"}) {
            for (bool upstream : {false, true}) run_projection(backend, guard, upstream);
        }
        run_projection(backend, "observed-upstream", true);
        run_projection(backend, "other-upstream-scale", true);
        ggml_backend_free(backend);
        std::puts("H3_SWIGLU_QUANT_PASS");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
