#include "ggml.h"
#include "ggml-backend.h"

#if defined(GGML_TEST_CONVROT_CUDA)
#include "ggml-cuda.h"
static ggml_backend_t convrot_backend_init() { return ggml_backend_cuda_init(0); }
static constexpr const char * kBackendName = "CUDA";
#elif defined(GGML_TEST_CONVROT_VULKAN)
#include "ggml-vulkan.h"
static ggml_backend_t convrot_backend_init() { return ggml_backend_vk_init(0); }
static constexpr const char * kBackendName = "Vulkan";
#else
#error "Select a ConvRot GPU backend"
#endif

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

namespace {

constexpr int64_t kGroupSize = 256;
constexpr int64_t kTestInputFeatures = 2 * kGroupSize;
constexpr int64_t kTestOutFeatures = 3;
constexpr int64_t kTestColumns = 2;
constexpr int64_t kBenchmarkInputFeatures = 21 * kGroupSize;
constexpr int64_t kBenchmarkOutFeatures = 14336;
constexpr int64_t kBenchmarkColumns = 1;

void hadamard_4(float * values, size_t stride) {
    const float a = values[0 * stride];
    const float b = values[1 * stride];
    const float c = values[2 * stride];
    const float d = values[3 * stride];
    values[0 * stride] = ( a + b + c - d) * 0.5f;
    values[1 * stride] = ( a + b - c + d) * 0.5f;
    values[2 * stride] = ( a - b + c + d) * 0.5f;
    values[3 * stride] = (-a + b + c + d) * 0.5f;
}

bool run_case(ggml_type activation_type) {
    const bool benchmark = std::getenv("GGML_CONVROT_BENCH") != nullptr;
    const int64_t input_features = benchmark ? kBenchmarkInputFeatures : kTestInputFeatures;
    const int64_t out_features = benchmark ? kBenchmarkOutFeatures : kTestOutFeatures;
    const int64_t columns = benchmark ? kBenchmarkColumns : kTestColumns;
    const int iterations = benchmark ? 10 : 1;
    ggml_context * ctx = ggml_init({ 4 * 1024 * 1024, nullptr, true });
    if (!ctx) return false;

    ggml_tensor * activations = ggml_new_tensor_2d(ctx, activation_type, input_features, columns);
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, input_features, out_features);
    ggml_tensor * scales = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_features);
    ggml_tensor * result = ggml_mul_mat_convrot(ctx, activations, weights, scales, kGroupSize);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    ggml_backend_t backend = convrot_backend_init();
    if (!backend || !ggml_backend_supports_convrot(backend, activation_type, kGroupSize) ||
        ggml_backend_supports_convrot(backend, activation_type, kGroupSize / 2) ||
        !ggml_backend_supports_op(backend, result)) {
        std::fprintf(stderr, "%s backend does not advertise expected ConvRot support\n", kBackendName);
        if (backend) ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> activation_f32(input_features * columns);
    std::vector<ggml_fp16_t> activation_f16(input_features * columns);
    std::vector<int8_t> weight_data(input_features * out_features);
    std::vector<float> scale_data(out_features);
    std::vector<float> expected(out_features * columns);
    for (int64_t column = 0; column < columns; ++column) {
        for (int64_t i = 0; i < input_features; ++i) {
            const float value = (float) ((i * 13 + column * 29) % 97 - 48) * 0.03125f;
            activation_f32[i + column * input_features] = value;
            activation_f16[i + column * input_features] = ggml_fp32_to_fp16(value);
        }
    }
    for (int64_t row = 0; row < out_features; ++row) {
        scale_data[row] = 0.00390625f * (float) (row + 1);
        for (int64_t i = 0; i < input_features; ++i) {
            weight_data[i + row * input_features] = (int8_t) ((i * 17 + row * 31) % 255 - 127);
        }
    }
    for (int64_t column = 0; column < columns; ++column) {
        for (int64_t row = 0; row < out_features; ++row) {
            float sum = 0.0f;
            for (int64_t offset = 0; offset < input_features; offset += kGroupSize) {
                std::array<float, kGroupSize> block;
                for (int64_t i = 0; i < kGroupSize; ++i) block[i] = weight_data[offset + i + row * input_features] * scale_data[row];
                for (size_t stride = 1; stride < kGroupSize; stride *= 4)
                    for (size_t base = 0; base < kGroupSize; base += 4 * stride)
                        for (size_t i = 0; i < stride; ++i) hadamard_4(block.data() + base + i, stride);
                for (int64_t i = 0; i < kGroupSize; ++i) {
                    const int64_t index = offset + i + column * input_features;
                    const float activation = activation_type == GGML_TYPE_F32 ? activation_f32[index] : ggml_fp16_to_fp32(activation_f16[index]);
                    sum += block[i] * activation;
                }
            }
            expected[row + column * out_features] = sum;
        }
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        std::fprintf(stderr, "failed to allocate %s test tensors\n", kBackendName);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(activations, activation_type == GGML_TYPE_F32 ? static_cast<const void *>(activation_f32.data()) : static_cast<const void *>(activation_f16.data()), 0, ggml_nbytes(activations));
    ggml_backend_tensor_set(weights, weight_data.data(), 0, ggml_nbytes(weights));
    ggml_backend_tensor_set(scales, scale_data.data(), 0, ggml_nbytes(scales));
    bool computed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (benchmark && computed) {
        const auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; computed && iteration < iterations; ++iteration) {
            computed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        }
        const double milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::printf("ConvRot %s %s benchmark: %.3f ms/run (%d runs, %lldx%lld)\n",
                    kBackendName, ggml_type_name(activation_type), milliseconds / iterations, iterations,
                    (long long) input_features, (long long) out_features);
    }
    std::vector<float> actual(expected.size());
    if (computed) ggml_backend_tensor_get(result, actual.data(), 0, ggml_nbytes(result));

    bool passed = computed;
    const float relative_tolerance = benchmark ? 1e-4f : 3e-5f;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > relative_tolerance * (1.0f + std::fabs(expected[i]))) {
            std::fprintf(stderr, "%s ConvRot mismatch at %zu: got %.8f, expected %.8f\n", kBackendName, i, actual[i], expected[i]);
            passed = false;
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return passed;
}

} // namespace

int main() {
    const bool passed = run_case(GGML_TYPE_F32) && run_case(GGML_TYPE_F16);
    if (passed) std::printf("ConvRot %s test passed\n", kBackendName);
    return passed ? 0 : 1;
}
