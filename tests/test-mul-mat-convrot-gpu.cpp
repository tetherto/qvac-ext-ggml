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
#elif defined(GGML_TEST_CONVROT_METAL)
#include "ggml-metal.h"
static ggml_backend_t convrot_backend_init() { return ggml_backend_metal_init(); }
static constexpr const char * kBackendName = "Metal";
#else
#error "Select a ConvRot GPU backend"
#endif

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

namespace {

constexpr int64_t kGroupSize = 256;
constexpr int64_t kTestInputFeatures = 2 * kGroupSize;
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

enum class reference_probe { none, weights, activations, range };

bool run_case(ggml_type activation_type, int64_t test_columns, bool f16_compat,
              reference_probe probe = reference_probe::none,
              int64_t test_features = kTestInputFeatures, float range_gain = 1.0f) {
    const bool benchmark = std::getenv("GGML_CONVROT_BENCH") != nullptr;
    const int64_t input_features = benchmark && std::getenv("GGML_CONVROT_BENCH_K") ? std::atoi(std::getenv("GGML_CONVROT_BENCH_K")) : benchmark ? kBenchmarkInputFeatures : test_features;
    const int64_t out_features = benchmark && std::getenv("GGML_CONVROT_BENCH_N") ? std::atoi(std::getenv("GGML_CONVROT_BENCH_N")) : benchmark ? kBenchmarkOutFeatures : 37;
    if (input_features <= 0 || input_features > 32768 || input_features % 256 || out_features <= 0 || out_features > 65536) return false;
    int64_t columns = benchmark ? kBenchmarkColumns : test_columns;
    if (probe == reference_probe::weights) columns = input_features;
    if (benchmark && std::getenv("GGML_CONVROT_BENCH_COLS")) {
        char * end = nullptr;
        columns = std::strtoll(std::getenv("GGML_CONVROT_BENCH_COLS"), &end, 10);
        if (*end || columns < 1 || columns > 32768) {
            std::fprintf(stderr, "GGML_CONVROT_BENCH_COLS must be between 1 and 32768\n");
            return false;
        }
    }
    const int iterations = benchmark ? (columns > 1 ? 3 : 10) : 2;
    ggml_context * ctx = ggml_init({ 4 * 1024 * 1024, nullptr, true });
    if (!ctx) return false;

    ggml_tensor * activations = ggml_new_tensor_2d(ctx, activation_type, input_features, columns);
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, input_features, out_features);
    ggml_tensor * scales = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_features);
    ggml_tensor * result = ggml_mul_mat_convrot(ctx, activations, weights, scales, kGroupSize);
    ggml_mul_mat_convrot_set_f16_compat(result, f16_compat);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    ggml_tensor * second_scales = nullptr;
    ggml_tensor * second_result = nullptr;
    if (!benchmark && probe == reference_probe::none) {
        second_scales = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_features);
        second_result = ggml_mul_mat_convrot(ctx, activations, weights, second_scales, kGroupSize);
        ggml_mul_mat_convrot_set_f16_compat(second_result, f16_compat);
        ggml_build_forward_expand(graph, second_result);
    }

    ggml_backend_t backend = convrot_backend_init();
    if (!backend || !ggml_backend_supports_op(backend, result)) {
        std::fprintf(stderr, "%s backend does not advertise expected ConvRot support\n", kBackendName);
        if (backend) ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> activation_f32(input_features * columns);
    std::vector<ggml_fp16_t> activation_f16(input_features * columns);
    std::vector<int8_t> weight_data(input_features * out_features);
    std::vector<float> scale_data(out_features);
    struct reference_value { size_t index; float value; double sum_abs; };
    std::vector<reference_value> expected;
    for (int64_t column = 0; column < columns; ++column) {
        for (int64_t i = 0; i < input_features; ++i) {
            const float value = probe == reference_probe::range ? (i == column ? range_gain : 0.0f) :
                probe == reference_probe::weights ? float(i == column) :
                (float) ((i * 13 + column * 29) % 97 - 48) * 0.03137f;
            activation_f32[i + column * input_features] = value;
            activation_f16[i + column * input_features] = ggml_fp32_to_fp16(value);
        }
    }
    for (int64_t row = 0; row < out_features; ++row) {
        scale_data[row] = 0.00137f * (float) (row % 17 + 1);
        for (int64_t i = 0; i < input_features; ++i) {
            weight_data[i + row * input_features] = (int8_t) ((i * 17 + row * 31) % 255 - 127);
        }
    }
    if (probe == reference_probe::activations) {
        // H256 is symmetric and self-inverse. These I8 rows reconstruct a
        // selection matrix, exposing activation conversion without cancellation.
        for (int64_t row = 0; row < out_features; ++row) {
            std::array<float, kGroupSize> block = {};
            block[row % kGroupSize] = 1;
            for (size_t stride = 1; stride < kGroupSize; stride *= 4)
                for (size_t base = 0; base < kGroupSize; base += 4 * stride)
                    for (size_t i = 0; i < stride; ++i) hadamard_4(block.data() + base + i, stride);
            scale_data[row] = 1.0f / 16;
            for (int64_t i = 0; i < input_features; ++i)
                weight_data[i + row * input_features] = i < kGroupSize ? int8_t(block[i] * 16) : 0;
        }
    }
    // Sample the large benchmark grid, including its final row/column. Unit
    // cases check every element; benchmarks must not spend minutes on a CPU GEMM.
    const int64_t column_stride = benchmark ? std::max<int64_t>(1, (columns - 1) / 8) : 1;
    const int64_t row_stride = benchmark ? std::max<int64_t>(1, (out_features - 1) / 16) : 1;
    std::vector<int64_t> sampled_columns;
    for (int64_t column = 0; column < columns; column = std::min(column + column_stride, columns - 1)) {
        sampled_columns.push_back(column);
        if (column == columns - 1) break;
    }
    if (benchmark) for (int64_t boundary = 1024; boundary < columns; boundary += 1024) {
        sampled_columns.push_back(boundary - 1);
        sampled_columns.push_back(boundary);
    }
    std::sort(sampled_columns.begin(), sampled_columns.end());
    sampled_columns.erase(std::unique(sampled_columns.begin(), sampled_columns.end()), sampled_columns.end());
    for (int64_t column : sampled_columns) {
        for (int64_t row = 0; row < out_features; row = std::min(row + row_stride, out_features - 1)) {
            double sum = 0.0;
            double sum_abs = 0.0;
            for (int64_t offset = 0; offset < input_features; offset += kGroupSize) {
                std::array<float, kGroupSize> block;
                for (int64_t i = 0; i < kGroupSize; ++i) block[i] = weight_data[offset + i + row * input_features] * scale_data[row];
                for (size_t stride = 1; stride < kGroupSize; stride *= 4)
                    for (size_t base = 0; base < kGroupSize; base += 4 * stride)
                        for (size_t i = 0; i < stride; ++i) hadamard_4(block.data() + base + i, stride);
                for (int64_t i = 0; i < kGroupSize; ++i) {
                    const int64_t index = offset + i + column * input_features;
                    const float activation = activation_type == GGML_TYPE_F32 && !f16_compat ? activation_f32[index] : ggml_fp16_to_fp32(activation_f16[index]);
                    const float weight = f16_compat ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(block[i])) : block[i];
                    const double product = double(weight) * activation;
                    sum += product;
                    sum_abs += std::fabs(product);
                }
            }
            expected.push_back({size_t(row + column * out_features),
                f16_compat ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(float(sum))) : float(sum), sum_abs});
            if (row == out_features - 1) break;
        }
        if (column == columns - 1) break;
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
    if (second_scales) {
        std::vector<float> doubled_scales = scale_data;
        for (float & value : doubled_scales) value *= 2;
        ggml_backend_tensor_set(second_scales, doubled_scales.data(), 0, ggml_nbytes(second_scales));
        // Different reconstruction data in the same graph catches scratch
        // reuse that overwrites a previous GEMM's still-pending inputs.
        const size_t first_count = expected.size();
        for (size_t i = 0; i < first_count; ++i) {
            const reference_value first = expected[i];
            expected.push_back({first.index + size_t(out_features * columns), first.value * 2, first.sum_abs * 2});
        }
    }
    bool computed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (computed) {
        const auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; computed && iteration < iterations; ++iteration) {
            computed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        }
        const double milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::printf("ConvRot %s %s %s: %.3f ms/run (%d runs, K=%lld N=%lld columns=%lld probe=%d)\n",
                    kBackendName, ggml_type_name(activation_type), f16_compat ? "f16-compat" : "native",
                    milliseconds / iterations, iterations,
                    (long long) input_features, (long long) out_features, (long long) columns, int(probe));
    }
    std::vector<float> actual(out_features * columns * (second_result ? 2 : 1));
    if (computed) ggml_backend_tensor_get(result, actual.data(), 0, ggml_nbytes(result));
    if (computed && second_result) ggml_backend_tensor_get(second_result, actual.data() + out_features * columns, 0, ggml_nbytes(second_result));

    bool passed = computed;
    const float relative_tolerance = f16_compat ? 2e-3f : (benchmark ? 1e-4f : 3e-5f);
    for (float value : actual) {
        if (!std::isfinite(value)) {
            std::fprintf(stderr, "%s ConvRot produced a non-finite result\n", kBackendName);
            passed = false;
            break;
        }
    }
    float max_error = 0;
    int mismatches = 0;
    for (const auto & reference : expected) {
        const size_t i = reference.index;
        const float error = std::fabs(actual[i] - reference.value) / (1.0f + std::fabs(reference.value));
        max_error = std::max(max_error, error);
        // F16 compatibility allows the backend's F16 reduction order. Use a
        // forward-error bound scaled by sum(abs(W*X)), not the cancellation-
        // sensitive output magnitude. Native F32 keeps its tighter tolerance.
        const double limit = probe != reference_probe::none ? 0.0 :
                             f16_compat ? relative_tolerance * (1.0 + reference.sum_abs) :
                                         relative_tolerance * (1.0 + std::fabs(reference.value));
        const bool output_is_half = !f16_compat || actual[i] == ggml_fp16_to_fp32(ggml_fp32_to_fp16(actual[i]));
        if (std::fabs(actual[i] - reference.value) > limit || !output_is_half) {
            if (mismatches++ < 8) std::fprintf(stderr, "%s ConvRot mismatch at %zu: got %.8f, expected %.8f\n", kBackendName, i, actual[i], reference.value);
            passed = false;
        }
    }
    std::printf("  checked %zu outputs, max normalized error %.8g, mismatches %d\n", expected.size(), max_error, mismatches);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return passed;
}

} // namespace

int main() {
    bool passed = true;
#if defined(GGML_TEST_CONVROT_VULKAN)
    // Exercise each selected hybrid reduction and its overflow/underflow
    // guard. A single nonzero activation gives an exact reconstruction
    // reference, so these cases require zero error, not a relaxed bound.
    if (!std::getenv("GGML_CONVROT_BENCH")) {
        for (int64_t k : {5376, 7168, 14336}) for (float gain : {1e10f, 1e-20f})
            passed = run_case(GGML_TYPE_F32, 2, false, reference_probe::range, k, gain) && passed;
    }
#endif
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (bool compat : {false, true}) {
            if (compat && std::getenv("GGML_CONVROT_BENCH_NATIVE_ONLY")) continue;
#if defined(GGML_TEST_CONVROT_CUDA)
            // CUDA currently implements the optional compatibility hint only
            // for F32 activations; F16 activations retain native arithmetic.
            if (compat && type == GGML_TYPE_F16) continue;
#endif
            if (std::getenv("GGML_CONVROT_BENCH")) {
                passed = run_case(type, kBenchmarkColumns, compat) && passed;
            } else {
                for (int64_t columns : {int64_t(1), int64_t(2), int64_t(33)}) {
                    passed = run_case(type, columns, compat) && passed;
                }
                if (compat) {
                    passed = run_case(type, 1, true, reference_probe::weights) && passed;
                    passed = run_case(type, 33, true, reference_probe::activations) && passed;
                }
            }
        }
    }
    if (passed) std::printf("ConvRot %s test passed\n", kBackendName);
    return passed ? 0 : 1;
}
