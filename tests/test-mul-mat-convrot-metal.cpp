#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <vector>

namespace {

constexpr int64_t kGroupSize = 256;

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

bool run_case(ggml_type activation_type, int64_t kColumns, bool compat, bool benchmark = false, float magnitude = 1.0f, bool pair = false) {
    const int64_t kInputFeatures = benchmark ? 5376 : 512;
    const int64_t kOutFeatures = benchmark ? 14336 : 37;
    ggml_init_params params = { 4 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * activations = ggml_new_tensor_2d(ctx, activation_type, kInputFeatures, kColumns);
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, kInputFeatures, kOutFeatures);
    ggml_tensor * scales = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kOutFeatures);
    ggml_tensor * result = ggml_mul_mat_convrot(ctx, activations, weights, scales, kGroupSize);
    ggml_mul_mat_convrot_set_f16_compat(result, compat);
    if (pair) {
        ggml_tensor * other = ggml_mul_mat_convrot(ctx, activations, weights, scales, kGroupSize);
        ggml_mul_mat_convrot_set_f16_compat(other, compat);
        result = ggml_add(ctx, result, other);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    ggml_backend_t backend = ggml_backend_metal_init();
    if (!backend || !ggml_backend_supports_op(backend, result)) {
        std::fprintf(stderr, "Metal backend does not advertise expected ConvRot support\n");
        if (backend) ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> activation_f32(kInputFeatures * kColumns);
    std::vector<ggml_fp16_t> activation_f16(kInputFeatures * kColumns);
    std::vector<int8_t> weight_data(kInputFeatures * kOutFeatures);
    std::vector<float> scale_data(kOutFeatures);
    std::vector<float> expected(kOutFeatures * kColumns);
    for (int64_t column = 0; column < kColumns; ++column) {
        for (int64_t i = 0; i < kInputFeatures; ++i) {
            const float value = (float) ((i * 13 + column * 29) % 97 - 48) *
                (magnitude == 1.0f ? 0.03127f : 0.03125f) * magnitude;
            activation_f32[i + column * kInputFeatures] = value;
            activation_f16[i + column * kInputFeatures] = ggml_fp32_to_fp16(value);
        }
    }
    for (int64_t row = 0; row < kOutFeatures; ++row) {
        scale_data[row] = (magnitude == 1.0f ? 0.0039183f : 0.00390625f) * (float) (row % 37 + 1);
        for (int64_t i = 0; i < kInputFeatures; ++i) {
            weight_data[i + row * kInputFeatures] = (int8_t) ((i * 17 + row * 31) % 255 - 127);
        }
    }
    for (int64_t column = 0; column < kColumns; ++column) {
        if (benchmark && column != 0 && column != kColumns - 1) continue;
        for (int64_t row = 0; row < kOutFeatures; ++row) {
            if (benchmark && row != 0 && row != kOutFeatures - 1) continue;
            double sum = 0.0;
            for (int64_t offset = 0; offset < kInputFeatures; offset += kGroupSize) {
                std::array<float, kGroupSize> block;
                for (int64_t i = 0; i < kGroupSize; ++i) block[i] = weight_data[offset + i + row*kInputFeatures] * scale_data[row];
                for (size_t stride = 1; stride < kGroupSize; stride *= 4)
                    for (size_t base = 0; base < kGroupSize; base += 4*stride)
                        for (size_t i = 0; i < stride; ++i) hadamard_4(block.data() + base + i, stride);
                for (int64_t i = 0; i < kGroupSize; ++i) {
                    const int64_t index = offset + i + column*kInputFeatures;
                    const float activation = activation_type == GGML_TYPE_F32 && !compat ? activation_f32[index] : ggml_fp16_to_fp32(activation_f16[index]);
                    const float weight = compat ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(block[i])) : block[i];
                    sum += double(weight) * double(activation);
                }
            }
            expected[row + column*kOutFeatures] = compat ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(sum)) : sum;
            if (pair) expected[row + column*kOutFeatures] *= 2.0f;
        }
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        std::fprintf(stderr, "failed to allocate Metal test tensors\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(activations, activation_type == GGML_TYPE_F32 ? static_cast<const void *>(activation_f32.data()) : static_cast<const void *>(activation_f16.data()), 0, ggml_nbytes(activations));
    ggml_backend_tensor_set(weights, weight_data.data(), 0, ggml_nbytes(weights));
    ggml_backend_tensor_set(scales, scale_data.data(), 0, ggml_nbytes(scales));
    const bool computed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    std::vector<float> actual(expected.size());
    if (computed) ggml_backend_tensor_get(result, actual.data(), 0, ggml_nbytes(result));

    bool passed = computed;
    if (benchmark && passed) {
        constexpr int runs = 3;
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < runs; ++i) passed = passed && ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        ggml_backend_synchronize(backend);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count()/runs;
        std::printf("ConvRot Metal %s compat=%d K=%lld N=%lld columns=%lld %.3f ms/run\n", ggml_type_name(activation_type), compat, (long long)kInputFeatures, (long long)kOutFeatures, (long long)kColumns, ms);
    }
    for (size_t i = 0; passed && i < actual.size(); ++i) {
        const bool checked = !benchmark || ((i % kOutFeatures == 0 || i % kOutFeatures == size_t(kOutFeatures - 1)) &&
            (i / kOutFeatures == 0 || i / kOutFeatures == size_t(kColumns - 1)));
        // Scale the absolute term with the input in the explicit range tests;
        // otherwise multiplying the entire problem by 1e6 changes its allowed
        // relative numerical error solely because the output nearly cancels.
        if (!std::isfinite(actual[i]) || (checked && std::fabs(actual[i] - expected[i]) > (compat ? 1e-3f : 3e-5f) * (std::fabs(magnitude) + std::fabs(expected[i])))) {
            std::fprintf(stderr, "Metal ConvRot mismatch at %zu: got %.8f, expected %.8f\n", i, actual[i], expected[i]);
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
    bool passed = true;
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16})
        for (int columns : {1, 2, 33})
            for (bool compat : {false, true}) passed = run_case(type, columns, compat) && passed;
    // Exactly representable patterns isolate range handling from cancellation
    // differences between rotating the activation and reconstructing weights.
    passed = run_case(GGML_TYPE_F32, 33, false, false, std::ldexp(1.0f, -30)) && passed;
    passed = run_case(GGML_TYPE_F32, 33, false, false, std::ldexp(1.0f, 20)) && passed;
    passed = run_case(GGML_TYPE_F32, 33, false, false, 1.0f, true) && passed;
    const char * columns = std::getenv("GGML_CONVROT_BENCH_COLUMNS");
    if (columns) passed = run_case(GGML_TYPE_F32, std::max(1, std::atoi(columns)), false, true) && passed;
    if (passed) std::puts("ConvRot Metal test passed");
    return passed ? 0 : 1;
}
