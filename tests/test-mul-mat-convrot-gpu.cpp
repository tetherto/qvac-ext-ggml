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
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int64_t kGroupSize = 256;
constexpr int64_t kInputFeatures = 2 * kGroupSize;
constexpr int64_t kOutFeatures = 3;
constexpr int64_t kColumns = 2;

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
    ggml_context * ctx = ggml_init({ 4 * 1024 * 1024, nullptr, true });
    if (!ctx) return false;

    ggml_tensor * activations = ggml_new_tensor_2d(ctx, activation_type, kInputFeatures, kColumns);
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, kInputFeatures, kOutFeatures);
    ggml_tensor * scales = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kOutFeatures);
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

    std::vector<float> activation_f32(kInputFeatures * kColumns);
    std::vector<ggml_fp16_t> activation_f16(kInputFeatures * kColumns);
    std::vector<int8_t> weight_data(kInputFeatures * kOutFeatures);
    std::vector<float> scale_data(kOutFeatures);
    std::vector<float> expected(kOutFeatures * kColumns);
    for (int64_t column = 0; column < kColumns; ++column) {
        for (int64_t i = 0; i < kInputFeatures; ++i) {
            const float value = (float) ((i * 13 + column * 29) % 97 - 48) * 0.03125f;
            activation_f32[i + column * kInputFeatures] = value;
            activation_f16[i + column * kInputFeatures] = ggml_fp32_to_fp16(value);
        }
    }
    for (int64_t row = 0; row < kOutFeatures; ++row) {
        scale_data[row] = 0.00390625f * (float) (row + 1);
        for (int64_t i = 0; i < kInputFeatures; ++i) {
            weight_data[i + row * kInputFeatures] = (int8_t) ((i * 17 + row * 31) % 255 - 127);
        }
    }
    for (int64_t column = 0; column < kColumns; ++column) {
        for (int64_t row = 0; row < kOutFeatures; ++row) {
            float sum = 0.0f;
            for (int64_t offset = 0; offset < kInputFeatures; offset += kGroupSize) {
                std::array<float, kGroupSize> block;
                for (int64_t i = 0; i < kGroupSize; ++i) block[i] = weight_data[offset + i + row * kInputFeatures] * scale_data[row];
                for (size_t stride = 1; stride < kGroupSize; stride *= 4)
                    for (size_t base = 0; base < kGroupSize; base += 4 * stride)
                        for (size_t i = 0; i < stride; ++i) hadamard_4(block.data() + base + i, stride);
                for (int64_t i = 0; i < kGroupSize; ++i) {
                    const int64_t index = offset + i + column * kInputFeatures;
                    const float activation = activation_type == GGML_TYPE_F32 ? activation_f32[index] : ggml_fp16_to_fp32(activation_f16[index]);
                    sum += block[i] * activation;
                }
            }
            expected[row + column * kOutFeatures] = sum;
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
    const bool computed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    std::vector<float> actual(expected.size());
    if (computed) ggml_backend_tensor_get(result, actual.data(), 0, ggml_nbytes(result));

    bool passed = computed;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > 3e-5f * (1.0f + std::fabs(expected[i]))) {
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
