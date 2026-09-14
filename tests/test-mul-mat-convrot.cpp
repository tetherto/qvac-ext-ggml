#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int64_t kGroupSize = 256;
constexpr int64_t kInputFeatures = 2 * kGroupSize;
constexpr int64_t kOutFeatures = 3;

void apply_regular_hadamard_4(float * values, size_t stride) {
    const float a = values[0 * stride];
    const float b = values[1 * stride];
    const float c = values[2 * stride];
    const float d = values[3 * stride];

    values[0 * stride] = ( a + b + c - d) * 0.5f;
    values[1 * stride] = ( a + b - c + d) * 0.5f;
    values[2 * stride] = ( a - b + c + d) * 0.5f;
    values[3 * stride] = (-a + b + c + d) * 0.5f;
}

float read_activation(const ggml_tensor * tensor, const void * data, int64_t i0, int64_t i1, int64_t i2) {
    const char * ptr = (const char *) data + i0 * tensor->nb[0] + i1 * tensor->nb[1] + i2 * tensor->nb[2];
    if (tensor->type == GGML_TYPE_F32) {
        float value;
        std::memcpy(&value, ptr, sizeof(value));
        return value;
    }

    ggml_fp16_t value;
    std::memcpy(&value, ptr, sizeof(value));
    return ggml_fp16_to_fp32(value);
}

float compatibility_reference(const ggml_tensor * activations, const void * activation_data,
                              const ggml_tensor * weights, const void * weight_data,
                              const ggml_tensor * scales, const void * scale_data,
                              int64_t row, int64_t i1, int64_t i2, bool round_weights_to_f16) {
    float scale;
    std::memcpy(&scale, (const char *) scale_data + row * scales->nb[0], sizeof(scale));

    std::array<float, kGroupSize> values;
    float sum = 0.0f;
    for (int64_t offset = 0; offset < weights->ne[0]; offset += kGroupSize) {
        for (int64_t i = 0; i < kGroupSize; ++i) {
            int8_t quantized;
            std::memcpy(&quantized, (const char *) weight_data + row * weights->nb[1] + (offset + i) * weights->nb[0], sizeof(quantized));
            values[i] = (float) quantized * scale;
        }
        for (size_t stride = 1; stride < kGroupSize; stride *= 4) {
            for (size_t base = 0; base < kGroupSize; base += stride * 4) {
                for (size_t i = 0; i < stride; ++i) {
                    apply_regular_hadamard_4(values.data() + base + i, stride);
                }
            }
        }
        for (int64_t i = 0; i < kGroupSize; ++i) {
            const float weight = round_weights_to_f16
                ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(values[i]))
                : values[i];
            sum += weight * read_activation(activations, activation_data, offset + i, i1, i2);
        }
    }
    return sum;
}

ggml_backend_t init_cpu_backend() {
    ggml_backend_load_all();
    return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
}

bool close(float actual, float expected, float tolerance) {
    return std::fabs(actual - expected) <= tolerance * (1.0f + std::fabs(expected));
}

bool run_case(ggml_type activation_type, bool non_contiguous) {
    ggml_init_params params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "failed to create ggml context\n");
        return false;
    }

    ggml_tensor * activation_storage = ggml_new_tensor_3d(ctx, activation_type, kInputFeatures, non_contiguous ? 4 : 2, 2);
    ggml_tensor * activations = activation_storage;
    if (non_contiguous) {
        activations = ggml_view_3d(ctx, activation_storage, kInputFeatures, 2, 2,
            2 * activation_storage->nb[1], activation_storage->nb[2], 0);
    }
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, kInputFeatures, kOutFeatures);
    ggml_tensor * scales = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kOutFeatures, 1);
    std::vector<uint8_t> activation_data(ggml_nbytes(activation_storage));
    std::vector<uint8_t> weight_data(ggml_nbytes(weights));
    std::vector<uint8_t> scale_data(ggml_nbytes(scales));

    for (int64_t i2 = 0; i2 < activation_storage->ne[2]; ++i2) {
        for (int64_t i1 = 0; i1 < activation_storage->ne[1]; ++i1) {
            for (int64_t i0 = 0; i0 < activation_storage->ne[0]; ++i0) {
                const float value = ((float) ((i0 * 13 + i1 * 29 + i2 * 47) % 97) - 48.0f) * 0.03125f;
                char * ptr = (char *) activation_data.data() + i0 * activation_storage->nb[0] + i1 * activation_storage->nb[1] + i2 * activation_storage->nb[2];
                if (activation_type == GGML_TYPE_F32) {
                    std::memcpy(ptr, &value, sizeof(value));
                } else {
                    const ggml_fp16_t value_f16 = ggml_fp32_to_fp16(value);
                    std::memcpy(ptr, &value_f16, sizeof(value_f16));
                }
            }
        }
    }
    for (int64_t row = 0; row < kOutFeatures; ++row) {
        const float scale = 0.00390625f * (float) (row + 1);
        std::memcpy((char *) scale_data.data() + row * scales->nb[0], &scale, sizeof(scale));
        for (int64_t column = 0; column < kInputFeatures; ++column) {
            const int8_t value = (int8_t) ((column * 17 + row * 31) % 255 - 127);
            std::memcpy((char *) weight_data.data() + row * weights->nb[1] + column * weights->nb[0], &value, sizeof(value));
        }
    }

    ggml_tensor * result = ggml_mul_mat_convrot(ctx, activations, weights, scales, kGroupSize);
    ggml_backend_t backend = init_cpu_backend();
    if (!backend || !ggml_backend_supports_op(backend, result)) {
        std::fprintf(stderr, "CPU backend unexpectedly does not support ConvRot\n");
        if (backend) ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        std::fprintf(stderr, "failed to allocate CPU test tensors\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(activation_storage, activation_data.data(), 0, activation_data.size());
    ggml_backend_tensor_set(weights, weight_data.data(), 0, weight_data.size());
    ggml_backend_tensor_set(scales, scale_data.data(), 0, scale_data.size());
    const bool computed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    std::vector<uint8_t> result_data(ggml_nbytes(result));
    if (computed) {
        ggml_backend_tensor_get(result, result_data.data(), 0, result_data.size());
    }
    bool passed = computed;
    for (int64_t i2 = 0; passed && i2 < result->ne[2]; ++i2) {
        for (int64_t i1 = 0; passed && i1 < result->ne[1]; ++i1) {
            for (int64_t row = 0; passed && row < result->ne[0]; ++row) {
                float actual;
                std::memcpy(&actual, result_data.data() + row * result->nb[0] + i1 * result->nb[1] + i2 * result->nb[2], sizeof(actual));
                const float reference_f32 = compatibility_reference(activations, activation_data.data(), weights, weight_data.data(), scales, scale_data.data(), row, i1, i2, false);
                const float reference_f16 = compatibility_reference(activations, activation_data.data(), weights, weight_data.data(), scales, scale_data.data(), row, i1, i2, true);
                if (!close(actual, reference_f32, 1e-6f) || !close(actual, reference_f16, 4e-3f)) {
                    std::fprintf(stderr, "ConvRot mismatch at [%lld, %lld, %lld]: got %.8f, F32 %.8f, F16 %.8f\n",
                        (long long) row, (long long) i1, (long long) i2, actual, reference_f32, reference_f16);
                    passed = false;
                }
            }
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return passed;
}

bool run_empty_batch_case() {
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * activations = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kGroupSize, 0);
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, kGroupSize, kOutFeatures);
    ggml_tensor * scales = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kOutFeatures);
    ggml_tensor * result = ggml_mul_mat_convrot(ctx, activations, weights, scales, kGroupSize);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    ggml_backend_t backend = init_cpu_backend();
    ggml_backend_buffer_t buffer = backend ? ggml_backend_alloc_ctx_tensors(ctx, backend) : nullptr;
    const bool passed = backend && buffer && result->ne[0] == kOutFeatures && result->ne[1] == 0 &&
        ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (buffer) ggml_backend_buffer_free(buffer);
    if (backend) ggml_backend_free(backend);
    ggml_free(ctx);
    return passed;
}

} // namespace

int main() {
    const bool passed = run_case(GGML_TYPE_F32, true) && run_case(GGML_TYPE_F16, false) && run_empty_batch_case();
    if (!passed) {
        return 1;
    }

    std::puts("ConvRot CPU reference test passed");
    return 0;
}
