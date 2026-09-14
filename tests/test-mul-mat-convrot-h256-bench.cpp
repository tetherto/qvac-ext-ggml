#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <vector>

namespace {
constexpr int64_t kH = 256;
constexpr int64_t kGroups = 21;
constexpr int64_t kK = kH * kGroups;
constexpr int64_t kN = 14336;
constexpr int kWarmup = 3;
constexpr int kIterations = 20;

void convrot(float * values) {
    std::array<float, kH> tmp;
    for (int stride = 1; stride < kH; stride *= 4) {
        for (int base = 0; base < kH; base += 4 * stride) for (int i = 0; i < stride; ++i) {
            const float a = values[base + i], b = values[base + stride + i];
            const float c = values[base + 2 * stride + i], d = values[base + 3 * stride + i];
            tmp[base + i] = (a + b + c - d) * 0.5f;
            tmp[base + stride + i] = (a + b - c + d) * 0.5f;
            tmp[base + 2 * stride + i] = (a - b + c + d) * 0.5f;
            tmp[base + 3 * stride + i] = (-a + b + c + d) * 0.5f;
        }
        std::memcpy(values, tmp.data(), sizeof(float) * kH);
    }
}

bool compute(ggml_backend_t backend, ggml_cgraph * graph) {
    for (int i = 0; i < kWarmup; ++i) if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kIterations; ++i) if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / kIterations;
    std::printf("%.3f", ms);
    return true;
}
}

int main() {
    ggml_context * ctx = ggml_init({ 256 * 1024 * 1024, nullptr, true });
    if (!ctx) return 1;
    // H is deliberately materialized. An unaware backend therefore runs a
    // correct regular MUL_MAT; CUDA replaces it with the hint's H256 kernel.
    ggml_tensor * h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kH, kH);
    ggml_tensor * x_groups = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kH, kGroups);
    ggml_tensor * x = ggml_reshape_2d(ctx, x_groups, kK, 1);
    ggml_tensor * wi8 = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, kK, kN);
    ggml_tensor * scales = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kN);
    ggml_tensor * wf32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kK, kN);

    ggml_tensor * fused = ggml_mul_mat_convrot(ctx, x, wi8, scales, kH);
    ggml_tensor * rotated_groups = ggml_mul_mat(ctx, h, x_groups);
    ggml_mul_mat_set_hint(rotated_groups, GGML_HINT_SRC0_IS_CONVROT_H256);
    ggml_tensor * rotated = ggml_reshape_2d(ctx, rotated_groups, kK, 1);
    ggml_tensor * hinted = ggml_mul_mat(ctx, wf32, rotated);
    ggml_cgraph * fused_graph = ggml_new_graph(ctx);
    ggml_cgraph * hint_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(fused_graph, fused);
    ggml_build_forward_expand(hint_graph, hinted);

    std::vector<float> h_data(kH * kH), x_data(kK), scale_data(kN);
    std::vector<int8_t> i8_data(kK * kN);
    std::vector<float> f32_data(kK * kN);
    for (int col = 0; col < kH; ++col) {
        std::array<float, kH> basis = {}; basis[col] = 1.0f; convrot(basis.data());
        for (int row = 0; row < kH; ++row) h_data[col + row * kH] = basis[row];
    }
    for (int64_t i = 0; i < kK; ++i) x_data[i] = ((i * 13) % 97 - 48) * 0.03125f;
    for (int64_t row = 0; row < kN; ++row) {
        scale_data[row] = 0.00390625f * (row % 7 + 1);
        for (int64_t col = 0; col < kK; ++col) {
            const int8_t q = (int8_t) ((col * 17 + row * 31) % 255 - 127);
            i8_data[col + row * kK] = q;
            f32_data[col + row * kK] = q * scale_data[row];
        }
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    ggml_backend_buffer_t buffer = backend ? ggml_backend_alloc_ctx_tensors(ctx, backend) : nullptr;
    if (!buffer) return 2;
    ggml_backend_tensor_set(h, h_data.data(), 0, ggml_nbytes(h));
    ggml_backend_tensor_set(x_groups, x_data.data(), 0, ggml_nbytes(x_groups));
    ggml_backend_tensor_set(wi8, i8_data.data(), 0, ggml_nbytes(wi8));
    ggml_backend_tensor_set(scales, scale_data.data(), 0, ggml_nbytes(scales));
    ggml_backend_tensor_set(wf32, f32_data.data(), 0, ggml_nbytes(wf32));

    std::vector<float> fused_data(kN), hinted_data(kN);
    if (ggml_backend_graph_compute(backend, fused_graph) != GGML_STATUS_SUCCESS ||
        ggml_backend_graph_compute(backend, hint_graph) != GGML_STATUS_SUCCESS) return 3;
    ggml_backend_tensor_get(fused, fused_data.data(), 0, ggml_nbytes(fused));
    ggml_backend_tensor_get(hinted, hinted_data.data(), 0, ggml_nbytes(hinted));
    // Both paths retain FP32 scaled weight values. Any residual is only the
    // expected change in floating-point reduction order.
    float max_relative_error = 0.0f;
    for (int64_t i = 0; i < kN; ++i) {
        max_relative_error = std::max(max_relative_error,
            std::fabs(fused_data[i] - hinted_data[i]) / (1.0f + std::fabs(fused_data[i])));
    }
    std::printf("max FP32 path relative error: %.6f%%\n", 100.0f * max_relative_error);

    std::printf("ConvRot CUDA F32 benchmark (%lldx%lld): fused=", (long long) kK, (long long) kN);
    if (!compute(backend, fused_graph)) return 5;
    std::printf(" ms/run, hinted-H256+F32= ");
    if (!compute(backend, hint_graph)) return 6;
    std::printf(" ms/run\n");
    ggml_backend_buffer_free(buffer); ggml_backend_free(backend); ggml_free(ctx);
    return 0;
}
