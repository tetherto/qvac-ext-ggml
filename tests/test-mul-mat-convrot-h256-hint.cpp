#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace {

constexpr int kH = 256;

void convrot_h256(float * values) {
    std::array<float, kH> tmp;
    for (int stride = 1; stride < kH; stride *= 4) {
        for (int base = 0; base < kH; base += 4 * stride) {
            for (int i = 0; i < stride; ++i) {
                const float a = values[base + 0 * stride + i];
                const float b = values[base + 1 * stride + i];
                const float c = values[base + 2 * stride + i];
                const float d = values[base + 3 * stride + i];
                tmp[base + 0 * stride + i] = ( a + b + c - d) * 0.5f;
                tmp[base + 1 * stride + i] = ( a + b - c + d) * 0.5f;
                tmp[base + 2 * stride + i] = ( a - b + c + d) * 0.5f;
                tmp[base + 3 * stride + i] = (-a + b + c + d) * 0.5f;
            }
        }
        std::memcpy(values, tmp.data(), sizeof(float) * kH);
    }
}

} // namespace

int main() {
    ggml_backend_load_all();
    ggml_backend_t backend = nullptr;
#if defined(GGML_TEST_CONVROT_H256_CUDA)
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "CUDA") == 0) {
            backend = ggml_backend_dev_init(dev, nullptr);
            break;
        }
    }
#else
    backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
#endif
    if (!backend) return 1;

    ggml_init_params params = { 2 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return 1;

    // A genuine matrix is deliberately supplied: unknown hints are therefore
    // correct ordinary matmuls on CPU, BLAS, and any third-party backend.
    ggml_tensor * h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kH, kH);
    std::array<float, kH * kH> h_data;
    for (int col = 0; col < kH; ++col) {
        std::array<float, kH> basis = {};
        basis[col] = 1.0f;
        convrot_h256(basis.data());
        for (int row = 0; row < kH; ++row) {
            h_data[col + row * kH] = basis[row];
        }
    }

    constexpr int kRows = 3;
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kH, kRows);
    std::array<float, kH * kRows> x_data;
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kH; ++col) {
            x_data[col + row * kH] = ((col * 17 + row * 31) % 97 - 48) * 0.03125f;
        }
    }

    ggml_tensor * y = ggml_mul_mat(ctx, h, x);
    ggml_mul_mat_set_hint(y, GGML_HINT_SRC0_IS_CONVROT_H256);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    std::array<float, kH * kRows> y_data;
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 2;
    ggml_backend_tensor_set(h, h_data.data(), 0, ggml_nbytes(h));
    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 2;
    ggml_backend_tensor_get(y, y_data.data(), 0, ggml_nbytes(y));

    for (int row = 0; row < kRows; ++row) {
        std::array<float, kH> expected;
        std::memcpy(expected.data(), x_data.data() + row * kH, sizeof(float) * kH);
        convrot_h256(expected.data());
        for (int col = 0; col < kH; ++col) {
            const float actual = y_data[col + row * kH];
            if (std::fabs(actual - expected[col]) > 2e-5f) {
                std::fprintf(stderr, "mismatch at [%d, %d]: %f != %f\n", col, row, actual, expected[col]);
                return 3;
            }
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
