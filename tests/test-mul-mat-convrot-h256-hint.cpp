#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#if defined(GGML_TEST_CONVROT_H256_CUDA)
#include "ggml-cuda.h"
#endif

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
    ggml_init_params params = { 2 * 1024 * 1024, nullptr,
#if defined(GGML_TEST_CONVROT_H256_CUDA)
        true
#else
        false
#endif
    };
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
#if !defined(GGML_TEST_CONVROT_H256_CUDA)
            ((float *) h->data)[col + row * kH] = basis[row];
#endif
        }
    }

    constexpr int kRows = 3;
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kH, kRows);
    std::array<float, kH * kRows> x_data;
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kH; ++col) {
            x_data[col + row * kH] = ((col * 17 + row * 31) % 97 - 48) * 0.03125f;
#if !defined(GGML_TEST_CONVROT_H256_CUDA)
            ((float *) x->data)[col + row * kH] = x_data[col + row * kH];
#endif
        }
    }

    ggml_tensor * y = ggml_mul_mat(ctx, h, x);
    ggml_mul_mat_set_hint(y, GGML_HINT_SRC0_IS_CONVROT_H256);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    std::array<float, kH * kRows> y_data;
#if defined(GGML_TEST_CONVROT_H256_CUDA)
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    ggml_backend_buffer_t buffer = backend ? ggml_backend_alloc_ctx_tensors(ctx, backend) : nullptr;
    if (!buffer) return 2;
    ggml_backend_tensor_set(h, h_data.data(), 0, ggml_nbytes(h));
    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 2;
    ggml_backend_tensor_get(y, y_data.data(), 0, ggml_nbytes(y));
#else
    if (ggml_graph_compute_with_ctx(ctx, graph, 4) != GGML_STATUS_SUCCESS) return 2;
#endif

    for (int row = 0; row < kRows; ++row) {
        std::array<float, kH> expected;
        std::memcpy(expected.data(), x_data.data() + row * kH, sizeof(float) * kH);
        convrot_h256(expected.data());
        for (int col = 0; col < kH; ++col) {
            const float actual =
#if defined(GGML_TEST_CONVROT_H256_CUDA)
                y_data[col + row * kH];
#else
                ((float *) y->data)[col + row * kH];
#endif
            if (std::fabs(actual - expected[col]) > 2e-5f) {
                std::fprintf(stderr, "mismatch at [%d, %d]: %f != %f\n", col, row, actual, expected[col]);
                return 3;
            }
        }
    }
#if defined(GGML_TEST_CONVROT_H256_CUDA)
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
#endif
    ggml_free(ctx);
    return 0;
}
