#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

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
    ggml_init_params params = { 2 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return 1;

    // A genuine matrix is deliberately supplied: unknown hints are therefore
    // correct ordinary matmuls on CPU, BLAS, and any third-party backend.
    ggml_tensor * h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kH, kH);
    for (int col = 0; col < kH; ++col) {
        std::array<float, kH> basis = {};
        basis[col] = 1.0f;
        convrot_h256(basis.data());
        for (int row = 0; row < kH; ++row) {
            ((float *) h->data)[col + row * kH] = basis[row];
        }
    }

    constexpr int kRows = 3;
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kH, kRows);
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kH; ++col) {
            ((float *) x->data)[col + row * kH] = ((col * 17 + row * 31) % 97 - 48) * 0.03125f;
        }
    }

    ggml_tensor * y = ggml_mul_mat(ctx, h, x);
    ggml_mul_mat_set_hint(y, GGML_HINT_SRC0_IS_CONVROT_H256);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    if (ggml_graph_compute_with_ctx(ctx, graph, 4) != GGML_STATUS_SUCCESS) return 2;

    for (int row = 0; row < kRows; ++row) {
        std::array<float, kH> expected;
        std::memcpy(expected.data(), (float *) x->data + row * kH, sizeof(float) * kH);
        convrot_h256(expected.data());
        for (int col = 0; col < kH; ++col) {
            const float actual = ((float *) y->data)[col + row * kH];
            if (std::fabs(actual - expected[col]) > 2e-5f) {
                std::fprintf(stderr, "mismatch at [%d, %d]: %f != %f\n", col, row, actual, expected[col]);
                return 3;
            }
        }
    }
    ggml_free(ctx);
    return 0;
}
