#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int kH = 256;

// Closed-form Kronecker product of the four ComfyUI radix-4 sign matrices.
// Unlike the CUDA kernel, this oracle does not execute any butterfly stages.
float dense_h256_coefficient(int row, int col) {
    constexpr int signs[4][4] = {
        { 1,  1,  1, -1},
        { 1,  1, -1,  1},
        { 1, -1,  1,  1},
        {-1,  1,  1,  1},
    };
    int sign = 1;
    for (int digit = 0; digit < 4; ++digit) {
        sign *= signs[row % 4][col % 4];
        row /= 4;
        col /= 4;
    }
    return sign * (1.0f / 16.0f);
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
    if (!backend) {
#if defined(GGML_TEST_CONVROT_H256_CUDA)
        std::fprintf(stderr, "CUDA backend unavailable; skipping H256 hint test\n");
        return 77;
#else
        return 1;
#endif
    }

    ggml_init_params params = { 2 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        return 1;
    }

    // CPU must fall back to a genuine matmul. CUDA gets a zero matrix so a
    // silent fallback cannot pass: only the hinted rotation yields the oracle.
    ggml_tensor * h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kH, kH);
    std::array<float, kH * kH> h_data = {};
#if !defined(GGML_TEST_CONVROT_H256_CUDA)
    for (int row = 0; row < kH; ++row) {
        for (int col = 0; col < kH; ++col) {
            h_data[col + row * kH] = dense_h256_coefficient(row, col);
        }
    }
#endif

    constexpr int kRows = 257;
    constexpr int kBatches = 3;
    constexpr float kScale = 1.25f;
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kH, kRows, kBatches);
    std::vector<float> x_data(kH * kRows * kBatches);
    for (int row = 0; row < kRows * kBatches; ++row) {
        for (int col = 0; col < kH; ++col) {
            x_data[col + row * kH] = ((col * 17 + row * 31) % 97 - 48) * 0.03125f;
        }
    }

    // Force a GPU producer immediately before the hinted kernel. This catches
    // a missing PDL dependency wait, which a host-upload-only test cannot.
    ggml_tensor * scaled = ggml_scale(ctx, x, kScale);
    ggml_tensor * y = ggml_mul_mat(ctx, h, scaled);
    ggml_mul_mat_set_hint(y, GGML_HINT_SRC0_IS_CONVROT_H256);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    std::vector<float> y_data(kH * kRows * kBatches);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 2;
    }
    ggml_backend_tensor_set(h, h_data.data(), 0, ggml_nbytes(h));
    ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));
    int status = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS ? 0 : 2;
    if (status == 0) ggml_backend_tensor_get(y, y_data.data(), 0, ggml_nbytes(y));

    for (int row = 0; row < kRows * kBatches && status == 0; ++row) {
        for (int col = 0; col < kH; ++col) {
            float expected = 0.0f;
            for (int input = 0; input < kH; ++input) {
                expected += dense_h256_coefficient(col, input) *
                            (x_data[input + row * kH] * kScale);
            }
            const float actual = y_data[col + row * kH];
            if (std::fabs(actual - expected) > 2e-5f) {
                std::fprintf(stderr, "mismatch at [%d, %d]: %f != %f\n", col, row, actual, expected);
                status = 3;
                break;
            }
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return status;
}
