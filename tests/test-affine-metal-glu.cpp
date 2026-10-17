#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// Ordinary and affine projections have the same quantized type and input.
// A generic SwiGLU kernel must not interpret the affine carrier as a weight.
static bool check_case(ggml_backend_t backend, ggml_type type, bool affine_gate, bool up_first) {
    constexpr int K = 32, N = 64;
    auto * ctx = ggml_init({ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false), nullptr, true});
    auto * w = ggml_new_tensor_2d(ctx, type, K, N);
    auto * codes = ggml_new_tensor_2d(ctx, type, K, N);
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
    auto * step = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    auto * minimum = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); return false; }
    const size_t block_size = ggml_type_size(type);
    std::vector<unsigned char> ordinary(ggml_nbytes(w)), carrier(ggml_nbytes(codes));
    for (int row = 0; row < N; ++row) {
        const ggml_fp16_t unit = ggml_fp32_to_fp16(1.0f);
        std::memcpy(ordinary.data() + row * block_size, &unit, 2);
        std::memcpy(carrier.data() + row * block_size, &unit, 2);
        std::memset(ordinary.data() + row * block_size + 2, type == GGML_TYPE_Q4_0 ? 0x99 : 1, block_size - 2);
        // Unsigned code 8 decodes as zero in ordinary Q4_0, -120 in Q8_0.
        std::memset(carrier.data() + row * block_size + 2, 0x88, block_size - 2);
    }
    std::vector<float> ones(N, 1.0f), zeros(N, 0.0f);
    ggml_backend_tensor_set(w, ordinary.data(), 0, ordinary.size());
    ggml_backend_tensor_set(codes, carrier.data(), 0, carrier.size());
    ggml_backend_tensor_set(x, ones.data(), 0, K * sizeof(float));
    ggml_backend_tensor_set(step, ones.data(), 0, N * sizeof(float));
    ggml_backend_tensor_set(minimum, zeros.data(), 0, N * sizeof(float));
    auto * normal = ggml_mul_mat(ctx, w, x);
    auto * affine = ggml_mul_mat_affine_v1(ctx, codes, x, step, minimum);
    auto * gate = affine_gate ? affine : normal;
    auto * up = affine_gate ? normal : affine;
    auto * out = ggml_swiglu_split(ctx, gate, up);
    auto * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, up_first ? up : gate);
    ggml_build_forward_expand(graph, out);
    auto * alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool ok = ggml_gallocr_alloc_graph(alloc, graph) &&
              ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    std::vector<float> actual(N);
    if (ok) {
        ggml_backend_tensor_get(out, actual.data(), 0, N * sizeof(float));
        const float g = affine_gate ? 256.0f : 32.0f;
        const float u = affine_gate ? 32.0f : 256.0f;
        const float expected = g / (1.0f + std::exp(-g)) * u;
        for (float v : actual) ok = ok && std::isfinite(v) && std::abs(v - expected) <= 0.001f * expected;
        if (!ok) std::fprintf(stderr, "FAIL type=%s affine_gate=%d up_first=%d got=%g expected=%g\n",
                              ggml_type_name(type), affine_gate, up_first, actual[0], expected);
    }
    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main() {
    ggml_backend_load_all();
    auto * dev = ggml_backend_dev_by_name("MTL0");
    if (!dev) return 77;
    auto * backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) return 1;
    auto * probe = ggml_init({ggml_tensor_overhead() * 8, nullptr, true});
    auto * codes = ggml_new_tensor_2d(probe, GGML_TYPE_Q4_0, 32, 64);
    auto * input = ggml_new_tensor_2d(probe, GGML_TYPE_F32, 32, 1);
    auto * step = ggml_new_tensor_1d(probe, GGML_TYPE_F32, 64);
    auto * minimum = ggml_new_tensor_1d(probe, GGML_TYPE_F32, 64);
    const bool supported = ggml_backend_supports_op(backend,
        ggml_mul_mat_affine_v1(probe, codes, input, step, minimum));
    ggml_free(probe);
    if (!supported) { ggml_backend_free(backend); return 77; }
    bool ok = true;
    for (auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0})
        for (bool affine_gate : {false, true})
            for (bool up_first : {false, true})
                ok = check_case(backend, type, affine_gate, up_first) && ok;
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
