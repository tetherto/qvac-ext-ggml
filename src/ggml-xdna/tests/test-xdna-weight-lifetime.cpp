#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static constexpr int64_t WEIGHT_WIDTH = 1024;
static constexpr int64_t OUTPUT_WIDTH = 1024;
static constexpr int64_t BATCH_SIZE   = 32;
static constexpr float ACTIVATION    = 0.25f;
static constexpr float TOLERANCE     = 1e-4f;

struct resources {
    ggml_backend_t        backend = nullptr;
    ggml_backend_t        other   = nullptr;
    ggml_backend_buffer_t marker  = nullptr;
    ggml_backend_buffer_t tensors = nullptr;
    ggml_context *        ctx     = nullptr;

    ~resources() {
        ggml_backend_buffer_free(marker);
        ggml_backend_buffer_free(tensors);
        ggml_free(ctx);
        ggml_backend_free(backend);
        ggml_backend_free(other);
    }
};

static bool check_projection(ggml_backend_t backend, ggml_cgraph * graph,
                             ggml_tensor * weights, ggml_tensor * output, float value) {
    std::vector<float> values((size_t) ggml_nelements(weights), value);
    if (weights->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> packed(values.size());
        ggml_fp32_to_bf16_row(values.data(), packed.data(), (int64_t) values.size());
        ggml_backend_tensor_set(weights, packed.data(), 0, ggml_nbytes(weights));
    } else {
        std::vector<ggml_fp16_t> packed(values.size());
        ggml_fp32_to_fp16_row(values.data(), packed.data(), (int64_t) values.size());
        ggml_backend_tensor_set(weights, packed.data(), 0, ggml_nbytes(weights));
    }
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "XDNA projection failed\n");
        return false;
    }
    std::vector<float> actual((size_t) ggml_nelements(output));
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    const float expected = WEIGHT_WIDTH * ACTIVATION * value;
    if (!std::all_of(actual.begin(), actual.end(), [expected](float v) {
            return std::isfinite(v) && std::abs(v - expected) <= TOLERANCE;
        })) {
        std::fprintf(stderr, "Projection retained stale weights: expected %g, first value %g\n",
                     expected, actual.front());
        return false;
    }
    return true;
}

static bool check_lifetime(ggml_type type, ggml_backend_buffer_usage usage) {
    resources r;
    r.backend = ggml_backend_init_by_name("XDNA", nullptr);
    r.other   = ggml_backend_init_by_name("XDNA", nullptr);
    if (!r.backend || !r.other) {
        std::fprintf(stderr, "This test requires an AMD XDNA NPU and its backend\n");
        return false;
    }
    r.marker = ggml_backend_alloc_buffer(r.backend, 4096);
    if (!r.marker || std::strcmp(ggml_backend_buft_name(ggml_backend_buffer_get_type(r.marker)), "XDNA_Host")) {
        std::fprintf(stderr, "This test requires an XDNA weight buffer, without host allocation fallback\n");
        return false;
    }
    ggml_backend_buffer_set_usage(r.marker, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    const ggml_init_params params = { ggml_tensor_overhead() * 3 + ggml_graph_overhead(), nullptr, true };
    r.ctx = ggml_init(params);
    if (!r.ctx) {
        return false;
    }
    ggml_tensor * weights = ggml_new_tensor_2d(r.ctx, type, WEIGHT_WIDTH, OUTPUT_WIDTH);
    ggml_tensor * input   = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, WEIGHT_WIDTH, BATCH_SIZE);
    ggml_tensor * output  = ggml_mul_mat(r.ctx, weights, input);
    ggml_cgraph * graph   = ggml_new_graph(r.ctx);
    ggml_build_forward_expand(graph, output);
    r.tensors = ggml_backend_alloc_ctx_tensors_from_buft(r.ctx, ggml_backend_cpu_buffer_type());
    if (!r.tensors || !ggml_backend_supports_op(r.backend, output)) {
        std::fprintf(stderr, "This test requires a matching XDNA GEMM kernel for its projection\n");
        return false;
    }
    ggml_backend_buffer_set_usage(r.tensors, usage);
    std::vector<float> inputs((size_t) ggml_nelements(input), ACTIVATION);
    ggml_backend_tensor_set(input, inputs.data(), 0, ggml_nbytes(input));
    if (!check_projection(r.backend, graph, weights, output, 1.0f)) {
        return false;
    }
    if (!check_projection(r.backend, graph, weights, output, 2.0f)) {
        return false;
    }

    ggml_backend_buffer_free(r.marker);
    r.marker = nullptr;
    ggml_backend_free(r.backend);
    r.backend = nullptr;
    if (!check_projection(r.other, graph, weights, output, 3.0f)) {
        return false;
    }
    return true;
}

int main() {
    setenv("GGML_XDNA_GLUE", "0", 1);
    ggml_backend_load_all();
    if (!check_lifetime(GGML_TYPE_BF16, GGML_BACKEND_BUFFER_USAGE_COMPUTE) ||
        !check_lifetime(GGML_TYPE_F16, GGML_BACKEND_BUFFER_USAGE_COMPUTE) ||
        !check_lifetime(GGML_TYPE_BF16, GGML_BACKEND_BUFFER_USAGE_WEIGHTS) ||
        !check_lifetime(GGML_TYPE_F16, GGML_BACKEND_BUFFER_USAGE_WEIGHTS)) {
        return 1;
    }
    std::puts("test-xdna-weight-lifetime: all passed");
    return 0;
}
