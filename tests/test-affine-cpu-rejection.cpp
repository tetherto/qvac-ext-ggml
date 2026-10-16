#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Terminate successfully only at the intended execution guard. This avoids
// platform-specific subprocess/death-test machinery and assertion dialogs.
static void rejected(const char * message) {
    if (std::strstr(message, "affine v1 requires its Metal kernel")) {
        std::puts("PASS: affine CPU execution rejected before kernel dispatch");
        std::fflush(stdout);
        std::_Exit(0);
    }
    std::fprintf(stderr, "FAIL: unexpected assertion: %s\n", message);
    std::_Exit(1);
}

int main(int argc, char ** argv) {
    if (argc != 3) return 1;
    const bool repack = std::strcmp(argv[1], "repack") == 0;
    const ggml_type type = std::strcmp(argv[2], "q4") == 0 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0;
    ggml_backend_load_all();
    auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) return 77;
    auto * backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) return 1;
    auto * buft = ggml_backend_dev_buffer_type(dev);
    if (repack) {
        auto extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts"));
        buft = nullptr;
        if (extra) {
            for (auto ** p = extra(dev); p && *p; ++p) {
                if (std::strcmp(ggml_backend_buft_name(*p), "CPU_REPACK") == 0) buft = *p;
            }
        }
        if (!buft) { std::puts("SKIP: CPU_REPACK unavailable"); return 77; }
    }
    constexpr int K = 32, N = 64;
    auto * weights = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
    auto * codes = ggml_new_tensor_2d(weights, type, K, N);
    auto * wbuf = ggml_backend_alloc_ctx_tensors_from_buft(weights, buft);
    if (!wbuf) return 1;
    // Buffer presence alone is insufficient: require actual repacking traits.
    if (repack && !codes->extra) { std::puts("SKIP: no repack kernel for this carrier/CPU"); return 77; }
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto * ctx = ggml_init({ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(32, false), nullptr, true});
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
    auto * step = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    auto * minimum = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    auto * ordinary = ggml_mul_mat(ctx, codes, x);
    auto * affine = ggml_mul_mat_affine_v1(ctx, codes, x, step, minimum);
    auto * buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) return 1;
    const size_t block_size = ggml_type_size(type);
    std::vector<unsigned char> data(ggml_nbytes(codes), 0x88);
    for (int row = 0; row < N; ++row) {
        const ggml_fp16_t unit = ggml_fp32_to_fp16(1.0f);
        std::memcpy(data.data() + row * block_size, &unit, sizeof(unit));
    }
    std::vector<float> ones(N, 1.0f), zeros(N, 0.0f);
    ggml_backend_tensor_set(codes, data.data(), 0, data.size());
    ggml_backend_tensor_set(x, ones.data(), 0, K * sizeof(float));
    ggml_backend_tensor_set(step, ones.data(), 0, N * sizeof(float));
    ggml_backend_tensor_set(minimum, zeros.data(), 0, N * sizeof(float));
    auto * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, ordinary);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 1;
    std::vector<float> out(N);
    ggml_backend_tensor_get(ordinary, out.data(), 0, N * sizeof(float));
    // Q4 code 8 is ordinary zero; the signed Q8 byte 0x88 is -120.
    const float expected = type == GGML_TYPE_Q4_0 ? 0.0f : -3840.0f;
    for (float value : out) {
        if (!std::isfinite(value) || std::fabs(value - expected) > 0.001f) {
            std::fprintf(stderr, "FAIL: ordinary %s in %s got %.9g, expected %.9g\n",
                         ggml_type_name(type), ggml_backend_buft_name(buft), value, expected);
            return 1;
        }
    }
    std::printf("Ordinary %s matmul passed in %s; attempting direct affine execution\n",
                ggml_type_name(type), ggml_backend_buft_name(buft));
    std::fflush(stdout);
    ggml_graph_clear(graph);
    ggml_build_forward_expand(graph, affine);
    ggml_set_abort_callback(rejected);
    // Deliberately bypass supports_op, as direct callers can do.
    ggml_backend_graph_compute(backend, graph);
    std::fputs("FAIL: affine CPU execution reached a kernel instead of rejecting\n", stderr);
    return 1;
}
