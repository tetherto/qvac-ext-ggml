// Force the VMM reservation to fail, then check a real CUDA matmul against a
// host reference. A second backend must reuse the cached cudaMalloc fallback.
#include "ggml.h"
#include "ggml-backend.h"
#include <cuda.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static std::atomic<int> fallback_count{0};

static void capture_log(enum ggml_log_level, const char * text, void *) {
    std::fputs(text, stderr);
    if (std::strstr(text, "falling back to cudaMalloc")) {
        ++fallback_count;
    }
}

static bool run_matmul(ggml_backend_t backend) {
    constexpr int k = 128, m = 64, n = 64;
    ggml_context * ctx = ggml_init({ggml_graph_overhead() + 3 * ggml_tensor_overhead(), nullptr, true});
    if (!ctx) return false;
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, m);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * c = ggml_mul_mat(ctx, a, b);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }
    std::vector<ggml_fp16_t> av(k * m);
    std::vector<float> bv(k * n), actual(m * n);
    for (int i = 0; i < k * m; ++i) av[i] = ggml_fp32_to_fp16((i % 17 - 8) * 0.125f);
    for (int i = 0; i < k * n; ++i) bv[i] = (i % 13 - 6) * 0.125f;
    ggml_backend_tensor_set(a, av.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, bv.data(), 0, ggml_nbytes(b));
    bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (ok) {
        ggml_backend_tensor_get(c, actual.data(), 0, ggml_nbytes(c));
        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < m; ++row) {
                float expected = 0;
                for (int j = 0; j < k; ++j) expected += ggml_fp16_to_fp32(av[row * k + j]) * bv[col * k + j];
                const float value = actual[col * m + row];
                if (!std::isfinite(value) || std::fabs(value - expected) > 1e-3f) ok = false;
            }
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main() {
    if (!std::getenv("GGML_CUDA_TEST_VMM_RESERVE_FAIL")) {
        std::fprintf(stderr, "This test requires GGML_CUDA_TEST_VMM_RESERVE_FAIL=1\n");
        return 1;
    }
    CUdevice device;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&device, 0) != CUDA_SUCCESS) return 77;
    int vmm = 0;
    if (cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device) != CUDA_SUCCESS) return 1;
    if (!vmm) return 77;

    ggml_log_set(capture_log, nullptr);
    ggml_backend_load_all();
    for (int repetition = 0; repetition < 2; ++repetition) {
        // Select by name so integrated GPUs such as GB10 are included.
        ggml_backend_t backend = ggml_backend_init_by_name("CUDA0", nullptr);
        if (!backend) return 77;
        const bool ok = run_matmul(backend);
        ggml_backend_free(backend);
        if (!ok) {
            std::fprintf(stderr, "CUDA matmul failed after VMM reservation failure\n");
            return 1;
        }
    }
    if (fallback_count != 1) {
        std::fprintf(stderr, "Expected one VMM fallback, observed %d\n", fallback_count.load());
        return 1;
    }
    std::puts("CUDA VMM fallback: both matrix results correct; failed reservation cached");
    return 0;
}
