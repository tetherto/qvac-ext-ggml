// Unit test for ggml_col2im_1d (ACE-Step Oobleck VAE op).
//
// The kernel computes the scatter-add of GEMM columns back into a 1D signal
// via a *gather* (each output time reads the columns that land on it).  Here we
// cross-check it against an independent *scatter* reference, so a bug in the
// index math on either side shows up as a mismatch.

#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include "col2im-1d-layout.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

// Scatter-add definition of col2im_1d.
//   col: [K*OC, T_in] contiguous, element (oc*K + k, t_in)
//   dst: [T_out, OC]  contiguous, element (t_out, oc)
//   T_out = (T_in - 1)*s0 + K - 2*p0
static std::vector<float> col2im_1d_reference(
        const std::vector<float> & col,
        int64_t K, int64_t OC, int64_t T_in, int s0, int p0) {

    const int64_t K_OC  = K * OC;
    const int64_t T_out = (T_in - 1) * s0 + K - 2 * p0;
    std::vector<float> dst(T_out * OC, 0.0f);

    for (int64_t t_in = 0; t_in < T_in; ++t_in) {
        for (int64_t k = 0; k < K; ++k) {
            const int64_t t_abs = t_in * s0 + k;   // uncropped position
            const int64_t t_out = t_abs - p0;
            if (t_out < 0 || t_out >= T_out) {
                continue;
            }
            for (int64_t oc = 0; oc < OC; ++oc) {
                dst[t_out + oc * T_out] += col[(oc * K + k) + t_in * K_OC];
            }
        }
    }
    return dst;
}

static bool check_equal(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        printf("  size mismatch: %d != %d\n", (int) a.size(), (int) b.size());
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > 1e-4f) {
            printf("  [%d] %f != %f\n", (int) i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

static bool test_col2im_1d(int64_t K, int64_t OC, int64_t T_in, int s0, int p0) {
    ggml_init_params params {
        /*.mem_size   =*/ 16 * ggml_tensor_overhead() + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr{ggml_init(params)};
    ggml_context * ctx = ctx_ptr.get();
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K * OC, T_in); // [K*OC, T_in]
    ggml_tensor * res = ggml_col2im_1d(ctx, a, s0, (int) OC, p0);
    ggml_build_forward_expand(gf, res);

    ggml_backend_ptr backend_ptr{ggml_backend_cpu_init()};
    ggml_backend_t backend = backend_ptr.get();
    ggml_backend_cpu_set_n_threads(backend, 2);
    ggml_backend_buffer_ptr buffer{ggml_backend_alloc_ctx_tensors(ctx, backend)};

    std::vector<float> a_values(ggml_nelements(a));
    std::iota(a_values.begin(), a_values.end(), 1.0f);
    ggml_backend_tensor_set(a, a_values.data(), 0, ggml_nbytes(a));

    ggml_backend_graph_compute(backend, gf);

    std::vector<float> res_values(ggml_nelements(res));
    ggml_backend_tensor_get(res, res_values.data(), 0, ggml_nbytes(res));

    const std::vector<float> expected = col2im_1d_reference(a_values, K, OC, T_in, s0, p0);

    const bool passed = check_equal(res_values, expected);
    printf("ggml_col2im_1d(K=%d, OC=%d, T_in=%d, s0=%d, p0=%d) -> T_out=%d: %s\n",
        (int) K, (int) OC, (int) T_in, s0, p0, (int) res->ne[0],
        passed ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");
    return passed;
}

static bool test_hexagon_layout() {
#define CHECK_LAYOUT(value) do { if (!(value)) { printf("FAIL layout line %d: %s\n", __LINE__, #value); return false; } } while (0)
    const size_t budget = 8 * 1024 * 1024;
    htp_col2im_1d_vtcm_layout layout = {};

    // Oversized staging must not wrap into a small worker allocation.
    ggml_context_ptr ctx{ggml_init({4 * ggml_tensor_overhead(), nullptr, true})};
    CHECK_LAYOUT(ctx != nullptr);
    ggml_tensor * cols = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 320 * 5808, 1);
    ggml_tensor * out = ggml_col2im_1d(ctx.get(), cols, 1, 5808, 0);
    CHECK_LAYOUT(ggml_nbytes(cols) == 7434240 && ggml_nbytes(out) == 7434240);
    CHECK_LAYOUT(!htp_col2im_1d_vtcm_layout_build(320, 5808, 1, 6, budget, &layout));

    CHECK_LAYOUT(htp_col2im_1d_vtcm_layout_build(16, 1, 4, 6, budget, &layout));
    CHECK_LAYOUT(layout.band == 256 && layout.taps_per_output == 4 && layout.off_tmp == 4480 &&
                 layout.off_out == 4992 && layout.bytes_per_thread == 6016 && layout.plan_bytes == 512);
    CHECK_LAYOUT(htp_col2im_1d_vtcm_layout_build(16, 1, 4, 6, 9728, &layout));
    CHECK_LAYOUT(layout.band == 32 && layout.bytes_per_thread == 1536);
    CHECK_LAYOUT(!htp_col2im_1d_vtcm_layout_build(16, 1, 4, 6, 9727, &layout));
    CHECK_LAYOUT(!htp_col2im_1d_vtcm_layout_build(16, 1, 4, 0, budget, &layout));
    CHECK_LAYOUT(!htp_col2im_1d_vtcm_layout_build(16, 0, 4, 6, budget, &layout));
    CHECK_LAYOUT(!htp_col2im_1d_vtcm_layout_build(16, 1, 0, 6, budget, &layout));
    CHECK_LAYOUT(!htp_col2im_1d_vtcm_layout_build(7, 1, 4, 6, budget, &layout));
    CHECK_LAYOUT(!htp_col2im_1d_vtcm_layout_build(16, 1, 4, 6, 0, &layout));
    CHECK_LAYOUT(!htp_col2im_1d_vtcm_layout_build(UINT32_MAX, UINT32_MAX, 1, 6, SIZE_MAX, &layout));
    printf("Hexagon col2im layout guards: PASSED\n");
#undef CHECK_LAYOUT
    return true;
}

int main() {
    bool ok = true;
    ok &= test_hexagon_layout();
    ok &= test_col2im_1d(/*K=*/4, /*OC=*/3, /*T_in=*/10, /*s0=*/2, /*p0=*/1);
    ok &= test_col2im_1d(/*K=*/7, /*OC=*/1, /*T_in=*/32, /*s0=*/4, /*p0=*/2); // mono, stride 4
    ok &= test_col2im_1d(/*K=*/3, /*OC=*/8, /*T_in=*/16, /*s0=*/1, /*p0=*/0); // no stride/pad
    ok &= test_col2im_1d(/*K=*/6, /*OC=*/2, /*T_in=*/5,  /*s0=*/3, /*p0=*/2); // K > s0 overlap
    return ok ? 0 : 1;
}
