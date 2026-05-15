#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

struct test_config {
    const char * name;
    int d_head;
    int n_head;
    int L;
    int N;
};

// Build the OLD multi-op apply_rope graph (interleaved)
static ggml_tensor * build_rope_old(ggml_context * ctx,
                                     ggml_tensor * x,   // [d_head, n_head, L, N]
                                     ggml_tensor * pe) { // [2, 2, d_head/2, L]
    int64_t d_head = x->ne[0];
    int64_t n_head = x->ne[1];
    int64_t L      = x->ne[2];
    int64_t N      = x->ne[3];

    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    x = ggml_reshape_4d(ctx, x, 2, d_head / 2, L, n_head * N);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 3, 0, 1, 2));

    int64_t offset = x->nb[2] * x->ne[2];
    auto x_0 = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], offset * 0);
    auto x_1 = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], offset * 1);
    x_0 = ggml_reshape_4d(ctx, x_0, 1, x_0->ne[0], x_0->ne[1], x_0->ne[2]);
    x_1 = ggml_reshape_4d(ctx, x_1, 1, x_1->ne[0], x_1->ne[1], x_1->ne[2]);
    auto temp_x = ggml_new_tensor_4d(ctx, x_0->type, 2, x_0->ne[1], x_0->ne[2], x_0->ne[3]);
    x_0 = ggml_repeat(ctx, x_0, temp_x);
    x_1 = ggml_repeat(ctx, x_1, temp_x);

    pe = ggml_cont(ctx, ggml_permute(ctx, pe, 3, 0, 1, 2));
    offset = pe->nb[2] * pe->ne[2];
    auto pe_0 = ggml_view_3d(ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->nb[1], pe->nb[2], offset * 0);
    auto pe_1 = ggml_view_3d(ctx, pe, pe->ne[0], pe->ne[1], pe->ne[2], pe->nb[1], pe->nb[2], offset * 1);

    auto x_out = ggml_add(ctx, ggml_mul(ctx, x_0, pe_0), ggml_mul(ctx, x_1, pe_1));
    x_out = ggml_reshape_3d(ctx, x_out, d_head, L, n_head * N);
    return x_out;
}

static ggml_tensor * build_permute_old(ggml_context * ctx,
                                        ggml_tensor * x,
                                        ggml_tensor *) {
    int64_t d_head = x->ne[0];
    int64_t n_head = x->ne[1];
    int64_t L      = x->ne[2];
    int64_t N      = x->ne[3];

    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    return ggml_reshape_3d(ctx, x, d_head, L, n_head * N);
}

// Build the NEW fused graph
static ggml_tensor * build_rope_fused(ggml_context * ctx,
                                       ggml_tensor * x,
                                       ggml_tensor * pe) {
    return ggml_rope_flux(ctx, x, pe);
}

static ggml_tensor * build_permute_fused(ggml_context * ctx,
                                          ggml_tensor * x,
                                          ggml_tensor *) {
    return ggml_rope_flux(ctx, x, nullptr);
}

static float * run_graph(ggml_backend_t backend,
                          ggml_tensor * x_param,
                          ggml_tensor * pe_param,
                          const float * x_data,
                          const float * pe_data,
                          ggml_tensor * (*builder)(ggml_context *, ggml_tensor *, ggml_tensor *),
                          int * out_nodes) {
    size_t buf_size = ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(256, false);
    std::vector<uint8_t> buf(buf_size);

    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context * ctx = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 256, false);

    ggml_tensor * result = builder(ctx, x_param, pe_param);
    ggml_set_name(result, "result");
    ggml_build_forward_expand(gf, result);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_reserve(allocr, gf);

    // re-build for actual compute
    ggml_free(ctx);
    ctx = ggml_init(p);
    gf = ggml_new_graph_custom(ctx, 256, false);
    result = builder(ctx, x_param, pe_param);
    ggml_set_name(result, "result");
    ggml_build_forward_expand(gf, result);
    ggml_gallocr_alloc_graph(allocr, gf);

    // set input data
    ggml_backend_tensor_set(x_param, x_data, 0, ggml_nbytes(x_param));
    if (pe_param != nullptr) {
        ggml_backend_tensor_set(pe_param, pe_data, 0, ggml_nbytes(pe_param));
    }

    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, 1);
    }

    ggml_backend_synchronize(backend);
    int64_t t0 = ggml_time_us();
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    int64_t t1 = ggml_time_us();

    *out_nodes = ggml_graph_n_nodes(gf);

    result = ggml_graph_get_tensor(gf, "result");
    size_t nbytes = ggml_nbytes(result);
    float * out = (float *)malloc(nbytes);
    assert(out != nullptr);
    ggml_backend_tensor_get(result, out, 0, nbytes);

    printf("    time: %.2f ms, nodes: %d, output shape: [%lld, %lld, %lld]\n",
           (t1 - t0) / 1000.0, *out_nodes,
           (long long)result->ne[0], (long long)result->ne[1], (long long)result->ne[2]);

    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return out;
}

int main(void) {
    ggml_time_init();
    ggml_log_set(ggml_log_callback_default, nullptr);

    test_config configs[] = {
        { "small",       16,   4,   32, 1 },
        { "medium",      64,   8,  128, 1 },
        { "flux_klein", 128,  24, 4352, 1 },
        { "flux_batch", 128,  24,  256, 2 },
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    ggml_backend_t backend = nullptr;
#ifdef GGML_USE_METAL
    backend = ggml_backend_metal_init();
    if (backend) {
        printf("Using Metal backend\n\n");
    }
#endif
    if (!backend) {
        backend = ggml_backend_cpu_init();
        if (!backend) {
            fprintf(stderr, "Backend init failed\n");
            return 1;
        }
        printf("Using CPU backend\n\n");
    }

    int pass = 0, fail = 0;

    for (int c = 0; c < n_configs; c++) {
        auto & cfg = configs[c];
        printf("=== %s: d_head=%d, n_head=%d, L=%d, N=%d ===\n",
               cfg.name, cfg.d_head, cfg.n_head, cfg.L, cfg.N);

        int x_elems  = cfg.d_head * cfg.n_head * cfg.L * cfg.N;
        int pe_elems = 2 * 2 * (cfg.d_head / 2) * cfg.L;

        // generate deterministic test data
        std::vector<float> x_data(x_elems);
        std::vector<float> pe_data(pe_elems);
        srand(42 + c);
        for (int i = 0; i < x_elems; i++) {
            x_data[i] = ((float)(rand() % 2000) - 1000.0f) / 1000.0f;
        }
        for (int i = 0; i < cfg.L; i++) {
            for (int p = 0; p < cfg.d_head / 2; p++) {
                float theta = (float)i / powf(10000.0f, 2.0f * p / cfg.d_head);
                float cos_v = cosf(theta);
                float sin_v = sinf(theta);
                int base = i * (cfg.d_head / 2) * 4 + p * 4;
                pe_data[base + 0] = cos_v;   // [0,0] = cos
                pe_data[base + 1] = -sin_v;  // [1,0] = -sin
                pe_data[base + 2] = sin_v;   // [0,1] = sin
                pe_data[base + 3] = cos_v;   // [1,1] = cos
            }
        }

        // allocate parameter tensors on backend
        size_t buf_size = x_elems * sizeof(float) + pe_elems * sizeof(float) + 4096;
        ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(backend, buf_size);

        ggml_init_params mp = { ggml_tensor_overhead() * 4, NULL, true };
        ggml_context * mctx = ggml_init(mp);

        ggml_tensor * x_param  = ggml_new_tensor_4d(mctx, GGML_TYPE_F32, cfg.d_head, cfg.n_head, cfg.L, cfg.N);
        ggml_tensor * pe_param = ggml_new_tensor_4d(mctx, GGML_TYPE_F32, 2, 2, cfg.d_head / 2, cfg.L);

        ggml_tallocr alloc = ggml_tallocr_new(buffer);
        ggml_tallocr_alloc(&alloc, x_param);
        ggml_tallocr_alloc(&alloc, pe_param);

        // run old pipeline
        printf("  OLD (multi-op):\n");
        int old_nodes = 0;
        float * old_out = run_graph(backend, x_param, pe_param, x_data.data(), pe_data.data(),
                                    build_rope_old, &old_nodes);

        // run new fused kernel
        printf("  NEW (fused):\n");
        int new_nodes = 0;
        float * new_out = run_graph(backend, x_param, pe_param, x_data.data(), pe_data.data(),
                                    build_rope_fused, &new_nodes);

        // compare outputs
        int out_elems = cfg.d_head * cfg.L * cfg.N * cfg.n_head;
        float max_abs = 0.0f;
        float max_rel = 0.0f;
        int max_abs_idx = 0;
        double old_sum = 0, new_sum = 0, old_sqsum = 0, new_sqsum = 0;
        int n_nonzero_old = 0, n_nonzero_new = 0;
        int n_exact = 0;

        for (int i = 0; i < out_elems; i++) {
            float diff = fabsf(old_out[i] - new_out[i]);
            float rel = (fabsf(old_out[i]) > 1e-6f) ? diff / fabsf(old_out[i]) : 0.0f;
            if (diff > max_abs) {
                max_abs = diff;
                max_abs_idx = i;
            }
            if (rel > max_rel) max_rel = rel;
            if (old_out[i] == new_out[i]) n_exact++;
            if (fabsf(old_out[i]) > 1e-8f) n_nonzero_old++;
            if (fabsf(new_out[i]) > 1e-8f) n_nonzero_new++;
            old_sum += old_out[i];
            new_sum += new_out[i];
            old_sqsum += (double)old_out[i] * old_out[i];
            new_sqsum += (double)new_out[i] * new_out[i];
        }

        double old_mean = old_sum / out_elems;
        double new_mean = new_sum / out_elems;
        double old_std = sqrt(old_sqsum / out_elems - old_mean * old_mean);
        double new_std = sqrt(new_sqsum / out_elems - new_mean * new_mean);

        printf("  OLD stats: mean=%.6f, std=%.6f, nonzero=%d/%d\n",
               old_mean, old_std, n_nonzero_old, out_elems);
        printf("  NEW stats: mean=%.6f, std=%.6f, nonzero=%d/%d\n",
               new_mean, new_std, n_nonzero_new, out_elems);
        printf("  Sample values [0..4]: old=[%.4f, %.4f, %.4f, %.4f, %.4f]\n",
               old_out[0], old_out[1], old_out[2], old_out[3], old_out[4]);
        printf("                        new=[%.4f, %.4f, %.4f, %.4f, %.4f]\n",
               new_out[0], new_out[1], new_out[2], new_out[3], new_out[4]);
        printf("  Exact matches: %d/%d (%.1f%%)\n", n_exact, out_elems, 100.0f * n_exact / out_elems);

        bool ok = max_abs < 1e-4f && n_nonzero_old > out_elems / 2 && n_nonzero_new > out_elems / 2;
        printf("  COMPARE: max_abs=%.6f (at idx %d: old=%.6f new=%.6f), max_rel=%.4f%% => %s\n",
               max_abs, max_abs_idx,
               old_out[max_abs_idx], new_out[max_abs_idx],
               max_rel * 100.0f,
               ok ? "PASS" : "FAIL");
        printf("  nodes: old=%d, new=%d (-%d)\n\n",
               old_nodes, new_nodes, old_nodes - new_nodes);

        if (ok) pass++; else fail++;

        free(old_out);
        free(new_out);

        // compare permute-only path (b == NULL)
        printf("  OLD permute-only:\n");
        int old_perm_nodes = 0;
        float * old_perm_out = run_graph(backend, x_param, nullptr, x_data.data(), nullptr,
                                         build_permute_old, &old_perm_nodes);

        printf("  NEW permute-only (fused):\n");
        int new_perm_nodes = 0;
        float * new_perm_out = run_graph(backend, x_param, nullptr, x_data.data(), nullptr,
                                         build_permute_fused, &new_perm_nodes);

        float perm_max_abs = 0.0f;
        int perm_max_abs_idx = 0;
        for (int i = 0; i < out_elems; i++) {
            float diff = fabsf(old_perm_out[i] - new_perm_out[i]);
            if (diff > perm_max_abs) {
                perm_max_abs = diff;
                perm_max_abs_idx = i;
            }
        }

        bool perm_ok = perm_max_abs < 1e-6f;
        printf("  PERMUTE COMPARE: max_abs=%.6f (at idx %d: old=%.6f new=%.6f) => %s\n",
               perm_max_abs, perm_max_abs_idx,
               old_perm_out[perm_max_abs_idx], new_perm_out[perm_max_abs_idx],
               perm_ok ? "PASS" : "FAIL");
        printf("  nodes: old=%d, new=%d (-%d)\n\n",
               old_perm_nodes, new_perm_nodes, old_perm_nodes - new_perm_nodes);

        if (perm_ok) pass++; else fail++;

        free(old_perm_out);
        free(new_perm_out);
        ggml_free(mctx);
        ggml_backend_buffer_free(buffer);
    }

    ggml_backend_free(backend);

    printf("=== Results: %d/%d passed", pass, pass + fail);
    if (fail > 0) printf(", %d FAILED", fail);
    printf(" ===\n");

    return fail > 0 ? 1 : 0;
}
