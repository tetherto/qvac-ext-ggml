#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

// CPU F16 MUL_MAT converts F32 activations to F16 even with GGML_PREC_F32.
// Instead, check explicitly dequantized F32 weights against a double-precision
// host dot product, then use that independent result for the Hexagon candidate.
// The cancellation inputs below lose their entire signal if narrowed to F16.
struct shape {
    const char * name;
    int k, channels, rows, batches;
    bool views, bias, panel;
};

static const shape cases[] = {
    {"k64-odd",       64,  7,  5, 1, false, false, true},
    {"k96-odd",       96,  7,  5, 1, false, false, true},
    {"k96-views",     96,  7,  5, 1, true,  false, true},
    {"k192-odd",     192, 11,  9, 1, false, false, true},
    {"k384-views",   384,  9,  7, 1, true,  false, true},
    // M>1 explicit-F32 MUL_MAT cannot fuse with ADD on the current backend;
    // the standalone panel product must still feed the normal bias operation.
    {"separate-bias",  96,  7,  5, 1, false, true,  true},
    {"channels96",    96, 96, 33, 1, false, false, true},
    {"channels192",  192,192, 33, 1, false, false, true},
    {"channels384",  384,384, 33, 1, false, false, true},
    // Ineligible shapes retain the existing F16-activation arithmetic.
    {"guard-k32",     32,  7,  5, 1, false, false, false},
    {"guard-batched", 96,  7,  5, 2, false, false, false},
    {"guard-k97",     97,  7,  5, 1, true,  false, false},
    {"guard-one-col", 96,  1,  5, 1, false, false, false},
    {"guard-one-row", 96,  7,  1, 1, false, false, false},
};

struct inputs {
    std::vector<float> weights, activations, expected;
};

static inputs make_inputs(const shape & s, bool narrow) {
    inputs in;
    in.weights.resize(size_t(s.k) * s.channels * s.batches);
    in.activations.resize(size_t(s.k) * s.rows * s.batches);
    in.expected.resize(size_t(s.channels) * s.rows * s.batches);
    for (int batch = 0; batch < s.batches; ++batch) {
        for (int c = 0; c < s.channels; ++c) {
            for (int k = 0; k < s.k; ++k) {
                const float weight = k == s.k - 1 && s.k % 2 ? 0.0f
                                     : float(1 + c % 4) * (k % 2 ? -1.0f : 1.0f);
                in.weights[(size_t(batch) * s.channels + c) * s.k + k] =
                    ggml_fp16_to_fp32(ggml_fp32_to_fp16(weight));
            }
        }
        for (int r = 0; r < s.rows; ++r) {
            for (int k = 0; k < s.k; ++k) {
                const float base = 1.0f + float((k / 2 + r) % 8) / 16.0f;
                const float delta = float(1 + (r + batch) % 3) / 8192.0f;
                float x = base + (k % 2 ? 0.0f : delta);
                if (narrow) x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
                in.activations[(size_t(batch) * s.rows + r) * s.k + k] = x;
            }
            for (int c = 0; c < s.channels; ++c) {
                double sum = 0;
                for (int k = 0; k < s.k; ++k) {
                    sum += double(in.weights[(size_t(batch) * s.channels + c) * s.k + k]) *
                           in.activations[(size_t(batch) * s.rows + r) * s.k + k];
                }
                if (s.bias) sum += float(c + 1) / 16.0f;
                in.expected[(size_t(batch) * s.rows + r) * s.channels + c] = float(sum);
            }
        }
    }
    return in;
}

struct graph_storage {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~graph_storage() { ggml_backend_buffer_free(buffer); ggml_free(ctx); }
};

// Fill row padding with NaNs: a partial-K vector must mask both operands.
static void upload(ggml_tensor * storage, const std::vector<float> & logical,
                   int k, int rows, int batches, int offset) {
    std::vector<float> data(ggml_nelements(storage), std::numeric_limits<float>::quiet_NaN());
    for (int b = 0; b < batches; ++b) {
        for (int r = 0; r < rows; ++r) {
            std::copy_n(logical.data() + (size_t(b) * rows + r) * k, k,
                        data.data() + (size_t(b) * rows + r) * storage->ne[0] + offset);
        }
    }
    if (storage->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(data.size());
        for (size_t i = 0; i < data.size(); ++i) half[i] = ggml_fp32_to_fp16(data[i]);
        ggml_backend_tensor_set(storage, half.data(), 0, half.size() * sizeof(half[0]));
    } else {
        ggml_backend_tensor_set(storage, data.data(), 0, data.size() * sizeof(data[0]));
    }
}

static bool compute(ggml_backend_t backend, const shape & s, const inputs & in,
                    ggml_type weight_type, std::vector<float> & result) {
    graph_storage storage;
    storage.ctx = ggml_init({ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true});
    if (!storage.ctx) return false;
    auto * ctx = storage.ctx;
    auto * graph = ggml_new_graph(ctx);
    auto * a = ggml_new_tensor_3d(ctx, weight_type, s.k + (s.views ? 13 : 0), s.channels, s.batches);
    auto * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.k + (s.views ? 11 : 0), s.rows, s.batches);
    auto * av = s.views ? ggml_view_3d(ctx, a, s.k, s.channels, s.batches, a->nb[1], a->nb[2], a->nb[0]) : a;
    auto * bv = s.views ? ggml_view_3d(ctx, b, s.k, s.rows, s.batches, b->nb[1], b->nb[2], b->nb[0]) : b;
    auto * out = ggml_mul_mat(ctx, av, bv);
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    ggml_tensor * bias = nullptr;
    if (s.bias) {
        bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, s.channels);
        out = ggml_add(ctx, out, bias);
    }
    ggml_build_forward_expand(graph, out);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (!ggml_backend_supports_op(backend, ggml_graph_node(graph, i))) {
            std::printf("%s: unsupported node on %s\n", s.name, ggml_backend_name(backend));
            return false;
        }
    }
    storage.buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!storage.buffer) return false;
    upload(a, in.weights, s.k, s.channels, s.batches, s.views ? 1 : 0);
    upload(b, in.activations, s.k, s.rows, s.batches, s.views ? 1 : 0);
    if (bias) {
        std::vector<float> data(s.channels);
        for (int c = 0; c < s.channels; ++c) data[c] = float(c + 1) / 16.0f;
        ggml_backend_tensor_set(bias, data.data(), 0, data.size() * sizeof(float));
    }
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
    result.resize(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    return true;
}

static bool compare(const char * backend, const shape & s, const std::vector<float> & actual,
                    const std::vector<float> & expected) {
    bool ok = actual.size() == expected.size();
    double max_abs = 0;
    for (size_t i = 0; i < actual.size() && i < expected.size(); ++i) {
        const double error = std::abs(double(actual[i]) - expected[i]);
        ok = std::isfinite(actual[i]) && error <= 2e-6 && ok;
        max_abs = std::max(max_abs, error);
    }
    std::printf("%s %s k=%d channels=%d rows=%d batches=%d max_abs=%.9g %s\n",
                backend, s.name, s.k, s.channels, s.rows, s.batches, max_abs, ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char ** argv) {
    const bool reference_only = argc == 2 && std::strcmp(argv[1], "--reference-only") == 0;
    if (argc != 1 && !reference_only) {
        std::fprintf(stderr, "usage: %s [--reference-only]\n", argv[0]);
        return 1;
    }
    ggml_backend_load_all();
    auto cpu = ggml_backend_init_by_name("CPU", nullptr);
    if (!cpu) return 1;
    const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu));
    using set_ref_fn = void (*)(ggml_backend_t, bool);
    const auto set_ref = reinterpret_cast<set_ref_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_use_ref"));
    if (!set_ref) { ggml_backend_free(cpu); return 1; }
    set_ref(cpu, true);
    auto * dev = reference_only ? nullptr : ggml_backend_dev_by_name("HTP0");
    auto hexagon = dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
    bool ok = !dev || hexagon;
    for (const auto & s : cases) {
        const auto full = make_inputs(s, false);
        const auto narrowed = make_inputs(s, true);
        const auto & expected = s.panel ? full : narrowed;
        std::vector<float> result;
        const bool ref_ok = compute(cpu, s, expected, GGML_TYPE_F32, result);
        ok = ref_ok && ok;
        if (ref_ok) ok = compare("CPU-dequantized-F32", s, result, expected.expected) && ok;
        // Confirm this fixture would catch activation narrowing on every
        // eligible shape, rather than accidentally using F16-exact inputs.
        if (s.panel && full.expected == narrowed.expected) ok = false;
        if (hexagon) {
            const bool ran = compute(hexagon, s, full, GGML_TYPE_F16, result);
            ok = ran && ok;
            if (ran) ok = compare("Hexagon", s, result, expected.expected) && ok;
        }
    }
    ggml_backend_free(hexagon);
    ggml_backend_free(cpu);
    if (!ok) return 1;
    if (!reference_only && !dev) { std::puts("SKIP: Hexagon unavailable; independent F32 reference checks passed"); return 77; }
    return 0;
}
