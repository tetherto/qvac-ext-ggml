#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static constexpr size_t context_bytes = 4*1024*1024;
static constexpr float tolerance = 1e-6f;

static std::vector<float> pattern(size_t count, unsigned seed) {
    std::vector<float> values(count);
    for (float & value : values) {
        seed = seed*1664525u + 1013904223u;
        value = float(seed >> 8)/8388608.0f - 1.0f;
    }
    return values;
}

static ggml_tensor * reference(ggml_context * ctx, ggml_tensor * x, ggml_tensor * pe, bool split_half) {
    if (!split_half || !pe) {
        return ggml_rope_flux(ctx, x, pe);
    }
    auto input = ggml_cont(ctx, x);
    input = ggml_reshape_4d(ctx, input, x->ne[0]/2, 2, x->ne[1], x->ne[2]*x->ne[3]);
    input = ggml_cont(ctx, ggml_permute(ctx, input, 1, 0, 2, 3));
    input = ggml_reshape_4d(ctx, input, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
    auto output = ggml_rope_flux(ctx, input, pe);
    output = ggml_reshape_4d(ctx, output, 2, x->ne[0]/2, x->ne[2], x->ne[1]*x->ne[3]);
    output = ggml_cont(ctx, ggml_permute(ctx, output, 1, 0, 2, 3));
    return ggml_reshape_3d(ctx, output, x->ne[0], x->ne[2], x->ne[1]*x->ne[3]);
}

static bool compare(ggml_backend_t backend, ggml_tensor * got, ggml_tensor * expected, bool exact) {
    if (!ggml_are_same_shape(got, expected)) {
        return false;
    }
    std::vector<float> actual(ggml_nelements(got)), wanted(actual.size());
    ggml_backend_tensor_get(got, actual.data(), 0, actual.size()*sizeof(float));
    ggml_backend_tensor_get(expected, wanted.data(), 0, wanted.size()*sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(wanted[i]) ||
            (exact ? std::memcmp(&actual[i], &wanted[i], sizeof(float)) != 0 :
             std::fabs(actual[i] - wanted[i]) > tolerance*(1.0f + std::fabs(wanted[i])))) {
            std::fprintf(stderr, "%s: mismatch at %zu: %.9g vs %.9g\n",
                         ggml_backend_name(backend), i, actual[i], wanted[i]);
            return false;
        }
    }
    return true;
}

static bool run_case(ggml_backend_t backend, int dim, int heads, int tokens, int batch,
                     bool split_half, bool with_pe, bool strided, bool & supported) {
    ggml_init_params params = { context_bytes, nullptr, true };
    auto ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }
    auto storage = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, heads, tokens*(strided ? 2 : 1) + 2, batch);
    auto x = ggml_view_4d(ctx, storage, dim, heads, tokens, batch, storage->nb[1],
                          storage->nb[2]*(strided ? 2 : 1), storage->nb[3], storage->nb[2]);
    ggml_tensor * pe_storage = nullptr;
    ggml_tensor * pe = nullptr;
    if (with_pe) {
        pe_storage = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, dim/2, tokens*(strided ? 2 : 1) + 2);
        pe = ggml_view_4d(ctx, pe_storage, 2, 2, dim/2, tokens, pe_storage->nb[1], pe_storage->nb[2],
                          pe_storage->nb[3]*(strided ? 2 : 1), pe_storage->nb[3]);
    }
    auto candidate = ggml_rope_flux_ext(ctx, x, pe, split_half);
    auto expected = reference(ctx, x, pe, split_half);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, expected);
    ggml_build_forward_expand(graph, candidate);
    supported = true;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (!ggml_backend_supports_op(backend, ggml_graph_node(graph, i))) {
            supported = false;
        }
    }
    if (!supported) {
        ggml_free(ctx);
        return true;
    }
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }
    auto values = pattern(ggml_nelements(storage), 42);
    ggml_backend_tensor_set(storage, values.data(), 0, values.size()*sizeof(float));
    if (pe_storage) {
        values = pattern(ggml_nelements(pe_storage), 17);
        ggml_backend_tensor_set(pe_storage, values.data(), 0, values.size()*sizeof(float));
    }
    const bool computed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    const bool passed = computed && compare(backend, candidate, expected, !split_half || !with_pe);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return passed;
}

static bool run_backend(ggml_backend_t backend) {
    int passed = 0, skipped = 0, failed = 0;
    for (int dim : { 2, 64, 128 }) {
        for (int heads : { 1, 4 }) {
            for (int tokens : { 1, 7, 33 }) {
                for (int batch : { 1, 2 }) {
                    for (bool split_half : { false, true }) {
                        for (bool with_pe : { false, true }) {
                            for (bool strided : { false, true }) {
                                bool supported = false;
                                if (!run_case(backend, dim, heads, tokens, batch,
                                              split_half, with_pe, strided, supported)) {
                                    ++failed;
                                } else if (supported) {
                                    ++passed;
                                } else {
                                    ++skipped;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    const bool cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
    std::printf("%s: passed=%d skipped=%d failed=%d\n", ggml_backend_name(backend), passed, skipped, failed);
    return failed == 0 && (!cpu || (passed > 0 && skipped == 0));
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    auto backend = ggml_backend_init_by_name(argc > 1 ? argv[1] : "CPU", nullptr);
    if (!backend) {
        std::fprintf(stderr, "Requested backend is unavailable\n");
        return 1;
    }
    const bool passed = run_backend(backend);
    ggml_backend_free(backend);
    return passed ? 0 : 1;
}
