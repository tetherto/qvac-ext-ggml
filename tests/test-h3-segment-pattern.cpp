#include "ggml-cuda/h3-segment-fusion.h"

#include <cstdio>
#include <stdexcept>

static void require(bool value, const char *message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

struct segment_graph {
    ggml_context *context;
    ggml_cgraph *graph;
    ggml_tensor *product;
    ggml_tensor *sum;
    ggml_tensor *shift;
    ggml_tensor *output;
    int first = -1;

    segment_graph(bool modulation, bool metadata = true, bool padded = false, int width = 128,
                  bool computed_shift = false) {
        context = ggml_init({ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(128, false), nullptr, true});
        require(context != nullptr, "context allocation failed");
        graph = ggml_new_graph_custom(context, 128, false);
        auto *raw = ggml_new_tensor_3d(context, GGML_TYPE_F32, width + (padded ? 4 : 0), 8, 2);
        auto *x = ggml_view_3d(context, raw, width, 7, 2, raw->nb[1], raw->nb[2], padded ? 16 : 0);
        auto *base = modulation ? x : ggml_new_tensor_3d(context, GGML_TYPE_F32, width, 7, 2);
        auto *scale = ggml_new_tensor_1d(context, GGML_TYPE_F32, width);
        auto *projection = ggml_new_tensor_1d(context, GGML_TYPE_F32, width * 6);
        product = ggml_mul(context, x, scale);
        sum = ggml_add(context, base, product);
        shift = metadata ? ggml_view_1d(context, projection, width, width * sizeof(float)) : scale;
        if (computed_shift) {
            shift = ggml_dup(context, shift);
        }
        output = modulation ? ggml_add(context, sum, shift) : sum;
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            if (ggml_graph_node(graph, i) == product) {
                first = i;
            }
        }
        require(first >= 0, "product not in graph");
    }

    ~segment_graph() { ggml_free(context); }

    bool matches() const {
        ggml_cuda_h3::segment_fusion_match match;
        const bool found = ggml_cuda_h3::match_segment_fusion(graph, first, match);
        if (found) {
            require(match.output == output, "wrong output selected");
        }
        return found;
    }
};

static void memory_guards(bool modulation) {
    segment_graph fixture(modulation);
    ggml_cuda_h3::segment_fusion_match match;
    require(ggml_cuda_h3::match_segment_fusion(fixture.graph, fixture.first, match), "missing memory fixture");
    ggml_tensor base = *match.base, update = *match.update, scale = *match.scale, output = *match.output;
    ggml_tensor shift = match.shift ? *match.shift : scale;
    base.data = reinterpret_cast<void *>(uintptr_t(0x10000000));
    update.data = reinterpret_cast<void *>(uintptr_t(0x20000000));
    scale.data = reinterpret_cast<void *>(uintptr_t(0x30000000));
    shift.data = reinterpret_cast<void *>(uintptr_t(0x40000000));
    output.data = reinterpret_cast<void *>(uintptr_t(0x50000000));
    match.base = &base;
    match.update = &update;
    match.scale = &scale;
    match.shift = modulation ? &shift : nullptr;
    match.output = &output;
    require(ggml_cuda_h3::segment_memory_safe(match), "disjoint operands rejected");
    base = output;
    require(ggml_cuda_h3::segment_memory_safe(match), "exact pointwise alias rejected");
    base.data = reinterpret_cast<void *>(uintptr_t(output.data) + 16);
    require(!ggml_cuda_h3::segment_memory_safe(match), "shifted alias accepted");
    base.data = output.data;
    base.nb[2] += 16;
    require(!ggml_cuda_h3::segment_memory_safe(match), "different-stride alias accepted");
    base.data = reinterpret_cast<void *>(uintptr_t(0x10000000));
    scale.data = output.data;
    require(!ggml_cuda_h3::segment_memory_safe(match), "broadcast scale alias accepted");
    scale.data = reinterpret_cast<void *>(uintptr_t(0x30000000));
    shift.data = output.data;
    if (modulation) {
        require(!ggml_cuda_h3::segment_memory_safe(match), "broadcast shift alias accepted");
    }
}

int main() {
    try {
        for (bool modulation : {false, true}) {
            for (bool metadata : {false, true}) {
                for (bool padded : {false, true}) {
                    segment_graph fixture(modulation, metadata, padded);
                    require(fixture.matches(), "supported pattern rejected");
                }
            }
            {
                segment_graph fixture(modulation);
                ggml_set_output(fixture.product);
                require(!fixture.matches(), "observed product accepted");
            }
            {
                segment_graph fixture(modulation);
                auto *other = ggml_scale(fixture.context, fixture.product, 2.f);
                ggml_set_output(other);
                ggml_build_forward_expand(fixture.graph, other);
                require(!fixture.matches(), "outside product consumer accepted");
            }
            {
                segment_graph fixture(modulation);
                auto *a = fixture.sum->src[0];
                fixture.sum->src[0] = fixture.product;
                fixture.sum->src[1] = a;
                require(!fixture.matches(), "reordered operands accepted");
            }
            {
                segment_graph fixture(modulation);
                fixture.product->src[1]->ne[1] = 7;
                require(!fixture.matches(), "per-token scale accepted");
            }
            {
                segment_graph fixture(modulation);
                fixture.product->view_src = fixture.product->src[0];
                require(!fixture.matches(), "in-place product accepted");
            }
            {
                segment_graph fixture(modulation);
                fixture.product->src[0]->nb[0] = 8;
                require(!fixture.matches(), "non-unit channel stride accepted");
            }
            {
                segment_graph fixture(modulation);
                fixture.product->src[0]->ne[3] = 2;
                require(!fixture.matches(), "unsupported fourth dimension accepted");
            }
            {
                segment_graph fixture(modulation, true, false, 33);
                require(!fixture.matches(), "unsupported vector width accepted");
            }
            memory_guards(modulation);
        }
        {
            segment_graph fixture(true);
            ggml_set_output(fixture.sum);
            require(!fixture.matches(), "observed inner sum accepted");
        }
        {
            segment_graph fixture(true);
            ggml_set_output(fixture.shift);
            require(!fixture.matches(), "observed skipped shift view accepted");
        }
        {
            segment_graph fixture(true, true, false, 128, true);
            require(!fixture.matches(), "unprepared shift computation accepted");
        }
        std::puts("PASS H3 segment patterns, observers, consumer ownership, layouts and alias guards");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
