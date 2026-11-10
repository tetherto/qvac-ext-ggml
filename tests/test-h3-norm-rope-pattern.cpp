#include "ggml.h"
#include "ggml-cuda/h3-norm-rope.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

static void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct pattern_graph {
    ggml_context *context;
    ggml_cgraph *graph;
    ggml_tensor *norms[2]{};
    ggml_tensor *weighted[2]{};
    ggml_tensor *coefficient_views[2][4]{};
    ggml_tensor *coefficient_copies[2][4]{};
    ggml_tensor *outputs[2]{};
    int norm_indices[2]{};

    pattern_graph(int rotation = 96, bool computed_pe = false, bool strided_pe = false, bool identity_layout = false,
                  bool mixed_pe = false, bool per_head_weight = false, int width = 128) {
        constexpr int heads = 3;
        constexpr int tokens = 3;
        constexpr int batch = 2;
        const int half = rotation / 2;
        context = ggml_init({ggml_tensor_overhead() * 2048 + ggml_graph_overhead_custom(2048, false), nullptr, true});
        require(context != nullptr, "metadata context allocation failed");
        graph = ggml_new_graph_custom(context, 2048, false);
        auto *raw = ggml_new_tensor_3d(context, GGML_TYPE_F32, 3 * width * heads, tokens, batch);
        ggml_set_input(raw);
        auto *pe = strided_pe ? ggml_new_tensor_4d(context, GGML_TYPE_F32, 5, 4, half + 3, tokens + 1)
                              : ggml_new_tensor_4d(context, GGML_TYPE_F32, 2, 2, half, tokens);
        ggml_set_input(pe);
        if (strided_pe) {
            pe = ggml_view_4d(context, pe, 2, 2, half, tokens, pe->nb[1], pe->nb[2], pe->nb[3],
                              sizeof(float) + pe->nb[1]);
            pe = ggml_permute(context, pe, 1, 0, 2, 3);
        }
        if (computed_pe) {
            pe = ggml_dup(context, pe);
        }
        auto *other_pe = ggml_new_tensor_4d(context, GGML_TYPE_F32, 2, 2, half, tokens);
        ggml_set_input(other_pe);
        for (int copy = 0; copy < 2; ++copy) {
            auto *input = ggml_view_4d(context, raw, width, heads, tokens, batch, width * sizeof(float), raw->nb[1],
                                       raw->nb[2], copy * width * heads * sizeof(float));
            auto *weight = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, per_head_weight ? heads : 1);
            ggml_set_input(weight);
            norms[copy] = ggml_rms_norm(context, input, 1e-6f);
            auto *normalized = ggml_mul_inplace(context, norms[copy], weight);
            weighted[copy] = normalized;
            auto half_view = [&](int offset) {
                return ggml_view_4d(context, normalized, half, heads, tokens, batch, normalized->nb[1],
                                    normalized->nb[2], normalized->nb[3], offset * sizeof(float));
            };
            auto *a = half_view(0);
            auto *b = half_view(half);
            auto coefficient = [&](int index) {
                auto *source = mixed_pe && index == 1 ? other_pe : pe;
                const size_t offset = (index % 2) * source->nb[0] + (index / 2) * source->nb[1];
                auto *view = ggml_view_4d(context, source, 1, 1, half, tokens, source->nb[1], source->nb[2],
                                          source->nb[3], offset);
                auto *permute = ggml_permute(context, view, 1, 3, 0, 2);
                auto *contiguous = ggml_cont(context, permute);
                coefficient_views[copy][index] = view;
                coefficient_copies[copy][index] = contiguous;
                return contiguous;
            };
            auto *first =
                ggml_add_inplace(context, ggml_mul(context, a, coefficient(0)), ggml_mul(context, b, coefficient(1)));
            auto *second =
                ggml_add_inplace(context, ggml_mul(context, a, coefficient(2)), ggml_mul(context, b, coefficient(3)));
            auto *rotated = ggml_concat(context, first, second, 0);
            if (rotation < width) {
                auto *tail =
                    ggml_view_4d(context, normalized, width - rotation, heads, tokens, batch, normalized->nb[1],
                                 normalized->nb[2], normalized->nb[3], rotation * sizeof(float));
                rotated = ggml_concat(context, rotated, tail, 0);
            }
            auto *permute = identity_layout ? ggml_permute(context, rotated, 0, 1, 2, 3)
                                            : ggml_permute(context, rotated, 0, 2, 1, 3);
            auto *packed = ggml_cont(context, permute);
            outputs[copy] = packed;
            auto *output = ggml_reshape_3d(context, packed, width, tokens, heads * batch);
            ggml_set_output(output);
            ggml_build_forward_expand(graph, output);
        }
        for (int copy = 0; copy < 2; ++copy) {
            norm_indices[copy] = -1;
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                if (ggml_graph_node(graph, i) == norms[copy]) {
                    norm_indices[copy] = i;
                }
            }
            require(norm_indices[copy] >= 0, "normalization missing from reference graph");
        }
    }

    ~pattern_graph() { ggml_free(context); }

    bool matches(int copy) const {
        ggml_cuda_h3::norm_rope_match match;
        if (!ggml_cuda_h3::match_norm_rope(graph, norm_indices[copy], match)) {
            return false;
        }
        require(match.output == outputs[copy], "matched a different output tensor");
        return true;
    }
};

int main() {
    try {
        for (int rotation : {96, 128}) {
            for (bool strided : {false, true}) {
                pattern_graph fixture(rotation, false, strided);
                // A first-use PE view/permutation must execute before the fusion.
                require(fixture.matches(0) == !strided && fixture.matches(1),
                        "prepared PE pattern or first-use dependency guard failed");
            }
        }
        {
            pattern_graph fixture(96, true);
            require(!fixture.matches(0) && fixture.matches(1), "intervening PE computation was not preserved");
        }
        {
            pattern_graph fixture(96, false, false, true);
            require(!fixture.matches(0), "shape-colliding wrong output permutation accepted");
        }
        {
            pattern_graph fixture(96, false, false, false, true);
            require(!fixture.matches(0), "coefficients from different PE tensors accepted");
        }
        {
            pattern_graph fixture(96, false, false, false, false, true);
            require(!fixture.matches(0), "per-head normalization weights accepted");
        }
        {
            pattern_graph fixture(96, false, false, false, false, false, 256);
            require(!fixture.matches(0), "unsupported head dimension accepted");
        }
        {
            pattern_graph fixture;
            ggml_set_output(fixture.norms[0]);
            require(!fixture.matches(0), "observed normalization accepted");
        }
        {
            pattern_graph fixture;
            ggml_set_output(fixture.coefficient_copies[0][0]);
            require(!fixture.matches(0), "observed coefficient copy accepted");
        }
        {
            pattern_graph fixture;
            auto *extra = ggml_scale(fixture.context, fixture.weighted[0], 2.f);
            ggml_set_output(extra);
            ggml_build_forward_expand(fixture.graph, extra);
            require(!fixture.matches(0), "outside normalized-value consumer accepted");
        }
        {
            pattern_graph fixture;
            auto *extra = ggml_scale(fixture.context, fixture.coefficient_copies[0][1], 2.f);
            ggml_set_output(extra);
            ggml_build_forward_expand(fixture.graph, extra);
            require(!fixture.matches(0), "outside coefficient consumer accepted");
        }
        std::puts("PASS H3 norm/RoPE pattern, strided PE, batch2, ownership, observers and near-miss arithmetic/layout "
                  "guards");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
