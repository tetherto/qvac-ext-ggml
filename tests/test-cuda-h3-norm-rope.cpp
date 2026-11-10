#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda/h3-norm-rope.h"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static ggml_tensor *partial_rope(ggml_context *ctx, ggml_tensor *x, ggml_tensor *pe) {
    const int half = (int)pe->ne[2];
    auto half_view = [&](int start) {
        return ggml_view_4d(ctx, x, half, x->ne[1], x->ne[2], x->ne[3], x->nb[1], x->nb[2], x->nb[3],
                            start * sizeof(float));
    };
    auto coefficient = [&](int index) {
        auto *view = ggml_view_4d(ctx, pe, 1, 1, half, pe->ne[3], pe->nb[1], pe->nb[2], pe->nb[3],
                                  (index % 2) * pe->nb[0] + (index / 2) * pe->nb[1]);
        return ggml_cont(ctx, ggml_permute(ctx, view, 1, 3, 0, 2));
    };
    auto *a = half_view(0);
    auto *b = half_view(half);
    auto *first = ggml_add_inplace(ctx, ggml_mul(ctx, a, coefficient(0)), ggml_mul(ctx, b, coefficient(1)));
    auto *second = ggml_add_inplace(ctx, ggml_mul(ctx, a, coefficient(2)), ggml_mul(ctx, b, coefficient(3)));
    auto *rotated = ggml_concat(ctx, first, second, 0);
    if (2 * half < 128) {
        auto *tail = ggml_view_4d(ctx, x, 128 - 2 * half, x->ne[1], x->ne[2], x->ne[3], x->nb[1], x->nb[2], x->nb[3],
                                  2 * half * sizeof(float));
        rotated = ggml_concat(ctx, rotated, tail, 0);
    }
    auto *packed = ggml_cont(ctx, ggml_permute(ctx, rotated, 0, 2, 1, 3));
    return ggml_reshape_3d(ctx, packed, packed->ne[0], packed->ne[1], packed->ne[2] * packed->ne[3]);
}

static void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static std::atomic<int> replayed_fusions{0};
static int fusions_per_replay = 0;

static void log_fusions(ggml_log_level, const char *text, void *user_data) {
    if (std::strstr(text, "h3_norm_rope_fusion ")) {
        ++*static_cast<std::atomic<int> *>(user_data);
    }
    if (std::strstr(text, "cuda_graph_execute ") && std::strstr(text, "mode=replay")) {
        replayed_fusions += fusions_per_replay;
    }
    std::fputs(text, stderr);
}

static float value(size_t index, int seed, float scale) {
    return scale * (float((index * 17 + seed * 29) % 251) - 125.f) / 128.f;
}

static void fill_view(ggml_tensor *base, ggml_tensor *view, std::vector<float> &data, int seed, float scale) {
    data.assign(ggml_nelements(base), 13.75f);
    size_t logical = 0;
    for (int64_t b = 0; b < view->ne[3]; ++b)
        for (int64_t t = 0; t < view->ne[2]; ++t)
            for (int64_t h = 0; h < view->ne[1]; ++h)
                for (int64_t c = 0; c < view->ne[0]; ++c) {
                    const size_t offset =
                        view->view_offs + c * view->nb[0] + h * view->nb[1] + t * view->nb[2] + b * view->nb[3];
                    require(offset % sizeof(float) == 0 && offset / sizeof(float) < data.size(),
                            "view exceeds its backing allocation");
                    data[offset / sizeof(float)] = value(logical++, seed, scale);
                }
}

static std::vector<unsigned char> bytes(ggml_tensor *tensor) {
    std::vector<unsigned char> result(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, result.data(), 0, result.size());
    return result;
}

static void run_case(ggml_backend_t backend, int rotation, int heads, int tokens, int batch, int layout, bool arbitrary,
                     float input_scale, bool observed) {
    fusions_per_replay = observed ? 0 : 1;
    auto *ctx = ggml_init({ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false), nullptr, true});
    auto *graph = ggml_new_graph_custom(ctx, 4096, false);
    auto *raw =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3 * 128 * heads + (layout ? 16 : 0), tokens + (layout ? 1 : 0), batch);
    ggml_set_input(raw);
    ggml_set_output(raw);
    auto *input = ggml_view_4d(ctx, raw, 128, heads, tokens, batch, 128 * sizeof(float), raw->nb[1], raw->nb[2],
                               layout ? 8 * sizeof(float) : 0);
    auto *pe_base = layout ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 5, 4, rotation / 2 + 3, tokens + 1)
                           : ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, rotation / 2, tokens);
    ggml_set_input(pe_base);
    ggml_set_output(pe_base);
    auto *pe = layout ? ggml_view_4d(ctx, pe_base, 2, 2, rotation / 2, tokens, pe_base->nb[1], pe_base->nb[2],
                                     pe_base->nb[3], sizeof(float) + pe_base->nb[1])
                      : pe_base;
    if (layout == 2) {
        pe = ggml_permute(ctx, pe, 1, 0, 2, 3);
    }
    auto *weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    ggml_set_input(weight);
    ggml_set_output(weight);
    ggml_tensor *outputs[2];
    for (int mode = 0; mode < 2; ++mode) {
        auto *normalized = ggml_rms_norm(ctx, input, 1e-6f);
        if (mode == 0 || observed) {
            ggml_set_output(normalized);
        }
        auto *weighted = ggml_mul_inplace(ctx, normalized, weight);
        outputs[mode] = partial_rope(ctx, weighted, pe);
        ggml_set_output(outputs[mode]);
        ggml_build_forward_expand(graph, outputs[mode]);
    }
    int matches = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_cuda_h3::norm_rope_match match;
        matches += ggml_cuda_h3::match_norm_rope(graph, i, match);
    }
    require(matches == (observed ? 0 : 1), "unexpected positive fusion-pattern count");
    auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    require(ggml_gallocr_alloc_graph(allocator, graph), "graph allocation failed");
    std::vector<float> input_data, pe_data, weight_data(128);
    std::vector<unsigned char> first_output;
    for (int pass = 0; pass < 3; ++pass) {
        const int seed = pass == 1 ? 41 : 19;
        fill_view(raw, input, input_data, seed, input_scale);
        fill_view(pe_base, pe, pe_data, seed + 1, 0.75f);
        if (!arbitrary) {
            for (int t = 0; t < tokens; ++t)
                for (int c = 0; c < rotation / 2; ++c) {
                    const float angle = (t * 0.03125f + c * 0.015625f);
                    const float coefficients[] = {std::cos(angle), -std::sin(angle), std::sin(angle), std::cos(angle)};
                    for (int j = 0; j < 4; ++j) {
                        const size_t offset =
                            pe->view_offs + (j % 2) * pe->nb[0] + (j / 2) * pe->nb[1] + c * pe->nb[2] + t * pe->nb[3];
                        pe_data[offset / sizeof(float)] = coefficients[j];
                    }
                }
        }
        for (int i = 0; i < 128; ++i)
            weight_data[i] = value(i, seed + 2, 0.75f);
        ggml_backend_tensor_set(raw, input_data.data(), 0, input_data.size() * sizeof(float));
        ggml_backend_tensor_set(pe_base, pe_data.data(), 0, pe_data.size() * sizeof(float));
        ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph computation failed");
        ggml_backend_synchronize(backend);
        const auto reference = bytes(outputs[0]);
        const auto candidate = bytes(outputs[1]);
        if (reference != candidate) {
            size_t different = 0;
            float maximum = 0.f;
            for (size_t i = 0; i < reference.size(); i += sizeof(float)) {
                float a, b;
                std::memcpy(&a, reference.data() + i, sizeof(float));
                std::memcpy(&b, candidate.data() + i, sizeof(float));
                different += std::memcmp(reference.data() + i, candidate.data() + i, sizeof(float)) != 0;
                maximum = std::max(maximum, std::abs(a - b));
            }
            std::fprintf(stderr,
                         "MISMATCH values=%zu max_absolute=%.9g rotation=%d heads=%d tokens=%d batch=%d layout=%d "
                         "arbitrary=%d scale=%.9g observed=%d pass=%d\n",
                         different, maximum, rotation, heads, tokens, batch, layout, arbitrary, input_scale, observed,
                         pass);
            throw std::runtime_error("byte equality failed");
        }
        const auto raw_after = bytes(raw);
        const auto pe_after = bytes(pe_base);
        const auto weight_after = bytes(weight);
        require(raw_after.size() == input_data.size() * sizeof(float) &&
                    std::memcmp(raw_after.data(), input_data.data(), raw_after.size()) == 0,
                "input backing storage mutated");
        require(std::memcmp(pe_after.data(), pe_data.data(), pe_after.size()) == 0, "PE backing storage mutated");
        require(std::memcmp(weight_after.data(), weight_data.data(), weight_after.size()) == 0, "norm weights mutated");
        if (pass == 0)
            first_output = candidate;
        if (pass == 2)
            require(first_output == candidate, "A/B/A output changed");
    }
    std::printf("PASS rotation=%d heads=%d tokens=%d batch=%d layout=%d arbitrary=%d scale=%.9g observed=%d A/B/A "
                "byte-identical\n",
                rotation, heads, tokens, batch, layout, arbitrary, input_scale, observed);
    std::fflush(stdout);
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
}

int main(int argc, char **argv) {
    try {
        const char *visible = std::getenv("CUDA_VISIBLE_DEVICES");
        if (!visible || std::string(visible) != "0") {
            std::puts("SKIP: explicitly select physical GPU0 with CUDA_VISIBLE_DEVICES=0");
            return 77;
        }
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess || properties.major != 12 || properties.minor != 0) {
            std::puts("SKIP: SM120 CUDA device required");
            return 77;
        }
        ggml_backend_load_all();
        auto backend = ggml_backend_init_by_name("CUDA0", nullptr);
        require(backend != nullptr, "CUDA0 unavailable");
        bool full = false;
        bool expect_fusion = false;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--full-shape") {
                full = true;
            } else if (std::string(argv[i]) == "--expect-fusion") {
                expect_fusion = true;
            } else {
                throw std::runtime_error("unknown argument");
            }
        }
        std::atomic<int> fusion_launches{0};
        ggml_log_set(log_fusions, &fusion_launches);
        if (full) {
            run_case(backend, 96, 56, 15429, 1, 0, false, 0.5f, false);
        } else {
            for (int rotation : {96, 128})
                for (int layout : {0, 1, 2})
                    for (bool arbitrary : {false, true})
                        run_case(backend, rotation, 3, 17, 2, layout, arbitrary, 0.5f, false);
            run_case(backend, 96, 56, 17, 1, 0, false, 0.5f, false);
            run_case(backend, 96, 3, 1, 1, 0, true, 1e-20f, false);
            run_case(backend, 96, 3, 7, 1, 0, true, 1e18f, false);
            run_case(backend, 96, 3, 17, 2, 1, true, 0.5f, true);
        }
        require((expect_fusion ? fusion_launches.load() + replayed_fusions.load() : fusion_launches.load()) ==
                    (expect_fusion ? (full ? 3 : 45) : 0),
                "unexpected actual fusion launch count; enable fusion and tracing for --expect-fusion");
        std::printf("NORM_ROPE host_dispatches=%d replayed_fusions=%d\n", fusion_launches.load(), replayed_fusions.load());
        ggml_log_set(nullptr, nullptr);
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
