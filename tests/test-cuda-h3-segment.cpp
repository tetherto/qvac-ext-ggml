#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda/h3-segment-fusion.h"
#include <cuda_runtime_api.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static ggml_tensor *h3_segmented_run(ggml_context *ctx, ggml_tensor *x, ggml_tensor *update, ggml_tensor *projection,
                                     int kind, bool whole, bool views) {
    const int64_t width = x->ne[0], tokens = x->ne[1];
    struct span {
        int64_t start, end, row;
    };
    const std::vector<span> spans =
        whole ? std::vector<span>{{0, tokens, 4}}
              : std::vector<span>{{0, 1, 0}, {1, 3, 1}, {3, 4, 2}, {4, 8, 3}, {8, 12, 4}, {12, tokens, 5}};
    ggml_tensor *output = nullptr;
    for (const auto &span : spans) {
        auto *matrix = ggml_reshape_2d(ctx, projection, width * 6, 6);
        auto *row = ggml_view_2d(ctx, matrix, width * 6, 1, matrix->nb[1], span.row * matrix->nb[1]);
        if (!views)
            row = ggml_cont(ctx, row);
        auto parameter = [&](int index) {
            auto *value = ggml_view_1d(ctx, row, width, index * width * sizeof(float));
            return views ? value : ggml_cont(ctx, value);
        };
        auto slice = [&](ggml_tensor *input) {
            auto *value = ggml_view_3d(ctx, input, width, span.end - span.start, input->ne[2], input->nb[1],
                                       input->nb[2], span.start * input->nb[1]);
            return views ? value : ggml_cont(ctx, value);
        };
        auto *base = slice(x);
        ggml_tensor *part;
        if (kind < 2) {
            auto *shift = parameter(3 * kind);
            auto *scale = parameter(3 * kind + 1);
            part = ggml_add(ctx, ggml_add(ctx, base, ggml_mul(ctx, base, scale)), shift);
        } else {
            auto *gate = parameter(kind == 2 ? 2 : 5);
            part = ggml_add(ctx, base, ggml_mul(ctx, slice(update), gate));
        }
        output = output ? ggml_concat(ctx, output, part, 1) : part;
    }
    return output;
}

static void require(bool condition, const char *text) {
    if (!condition)
        throw std::runtime_error(text);
}

static void capture(ggml_log_level, const char *text, void *data) {
    if (std::strstr(text, "h3_segment_fusion "))
        ++*static_cast<std::atomic<int> *>(data);
    std::fputs(text, stderr);
}

static float value(size_t index, int seed, float scale) {
    return scale * (float((index * 17 + seed * 29) % 251) - 125.f) / 128.f;
}

static void fill(ggml_tensor *raw, ggml_tensor *view, std::vector<float> &values, int seed, float scale) {
    values.assign(ggml_nelements(raw), 13.75f);
    size_t logical = 0;
    for (int64_t b = 0; b < view->ne[2]; ++b)
        for (int64_t t = 0; t < view->ne[1]; ++t)
            for (int64_t c = 0; c < view->ne[0]; ++c) {
                size_t offset = view->view_offs + b * view->nb[2] + t * view->nb[1] + c * sizeof(float);
                require(offset / sizeof(float) < values.size(), "input view outside backing");
                values[offset / sizeof(float)] = value(logical++, seed, scale);
            }
}

static std::vector<unsigned char> download(ggml_tensor *tensor) {
    std::vector<unsigned char> result(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, result.data(), 0, result.size());
    return result;
}

static void run_case(ggml_backend_t backend, std::atomic<int> &launches, bool enabled, int width, int tokens, int batch,
                     int layout, bool whole, bool observed, int alias = 0, int selected_kind = -1,
                     float input_scale = 1.f) {
    auto *ctx = ggml_init({ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false), nullptr, true});
    require(ctx, "context allocation failed");
    auto *graph = ggml_new_graph_custom(ctx, 4096, false);
    auto *raw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width + (layout ? 16 : 0), tokens + (layout ? 1 : 0), batch);
    auto *updates = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, raw->ne[0], raw->ne[1], batch);
    auto *projection = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width * 6 * 3, 2);
    for (auto *tensor : {raw, updates, projection}) {
        ggml_set_input(tensor);
        ggml_set_output(tensor);
        ggml_build_forward_expand(graph, tensor);
    }
    const size_t offset = layout == 1 ? 16 : layout == 2 ? sizeof(float) : 0;
    auto *x = ggml_view_3d(ctx, raw, width, tokens, batch, raw->nb[1], raw->nb[2], offset);
    auto *u = ggml_view_3d(ctx, updates, width, tokens, batch, updates->nb[1], updates->nb[2], offset);
    std::vector<ggml_tensor *> reference, candidate;
    for (int kind = 0; kind < 4; ++kind)
        if (selected_kind < 0 || selected_kind == kind) {
            auto *out = h3_segmented_run(ctx, x, u, projection, kind, whole, false);
            reference.push_back(out);
            ggml_set_output(out);
            ggml_build_forward_expand(graph, out);
        }
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto *node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_MUL)
            ggml_set_output(node);
    }
    const int candidate_start = ggml_graph_n_nodes(graph);
    for (int kind = 0; kind < 4; ++kind)
        if (selected_kind < 0 || selected_kind == kind) {
            auto *out = h3_segmented_run(ctx, x, u, projection, kind, whole, true);
            candidate.push_back(out);
            ggml_set_output(out);
            ggml_build_forward_expand(graph, out);
        }
    if (observed)
        for (int i = candidate_start; i < ggml_graph_n_nodes(graph); ++i) {
            auto *node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_MUL)
                ggml_set_output(node);
        }
    int matches = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_cuda_h3::segment_fusion_match match;
        matches += ggml_cuda_h3::match_segment_fusion(graph, i, match);
    }
    const int expected_patterns = observed || width % 4 ? 0 : int(candidate.size()) * (whole ? 1 : 6);
    require(matches == expected_patterns, "unexpected metadata pattern count");
    auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    require(ggml_gallocr_alloc_graph(allocator, graph), "graph allocation failed");
    if (alias) {
        require(whole && layout == 0 && candidate.size() == 1, "invalid alias fixture");
        auto *target = alias == 1 ? x : u;
        candidate[0]->data = target->data;
        candidate[0]->buffer = target->buffer;
    }
    const int before = launches.load();
    std::vector<float> base_data, update_data, projection_data(ggml_nelements(projection));
    std::vector<std::vector<unsigned char>> first;
    for (int pass = 0; pass < 3; ++pass) {
        const int seed = pass == 1 ? 41 : 19;
        fill(raw, x, base_data, seed, input_scale);
        fill(updates, u, update_data, seed + 1, input_scale);
        for (size_t i = 0; i < projection_data.size(); ++i)
            projection_data[i] = value(i, seed + 2, 0.75f);
        if (input_scale == 1.f && width >= 4) {
            const float a = std::ldexp(1.f, -13);
            for (int b = 0; b < batch; ++b)
                for (int t = 0; t < tokens; ++t) {
                    const size_t bi = (x->view_offs + b * x->nb[2] + t * x->nb[1]) / sizeof(float);
                    base_data[bi] = -1.f;
                    update_data[bi] = 1.f + a;
                    base_data[bi + 1] = 1.f + a;
                }
            for (int row = 0; row < 6; ++row)
                for (int slot = 0; slot < 6; ++slot) {
                    projection_data[(row * 6 + slot) * width] = 1.f - a;
                    projection_data[(row * 6 + slot) * width + 1] = (slot == 1 || slot == 4) ? -1.f + a : 0.f;
                }
        }
        ggml_backend_tensor_set(raw, base_data.data(), 0, base_data.size() * sizeof(float));
        ggml_backend_tensor_set(updates, update_data.data(), 0, update_data.size() * sizeof(float));
        ggml_backend_tensor_set(projection, projection_data.data(), 0, projection_data.size() * sizeof(float));
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph execution failed");
        ggml_backend_synchronize(backend);
        for (size_t i = 0; i < candidate.size(); ++i) {
            const auto expected = download(reference[i]);
            const auto actual = download(candidate[i]);
            require(expected == actual, "reference/candidate byte mismatch");
            if (pass == 0)
                first.push_back(actual);
            if (pass == 2)
                require(first[i] == actual, "A/B/A output mismatch");
            if (alias) {
                auto *backing = alias == 1 ? raw : updates;
                require(download(backing) == expected, "pointwise alias side effect mismatch");
            }
        }
        if (alias != 1) {
            auto data = download(raw);
            require(data.size() == base_data.size() * sizeof(float) &&
                        std::memcmp(data.data(), base_data.data(), data.size()) == 0,
                    "base input/padding changed");
        }
        if (alias != 2) {
            auto data = download(updates);
            require(data.size() == update_data.size() * sizeof(float) &&
                        std::memcmp(data.data(), update_data.data(), data.size()) == 0,
                    "update input/padding changed");
        }
        auto data = download(projection);
        require(std::memcmp(data.data(), projection_data.data(), data.size()) == 0, "modulation projection changed");
    }
    require(launches.load() - before == (enabled && layout != 2 ? expected_patterns * 3 : 0),
            "actual fusion count mismatch");
    std::printf("PASS width=%d tokens=%d batch=%d layout=%d whole=%d observed=%d alias=%d kind=%d scale=%.9g "
                "fusions=%d A/B/A exact\n",
                width, tokens, batch, layout, whole, observed, alias, selected_kind, input_scale,
                launches.load() - before);
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
        require(backend, "CUDA0 unavailable");
        std::atomic<int> launches{0};
        ggml_log_set(capture, &launches);
        const char *flag = std::getenv("GGML_CUDA_H3_SEGMENT_FUSION");
        const char *disabled = std::getenv("GGML_CUDA_DISABLE_FUSION");
        const char *graph_opt = std::getenv("GGML_CUDA_GRAPH_OPT");
        const bool enabled = (!flag || std::string(flag) == "1") && !(disabled && std::atoi(disabled)) &&
                             !(graph_opt && std::atoi(graph_opt) == 1);
        bool full = false, expect = false;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--full-shape")
                full = true;
            else if (std::string(argv[i]) == "--expect-fusion")
                expect = true;
            else
                throw std::runtime_error("unknown argument");
        }
        require(enabled == expect, "fusion environment does not match requested test mode");
        if (full) {
            run_case(backend, launches, enabled, 5376, 15429, 1, 0, false, false, 0, 0);
        } else {
            run_case(backend, launches, enabled, 4, 1, 1, 0, true, false);
            run_case(backend, launches, enabled, 132, 17, 3, 1, true, false);
            run_case(backend, launches, enabled, 65536, 1, 1, 0, true, false);
            run_case(backend, launches, enabled, 4, 65535, 1, 0, true, false);
            run_case(backend, launches, enabled, 32, 13, 2, 1, false, false, 0, -1, 1e-40f);
            run_case(backend, launches, enabled, 32, 13, 2, 1, false, false, 0, -1, 1e30f);
            run_case(backend, launches, enabled, 32, 13, 2, 1, false, false, 0, -1, -0.0f);
            run_case(backend, launches, enabled, 4, 13, 2, 0, false, false);
            for (int width : {32, 128, 5376})
                for (int layout : {0, 1})
                    run_case(backend, launches, enabled, width, 13, 2, layout, false, false);
            run_case(backend, launches, enabled, 128, 13, 2, 2, false, false);
            run_case(backend, launches, enabled, 33, 13, 2, 0, false, false);
            run_case(backend, launches, enabled, 128, 1, 1, 0, true, false);
            run_case(backend, launches, enabled, 128, 5, 2, 0, true, true);
            run_case(backend, launches, enabled, 128, 5, 2, 0, true, false, 1, 0);
            run_case(backend, launches, enabled, 128, 5, 2, 0, true, false, 1, 2);
            run_case(backend, launches, enabled, 128, 5, 2, 0, true, false, 2, 2);
            run_case(backend, launches, enabled, 32, 13, 2, 1, false, false, 0, -1, 1e18f);
            run_case(backend, launches, enabled, 32, 13, 2, 1, false, false, 0, -1, 1e-30f);
        }
        ggml_log_set(nullptr, nullptr);
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception &error) {
        ggml_log_set(nullptr, nullptr);
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
