#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <numeric>
#include <vector>

namespace {

constexpr size_t   kContextBytes   = 16u*1024*1024;
constexpr uint32_t kPatternPrime   = 2654435761u;
constexpr uint32_t kPatternRange   = 2001;
constexpr float    kPatternScale   = 1000.0f;
constexpr int      kGroups         = 32;
constexpr float    kNormEps        = 1e-6f;
constexpr double   kNormTolerance  = 1e-3;
constexpr int64_t  kHeadDim        = 64;
constexpr int64_t  kHeads          = 2;
constexpr int64_t  kKeys           = 256;
constexpr int64_t  kShortQueries   = 1024;
constexpr int64_t  kLongQueries    = 4100;
constexpr double   kAttnTolerance  = 5e-3;

enum class outcome { passed, failed, skipped };

struct norm_shape {
    int64_t width;
    int64_t height;
    int64_t channels;
    int64_t batch;

    int64_t plane() const { return width*height; }
    int64_t item() const { return plane()*channels; }
    int64_t elements() const { return item()*batch; }
};

struct input_tensor {
    ggml_tensor * tensor;
    std::vector<float> values;
};

struct graph_case {
    ggml_tensor * output;
    std::vector<input_tensor> inputs;
};

float pattern_value(size_t index, uint32_t seed) {
    const uint32_t mixed = (uint32_t(index) + seed)*kPatternPrime;
    return float(int32_t(mixed % kPatternRange) - int32_t(kPatternRange/2)) / kPatternScale;
}

std::vector<float> make_pattern(size_t count, uint32_t seed) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = pattern_value(i, seed);
    }
    return values;
}

std::vector<float> round_to_f16(const std::vector<float> & values) {
    std::vector<float> rounded(values.size());
    std::transform(values.begin(), values.end(), rounded.begin(),
                   [](float v) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(v)); });
    return rounded;
}

void upload(const input_tensor & input) {
    if (input.tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(input.values.size());
        ggml_fp32_to_fp16_row(input.values.data(), half.data(), int64_t(half.size()));
        ggml_backend_tensor_set(input.tensor, half.data(), 0, ggml_nbytes(input.tensor));
        return;
    }
    ggml_backend_tensor_set(input.tensor, input.values.data(), 0, ggml_nbytes(input.tensor));
}

bool graph_supported(ggml_backend_t backend, ggml_cgraph * graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (!ggml_backend_supports_op(backend, ggml_graph_node(graph, i))) {
            return false;
        }
    }
    return true;
}

outcome run_case(ggml_backend_t backend,
                 const std::function<graph_case(ggml_context *)> & build,
                 std::vector<float> & output) {
    ggml_init_params params = { kContextBytes, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    graph_case built = build(ctx);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, built.output);
    if (!graph_supported(backend, graph)) {
        ggml_free(ctx);
        return outcome::skipped;
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    std::for_each(built.inputs.begin(), built.inputs.end(), upload);
    const bool computed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    output.resize(ggml_nelements(built.output));
    if (computed) {
        ggml_backend_tensor_get(built.output, output.data(), 0, ggml_nbytes(built.output));
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return computed ? outcome::passed : outcome::failed;
}

double max_abs_error(const std::vector<float> & got, const std::vector<float> & expected) {
    double worst = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double error = std::fabs(double(got[i]) - double(expected[i]));
        worst = std::isfinite(error) ? std::max(worst, error) : INFINITY;
    }
    return worst;
}

struct moments {
    double mean;
    double inv_std;
};

moments block_moments(const float * src, int64_t count) {
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int64_t i = 0; i < count; ++i) {
        sum += src[i];
        sum_sq += double(src[i])*src[i];
    }
    const double mean = sum / count;
    return { mean, 1.0 / std::sqrt(sum_sq / count - mean*mean + kNormEps) };
}

void normalize_block(const float * src, float * dst, int64_t count) {
    const moments m = block_moments(src, count);
    std::transform(src, src + count, dst, [&](float v) { return float((v - m.mean)*m.inv_std); });
}

std::vector<float> reference_group_norm(const std::vector<float> & input, const norm_shape & shape) {
    std::vector<float> output(input.size());
    const int64_t block = shape.plane()*(shape.channels/kGroups);
    for (int64_t start = 0; start < shape.elements(); start += block) {
        normalize_block(input.data() + start, output.data() + start, block);
    }
    return output;
}

void apply_channel_affine(std::vector<float> & values, const std::vector<float> & scale,
                          const std::vector<float> & shift, const norm_shape & shape) {
    for (int64_t i = 0; i < shape.elements(); ++i) {
        const int64_t channel = (i / shape.plane()) % shape.channels;
        values[i] = values[i]*scale[channel] + shift[channel];
    }
}

ggml_tensor * new_norm_input(ggml_context * ctx, const norm_shape & shape) {
    return ggml_new_tensor_4d(ctx, GGML_TYPE_F32, shape.width, shape.height, shape.channels, shape.batch);
}

outcome test_group_norm(ggml_backend_t backend, const norm_shape & shape) {
    const std::vector<float> input = make_pattern(shape.elements(), 1);
    std::vector<float> output;
    const outcome result = run_case(backend, [&](ggml_context * ctx) {
        ggml_tensor * x = new_norm_input(ctx, shape);
        return graph_case{ ggml_group_norm(ctx, x, kGroups, kNormEps), { { x, input } } };
    }, output);
    if (result != outcome::passed) {
        return result;
    }
    const double error = max_abs_error(output, reference_group_norm(input, shape));
    std::printf("    group_norm batch=%lld: max abs err %.2e\n", (long long) shape.batch, error);
    return error <= kNormTolerance ? outcome::passed : outcome::failed;
}

outcome test_group_norm_affine(ggml_backend_t backend, const norm_shape & shape) {
    const std::vector<float> input = make_pattern(shape.elements(), 2);
    const std::vector<float> scale = make_pattern(shape.channels, 3);
    const std::vector<float> shift = make_pattern(shape.channels, 4);
    std::vector<float> output;
    const outcome result = run_case(backend, [&](ggml_context * ctx) {
        ggml_tensor * x = new_norm_input(ctx, shape);
        ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, shape.channels, 1);
        ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, shape.channels, 1);
        ggml_tensor * y = ggml_add(ctx, ggml_mul(ctx, ggml_group_norm(ctx, x, kGroups, kNormEps), w), b);
        return graph_case{ y, { { x, input }, { w, scale }, { b, shift } } };
    }, output);
    if (result != outcome::passed) {
        return result;
    }
    std::vector<float> expected = reference_group_norm(input, shape);
    apply_channel_affine(expected, scale, shift, shape);
    const double error = max_abs_error(output, expected);
    std::printf("    group_norm+mul+add batch=%lld: max abs err %.2e\n", (long long) shape.batch, error);
    return error <= kNormTolerance ? outcome::passed : outcome::failed;
}

double dot(const float * a, const float * b) {
    double acc = 0.0;
    for (int64_t d = 0; d < kHeadDim; ++d) {
        acc += double(a[d])*b[d];
    }
    return acc;
}

std::vector<double> attention_logits(const float * query, const std::vector<float> & keys, int64_t head) {
    std::vector<double> logits(kKeys);
    const double scale = 1.0 / std::sqrt(double(kHeadDim));
    for (int64_t key = 0; key < kKeys; ++key) {
        logits[key] = dot(query, keys.data() + (head*kKeys + key)*kHeadDim)*scale;
    }
    return logits;
}

std::vector<double> softmax(std::vector<double> logits) {
    const double peak = *std::max_element(logits.begin(), logits.end());
    std::transform(logits.begin(), logits.end(), logits.begin(), [&](double v) { return std::exp(v - peak); });
    const double total = std::accumulate(logits.begin(), logits.end(), 0.0);
    std::transform(logits.begin(), logits.end(), logits.begin(), [&](double v) { return v / total; });
    return logits;
}

double weighted_value(const std::vector<double> & weights, const std::vector<float> & values, int64_t head, int64_t d) {
    double acc = 0.0;
    for (int64_t key = 0; key < kKeys; ++key) {
        acc += weights[key]*values[(head*kKeys + key)*kHeadDim + d];
    }
    return acc;
}

void attend_query(const float * query, const std::vector<float> & keys, const std::vector<float> & values,
                  int64_t head, float * out) {
    const std::vector<double> weights = softmax(attention_logits(query, keys, head));
    for (int64_t d = 0; d < kHeadDim; ++d) {
        out[d] = float(weighted_value(weights, values, head, d));
    }
}

std::vector<float> reference_attention(const std::vector<float> & queries, const std::vector<float> & keys,
                                       const std::vector<float> & values, int64_t n_queries) {
    std::vector<float> output(kHeadDim*kHeads*n_queries);
    for (int64_t index = 0; index < kHeads*n_queries; ++index) {
        const int64_t head = index / n_queries;
        const int64_t row = index % n_queries;
        attend_query(queries.data() + (head*n_queries + row)*kHeadDim, keys, values, head,
                     output.data() + (row*kHeads + head)*kHeadDim);
    }
    return output;
}

outcome test_flash_attention(ggml_backend_t backend, int64_t n_queries) {
    const std::vector<float> queries = make_pattern(kHeadDim*n_queries*kHeads, 5);
    const std::vector<float> keys = round_to_f16(make_pattern(kHeadDim*kKeys*kHeads, 6));
    const std::vector<float> values = round_to_f16(make_pattern(kHeadDim*kKeys*kHeads, 7));
    std::vector<float> output;
    const outcome result = run_case(backend, [&](ggml_context * ctx) {
        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kHeadDim, n_queries, kHeads, 1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kHeadDim, kKeys, kHeads, 1);
        ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kHeadDim, kKeys, kHeads, 1);
        ggml_tensor * y = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f / std::sqrt(float(kHeadDim)), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(y, GGML_PREC_F32);
        return graph_case{ y, { { q, queries }, { k, keys }, { v, values } } };
    }, output);
    if (result != outcome::passed) {
        return result;
    }
    const double error = max_abs_error(output, reference_attention(queries, keys, values, n_queries));
    std::printf("    flash_attn_ext queries=%lld: max abs err %.2e\n", (long long) n_queries, error);
    return error <= kAttnTolerance ? outcome::passed : outcome::failed;
}

bool report(const char * name, outcome result) {
    const char * label = result == outcome::passed ? "ok" : result == outcome::skipped ? "skipped" : "FAILED";
    std::printf("  %-36s %s\n", name, label);
    return result != outcome::failed;
}

bool test_backend(ggml_backend_t backend) {
    const norm_shape single = { 16, 16, 64, 1 };
    const norm_shape batched = { 16, 16, 64, 2 };
    bool ok = true;
    ok = report("group_norm batch 1", test_group_norm(backend, single)) && ok;
    ok = report("group_norm batch 2", test_group_norm(backend, batched)) && ok;
    ok = report("group_norm+mul+add batch 1", test_group_norm_affine(backend, single)) && ok;
    ok = report("group_norm+mul+add batch 2", test_group_norm_affine(backend, batched)) && ok;
    ok = report("flash_attn_ext 1024 queries", test_flash_attention(backend, kShortQueries)) && ok;
    ok = report("flash_attn_ext 4100 queries", test_flash_attention(backend, kLongQueries)) && ok;
    return ok;
}

} // namespace

int main() {
    ggml_backend_load_all();
    bool all_ok = true;
    int tested = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            continue;
        }
        std::printf("backend %s (%s)\n", ggml_backend_name(backend), ggml_backend_dev_description(dev));
        all_ok = test_backend(backend) && all_ok;
        ++tested;
        ggml_backend_free(backend);
    }
    if (tested == 0) {
        std::fprintf(stderr, "no backend available\n");
        return 1;
    }
    std::puts(all_ok ? "diffusion batch ops test passed" : "diffusion batch ops test FAILED");
    return all_ok ? 0 : 1;
}
