#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

namespace {
constexpr int heads = 56, dim = 128;
constexpr float kv_scale = 1.0f / 128.0f;
const float scale = (1.0f / std::sqrt(float(dim))) / kv_scale;
int activated = 0;

void log_message(ggml_log_level, const char * message, void *) {
    if (std::strstr(message, "cuDNN SDPA active:")) {
        ++activated;
    }
    std::fputs(message, stderr);
}

// Alternating channel values exercise every dimension. Their repeated structure
// permits an independent FP64 dot product at the full sequence lengths.
float q0(int t, int h) {
    return float((t + h * 5) % 13 - 6) / 8.0f + float((t * 13 + h) % 7 - 3) / 8192.0f;
}
float q1(int t, int h) {
    return float((t * 3 + h) % 11 - 5) / 8.0f + float((t + h * 7) % 5 - 2) / 16384.0f;
}
float k0(int t, int h) { return float((t + h * 2) % 17 - 8) / 8.0f; }
float k1(int t, int h) { return float((t * 5 + h) % 19 - 9) / 8.0f; }
float value(int t, int h, int d) { return float((t * 3 + h * 5 + d) % 23 - 11 + 2 * ((t + h * 2) % 17 - 8)) / 128.0f; }

bool run(ggml_backend_t backend, int tokens, bool masked, bool cudnn_enabled) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false), nullptr, true}), ggml_free);
    if (!ctx) {
        return false;
    }
    auto * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, dim, tokens, heads);
    auto * k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, dim, tokens, heads);
    auto * packed = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, dim, heads, tokens);
    auto * v = ggml_permute(ctx.get(), packed, 0, 2, 1, 3);
    auto * mask = masked ? ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, tokens, tokens) : nullptr;
    auto * dst = ggml_flash_attn_ext(ctx.get(), q, k, v, mask, scale, 0, 0);
    ggml_flash_attn_ext_set_prec(dst, GGML_PREC_F32);
    auto * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, dst);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    if (!buffer) {
        return false;
    }
    const size_t count = size_t(dim) * tokens * heads;
    std::vector<float> qs(count), output(count);
    std::vector<ggml_fp16_t> ks(count), vs(count);
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < heads; ++h) {
            const size_t key = size_t(dim) * (t + h * tokens);
            const auto even = ggml_fp32_to_fp16(k0(t, h) * kv_scale);
            const auto odd = ggml_fp32_to_fp16(k1(t, h) * kv_scale);
            for (int d = 0; d < dim; ++d) {
                ks[key + d] = (d % 2) ? odd : even;
                vs[d + dim * (h + heads * t)] = ggml_fp32_to_fp16(value(t, h, d) * kv_scale);
            }
        }
    }
    ggml_backend_tensor_set(k, ks.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(packed, vs.data(), 0, ggml_nbytes(packed));
    ks.clear(); ks.shrink_to_fit();
    vs.clear(); vs.shrink_to_fit();
    if (mask) {
        std::vector<ggml_fp16_t> m(size_t(tokens) * tokens, ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        for (int t = 0; t < tokens; ++t) {
            for (int i = 0; i < 8; ++i) {
                m[t * tokens + i] = ggml_fp32_to_fp16(0.0f);
            }
        }
        ggml_backend_tensor_set(mask, m.data(), 0, ggml_nbytes(mask));
    }
    for (int phase : {0, 3, 7}) {
        for (int h = 0; h < heads; ++h) {
            for (int t = 0; t < tokens; ++t) {
                const size_t query = size_t(dim) * (t + h * tokens);
                const float even = q0(t + phase, h), odd = q1(t + phase, h);
                for (int d = 0; d < dim; d += 2) {
                    qs[query + d] = even;
                    qs[query + d + 1] = odd;
                }
            }
        }
        ggml_backend_tensor_set(q, qs.data(), 0, ggml_nbytes(q));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            return false;
        }
        ggml_backend_tensor_get(dst, output.data(), 0, ggml_nbytes(dst));
        for (float x : output) {
            if (!std::isfinite(x)) {
                return false;
            }
        }
        double max_abs = 0.0, sum_error = 0.0, sum_expected = 0.0;
        for (int t : {0, tokens / 2, tokens - 1}) {
            for (int h : {0, heads / 2, heads - 1}) {
                for (int d : {0, 7, dim - 1}) {
                    double sum = 0.0, total = 0.0;
                    for (int i = 0; i < (masked ? 8 : tokens); ++i) {
                        const double dot = (dim / 2) * double(kv_scale) *
                            (double(q0(t + phase, h)) * k0(i, h) + double(q1(t + phase, h)) * k1(i, h));
                        const double weight = std::exp(double(scale) * dot);
                        sum += weight * value(i, h, d);
                        total += weight;
                    }
                    const double actual = output[d + dim * (h + heads * t)] / kv_scale;
                    const double expected = sum / total;
                    const double error = actual - expected;
                    max_abs = std::max(max_abs, std::abs(error));
                    sum_error += error * error;
                    sum_expected += expected * expected;
                }
            }
        }
        const double rel_l2 = std::sqrt(sum_error / sum_expected);
        std::printf("tokens=%d masked=%d phase=%d scale=%.8g max_abs=%.8g rel_l2=%.8g\n",
                    tokens, int(masked), phase, scale, max_abs, rel_l2);
        // On this fixture, pristine Spark attention reached 1.0834e-3 / 4.8522e-3 at long
        // sequences. Preserve its measured range. Check the opted-in kernel and short
        // native cases against the tighter post-scale FP64 reference throughout.
        const bool long_native = !cudnn_enabled && tokens > 256;
        const double abs_limit = long_native ? 1.2e-3 : 1e-4;
        const double rel_limit = long_native ? 5.5e-3 : 1e-3;
        if (!(max_abs <= abs_limit && rel_l2 <= rel_limit)) {
            return false;
        }
    }
    return true;
}
} // namespace

int main(int argc, char ** argv) {
    ggml_log_set(log_message, nullptr);
    ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * candidate = ggml_backend_dev_get(i);
        if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(candidate)), "CUDA") == 0 &&
            std::strstr(ggml_backend_dev_description(candidate), "GB10")) {
            dev = candidate;
            break;
        }
    }
    if (!dev) {
        return 77;
    }
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(dev, nullptr), ggml_backend_free);
    bool enabled = argc > 1 && std::strcmp(argv[1], "--enabled") == 0;
    if (!backend || !run(backend.get(), 256, true, enabled) || activated != 0 ||
        !run(backend.get(), 256, false, enabled) || activated != int(enabled) ||
        !run(backend.get(), 5912, false, enabled) || activated != 2 * int(enabled) ||
        !run(backend.get(), 15435, false, enabled) || activated != 3 * int(enabled)) {
        return 1;
    }
    return 0;
}
