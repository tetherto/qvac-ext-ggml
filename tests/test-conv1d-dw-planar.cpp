#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
constexpr int k_benchmark_runs = 11;
constexpr int k_benchmark_warmups = 3;
constexpr float k_tolerance = 2e-5f;

struct ConvCase {
    int width;
    int channels;
    int kernel;
    int padding;
    int stride = 1;
    int dilation = 1;
    int batch = 2;
    ggml_type weight_type = GGML_TYPE_F32;
};

struct ConvFixture {
    ggml_context_ptr ctx;
    ggml_backend_ptr backend;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * input;
    ggml_tensor * kernel;
    ggml_tensor * output;
    ggml_cgraph * graph;
    std::vector<float> source;
    std::vector<float> weights;
};

ConvFixture create_fixture(const ConvCase & c, int threads) {
    ggml_init_params params = {64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true};
    ConvFixture f;
    f.ctx.reset(ggml_init(params));
    f.backend.reset(ggml_backend_cpu_init());
    ggml_backend_cpu_set_n_threads(f.backend.get(), threads);
    f.input = ggml_new_tensor_4d(f.ctx.get(), GGML_TYPE_F32, c.width, 1, c.channels, c.batch);
    f.kernel = ggml_new_tensor_4d(f.ctx.get(), c.weight_type, c.kernel, 1, 1, c.channels);
    f.output = ggml_conv_2d_dw_direct(f.ctx.get(), f.kernel, f.input,
        c.stride, 1, c.padding, 0, c.dilation, 1);
    f.graph = ggml_new_graph(f.ctx.get());
    ggml_build_forward_expand(f.graph, f.output);
    f.buffer.reset(ggml_backend_alloc_ctx_tensors(f.ctx.get(), f.backend.get()));
    return f;
}

void initialize_fixture(ConvFixture & f, const ConvCase & c) {
    constexpr float source_frequency = 0.17f;
    constexpr float weight_frequency = 0.31f;
    f.source.resize(ggml_nelements(f.input));
    f.weights.resize(ggml_nelements(f.kernel));
    for (size_t i = 0; i < f.source.size(); ++i) {
        f.source[i] = std::sin(float(i)*source_frequency);
    }
    for (size_t i = 0; i < f.weights.size(); ++i) {
        f.weights[i] = std::cos(float(i)*weight_frequency);
    }
    ggml_backend_tensor_set(f.input, f.source.data(), 0, ggml_nbytes(f.input));
    if (c.weight_type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(f.weights.size());
        ggml_fp32_to_fp16_row(f.weights.data(), half.data(), f.weights.size());
        ggml_backend_tensor_set(f.kernel, half.data(), 0, ggml_nbytes(f.kernel));
        ggml_fp16_to_fp32_row(half.data(), f.weights.data(), f.weights.size());
    } else {
        ggml_backend_tensor_set(f.kernel, f.weights.data(), 0, ggml_nbytes(f.kernel));
    }
}

bool check_output(const ConvFixture & f, const ConvCase & c) {
    const int outputs = int(f.output->ne[0]);
    std::vector<float> result(ggml_nelements(f.output));
    ggml_backend_tensor_get(f.output, result.data(), 0, ggml_nbytes(f.output));
    for (int row = 0; row < c.channels*c.batch; ++row) {
        for (int x = 0; x < outputs; ++x) {
            float expected = 0;
            for (int tap = 0; tap < c.kernel; ++tap) {
                const int sx = x*c.stride - c.padding + tap*c.dilation;
                if (sx >= 0 && sx < c.width) {
                    expected += f.source[row*c.width + sx]*f.weights[(row%c.channels)*c.kernel + tap];
                }
            }
            const float actual = result[row*outputs + x];
            if (!std::isfinite(actual) || std::abs(actual - expected) > k_tolerance*(1 + std::abs(expected))) {
                std::fprintf(stderr, "Mismatch: width=%d kernel=%d pad=%d row=%d x=%d actual=%g expected=%g\n",
                    c.width, c.kernel, c.padding, row, x, actual, expected);
                return false;
            }
        }
    }
    return true;
}

bool benchmark_fixture(const ConvFixture & f, const ConvCase & c, int threads) {
    std::vector<double> timings;
    for (int run = -k_benchmark_warmups; run < k_benchmark_runs; ++run) {
        const auto start = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(f.backend.get(), f.graph) != GGML_STATUS_SUCCESS) {
            return false;
        }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        if (run >= 0) {
            timings.push_back(ms);
        }
    }
    std::sort(timings.begin(), timings.end());
    std::printf("width=%d channels=%d kernel=%d threads=%d median_ms=%.6f\n",
        c.width, c.channels, c.kernel, threads, timings[timings.size()/2]);
    return true;
}

bool run_case(const ConvCase & c, int threads, bool benchmark) {
    ConvFixture fixture = create_fixture(c, threads);
    if (!fixture.buffer) {
        return false;
    }
    initialize_fixture(fixture, c);
    if (ggml_backend_graph_compute(fixture.backend.get(), fixture.graph) != GGML_STATUS_SUCCESS ||
        !check_output(fixture, c)) {
        return false;
    }
    return !benchmark || benchmark_fixture(fixture, c, threads);
}

bool run_regressions() {
    for (int threads : {1, 8}) {
        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            for (int width : {1, 7, 17, 33, 137}) {
                for (int kernel : {1, 9, 31}) {
                    ConvCase c{width, 5, kernel, kernel/2};
                    c.weight_type = type;
                    if (!run_case(c, threads, false)) {
                        return false;
                    }
                }
            }
        }
        for (const ConvCase & c : {ConvCase{137, 5, 9, 0}, ConvCase{137, 5, 9, 8},
                                  ConvCase{137, 5, 9, 8, 1, 2}, ConvCase{137, 5, 9, 4, 2}}) {
            if (!run_case(c, threads, false)) {
                return false;
            }
        }
    }
    std::puts("Planar depthwise convolution regression cases passed.");
    return true;
}
} // namespace

int main(int argc, char ** argv) {
    ggml_time_init();
    if (argc == 2 && std::strcmp(argv[1], "--bench") == 0) {
        for (int threads : {1, 8}) {
            for (int channels : {640, 1024}) {
                for (int width : {137, 750, 3750}) {
                    if (!run_case({width, channels, 9, 4, 1, 1, 1}, threads, true)) {
                        return 1;
                    }
                }
            }
        }
        return 0;
    }
    return run_regressions() ? 0 : 1;
}
