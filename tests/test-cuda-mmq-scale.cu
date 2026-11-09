#include "ggml-backend.h"
#include "ggml-cuda/quantize.cuh"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;

static void require(bool value, const std::string& message) {
    if (!value)
        throw std::runtime_error(message);
}

static void cuda_check(cudaError_t status, const char* operation) {
    require(status == cudaSuccess, std::string(operation) + ": " + cudaGetErrorString(status));
}

struct DeviceBytes {
    unsigned char* data = nullptr;
    explicit DeviceBytes(size_t size) { cuda_check(cudaMalloc(reinterpret_cast<void**>(&data), size), "cudaMalloc"); }
    ~DeviceBytes() {
        if (data)
            cudaFree(data);
    }
    DeviceBytes(const DeviceBytes&) = delete;
    DeviceBytes& operator=(const DeviceBytes&) = delete;
};

struct Stream {
    cudaStream_t value = nullptr;
    Stream() { cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "cudaStreamCreate"); }
    ~Stream() {
        if (value)
            cudaStreamDestroy(value);
    }
};

struct CapturedGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ~CapturedGraph() {
        if (executable)
            cudaGraphExecDestroy(executable);
        if (graph)
            cudaGraphDestroy(graph);
    }
};

struct Shape {
    int64_t k, n, b2, b3;
    bool padded;
};

static float random_value(uint64_t& state) {
    state += 0x9e3779b97f4a7c15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return static_cast<float>((z ^ (z >> 31)) >> 40) / 8388608.f - 1.f;
}

static float fixture_value(const std::string& fixture, size_t i, uint64_t& seed, int iteration) {
    if (fixture == "random" || fixture == "bias-padding")
        return 3.f * random_value(seed);
    const float tiny = std::numeric_limits<float>::denorm_min();
    const float normal = std::numeric_limits<float>::min();
    if (fixture == "negative-underflow") {
        // Keep complete DS4 groups negative: mixed signs or +0 padding can
        // otherwise hide a fused-versus-separate signed-zero difference.
        const float values[] = {-tiny,         -2.f * tiny, -32.f * tiny,   -64.f * tiny,
                                -127.f * tiny, -normal,     -128.f * normal};
        return values[(i / 128 + iteration) % (sizeof(values) / sizeof(values[0]))];
    }
    if (fixture == "zeros")
        return (i + iteration) % 2 ? -0.f : 0.f;
    if (fixture == "tiny") {
        const float values[] = {tiny, -tiny,        normal,        -normal,        0.f,
                                -0.f, 127.f * tiny, -127.f * tiny, 128.f * normal, -128.f * normal};
        return values[(i + iteration) % (sizeof(values) / sizeof(values[0]))];
    }
    if (fixture == "large") {
        const float maximum = std::numeric_limits<float>::max();
        const float values[] = {maximum,  -maximum, 0.f,       -0.f,    65504.f,
                                -65504.f, 0x1p100f, -0x1p100f, 0x1p20f, -0x1p20f};
        return values[(i + iteration) % (sizeof(values) / sizeof(values[0]))];
    }
    const int exponent = static_cast<int>((i / 32 + iteration) % 121) - 100;
    float value = std::ldexp(1.f, exponent);
    if (i % 3 == 0)
        value = std::nextafter(value, 0.f);
    else if (i % 3 == 1)
        value = std::nextafter(value, std::numeric_limits<float>::infinity());
    return i % 2 ? -value : value;
}

static std::vector<unsigned char> download(const DeviceBytes& allocation, size_t size) {
    std::vector<unsigned char> result(size);
    cuda_check(cudaMemcpy(result.data(), allocation.data, size, cudaMemcpyDeviceToHost), "cudaMemcpy result");
    return result;
}

static void check_guards(const std::vector<unsigned char>& bytes, size_t guard, size_t payload) {
    for (size_t i = 0; i < guard; ++i)
        require(bytes[i] == 0xa5 && bytes[guard + payload + i] == 0xa5, "quantizer wrote outside payload");
}

static void check_padding(const std::vector<unsigned char>& bytes, size_t guard, const Shape& shape, int64_t padded_k) {
    // Whole padded DS4 blocks must use literal +0, including with affine bias.
    if (shape.k % QK8_1_MMQ != 0)
        return;
    const size_t channels = static_cast<size_t>(shape.b2 * shape.b3);
    const size_t blocks_per_channel = static_cast<size_t>(padded_k / QK8_1_MMQ * shape.n);
    for (size_t channel = 0; channel < channels; ++channel)
        for (int64_t block = shape.k / QK8_1_MMQ; block < padded_k / QK8_1_MMQ; ++block)
            for (int64_t row = 0; row < shape.n; ++row) {
                const size_t begin =
                    guard + (channel * blocks_per_channel + block * shape.n + row) * sizeof(block_q8_1_mmq);
                for (size_t i = 0; i < sizeof(block_q8_1_mmq); ++i)
                    require(bytes[begin + i] == 0, "padding differs from literal +0 quantization");
            }
}

static void run_case(ggml_backend_t backend,
                     cudaStream_t stream,
                     const Shape& shape,
                     const std::string& fixture,
                     bool capture = false) {
    constexpr size_t guard = 256;
    constexpr int64_t input_guard = 32;
    const float scale = 1.f / 128.f;
    const float bias = fixture == "bias-padding" ? 0.125f : 0.f;
    const int64_t s01 = shape.k + (shape.padded ? 12 : 0);
    const int64_t s02 = s01 * shape.n + (shape.padded ? 16 : 0);
    const int64_t s03 = s02 * shape.b2 + (shape.padded ? 20 : 0);
    const int64_t input_size = input_guard + s03 * shape.b3 + input_guard;
    const int64_t padded_k = (shape.k + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING * MATRIX_ROW_PADDING;
    const size_t payload =
        static_cast<size_t>(padded_k / QK8_1_MMQ * shape.n * shape.b2 * shape.b3) * sizeof(block_q8_1_mmq);
    const size_t allocation_size = payload + 2 * guard;
    Context context(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    require(context != nullptr, "context allocation failed");
    auto base = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, input_size);
    auto input = ggml_view_4d(context.get(), base, shape.k, shape.n, shape.b2, shape.b3, s01 * sizeof(float),
                              s02 * sizeof(float), s03 * sizeof(float), input_guard * sizeof(float));
    ggml_set_input(base);
    auto packed = ggml_is_contiguous(input) ? input : ggml_cont(context.get(), input);
    auto scaled = ggml_scale_bias(context.get(), packed, scale, bias);
    ggml_set_output(scaled);
    auto graph = ggml_new_graph(context.get());
    ggml_build_forward_expand(graph, scaled);
    // Allocate every node independently: raw input and pre-quantization output
    // cannot be recycled by gallocr or changed by an in-place graph rewrite.
    Buffer buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend), ggml_backend_buffer_free);
    require(buffer != nullptr, "CUDA graph buffer allocation failed");
    ggml_backend_buffer_clear(buffer.get(), 0);
    require(ggml_is_contiguous(scaled), "baseline scale must be packed");
    DeviceBytes original(allocation_size), fused(allocation_size);
    const auto launch_affine = [&] {
        quantize_mmq_q8_1_affine_cuda(static_cast<const float*>(input->data), fused.data + guard, shape.k, s01, s02,
                                      s03, padded_k, shape.n, shape.b2, shape.b3, scale, bias, stream);
    };
    CapturedGraph captured;
    if (capture) {
        ggml_backend_synchronize(backend);
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "quantizer begin capture");
        launch_affine();
        cuda_check(cudaStreamEndCapture(stream, &captured.graph), "quantizer end capture");
        cuda_check(cudaGraphInstantiate(&captured.executable, captured.graph, nullptr, nullptr, 0),
                   "quantizer graph instantiate");
    }
    std::vector<unsigned char> first_a;
    for (int execution = 0; execution < 3; ++execution) {
        const int iteration = execution == 1 ? 1 : 0;
        uint64_t seed = 424242 + iteration;
        std::vector<float> host(static_cast<size_t>(input_size), -91.25f);
        size_t logical = 0;
        for (int64_t b3 = 0; b3 < shape.b3; ++b3)
            for (int64_t b2 = 0; b2 < shape.b2; ++b2)
                for (int64_t row = 0; row < shape.n; ++row)
                    for (int64_t k = 0; k < shape.k; ++k, ++logical)
                        host[input_guard + b3 * s03 + b2 * s02 + row * s01 + k] =
                            fixture_value(fixture, logical, seed, iteration);
        // INPUT flags do not retain storage after a consumer: refresh every run.
        ggml_backend_tensor_set(base, host.data(), 0, host.size() * sizeof(float));
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "baseline SCALE graph failed");
        ggml_backend_synchronize(backend);
        cuda_check(cudaMemsetAsync(original.data, 0xa5, allocation_size, stream), "memset original");
        cuda_check(cudaMemsetAsync(fused.data, 0xa5, allocation_size, stream), "memset fused");
        quantize_mmq_q8_1_cuda(static_cast<const float*>(scaled->data), nullptr, original.data + guard, GGML_TYPE_Q4_K,
                               shape.k, shape.k, shape.k * shape.n, shape.k * shape.n * shape.b2, padded_k, shape.n,
                               shape.b2, shape.b3, stream);
        if (capture)
            cuda_check(cudaGraphLaunch(captured.executable, stream), "quantizer graph replay");
        else
            launch_affine();
        cuda_check(cudaGetLastError(), "quantizer launch");
        cuda_check(cudaStreamSynchronize(stream), "quantizer synchronize");
        const auto expected = download(original, allocation_size);
        const auto actual = download(fused, allocation_size);
        check_guards(expected, guard, payload);
        check_guards(actual, guard, payload);
        check_padding(expected, guard, shape, padded_k);
        check_padding(actual, guard, shape, padded_k);
        if (actual != expected) {
            const size_t offset = static_cast<size_t>(
                std::mismatch(actual.begin(), actual.end(), expected.begin()).first - actual.begin());
            throw std::runtime_error("DS4 payload differs at allocation byte " + std::to_string(offset) + " expected=" +
                                     std::to_string(expected[offset]) + " actual=" + std::to_string(actual[offset]));
        }
        if (execution == 0)
            first_a = actual;
        else if (execution == 2)
            require(actual == first_a, "A/B/A quantized payload is not byte-exact");
        std::vector<float> after(host.size());
        ggml_backend_tensor_get(base, after.data(), 0, after.size() * sizeof(float));
        require(std::memcmp(host.data(), after.data(), host.size() * sizeof(float)) == 0,
                "quantizer or scale changed input/stride padding/guards");
    }
    std::printf(
        "DS4 exact: K=%lld N=%lld B=%lldx%lld padded=%d fixture=%s "
        "payload=%zu A/B/A guards readonly\n",
        static_cast<long long>(shape.k), static_cast<long long>(shape.n), static_cast<long long>(shape.b2),
        static_cast<long long>(shape.b3), shape.padded, fixture.c_str(), payload);
}

struct ScaledLinearGraph {
    Context context;
    Buffer buffer;
    ggml_tensor* input = nullptr;
    ggml_tensor* prescale = nullptr;
    ggml_tensor* matmul = nullptr;
    ggml_tensor* output = nullptr;
    ggml_tensor* observed = nullptr;
    ggml_cgraph* graph = nullptr;
    bool inplace;

    ScaledLinearGraph(ggml_backend_t backend,
                      ggml_tensor* weights,
                      int64_t n,
                      int64_t batch,
                      const std::string& guard,
                      bool reference)
        : context(ggml_init({1024 * 1024, nullptr, true}), ggml_free),
          buffer(nullptr, ggml_backend_buffer_free),
          inplace(guard == "inplace-scale") {
        require(context != nullptr, "scaled Linear context allocation failed");
        input = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, weights->ne[0], n, batch);
        ggml_set_input(input);
        prescale = inplace ? ggml_scale_inplace(context.get(), input, 1.f / 128.f)
                           : ggml_scale(context.get(), input, 1.f / 128.f);
        matmul = ggml_mul_mat(context.get(), weights, prescale);
        ggml_mul_mat_set_prec(matmul, GGML_PREC_F32);
        output = ggml_scale(context.get(), matmul, 128.f);
        ggml_set_output(output);
        // Observation forces the actual original SCALE and quantizer path.
        if (reference || guard == "observed-scale")
            ggml_set_output(prescale);
        if (guard == "observed-mm")
            observed = matmul;
        else if (guard == "multi-scale")
            observed = ggml_scale(context.get(), prescale, 2.f);
        else if (guard == "multi-mm")
            observed = ggml_scale(context.get(), matmul, 2.f);
        if (observed)
            ggml_set_output(observed);
        graph = ggml_new_graph(context.get());
        ggml_build_forward_expand(graph, output);
        if (observed)
            ggml_build_forward_expand(graph, observed);
        // Native independent allocations retain cuts without OUTPUT flags on
        // the eligible candidate or graph-allocator recycling of viewed bases.
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), backend));
        require(buffer != nullptr, "scaled Linear buffer allocation failed");
        ggml_backend_buffer_clear(buffer.get(), 0);
        require(prescale->op == GGML_OP_SCALE && matmul->op == GGML_OP_MUL_MAT && output->op == GGML_OP_SCALE,
                "scaled Linear topology changed");
        require(matmul->src[0] == weights && matmul->src[1] == prescale && output->src[0] == matmul,
                "scaled Linear must reach actual shared-weight matmul");
        require(weights->type == GGML_TYPE_Q4_K && input->type == GGML_TYPE_F32 && matmul->type == GGML_TYPE_F32 &&
                    ggml_get_op_params_i32(matmul, 0) == GGML_PREC_F32,
                "scaled Linear type or precision changed");
        uint32_t bits[2];
        std::memcpy(bits, prescale->op_params, sizeof(bits));
        require(bits[0] == 0x3c000000u && bits[1] == 0, "prescale must be exactly 1/128 and +0 bias");
        std::memcpy(bits, output->op_params, sizeof(bits));
        require(bits[0] == 0x43000000u && bits[1] == 0, "postscale must be exactly 128 and +0 bias");
        require(ggml_is_contiguous(input) && ggml_is_contiguous(prescale) && ggml_is_contiguous(matmul) &&
                    ggml_is_contiguous(output),
                "scaled Linear fixture must be dense");
        require(output->ne[0] == weights->ne[1] && output->ne[1] == n && output->ne[2] == batch,
                "scaled Linear output dimensions changed");
        require((prescale->view_src != nullptr) == inplace, "in-place fixture alias missing");
        require(((prescale->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) == (reference || guard == "observed-scale"),
                "prescale observability differs from intended policy");
        int scale_index = -1, mm_index = -1;
        for (int i = 0; i < graph->n_nodes; ++i) {
            if (graph->nodes[i] == prescale)
                scale_index = i;
            if (graph->nodes[i] == matmul)
                mm_index = i;
        }
        require(scale_index >= 0 && mm_index == scale_index + 1, "actual SCALE->MM nodes must be adjacent");
        require(ggml_node_get_use_count(graph, scale_index) == (guard == "multi-scale" ? 2 : 1) &&
                    ggml_node_get_use_count(graph, mm_index) == (guard == "multi-mm" ? 2 : 1),
                "actual intermediate consumers differ from intended guard");
        require(prescale->data != matmul->data && matmul->data != output->data &&
                    (inplace || prescale->data != input->data),
                "fixture cuts unexpectedly alias");
    }
};

struct ScaledLinearResult {
    std::vector<float> output, matmul, prescale, observed, input_after;
};

static std::vector<float> read_f32(ggml_tensor* tensor) {
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}

static bool byte_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

static void require_result_exact(const ScaledLinearResult& a, const ScaledLinearResult& b) {
    require(byte_equal(a.output, b.output) && byte_equal(a.matmul, b.matmul) && byte_equal(a.prescale, b.prescale) &&
                byte_equal(a.observed, b.observed) && byte_equal(a.input_after, b.input_after),
            "scaled Linear A/B/A differs");
}

static void poison_f32(ggml_tensor* tensor) {
    std::vector<uint32_t> poison(ggml_nelements(tensor), 0x7fc12345u);
    ggml_backend_tensor_set(tensor, poison.data(), 0, poison.size() * sizeof(uint32_t));
}

static ScaledLinearResult execute_scaled_linear(ggml_backend_t backend,
                                                ScaledLinearGraph& graph,
                                                const std::vector<float>& input,
                                                bool expect_elided) {
    ggml_backend_tensor_set(graph.input, input.data(), 0, input.size() * sizeof(float));
    if (!graph.inplace)
        poison_f32(graph.prescale);
    poison_f32(graph.matmul);
    poison_f32(graph.output);
    if (graph.observed && graph.observed != graph.matmul && graph.observed != graph.prescale)
        poison_f32(graph.observed);
    require(ggml_backend_graph_compute(backend, graph.graph) == GGML_STATUS_SUCCESS, "scaled Linear graph failed");
    ggml_backend_synchronize(backend);
    ScaledLinearResult result;
    result.output = read_f32(graph.output);
    result.matmul = read_f32(graph.matmul);
    result.prescale = read_f32(graph.prescale);
    result.input_after = read_f32(graph.input);
    if (graph.observed)
        result.observed = read_f32(graph.observed);
    for (const auto* values : {&result.output, &result.matmul, &result.observed})
        for (float value : *values)
            require(std::isfinite(value), "scaled Linear output/cut is nonfinite");
    if (expect_elided) {
        for (float value : result.prescale) {
            uint32_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            require(bits == 0x7fc12345u, "eligible CUDA graph materialized its supposedly elided prescale");
        }
    } else {
        for (float value : result.prescale)
            require(std::isfinite(value), "required prescale write was elided");
    }
    if (graph.inplace)
        require(byte_equal(result.input_after, result.prescale) && !byte_equal(result.input_after, input),
                "in-place SCALE did not preserve its observable input write");
    else
        require(byte_equal(result.input_after, input), "scaled Linear changed readonly input");
    return result;
}

static std::vector<unsigned char> linear_weights(int64_t k, int64_t m) {
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_K, k);
    std::vector<unsigned char> result(row_bytes * m);
    uint64_t seed = 1337;
    for (int64_t first = 0; first < m; first += 64) {
        const int64_t rows = std::min<int64_t>(64, m - first);
        std::vector<float> values(rows * k);
        for (float& value : values)
            value = 0.02f * random_value(seed);
        require(ggml_quantize_chunk(GGML_TYPE_Q4_K, values.data(), result.data() + first * row_bytes, 0, rows, k,
                                    nullptr) == rows * row_bytes,
                "Q4_K native weight quantization failed");
    }
    return result;
}

static void run_scaled_linear_case(ggml_backend_t backend,
                                   ggml_tensor* weights,
                                   int64_t n,
                                   int64_t batch,
                                   const std::string& guard,
                                   bool cuda_policy) {
    ScaledLinearGraph reference(backend, weights, n, batch, guard, true);
    ScaledLinearGraph candidate(backend, weights, n, batch, guard, false);
    const bool expect_elided = cuda_policy && n >= 512 && batch == 1 && guard == "dense";
    uint64_t seed_a = 424242, seed_b = 424243;
    std::vector<float> host_a(ggml_nelements(reference.input)), host_b(host_a.size());
    for (size_t i = 0; i < host_a.size(); ++i) {
        host_a[i] = 0.25f * random_value(seed_a);
        host_b[i] = 0.25f * random_value(seed_b);
    }
    ScaledLinearResult first_reference, first_candidate;
    for (int iteration = 0; iteration < 3; ++iteration) {
        const auto& input = iteration == 1 ? host_b : host_a;
        auto expected = execute_scaled_linear(backend, reference, input, false);
        auto actual = execute_scaled_linear(backend, candidate, input, expect_elided);
        require(byte_equal(actual.output, expected.output) && byte_equal(actual.matmul, expected.matmul) &&
                    byte_equal(actual.observed, expected.observed) &&
                    byte_equal(actual.input_after, expected.input_after),
                "scaled Linear differs from actual materialized reference");
        if (!expect_elided)
            require(byte_equal(actual.prescale, expected.prescale), "fallback prescale differs from reference");
        if (iteration == 0) {
            first_reference = std::move(expected);
            first_candidate = std::move(actual);
        } else if (iteration == 1) {
            require(!byte_equal(actual.output, first_candidate.output) &&
                        !byte_equal(expected.output, first_reference.output),
                    "B input did not change scaled Linear output");
        } else {
            require_result_exact(expected, first_reference);
            require_result_exact(actual, first_candidate);
        }
    }
    std::printf(
        "scaled Linear exact: K=%lld M=%lld N=%lld batch=%lld guard=%s prescale_elided=%d A/B/A shared-Q4_K-weights\n",
        static_cast<long long>(weights->ne[0]), static_cast<long long>(weights->ne[1]), static_cast<long long>(n),
        static_cast<long long>(batch), guard.c_str(), expect_elided);
}

static void run_scaled_linear_graphs(ggml_backend_t backend) {
    cudaDeviceProp device;
    cuda_check(cudaGetDeviceProperties(&device, 0), "cudaGetDeviceProperties");
    const char* disabled = std::getenv("GGML_CUDA_DISABLE_FUSION");
    bool cuda_policy = device.major == 12 && device.minor == 0 && !(disabled && std::atoi(disabled));
#ifdef GGML_TEST_MMQ_SCALE_FORCE_CUBLAS
    cuda_policy = false;
#endif
    const int64_t shapes[][2] = {{5376, 28672}, {14336, 5376}};
    for (const auto& shape : shapes) {
        const auto host_weights = linear_weights(shape[0], shape[1]);
        Context context(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
        require(context != nullptr, "shared weight context allocation failed");
        auto weights = ggml_new_tensor_2d(context.get(), GGML_TYPE_Q4_K, shape[0], shape[1]);
        Buffer buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend), ggml_backend_buffer_free);
        require(buffer != nullptr, "shared Q4_K weight allocation failed");
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_clear(buffer.get(), 0);
        ggml_backend_tensor_set(weights, host_weights.data(), 0, host_weights.size());
        run_scaled_linear_case(backend, weights, 513, 1, "dense", cuda_policy);
        run_scaled_linear_case(backend, weights, 30, 1, "dense", cuda_policy);
        run_scaled_linear_case(backend, weights, 513, 2, "dense", cuda_policy);
        for (const char* guard : {"observed-scale", "observed-mm", "multi-scale", "multi-mm", "inplace-scale"})
            run_scaled_linear_case(backend, weights, 513, 1, guard, cuda_policy);
        std::vector<unsigned char> after(host_weights.size());
        ggml_backend_tensor_get(weights, after.data(), 0, after.size());
        require(after == host_weights, "scaled Linear changed shared readonly Q4_K weights");
    }
    ggml_quantize_free();
}

int main(int argc, char** argv) {
    try {
        bool large = false;
        if (argc == 2 && std::string(argv[1]) == "--large")
            large = true;
        else
            require(argc == 1, "usage: test-cuda-mmq-scale [--large]");
        const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
        const char* order = std::getenv("CUDA_DEVICE_ORDER");
        if (!visible || std::string(visible) != "0" || !order || std::string(order) != "PCI_BUS_ID") {
            std::fprintf(stderr,
                         "SKIP: run explicitly with CUDA_VISIBLE_DEVICES=0 "
                         "CUDA_DEVICE_ORDER=PCI_BUS_ID\n");
            return 77;
        }
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
            return 77;
        cuda_check(cudaSetDevice(0), "cudaSetDevice");
        ggml_backend_load_all();
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_init_by_name("CUDA0", nullptr),
                                                                            ggml_backend_free);
        require(backend != nullptr, "CUDA0 backend unavailable");
        Stream stream;
        std::vector<Shape> shapes = {
            {4, 1, 1, 1, false},    {128, 3, 1, 1, true},    {256, 17, 2, 2, true},     {260, 9, 1, 2, true},
            {512, 65, 2, 1, false}, {5376, 17, 1, 1, false}, {14336, 513, 1, 1, false},
        };
        if (large) {
            shapes.push_back({5376, 15429, 1, 1, false});
            shapes.push_back({14336, 15429, 1, 1, false});
        }
        size_t count = 0;
        for (const auto& shape : shapes)
            for (const char* fixture :
                 {"random", "zeros", "tiny", "negative-underflow", "large", "powers", "bias-padding"}) {
                run_case(backend.get(), stream.value, shape, fixture);
                ++count;
            }
        std::printf(
            "CUDA SCALE + original DS4 quantizer vs affine quantizer: %zu "
            "cases passed byte-exact\n",
            count);
        run_case(backend.get(), stream.value, {5376, 17, 1, 1, false}, "random", true);
        std::puts("Affine quantizer CUDA stream capture/replay: A/B/A bytes, canaries and readonly input passed");
        run_scaled_linear_graphs(backend.get());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "test-cuda-mmq-scale: %s\n", error.what());
        return 1;
    }
}
