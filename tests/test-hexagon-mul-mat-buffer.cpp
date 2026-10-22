#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hexagon.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static constexpr int SKIP_RETURN_CODE = 77;
// The ordinary MUL_MAT harness uses this same NMSE limit.
static constexpr double MAX_NMSE = 5e-4;

struct shape {
    int64_t m;
    int64_t n;
    int64_t k;
};

struct tensor_storage {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    ~tensor_storage() {
        ggml_backend_buffer_free(buffer);
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

static bool support_probe(ggml_backend_t backend, ggml_type type, ggml_backend_buffer_type_t buft,
                          bool dummy, bool expected, const char * label) {
    tensor_storage storage;
    storage.ctx = ggml_init({ ggml_tensor_overhead() * 8, nullptr, true });
    if (!storage.ctx) {
        std::fprintf(stderr, "%s: context allocation failed\n", label);
        return false;
    }
    ggml_tensor * weight = ggml_new_tensor_2d(storage.ctx, type, 256, 16);
    ggml_tensor * input = ggml_new_tensor_2d(storage.ctx, GGML_TYPE_F32, 256, 1);
    ggml_tensor * output = ggml_mul_mat(storage.ctx, weight, input);
    if (buft) {
        storage.buffer = dummy ? ggml_backend_buft_alloc_buffer(buft, 0) :
                                 ggml_backend_alloc_ctx_tensors_from_buft(storage.ctx, buft);
        if (!storage.buffer) {
            std::fprintf(stderr, "%s: %s buffer allocation failed\n", label, ggml_backend_buft_name(buft));
            return false;
        }
        if (dummy) {
            // Model loaders attach this tag to query an intended weight layout.
            weight->buffer = storage.buffer;
        }
    }
    const bool supported = ggml_backend_supports_op(backend, output);
    const bool ok = supported == expected;
    std::printf("support %s %s: supported=%d expected=%d %s\n", ggml_type_name(type), label,
                int(supported), int(expected), ok ? "PASS" : "FAIL");
    return ok;
}

static ggml_backend_buffer_type_t repack_buffer_type(ggml_backend_dev_t dev) {
    auto extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");
    for (auto bufts = extra ? extra(dev) : nullptr; bufts && *bufts; ++bufts) {
        if (!ggml_backend_buft_is_host(*bufts) && ggml_backend_dev_supports_buft(dev, *bufts)) {
            return *bufts;
        }
    }
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    return !ggml_backend_buft_is_host(buft) ? buft : nullptr;
}

static float value(size_t index, uint32_t seed) {
    uint32_t bits = uint32_t(index) + seed;
    bits ^= bits >> 16;
    bits *= 0x7feb352dU;
    bits ^= bits >> 15;
    bits *= 0x846ca68bU;
    bits ^= bits >> 16;
    return float(int(bits & 0xffffU) - 32768) / 65536.0f;
}

static bool compute(ggml_backend_t backend, ggml_backend_buffer_type_t weight_buft, const shape & s,
                    const std::vector<uint8_t> & weights, const std::vector<float> & input,
                    std::vector<float> & output) {
    tensor_storage weight_storage;
    weight_storage.ctx = ggml_init({ ggml_tensor_overhead() * 2, nullptr, true });
    tensor_storage graph_storage;
    graph_storage.ctx = ggml_init({ ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true });
    if (!weight_storage.ctx || !graph_storage.ctx) {
        std::fprintf(stderr, "%s: context allocation failed\n", ggml_backend_name(backend));
        return false;
    }
    ggml_tensor * weight = ggml_new_tensor_2d(weight_storage.ctx, GGML_TYPE_Q8_0, s.k, s.m);
    weight_storage.buffer = ggml_backend_alloc_ctx_tensors_from_buft(weight_storage.ctx, weight_buft);
    if (!weight_storage.buffer) {
        std::fprintf(stderr, "%s: weight allocation failed\n", ggml_backend_name(backend));
        return false;
    }
    ggml_backend_buffer_set_usage(weight_storage.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_tensor * x = ggml_new_tensor_2d(graph_storage.ctx, GGML_TYPE_F32, s.k, s.n);
    ggml_tensor * y = ggml_mul_mat(graph_storage.ctx, weight, x);
    ggml_cgraph * graph = ggml_new_graph(graph_storage.ctx);
    ggml_build_forward_expand(graph, y);
    graph_storage.buffer = ggml_backend_alloc_ctx_tensors(graph_storage.ctx, backend);
    if (!graph_storage.buffer || !ggml_backend_supports_op(backend, y)) {
        std::fprintf(stderr, "%s: allocated MUL_MAT graph is not supported\n", ggml_backend_name(backend));
        return false;
    }
    ggml_backend_tensor_set(weight, weights.data(), 0, weights.size());
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    // Direct execution on this one backend: there is no scheduler/CPU fallback.
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: graph execution returned %d\n", ggml_backend_name(backend), int(status));
        return false;
    }
    output.resize(size_t(s.m * s.n));
    ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
    return true;
}

static bool math_case(ggml_backend_t cpu, ggml_backend_t htp, ggml_backend_buffer_type_t repack,
                      const shape & s) {
    std::vector<float> original(size_t(s.k * s.m));
    std::vector<float> input(size_t(s.k * s.n));
    for (size_t i = 0; i < original.size(); ++i) {
        original[i] = value(i, 0x31415926U);
    }
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = value(i, 0x9e3779b9U);
    }
    std::vector<uint8_t> canonical(size_t(ggml_row_size(GGML_TYPE_Q8_0, s.k) * s.m));
    if (ggml_quantize_chunk(GGML_TYPE_Q8_0, original.data(), canonical.data(), 0, s.m, s.k, nullptr) != canonical.size()) {
        std::fprintf(stderr, "Q8_0 quantization produced an unexpected byte count\n");
        return false;
    }
    std::vector<float> reference, actual;
    if (!compute(cpu, ggml_backend_get_default_buffer_type(cpu), s, canonical, input, reference)) {
        std::fprintf(stderr, "Q8_0 (m,n,k)=(%lld,%lld,%lld): CPU reference execution FAIL\n",
                     (long long) s.m, (long long) s.n, (long long) s.k);
        return false;
    }
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i])) {
            std::fprintf(stderr, "Q8_0 (m,n,k)=(%lld,%lld,%lld): CPU reference element %zu=%g FAIL\n",
                         (long long) s.m, (long long) s.n, (long long) s.k, i, double(reference[i]));
            return false;
        }
    }
    if (!compute(htp, repack, s, canonical, input, actual)) {
        std::fprintf(stderr, "Q8_0 (m,n,k)=(%lld,%lld,%lld): execution FAIL\n",
                     (long long) s.m, (long long) s.n, (long long) s.k);
        return false;
    }
    double error = 0.0, norm = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(actual[i])) {
            std::fprintf(stderr, "Q8_0 (m,n,k)=(%lld,%lld,%lld): element %zu CPU=%g HTP0=%g FAIL\n",
                         (long long) s.m, (long long) s.n, (long long) s.k, i,
                         double(reference[i]), double(actual[i]));
            return false;
        }
        const double difference = double(actual[i]) - double(reference[i]);
        error += difference * difference;
        norm += double(reference[i]) * double(reference[i]);
    }
    const double nmse = norm > 0.0 ? error / norm : error;
    const bool ok = std::isfinite(nmse) && nmse <= MAX_NMSE;
    std::printf("Q8_0 HTP0 explicit-repack (m,n,k)=(%lld,%lld,%lld): NMSE=%.9g limit=%.9g %s\n",
                (long long) s.m, (long long) s.n, (long long) s.k, nmse, MAX_NMSE, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("HTP0");
    if (!dev) {
        std::puts("SKIP: HTP0 is unavailable; no Hexagon contract or math cases executed");
        return SKIP_RETURN_CODE;
    }
    ggml_backend_t htp = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_name("CPU", nullptr);
    if (!htp || !cpu || ggml_backend_get_device(htp) != dev ||
        std::strcmp(ggml_backend_dev_name(ggml_backend_get_device(htp)), "HTP0") != 0 ||
        ggml_backend_dev_type(ggml_backend_get_device(cpu)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        std::fprintf(stderr, "FAIL: actual HTP0 and CPU reference initialization are required\n");
        ggml_backend_free(htp);
        ggml_backend_free(cpu);
        return 1;
    }
    auto query_runtime = reinterpret_cast<ggml_backend_hexagon_get_runtime_info_v1_t>(
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),
                                         "ggml_backend_hexagon_get_runtime_info_v1"));
    ggml_backend_hexagon_runtime_info_v1 runtime = {};
    if (!query_runtime || !query_runtime(htp, 0, &runtime, sizeof(runtime)) ||
        runtime.version != GGML_HEXAGON_RUNTIME_INFO_VERSION || runtime.session_count == 0 ||
        (runtime.op_stage & 3) != 3) {
        std::fprintf(stderr, "FAIL: confirmed HTP queue+compute execution is required (runtime version=%u stage=%u sessions=%u)\n",
                     runtime.version, runtime.op_stage, runtime.session_count);
        ggml_backend_free(htp);
        ggml_backend_free(cpu);
        return 1;
    }
    const auto default_buft = ggml_backend_get_default_buffer_type(htp);
    const auto cpu_buft = ggml_backend_get_default_buffer_type(cpu);
    const auto repack = repack_buffer_type(dev);
    if (!repack) {
        std::fprintf(stderr, "FAIL: HTP0 exposes no non-host repack weight buffer\n");
        ggml_backend_free(htp);
        ggml_backend_free(cpu);
        return 1;
    }
    const bool default_host = ggml_backend_buft_is_host(default_buft);
    std::printf("backend=HTP0 default=%s host=%d weights=%s reference=%s op_stage=%u\n",
                ggml_backend_buft_name(default_buft), int(default_host), ggml_backend_buft_name(repack),
                ggml_backend_name(cpu), runtime.op_stage);

    int contract_pass = 0, contract_fail = 0;
    const auto record = [&](bool ok) { ok ? ++contract_pass : ++contract_fail; };
    const ggml_type types[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL, GGML_TYPE_MXFP4 };
    for (ggml_type type : types) {
        record(support_probe(htp, type, nullptr, false, !default_host, "bufferless-default"));
        record(support_probe(htp, type, default_buft, false, !default_host, "allocated-default"));
        record(support_probe(htp, type, repack, true, true, "zero-size-repack"));
        record(support_probe(htp, type, repack, false, true, "allocated-repack"));
        record(support_probe(htp, type, cpu_buft, false, false, "canonical-CPU"));
    }
    record(support_probe(htp, GGML_TYPE_F16, nullptr, false, true, "bufferless-default"));

    // The five unique shapes from the previously failing default Q8_0 groups.
    const shape shapes[] = { {16, 1, 256}, {16, 16, 256}, {2048, 90, 512}, {2048, 96, 512}, {2048, 97, 512} };
    int math_pass = 0, math_fail = 0;
    for (const shape & s : shapes) {
        math_case(cpu, htp, repack, s) ? ++math_pass : ++math_fail;
    }
    std::printf("support-contract: %d passed, %d failed; HTP0 explicit-repack CPU-reference: %d passed, %d failed\n",
                contract_pass, contract_fail, math_pass, math_fail);
    ggml_backend_free(htp);
    ggml_backend_free(cpu);
    ggml_quantize_free();
    return contract_fail == 0 && math_fail == 0 ? 0 : 1;
}
