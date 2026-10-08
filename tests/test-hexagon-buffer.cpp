#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// Host-visible buffers must contain canonical ggml bytes: CPU fallback and
// ggml_backend_tensor_copy may read their data pointers without get_tensor.
static bool check_buffer(ggml_backend_buffer_type_t buft, ggml_backend_t cpu, ggml_type type) {
    const int64_t k = 128, rows = 33;
    ggml_context * ctx = ggml_init({ggml_tensor_overhead() * 2, nullptr, true});
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, type, k, rows);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    GGML_ASSERT(buffer);

    ggml_context * copy_ctx = ggml_init({ggml_tensor_overhead() * 2, nullptr, true});
    ggml_tensor * copy = ggml_new_tensor_2d(copy_ctx, type, k, rows);
    ggml_backend_buffer_t copy_buffer = ggml_backend_alloc_ctx_tensors(copy_ctx, cpu);
    GGML_ASSERT(copy_buffer);

    std::vector<float> values(k * rows);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = std::sin(float(i) * 0.13f) * (0.1f + float(i % 11));
    }
    std::vector<uint8_t> canonical(ggml_nbytes(weight)), actual(canonical.size());
    const size_t size = ggml_quantize_chunk(type, values.data(), canonical.data(), 0, rows, k, nullptr);
    GGML_ASSERT(size == canonical.size());
    ggml_backend_tensor_set(weight, canonical.data(), 0, size);

    ggml_backend_tensor_get(weight, actual.data(), 0, size);
    const bool api_ok = actual == canonical;
    const bool host = ggml_backend_buft_is_host(buft);
    const bool host_ok = !host || std::memcmp(weight->data, canonical.data(), size) == 0;
    ggml_backend_tensor_copy(weight, copy);
    ggml_backend_tensor_get(copy, actual.data(), 0, size);
    const bool copy_ok = actual == canonical;
    bool partial_ok = true;
    if (host) {
        // Streaming loaders may transfer canonical blocks at nonzero offsets.
        const size_t offset = ggml_row_size(type, k);
        const size_t chunk = offset;
        std::vector<uint8_t> row(chunk);
        ggml_backend_tensor_get(weight, row.data(), offset, chunk);
        partial_ok = std::memcmp(row.data(), canonical.data() + offset, chunk) == 0;
        ggml_backend_tensor_set(weight, canonical.data(), offset, chunk);
        ggml_backend_tensor_get(weight, actual.data(), 0, size);
        std::memcpy(canonical.data() + offset, canonical.data(), chunk);
        partial_ok = actual == canonical && partial_ok;
    }
    const bool ok = api_ok && host_ok && copy_ok && partial_ok;
    std::printf("%s %s: API=%s host-bytes=%s CPU-copy=%s partial=%s\n", ggml_backend_buft_name(buft),
                ggml_type_name(type), api_ok ? "OK" : "FAIL", host ? (host_ok ? "OK" : "FAIL") : "N/A",
                copy_ok ? "OK" : "FAIL", host ? (partial_ok ? "OK" : "FAIL") : "N/A");
    ggml_backend_buffer_free(copy_buffer);
    ggml_free(copy_ctx);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_name("CPU", nullptr);
    GGML_ASSERT(cpu);
    const ggml_type types[] = {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL, GGML_TYPE_MXFP4};
    bool ok = true;
    for (auto type : types) {
        ok = check_buffer(ggml_backend_get_default_buffer_type(cpu), cpu, type) && ok;
    }
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("HTP0");
    if (!dev) {
        std::puts("SKIP: Hexagon device unavailable");
        ggml_backend_free(cpu);
        ggml_quantize_free();
        return ok ? 77 : 1;
    }
    ggml_backend_t hexagon = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(hexagon);
    std::vector<ggml_backend_buffer_type_t> bufts = { ggml_backend_get_default_buffer_type(hexagon) };
    auto extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");
    if (extra) {
        for (auto p = extra(dev); p && *p; ++p) {
            bufts.push_back(*p);
        }
    }
    for (auto buft : bufts) {
        for (auto type : types) {
            ok = check_buffer(buft, cpu, type) && ok;
        }
    }
    ggml_backend_free(cpu);
    ggml_backend_free(hexagon);
    ggml_quantize_free();
    return ok ? 0 : 1;
}
