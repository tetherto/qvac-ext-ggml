#include "../xdna-weight-cache.h"

#include <cstdio>
#include <initializer_list>

static bool check_fingerprint() {
    unsigned char data[98]{};
    const uint64_t original = xdna_weight_fingerprint(data + 1, 95);
    for (const size_t index : { 1, 49, 95 }) {
        data[index] = 1;
        if (xdna_weight_fingerprint(data + 1, 95) == original) {
            return false;
        }
        data[index] = 0;
    }
    return xdna_weight_fingerprint(data + 1, 95) == original;
}

static bool check_rewrite(ggml_tensor * tensor) {
    const xdna_weight_signature before(tensor);
    if (!(before == xdna_weight_signature(tensor))) {
        return false;
    }
    static_cast<unsigned char *>(tensor->data)[ggml_nbytes(tensor) - 1] ^= 1;
    if (before == xdna_weight_signature(tensor)) {
        return false;
    }
    const xdna_weight_signature rewritten(tensor);
    tensor->type = GGML_TYPE_BF16;
    return !(rewritten == xdna_weight_signature(tensor));
}

struct cache_case {
    const char * what;
    const ggml_tensor * tensor;
    bool expected;
};

int main() {
    const ggml_init_params params = { ggml_tensor_overhead() * 5, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 4096);
    if (!ctx || !buffer) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return 1;
    }
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8, 8);
    weight->buffer = buffer;
    weight->data = ggml_backend_buffer_get_base(buffer);
    std::memset(weight->data, 0, ggml_nbytes(weight));
    ggml_tensor * view = ggml_view_2d(ctx, weight, 8, 4, weight->nb[1], 0);
    ggml_tensor * unallocated = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 8, 8);
    ggml_tensor * intermediate = ggml_add(ctx, weight, weight);
    intermediate->buffer = buffer;

    int failures = 0;
    for (const auto usage : { GGML_BACKEND_BUFFER_USAGE_ANY, GGML_BACKEND_BUFFER_USAGE_COMPUTE,
                             GGML_BACKEND_BUFFER_USAGE_WEIGHTS }) {
        ggml_backend_buffer_set_usage(buffer, usage);
        const bool cacheable = usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        const cache_case cases[] = {
            { "leaf buffer classification", weight, cacheable },
            { "view follows its source", view, cacheable },
            { "unallocated operand is mutable", unallocated, false },
            { "computed operand is mutable", intermediate, false },
            { "null operand is mutable", nullptr, false },
        };
        for (const auto & item : cases) {
            if (xdna_weight_is_cacheable(item.tensor) != item.expected) {
                std::fprintf(stderr, "FAIL: %s (usage %d)\n", item.what, (int) usage);
                failures++;
            }
        }
        if (!cacheable && xdna_weight_signature(weight) == xdna_weight_signature(weight)) {
            std::fprintf(stderr, "FAIL: mutable operand signature permits cache reuse\n");
            failures++;
        }
    }
    if (!check_fingerprint() || !check_rewrite(weight)) {
        std::fprintf(stderr, "FAIL: WEIGHTS rewrite or operand identity did not invalidate its signature\n");
        failures++;
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    if (failures) {
        return 1;
    }
    std::puts("test-xdna-weight-cache: all passed");
    return 0;
}
