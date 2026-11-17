#include "../xdna-glue.h"

#include <cstdio>

struct glue_case {
    const char * name;
    const ggml_tensor * op;
    bool expected;
};

template <size_t N> static int check_cases(const glue_case (&cases)[N]) {
    int failures = 0;
    for (const glue_case & c : cases) {
        const bool accepted = xdna_glue_supports_op(c.op);
        const bool ok = accepted == c.expected;
        std::printf("%s: %s\n", ok ? "ok" : "FAIL", c.name);
        failures += ok ? 0 : 1;
    }
    return failures;
}

int main() {
    constexpr size_t BUFFER_BYTES = 64 * 1024;
    ggml_context * ctx = ggml_init({BUFFER_BYTES, nullptr, true});
    if (!ctx) {
        return 1;
    }
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2);
    ggml_tensor * rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 2);
    ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8, 4);
    ggml_tensor * biased = ggml_soft_max_ext_back(ctx, a, b, 1.0f, 1.0f);
    ggml_tensor * unbiased = ggml_soft_max_ext_back(ctx, a, b, 1.0f, 0.0f);
    const glue_case cases[] = {
        { "CPU-rejected softmax backward bias is rejected by glue", biased, false },
        { "CPU-supported softmax backward remains eligible", unbiased, true },
        { "contiguous F32 addition remains eligible", ggml_add(ctx, a, b), true },
        { "F16 cache row writes remain eligible", ggml_set_rows(ctx, cache, a, rows), true },
        { "metadata-only tensors remain excluded", a, false },
        { "tensor views remain excluded", ggml_transpose(ctx, a), false },
    };
    const int failures = check_cases(cases);
    ggml_free(ctx);
    return failures ? 1 : 0;
}
