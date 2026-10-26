#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>

static bool check_offset(ggml_context * ctx, ggml_backend_dev_t device, size_t offset, bool expected) {
    static constexpr int64_t width = 16;
    ggml_tensor * storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width + offset / sizeof(float));
    ggml_tensor * view = ggml_view_1d(ctx, storage, width, offset);
    ggml_tensor * op = ggml_zero_upsample(ctx, view, 2);
    if (ggml_backend_dev_supports_op(device, op) == expected) {
        return true;
    }
    std::fprintf(stderr, "Unexpected ZERO_UPSAMPLE support for offset %zu\n", offset);
    return false;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_name("Vulkan0");
    if (device == nullptr) {
        return 1;
    }
    const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_dev_buffer_type(device));
    const ggml_init_params params = {ggml_tensor_overhead() * 12, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return 1;
    }
    bool ok = check_offset(ctx, device, 0, true);
    ok &= check_offset(ctx, device, alignment, true);
    if (alignment > sizeof(float)) {
        ok &= check_offset(ctx, device, sizeof(float), false);
    }
    ggml_free(ctx);
    return ok ? 0 : 1;
}
