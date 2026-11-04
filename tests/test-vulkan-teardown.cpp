#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>

static constexpr size_t buffer_bytes = 1 << 20;
static constexpr const char * keep_buffer_flag = "--keep-buffer";

static ggml_backend_buffer_t alloc_on(ggml_backend_t backend) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), buffer_bytes);
}

static bool run_on(ggml_backend_dev_t device, bool keep_buffer) {
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (backend == nullptr) {
        std::fprintf(stderr, "Vulkan backend init failed\n");
        return false;
    }
    ggml_backend_buffer_t buffer = alloc_on(backend);
    const bool allocated = buffer != nullptr;
    if (allocated && !keep_buffer) {
        ggml_backend_buffer_free(buffer);
    }
    ggml_backend_free(backend);
    return allocated;
}

static bool wants_kept_buffer(int argc, char ** argv) {
    return argc > 1 && std::strcmp(argv[1], keep_buffer_flag) == 0;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_name("Vulkan0");
    if (device == nullptr) {
        return 1;
    }
    return run_on(device, wants_kept_buffer(argc, argv)) ? 0 : 1;
}
