#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-vulkan/ggml-vulkan-capacity.h"

#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>

static int allocation_calls = 0;

static const char * test_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "capacity-test";
}

static ggml_backend_buffer_t test_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_UNUSED(buft);
    GGML_UNUSED(size);
    ++allocation_calls;
    return nullptr;
}

static size_t test_buft_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 16;
}

static size_t test_buft_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 4096;
}

static size_t test_buft_max_capacity(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 8192;
}

static void * test_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    return std::strcmp(name, "ggml_backend_get_buffer_capacity") == 0
        ? (void *) test_buft_max_capacity
        : nullptr;
}

static ggml_backend_buffer_type make_test_buft(bool finite) {
    return {
        /* .iface   = */ {
            /* .get_name       = */ test_buft_name,
            /* .alloc_buffer   = */ test_buft_alloc,
            /* .get_alignment  = */ test_buft_alignment,
            /* .get_max_size   = */ finite ? test_buft_max_size : nullptr,
            /* .get_alloc_size = */ nullptr,
            /* .is_host        = */ nullptr,
        },
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
}

int main() {
    ggml_backend_buffer_type finite = make_test_buft(true);

    assert(ggml_backend_buft_get_max_capacity(&finite) == 4096);
    assert(ggml_backend_buft_is_size_supported(&finite, 0));
    assert(ggml_backend_buft_is_size_supported(&finite, 4096));
    assert(!ggml_backend_buft_is_size_supported(&finite, 4097));
    assert(allocation_calls == 0);

    ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ {
            /* .get_name         = */ nullptr,
            /* .get_device_count = */ nullptr,
            /* .get_device       = */ nullptr,
            /* .get_proc_address = */ test_reg_get_proc_address,
        },
        /* .context     = */ nullptr,
    };
    ggml_backend_device device = {
        /* .iface   = */ {},
        /* .reg     = */ &reg,
        /* .context = */ nullptr,
    };
    finite.device = &device;

    assert(ggml_backend_buft_get_max_size(&finite) == 4096);
    assert(ggml_backend_buft_get_max_capacity(&finite) == 8192);
    assert(ggml_backend_buft_is_size_supported(&finite, 8192));
    assert(!ggml_backend_buft_is_size_supported(&finite, 8193));
    assert(allocation_calls == 0);

    ggml_backend_buffer_type unbounded = make_test_buft(false);

    assert(ggml_backend_buft_get_max_capacity(&unbounded) == SIZE_MAX);
    assert(ggml_backend_buft_is_size_supported(&unbounded, SIZE_MAX));
    assert(allocation_calls == 0);

    const uint64_t gib = UINT64_C(1024) * 1024 * 1024;
    assert(ggml_vk_buffer_capacity(16 * gib, 12 * gib, 4 * gib - 1, false) == 4 * gib - 1);
    assert(ggml_vk_buffer_capacity(16 * gib, 12 * gib, 4 * gib - 1, true) == 12 * gib);
    assert(ggml_vk_buffer_capacity(8 * gib, 16 * gib, 4 * gib - 1, true) == 8 * gib);
    assert(ggml_vk_buffer_capacity(16 * gib, 8 * gib, 4 * gib - 1, true) == 8 * gib);

    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
        const size_t capacity = ggml_backend_buft_get_max_capacity(buft);

        assert(capacity > 0);
        assert(ggml_backend_buft_is_size_supported(buft, capacity));
        if (capacity < SIZE_MAX) {
            assert(!ggml_backend_buft_is_size_supported(buft, capacity + 1));
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend != nullptr) {
            assert(ggml_backend_get_max_buffer_capacity(backend) == capacity);
            assert(ggml_backend_is_buffer_size_supported(backend, capacity));
            ggml_backend_free(backend);
        }

        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
        if (host_buft != nullptr && std::strcmp(ggml_backend_buft_name(host_buft), "Vulkan_Host") == 0) {
            assert(ggml_backend_buft_get_max_capacity(host_buft) == SIZE_MAX);
            assert(ggml_backend_buft_is_size_supported(host_buft, SIZE_MAX));
        }

        std::printf("%s capacity: %" PRIu64 " bytes\n",
                    ggml_backend_dev_name(dev), (uint64_t) capacity);
    }

    assert(allocation_calls == 0);
    return 0;
}
