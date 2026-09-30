#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const size_t mib = 1024ull * 1024;
static const size_t chunk_bytes = 256 * mib;
static const size_t max_overcommit_total = 32 * 1024 * mib;

struct device_memory {
    size_t free;
    size_t total;
};

static device_memory query_memory(ggml_backend_dev_t dev) {
    device_memory memory = { 0, 0 };
    ggml_backend_dev_memory(dev, &memory.free, &memory.total);
    return memory;
}

static bool check(bool ok, const char * device, const char * what) {
    if (!ok) {
        fprintf(stderr, "  %s: %s\n", device, what);
    }
    return ok;
}

static bool free_is_within_total(ggml_backend_dev_t dev, const char * what) {
    device_memory memory = query_memory(dev);
    return check(memory.free <= memory.total, ggml_backend_dev_name(dev), what);
}

static bool can_overcommit(ggml_backend_dev_t dev) {
    device_memory memory = query_memory(dev);
    return ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU &&
           memory.total > 0 && memory.total <= max_overcommit_total;
}

static std::vector<ggml_backend_buffer_t> allocate_past(ggml_backend_buffer_type_t buft, size_t target) {
    std::vector<ggml_backend_buffer_t> buffers;
    const size_t chunk = std::min(chunk_bytes, ggml_backend_buft_get_max_size(buft));
    size_t allocated = 0;
    while (allocated <= target) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, chunk);
        if (buffer == nullptr) {
            break;
        }
        buffers.push_back(buffer);
        allocated += chunk;
    }
    return buffers;
}

static size_t total_size(const std::vector<ggml_backend_buffer_t> & buffers) {
    size_t size = 0;
    for (ggml_backend_buffer_t buffer : buffers) {
        size += ggml_backend_buffer_get_size(buffer);
    }
    return size;
}

static void free_buffers(std::vector<ggml_backend_buffer_t> & buffers) {
    for (ggml_backend_buffer_t buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
    buffers.clear();
}

static bool probe_overcommitted(ggml_backend_dev_t dev) {
    const size_t total = query_memory(dev).total;
    std::vector<ggml_backend_buffer_t> buffers = allocate_past(ggml_backend_dev_buffer_type(dev), total);
    printf("  %-40s allocated %zu MiB against a %zu MiB total\n",
           ggml_backend_dev_name(dev), total_size(buffers) / mib, total / mib);
    bool ok = free_is_within_total(dev, "free exceeds total after allocating past the reported total");
    free_buffers(buffers);
    return ok;
}

static bool probe_device(size_t index) {
    ggml_backend_dev_t dev = ggml_backend_dev_get(index);
    bool ok = free_is_within_total(dev, "free exceeds total before allocating");
    if (can_overcommit(dev)) {
        ok = probe_overcommitted(dev) && ok;
    }
    printf("  %-40s %s\n", ggml_backend_dev_name(dev), ok ? "OK" : "FAIL");
    return ok;
}

static bool probe_all_devices() {
    bool ok = true;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ok = probe_device(i) && ok;
    }
    return ok;
}

int main() {
    ggml_backend_load_all();
    printf("device memory queries on %zu device(s)\n", ggml_backend_dev_count());
    return probe_all_devices() ? EXIT_SUCCESS : EXIT_FAILURE;
}
