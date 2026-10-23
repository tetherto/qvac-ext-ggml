#include "ggml-backend.h"
#include "ggml-hexagon.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

int main() {
    ggml_backend_load_all();
    auto * device = ggml_backend_dev_by_name("HTP0");
    if (!device) {
        std::puts("SKIP: Hexagon device unavailable");
        return 77;
    }
    auto * backend = ggml_backend_dev_init(device, nullptr);
    CHECK(backend);
    auto query = reinterpret_cast<ggml_backend_hexagon_get_runtime_info_v1_t>(
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device),
                    "ggml_backend_hexagon_get_runtime_info_v1"));
    CHECK(query);
    ggml_backend_hexagon_runtime_info_v1 info, sentinel;
    std::memset(&sentinel, 0x5a, sizeof(sentinel));
    info = sentinel;
    CHECK(!query(nullptr, 0, &info, sizeof(info)));
    CHECK(!query(backend, 0, nullptr, sizeof(info)));
    CHECK(!query(backend, 0, &info, sizeof(info) - 1));
    CHECK(!query(backend, 0, &info, sizeof(info) + 1));
    CHECK(!query(backend, std::numeric_limits<uint64_t>::max(), &info, sizeof(info)));
    CHECK(std::memcmp(&info, &sentinel, sizeof(info)) == 0);
    auto * cpu = ggml_backend_init_by_name("CPU", nullptr);
    CHECK(cpu && !query(cpu, 0, &info, sizeof(info)));
    ggml_backend_free(cpu);

    ggml_backend_hexagon_runtime_info_v1 base, future;
    CHECK(query(backend, 0, &base, sizeof(base)));
    CHECK(query(backend, 32768, &future, sizeof(future)));
    CHECK(base.version == 1 && base.session_count > 0 && base.host_bytes_bound > 0 && base.shared_bytes > 0);
    CHECK(base.max_buffer_bytes > 0 && base.max_batch_vmem_bytes > 0 && base.vtcm_bytes > 0);
    CHECK(base.tensor_guard_bytes == 4096 && base.buffer_alignment == 128 && base.allocation_page_bytes > 0);
    CHECK(future.host_bytes_bound > base.host_bytes_bound && future.shared_bytes == base.shared_bytes);

    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(device, &props);
    CHECK(props.memory_unified);
    auto check_buffer_size = [&](ggml_backend_buffer_type_t type) {
        if (ggml_backend_buft_get_buffer_alloc_size(type, 0) != 0 ||
            ggml_backend_buft_get_buffer_alloc_size(type, 128) != 128 + base.tensor_guard_bytes ||
            ggml_backend_buft_get_buffer_alloc_size(type, SIZE_MAX - 20) != SIZE_MAX) {
            return false;
        }
        auto * buffer = ggml_backend_buft_alloc_buffer(type, 128);
        const bool ok = buffer && ggml_backend_buffer_get_size(buffer) ==
            ggml_backend_buft_get_buffer_alloc_size(type, 128);
        ggml_backend_buffer_free(buffer);
        if (!ok || ggml_backend_buft_alloc_buffer(type, SIZE_MAX - 20) != nullptr) {
            return false;
        }
        return true;
    };
    CHECK(check_buffer_size(ggml_backend_dev_buffer_type(device)));
    auto extras = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device), "ggml_backend_dev_get_extra_bufts"));
    if (extras) {
        for (auto * type = extras(device); type && *type; ++type) {
            CHECK(check_buffer_size(*type));
        }
    }

    // The registry has retained its effective options. Mutating the process
    // environment after initialization must not change reported execution mode
    // or underprice persistent queues (especially PROFILE=3 trace storage).
#ifdef _WIN32
    _putenv_s("GGML_HEXAGON_OPSTAGE", "0");
    _putenv_s("GGML_HEXAGON_OPQUEUE", "1");
    _putenv_s("GGML_HEXAGON_OPBATCH", "1");
    _putenv_s("GGML_HEXAGON_PROFILE", "3");
    _putenv_s("GGML_HEXAGON_OPTRACE", "1");
#else
    setenv("GGML_HEXAGON_OPSTAGE", "0", 1);
    setenv("GGML_HEXAGON_OPQUEUE", "1", 1);
    setenv("GGML_HEXAGON_OPBATCH", "1", 1);
    setenv("GGML_HEXAGON_PROFILE", "3", 1);
    setenv("GGML_HEXAGON_OPTRACE", "1", 1);
#endif
    CHECK(query(backend, 0, &info, sizeof(info)));
    CHECK(std::memcmp(&info, &base, sizeof(info)) == 0);
    std::printf("runtime-info v%u: stage=%u sessions=%u host=%llu projected-host=%llu shared=%llu page=%llu\n",
            base.version, base.op_stage, base.session_count,
            (unsigned long long) base.host_bytes_bound, (unsigned long long) future.host_bytes_bound,
            (unsigned long long) base.shared_bytes, (unsigned long long) base.allocation_page_bytes);
    ggml_backend_free(backend);
    return 0;
}
