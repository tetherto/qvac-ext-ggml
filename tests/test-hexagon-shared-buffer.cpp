// Exercise the production RPC buffer constructor without a DSP or the SDK.
// Only the session and transport API boundaries are faked.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class failure { none, allocation, fd, mapping };
enum fastrpc_map_flags { FASTRPC_MAP_FD, FASTRPC_MAP_FD_DELAYED };
constexpr int RPCMEM_HEAP_ID_SYSTEM = 1;
constexpr int RPCMEM_DEFAULT_FLAGS = 0;

failure fail_at = failure::none;
std::vector<std::string> events;
std::set<void *> allocations;
fastrpc_map_flags last_map_flag = FASTRPC_MAP_FD_DELAYED;
bool arguments_ok = true;

struct ggml_hexagon_session {
    uint32_t domain_id = 3;
    uint64_t handle = 42;
    const char * c_name() const { return "HTP-test"; }
};

void * rpcmem_alloc2(int heap, int flags, size_t size) {
    events.push_back("allocate");
    arguments_ok &= heap == RPCMEM_HEAP_ID_SYSTEM && flags == RPCMEM_DEFAULT_FLAGS && size == 128;
    if (fail_at == failure::allocation) return nullptr;
    void * ptr = std::malloc(size);
    if (!ptr) throw std::bad_alloc();
    allocations.insert(ptr);
    return ptr;
}

int rpcmem_to_fd(void * ptr) {
    events.push_back("fd");
    arguments_ok &= allocations.count(ptr) == 1;
    return fail_at == failure::fd ? -1 : 7;
}

int fastrpc_mmap(uint32_t domain, int fd, void * ptr, size_t offset, size_t size, fastrpc_map_flags flags) {
    events.push_back("map");
    arguments_ok &= domain == 3 && fd == 7 && allocations.count(ptr) == 1 && offset == 0 && size == 128;
    last_map_flag = flags;
    return fail_at == failure::mapping ? -1 : 0;
}

int htp_iface_munmap(uint64_t handle, int fd) {
    events.push_back("drop-dsp-reference");
    arguments_ok &= handle == 42 && fd == 7;
    return 0;
}

int fastrpc_munmap(uint32_t domain, int fd, void * ptr, size_t size) {
    events.push_back("unmap");
    arguments_ok &= domain == 3 && fd == 7 && allocations.count(ptr) == 1 && size == 128;
    return 0;
}

void rpcmem_free(void * ptr) {
    events.push_back("free");
    const bool owned = allocations.erase(ptr) == 1;
    arguments_ok &= owned;
    if (owned) std::free(ptr);
}

#define HEX_VERBOSE(...) ((void) 0)
#define GGML_LOG_ERROR(...) ((void) 0)
#include "htp-shared-buffer.h"
#undef GGML_LOG_ERROR
#undef HEX_VERBOSE

int failures = 0;

void check(bool ok, const char * label) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", label);
        ++failures;
    }
}

void reset(failure next) {
    // Let a failing baseline finish all cases while reporting each leaked
    // allocation; this cleanup is outside the code under test.
    for (void * ptr : allocations) std::free(ptr);
    allocations.clear();
    events.clear();
    arguments_ok = true;
    fail_at = next;
}

void success_case(const char * label, bool pinned) {
    reset(failure::none);
    ggml_hexagon_session session;
    {
        ggml_hexagon_shared_buffer buffer(&session, 128, pinned);
        check(buffer.base != nullptr && buffer.fd == 7 && buffer.size == 128 && buffer.mapped,
              "successful constructor state");
        check(buffer.pinned == pinned && buffer.sess == &session, "buffer ownership");
        check(last_map_flag == (pinned ? FASTRPC_MAP_FD : FASTRPC_MAP_FD_DELAYED), "mapping mode");
    }
    const std::vector<std::string> expected = pinned
        ? std::vector<std::string>{"allocate", "fd", "map", "unmap", "free"}
        : std::vector<std::string>{"allocate", "fd", "map", "drop-dsp-reference", "unmap", "free"};
    check(events == expected, "successful destruction order");
    check(allocations.empty() && arguments_ok, "successful destruction ownership");
    std::printf("%s: live=%zu transport=%s\n", label, allocations.size(), arguments_ok ? "OK" : "FAIL");
}

void failure_case(const char * label, failure where, const char * error,
                  const std::vector<std::string> & expected) {
    reset(where);
    bool threw = false;
    ggml_hexagon_session session;
    try {
        ggml_hexagon_shared_buffer buffer(&session, 128);
    } catch (const std::runtime_error & ex) {
        threw = true;
        check(std::string(ex.what()).find(error) != std::string::npos, "original failure preserved");
    }
    check(threw, "constructor must propagate failure");
    check(events == expected, "failure cleanup order");
    check(allocations.empty(), "failed constructor must release RPC allocation");
    check(arguments_ok, "failure transport arguments");
    std::printf("%s: threw=%d live=%zu transport=%s\n", label, threw, allocations.size(), arguments_ok ? "OK" : "FAIL");
    success_case("recovery after failure", false);
}

} // namespace

int main() {
    failure_case("allocation failure", failure::allocation, "rpcmem_alloc", {"allocate"});
    failure_case("FD failure", failure::fd, "rpcmem_to_fd", {"allocate", "fd", "free"});
    failure_case("mapping failure", failure::mapping, "fastrpc_mmap", {"allocate", "fd", "map", "free"});
    success_case("normal pinned destruction", true);
    reset(failure::none);
    std::printf("test-hexagon-shared-buffer: %d failures\n", failures);
    return failures ? 1 : 0;
}
