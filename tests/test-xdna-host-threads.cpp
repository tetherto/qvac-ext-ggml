#include "../src/ggml-xdna/xdna-host-threads.h"

#include <cstdio>

struct thread_case {
    const char * name;
    int configured_threads;
    int default_threads;
    int expected_threads;
};

int main() {
    const thread_case cases[] = {
        { "disabled host decode still permits host prefill", 0, 16, 16 },
        { "negative configured count uses the default", -1, 8, 8 },
        { "explicit count overrides the default", 4, 16, 4 },
        { "single-thread host prefill remains valid", 1, 16, 1 },
        { "missing default has a positive fallback", 0, 0, 1 },
        { "invalid default has a positive fallback", 0, -1, 1 },
    };
    int failures = 0;
    for (const thread_case & c : cases) {
        const int actual = xdna_host_compute_thread_count(c.configured_threads, c.default_threads);
        const bool ok = actual == c.expected_threads && actual > 0;
        std::printf("%s: %s (threads=%d)\n", ok ? "ok" : "FAIL", c.name, actual);
        failures += ok ? 0 : 1;
    }
    return failures ? 1 : 0;
}
