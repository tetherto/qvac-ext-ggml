#pragma once

#include <algorithm>

static inline int xdna_host_compute_thread_count(int configured_threads, int default_threads) {
    return configured_threads > 0 ? configured_threads : std::max(1, default_threads);
}
