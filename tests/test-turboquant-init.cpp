#include "ggml.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

static constexpr size_t element_count = 256;
static constexpr size_t worker_count = 8;
static constexpr ggml_type types[] = {
    GGML_TYPE_TBQ3_0, GGML_TYPE_TBQ4_0, GGML_TYPE_TBQ3_0_64, GGML_TYPE_TBQ4_0_64,
    GGML_TYPE_PQ3_0, GGML_TYPE_PQ4_0, GGML_TYPE_PQ3_0_64, GGML_TYPE_PQ4_0_64,
};
using result = std::vector<uint8_t>;

static std::array<float, element_count> make_input() {
    std::array<float, element_count> input;
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(float(i));
    }
    return input;
}

static result quantize(const std::array<float, element_count> & input) {
    result output;
    for (ggml_type type : types) {
        const size_t offset = output.size();
        output.resize(offset + ggml_row_size(type, element_count));
        ggml_quantize_chunk(type, input.data(), output.data() + offset, 0, 1, element_count, nullptr);
    }
    return output;
}

static void await_start(std::atomic<size_t> & ready, std::atomic<bool> & start) {
    ready.fetch_add(1);
    while (!start.load()) {
        std::this_thread::yield();
    }
}

static void start_workers(std::array<std::thread, worker_count> & workers,
                          std::array<result, worker_count> & outputs,
                          const std::array<float, element_count> & input,
                          std::atomic<size_t> & ready, std::atomic<bool> & start) {
    for (size_t i = 0; i < worker_count; ++i) {
        workers[i] = std::thread([&, i] {
            await_start(ready, start);
            outputs[i] = quantize(input);
        });
    }
}

static void release_workers(std::atomic<size_t> & ready, std::atomic<bool> & start) {
    while (ready.load() != worker_count) {
        std::this_thread::yield();
    }
    start.store(true);
}

static void join_workers(std::array<std::thread, worker_count> & workers) {
    for (auto & worker : workers) {
        worker.join();
    }
}

static bool matches_reference(const std::array<result, worker_count> & outputs, const result & reference) {
    for (const auto & output : outputs) {
        if (output != reference) {
            return false;
        }
    }
    return true;
}

int main() {
    const auto input = make_input();
    std::array<std::thread, worker_count> workers;
    std::array<result, worker_count> outputs;
    std::atomic<size_t> ready{0};
    std::atomic<bool> start{false};
    start_workers(workers, outputs, input, ready, start);
    release_workers(ready, start);
    join_workers(workers);
    if (!matches_reference(outputs, quantize(input))) {
        std::fprintf(stderr, "Concurrent first-use quantization differs from serial output\n");
        return 1;
    }
    return 0;
}
