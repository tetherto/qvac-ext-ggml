#include "mean-ops.h"

#include <cstdio>
#include <initializer_list>
#include <vector>

#define CHECK(value) do { if (!(value)) { \
    std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #value); return 1; } } while (0)

static htp_mean_tensor tensor(uint64_t width, uint64_t rows, uint64_t c, uint64_t b, uint64_t padding = 0) {
    htp_mean_tensor t = {};
    t.ne[0] = width; t.ne[1] = rows; t.ne[2] = c; t.ne[3] = b;
    t.nb[0] = sizeof(float);
    t.nb[1] = (width + padding) * sizeof(float);
    t.nb[2] = t.nb[1] * rows;
    t.nb[3] = t.nb[2] * c;
    t.size = t.nb[3] * b - padding * sizeof(float);
    return t;
}

int main() {
    htp_mean_layout layout;
    CHECK(!htp_mean_span_fits(UINT32_MAX - 3, 16, UINT32_MAX));
    CHECK(htp_mean_span_fits(UINT32_MAX - 3, 16, UINT64_MAX));
    CHECK(!htp_mean_span_fits(UINT64_MAX - 3, 16, UINT64_MAX));
    for (uint32_t width : {1u, 2u, 12u, 16u, 24u, 31u, 32u, 33u, 63u, 64u, 65u, 127u, 128u, 129u}) {
        for (uint32_t padding : {0u, 1u, 31u}) {
            auto x = tensor(width, 5, 3, 2, padding), y = tensor(1, 5, 3, 2);
            CHECK(htp_mean_validate(&x, &y, 6, &layout));
            CHECK(layout.width == width && layout.rows == 30 && layout.threads == 6);
            CHECK(layout.stride == (width + padding) * sizeof(float));
            for (uint32_t row = 0; row < layout.rows; ++row) {
                const uint64_t tail_end = uint64_t(row) * layout.stride + ((width + 31) / 32) * 128;
                CHECK(htp_mean_direct_tail(&layout, row) == (tail_end <= x.size));
                if (!htp_mean_direct_tail(&layout, row)) {
                    const uint64_t copy_start = uint64_t(row) * layout.stride + (width / 32) * 128;
                    CHECK(copy_start + (width % 32) * sizeof(float) <= x.size);
                }
            }
            CHECK(!htp_mean_direct_tail(&layout, layout.rows));
            x.address = 0x1004; y.address = 0x1000004;
            CHECK(htp_mean_validate(&x, &y, 6, &layout));
        }
    }
    // Tag every output row, including the first +4-byte destination view.
    // Exhaust worker boundaries and partial/full output vectors independently
    // of input width, so adjacent workers cannot overwrite one another.
    for (uint32_t rows : {1u, 5u, 31u, 32u, 33u, 63u, 64u, 65u, 127u, 128u, 129u, 192u, 198u, 5292u}) {
        for (uint32_t maximum_threads : {1u, 6u, 10u}) {
            auto source = tensor(33, rows, 1, 1), output = tensor(1, rows, 1, 1);
            source.address = 0x1004; output.address = 0x1000004;
            CHECK(htp_mean_validate(&source, &output, maximum_threads, &layout));
            const uint32_t rows_per_thread = (layout.rows - 1) / layout.threads + 1;
            std::vector<uint32_t> writes(rows, 0);
            std::vector<float> storage(rows + 2, -1000.0f);
            float * destination = storage.data() + 1;
            for (uint32_t worker = 0; worker < layout.threads; ++worker) {
                const uint32_t begin = worker * rows_per_thread;
                const uint32_t end = begin + rows_per_thread < rows ? begin + rows_per_thread : rows;
                for (uint32_t row = begin; row < end;) {
                    const uint32_t valid_rows = htp_mean_block_rows(row, end);
                    CHECK(valid_rows > 0 && valid_rows <= HTP_MEAN_LANES && row + valid_rows <= end);
                    CHECK(valid_rows == HTP_MEAN_LANES || row + valid_rows == end);
                    for (uint32_t lane = 0; lane < valid_rows; ++lane) {
                        destination[row + lane] = float(row + lane) + 0.25f;
                        ++writes[row + lane];
                    }
                    row += valid_rows;
                }
            }
            CHECK(storage.front() == -1000.0f && storage.back() == -1000.0f);
            for (uint32_t row = 0; row < rows; ++row) {
                CHECK(writes[row] == 1 && destination[row] == float(row) + 0.25f);
            }
        }
    }
    CHECK(htp_mean_block_rows(33, 32) == 0 && htp_mean_block_rows(32, 32) == 0);
    CHECK(htp_mean_block_rows(UINT32_MAX - 31, UINT32_MAX) == 31);
    auto x = tensor(65, 63, 12, 7), y = tensor(1, 63, 12, 7);
    const auto original_x = x, original_y = y;
    const auto valid = [&]() { return htp_mean_validate(&x, &y, 6, &layout); };
    CHECK(valid());
    for (int d = 0; d < 4; ++d) {
        x.ne[d] = 0; CHECK(!valid()); x = original_x;
        x.ne[d] = UINT64_MAX; CHECK(!valid()); x = original_x;
        x.nb[d] = UINT64_MAX; CHECK(!valid()); x = original_x;
        y.nb[d] += 4; CHECK(!valid()); y = original_y;
    }
    x.nb[2] += 4; CHECK(!valid()); x = original_x;
    x.nb[3] += 4; CHECK(!valid()); x = original_x;
    --x.size; CHECK(!valid()); x = original_x;
    --y.size; CHECK(!valid()); y = original_y;
    x.type = 1; CHECK(!valid()); x = original_x;
    y.type = 1; CHECK(!valid()); y = original_y;
    y.ne[0] = 2; CHECK(!valid()); y = original_y;
    x.address = 1; CHECK(!valid()); x = original_x;
    x.address = UINTPTR_MAX - 3; CHECK(!valid()); x = original_x;
    x.address = 0x1000; y.address = 0x1004; CHECK(!valid());
    x = original_x; y = original_y;
    CHECK(!htp_mean_validate(&x, &y, 0, &layout));
    CHECK(!htp_mean_validate(&x, &y, 11, &layout));
    CHECK(!htp_mean_validate(&x, &y, 6, nullptr));
    auto huge = tensor(1, 1, 1, 1);
    huge.ne[1] = UINT32_MAX; huge.ne[2] = UINT32_MAX;
    CHECK(!htp_mean_tensor_bytes(&huge, true, &x.size));
    std::puts("MEAN F32 shape/stride/span/alias and bounded-tail guards passed");
    std::puts("MEAN F32 packed row blocks, worker boundaries and exact output tails passed");
    return 0;
}
