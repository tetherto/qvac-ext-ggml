#include "energy-ops.h"

#include <algorithm>
#include <cstdio>
#include <initializer_list>
#include <vector>

#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #value); return 1; } } while (0)

static htp_energy_tensor rows(uint64_t width, uint64_t r, uint64_t c, uint64_t b, uint64_t padding = 0) {
    htp_energy_tensor t = {};
    t.ne[0] = width; t.ne[1] = r; t.ne[2] = c; t.ne[3] = b;
    t.nb[0] = 4; t.nb[1] = (width + padding) * 4;
    t.nb[2] = t.nb[1] * r; t.nb[3] = t.nb[2] * c;
    t.size = t.nb[3] * b - padding * 4;
    return t;
}

static htp_energy_tensor planes(uint64_t channels, uint64_t spatial, uint64_t c, uint64_t b) {
    htp_energy_tensor t = rows(channels, spatial, c, b);
    t.nb[0] = spatial * 4; t.nb[1] = 4;
    t.nb[2] = spatial * channels * 4; t.nb[3] = t.nb[2] * c;
    t.size = t.nb[3] * b;
    return t;
}

static bool mapping(const htp_energy_tensor & x, const htp_energy_layout & layout) {
    std::vector<unsigned> writes(layout.rows, 0);
    for (uint32_t worker = 0; worker < layout.threads; ++worker) {
        for (uint32_t job = worker; job < layout.jobs; job += layout.threads) {
            uint32_t start, valid, group = 0, spatial = 0;
            if (layout.mode == HTP_ENERGY_ROWS) {
                start = job * 32; valid = std::min(32u, layout.rows - start);
            } else {
                const uint32_t tiles = (layout.spatial + 31) / 32;
                group = job / tiles; spatial = (job % tiles) * 32;
                start = group * layout.spatial + spatial;
                valid = std::min(32u, layout.spatial - spatial);
            }
            if (!valid || valid > 32 || start + valid > layout.rows) return false;
            for (uint32_t lane = 0; lane < valid; ++lane) {
                const uint32_t row = start + lane;
                ++writes[row];
                for (uint32_t item = 0; item < layout.width; ++item) {
                    const uint64_t offset = layout.mode == HTP_ENERGY_ROWS ?
                        uint64_t(row) * layout.row_stride + item * 4 :
                        uint64_t(group) * layout.group_stride + uint64_t(item) * layout.channel_stride + (spatial + lane) * 4;
                    if (offset + 4 > x.size) return false;
                }
            }
            if (layout.mode == HTP_ENERGY_CHANNELS && valid == 32) {
                for (uint32_t item = 0; item < layout.width; ++item) {
                    const uint64_t offset = uint64_t(group) * layout.group_stride + uint64_t(item) * layout.channel_stride + spatial * 4;
                    if (offset + 128 > x.size || spatial + 32 > layout.spatial) return false;
                }
            }
        }
    }
    return std::all_of(writes.begin(), writes.end(), [](unsigned count) { return count == 1; });
}

int main() {
    htp_energy_layout layout;
    for (uint32_t width : {1u, 2u, 12u, 24u, 31u, 32u, 33u, 63u, 64u, 65u, 127u, 128u, 129u}) {
        for (uint32_t r : {1u, 5u, 31u, 32u, 33u, 63u, 65u}) {
            for (uint32_t padding : {0u, 1u, 31u}) {
                auto x = rows(width, r, 3, 2, padding), y = rows(1, r, 3, 2);
                x.address = 0x1004; y.address = 0x1000004;
                for (uint32_t threads : {1u, 6u, 10u}) {
                    CHECK(htp_energy_validate(&x, &y, threads, &layout));
                    CHECK(layout.mode == HTP_ENERGY_ROWS && mapping(x, layout));
                    for (uint32_t row = 0; row < layout.rows; ++row) {
                        const uint64_t end = uint64_t(row) * x.nb[1] + ((width + 31) / 32) * 128;
                        CHECK(htp_energy_direct_row_tail(&layout, row) == (end <= x.size));
                    }
                }
            }
        }
    }
    for (uint32_t channels : {1u, 3u, 12u, 16u, 24u, 32u, 64u, 128u}) {
        for (uint32_t spatial : {1u, 31u, 32u, 33u, 63u, 64u, 65u, 2079u}) {
            auto x = planes(channels, spatial, 2, 3), y = rows(1, spatial, 2, 3);
            for (uint32_t threads : {1u, 6u, 10u}) {
                CHECK(htp_energy_validate(&x, &y, threads, &layout));
                CHECK(mapping(x, layout));
            }
        }
    }
    auto x = planes(24, 65, 3, 2), y = rows(1, 65, 3, 2);
    const auto original_x = x, original_y = y;
    const auto valid = [&]() { return htp_energy_validate(&x, &y, 6, &layout); };
    CHECK(valid());
    for (int d = 0; d < 4; ++d) {
        x.ne[d] = 0; CHECK(!valid()); x = original_x;
        x.ne[d] = UINT64_MAX; CHECK(!valid()); x = original_x;
        x.nb[d] = UINT64_MAX; CHECK(!valid()); x = original_x;
        y.nb[d] += 4; CHECK(!valid()); y = original_y;
    }
    x.nb[0] += 4; CHECK(!valid()); x = original_x;
    --x.size; CHECK(!valid()); x = original_x;
    --y.size; CHECK(!valid()); y = original_y;
    x.type = 1; CHECK(!valid()); x = original_x;
    y.type = 1; CHECK(!valid()); y = original_y;
    x.address = 1; CHECK(!valid()); x = original_x;
    x.address = UINTPTR_MAX - 3; CHECK(!valid()); x = original_x;
    x.address = 0x1000; y.address = 0x1004; CHECK(!valid());
    x = original_x; y = original_y;
    CHECK(!htp_energy_validate(&x, &y, 0, &layout));
    CHECK(!htp_energy_validate(&x, &y, 11, &layout));
    CHECK(!htp_energy_validate(&x, &y, 6, nullptr));
    CHECK(!htp_energy_span_fits(UINT32_MAX - 3, 16, UINT32_MAX));
    CHECK(htp_energy_span_fits(UINT32_MAX - 3, 16, UINT64_MAX));
    CHECK(!htp_energy_span_fits(UINT64_MAX - 3, 16, UINT64_MAX));
    x = planes(129, 33, 1, 1); y = rows(1, 33, 1, 1); CHECK(!valid());
    std::puts("Energy F32 row/channel layout, span/alias, tail and job ownership guards passed");
    return 0;
}
