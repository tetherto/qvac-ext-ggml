#include "conv-2d-dw-guards.h"

#include <array>
#include <cstdio>
#include <initializer_list>

#define CHECK(value)                                                                                                   \
    do {                                                                                                               \
        if (!(value)) {                                                                                                \
            std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #value);                                              \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

int main() {
    for (uint32_t kw : {1u, 2u, 3u, 5u, 7u})
        for (uint32_t kh : {0u, 1u, 2u, 3u, 4u, 5u})
            for (int32_t sy : {0, 1, 2, 3, 4}) {
                const bool expected = (kw == 3 || kw == 5) && kh >= 1 && kh <= 4 && sy >= 1 && sy <= 3;
                CHECK(htp_conv_dw_planar_stride1_kw35_eligible(false, 4, 4, kw, kh, 1, sy, 1, 1, true) == expected);
                CHECK(!htp_conv_dw_planar_stride1_kw35_eligible(false, 4, 4, kw, kh, 1, sy, 1, 1, false));
            }
    CHECK(!htp_conv_dw_planar_stride1_kw35_eligible(true, 4, 4, 5, 1, 1, 1, 1, 1, true));
    CHECK(!htp_conv_dw_planar_stride1_kw35_eligible(false, 8, 4, 5, 1, 1, 1, 1, 1, true));
    CHECK(!htp_conv_dw_planar_stride1_kw35_eligible(false, 4, 8, 5, 1, 1, 1, 1, 1, true));
    CHECK(!htp_conv_dw_planar_stride1_kw35_eligible(false, 4, 4, 5, 1, 2, 1, 1, 1, true));
    CHECK(!htp_conv_dw_planar_stride1_kw35_eligible(false, 4, 4, 5, 1, 1, 1, 2, 1, true));
    CHECK(!htp_conv_dw_planar_stride1_kw35_eligible(false, 4, 4, 5, 1, 1, 1, 1, 2, true));
    for (int64_t ix : {INT64_MIN, INT64_MAX}) {
        uint32_t source, target, count;
        htp_conv_dw_stride1_window(65, ix, 36, &source, &target, &count);
        CHECK(source == 0 && target == 0 && count == 0);
    }
    for (uint32_t width : {1u, 2u, 31u, 32u, 33u, 63u, 64u, 65u, 127u, 128u, 129u})
        for (uint32_t kw : {3u, 5u})
            for (uint32_t pad : {0u, kw / 2, 4u, 20u}) {
                if (width + 2 * pad < kw)
                    continue;
                const uint32_t outputs = width + 2 * pad - kw + 1;
                for (uint32_t ox = 0; ox < outputs; ox += 32) {
                    const uint32_t lanes = outputs - ox < 32 ? outputs - ox : 32;
                    const uint32_t span = lanes + kw - 1;
                    CHECK(span <= 36);
                    const int64_t ix = (int64_t)ox - pad;
                    uint32_t source, target, count;
                    htp_conv_dw_stride1_window(width, ix, span, &source, &target, &count);
                    CHECK(count == 0 || (source + count <= width && target + count <= span));
                    std::array<uint32_t, 64> tile = {};
                    for (uint32_t i = 0; i < count; ++i) {
                        const uint32_t index = source + i;
                        // Distinct per-position tags and preserved signed zero.
                        tile[target + i] = index % 7 ? 0x3f800000u + index : 0x80000000u;
                    }
                    for (uint32_t kx = 0; kx < kw; ++kx)
                        for (uint32_t lane = 0; lane < lanes; ++lane) {
                            const int64_t expected_x = ix + lane + kx;
                            uint32_t expected = 0;
                            if (expected_x >= 0 && expected_x < width) {
                                expected = expected_x % 7 ? 0x3f800000u + expected_x : 0x80000000u;
                            }
                            CHECK(tile[lane + kx] == expected);
                        }
                }
            }
    std::puts("Depthwise KW3/KW5 stride1 selection/window/tap guards passed");
    return 0;
}
