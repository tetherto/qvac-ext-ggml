#include "conv-2d-dw-guards.h"

#include <cstdio>
#include <initializer_list>

#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #value); return 1; } } while (0)

int main() {
    auto accepts = [](uint32_t kh, int sy, bool f32) {
        return htp_conv_dw_planar_stride2_kw3_eligible(false, 4, 4, 3, kh, 2, sy, 1, 1, f32);
    };
    for (uint32_t kh : {1u, 2u, 3u, 4u, 7u}) {
        for (int sy : {1, 2, 3}) {
            CHECK(accepts(kh, sy, true));
            CHECK(accepts(kh, sy, false) == (kh == 3 && sy == 2));
        }
    }
    CHECK(!accepts(0, 1, true));
    CHECK(!accepts(2, 0, true));
    CHECK(!accepts(2, -1, true));
    CHECK(!htp_conv_dw_planar_stride2_kw3_eligible(true, 4, 4, 3, 2, 2, 1, 1, 1, true));
    CHECK(!htp_conv_dw_planar_stride2_kw3_eligible(false, 8, 4, 3, 2, 2, 1, 1, 1, true));
    CHECK(!htp_conv_dw_planar_stride2_kw3_eligible(false, 4, 8, 3, 2, 2, 1, 1, 1, true));
    CHECK(!htp_conv_dw_planar_stride2_kw3_eligible(false, 4, 4, 5, 2, 2, 1, 1, 1, true));
    CHECK(!htp_conv_dw_planar_stride2_kw3_eligible(false, 4, 4, 3, 2, 1, 1, 1, 1, true));
    CHECK(!htp_conv_dw_planar_stride2_kw3_eligible(false, 4, 4, 3, 2, 2, 1, 2, 1, true));
    CHECK(!htp_conv_dw_planar_stride2_kw3_eligible(false, 4, 4, 3, 2, 2, 1, 1, 2, true));
    std::puts("Depthwise KW3 stride2 layout/type/vertical guards passed");
    return 0;
}
