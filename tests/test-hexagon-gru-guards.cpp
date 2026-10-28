#include "gru-ops.h"

#include <cstdio>
#include <initializer_list>
#include <limits>

#define CHECK(value)                                                                                                   \
    do {                                                                                                               \
        if (!(value)) {                                                                                                \
            std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #value);                                              \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

static htp_gru_tensor tensor(uint64_t a, uint64_t b = 1, uint64_t c = 1) {
    htp_gru_tensor result = {};
    result.ne[0] = a;
    result.ne[1] = b;
    result.ne[2] = c;
    result.ne[3] = 1;
    uint64_t stride = sizeof(float);
    for (int i = 0; i < 4; ++i) {
        result.nb[i] = stride;
        stride *= result.ne[i];
    }
    result.size = stride;
    return result;
}

int main() {
    htp_gru_layout layout;
    CHECK(!htp_gru_span_fits(UINT32_MAX - 15, 64, UINT32_MAX));
    CHECK(htp_gru_span_fits(UINT32_MAX - 15, 64, UINT64_MAX));
    CHECK(!htp_gru_span_fits(UINT64_MAX - 15, 64, UINT64_MAX));
    for (uint32_t h : {1u, 2u, 3u, 5u, 31u, 32u, 33u, 64u, 127u, 128u}) {
        for (uint32_t b : {1u, 3u, 17u})
            for (uint32_t l : {1u, 2u, 63u})
                for (int reverse : {0, 1}) {
                    auto w = tensor(h, 3 * h), gi = tensor(3 * h, b, l), bias = tensor(3 * h), dst = tensor(h, b, l);
                    CHECK(htp_gru_validate(&w, &gi, &bias, &dst, reverse, 6, 1024 * 1024, &layout));
                    CHECK(layout.h == h && layout.b == b && layout.l == l && layout.padded_h % 32 == 0);
                    CHECK(layout.threads == (b < 6 ? b : 6));
                    CHECK(layout.weights_bytes == 3 * h * layout.padded_h * sizeof(float));
                    CHECK(!htp_gru_validate(&w, &gi, &bias, &dst, reverse, 6, layout.total_bytes - 1, &layout));
                }
    }
    auto w = tensor(3, 9), gi = tensor(9, 2, 7), bias = tensor(9), dst = tensor(3, 2, 7);
    const auto valid = [&]() { return htp_gru_validate(&w, &gi, &bias, &dst, 0, 6, 1024 * 1024, &layout); };
    CHECK(valid());
    auto saved = gi;
    gi.type = 1;
    CHECK(!valid());
    gi = saved;
    gi.nb[1] += 4;
    CHECK(!valid());
    gi = saved;
    --gi.size;
    CHECK(!valid());
    gi = saved;
    gi.ne[0] = 8;
    CHECK(!valid());
    gi = saved;
    gi.ne[1] = 0;
    CHECK(!valid());
    gi = saved;
    gi.ne[2] = std::numeric_limits<uint64_t>::max();
    CHECK(!valid());
    gi = saved;
    gi.ne[1] = UINT32_MAX;
    CHECK(!valid());
    gi = saved;
    gi.address = 2;
    CHECK(!valid());
    gi = saved;
    gi.address = UINT64_MAX - 3;
    CHECK(!valid());
    gi = saved;
    w.address = 4096;
    dst.address = 4100;
    CHECK(!valid());
    dst.address = w.address + w.size;
    CHECK(valid());
    w.address = 0;
    dst.address = 0;
    CHECK(!htp_gru_validate(&w, &gi, &bias, &dst, -1, 6, 1024 * 1024, &layout));
    CHECK(!htp_gru_validate(&w, &gi, &bias, &dst, 2, 6, 1024 * 1024, &layout));
    CHECK(!htp_gru_validate(&w, &gi, &bias, &dst, 0, 0, 1024 * 1024, &layout));
    CHECK(!htp_gru_validate(&w, &gi, &bias, &dst, 0, 11, 1024 * 1024, &layout));
    w = tensor(129, 387);
    gi = tensor(387, 2, 7);
    bias = tensor(387);
    dst = tensor(129, 2, 7);
    CHECK(!valid());
    std::puts("GRU F32 shape/span/reverse/alias/VTCM guards passed");
    return 0;
}
