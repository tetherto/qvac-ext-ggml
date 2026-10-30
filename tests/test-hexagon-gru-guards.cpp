#include "gru-ops.h"

#include <array>
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

static bool packed_lane_mapping() {
    // Independent byte-tag emulation of vror followed by vdelta's descending
    // butterfly. Check every byte, not just repeated float values: a bad
    // control can otherwise mix bytes while appearing to preserve zeroes.
    for (uint32_t h : {4u, 8u}) {
        const uint32_t packed = HTP_GRU_LANES / h;
        for (uint32_t k = 0; k < h; ++k) {
            std::array<uint8_t, 128> current;
            for (uint32_t byte = 0; byte < current.size(); ++byte) {
                current[byte] = (byte + k * sizeof(float)) % current.size();
            }
            for (uint32_t delta = 64; delta; delta /= 2) {
                const auto previous = current;
                for (uint32_t byte = 0; byte < current.size(); ++byte) {
                    const uint32_t source = byte ^ (htp_gru_packed_repl_control(h, byte) & delta);
                    current[byte] = previous[source];
                }
            }
            for (uint32_t batch = 0; batch < packed; ++batch) {
                for (uint32_t j = 0; j < h; ++j) {
                    for (uint32_t byte = 0; byte < sizeof(float); ++byte) {
                        if (current[(batch * h + j) * sizeof(float) + byte] != (batch * h + k) * sizeof(float) + byte) {
                            return false;
                        }
                    }
                }
            }
        }
        for (uint32_t batch = 0; batch < packed; ++batch) {
            for (uint32_t gate = 0; gate < 3; ++gate) {
                for (uint32_t j = 0; j < h; ++j) {
                    const uint32_t actual = htp_gru_packed_input_word(h, gate, batch * h + j);
                    if (actual != batch * 3 * h + gate * h + j || actual >= 3 * HTP_GRU_LANES) {
                        return false;
                    }
                }
            }
        }
        // Include both sides of the full-tile cutoff and partial final tiles.
        for (uint32_t b : {1u, 3u, 4u, 5u, 7u, 8u, 9u, 11u, 17u, 31u, 32u, 33u, 231u, 441u}) {
            const uint32_t actual = htp_gru_packed_batches(h, b);
            if (actual != (b >= packed ? packed : 1)) {
                return false;
            }
            uint32_t covered = 0;
            for (uint32_t first = 0; first < b; first += actual) {
                const uint32_t valid = b - first < actual ? b - first : actual;
                if (!valid || valid > actual || first + valid > b) {
                    return false;
                }
                covered += valid;
            }
            if (covered != b) {
                return false;
            }
        }
    }
    for (uint32_t h : {1u, 2u, 3u, 7u, 16u, 32u, 64u, 128u}) {
        if (htp_gru_packed_batches(h, 441) != 1) {
            return false;
        }
    }
    return true;
}

int main() {
    htp_gru_layout layout;
    CHECK(packed_lane_mapping());
    CHECK(!htp_gru_span_fits(UINT32_MAX - 15, 64, UINT32_MAX));
    CHECK(htp_gru_span_fits(UINT32_MAX - 15, 64, UINT64_MAX));
    CHECK(!htp_gru_span_fits(UINT64_MAX - 15, 64, UINT64_MAX));
    for (uint32_t h : {1u, 2u, 3u, 4u, 5u, 8u, 31u, 32u, 33u, 64u, 127u, 128u}) {
        for (uint32_t b : {1u, 3u, 4u, 5u, 7u, 8u, 9u, 11u, 17u, 231u, 441u})
            for (uint32_t l : {1u, 2u, 63u})
                for (int reverse : {0, 1}) {
                    auto w = tensor(h, 3 * h), gi = tensor(3 * h, b, l), bias = tensor(3 * h), dst = tensor(h, b, l);
                    CHECK(htp_gru_validate(&w, &gi, &bias, &dst, reverse, 6, 1024 * 1024, &layout));
                    CHECK(layout.h == h && layout.b == b && layout.l == l && layout.padded_h % 32 == 0);
                    CHECK(layout.threads == (b < 6 ? b : 6));
                    CHECK(layout.weights_bytes == 3 * h * layout.padded_h * sizeof(float));
                    CHECK(layout.worker_bytes ==
                          (7 * layout.padded_h + (htp_gru_packed_batches(h, b) > 1 ? 96 : 0)) * sizeof(float));
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
    std::puts("GRU F32 packed byte/subgroup/tail mapping passed");
    return 0;
}
