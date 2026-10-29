#ifndef HTP_CONV_2D_DW_GUARDS_H
#define HTP_CONV_2D_DW_GUARDS_H

#include <stdbool.h>
#include <stdint.h>

// The caller has already validated F32 activation/output and F32/F16 weights.
// Only F32 weights gain new vertical geometries. F16 keeps its original 3x3,
// stride-2-in-both-axes route; all other layouts/geometries use the fallback.
static inline bool htp_conv_dw_planar_stride2_kw3_eligible(
        bool channels_contiguous, uint32_t x_stride0, uint32_t y_stride0,
        uint32_t kw, uint32_t kh, int32_t sx, int32_t sy, int32_t dx, int32_t dy,
        bool weights_f32) {
    return !channels_contiguous && x_stride0 == sizeof(float) && y_stride0 == sizeof(float) &&
           kw == 3 && kh > 0 && sx == 2 && sy > 0 && dx == 1 && dy == 1 &&
           (weights_f32 || (kh == 3 && sy == 2));
}

// New row-window reuse is deliberately F32-only and bounded to small spatial
// kernels. The caller retains the original type/layout/shape/span validation.
static inline bool htp_conv_dw_planar_stride1_kw35_eligible(bool channels_contiguous, uint32_t x_stride0,
                                                            uint32_t y_stride0, uint32_t kw, uint32_t kh, int32_t sx,
                                                            int32_t sy, int32_t dx, int32_t dy, bool weights_f32) {
    return weights_f32 && !channels_contiguous && x_stride0 == sizeof(float) && y_stride0 == sizeof(float) &&
           (kw == 3 || kw == 5) && kh >= 1 && kh <= 4 && sx == 1 && sy >= 1 && sy <= 3 && dx == 1 && dy == 1;
}

// A shared, testable copy window avoids constructing pointers before a row.
// The two-vector stage bounds the supported span, including all tail lanes.
static inline void htp_conv_dw_stride1_window(uint32_t width, int64_t ix, uint32_t span, uint32_t *source,
                                              uint32_t *target, uint32_t *count) {
    *source = *target = *count = 0;
    if (!width || !span || span > 64 || ix >= (int64_t)width || ix <= -(int64_t)span) {
        return;
    }
    const int64_t first = ix < 0 ? -ix : 0;
    const int64_t available = (int64_t)width - ix;
    const int64_t end = available < span ? available : span;
    if (first < end) {
        *source = (uint32_t)(ix + first);
        *target = (uint32_t)first;
        *count = (uint32_t)(end - first);
    }
}

#endif
