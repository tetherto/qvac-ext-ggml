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

#endif
