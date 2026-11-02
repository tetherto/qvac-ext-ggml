#ifndef HTP_COL2IM_1D_LAYOUT_H
#define HTP_COL2IM_1D_LAYOUT_H

#include "hex-common.h"

struct htp_col2im_1d_vtcm_layout {
    uint32_t taps_per_output;
    uint32_t band;
    uint32_t off_tmp;
    uint32_t off_out;
    uint32_t bytes_per_thread;
    uint32_t plan_bytes;
};

static inline bool htp_col2im_1d_vtcm_layout_build(uint32_t kernel, uint32_t channels, uint32_t stride,
                                                 uint32_t n_threads, size_t vtcm_size,
                                                 struct htp_col2im_1d_vtcm_layout * out) {
    size_t k_oc, plan_bytes;
    if (!kernel || !channels || !stride || stride > 32 || kernel % stride || !n_threads ||
        hex_mul_overflow(kernel, channels, &k_oc) || hex_mul_overflow(stride, 128, &plan_bytes) ||
        plan_bytes > vtcm_size || plan_bytes > UINT32_MAX) {
        return false;
    }
    const size_t taps = kernel / stride;
    const size_t worker_budget = MIN((vtcm_size - plan_bytes) / n_threads, UINT32_MAX);
    // Check sizes and padding before narrowing the DSP offsets.
    for (uint32_t band = 256; band >= 32; band /= 2) {
        size_t cols, cols_bytes, tmp_bytes, out_bytes, off_tmp, off_out, worker_bytes;
        if (hex_add_overflow(band / stride, taps, &cols) || hex_add_overflow(cols, 1, &cols) ||
            hex_mul_overflow(cols, k_oc, &cols_bytes) || hex_mul_overflow(cols_bytes, sizeof(float), &cols_bytes) ||
            hex_mul_overflow(taps, 128, &tmp_bytes) || hex_mul_overflow(channels, band * sizeof(float), &out_bytes) ||
            hex_add_overflow(cols_bytes, 127, &off_tmp) || hex_add_overflow(out_bytes, 127, &out_bytes)) {
            continue;
        }
        off_tmp = hex_align_down(off_tmp, 128);
        out_bytes = hex_align_down(out_bytes, 128);
        if (hex_add_overflow(off_tmp, tmp_bytes, &off_out) ||
            hex_add_overflow(off_out, out_bytes, &worker_bytes) || worker_bytes > worker_budget) {
            continue;
        }
        out->taps_per_output = (uint32_t) taps;
        out->band = band;
        out->off_tmp = (uint32_t) off_tmp;
        out->off_out = (uint32_t) off_out;
        out->bytes_per_thread = (uint32_t) worker_bytes;
        out->plan_bytes = (uint32_t) plan_bytes;
        return true;
    }
    return false;
}

#endif
