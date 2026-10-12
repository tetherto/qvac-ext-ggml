#pragma once

#include "htp/matmul-ops.h"

#include <cstring>

#define HTP_MM_F16_F32_PANEL_DEFAULT HTP_MM_F16_F32_PANEL_4X2

struct htp_f16_f32_panel_shape_name {
    const char * name;
    int          shape;
};

static constexpr htp_f16_f32_panel_shape_name htp_f16_f32_panel_shape_names[] = {
    { "2x2", HTP_MM_F16_F32_PANEL_2X2 },
    { "4x2", HTP_MM_F16_F32_PANEL_4X2 },
};

static inline bool htp_parse_f16_f32_panel_shape(const char * name, int * shape) {
    for (const auto & entry : htp_f16_f32_panel_shape_names) {
        if (std::strcmp(name, entry.name) == 0) {
            *shape = entry.shape;
            return true;
        }
    }
    *shape = HTP_MM_F16_F32_PANEL_DEFAULT;
    return false;
}

static inline const char * htp_f16_f32_panel_shape_label(int shape) {
    for (const auto & entry : htp_f16_f32_panel_shape_names) {
        if (entry.shape == shape) {
            return entry.name;
        }
    }
    return "unknown";
}
