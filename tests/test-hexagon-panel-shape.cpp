#include "ggml-impl.h"
#if !defined(__clang__) && !defined(__arm__) && !defined(__aarch64__)
#define __fp16 ggml_fp16_t
#define TEST_PANEL_FP16_ALIAS
#endif
#include "htp-panel.h"
#ifdef TEST_PANEL_FP16_ALIAS
#undef __fp16
#undef TEST_PANEL_FP16_ALIAS
#endif

#include <cstdio>
#include <cstring>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

static bool parses_to(const char * name, int expected) {
    int shape = -1;
    return htp_parse_f16_f32_panel_shape(name, &shape) && shape == expected;
}

static bool rejects_to_default(const char * name) {
    int shape = -1;
    return !htp_parse_f16_f32_panel_shape(name, &shape) && shape == HTP_MM_F16_F32_PANEL_DEFAULT;
}

int main() {
    CHECK(HTP_MM_F16_F32_PANEL_DEFAULT == HTP_MM_F16_F32_PANEL_4X2);
    CHECK(parses_to("4x2", HTP_MM_F16_F32_PANEL_4X2));
    CHECK(parses_to("2x2", HTP_MM_F16_F32_PANEL_2X2));
    CHECK(rejects_to_default(""));
    CHECK(rejects_to_default("4X2"));
    CHECK(rejects_to_default("2x4"));
    CHECK(rejects_to_default("4x2 "));
    CHECK(std::strcmp(htp_f16_f32_panel_shape_label(HTP_MM_F16_F32_PANEL_4X2), "4x2") == 0);
    CHECK(std::strcmp(htp_f16_f32_panel_shape_label(HTP_MM_F16_F32_PANEL_2X2), "2x2") == 0);
    CHECK(std::strcmp(htp_f16_f32_panel_shape_label(HTP_MM_F16_F32_PANEL_DEFAULT), "4x2") == 0);
    std::puts("PASS: 4x2 default, 2x2 opt-out, unsupported names keep the default");
    return 0;
}
