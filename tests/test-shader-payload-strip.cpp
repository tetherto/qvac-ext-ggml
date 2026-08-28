#include "shader-payload-strip.hpp"

#include <cstdio>

static int g_failures = 0;

static void expect_strip(const char * name, bool expected) {
    const bool got = should_strip_shader_payload(name);
    if (got != expected) {
        std::printf("FAIL: \"%s\" strip=%d (expected %d)\n", name, got ? 1 : 0, expected ? 1 : 0);
        g_failures++;
        return;
    }
    std::printf("ok:   \"%s\" strip=%d\n", name, got ? 1 : 0);
}

static void expect_keep_names(const char * const * names, int count) {
    for (int i = 0; i < count; ++i) {
        expect_strip(names[i], false);
    }
}

static void expect_strip_names(const char * const * names, int count) {
    for (int i = 0; i < count; ++i) {
        expect_strip(names[i], true);
    }
}

int main() {
    const char * const keep[] = {
        "matmul_q8_0_f32",
        "mul_mat_vec_q4_0_f32_f32",
        "rms_norm_f32",
        "silu_f32",
        "soft_max_f32",
        "step_f32",
        "geglu_f32",
        "matmul_q4_0_f32",
        "matmul_q5_0_f32",
        "matmul_q6_k_f32",
        "dequant_q8_0",
        "matmul_expq_f32",
    };
    const char * const strip[] = {
        "matmul_iq4_nl_f32",
        "matmul_q1_0_f32",
        "dequant_mxfp4",
        "dequant_nvfp4",
        "rms_norm_back_f32",
        "opt_step_adamw_f32",
        "mul_mat_vec_iq2_xxs_f32_f32",
        "silu_back_f32",
        "geglu_back_f32",
        "soft_max_back_f32",
        "repeat_back_f32",
        "out_prod_f32",
        "cross_entropy_loss_f32",
    };

    expect_keep_names(keep, static_cast<int>(sizeof(keep) / sizeof(keep[0])));
    expect_strip_names(strip, static_cast<int>(sizeof(strip) / sizeof(strip[0])));

    if (g_failures != 0) {
        std::printf("%d failure(s)\n", g_failures);
        return 1;
    }
    return 0;
}
