#ifndef HVX_TANH_H
#define HVX_TANH_H

#include "hvx-base.h"
#include "hvx-erf.h"
#include "hvx-inverse.h"

// tanh(x) = x * P(x^2) / Q(x^2) on [-c, c] (tanh is +-1 in f32 outside it),
// the minimax rational form Eigen uses for float.
#define HVX_TANH_CLAMP 7.90531110763549805f

#define HVX_TANH_P1  4.89352455891786e-03f
#define HVX_TANH_P3  6.37261928875436e-04f
#define HVX_TANH_P5  1.48572235717979e-05f
#define HVX_TANH_P7  5.12229709037114e-08f
#define HVX_TANH_P9  -8.60467152213735e-11f
#define HVX_TANH_P11 2.00018790482477e-13f
#define HVX_TANH_P13 -2.76076847742355e-16f

#define HVX_TANH_Q0 4.89352518554385e-03f
#define HVX_TANH_Q2 2.26843463243900e-03f
#define HVX_TANH_Q4 1.18534705686654e-04f
#define HVX_TANH_Q6 1.19825839466702e-06f

#define HVX_GELU_SQRT_2_OVER_PI 0.79788456080286535588f
#define HVX_GELU_COEF_A         0.044715f

static inline HVX_Vector hvx_vec_tanh_numerator_f32(HVX_Vector x, HVX_Vector x2) {
    HVX_Vector p = hvx_vec_madd_f32(x2, hvx_vec_splat_f32(HVX_TANH_P13), HVX_TANH_P11);
    p = hvx_vec_madd_f32(x2, p, HVX_TANH_P9);
    p = hvx_vec_madd_f32(x2, p, HVX_TANH_P7);
    p = hvx_vec_madd_f32(x2, p, HVX_TANH_P5);
    p = hvx_vec_madd_f32(x2, p, HVX_TANH_P3);
    p = hvx_vec_madd_f32(x2, p, HVX_TANH_P1);
    return hvx_vec_mul_f32_f32(x, p);
}

static inline HVX_Vector hvx_vec_tanh_denominator_f32(HVX_Vector x2) {
    HVX_Vector q = hvx_vec_madd_f32(x2, hvx_vec_splat_f32(HVX_TANH_Q6), HVX_TANH_Q4);
    q = hvx_vec_madd_f32(x2, q, HVX_TANH_Q2);
    return hvx_vec_madd_f32(x2, q, HVX_TANH_Q0);
}

static inline HVX_Vector hvx_vec_tanh_rational_f32(HVX_Vector v) {
    const HVX_Vector hi = hvx_vec_splat_f32(HVX_TANH_CLAMP);
    const HVX_Vector lo = hvx_vec_splat_f32(-HVX_TANH_CLAMP);
    const HVX_Vector x  = Q6_Vsf_vmax_VsfVsf(Q6_Vsf_vmin_VsfVsf(v, hi), lo);
    const HVX_Vector x2 = hvx_vec_mul_f32_f32(x, x);
    const HVX_Vector p  = hvx_vec_tanh_numerator_f32(x, x2);
    const HVX_Vector q  = hvx_vec_tanh_denominator_f32(x2);
    return hvx_vec_mul_f32_f32(p, hvx_vec_inverse_f32(q));
}

// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3))), ggml's GELU.
static inline HVX_Vector hvx_vec_gelu_tanh_f32(HVX_Vector x) {
    const HVX_Vector x2    = hvx_vec_mul_f32_f32(x, x);
    const HVX_Vector cubic = hvx_vec_madd_f32(x2, hvx_vec_splat_f32(HVX_GELU_COEF_A), 1.0f);
    const HVX_Vector z     = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(x, cubic), hvx_vec_splat_f32(HVX_GELU_SQRT_2_OVER_PI));
    const HVX_Vector one_plus_tanh = hvx_vec_add_f32_f32(hvx_vec_tanh_rational_f32(z), hvx_vec_splat_f32(1.0f));
    const HVX_Vector half_x = hvx_vec_mul_f32_f32(x, hvx_vec_splat_f32(0.5f));
    return hvx_vec_mul_f32_f32(half_x, one_plus_tanh);
}

static inline void hvx_gelu_tanh_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);

    HVX_Vector * restrict vdst       = (HVX_Vector *) dst;
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;

    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    #pragma unroll(4)
    for (; i < nvec; i++) {
        vdst[i] = hvx_vec_gelu_tanh_f32(vsrc[i]);
    }
    if (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * sizeof(float), hvx_vec_gelu_tanh_f32(vsrc[i]));
    }
}

#endif  // HVX_TANH_H
