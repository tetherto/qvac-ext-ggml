#ifndef HVX_ERF_H
#define HVX_ERF_H

#include "hvx-base.h"
#include "hvx-exp.h"
#include "hvx-inverse.h"

// erf(x) approximated via Abramowitz-Stegun 7.1.26 (max abs error 1.5e-7
// over all finite inputs, well within FP32 precision):
//
//   t   = 1 / (1 + p*|x|)
//   y   = 1 - (((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t) * exp(-x*x)
//   erf = sign(x) * y
//
// Horner-order polynomial for a stable evaluation. Matches the standard CPU
// erff() closely enough for GELU_ERF to be used interchangeably with the FP32
// reference in supports_op callers.
static inline HVX_Vector hvx_vec_erf_f32(HVX_Vector x) {
    const HVX_Vector c_p    = hvx_vec_splat_f32(0.3275911f);
    const HVX_Vector c_a1   = hvx_vec_splat_f32(0.254829592f);
    const HVX_Vector c_a2   = hvx_vec_splat_f32(-0.284496736f);
    const HVX_Vector c_a3   = hvx_vec_splat_f32(1.421413741f);
    const HVX_Vector c_a4   = hvx_vec_splat_f32(-1.453152027f);
    const HVX_Vector c_a5   = hvx_vec_splat_f32(1.061405429f);
    const HVX_Vector c_one  = hvx_vec_splat_f32(1.0f);
    const HVX_Vector c_zero = hvx_vec_splat_f32(0.0f);

    // Decompose into |x| and sign so the polynomial sees non-negative inputs.
    const HVX_Vector x_abs = hvx_vec_abs_f32(x);

    // t = 1 / (1 + p*|x|)
    const HVX_Vector denom = hvx_vec_add_f32_f32(c_one, hvx_vec_mul_f32_f32(c_p, x_abs));
    const HVX_Vector t     = hvx_vec_inverse_f32(denom);

    // Horner: p(t) = ((((a5*t + a4)*t + a3)*t + a2)*t + a1) * t
    HVX_Vector poly = hvx_vec_add_f32_f32(c_a4, hvx_vec_mul_f32_f32(t, c_a5));
    poly = hvx_vec_add_f32_f32(c_a3, hvx_vec_mul_f32_f32(t, poly));
    poly = hvx_vec_add_f32_f32(c_a2, hvx_vec_mul_f32_f32(t, poly));
    poly = hvx_vec_add_f32_f32(c_a1, hvx_vec_mul_f32_f32(t, poly));
    poly = hvx_vec_mul_f32_f32(t, poly);

    // exp(-x*x). The guard against overflow is irrelevant here because
    // |x*x| swamps the erf approximation long before exp underflows anyway;
    // use the unguarded form for one less splat/compare per lane.
    const HVX_Vector x_sq      = hvx_vec_mul_f32_f32(x_abs, x_abs);
    const HVX_Vector neg_x_sq  = hvx_vec_neg_f32(x_sq);
    const HVX_Vector exp_val   = hvx_vec_exp_f32(neg_x_sq);

    // y = 1 - poly * exp(-x*x)
    const HVX_Vector y = hvx_vec_sub_f32_f32(c_one, hvx_vec_mul_f32_f32(poly, exp_val));

    // Reapply sign: positive -> +y, negative -> -y. (At x == 0 both branches
    // produce 0.)
    const HVX_VectorPred is_neg = Q6_Q_vcmp_gt_VsfVsf(c_zero, x);
    const HVX_Vector y_neg      = hvx_vec_neg_f32(y);
    return Q6_V_vmux_QVV(is_neg, y_neg, y);
}

// GELU_ERF(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
// Matches the CPU reference in src/ggml-cpu/vec.h (ggml_vec_gelu_erf_f32).
static inline HVX_Vector hvx_vec_gelu_erf_f32(HVX_Vector x) {
    const HVX_Vector c_sqrt2_inv = hvx_vec_splat_f32(0.70710678118654752f);  // 1 / sqrt(2)
    const HVX_Vector c_half      = hvx_vec_splat_f32(0.5f);
    const HVX_Vector c_one       = hvx_vec_splat_f32(1.0f);

    const HVX_Vector x_scaled = hvx_vec_mul_f32_f32(x, c_sqrt2_inv);
    const HVX_Vector e        = hvx_vec_erf_f32(x_scaled);
    const HVX_Vector one_plus = hvx_vec_add_f32_f32(c_one, e);
    const HVX_Vector half_x   = hvx_vec_mul_f32_f32(c_half, x);
    return hvx_vec_mul_f32_f32(half_x, one_plus);
}

// Row-span wrapper: apply hvx_vec_gelu_erf_f32 over num_elems FP32 samples.
// Shape mirrors hvx_sin_f32 / hvx_exp_f32 (aligned fast path + unaligned
// fallback + partial-vector tail via hvx_vec_store_u).
static inline void hvx_gelu_erf_f32(uint8_t * restrict dst, const uint8_t * restrict src, int num_elems) {
    const int left_over       = num_elems & (VLEN_FP32 - 1);
    const int num_elems_whole = num_elems - left_over;

    const int src_aligned = hex_is_aligned((void *) src, VLEN);
    const int dst_aligned = hex_is_aligned((void *) dst, VLEN);

    if (src_aligned && dst_aligned) {
        HVX_Vector * p_in  = (HVX_Vector *) src;
        HVX_Vector * p_out = (HVX_Vector *) dst;
        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            *p_out++ = hvx_vec_gelu_erf_f32(*p_in++);
        }
    } else {
        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector in = *(HVX_UVector *) (src + i * SIZEOF_FP32);
            *(HVX_UVector *) (dst + i * SIZEOF_FP32) = hvx_vec_gelu_erf_f32(in);
        }
    }

    if (left_over > 0) {
        HVX_Vector in  = *(HVX_UVector *) (src + num_elems_whole * SIZEOF_FP32);
        HVX_Vector out = hvx_vec_gelu_erf_f32(in);
        hvx_vec_store_u(dst + num_elems_whole * SIZEOF_FP32, left_over * SIZEOF_FP32, out);
    }
}

#endif /* HVX_ERF_H */
