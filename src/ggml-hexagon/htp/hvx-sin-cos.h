#ifndef HVX_SIN_COS_H
#define HVX_SIN_COS_H

#include "hvx-base.h"
#include "hvx-floor.h"

static inline HVX_Vector hvx_vec_cos_f32(HVX_Vector x) {
    HVX_Vector const_inv_pi = hvx_vec_splat_f32(0.3183098861837907f);
    HVX_Vector const_half   = hvx_vec_splat_f32(0.5f);
    HVX_Vector const_pi     = hvx_vec_splat_f32(3.141592653589793f);
    HVX_Vector const_one    = hvx_vec_splat_f32(1.0f);
    HVX_Vector const_neg_one = hvx_vec_splat_f32(-1.0f);

    // n = floor(x * (1/pi) + 0.5)
    HVX_Vector n_float = hvx_vec_floor_f32(hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(x, const_inv_pi), const_half));

    // y = x - n * pi
    HVX_Vector y = hvx_vec_sub_f32_f32(x, hvx_vec_mul_f32_f32(n_float, const_pi));

    // Sign determination: if n is odd, sign is -1.0f, else 1.0f
    // half_n = n * 0.5f
    HVX_Vector half_n = hvx_vec_mul_f32_f32(n_float, const_half);
    // floor_half_n = floor(half_n)
    HVX_Vector floor_half_n = hvx_vec_floor_f32(half_n);
    // is_odd = half_n > floor_half_n
    HVX_VectorPred is_odd = Q6_Q_vcmp_gt_VsfVsf(half_n, floor_half_n);
    // sign = vmux(is_odd, -1.0f, 1.0f)
    HVX_Vector sign = Q6_V_vmux_QVV(is_odd, const_neg_one, const_one);

    // z = y^2
    HVX_Vector z = hvx_vec_mul_f32_f32(y, y);

    // Chebyshev approximation for cos(y)
    HVX_Vector c4 = hvx_vec_splat_f32(2.3557242013849433e-05f);
    HVX_Vector c3 = hvx_vec_splat_f32(-0.0013871428263450528f);
    HVX_Vector c2 = hvx_vec_splat_f32(0.041665895266688284f);
    HVX_Vector c1 = hvx_vec_splat_f32(-0.4999999360426369f);
    HVX_Vector c0 = hvx_vec_splat_f32(0.9999999999071725f);

    HVX_Vector cos_y = hvx_vec_add_f32_f32(c3, hvx_vec_mul_f32_f32(z, c4));
    cos_y = hvx_vec_add_f32_f32(c2, hvx_vec_mul_f32_f32(z, cos_y));
    cos_y = hvx_vec_add_f32_f32(c1, hvx_vec_mul_f32_f32(z, cos_y));
    cos_y = hvx_vec_add_f32_f32(c0, hvx_vec_mul_f32_f32(z, cos_y));

    return hvx_vec_mul_f32_f32(cos_y, sign);
}

static inline HVX_Vector hvx_vec_sin_f32(HVX_Vector x) {
    HVX_Vector const_inv_pi = hvx_vec_splat_f32(0.3183098861837907f);
    HVX_Vector const_half   = hvx_vec_splat_f32(0.5f);
    HVX_Vector const_pi     = hvx_vec_splat_f32(3.141592653589793f);
    HVX_Vector const_one    = hvx_vec_splat_f32(1.0f);
    HVX_Vector const_neg_one = hvx_vec_splat_f32(-1.0f);

    // n = floor(x * (1/pi) + 0.5)
    HVX_Vector n_float = hvx_vec_floor_f32(hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(x, const_inv_pi), const_half));

    // y = x - n * pi
    HVX_Vector y = hvx_vec_sub_f32_f32(x, hvx_vec_mul_f32_f32(n_float, const_pi));

    // Sign determination: if n is odd, sign is -1.0f, else 1.0f
    // half_n = n * 0.5f
    HVX_Vector half_n = hvx_vec_mul_f32_f32(n_float, const_half);
    // floor_half_n = floor(half_n)
    HVX_Vector floor_half_n = hvx_vec_floor_f32(half_n);
    // is_odd = half_n > floor_half_n
    HVX_VectorPred is_odd = Q6_Q_vcmp_gt_VsfVsf(half_n, floor_half_n);
    // sign = vmux(is_odd, -1.0f, 1.0f)
    HVX_Vector sign = Q6_V_vmux_QVV(is_odd, const_neg_one, const_one);

    // z = y^2
    HVX_Vector z = hvx_vec_mul_f32_f32(y, y);

    // Chebyshev approximation for sin(y)
    HVX_Vector s4 = hvx_vec_splat_f32(2.642186986152672e-06f);
    HVX_Vector s3 = hvx_vec_splat_f32(-0.00019825318964070864f);
    HVX_Vector s2 = hvx_vec_splat_f32(0.00833326283319605f);
    HVX_Vector s1 = hvx_vec_splat_f32(-0.16666666082087775f);
    HVX_Vector s0 = hvx_vec_splat_f32(0.999999999915155f);

    HVX_Vector sin_y = hvx_vec_add_f32_f32(s3, hvx_vec_mul_f32_f32(z, s4));
    sin_y = hvx_vec_add_f32_f32(s2, hvx_vec_mul_f32_f32(z, sin_y));
    sin_y = hvx_vec_add_f32_f32(s1, hvx_vec_mul_f32_f32(z, sin_y));
    sin_y = hvx_vec_add_f32_f32(s0, hvx_vec_mul_f32_f32(z, sin_y));
    sin_y = hvx_vec_mul_f32_f32(y, sin_y);

    return hvx_vec_mul_f32_f32(sin_y, sign);
}

// Row-span wrapper: apply hvx_vec_sin_f32 element-wise over num_elems FP32
// samples. Handles aligned fast path and unaligned fallback the same way
// hvx_exp_f32 does, with a partial-vector tail via hvx_vec_store_u.
static inline void hvx_sin_f32(uint8_t * restrict dst, const uint8_t * restrict src, int num_elems) {
    const int left_over       = num_elems & (VLEN_FP32 - 1);
    const int num_elems_whole = num_elems - left_over;

    const int src_aligned = hex_is_aligned((void *) src, VLEN);
    const int dst_aligned = hex_is_aligned((void *) dst, VLEN);

    if (src_aligned && dst_aligned) {
        HVX_Vector * p_in  = (HVX_Vector *) src;
        HVX_Vector * p_out = (HVX_Vector *) dst;
        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            *p_out++ = hvx_vec_sin_f32(*p_in++);
        }
    } else {
        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector in = *(HVX_UVector *) (src + i * SIZEOF_FP32);
            *(HVX_UVector *) (dst + i * SIZEOF_FP32) = hvx_vec_sin_f32(in);
        }
    }

    if (left_over > 0) {
        HVX_Vector in = *(HVX_UVector *) (src + num_elems_whole * SIZEOF_FP32);
        HVX_Vector out = hvx_vec_sin_f32(in);
        hvx_vec_store_u(dst + num_elems_whole * SIZEOF_FP32, left_over * SIZEOF_FP32, out);
    }
}

#endif /* HVX_SIN_COS_H */
