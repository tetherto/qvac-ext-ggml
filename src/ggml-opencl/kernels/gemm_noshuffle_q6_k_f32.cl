#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#define K_PER_SCALE_PAIR 32

inline bool tile_finite(float8 c0, float8 c1, float8 c2, float8 c3) {
    return all(isfinite(c0)) && all(isfinite(c1)) && all(isfinite(c2)) && all(isfinite(c3));
}

inline float4 dequantize_rows(
        global const ushort * ptr_ql,
        global const uchar * ptr_qh,
        global const ushort * ptr_s,
        global const half * ptr_d,
        int m,
        int kk
) {
    ushort4 bits4 = vload4(0, ptr_ql + (kk/4)*m);
    uchar4  bits2 = vload4(0, ptr_qh + (kk/4)*m);
    char8 scale_s_8 = as_char8(vload4(0, ptr_s + (kk/K_PER_SCALE_PAIR)*m));
    char4 scale_s = ((kk/16) % 2) == 0 ? scale_s_8.s0246 : scale_s_8.s1357;
    float4 scale_d = convert_float4(vload4(0, ptr_d + (kk/256)*m));
    ushort4 low  = (bits4 >> (ushort) (4 * (kk % 4))) & (ushort) 0xF;
    ushort4 high = convert_ushort4((bits2 >> (uchar) (2 * (kk % 4))) & (uchar) 0x3);
    return convert_float4(convert_int4(low | (high << (ushort) 4)) - 32) * convert_float4(scale_s) * scale_d;
}

inline void accumulate_tile_fp32(
        read_only image1d_buffer_t src1,
        global const ushort * ptr_ql,
        global const uchar * ptr_qh,
        global const ushort * ptr_s,
        global const half * ptr_d,
        int m,
        int n_4,
        int k,
        int gy,
        float8 * c0,
        float8 * c1,
        float8 * c2,
        float8 * c3
) {
    *c0 = 0; *c1 = 0; *c2 = 0; *c3 = 0;
    for (int kk = 0; kk < k; ++kk) {
        half8 B;
        B.s0123 = read_imageh(src1, gy * 2 + kk * n_4);
        B.s4567 = read_imageh(src1, gy * 2 + kk * n_4 + 1);
        float8 activations = convert_float8(B);
        float4 weights = dequantize_rows(ptr_ql, ptr_qh, ptr_s, ptr_d, m, kk);
        *c0 += activations * weights.s0;
        *c1 += activations * weights.s1;
        *c2 += activations * weights.s2;
        *c3 += activations * weights.s3;
    }
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_gemm_noshuffle_q6_K_f32(
        global const ushort * src0_ql,
        global const uchar  * src0_qh,
        global const ushort * src0_s,
        global const half   * src0_d,
        read_only image1d_buffer_t src1,
        global float * dst,
        ulong offsetd,
        int m,
        int n,
        int k,
        int n_no_padding,
        ushort mask_f000,
        uchar  mask_c0
) {
    dst = (global float *)( (global char *)dst + offsetd );

    int n_4 = n >> 2;

    int gy = get_global_id(0); // n
    int gx = get_global_id(1); // m
    int gx_2 = gx << 2;

    float8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B;
    half4 dequantized_weights;

    global const ushort * ptr_ql = src0_ql + gx_2;
    global const uchar  * ptr_qh = src0_qh + gx_2;
    global const ushort * ptr_s  = src0_s  + gx_2;
    global const half   * ptr_d  = src0_d  + gx_2;

    for (int ib = 0; ib < k; ib += K_PER_SCALE_PAIR) {
        // load 4 consecutive scales
        char8 scale_s_8 = as_char8(vload4(0, ptr_s + (ib/K_PER_SCALE_PAIR)*m)); // 1 char scale every 16 elements, packed in 2s
        half4   scale_d = vload4(0, ptr_d + (ib/256)*m);  // 1 half scale every 256 elements
        half8 b0 = 0, b1 = 0, b2 = 0, b3 = 0;

        for (int i = ib; i < ib + K_PER_SCALE_PAIR; i += 4) {
            // load 4x elements (ushort) of ql on M, each ushort contains 4 weights
            // 4x ushort correspons to 4 rows on M
            ushort4 bits4 = vload4(0, ptr_ql + (i/4)*m); // ql packed in 4s in ushort
            uchar4  bits2 = vload4(0, ptr_qh + (i/4)*m); // qh packed in 4s in uchar

            char4   scale_s = ((i/16) % 2) == 0 ? scale_s_8.s0246 : scale_s_8.s1357; // transposed as ushort, 2 blocks

            // j=0
            // load 2x 4 elements of activations on N, corresponding to 8 rows on N
            B.s0123 = read_imageh(src1, gy*2 + (i + 0)*n_4 + 0);
            B.s4567 = read_imageh(src1, gy*2 + (i + 0)*n_4 + 1);
            dequantized_weights.s0 = (convert_half((bits4.s0 & 0x000F) | ((bits2.s0 & 0x03) << 4)) - 32.f) * scale_s.s0 * scale_d.s0;
            dequantized_weights.s1 = (convert_half((bits4.s1 & 0x000F) | ((bits2.s1 & 0x03) << 4)) - 32.f) * scale_s.s1 * scale_d.s1;
            dequantized_weights.s2 = (convert_half((bits4.s2 & 0x000F) | ((bits2.s2 & 0x03) << 4)) - 32.f) * scale_s.s2 * scale_d.s2;
            dequantized_weights.s3 = (convert_half((bits4.s3 & 0x000F) | ((bits2.s3 & 0x03) << 4)) - 32.f) * scale_s.s3 * scale_d.s3;
            b0 += B * dequantized_weights.s0;
            b1 += B * dequantized_weights.s1;
            b2 += B * dequantized_weights.s2;
            b3 += B * dequantized_weights.s3;

            // j=1
            B.s0123 = read_imageh(src1, gy*2 + (i + 1)*n_4 + 0);
            B.s4567 = read_imageh(src1, gy*2 + (i + 1)*n_4 + 1);
            dequantized_weights.s0 = (convert_half((((bits4.s0 & 0x00F0) >> 4) | ((bits2.s0 & 0x0C) << 2))) - 32.f) * scale_s.s0 * scale_d.s0;
            dequantized_weights.s1 = (convert_half((((bits4.s1 & 0x00F0) >> 4) | ((bits2.s1 & 0x0C) << 2))) - 32.f) * scale_s.s1 * scale_d.s1;
            dequantized_weights.s2 = (convert_half((((bits4.s2 & 0x00F0) >> 4) | ((bits2.s2 & 0x0C) << 2))) - 32.f) * scale_s.s2 * scale_d.s2;
            dequantized_weights.s3 = (convert_half((((bits4.s3 & 0x00F0) >> 4) | ((bits2.s3 & 0x0C) << 2))) - 32.f) * scale_s.s3 * scale_d.s3;
            b0 += B * dequantized_weights.s0;
            b1 += B * dequantized_weights.s1;
            b2 += B * dequantized_weights.s2;
            b3 += B * dequantized_weights.s3;

            // j=2
            B.s0123 = read_imageh(src1, gy*2 + (i + 2)*n_4 + 0);
            B.s4567 = read_imageh(src1, gy*2 + (i + 2)*n_4 + 1);
            dequantized_weights.s0 = (convert_half((((bits4.s0 & 0x0F00) >> 8) | (bits2.s0 & 0x30))) - 32.f) * scale_s.s0 * scale_d.s0;
            dequantized_weights.s1 = (convert_half((((bits4.s1 & 0x0F00) >> 8) | (bits2.s1 & 0x30))) - 32.f) * scale_s.s1 * scale_d.s1;
            dequantized_weights.s2 = (convert_half((((bits4.s2 & 0x0F00) >> 8) | (bits2.s2 & 0x30))) - 32.f) * scale_s.s2 * scale_d.s2;
            dequantized_weights.s3 = (convert_half((((bits4.s3 & 0x0F00) >> 8) | (bits2.s3 & 0x30))) - 32.f) * scale_s.s3 * scale_d.s3;
            b0 += B * dequantized_weights.s0;
            b1 += B * dequantized_weights.s1;
            b2 += B * dequantized_weights.s2;
            b3 += B * dequantized_weights.s3;

            // j=3
            B.s0123 = read_imageh(src1, gy*2 + (i + 3)*n_4 + 0);
            B.s4567 = read_imageh(src1, gy*2 + (i + 3)*n_4 + 1);
            dequantized_weights.s0 = (convert_half((((bits4.s0 & mask_f000) >> 12) | ((bits2.s0 & mask_c0) >> 2))) - 32.f) * scale_s.s0 * scale_d.s0;
            dequantized_weights.s1 = (convert_half((((bits4.s1 & mask_f000) >> 12) | ((bits2.s1 & mask_c0) >> 2))) - 32.f) * scale_s.s1 * scale_d.s1;
            dequantized_weights.s2 = (convert_half((((bits4.s2 & mask_f000) >> 12) | ((bits2.s2 & mask_c0) >> 2))) - 32.f) * scale_s.s2 * scale_d.s2;
            dequantized_weights.s3 = (convert_half((((bits4.s3 & mask_f000) >> 12) | ((bits2.s3 & mask_c0) >> 2))) - 32.f) * scale_s.s3 * scale_d.s3;
            b0 += B * dequantized_weights.s0;
            b1 += B * dequantized_weights.s1;
            b2 += B * dequantized_weights.s2;
            b3 += B * dequantized_weights.s3;
        }

        c0 += convert_float8(b0);
        c1 += convert_float8(b1);
        c2 += convert_float8(b2);
        c3 += convert_float8(b3);
    }

    if (!tile_finite(c0, c1, c2, c3)) {
        accumulate_tile_fp32(src1, ptr_ql, ptr_qh, ptr_s, ptr_d, m, n_4, k, gy, &c0, &c1, &c2, &c3);
    }

    int idx = (gy<<3)*m + (gx<<2);

    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, dst + idx);
    }
}
