#include "tdt-step.cuh"

// Greedy transducer step control, mirroring ggml_tdt_step_f32 in ggml-cpu/ops.cpp.
static __global__ void tdt_step_f32_kernel(
        const int * token, const int * dur_idx, const float * state, const float * dur_table,
        float * dst, const int n_dur, const int blank_id, const int max_symbols, const int rnnt) {
    const float t = state[GGML_TDT_STEP_IN_T];
    const float s = state[GGML_TDT_STEP_IN_S];
    const float n = state[GGML_TDT_STEP_IN_N];

    float t_next = t;
    float s_next = s;
    float update = 0.0f;

    if (t < n) {
        const int   raw = dur_idx[0];
        const int   di  = raw < 0 ? 0 : (raw < n_dur ? raw : n_dur - 1);
        const float dur = rnnt ? 0.0f : dur_table[di];
        const float adv = rnnt ? 1.0f : (dur > 1.0f ? dur : 1.0f);

        if (token[0] == blank_id) {
            t_next = t + adv;
            s_next = 0.0f;
        } else {
            update = 1.0f;
            s_next = s + 1.0f;
            if ((!rnnt && dur > 0.0f) || s_next >= (float) max_symbols) {
                t_next = t + adv;
                s_next = 0.0f;
            }
        }
    }

    float frame = t_next > n - 1.0f ? n - 1.0f : t_next;
    if (frame < 0.0f) {
        frame = 0.0f;
    }

    dst[GGML_TDT_STEP_OUT_T]      = t_next;
    dst[GGML_TDT_STEP_OUT_S]      = s_next;
    dst[GGML_TDT_STEP_OUT_N]      = n;
    dst[GGML_TDT_STEP_OUT_UPDATE] = update;
    dst[GGML_TDT_STEP_OUT_HOLD]   = 1.0f - update;
    dst[GGML_TDT_STEP_OUT_FRAME]  = frame;
}

void ggml_cuda_op_tdt_step(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * token     = dst->src[0];
    const ggml_tensor * dur_idx   = dst->src[1];
    const ggml_tensor * state     = dst->src[2];
    const ggml_tensor * dur_table = dst->src[3];

    GGML_ASSERT(token->type     == GGML_TYPE_I32);
    GGML_ASSERT(dur_idx->type   == GGML_TYPE_I32);
    GGML_ASSERT(state->type     == GGML_TYPE_F32);
    GGML_ASSERT(dur_table->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type       == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(dst) == GGML_TDT_STEP_N_OUTS);

    tdt_step_f32_kernel<<<1, 1, 0, ctx.stream()>>>(
        (const int *) token->data, (const int *) dur_idx->data,
        (const float *) state->data, (const float *) dur_table->data,
        (float *) dst->data,
        (int) ggml_nelements(dur_table),
        ggml_get_op_params_i32(dst, 0), ggml_get_op_params_i32(dst, 1), ggml_get_op_params_i32(dst, 2));
}
