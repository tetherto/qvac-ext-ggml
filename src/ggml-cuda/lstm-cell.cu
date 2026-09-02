#include "lstm-cell.cuh"
#include "unary.cuh"

// Fused LSTM cell.  gates [4H, N] holds the pre-activations chunked along ne0 as
// i | f | g | o; dst [2H, N] holds h_new above c_new.
static __global__ void lstm_cell_f32_kernel(
        const float * gates, const float * c_prev, float * dst,
        const int64_t H, const int64_t total) {
    const int64_t k = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= total) {
        return;
    }

    const int64_t n = k / H;
    const int64_t j = k - n*H;

    const float * g = gates + n*GGML_LSTM_N_GATES*H;

    const float gi = ggml_cuda_op_sigmoid_single(g[GGML_LSTM_GATE_INPUT *H + j]);
    const float gf = ggml_cuda_op_sigmoid_single(g[GGML_LSTM_GATE_FORGET*H + j]);
    const float gg = ggml_cuda_op_tanh_single   (g[GGML_LSTM_GATE_CELL  *H + j]);
    const float go = ggml_cuda_op_sigmoid_single(g[GGML_LSTM_GATE_OUTPUT*H + j]);

    // Rounded intrinsics: nvcc compiles with fast math and would otherwise contract the two
    // products into an FMA, which the decomposed mul/mul/add graph cannot do.
    const float c_new = __fadd_rn(__fmul_rn(gf, c_prev[k]), __fmul_rn(gi, gg));

    float * out = dst + n*GGML_LSTM_N_OUTS*H;

    out[GGML_LSTM_OUT_H*H + j] = __fmul_rn(go, ggml_cuda_op_tanh_single(c_new));
    out[GGML_LSTM_OUT_C*H + j] = c_new;
}

void ggml_cuda_op_lstm_cell(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * gates  = dst->src[0];
    const ggml_tensor * c_prev = dst->src[1];

    GGML_ASSERT(gates->type  == GGML_TYPE_F32);
    GGML_ASSERT(c_prev->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type    == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(gates));
    GGML_ASSERT(ggml_is_contiguous(c_prev));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t H     = c_prev->ne[0];
    const int64_t total = H*c_prev->ne[1];

    const int64_t num_blocks = (total + CUDA_LSTM_CELL_BLOCK_SIZE - 1) / CUDA_LSTM_CELL_BLOCK_SIZE;
    lstm_cell_f32_kernel<<<num_blocks, CUDA_LSTM_CELL_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const float *) gates->data, (const float *) c_prev->data, (float *) dst->data, H, total);
}
