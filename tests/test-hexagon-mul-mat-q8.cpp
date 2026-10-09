#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static constexpr int     SKIP_RETURN_CODE     = 77;
static constexpr int64_t BLOCK_ELEMENTS       = 32;
static constexpr int     BLOCK_MAGNITUDES     = 4;
static constexpr float   FP16_OVERFLOW_VALUE  = 1.0e5f;
static constexpr double  MAX_NMSE_VS_CPU      = 1e-10;
static constexpr double  MAX_NMSE_Q4_1_VS_CPU = 1e-4;
static constexpr int64_t STRIDED_COLUMNS      = 63;
static constexpr int64_t STRIDED_ROW_PADDING  = 32;

struct mul_mat_shape {
    int64_t k;
    int64_t m;
};

struct activation_pattern {
    const char * name;
    float (*value)(int64_t);
};

struct mul_mat_result {
    std::vector<float> one_column;
    std::vector<float> two_columns;
    std::vector<float> strided_columns;
};

static float weight_value(int64_t i) {
    return std::sin(float(i) * 0.71f) * 0.05f;
}

static float block_varying_activation(int64_t i) {
    const float exponent = float((i / BLOCK_ELEMENTS) % BLOCK_MAGNITUDES) - 2.0f;
    return std::sin(float(i) * 0.37f) * std::pow(10.0f, exponent);
}

static float fp16_overflow_activation(int64_t i) {
    return i % (BLOCK_ELEMENTS * BLOCK_MAGNITUDES) == 7 ? FP16_OVERFLOW_VALUE : std::sin(float(i) * 0.37f);
}

static const activation_pattern BLOCK_MAGNITUDES_PATTERN = { "block-magnitudes", block_varying_activation };
static const activation_pattern FP16_OVERFLOW_PATTERN    = { "fp16-overflow", fp16_overflow_activation };

static std::vector<float> fill(int64_t n, float (*value)(int64_t)) {
    std::vector<float> out(n);
    for (int64_t i = 0; i < n; i++) {
        out[i] = value(i);
    }
    return out;
}

static std::vector<float> two_identical_columns(const std::vector<float> & column) {
    std::vector<float> out(column);
    out.insert(out.end(), column.begin(), column.end());
    return out;
}

static std::vector<float> padded_rotated_columns(const std::vector<float> & column, int64_t n_columns, int64_t padding) {
    const int64_t      k = (int64_t) column.size();
    std::vector<float> out((size_t) ((k + padding) * n_columns), FP16_OVERFLOW_VALUE);
    for (int64_t c = 0; c < n_columns; c++) {
        std::rotate_copy(column.begin(), column.begin() + (c * BLOCK_ELEMENTS) % k, column.end(),
                         out.begin() + c * (k + padding));
    }
    return out;
}

static mul_mat_result run_mul_mat(ggml_backend_t backend, ggml_backend_buffer_type_t weight_buft, ggml_type type,
                                  const mul_mat_shape & s, const activation_pattern & a) {
    ggml_context * wctx   = ggml_init({ ggml_tensor_overhead() * 2, nullptr, true });
    ggml_tensor *  weight = ggml_new_tensor_2d(wctx, type, s.k, s.m);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, weight_buft);
    GGML_ASSERT(wbuf);
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    const std::vector<float> w = fill(s.k * s.m, weight_value);
    std::vector<uint8_t>     q(ggml_nbytes(weight));
    ggml_quantize_chunk(type, w.data(), q.data(), 0, s.m, s.k, nullptr);
    ggml_backend_tensor_set(weight, q.data(), 0, q.size());

    ggml_context * ctx = ggml_init({ ggml_tensor_overhead() * 12 + ggml_graph_overhead(), nullptr, true });
    ggml_tensor *  x1  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.k, 1);
    ggml_tensor *  x2  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.k, 2);
    ggml_tensor *  xp  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.k + STRIDED_ROW_PADDING, STRIDED_COLUMNS);
    ggml_tensor *  xs  = ggml_view_2d(ctx, xp, s.k, STRIDED_COLUMNS, xp->nb[1], 0);
    ggml_tensor *  y1  = ggml_mul_mat(ctx, weight, x1);
    ggml_tensor *  y2  = ggml_mul_mat(ctx, weight, x2);
    ggml_tensor *  ys  = ggml_mul_mat(ctx, weight, xs);
    ggml_mul_mat_set_prec(y1, GGML_PREC_F32);
    ggml_mul_mat_set_prec(y2, GGML_PREC_F32);
    ggml_mul_mat_set_prec(ys, GGML_PREC_F32);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y1);
    ggml_build_forward_expand(gf, y2);
    ggml_build_forward_expand(gf, ys);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf);

    const std::vector<float> column  = fill(s.k, a.value);
    const std::vector<float> both    = two_identical_columns(column);
    const std::vector<float> strided = padded_rotated_columns(column, STRIDED_COLUMNS, STRIDED_ROW_PADDING);
    ggml_backend_tensor_set(x1, column.data(), 0, column.size() * sizeof(float));
    ggml_backend_tensor_set(x2, both.data(), 0, both.size() * sizeof(float));
    ggml_backend_tensor_set(xp, strided.data(), 0, strided.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    mul_mat_result r;
    r.one_column.resize(s.m);
    r.two_columns.resize(2 * s.m);
    r.strided_columns.resize(STRIDED_COLUMNS * s.m);
    ggml_backend_tensor_get(y1, r.one_column.data(), 0, r.one_column.size() * sizeof(float));
    ggml_backend_tensor_get(y2, r.two_columns.data(), 0, r.two_columns.size() * sizeof(float));
    ggml_backend_tensor_get(ys, r.strided_columns.data(), 0, r.strided_columns.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    return r;
}

static double nmse(const std::vector<float> & ref, const float * got) {
    double err = 0.0, norm = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        const double d = double(got[i]) - double(ref[i]);
        err += d * d;
        norm += double(ref[i]) * double(ref[i]);
    }
    return norm > 0.0 ? err / norm : err;
}

static bool same_bits(const float * a, const float * b, int64_t n) {
    return std::memcmp(a, b, size_t(n) * sizeof(float)) == 0;
}

static ggml_backend_buffer_type_t hexagon_weight_buft(ggml_backend_dev_t dev, ggml_backend_t hexagon) {
    auto extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");
    ggml_backend_buffer_type_t * bufts = extra ? extra(dev) : nullptr;
    return bufts && bufts[0] ? bufts[0] : ggml_backend_get_default_buffer_type(hexagon);
}

static bool check_case(ggml_backend_t cpu, ggml_backend_t hexagon, ggml_backend_buffer_type_t buft, ggml_type type,
                       const mul_mat_shape & s, const activation_pattern & a, double max_nmse) {
    const mul_mat_result ref = run_mul_mat(cpu, ggml_backend_get_default_buffer_type(cpu), type, s, a);
    const mul_mat_result got = run_mul_mat(hexagon, buft, type, s, a);

    const double err          = nmse(ref.one_column, got.one_column.data());
    const double strided_err  = nmse(ref.strided_columns, got.strided_columns.data());
    const bool   columns_same = same_bits(got.two_columns.data(), got.two_columns.data() + s.m, s.m);
    const bool   batch_same   = same_bits(got.one_column.data(), got.two_columns.data(), s.m);
    const bool   ok = err <= max_nmse && strided_err <= max_nmse && columns_same && batch_same;
    std::printf("%s %s k=%lld m=%lld: nmse-vs-cpu=%.3e strided=%.3e columns=%s n1-vs-n2=%s %s\n",
                ggml_type_name(type), a.name, (long long) s.k, (long long) s.m, err, strided_err,
                columns_same ? "same" : "DIFF", batch_same ? "same" : "DIFF", ok ? "OK" : "FAIL");
    return ok;
}

static bool check_shapes(ggml_backend_t cpu, ggml_backend_t hexagon, ggml_backend_buffer_type_t buft, ggml_type type,
                         const activation_pattern & a, double max_nmse) {
    const mul_mat_shape shapes[] = { { 1024, 256 }, { 3072, 1024 }, { 1024, 32 } };
    bool ok = true;
    for (const mul_mat_shape & s : shapes) {
        ok = check_case(cpu, hexagon, buft, type, s, a, max_nmse) && ok;
    }
    return ok;
}

static bool check_type(ggml_backend_t cpu, ggml_backend_t hexagon, ggml_backend_buffer_type_t buft, ggml_type type) {
    const activation_pattern patterns[] = { BLOCK_MAGNITUDES_PATTERN, FP16_OVERFLOW_PATTERN };
    bool ok = true;
    for (const activation_pattern & a : patterns) {
        ok = check_shapes(cpu, hexagon, buft, type, a, MAX_NMSE_VS_CPU) && ok;
    }
    return ok;
}

static bool check_q4_1(ggml_backend_t cpu, ggml_backend_t hexagon, ggml_backend_buffer_type_t buft) {
    return check_shapes(cpu, hexagon, buft, GGML_TYPE_Q4_1, BLOCK_MAGNITUDES_PATTERN, MAX_NMSE_Q4_1_VS_CPU);
}

static bool check_types(ggml_backend_t cpu, ggml_backend_t hexagon, ggml_backend_buffer_type_t buft) {
    const ggml_type types[] = { GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_IQ4_NL, GGML_TYPE_MXFP4 };
    bool ok = true;
    for (ggml_type type : types) {
        ok = check_type(cpu, hexagon, buft, type) && ok;
    }
    return check_q4_1(cpu, hexagon, buft) && ok;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("HTP0");
    if (!dev) {
        std::puts("SKIP: Hexagon device unavailable");
        return SKIP_RETURN_CODE;
    }
    ggml_backend_t cpu     = ggml_backend_init_by_name("CPU", nullptr);
    ggml_backend_t hexagon = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(cpu && hexagon);

    const bool ok = check_types(cpu, hexagon, hexagon_weight_buft(dev, hexagon));

    ggml_backend_free(hexagon);
    ggml_backend_free(cpu);
    ggml_quantize_free();
    return ok ? 0 : 1;
}
