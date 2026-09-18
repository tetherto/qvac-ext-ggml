// ConvRot via the rotation op + standard quantized matmul.
//
// Runs on every backend found through the registry (static or GGML_BACKEND_DL).
// The oracle rotates the *weights* (the documented ConvRot definition) so the
// implementation, which rotates the *activations*, is checked independently.

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int32_t kGroup = 256;

void hadamard_4(float * v, size_t stride) {
    const float a = v[0], b = v[stride], c = v[2*stride], d = v[3*stride];
    v[0]        = ( a + b + c - d)*0.5f;
    v[stride]   = ( a + b - c + d)*0.5f;
    v[2*stride] = ( a - b + c + d)*0.5f;
    v[3*stride] = (-a + b + c + d)*0.5f;
}

void rotate_block(float * v, int n) {
    for (size_t stride = 1; stride < (size_t) n; stride *= 4)
        for (size_t base = 0; base < (size_t) n; base += 4*stride)
            for (size_t i = 0; i < stride; ++i) hadamard_4(v + base + i, stride);
}

struct result { bool ok = true; double max_err = 0.0; };

// ---- rotation op alone, on a strided view and in place --------------------

result test_rotation(ggml_backend_t be, bool inplace) {
    result r;
    const int64_t k = 2*kGroup, rows = 3, planes = 2;
    ggml_init_params ip = { 4*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    // storage has twice the rows; the op input is a view of every second row
    ggml_tensor * storage = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, inplace ? rows : 2*rows, planes);
    ggml_tensor * x = inplace ? storage
        : ggml_view_3d(ctx, storage, k, rows, planes, 2*storage->nb[1], storage->nb[2], 0);
    ggml_tensor * y = inplace ? ggml_convrot_inplace(ctx, x, kGroup) : ggml_convrot(ctx, x, kGroup);
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, y);

    if (!ggml_backend_supports_op(be, y)) {
        std::printf("    rotation%s: not supported, skipped\n", inplace ? " (in place)" : "");
        ggml_free(ctx);
        return r;
    }

    std::vector<float> data(ggml_nelements(storage));
    for (size_t i = 0; i < data.size(); ++i) data[i] = float((i*37 + 11) % 101) / 50.0f - 1.0f;

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(storage, data.data(), 0, ggml_nbytes(storage));
    r.ok = ggml_backend_graph_compute(be, g) == GGML_STATUS_SUCCESS;

    std::vector<float> out(ggml_nelements(y));
    if (r.ok) ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));

    for (int64_t p = 0; p < planes && r.ok; ++p) for (int64_t row = 0; row < rows; ++row) {
        const int64_t srow = inplace ? row : 2*row;
        std::vector<float> ref(data.begin() + (p*storage->ne[1] + srow)*k, data.begin() + (p*storage->ne[1] + srow)*k + k);
        for (int64_t k0 = 0; k0 < k; k0 += kGroup) rotate_block(ref.data() + k0, kGroup);
        for (int64_t i = 0; i < k; ++i) {
            const float got = out[(p*rows + row)*k + i];
            const double err = std::fabs(got - ref[i]) / (1.0 + std::fabs(ref[i]));
            r.max_err = std::fmax(r.max_err, err);
            if (err > 1e-5) r.ok = false;
        }
    }
    std::printf("    rotation%s: %s (max rel err %.2e)\n", inplace ? " (in place)" : "", r.ok ? "ok" : "FAILED", r.max_err);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

// ---- full linear: repack I8 -> Q8_0, rotate activations, stock matmul ------

result test_linear(ggml_backend_t be, int64_t m) {
    result r;
    const int64_t k = 2*kGroup, n = 37;   // n deliberately not a multiple of any tile size
    ggml_init_params ip = { 16*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    ggml_tensor * wq = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, n);
    ggml_tensor * y  = ggml_convrot_mul_mat(ctx, wq, x, kGroup);
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, y);

    bool supported = true;
    for (int i = 0; i < ggml_graph_n_nodes(g); ++i) supported = supported && ggml_backend_supports_op(be, ggml_graph_node(g, i));
    if (!supported) {
        std::printf("    linear M=%lld: not supported, skipped\n", (long long) m);
        ggml_free(ctx);
        return r;
    }

    // ConvRot source format: I8 [k, n] + one F32 scale per output row (row 1 is an all-zero row)
    std::vector<int8_t> w(k*n);
    std::vector<float>  s(n);
    for (int64_t row = 0; row < n; ++row) {
        s[row] = row == 1 ? 0.0f : 0.0007f + 0.00001f*float(row*7 % 13);
        for (int64_t i = 0; i < k; ++i) w[row*k + i] = row == 1 ? 0 : int8_t((i*17 + row*31) % 255 - 127);
    }
    std::vector<float> xv(k*m);
    for (size_t i = 0; i < xv.size(); ++i) xv[i] = float(int64_t((i*13 + 5) % 97) - 48) * 0.03125f;

    const size_t need = ggml_convrot_repack_q8_0(nullptr, nullptr, k, n, nullptr);
    if (need != ggml_nbytes(wq)) {
        std::printf("    repack size mismatch: %zu vs %zu\n", need, ggml_nbytes(wq));
        r.ok = false;
    }
    std::vector<uint8_t> q8(need);
    ggml_convrot_repack_q8_0(w.data(), s.data(), k, n, q8.data());

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(x,  xv.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(wq, q8.data(), 0, ggml_nbytes(wq));
    r.ok = r.ok && ggml_backend_graph_compute(be, g) == GGML_STATUS_SUCCESS;
    std::vector<float> out(n*m);
    if (r.ok) ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));

    // oracle: dequantize with the stored (F16-rounded) scale, rotate the WEIGHTS, dot with raw activations
    double max_abs_ref = 0.0;
    std::vector<double> ref(n*m);
    std::vector<float> wrow(k);
    for (int64_t row = 0; row < n; ++row) {
        const float sc = ggml_fp16_to_fp32(ggml_fp32_to_fp16(s[row]));
        for (int64_t i = 0; i < k; ++i) wrow[i] = float(w[row*k + i]) * sc;
        for (int64_t k0 = 0; k0 < k; k0 += kGroup) rotate_block(wrow.data() + k0, kGroup);
        for (int64_t c = 0; c < m; ++c) {
            double acc = 0.0;
            for (int64_t i = 0; i < k; ++i) acc += double(wrow[i]) * double(xv[c*k + i]);
            ref[c*n + row] = acc;
            max_abs_ref = std::fmax(max_abs_ref, std::fabs(acc));
        }
    }
    // Standard ggml quantized-matmul accuracy: activations are quantized to
    // Q8_0 (CPU) or cast to F16 (GPU batched path) inside the stock kernels.
    const double tol = 1e-2 * max_abs_ref;
    for (size_t i = 0; i < out.size() && r.ok; ++i) {
        if (!std::isfinite(out[i])) {
            std::printf("    linear M=%lld: non-finite output at %zu (%f); out[0..3] = %f %f %f %f, ref[0..3] = %f %f %f %f\n",
                (long long) m, i, out[i], out[0], out[1], out[2], out[3], ref[0], ref[1], ref[2], ref[3]);
            r.ok = false;
            break;
        }
        const double err = std::fabs(out[i] - ref[i]);
        r.max_err = std::fmax(r.max_err, err / max_abs_ref);
        if (err > tol) {
            std::printf("    linear M=%lld: mismatch at %zu: got %.6f expected %.6f\n", (long long) m, i, out[i], ref[i]);
            r.ok = false;
        }
    }
    // the all-zero row must be exactly zero
    for (int64_t c = 0; c < m && r.ok; ++c) if (out[c*n + 1] != 0.0f) { std::printf("    zero-scale row not zero\n"); r.ok = false; }

    std::printf("    linear M=%-3lld: %s (max err %.2e of max |ref|)\n", (long long) m, r.ok ? "ok" : "FAILED", r.max_err);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

} // namespace

int main() {
    ggml_backend_load_all();

    bool all_ok = true;
    int tested = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (!be) continue;
        std::printf("backend %s (%s)\n", ggml_backend_name(be), ggml_backend_dev_description(dev));
        bool ok = true;
        ok = test_rotation(be, false).ok && ok;
        ok = test_rotation(be, true).ok && ok;
        for (int64_t m : { int64_t(1), int64_t(2), int64_t(48) }) ok = test_linear(be, m).ok && ok;
        all_ok = all_ok && ok;
        ++tested;
        ggml_backend_free(be);
    }
    if (tested == 0) { std::fprintf(stderr, "no backend available\n"); return 1; }
    std::puts(all_ok ? "ConvRot decomposition test passed" : "ConvRot decomposition test FAILED");
    return all_ok ? 0 : 1;
}
