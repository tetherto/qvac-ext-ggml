#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <initializer_list>

// Simulate a backend whose ordinary MUL_MAT capability check ignores hints.
// This covers unsupported GPU registries even on hosts without their drivers.
static bool accepts_every_op(ggml_backend_dev_t, const ggml_tensor *) { return true; }
static const char * registry_name(ggml_backend_reg_t reg) {
    return static_cast<const char *>(reg->context);
}

int main() {
    ggml_backend_load_all();
    auto * ctx = ggml_init({ggml_tensor_overhead() * 16, nullptr, true});
    bool ok = true;
    for (auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}) {
        auto * codes = ggml_new_tensor_2d(ctx, type, 32, 64);
        auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 1);
        auto * step = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        auto * minimum = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        auto * affine = ggml_mul_mat_affine_v1(ctx, codes, x, step, minimum);
        auto * ordinary = ggml_mul_mat(ctx, codes, x);
        for (const char * name : {"CUDA", "Vulkan", "OpenCL", "SYCL", "RPC", "Hexagon", "CPU", "BLAS", "MTL"}) {
            ggml_backend_reg reg = {};
            reg.iface.get_name = registry_name;
            reg.context = const_cast<char *>(name);
            ggml_backend_device dev = {};
            dev.reg = &reg;
            dev.iface.supports_op = accepts_every_op;
            ggml_backend backend = {};
            backend.device = &dev;
            const bool expected = std::strcmp(name, "MTL") == 0;
            if (ggml_backend_dev_supports_op(&dev, affine) != expected ||
                    ggml_backend_supports_op(&backend, affine) != expected ||
                    !ggml_backend_dev_supports_op(&dev, ordinary)) {
                std::fprintf(stderr, "FAIL: %s affine/ordinary capability contract\n", name);
                ok = false;
            }
        }
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * dev = ggml_backend_dev_get(i);
            const bool metal = std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "MTL") == 0;
            if (metal && ggml_backend_dev_supports_op(dev, affine) != dev->iface.supports_op(dev, affine)) {
                std::fprintf(stderr, "FAIL: Metal capability query was overridden\n");
                ok = false;
            }
            if (!metal && ggml_backend_dev_supports_op(dev, affine)) {
                std::fprintf(stderr, "FAIL: %s accepts affine carriers\n", ggml_backend_dev_name(dev));
                ok = false;
            }
        }
    }
    ggml_free(ctx);
    return ok ? 0 : 1;
}
