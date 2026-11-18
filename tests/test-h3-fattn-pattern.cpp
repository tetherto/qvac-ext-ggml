#include "ggml-cuda/h3-fattn.h"

#include <cstdio>
#include <limits>
#include <stdexcept>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

struct Fixture {
    ggml_context * ctx;
    ggml_tensor * output;

    Fixture(int tokens = 1025, int heads = 56, int batch = 1) {
        ctx = ggml_init({ggml_tensor_overhead() * 32, nullptr, true});
        require(ctx != nullptr, "context allocation failed");
        auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, tokens, heads, batch);
        auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 128, tokens, heads, batch);
        auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 128, tokens, heads, batch);
        output = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 128.f / std::sqrt(128.f), 0.f, 0.f);
        ggml_flash_attn_ext_set_prec(output, GGML_PREC_F32);
    }

    ~Fixture() { ggml_free(ctx); }
    bool supported() const { return ggml_cuda_h3::fattn_tuning_supported(output); }
};

int main() {
    try {
        for (int tokens : {1024, 1025, 15429, 32768}) {
            Fixture f(tokens); require(f.supported(), "supported dense H3 shape rejected");
        }
        for (int tokens : {1, 32, 1023, 32769}) {
            Fixture f(tokens); require(!f.supported(), "unsupported token count accepted");
        }
        { Fixture f(1025, 28); require(!f.supported(), "unsupported head count accepted"); }
        { Fixture f(1025, 56, 2); require(!f.supported(), "unsupported batch accepted"); }
        for (int i : {0, 1, 2}) {
            Fixture f; f.output->src[i]->nb[1] += 16;
            require(!f.supported(), "strided input accepted");
        }
        { Fixture f; f.output->src[1]->ne[2] = 28; require(!f.supported(), "GQA accepted"); }
        { Fixture f; --f.output->src[2]->ne[1]; require(!f.supported(), "different KV lengths accepted"); }
        { Fixture f; f.output->src[1]->type = GGML_TYPE_BF16; require(!f.supported(), "changed precision accepted"); }
        { Fixture f; f.output->src[3] = f.output->src[1]; require(!f.supported(), "mask accepted"); }
        { Fixture f; f.output->src[4] = f.output->src[1]; require(!f.supported(), "attention sinks accepted"); }
        { Fixture f; f.output->view_src = f.output->src[0]; require(!f.supported(), "aliased output metadata accepted"); }
        { Fixture f; --f.output->ne[1]; require(!f.supported(), "wrong output shape accepted"); }
        { Fixture f; ggml_flash_attn_ext_set_prec(f.output, GGML_PREC_DEFAULT); require(!f.supported(), "different precision contract accepted"); }
        for (float scale : {0.f, -1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
            Fixture f; std::memcpy(f.output->op_params, &scale, sizeof(scale));
            require(!f.supported(), "invalid scale accepted");
        }
        for (int index : {1, 2}) {
            Fixture f; float params[] = {1.f, 0.f, 0.f}; params[index] = 1.f;
            std::memcpy(f.output->op_params, params, sizeof(params));
            require(!f.supported(), "ALiBi or softcap accepted");
        }
        require(!ggml_cuda_h3::fattn_tuning_supported(nullptr), "null output accepted");
        std::puts("PASS H3 attention shape, layout, precision and conditioning guards");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
