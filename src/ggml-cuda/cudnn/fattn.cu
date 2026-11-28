#include "../convert.cuh"
#include "../fattn.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

namespace {
constexpr int64_t heads = 56;
constexpr int64_t channels = 128;
constexpr int64_t row_size = heads * channels;
constexpr size_t max_scratch_bytes = 512 * 1024 * 1024;

__global__ void pack_q(const float * src, half * dst, int64_t count, int64_t token_stride, int64_t head_stride) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const int64_t token = i / row_size;
        const int64_t head  = i / channels % heads;
        dst[i] = __float2half_rn(src[token * token_stride + head * head_stride + i % channels]);
    }
}

static bool valid_shape(const ggml_tensor * t, ggml_type type, int64_t tokens) {
    return t->type == type && t->ne[0] == channels && t->ne[1] == tokens && t->ne[2] == heads && t->ne[3] == 1;
}

static bool valid_kv(const ggml_tensor * t, int64_t tokens) {
    return valid_shape(t, GGML_TYPE_F16, tokens) && t->nb[0] == sizeof(half) &&
           t->nb[1] % 16 == 0 && t->nb[2] % 16 == 0 && uintptr_t(t->data) % 16 == 0;
}
} // namespace

// One bounded plan on the backend's primary stream. Scratch is private so a rejected
// execution can safely fall back before anything writes the caller's destination.
struct ggml_cuda_cudnn_sdpa_plan {
    // Token count, K/V head and token strides, and the exact attention scale.
    using key_type = std::array<int64_t, 6>;
    key_type key;
    cudnnHandle_t handle = nullptr;
    fe::graph::Graph graph;
    void * scratch = nullptr;
    bool ready = false;
    bool reported = false;

    explicit ggml_cuda_cudnn_sdpa_plan(key_type key, float scale) : key(key) {
        if (cudnnGetVersion() / 10000 != 9) {
            GGML_LOG_WARN("cuDNN SDPA requires a cuDNN 9 runtime\n");
            return;
        }
        const auto create_status = cudnnCreate(&handle);
        if (create_status != CUDNN_STATUS_SUCCESS) {
            GGML_LOG_WARN("cuDNN SDPA: %s\n", cudnnGetErrorString(create_status));
            return;
        }
        const int64_t tokens = key[0];
        const std::vector<int64_t> dims   = {1, heads, tokens, channels};
        const std::vector<int64_t> packed = {row_size * tokens, channels, row_size, 1};
        graph.set_io_data_type(fe::DataType_t::HALF);
        graph.set_intermediate_data_type(fe::DataType_t::FLOAT);
        graph.set_compute_data_type(fe::DataType_t::FLOAT);
        auto tensor = [&](const char * name, int64_t uid, const std::vector<int64_t> & stride) {
            return graph.tensor(fe::graph::Tensor_attributes().set_name(name).set_uid(uid).set_dim(dims).set_stride(stride));
        };
        auto q = tensor("Q", 1, packed);
        auto k = tensor("K", 2, {row_size * tokens, key[1], key[2], 1});
        auto v = tensor("V", 3, {row_size * tokens, key[3], key[4], 1});
        auto outputs = graph.sdpa(q, k, v, fe::graph::SDPA_attributes().set_generate_stats(false).set_attn_scale(scale));
        outputs[0]->set_output(true).set_uid(4).set_dim(dims).set_stride(packed);
        auto status = graph.build(handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
        int64_t workspace = 0;
        if (!status.is_good()) {
            GGML_LOG_WARN("cuDNN SDPA: plan rejected for %lld tokens: %s\n", (long long) tokens, status.get_message().c_str());
            return;
        }
        auto workspace_status = graph.get_workspace_size(workspace);
        if (!workspace_status.is_good()) {
            GGML_LOG_WARN("cuDNN SDPA: workspace query failed: %s\n", workspace_status.get_message().c_str());
            return;
        }
        const size_t io_bytes = 2 * size_t(row_size * tokens) * sizeof(half);
        if (workspace < 0 || uint64_t(std::max<int64_t>(1, workspace)) > max_scratch_bytes - io_bytes) {
            return;
        }
        auto error = cudaMalloc(&scratch, io_bytes + size_t(std::max<int64_t>(1, workspace)));
        if (error == cudaErrorMemoryAllocation) {
            // Optional workspace allocation may fail while the stock op still fits.
            const auto pending = cudaGetLastError();
            if (pending != cudaSuccess && pending != cudaErrorMemoryAllocation) {
                CUDA_CHECK(pending);
            }
            GGML_LOG_WARN("cuDNN SDPA: insufficient scratch memory for %lld tokens\n", (long long) tokens);
            return;
        }
        CUDA_CHECK(error);
        ready = true;
    }

    void disable() {
        ready = false;
        if (scratch) {
            CUDA_CHECK(cudaFree(scratch));
            scratch = nullptr;
        }
    }

    ~ggml_cuda_cudnn_sdpa_plan() {
        disable();
        if (handle) {
            cudnnDestroy(handle);
        }
    }
};

void ggml_cuda_cudnn_sdpa_deleter::operator()(ggml_cuda_cudnn_sdpa_plan * plan) const {
    delete plan;
}

bool ggml_cuda_h3_cudnn_sdpa_enabled(int device) {
    const char * enable = std::getenv("GGML_CUDA_H3_CUDNN_SDPA");
    return (!enable || std::strcmp(enable, "1") == 0) &&
        ggml_cuda_info().devices[device].cc == GGML_CUDA_CC_DGX_SPARK;
}

bool ggml_cuda_try_cudnn_sdpa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ctx.curr_stream_no != 0 || !ggml_cuda_h3_cudnn_sdpa_enabled(ctx.device)) {
        return false;
    }
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    const int64_t tokens = q->ne[1];
    if (dst->src[3] || dst->src[4] || tokens < 256 ||
        uint64_t(tokens) > max_scratch_bytes / (2 * sizeof(half) * row_size) ||
        !valid_shape(q, GGML_TYPE_F32, tokens) || q->nb[0] != sizeof(float) || q->nb[1] % sizeof(float) != 0 ||
        q->nb[2] % sizeof(float) != 0 || !valid_kv(k, tokens) || !valid_kv(v, tokens) ||
        dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst)) {
        return false;
    }
    float params[3];
    std::memcpy(params, dst->op_params, sizeof(params));
    if (!std::isfinite(params[0]) || params[1] != 0.0f || params[2] != 0.0f) {
        return false;
    }
    auto stream = ctx.stream();
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        return false;
    }
    uint32_t scale_bits;
    std::memcpy(&scale_bits, &params[0], sizeof(scale_bits));
    const ggml_cuda_cudnn_sdpa_plan::key_type key = {tokens, int64_t(k->nb[2] / sizeof(half)), int64_t(k->nb[1] / sizeof(half)),
                                                    int64_t(v->nb[2] / sizeof(half)), int64_t(v->nb[1] / sizeof(half)), scale_bits};
    auto & plan = ctx.cudnn_sdpa;
    if (!plan || plan->key != key) {
        // Release old scratch before allocating for a new sequence length.
        plan.reset();
        plan.reset(new ggml_cuda_cudnn_sdpa_plan(key, params[0]));
    }
    if (!plan->ready) {
        return false;
    }
    const auto stream_status = cudnnSetStream(plan->handle, stream);
    if (stream_status != CUDNN_STATUS_SUCCESS) {
        GGML_LOG_WARN("cuDNN SDPA: %s\n", cudnnGetErrorString(stream_status));
        plan->disable();
        return false;
    }
    const int64_t count = tokens * row_size;
    auto * q_half = static_cast<half *>(plan->scratch);
    auto * output = q_half + count;
    void * workspace = output + count;
    constexpr int threads = 256;
    pack_q<<<(count + threads - 1) / threads, threads, 0, stream>>>(
        static_cast<const float *>(q->data), q_half, count, q->nb[1] / sizeof(float), q->nb[2] / sizeof(float));
    CUDA_CHECK(cudaGetLastError());
    std::unordered_map<int64_t, void *> buffers = {{1, q_half}, {2, k->data}, {3, v->data}, {4, output}};
    auto status = plan->graph.execute(plan->handle, buffers, workspace);
    if (!status.is_good()) {
        // Propagate a real CUDA fault. On a recoverable cuDNN error, retire this plan.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        GGML_LOG_WARN("cuDNN SDPA: execution failed: %s\n", status.get_message().c_str());
        plan->disable();
        return false;
    }
    ggml_get_to_fp32_cuda(GGML_TYPE_F16)(output, static_cast<float *>(dst->data), count, stream);
    CUDA_CHECK(cudaGetLastError());
    if (!plan->reported) {
        GGML_LOG_INFO("cuDNN SDPA active: tokens=%lld\n", (long long) tokens);
        plan->reported = true;
    }
    return true;
}
