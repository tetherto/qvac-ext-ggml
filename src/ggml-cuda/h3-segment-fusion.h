#ifndef GGML_CUDA_H3_SEGMENT_FUSION_H
#define GGML_CUDA_H3_SEGMENT_FUSION_H

#include "ggml-impl.h"

namespace ggml_cuda_h3 {

struct segment_fusion_match {
    const ggml_tensor * base = nullptr;
    const ggml_tensor * update = nullptr;
    const ggml_tensor * scale = nullptr;
    const ggml_tensor * shift = nullptr;
    ggml_tensor * output = nullptr;
    int last = 0;
};

static inline bool segment_input(const ggml_tensor * tensor) {
    if (!tensor || tensor->type != GGML_TYPE_F32 || tensor->nb[0] != sizeof(float) ||
        tensor->ne[0] < 4 || tensor->ne[0] > 65536 || tensor->ne[0] % 4 != 0 ||
        tensor->ne[1] < 1 || tensor->ne[1] > 65535 || tensor->ne[2] < 1 ||
        tensor->ne[2] > 65535 || tensor->ne[3] != 1) {
        return false;
    }
    return tensor->nb[1] % 16 == 0 && tensor->nb[2] % 16 == 0 &&
        tensor->nb[1] >= size_t(tensor->ne[0]) * sizeof(float) &&
        tensor->nb[2] >= size_t(tensor->ne[1]) * tensor->nb[1];
}

static inline bool segment_vector(const ggml_tensor * tensor, int64_t width) {
    return tensor && tensor->type == GGML_TYPE_F32 && tensor->ne[0] == width &&
        tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1 && ggml_is_contiguous(tensor);
}

static inline bool segment_result(const ggml_tensor * tensor, const ggml_tensor * source) {
    return tensor && tensor->type == GGML_TYPE_F32 && !tensor->view_src &&
        ggml_is_contiguous(tensor) && ggml_are_same_shape(tensor, source);
}

static inline bool segment_memory_safe(const segment_fusion_match & match) {
    const auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const uintptr_t ap = (uintptr_t) a->data;
        const uintptr_t bp = (uintptr_t) b->data;
        const size_t as = ggml_nbytes(a);
        const size_t bs = ggml_nbytes(b);
        return ap <= bp ? bp - ap < as : ap - bp < bs;
    };
    const auto pointwise = [&](const ggml_tensor * source) {
        return !overlap(match.output, source) ||
            (match.output->data == source->data && ggml_are_same_layout(match.output, source));
    };
    // Each lane reads its own base/update before writing; broadcast parameters cannot alias the output.
    return pointwise(match.base) && pointwise(match.update) && !overlap(match.output, match.scale) &&
        (!match.shift || !overlap(match.output, match.shift));
}

static inline bool match_segment_fusion(const ggml_cgraph * graph, int first, segment_fusion_match & match) {
    if (first < 0 || first >= graph->n_nodes - 1) {
        return false;
    }
    const auto * product = graph->nodes[first];
    auto * sum = graph->nodes[first + 1];
    if (product->op != GGML_OP_MUL || sum->op != GGML_OP_ADD || sum->src[1] != product ||
        !segment_input(product->src[0]) || !segment_input(sum->src[0])) {
        return false;
    }
    const auto * update = product->src[0];
    const auto * base = sum->src[0];
    if (!ggml_are_same_shape(base, update) || !segment_vector(product->src[1], update->ne[0]) ||
        !segment_result(product, update) || !segment_result(sum, update)) {
        return false;
    }
    int last = first + 1;
    int indices[] = {first, first + 1, 0};
    const ggml_op operations[] = {GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD};
    int count = 2;
    const ggml_tensor * shift = nullptr;
    ggml_tensor * output = sum;
    if (base == update) {
        // The shift may be a newly visited read-only view between the two adds.
        const ggml_tensor * metadata = nullptr;
        if (++last >= graph->n_nodes) {
            return false;
        }
        if (graph->nodes[last]->op == GGML_OP_VIEW) {
            metadata = graph->nodes[last];
            if ((metadata->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(graph, last) != 1 ||
                !metadata->src[0] || metadata->src[0] == product || metadata->src[0] == sum ||
                ++last >= graph->n_nodes) {
                return false;
            }
        }
        output = graph->nodes[last];
        shift = output->src[1];
        if (output->op != GGML_OP_ADD || output->src[0] != sum || !segment_result(output, update) ||
            !segment_vector(shift, update->ne[0]) || (metadata && metadata != shift)) {
            return false;
        }
        indices[2] = last;
        count = 3;
    }
    if (!ggml_can_fuse_subgraph_ext(graph, indices, count, operations, &last, 1)) {
        return false;
    }
    match.base = base;
    match.update = update;
    match.scale = product->src[1];
    match.shift = shift;
    match.output = output;
    match.last = last;
    return true;
}

} // namespace ggml_cuda_h3

#endif
