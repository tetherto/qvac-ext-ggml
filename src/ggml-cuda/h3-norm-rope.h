#ifndef GGML_CUDA_H3_NORM_ROPE_H
#define GGML_CUDA_H3_NORM_ROPE_H

#include "ggml-impl.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace ggml_cuda_h3 {

struct norm_rope_match {
    const ggml_tensor * input = nullptr;
    const ggml_tensor * weight = nullptr;
    const ggml_tensor * pe = nullptr;
    ggml_tensor * output = nullptr;
    int rotation = 0;
    int last = 0;
    std::array<const ggml_tensor *, 32> kept{};
    std::array<const ggml_tensor *, 8> pe_metadata{};
    int n_kept = 0;
    int n_pe_metadata = 0;

    bool keep(const ggml_tensor * tensor) {
        if (!tensor) {
            return false;
        }
        if (std::find(kept.begin(), kept.begin() + n_kept, tensor) != kept.begin() + n_kept) {
            return true;
        }
        if (n_kept == (int) kept.size()) {
            return false;
        }
        kept[n_kept++] = tensor;
        return true;
    }
};

static inline const ggml_tensor * view_root(const ggml_tensor * tensor) {
    return tensor->view_src ? tensor->view_src : tensor;
}

static inline bool view_of(const ggml_tensor * tensor, const ggml_tensor * source, size_t offset) {
    if (!tensor || tensor->op != GGML_OP_VIEW || tensor->src[0] != source ||
        tensor->type != GGML_TYPE_F32 || tensor->nb[0] != sizeof(float) ||
        tensor->view_src != view_root(source) ||
        offset > std::numeric_limits<size_t>::max() - source->view_offs ||
        tensor->view_offs != source->view_offs + offset) {
        return false;
    }
    size_t parameter;
    memcpy(&parameter, tensor->op_params, sizeof(parameter));
    return parameter == offset;
}

static inline bool permute_is(const ggml_tensor * tensor, const std::array<int, 4> & axes) {
    if (!tensor || tensor->op != GGML_OP_PERMUTE || !tensor->src[0]) {
        return false;
    }
    const auto * source = tensor->src[0];
    if (tensor->type != source->type || tensor->view_src != view_root(source) ||
        tensor->view_offs != source->view_offs) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        if (ggml_get_op_params_i32(tensor, i) != axes[i] ||
            tensor->ne[axes[i]] != source->ne[i] || tensor->nb[axes[i]] != source->nb[i]) {
            return false;
        }
    }
    return true;
}

static inline bool contiguous_f32(const ggml_tensor * tensor) {
    return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor);
}

static inline bool concat_zero(const ggml_tensor * tensor) {
    return contiguous_f32(tensor) && tensor->op == GGML_OP_CONCAT &&
        tensor->src[0] && tensor->src[1] && ggml_get_op_params_i32(tensor, 0) == 0;
}

static inline bool half_shape(const ggml_tensor * tensor, const ggml_tensor * normalized, int half) {
    if (!tensor || tensor->type != GGML_TYPE_F32 || tensor->ne[0] != half) {
        return false;
    }
    for (int i = 1; i < 4; ++i) {
        if (tensor->ne[i] != normalized->ne[i]) {
            return false;
        }
    }
    return true;
}

static inline bool coefficient(norm_rope_match & match, const ggml_tensor * tensor,
                               int index, int half, int64_t tokens) {
    if (index < 0 || index >= 4 || match.n_pe_metadata > 6 ||
        !contiguous_f32(tensor) || tensor->op != GGML_OP_CONT || tensor->view_src ||
        tensor->ne[0] != half || tensor->ne[1] != 1 || tensor->ne[2] != tokens || tensor->ne[3] != 1 ||
        !permute_is(tensor->src[0], {1, 3, 0, 2})) {
        return false;
    }
    const auto * permute = tensor->src[0];
    const auto * view = permute->src[0];
    const auto * pe = view->src[0];
    if (!pe || pe->type != GGML_TYPE_F32 || pe->ne[0] != 2 || pe->ne[1] != 2 ||
        pe->ne[2] != half || pe->ne[3] != tokens ||
        view->ne[0] != 1 || view->ne[1] != 1 || view->ne[2] != half || view->ne[3] != tokens ||
        !ggml_are_same_shape(tensor, permute)) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        if (pe->nb[i] == 0 || pe->nb[i] % sizeof(float) != 0 ||
            (i > 0 && view->nb[i] != pe->nb[i])) {
            return false;
        }
    }
    const size_t row_offset = index >= 2 ? pe->nb[1] : 0;
    if (index % 2 && pe->nb[0] > std::numeric_limits<size_t>::max() - row_offset) {
        return false;
    }
    const size_t offset = row_offset + (index % 2 ? pe->nb[0] : 0);
    if (!view_of(view, pe, offset) || (match.pe && match.pe != pe)) {
        return false;
    }
    match.pe = pe;
    match.pe_metadata[match.n_pe_metadata++] = view;
    match.pe_metadata[match.n_pe_metadata++] = permute;
    return match.keep(tensor);
}

static inline bool match_output(const ggml_tensor * norm, ggml_tensor * output, norm_rope_match & match) {
    if (!contiguous_f32(output) || output->op != GGML_OP_CONT || output->view_src ||
        !permute_is(output->src[0], {0, 2, 1, 3})) {
        return false;
    }
    const auto * permute = output->src[0];
    const auto * full = permute->src[0];
    if (!concat_zero(full) || !ggml_are_same_shape(output, permute) ||
        full->ne[0] != 128 || output->ne[0] != 128 || output->ne[1] != norm->ne[2] ||
        output->ne[2] != norm->ne[1] || output->ne[3] != norm->ne[3]) {
        return false;
    }
    const ggml_tensor * rotated = full;
    const ggml_tensor * tail = nullptr;
    if (full->src[0]->op == GGML_OP_CONCAT) {
        rotated = full->src[0];
        tail = full->src[1];
    }
    if (!concat_zero(rotated) || (rotated->ne[0] != 96 && rotated->ne[0] != 128) ||
        ((rotated->ne[0] == 96) != (tail != nullptr))) {
        return false;
    }
    const int half = (int) rotated->ne[0] / 2;
    const auto * first = rotated->src[0];
    const auto * second = rotated->src[1];
    if (!first || !second || first->op != GGML_OP_ADD || second->op != GGML_OP_ADD ||
        !contiguous_f32(first) || !contiguous_f32(second)) {
        return false;
    }
    const ggml_tensor * products[] = {first->src[0], first->src[1], second->src[0], second->src[1]};
    for (const auto * product : products) {
        if (!contiguous_f32(product) || product->op != GGML_OP_MUL || !product->src[0] || !product->src[1]) {
            return false;
        }
    }
    const auto * a = products[0]->src[0];
    const auto * b = products[1]->src[0];
    const auto * normalized = a->src[0];
    if (!normalized || normalized->op != GGML_OP_MUL || normalized->src[0] != norm ||
        normalized->view_src != norm || normalized->view_offs != 0 ||
        !contiguous_f32(normalized) || !ggml_are_same_shape(normalized, norm) ||
        products[2]->src[0] != a || products[3]->src[0] != b || b->src[0] != normalized ||
        !view_of(a, normalized, 0) || !view_of(b, normalized, half * sizeof(float)) ||
        !half_shape(a, normalized, half) || !half_shape(b, normalized, half)) {
        return false;
    }
    const auto * weight = normalized->src[1];
    if (!contiguous_f32(weight) || weight->ne[0] != 128 || ggml_nelements(weight) != 128) {
        return false;
    }
    for (int i = 1; i < 4; ++i) {
        if (a->nb[i] != normalized->nb[i] || b->nb[i] != normalized->nb[i] ||
            full->ne[i] != normalized->ne[i] || rotated->ne[i] != normalized->ne[i]) {
            return false;
        }
    }
    if (tail) {
        if (!view_of(tail, normalized, 2 * half * sizeof(float)) ||
            !half_shape(tail, normalized, 128 - 2 * half)) {
            return false;
        }
        for (int i = 1; i < 4; ++i) {
            if (tail->nb[i] != normalized->nb[i]) {
                return false;
            }
        }
    }
    if (!half_shape(first, normalized, half) || !half_shape(second, normalized, half)) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        if (!half_shape(products[i], normalized, half) ||
            !coefficient(match, products[i]->src[1], i, half, norm->ne[2]) || !match.keep(products[i])) {
            return false;
        }
    }
    const ggml_tensor * nodes[] = {norm, normalized, a, b, first, second, rotated, full, permute, output};
    for (const auto * tensor : nodes) {
        if (!match.keep(tensor)) {
            return false;
        }
    }
    if (tail && !match.keep(tail)) {
        return false;
    }
    match.input = norm->src[0];
    match.weight = weight;
    match.output = output;
    match.rotation = 2 * half;
    return true;
}

static inline bool match_norm_rope(const ggml_cgraph * graph, int first, norm_rope_match & result) {
    if (first < 0 || first >= graph->n_nodes) {
        return false;
    }
    const auto * norm = graph->nodes[first];
    const auto * input = norm->src[0];
    if (norm->op != GGML_OP_RMS_NORM || !input || norm->view_src ||
        !contiguous_f32(norm) || input->type != GGML_TYPE_F32 || input->nb[0] != sizeof(float) ||
        input->ne[0] != 128 || !ggml_are_same_shape(input, norm)) {
        return false;
    }
    int64_t rows = 1;
    for (int i = 1; i < 4; ++i) {
        if (input->ne[i] <= 0 || input->ne[i] > std::numeric_limits<int>::max() / rows) {
            return false;
        }
        rows *= input->ne[i];
    }
    const float eps = ggml_get_op_params_f32(norm, 0);
    if (!std::isfinite(eps) || eps < 0) {
        return false;
    }
    for (int last = first + 2; last < std::min(graph->n_nodes, first + 32); ++last) {
        auto * output = graph->nodes[last];
        if (output->op != GGML_OP_CONT || output->ne[0] != 128) {
            continue;
        }
        norm_rope_match candidate;
        if (!match_output(norm, output, candidate)) {
            continue;
        }
        std::array<int, 32> indices{};
        std::array<ggml_op, 32> operations{};
        int count = 0;
        bool valid = true;
        for (int index = first; index <= last; ++index) {
            const auto * node = graph->nodes[index];
            if (node != output && (node->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                valid = false;
                break;
            }
            const bool kept = std::find(candidate.kept.begin(), candidate.kept.begin() + candidate.n_kept, node) != candidate.kept.begin() + candidate.n_kept;
            const bool pe_view = std::find(candidate.pe_metadata.begin(), candidate.pe_metadata.begin() + candidate.n_pe_metadata, node) != candidate.pe_metadata.begin() + candidate.n_pe_metadata;
            if (kept) {
                indices[count] = index;
                operations[count++] = node->op;
            } else if (!pe_view || ggml_node_get_use_count(graph, index) != 1) {
                valid = false;
                break;
            }
        }
        if (!valid || count != candidate.n_kept ||
            !ggml_can_fuse_subgraph_ext(graph, indices.data(), count, operations.data(), &last, 1)) {
            continue;
        }
        candidate.last = last;
        result = candidate;
        return true;
    }
    return false;
}

} // namespace ggml_cuda_h3

#endif
