#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

static inline bool xdna_glue_supports_op(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return false;
        default:
            break;
    }
    if (op->op == GGML_OP_SET_ROWS) {
        if (!op->src[0] || op->src[0]->type != GGML_TYPE_F32 ||
            (op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_F32)) {
            return false;
        }
    } else if (op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) {
        return false;
    }
    ggml_backend_dev_t cpu = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
    return ggml_backend_dev_supports_op(cpu, op);
}
