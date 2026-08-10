#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

enum class ggml_vk_f32_matmul_pipeline_kind {
    none,
    scalar,
    cooperative_matrix,
};

enum class ggml_vk_matmul_pipeline_tier {
    none,
    small,
    medium,
    large,
};

static inline ggml_vk_matmul_pipeline_tier
ggml_vk_select_matmul_pipeline_tier(bool large_available,
                                    bool medium_available,
                                    bool small_available,
                                    uint32_t m,
                                    uint32_t n) {
    if ((small_available && (m <= 32 || n <= 32)) ||
        (!medium_available && !large_available)) {
        return small_available
            ? ggml_vk_matmul_pipeline_tier::small
            : ggml_vk_matmul_pipeline_tier::none;
    }
    if ((medium_available && (m <= 64 || n <= 64)) || !large_available) {
        return medium_available
            ? ggml_vk_matmul_pipeline_tier::medium
            : ggml_vk_matmul_pipeline_tier::none;
    }
    return large_available
        ? ggml_vk_matmul_pipeline_tier::large
        : ggml_vk_matmul_pipeline_tier::none;
}

static inline ggml_vk_f32_matmul_pipeline_kind
ggml_vk_select_f32_matmul_pipeline(bool cooperative_matrix_supported,
                                   bool cooperative_matrix_pipeline_empty,
                                   bool scalar_pipeline_empty) {
    if (cooperative_matrix_supported && !cooperative_matrix_pipeline_empty) {
        return ggml_vk_f32_matmul_pipeline_kind::cooperative_matrix;
    }
    if (!scalar_pipeline_empty) {
        return ggml_vk_f32_matmul_pipeline_kind::scalar;
    }
    return ggml_vk_f32_matmul_pipeline_kind::none;
}

struct ggml_vk_scalar_f32_matmul_config {
    std::vector<uint32_t> specialization_constants;
    std::array<uint32_t, 3> work_group_denominators;
    uint32_t alignment;
};

static inline std::array<ggml_vk_scalar_f32_matmul_config, 3>
ggml_vk_scalar_f32_matmul_configs(uint32_t subgroup_size) {
    const uint32_t subgroup_size_8 = std::max(subgroup_size, 8u);
    const uint32_t subgroup_size_32 = std::max(subgroup_size, 32u);
    const uint32_t small_warp_m = std::clamp(subgroup_size, 8u, 32u);

    return {{
            {{128, 128, 128, 16, subgroup_size_8 * 2, 64, 2, 4, 4, 1,
              subgroup_size_8},
             {128, 128, 1},
             128},
            {{128, 64, 64, 16, subgroup_size_8, 32, 2, 4, 2, 1,
              subgroup_size_8},
             {64, 64, 1},
             64},
            {{subgroup_size_32, 32, 32, 16, small_warp_m, 32, 2, 2, 2, 1,
              subgroup_size_8},
             {32, 32, 1},
             32},
    }};
}

static inline size_t ggml_vk_scalar_f32_matmul_shmem_size(
        const ggml_vk_scalar_f32_matmul_config &config) {
    // mul_mm.comp fixes BK at 32 for F32 and the scalar shader uses
    // SHMEM_STRIDE = BK / 2 + 1 with vec2 (two floats) as its element type.
    constexpr size_t shmem_stride = 32 / 2 + 1;
    constexpr size_t element_size = 2 * sizeof(float);
    return (config.specialization_constants[1] +
            config.specialization_constants[2]) *
           shmem_stride * element_size;
}

static inline bool ggml_vk_scalar_f32_matmul_shmem_supported(
        const ggml_vk_scalar_f32_matmul_config &config, size_t max_shmem_size) {
    return ggml_vk_scalar_f32_matmul_shmem_size(config) <= max_shmem_size;
}
