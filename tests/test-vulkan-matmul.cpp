#include "ggml-vulkan/ggml-vulkan-matmul.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

static void test_scalar_f32_configs(uint32_t subgroup_size) {
    const auto configs = ggml_vk_scalar_f32_matmul_configs(subgroup_size);
    constexpr size_t expected_shmem[] = {34816, 17408, 8704};

    for (size_t i = 0; i < configs.size(); ++i) {
        const auto &config = configs[i];
        const auto &spec = config.specialization_constants;

        assert(spec.size() == 11);
        assert(config.work_group_denominators[0] == spec[1]);
        assert(config.work_group_denominators[1] == spec[2]);
        assert(config.work_group_denominators[2] == 1);
        assert(spec[4] % spec[6] == 0);
        assert(spec[4] / spec[6] >= spec[7]);

        const uint64_t wniter_num = (uint64_t)spec[4] * spec[5];
        const uint64_t wniter_den =
                (uint64_t)spec[10] * spec[7] * spec[8] * spec[6];
        assert(wniter_den != 0);
        assert(wniter_num >= wniter_den);
        assert(wniter_num % wniter_den == 0);
        assert(ggml_vk_scalar_f32_matmul_shmem_size(config) ==
               expected_shmem[i]);
        assert(ggml_vk_scalar_f32_matmul_shmem_supported(config,
                                                         expected_shmem[i]));
        assert(!ggml_vk_scalar_f32_matmul_shmem_supported(
                config, expected_shmem[i] - 1));
    }
}

static void test_f32_pipeline_selection() {
    using kind = ggml_vk_f32_matmul_pipeline_kind;

    assert(ggml_vk_select_f32_matmul_pipeline(true, false, false) ==
           kind::cooperative_matrix);
    assert(ggml_vk_select_f32_matmul_pipeline(true, false, true) ==
           kind::cooperative_matrix);
    assert(ggml_vk_select_f32_matmul_pipeline(true, true, false) ==
           kind::scalar);
    assert(ggml_vk_select_f32_matmul_pipeline(false, false, false) ==
           kind::scalar);
    assert(ggml_vk_select_f32_matmul_pipeline(false, true, false) ==
           kind::scalar);
    assert(ggml_vk_select_f32_matmul_pipeline(true, true, true) == kind::none);
    assert(ggml_vk_select_f32_matmul_pipeline(false, false, true) ==
           kind::none);
}

int main() {
    test_scalar_f32_configs(8);
    test_scalar_f32_configs(16);
    test_scalar_f32_configs(32);
    test_scalar_f32_configs(64);
    test_f32_pipeline_selection();
    return 0;
}
