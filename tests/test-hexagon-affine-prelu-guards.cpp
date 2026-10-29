#include "affine-prelu-ops.h"

#include <array>
#include <cstdio>
#include <initializer_list>
#include <limits>

#define CHECK(value) do { if (!(value)) { \
    std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #value); return 1; } } while (0)

static htp_affine_f32_tensor tensor(uint64_t f, uint64_t t, uint64_t c, uint64_t b) {
    htp_affine_f32_tensor result = {};
    const uint64_t shape[] = {f, t, c, b};
    uint64_t stride = sizeof(float);
    for (int i = 0; i < 4; ++i) {
        result.ne[i] = shape[i];
        result.nb[i] = stride;
        stride *= shape[i];
    }
    result.size = stride;
    return result;
}

int main() {
    htp_affine_f32_layout layout;
    for (uint32_t f : {1u, 2u, 31u, 32u, 33u, 63u, 64u, 65u, 127u, 128u, 129u}) {
        for (uint32_t b : {1u, 3u, 7u}) {
            htp_affine_f32_tensor tensors[] = {
                tensor(f, 63, 24, b), tensor(f, 24, 1, 1), tensor(f, 24, 1, 1),
                tensor(24, 1, 1, 1), tensor(f, 63, 24, b)};
            CHECK(htp_affine_f32_validate(tensors, 6, 6 * HTP_AFFINE_F32_WORKER_BYTES, &layout));
            CHECK(layout.f == f && layout.t == 63 && layout.c == 24 && layout.b == b);
            CHECK(layout.tiles == (f + 31) / 32 && layout.jobs == layout.tiles * 24 * b);
            CHECK(!htp_affine_f32_validate(tensors, 6, 6 * HTP_AFFINE_F32_WORKER_BYTES - 1, &layout));
        }
    }
    std::array<htp_affine_f32_tensor, 5> tensors = {
        tensor(33, 9, 3, 2), tensor(33, 3, 1, 1), tensor(33, 3, 1, 1),
        tensor(3, 1, 1, 1), tensor(33, 9, 3, 2)};
    for (size_t i = 0; i < tensors.size(); ++i) tensors[i].address = 0x100004 + i * 0x100000;
    const auto valid = [&] { return htp_affine_f32_validate(tensors.data(), 6, 3072, &layout); };
    CHECK(valid()); // Non-vector-aligned views are supported without overreads.
    CHECK(!htp_affine_f32_validate(nullptr, 6, 3072, &layout));
    CHECK(!htp_affine_f32_validate(tensors.data(), 6, 3072, nullptr));
    for (uint32_t threads : {0u, 11u, UINT32_MAX}) {
        CHECK(!htp_affine_f32_validate(tensors.data(), threads, 1 << 20, &layout));
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto saved = tensors[i];
        tensors[i].size--;
        CHECK(!valid()); tensors[i] = saved;
        tensors[i].type = 1;
        CHECK(!valid()); tensors[i] = saved;
        tensors[i].address++;
        CHECK(!valid()); tensors[i] = saved;
        tensors[i].address = UINTPTR_MAX - 3;
        CHECK(!valid()); tensors[i] = saved;
        for (int d = 0; d < 4; ++d) {
            tensors[i].nb[d] += 4;
            CHECK(!valid()); tensors[i] = saved;
            tensors[i].ne[d] = 0;
            CHECK(!valid()); tensors[i] = saved;
            tensors[i].ne[d] = UINT64_MAX;
            CHECK(!valid()); tensors[i] = saved;
        }
    }
    // Canonical, well-sized tensors with wrong parameter/output shapes.
    for (size_t i = 1; i < tensors.size(); ++i) {
        const auto saved = tensors[i];
        for (int d = 0; d < 4; ++d) {
            auto ne = saved.ne;
            std::array<uint64_t, 4> shape = {ne[0], ne[1], ne[2], ne[3]};
            shape[d]++;
            tensors[i] = tensor(shape[0], shape[1], shape[2], shape[3]);
            CHECK(!valid());
            tensors[i] = saved;
        }
    }
    for (size_t i = 0; i < 4; ++i) {
        const uint64_t saved = tensors[4].address;
        tensors[4].address = tensors[i].address;
        CHECK(!valid());
        tensors[4].address += 4;
        // A one-float slope span is covered by the exact alias check above.
        if (tensors[i].size > 4) CHECK(!valid());
        tensors[4].address = saved;
    }
    uint64_t bytes;
    auto huge = tensor(1, 1, 1, 1);
    huge.ne[0] = UINT32_MAX;
    huge.size = UINT64_MAX;
    CHECK(!htp_affine_f32_tensor_bytes(&huge, &bytes));
    CHECK(!htp_affine_f32_tensor_bytes(nullptr, &bytes));
    CHECK(!htp_affine_f32_tensor_bytes(&huge, nullptr));
    auto smallest = tensor(1, 1, 1, 1);
    std::array<htp_affine_f32_tensor, 5> one = {smallest, smallest, smallest, smallest, smallest};
    CHECK(htp_affine_f32_validate(one.data(), 10, 512, &layout) && layout.threads == 1 && layout.jobs == 1);
    CHECK(!htp_affine_f32_validate(one.data(), 10, 511, &layout));
    std::puts("affine/PReLU F32 layout, span, shape, alias, VTCM and overflow guards: PASS");
    return 0;
}
