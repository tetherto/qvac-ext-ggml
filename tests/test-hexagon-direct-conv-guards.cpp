#include "conv-2d-ops.h"

#include <array>
#include <cstdio>
#include <initializer_list>
#include <limits>

#define CHECK(value)                                                                                                   \
    do {                                                                                                               \
        if (!(value)) {                                                                                                \
            std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #value);                                              \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

static htp_conv_f32_tensor tensor(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    htp_conv_f32_tensor t = {};
    const uint64_t ne[4] = {a, b, c, d};
    uint64_t stride = sizeof(float);
    for (int i = 0; i < 4; ++i) {
        t.ne[i] = ne[i];
        t.nb[i] = stride;
        stride *= ne[i];
    }
    t.size = stride;
    return t;
}

int main() {
    htp_conv_f32_layout layout;
    CHECK(!htp_conv_f32_span_fits(UINT32_MAX - 3, 16, UINT32_MAX));
    CHECK(htp_conv_f32_span_fits(UINT32_MAX - 3, 16, UINT64_MAX));
    CHECK(!htp_conv_f32_span_fits(UINT64_MAX - 3, 16, UINT64_MAX));
    for (int64_t position : {INT64_MIN, INT64_MAX}) {
        uint32_t source, target, count;
        htp_conv_f32_row_window(65, position, 67, &source, &target, &count);
        CHECK(source == 0 && target == 0 && count == 0);
    }
    for (uint32_t w : {1u, 2u, 31u, 32u, 33u, 63u, 64u, 65u, 127u, 128u, 129u}) {
        for (uint32_t kernel : {1u, 3u})
            for (int32_t sx : {1, 2})
                for (int32_t sy : {1, 2})
                    for (int32_t dx : {1, 2})
                        for (int32_t dy : {1, 2}) {
                            int32_t params[6] = {sx, sy, 4, 3, dx, dy};
                            const uint32_t ow = (w + 8 - dx * (kernel - 1) - 1) / sx + 1;
                            const uint32_t oh = (7 + 6 - dy * (kernel - 1) - 1) / sy + 1;
                            auto k = tensor(kernel, kernel, 3, 5), x = tensor(w, 7, 3, 2), y = tensor(ow, oh, 5, 2);
                            CHECK(htp_conv_f32_validate(&k, &x, &y, params, 6, 8 * 1024 * 1024, &layout));
                            CHECK(layout.w == w && layout.ow == ow && layout.oh == oh);
                            CHECK(layout.weights_bytes >= k.size && layout.weights_bytes % 128 == 0);
                            CHECK(layout.worker_bytes == 512 && layout.total_bytes % 128 == 0);
                            CHECK(!htp_conv_f32_validate(&k, &x, &y, params, 6, layout.total_bytes - 1, &layout));
                            for (uint32_t ox = 0; ox < ow; ox += 32) {
                                const uint32_t valid = ow - ox < 32 ? ow - ox : 32;
                                const uint32_t span = (valid - 1) * sx + (kernel - 1) * dx + 1;
                                CHECK(span <= 67);
                                const int64_t ix = (int64_t)ox * sx - params[2];
                                uint32_t source, target, count;
                                htp_conv_f32_row_window(w, ix, span, &source, &target, &count);
                                CHECK(count == 0 || (source + count <= w && target + count <= span));
                                for (uint32_t lane = 0; lane < span; ++lane) {
                                    const int64_t position = ix + lane;
                                    const bool expected = position >= 0 && position < w;
                                    const bool copied = lane >= target && lane < target + count;
                                    CHECK(expected == copied);
                                    CHECK(!copied || source + lane - target == position);
                                }
                            }
                        }
    }
    int32_t p[6] = {1, 1, 0, 0, 1, 1};
    auto k = tensor(1, 1, 6, 5), x = tensor(65, 63, 6, 7), y = tensor(65, 63, 5, 7);
    const auto valid = [&]() { return htp_conv_f32_validate(&k, &x, &y, p, 6, 8 * 1024 * 1024, &layout); };
    CHECK(valid() && layout.flat_pointwise);
    for (int i = 0; i < 6; ++i) {
        const int32_t saved = p[i];
        p[i] = -1;
        CHECK(!valid());
        p[i] = saved;
    }
    for (int i : {0, 1, 4, 5}) {
        p[i] = 3;
        CHECK(!valid());
        p[i] = 1;
    }
    const auto original_x = x, original_y = y, original_k = k;
    x.nb[1] += 4;
    CHECK(!valid());
    x = original_x;
    k.nb[3] += 4;
    CHECK(!valid());
    k = original_k;
    y.nb[1] += 4;
    CHECK(!valid());
    y = original_y;
    x.type = 1;
    CHECK(!valid());
    x = original_x;
    k.type = 1;
    CHECK(!valid());
    k = original_k;
    y.type = 1;
    CHECK(!valid());
    y = original_y;
    --x.size;
    CHECK(!valid());
    x = original_x;
    --k.size;
    CHECK(!valid());
    k = original_k;
    --y.size;
    CHECK(!valid());
    y = original_y;
    x.address = 2;
    CHECK(!valid());
    x = original_x;
    x.address = UINT64_MAX - 3;
    CHECK(!valid());
    x = original_x;
    x.ne[3] = UINT64_MAX;
    CHECK(!valid());
    x = original_x;
    x.ne[0] = UINT32_MAX;
    CHECK(!valid());
    x = original_x;
    x.ne[0] = 0;
    CHECK(!valid());
    x = original_x;
    y.ne[0] += 1;
    CHECK(!valid());
    y = original_y;
    y.ne[3] += 1;
    CHECK(!valid());
    y = original_y;
    k.ne[2] += 1;
    CHECK(!valid());
    k = original_k;
    k = tensor(2, 2, 6, 5);
    CHECK(!valid());
    k = original_k;
    k = tensor(1, 3, 6, 5);
    CHECK(!valid());
    k = original_k;
    x.address = 0x10000;
    y.address = x.address + 4;
    CHECK(!valid());
    y.address = x.address + x.size;
    CHECK(valid());
    x.address = 0;
    k.address = 0x1000;
    y.address = k.address;
    CHECK(!valid());
    k.address = 0;
    y.address = 0;
    CHECK(!htp_conv_f32_validate(&k, &x, &y, p, 0, 8 * 1024 * 1024, &layout));
    CHECK(!htp_conv_f32_validate(&k, &x, &y, p, 11, 8 * 1024 * 1024, &layout));
    k = tensor(3, 3, 1, 1);
    x = tensor(1, 1, 1, 1);
    y = tensor(1, 1, 1, 1);
    CHECK(!valid()); // Empty effective extent before padding.
    k = tensor(3, 3, 512, 512);
    x = tensor(3, 3, 512, 1);
    y = tensor(1, 1, 512, 1);
    CHECK(!valid()); // Valid tensor spans but padded weights exceed VTCM.

    htp_conv_f32_pointwise_layout staged;
    CHECK(!htp_conv_f32_plan_pointwise(nullptr, UINT64_MAX, &staged));
    CHECK(!htp_conv_f32_plan_pointwise(&layout, UINT64_MAX, nullptr));
    for (uint32_t plane : {1u, 2u, 31u, 32u, 33u, 63u, 64u, 65u, 127u, 128u, 129u, 2079u, 4095u}) {
        for (uint32_t ci : {1u, 3u, 6u, 16u}) for (uint32_t co : {1u, 3u, 4u, 5u, 6u, 12u, 16u})
            for (uint32_t batches : {1u, 2u, 7u}) {
                auto kernel = tensor(1, 1, ci, co), input = tensor(plane, 1, ci, batches);
                auto output = tensor(plane, 1, co, batches);
                CHECK(htp_conv_f32_validate(&kernel, &input, &output, p, 6, 8 * 1024 * 1024, &layout));
                CHECK(htp_conv_f32_plan_pointwise(&layout, 8 * 1024 * 1024, &staged));
                const uint32_t tiles = (plane + 31) / 32;
                CHECK(staged.jobs == tiles * batches);
                CHECK(staged.threads == (staged.jobs < layout.threads ? staged.jobs : layout.threads));
                CHECK(staged.input_bytes == ci * 128 && staged.worker_bytes == ci * 128 + 512);
                CHECK(staged.total_bytes == layout.weights_bytes + staged.threads * staged.worker_bytes);
                CHECK(staged.input_bytes % 128 == 0 && staged.worker_bytes % 128 == 0);
                CHECK(htp_conv_f32_plan_pointwise(&layout, staged.total_bytes, &staged));
                CHECK(!htp_conv_f32_plan_pointwise(&layout, staged.total_bytes - 1, &staged));
                for (uint32_t n = 0; n < batches; ++n) for (uint32_t tile : {0u, tiles / 2, tiles - 1}) {
                    const uint32_t start = tile * 32, lanes = plane - start < 32 ? plane - start : 32;
                    CHECK(lanes > 0 && lanes <= 32 && start + lanes <= plane);
                    for (uint32_t c = 0; c < ci; ++c) {
                        const uint64_t first = ((uint64_t)n * ci + c) * plane + start;
                        CHECK((first + lanes) * sizeof(float) <= input.size);
                        CHECK((uint64_t)c * 128 + lanes * sizeof(float) <= staged.input_bytes);
                    }
                    for (uint32_t oc = 0; oc < co; oc += 4) {
                        const uint32_t channels = co - oc < 4 ? co - oc : 4;
                        for (uint32_t c = 0; c < channels; ++c) {
                            const uint64_t first = ((uint64_t)n * co + oc + c) * plane + start;
                            CHECK((first + lanes) * sizeof(float) <= output.size);
                            CHECK(c * 128 + lanes * sizeof(float) <= 512);
                        }
                    }
                }
            }
    }
    // A smaller VTCM budget must keep the original accepted raw-DDR route.
    k = tensor(1, 1, 6, 5); x = tensor(65, 63, 6, 7); y = tensor(65, 63, 5, 7);
    CHECK(valid());
    const uint64_t original_budget = layout.total_bytes;
    CHECK(!htp_conv_f32_plan_pointwise(&layout, original_budget, &staged));
    CHECK(htp_conv_f32_validate(&k, &x, &y, p, 6, original_budget, &layout));
    const auto pointwise = layout;
    for (uint32_t field : {0u, 1u, 2u, 3u}) {
        auto malformed = pointwise;
        if (field == 0) malformed.flat_pointwise = false;
        if (field == 1) malformed.ci = UINT32_MAX;
        if (field == 2) malformed.tiles = malformed.n = UINT32_MAX;
        if (field == 3) malformed.weights_bytes = UINT32_MAX;
        CHECK(!htp_conv_f32_plan_pointwise(&malformed, UINT64_MAX, &staged));
    }
    // Large row strides stay exact: dma_queue_copy_rows handles descriptor
    // field limits with bounded 1-D rows instead of truncating a 2-D stride.
    k = tensor(1, 1, 1, 1); x = tensor(0x400001, 1, 1, 1); y = tensor(0x400001, 1, 1, 1);
    CHECK(valid() && (uint64_t)layout.plane * sizeof(float) > 0xffffffu);
    CHECK(htp_conv_f32_plan_pointwise(&layout, 8 * 1024 * 1024, &staged));
    // OC8 is optional: preserve the published pointwise worker count and the
    // four-channel weight padding, including OC9/10/15/17 remainders.
    htp_conv_f32_pointwise_layout eight;
    for (uint32_t channels : {1u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 12u, 15u, 16u, 17u}) {
        k = tensor(1, 1, 6, channels); x = tensor(65, 63, 6, 7); y = tensor(65, 63, channels, 7);
        CHECK(valid());
        CHECK(htp_conv_f32_plan_pointwise(&layout, 8 * 1024 * 1024, &staged));
        const bool selected = htp_conv_f32_plan_pointwise_eight(&layout, 8 * 1024 * 1024, &eight);
        CHECK(selected == (channels >= 8));
        if (!selected) continue;
        CHECK(eight.jobs == staged.jobs && eight.threads == staged.threads && eight.input_bytes == staged.input_bytes);
        CHECK(eight.worker_bytes == staged.worker_bytes + 512);
        CHECK(eight.total_bytes == layout.weights_bytes + eight.threads * eight.worker_bytes);
        CHECK(!htp_conv_f32_plan_pointwise_eight(&layout, eight.total_bytes - 1, &eight));
        CHECK(htp_conv_f32_plan_pointwise(&layout, staged.total_bytes, &staged));
        CHECK(!htp_conv_f32_plan_pointwise_eight(&layout, staged.total_bytes, &eight));
        for (uint32_t oc = 0; oc < channels;) {
            const uint32_t remaining = channels - oc, block = remaining >= 8 ? 8 : 4;
            const uint32_t actual = remaining < block ? remaining : block;
            const uint32_t computed = actual == 1 ? 1 : block;
            CHECK(oc + computed <= (channels + 3) / 4 * 4);
            CHECK(computed != 8 || oc + 8 <= channels);
            CHECK((uint64_t)(oc + computed) * layout.k_elements * sizeof(float) <= layout.weights_bytes);
            for (uint32_t c = 0; c < actual; ++c) {
                CHECK(c * 128 + 128 <= 8 * 128);
                const uint64_t last = ((uint64_t)6 * channels + oc + c + 1) * layout.out_plane;
                CHECK(last * sizeof(float) <= y.size);
            }
            oc += actual;
        }
    }

    // Prove the complete source-layout mapping, including absent padded rows,
    // dilation, vector tails, batch/channel separation and OC8/4/1 stores.
    htp_conv_f32_spatial_layout spatial;
    CHECK(!htp_conv_f32_plan_spatial(nullptr, UINT64_MAX, &spatial));
    CHECK(!htp_conv_f32_plan_spatial(&layout, UINT64_MAX, nullptr));
    for (uint32_t width : {1u, 2u, 31u, 32u, 33u, 63u, 64u, 65u, 127u, 128u, 129u}) {
        for (int sx : {1, 2}) for (int sy : {1, 2}) for (int dx : {1, 2}) for (int dy : {1, 2}) {
            int32_t params[6] = {sx, sy, dx, 2, dx, dy};
            const uint32_t ow = (width + 2 * dx - 2 * dx - 1) / sx + 1;
            const uint32_t oh = (7 + 4 - 2 * dy - 1) / sy + 1;
            auto kernel = tensor(3, 3, 3, 17), input = tensor(width, 7, 3, 2), output = tensor(ow, oh, 17, 2);
            CHECK(htp_conv_f32_validate(&kernel, &input, &output, params, 6, 8 * 1024 * 1024, &layout));
            const bool selected = htp_conv_f32_plan_spatial(&layout, 8 * 1024 * 1024, &spatial);
            CHECK(selected == ((uint64_t)layout.tiles * oh * 2 >= layout.threads));
            if (!selected) continue;
            CHECK(spatial.threads == layout.threads && spatial.jobs == layout.tiles * oh * 2);
            CHECK(spatial.row_floats == (sx == 1 ? 64u : 96u));
            CHECK(spatial.input_bytes == 3 * 3 * spatial.row_floats * sizeof(float));
            CHECK(spatial.worker_bytes == spatial.input_bytes + 128);
            CHECK(spatial.total_bytes == layout.weights_bytes + spatial.threads * spatial.worker_bytes);
            CHECK(spatial.input_bytes % 128 == 0 && spatial.worker_bytes % 128 == 0);
            CHECK(!htp_conv_f32_plan_spatial(&layout, spatial.total_bytes - 1, &spatial));
            CHECK(htp_conv_f32_plan_spatial(&layout, 8 * 1024 * 1024, &spatial));
            const uint64_t old_budget = layout.total_bytes;
            CHECK(!htp_conv_f32_plan_spatial(&layout, old_budget, &spatial));
            htp_conv_f32_layout accepted;
            CHECK(htp_conv_f32_validate(&kernel, &input, &output, params, 6, old_budget, &accepted));
            for (uint32_t job : {0u, spatial.jobs / 2, spatial.jobs - 1}) {
                const uint32_t start = job % layout.tiles * 32, oy = job / layout.tiles % oh;
                const uint32_t n = job / layout.tiles / oh;
                const uint32_t lanes = ow - start < 32 ? ow - start : 32;
                const uint32_t span = (lanes - 1) * sx + 2 * dx + 1;
                const int64_t ix = (int64_t)start * sx - dx;
                for (uint32_t ci = 0; ci < 3; ++ci) for (uint32_t ky = 0; ky < 3; ++ky) {
                    const int64_t iy = (int64_t)oy * sy + ky * dy - 2;
                    std::array<float, 98> storage;
                    storage.fill(-777.0f);
                    float * stage = storage.data() + 1;
                    for (uint32_t i = 0; i < spatial.row_floats; ++i) stage[i] = 0.0f;
                    uint32_t first = 0, target = 0, count = 0;
                    if (iy >= 0 && iy < 7) htp_conv_f32_row_window(width, ix, span, &first, &target, &count);
                    CHECK(first + count <= width && target + count <= spatial.row_floats);
                    for (uint32_t i = 0; i < count; ++i)
                        stage[target + i] = float(1 + n * 100000 + ci * 10000 + iy * width + first + i);
                    CHECK(storage.front() == -777.0f && stage[spatial.row_floats] == -777.0f);
                    for (uint32_t kx = 0; kx < 3; ++kx) {
                        CHECK(kx * dx + 31 * sx < spatial.row_floats);
                        for (uint32_t lane = 0; lane < lanes; ++lane) {
                            const int64_t source_x = ix + lane * sx + kx * dx;
                            const float expected = iy >= 0 && iy < 7 && source_x >= 0 && source_x < width ?
                                float(1 + n * 100000 + ci * 10000 + iy * width + source_x) : 0.0f;
                            CHECK(stage[lane * sx + kx * dx] == expected);
                        }
                    }
                }
                for (uint32_t oc = 0; oc < 17;) {
                    const uint32_t remaining = 17 - oc, block = remaining >= 8 ? 8 : 4;
                    const uint32_t actual = remaining < block ? remaining : block;
                    for (uint32_t c = 0; c < actual; ++c) {
                        const uint64_t first = ((uint64_t)n * 17 + oc + c) * ow * oh + oy * ow + start;
                        CHECK(first + lanes <= ((uint64_t)n * 17 + oc + c) * ow * oh + (oy + 1) * ow);
                        CHECK((first + lanes) * sizeof(float) <= output.size);
                    }
                    oc += actual;
                }
            }
        }
    }
    // Fewer spatial jobs cannot steal OC parallelism from the published path.
    p[0] = p[1] = p[4] = p[5] = 1; p[2] = p[3] = 1;
    k = tensor(3, 3, 3, 16); x = tensor(1, 1, 3, 1); y = tensor(1, 1, 16, 1);
    CHECK(valid() && layout.threads == 4);
    CHECK(!htp_conv_f32_plan_spatial(&layout, UINT64_MAX, &spatial));
    p[2] = p[3] = 1;
    k = tensor(3, 3, 3, 17); x = tensor(65, 7, 3, 2); y = tensor(65, 7, 17, 2);
    CHECK(valid() && htp_conv_f32_plan_spatial(&layout, 8 * 1024 * 1024, &spatial));
    for (uint32_t field = 0; field < 7; ++field) {
        auto malformed = layout;
        if (field == 0) malformed.ci = UINT32_MAX;
        if (field == 1) malformed.threads = 0;
        if (field == 2) malformed.threads = 11;
        if (field == 3) malformed.weights_bytes = UINT32_MAX;
        if (field == 4) malformed.sx = 3;
        if (field == 5) malformed.tiles = UINT32_MAX;
        if (field == 6) malformed.n = UINT32_MAX;
        CHECK(!htp_conv_f32_plan_spatial(&malformed, UINT64_MAX, &spatial));
    }
    std::puts("Direct F32 convolution shape/span/layout/padding/VTCM guards passed");
    std::puts("Direct F32 pointwise DMA jobs, VTCM fallback and bounded channel/tail rows passed");
    std::puts("Direct F32 OC8 and shared spatial-window bounds, worker preservation and fallbacks passed");
    return 0;
}
