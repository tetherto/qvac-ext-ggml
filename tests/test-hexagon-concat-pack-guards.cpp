#include "concat-pack-guards.h"

#include <algorithm>
#include <cstdio>
#include <initializer_list>
#include <vector>

#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #value); return 1; } } while (0)

static htp_concat_pack_tensor tensor(uint64_t width, uint64_t rows, uint64_t address) {
    htp_concat_pack_tensor t = {};
    t.ne[0] = width; t.ne[1] = rows; t.ne[2] = t.ne[3] = 1;
    t.address = address;
    uint64_t stride = sizeof(float);
    for (int i = 0; i < 4; ++i) { t.nb[i] = stride; stride *= t.ne[i]; }
    t.size = stride;
    return t;
}

int main() {
    htp_concat_pack_layout l;
    for (uint32_t width : {4u, 8u, 16u}) {
        for (uint32_t rows : {1u, 2u, 5u, 7u, 8u, 9u, 31u, 32u, 33u, 63u, 65u, 129u, 14553u}) {
            for (uint32_t threads : {1u, 6u, 10u}) {
                auto a = tensor(width, rows, 0x100004), b = tensor(width, rows, 0x100000c),
                     d = tensor(width * 2, rows, 0x3000004);
                CHECK(htp_concat_pack_plan(&a, &b, &d, 0, threads, &l));
                CHECK(l.rows == rows && l.row_elements == width && l.rows_per_vector * width == 32);
                std::vector<uint32_t> writes(rows, 0), out(size_t(rows) * width * 2 + 2, 0xdeadbeef);
                uint32_t * destination = out.data() + 1;
                for (uint32_t worker = 0; worker < l.threads; ++worker) {
                    uint32_t begin, end; htp_concat_pack_rows(&l, worker, &begin, &end);
                    CHECK(begin <= end && end <= rows);
                    // The vector body consumes exactly32 F32 values per input
                    // and writes64 output words. The remaining rows are exact.
                    uint32_t row = begin;
                    while (end - row >= l.rows_per_vector) {
                        CHECK(uint64_t(row + l.rows_per_vector) * width * 4 <= a.size);
                        CHECK(uint64_t(row + l.rows_per_vector) * width * 8 <= d.size);
                        for (uint32_t n = 0; n < l.rows_per_vector; ++n) {
                            for (uint32_t lane = 0; lane < width; ++lane) {
                                destination[size_t(row + n) * width * 2 + lane] = 1000 + (row + n) * width + lane;
                                destination[size_t(row + n) * width * 2 + width + lane] = 2000000 + (row + n) * width + lane;
                            }
                            ++writes[row + n];
                        }
                        row += l.rows_per_vector;
                    }
                    for (; row < end; ++row) {
                        for (uint32_t lane = 0; lane < width; ++lane) {
                            destination[size_t(row) * width * 2 + lane] = 1000 + row * width + lane;
                            destination[size_t(row) * width * 2 + width + lane] = 2000000 + row * width + lane;
                        }
                        ++writes[row];
                    }
                }
                CHECK(out.front() == 0xdeadbeef && out.back() == 0xdeadbeef);
                for (uint32_t row = 0; row < rows; ++row) {
                    CHECK(writes[row] == 1);
                    for (uint32_t lane = 0; lane < width; ++lane) {
                        CHECK(destination[size_t(row) * width * 2 + lane] == 1000 + row * width + lane);
                        CHECK(destination[size_t(row) * width * 2 + width + lane] == 2000000 + row * width + lane);
                    }
                }
            }
        }
    }
    auto a = tensor(4, 33, 0x1004), b = tensor(4, 33, 0x10004), d = tensor(8, 33, 0x20004);
    const auto old_a = a, old_b = b, old_d = d;
    const auto valid = [&] { return htp_concat_pack_plan(&a, &b, &d, 0, 6, &l); };
    CHECK(valid());
    for (int i = 0; i < 4; ++i) { a.nb[i] += 4; CHECK(!valid()); a = old_a; }
    for (int i = 0; i < 4; ++i) { d.nb[i] += 4; CHECK(!valid()); d = old_d; }
    --a.size; CHECK(!valid()); a = old_a;
    --b.size; CHECK(!valid()); b = old_b;
    --d.size; CHECK(!valid()); d = old_d;
    a.type = 1; CHECK(!valid()); a = old_a;
    b.ne[0] = 3; CHECK(!valid()); b = old_b;
    b.ne[2] = 2; CHECK(!valid()); b = old_b;
    d.address = a.address; CHECK(!valid()); d = old_d;
    d.address = b.address; CHECK(!valid()); d = old_d;
    a.address = UINTPTR_MAX - 3; CHECK(!valid()); a = old_a;
    ++a.address; CHECK(!valid()); a = old_a;
    a.address = 0; CHECK(!valid()); a = old_a;
    CHECK(!htp_concat_pack_plan(&a, &b, &d, 1, 6, &l));
    CHECK(!htp_concat_pack_plan(&a, &b, &d, 0, 0, &l));
    CHECK(!htp_concat_pack_plan(&a, &b, &d, 0, 11, &l));
    auto huge = tensor(4, UINT32_MAX, 0x1004); uint64_t bytes;
    CHECK(!htp_concat_pack_span(&huge, &bytes));
    huge = old_a; huge.ne[1] = UINT64_MAX; CHECK(!htp_concat_pack_span(&huge, &bytes));
    std::puts("CONCAT F32 packed span/stride/alias guards and exact worker/vector tails passed");
    return 0;
}
