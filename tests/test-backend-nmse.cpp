#include "backend-nmse.h"

#include <array>
#include <cstdio>

static bool check(const std::array<float, 2> & a, const std::array<float, 2> & b, double expected) {
    const double actual = nmse(a.data(), b.data(), a.size());
    if (actual == expected) {
        return true;
    }
    std::fprintf(stderr, "NMSE: expected %g, got %g\n", expected, actual);
    return false;
}

int main() {
    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    bool ok = true;
    ok &= check({1, -infinity}, {1, -FLT_MAX}, 0);
    ok &= check({-infinity, -FLT_MAX}, {-FLT_MAX, -infinity}, 0);
    ok &= check({0, 0}, {0, 0}, 0);
    ok &= check({0, 0}, {1, 0}, infinity);
    ok &= check({1, -infinity}, {1, 0}, infinity);
    ok &= check({1, 0}, {1, -FLT_MAX}, infinity);
    ok &= check({-infinity, 1}, {infinity, 1}, infinity);
    ok &= check({nan, 1}, {nan, 1}, infinity);
    ok &= check({1, 1}, {nan, 1}, infinity);
    ok &= check({infinity, 1}, {infinity, 1}, 0);
    ok &= check({infinity, 1}, {FLT_MAX, 1}, infinity);
    ok &= check({FLT_MAX, FLT_MAX}, {0, 0}, 1);
    ok &= check({1, 2}, {2, 4}, 1);
    ok &= check({1, -FLT_MAX}, {2, -infinity}, 1);
    return ok ? 0 : 1;
}
