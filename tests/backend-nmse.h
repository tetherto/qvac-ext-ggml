#pragma once

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <limits>

static constexpr float masked_threshold = -FLT_MAX / 2;

inline double nmse(const float * a, const float * b, size_t n) {
    double squared_error = 0.0;
    double squared_reference = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const bool a_masked = a[i] <= masked_threshold;
        const bool b_masked = b[i] <= masked_threshold;
        if ((a_masked && b_masked) || (std::isinf(a[i]) && a[i] == b[i])) {
            continue;
        }
        if (a_masked != b_masked || !std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return std::numeric_limits<double>::infinity();
        }
        const double difference = double(a[i]) - double(b[i]);
        squared_error += difference * difference;
        squared_reference += double(a[i]) * double(a[i]);
    }
    if (squared_reference == 0.0) {
        return squared_error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
    }
    return squared_error / squared_reference;
}
