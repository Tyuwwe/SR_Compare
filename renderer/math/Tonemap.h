#pragma once
// ============================================================================
// CPU mirror of renderer/shaders/tonemap.glsl (Stephen Hill fitted ACES +
// gamma 2.2).  Used by the Viewer HDR screenshot path so PNG pixels match
// present.frag.  Keep the constants and the operator order in lockstep.
// ============================================================================
#include "renderer/math/Math.h"

#include <algorithm>
#include <cmath>

namespace sr {

inline constexpr float kDisplayExposure = 1.f;

namespace tonemap_detail {

// Row-major 3x3 * column vector.
inline Vec3 mulMat3(const float m[3][3], const Vec3& v) {
    return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z};
}

inline Vec3 rrtAndOdtFit(const Vec3& v) {
    const Vec3 a{v.x * (v.x + 0.0245786f) - 0.000090537f,
                 v.y * (v.y + 0.0245786f) - 0.000090537f,
                 v.z * (v.z + 0.0245786f) - 0.000090537f};
    const Vec3 b{v.x * (0.983729f * v.x + 0.4329510f) + 0.238081f,
                 v.y * (0.983729f * v.y + 0.4329510f) + 0.238081f,
                 v.z * (0.983729f * v.z + 0.4329510f) + 0.238081f};
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

} // namespace tonemap_detail

// Linear Rec.709/sRGB HDR -> display-referred [0,1] with gamma 2.2.
inline Vec3 tonemapToDisplay(Vec3 color, float exposure = kDisplayExposure) {
    color = color * exposure;
    static const float kAcesInput[3][3] = {
        {0.59719f, 0.35458f, 0.04823f},
        {0.07600f, 0.90834f, 0.01566f},
        {0.02840f, 0.13383f, 0.83777f},
    };
    static const float kAcesOutput[3][3] = {
        {1.60475f, -0.53108f, -0.07367f},
        {-0.10208f, 1.10813f, -0.00605f},
        {-0.00327f, -0.07276f, 1.07602f},
    };
    color = tonemap_detail::mulMat3(kAcesInput, color);
    color = tonemap_detail::rrtAndOdtFit(color);
    color = tonemap_detail::mulMat3(kAcesOutput, color);
    color.x = std::clamp(color.x, 0.f, 1.f);
    color.y = std::clamp(color.y, 0.f, 1.f);
    color.z = std::clamp(color.z, 0.f, 1.f);
    const float invGamma = 1.f / 2.2f;
    color.x = std::pow(color.x, invGamma);
    color.y = std::pow(color.y, invGamma);
    color.z = std::pow(color.z, invGamma);
    return color;
}

} // namespace sr
