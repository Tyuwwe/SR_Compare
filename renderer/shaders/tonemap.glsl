// Shared display transform.  Mirrored by renderer/math/Tonemap.h (CPU screenshots).
// Stephen Hill fitted ACES (RRT+ODT) + gamma 2.2.  Input is linear Rec.709/sRGB HDR.
// Must stay identical across present.frag, compare_compose.frag, and
// compare_metrics_blocks.comp — the metric domain is this display encoding.
//
// Hill lists these row-major.  GLSL mat3(9 floats) is column-major, so each
// triplet below is one column of that row-major table (a compile-time
// constant; transpose() is not allowed in a global const initializer).

#ifndef SR_TONE_MAP_GLSL
#define SR_TONE_MAP_GLSL

// Default when a caller has no runtime exposure (kept as a named constant so
// the CPU mirror in Tonemap.h stays in lockstep).  All display-path shaders
// pass a push-constant exposure instead of this value.
const float kDisplayExposure = 1.0;

const mat3 kAcesInputMat = mat3(
    0.59719, 0.07600, 0.02840,
    0.35458, 0.90834, 0.13383,
    0.04823, 0.01566, 0.83777
);

const mat3 kAcesOutputMat = mat3(
     1.60475, -0.10208, -0.00327,
    -0.53108,  1.10813, -0.07276,
    -0.07367, -0.00605,  1.07602
);

vec3 rrtAndOdtFit(vec3 v) {
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

vec3 tonemapToDisplay(vec3 hdrLinear, float exposure) {
    vec3 color = hdrLinear * exposure;
    color = kAcesInputMat * color;
    color = rrtAndOdtFit(color);
    color = kAcesOutputMat * color;
    color = clamp(color, 0.0, 1.0);
    return pow(color, vec3(1.0 / 2.2));
}

#endif
