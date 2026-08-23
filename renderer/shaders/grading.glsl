// Shared log-domain color grading + HDR output encoding.  Applied AFTER the
// HDR lens chain and BEFORE the display transform (Hill fitted ACES, see
// tonemap.glsl) or the HDR output encode.  Mirrored by
// renderer/math/ColorGrading.h (CPU screenshots) — keep the constants and the
// operator order in lockstep.
//
// Grading pipeline (scene-linear Rec.709 input):
//   1. white balance — linear multiply by the (temperature, tint) white-point
//      ratio, computed on the CPU (Tanner Helland black-body approximation,
//      normalized to 6500K; tint moves along the green/magenta axis).
//   2. log encode — simplified ACEScc (SMPTE ST 2065-1): the sub-2^-15 toe is
//      replaced by a plain clamp (display-path only, and the CPU mirror does
//      the same so GPU/CPU screenshots stay identical).
//   3. contrast around the 18% grey pivot (0.18 in ACEScc = 0.4136).
//   4. saturation via Rec.709 luma in the log domain.
//   5. 3D LUT lookup (log domain on both sides; texel-center addressing).
//   6. log decode back to scene linear.
//
// HDR encodes (viewer present only; compare/bench stay SDR — see usage):
//   mode 1 HDR10 — scene nits = linear * exposure * paperWhite (BT.2408
//     graphics white, 203 nits default), Rec.709 -> Rec.2020 primaries,
//     PQ (ITU-R BT.2100 ST 2084), clamped to the 10000-nit PQ ceiling.
//   mode 2 scRGB — linear passthrough, 1.0 = 80 nits (Windows scRGB
//     convention), primaries stay Rec.709/sRGB.

#ifndef SR_GRADING_GLSL
#define SR_GRADING_GLSL

// ACEScc of 0.18 (18% grey): (log2(0.18) + 9.72) / 17.52.  Named constant so
// the CPU mirror stays in lockstep.
const float kLogPivot = 0.41359;
const float kAcesccMin = 0.000030517578125; // 2^-15 clamp (simplified toe)

vec3 linearToLog(vec3 c) {
    return (log2(max(c, vec3(kAcesccMin))) + 9.72) / 17.52;
}

vec3 logToLinear(vec3 l) {
    return exp2(l * 17.52 - 9.72);
}

// gradeA = (contrast, saturation, hdrMode, paperWhiteNits)
// gradeB = (whiteBalanceR, whiteBalanceG, whiteBalanceB, lutSize)
vec3 applyColorGrading(vec3 sceneLinear, vec4 gradeA, vec4 gradeB, sampler3D lut) {
    vec3 c = sceneLinear * gradeB.rgb;
    vec3 l = linearToLog(c);
    l = (l - kLogPivot) * gradeA.x + kLogPivot;
    const float luma = dot(l, vec3(0.2126, 0.7152, 0.0722));
    l = mix(vec3(luma), l, gradeA.y);
    // LUT lookup in the log domain; texel-center addressing maps [0,1] onto
    // the N lattice points (with linear filtering the identity LUT is exact).
    const float n = gradeB.w;
    const vec3 uvw = clamp(l, 0.0, 1.0) * ((n - 1.0) / n) + 0.5 / n;
    l = texture(lut, uvw).rgb;
    return logToLinear(l);
}

// ITU-R BT.2100-2 ST 2084 (PQ) inverse EOTF, input in nits.
vec3 pqEncode(vec3 nits) {
    const vec3 y = clamp(nits * (1.0 / 10000.0), 0.0, 1.0);
    const float m1 = 2610.0 / 16384.0;
    const float m2 = 2523.0 / 32.0;
    const float c1 = 3424.0 / 4096.0;
    const float c2 = 2413.0 / 128.0;
    const float c3 = 2392.0 / 128.0;
    const vec3 ym = pow(y, vec3(m1));
    return pow((c1 + c2 * ym) / (1.0 + c3 * ym), vec3(m2));
}

// Rec.709 -> Rec.2020 primaries (ITU-R BT.2087-0 conversion of the linear
// signal, rounded to 4 decimals as in BT.2408).  Column-major, like
// tonemap.glsl.
const mat3 kRec709ToRec2020 = mat3(
    1.6605, -0.1246, -0.0182,
    -0.5876,  1.1329, -0.1006,
    -0.0728, -0.0083,  1.1187
);

// HDR display encode for the viewer present pass.  hdrMode: 1 = HDR10 (PQ),
// 2 = scRGB.  SDR (mode 0) goes through tonemapToDisplay instead.
vec3 encodeHdrDisplay(vec3 gradedLinear, float exposure, float paperWhiteNits, float hdrMode) {
    const vec3 c = gradedLinear * exposure;
    if (hdrMode > 1.5) {
        // scRGB: linear output, 1.0 = 80 nits (Windows scRGB convention);
        // scale so the paper-white constant lands on its absolute value.
        return c * (paperWhiteNits / 80.0);
    }
    // HDR10: scene linear -> nits via the paper-white calibration, Rec.2020
    // primaries, PQ-encoded to the 10000-nit ceiling.
    return pqEncode(kRec709ToRec2020 * (c * paperWhiteNits));
}

#endif
