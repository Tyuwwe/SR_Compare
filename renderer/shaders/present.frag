#version 450
#extension GL_GOOGLE_include_directive : require
// Present pass: sample the final HDR image, run the terminal lens-effects
// chain, the log-domain color grading (Phase 6c), then the shared display
// transform (Hill fitted ACES + gamma 2.2) for SDR, or the HDR output encode
// (HDR10 PQ / scRGB, see grading.glsl) when the swapchain is HDR.
// The CPU screenshot path (tonemapToDisplay + applyColorGrading in
// renderer/math/) mirrors the tonemap and grading exactly; it intentionally
// does NOT reproduce the lens chain (screenshots capture the scene, not the
// lens).
//
// Lens chain (all effects default to weak, conservative strengths; a zero
// strength skips the effect entirely):
//   1. chromatic aberration — radial RGB split growing with the squared
//      distance from the image center (HDR domain, pre-grading);
//   2. lens dirt — procedural mask x accumulated bloom, additive in HDR
//      (dirt is only visible where bright light hits the lens; UE4 "Lens
//      Dirt", Kawase GDC 2003 frame-buffer post-processing);
//   3. vignette — quadratic falloff (display domain for SDR, linear domain
//      for HDR output — it is multiplicative either way);
//   4. film grain — frame-seeded integer-hash noise (SDR display domain
//      only; skipped for HDR output where a display-encoded offset has no
//      fixed meaning);
//      deterministic: no wall clock, no cross-run state.

#include "tonemap.glsl"
#include "grading.glsl"

layout(set = 0, binding = 0) uniform sampler2D uImage;
layout(set = 0, binding = 1) uniform sampler2D uBloom; // accumulated bloom mip 0 (path res)
layout(set = 0, binding = 2) uniform sampler2D uDirt;  // procedural lens-dirt mask
layout(set = 0, binding = 3) uniform sampler3D uLut;   // log-domain color grading LUT

layout(push_constant) uniform Push {
    vec4 exposure; // x = display exposure (scene linear multiplier)
    vec4 lensA;    // x = chromatic aberration (radial UV scale), y = vignette,
                   // z = film grain, w = frame index (grain hash seed)
    vec4 lensB;    // x = lens dirt strength, yzw unused
    vec4 gradeA;   // x = contrast, y = saturation, z = hdr mode (0 SDR / 1 HDR10 PQ
                   // / 2 scRGB), w = paper-white nits (HDR10 calibration)
    vec4 gradeB;   // xyz = white balance (temperature/tint), w = LUT size
} pc;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

// Deterministic integer hash -> [0,1) (LCG + xorshift mixing; cf. Wellons,
// "Hash functions for GPU rendering").  Float sin-hashes are avoided: their
// precision differs across GPUs.
float hash13(uvec3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    return float(v.x & 0x00FFFFFFu) * (1.0 / 16777216.0);
}

void main() {
    const vec2 d = vUV - 0.5;
    vec3 c;
    if (pc.lensA.x > 0.0) {
        // Radial channel split, proportional to the squared distance from the
        // center (zero on axis, strongest in the corners).
        const vec2 off = d * dot(d, d) * pc.lensA.x;
        c.r = texture(uImage, vUV - off).r;
        c.g = texture(uImage, vUV).g;
        c.b = texture(uImage, vUV + off).b;
    } else {
        c = texture(uImage, vUV).rgb;
    }
    // Lens dirt modulates the bloom only (a clean-lens frame is unchanged).
    if (pc.lensB.x > 0.0) {
        c += texture(uBloom, vUV).rgb * (texture(uDirt, vUV).r * pc.lensB.x);
    }
    // Log-domain color grading (Phase 6c), before the display transform.
    c = applyColorGrading(c, pc.gradeA, pc.gradeB, uLut);
    const bool hdr = pc.gradeA.z > 0.5;
    if (hdr) {
        // HDR output: no ACES/gamma — encodeHdrDisplay does the nits
        // calibration + PQ / scRGB mapping.  Vignette stays multiplicative
        // (applied in the linear domain here); film grain is skipped (a
        // display-encoded offset has no fixed meaning under PQ).
        if (pc.lensA.y > 0.0) {
            c *= 1.0 - pc.lensA.y * dot(d, d) * 2.0;
        }
        outColor = vec4(encodeHdrDisplay(c, pc.exposure.x, pc.gradeA.w, pc.gradeA.z), 1.0);
        return;
    }
    c = tonemapToDisplay(c, pc.exposure.x);
    // Quadratic vignette: 1 at the center, 1 - strength at the corners.
    if (pc.lensA.y > 0.0) {
        c *= 1.0 - pc.lensA.y * dot(d, d) * 2.0;
    }
    // Zero-centered film grain, seeded by the frame index.
    if (pc.lensA.z > 0.0) {
        const float n = hash13(uvec3(uvec2(gl_FragCoord.xy), uint(pc.lensA.w))) - 0.5;
        c += pc.lensA.z * n;
    }
    outColor = vec4(c, 1.0);
}
