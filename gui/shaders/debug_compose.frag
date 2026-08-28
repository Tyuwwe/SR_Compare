#version 450
#extension GL_GOOGLE_include_directive : require
#include "tonemap.glsl"
// Debug-tab fullscreen texture visualization (patch 0004 design, rewritten
// for the ImGui GUI).  Draws the selected render-res input / GBuffer texture
// or an upscaler output into the composite with a mode-dependent display
// transform.  Bound through the compare compose set layout (only binding 0
// is declared; the text/font/LUT bindings stay written but unused) and the
// compose pipeline layout (this 48-byte push block fits in its fragment
// range), so no extra layouts are needed.
//
// Modes:
//   0 HDR color — shared display transform (Hill fitted ACES + gamma, same
//                 as the column compose) with the LR path's display exposure.
//   1 depth     — D32 -> positive eye depth (Vulkan finite-far, non-reversed
//                 perspective), normalized to [near, range], optionally
//                 logarithmic.
//   2 motion    — stored previousUV - currentUV (framebuffer UV units, the
//                 canonical convention): hue = current-to-previous direction,
//                 brightness = magnitude in source pixels relative to range.
//   3 heat      — single-channel heat map with gain (reactive mask).
//   4 LDR color — plain clamp (plugin SDK views that are already display-
//                 ready).

layout(set = 0, binding = 0) uniform sampler2D uSource;

layout(push_constant) uniform Push {
    vec4 params; // xy = source size in pixels, z = 1 => nearest, w = exposure
    vec4 debug;  // x = mode, y = range/gain, z = near plane, w = far plane
    vec4 flags;  // x = logarithmic depth
} pc;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

vec3 hsvToRgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

vec3 heatMap(float v) {
    v = clamp(v, 0.0, 1.0);
    return clamp(vec3(1.5 * v, 1.5 - abs(3.0 * v - 1.5), 1.5 * (1.0 - v)), 0.0, 1.0);
}

void main() {
    vec4 sampleValue;
    if (pc.params.z > 0.5) {
        // Nearest: round the source-texel coordinate, clamp to the image.
        vec2 t = vUV * pc.params.xy - 0.5;
        ivec2 ip = ivec2(clamp(floor(t + 0.5), vec2(0.0), pc.params.xy - 1.0));
        sampleValue = texelFetch(uSource, ip, 0);
    } else {
        sampleValue = texture(uSource, vUV);
    }

    vec3 c;
    const int mode = int(pc.debug.x + 0.5);
    if (mode == 1) {
        float n = max(pc.debug.z, 1e-5);
        float f = max(pc.debug.w, n + 1e-4);
        float a = f / (f - n);
        float b = n * f / (f - n);
        float eyeDepth = b / max(a - sampleValue.r, 1e-6);
        float maxDepth = max(pc.debug.y, n + 1e-4);
        float v = pc.flags.x > 0.5
                      ? log(max(eyeDepth / n, 1.0)) / log(max(maxDepth / n, 1.0001))
                      : (eyeDepth - n) / (maxDepth - n);
        c = vec3(clamp(v, 0.0, 1.0));
    } else if (mode == 2) {
        // Keep the threshold in source-pixel units.
        vec2 motionPixels = sampleValue.rg * pc.params.xy;
        float magnitude = length(motionPixels);
        float hue = fract(atan(motionPixels.y, motionPixels.x) / 6.28318530718 + 1.0);
        float brightness = clamp(magnitude / max(pc.debug.y, 1e-4), 0.0, 1.0);
        c = hsvToRgb(vec3(hue, magnitude > 1e-5 ? 1.0 : 0.0, brightness));
    } else if (mode == 3) {
        c = heatMap(sampleValue.r * pc.debug.y);
    } else if (mode == 4) {
        c = clamp(sampleValue.rgb, 0.0, 1.0);
    } else {
        c = tonemapToDisplay(sampleValue.rgb, pc.params.w);
    }

    outColor = vec4(c, 1.0);
}
