#version 450
#extension GL_GOOGLE_include_directive : require
// Passthrough copy of the composed offscreen image into the swapchain image
// (swapchain images lack TRANSFER_DST usage, so a blit is not possible).
//
// HDR branch (Phase 6c, GUI checkbox): the composite is an SDR display-encoded
// (gamma 2.2) image, so the HDR modes re-linearize it and re-encode for the
// swapchain — SDR content in an HDR container.  True HDR headroom (scene HDR
// through grading -> PQ/scRGB without the SDR tonemap) is viewer-only.
// Also used as the GT-SSAA downsample shader; that path pushes mode 0
// (passthrough — the downsample target is HDR linear, not the swapchain).

#include "grading.glsl"

layout(set = 0, binding = 0) uniform sampler2D uImage;

layout(push_constant) uniform Push {
    vec4 params; // x = hdr mode (0 SDR passthrough / 1 HDR10 PQ / 2 scRGB),
                 // y = paper-white nits, zw unused
} pc;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

void main() {
    const vec3 c = texture(uImage, vUV).rgb;
    if (pc.params.x < 0.5) {
        outColor = vec4(c, 1.0);
        return;
    }
    // Undo the display gamma (composite is gamma-2.2 encoded), then encode.
    const vec3 linear = pow(max(c, vec3(0.0)), vec3(2.2));
    outColor = vec4(encodeHdrDisplay(linear, 1.0, pc.params.y, pc.params.x), 1.0);
}
