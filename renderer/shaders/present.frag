#version 450
#extension GL_GOOGLE_include_directive : require
// Present pass: sample the final HDR image and apply the shared display
// transform (Hill fitted ACES + gamma 2.2).  The CPU screenshot path
// (tonemapToDisplay in renderer/math/Tonemap.h) mirrors this exactly.

#include "tonemap.glsl"

layout(set = 0, binding = 0) uniform sampler2D uImage;

layout(push_constant) uniform Push {
    vec4 exposure; // x = display exposure (scene linear multiplier)
} pc;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

void main() {
    vec3 c = texture(uImage, vUV).rgb;
    outColor = vec4(tonemapToDisplay(c, pc.exposure.x), 1.0);
}
