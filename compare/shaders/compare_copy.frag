#version 450
// Passthrough copy of the composed offscreen image into the swapchain image
// (swapchain images lack TRANSFER_DST usage, so a blit is not possible).

layout(set = 0, binding = 0) uniform sampler2D uImage;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(texture(uImage, vUV).rgb, 1.0);
}
