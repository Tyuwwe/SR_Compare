#version 450
// Present pass: sample the final HDR image, apply Reinhard tone mapping and
// gamma 2.2.  The CPU screenshot path mirrors this exactly.

layout(set = 0, binding = 0) uniform sampler2D uImage;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

void main() {
    vec3 c = texture(uImage, vUV).rgb;
    c = c / (1.0 + c);
    c = pow(c, vec3(1.0 / 2.2));
    outColor = vec4(c, 1.0);
}
