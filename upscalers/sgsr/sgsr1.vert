#version 450
// Fullscreen triangle feeding the SGSR1 fragment shader.
// in_TEXCOORD0.xy = display UV (y down, Vulkan convention), matching the
// render-resolution input texture directly (identical aspect ratio).

layout(location = 0) out highp vec4 in_TEXCOORD0;

void main() {
    vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    in_TEXCOORD0 = vec4(p, 0.0, 0.0); // 0..2 range, interpolated to 0..1 across the screen
}
