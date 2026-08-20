#version 450
// Transparency pass vertex shader: identical contract to gbuffer.vert
// (current jittered clip position + un-jittered cur-vs-prev motion).

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;
layout(location = 3) in vec4 aTangent;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 viewProj;        // current, jitter applied when rendering the GBuffer
    mat4 viewProjNoJitter;
    mat4 prevViewProj;    // previous frame, no jitter
    vec4 cameraPos;
    vec4 light0Pos;
    vec4 light0Color;
    vec4 light1Pos;
    vec4 light1Color;
    vec4 ambient;
    vec4 renderSizeJitter; // xy = render size (pixels), zw = jitter (pixels)
} ubo;

layout(push_constant) uniform Push {
    mat4 model;
    mat4 prevModel;
    mat4 normalModel; // transpose(inverse(mat3(model))) in the upper 3x3
} pc;

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec2 vUV;
layout(location = 3) out vec4 vTangent;
layout(location = 4) out vec2 vMotion;

void main() {
    vec4 world = pc.model * vec4(aPos, 1.0);
    vWorldPos = world.xyz;
    vNormal = normalize(mat3(pc.normalModel) * aNormal);
    vUV = aUV;
    vTangent = vec4(normalize(mat3(pc.model) * aTangent.xyz), aTangent.w);

    gl_Position = ubo.viewProj * world;

    // Motion against the previous frame, un-jittered projections on both
    // sides (same convention as the GBuffer pass).
    vec4 curClipNoJitter = ubo.viewProjNoJitter * world;
    vec4 prevClip = ubo.prevViewProj * (pc.prevModel * vec4(aPos, 1.0));
    vec2 curNDC = curClipNoJitter.xy / curClipNoJitter.w;
    vec2 prevNDC = prevClip.xy / prevClip.w;
    vMotion = (curNDC - prevNDC) * 0.5 * ubo.renderSizeJitter.xy;
}
