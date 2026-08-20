#version 450
// Deferred GBuffer vertex shader: transforms to current (possibly jittered)
// clip space and computes the motion vector against the previous, un-jittered
// frame.  Same UBO/push-constant contract as the legacy forward scene.vert.

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
    // Tangents survive non-uniform scale acceptably with the plain model
    // matrix; the fragment shader re-orthogonalizes against the normal.
    vTangent = vec4(normalize(mat3(pc.model) * aTangent.xyz), aTangent.w);

    vec4 clip = ubo.viewProj * world;
    gl_Position = clip;

    // Motion against the previous frame, computed from the *un-jittered*
    // projections on both sides (jitter is resolved by TAA, not encoded here).
    vec4 curClipNoJitter = ubo.viewProjNoJitter * world;
    vec4 prevClip = ubo.prevViewProj * (pc.prevModel * vec4(aPos, 1.0));
    vec2 curNDC = curClipNoJitter.xy / curClipNoJitter.w;
    vec2 prevNDC = prevClip.xy / prevClip.w;
    // Pixel units, no jitter.  Points from previous position to current.
    vMotion = (curNDC - prevNDC) * 0.5 * ubo.renderSizeJitter.xy;
}
