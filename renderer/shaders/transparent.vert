#version 450
// Transparency pass vertex shader: identical contract to gbuffer.vert
// (current jittered clip position + unjittered current/previous clip data).

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;
layout(location = 3) in vec4 aTangent;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 viewProj;        // current, jitter applied when rendering the GBuffer
    mat4 viewProjNoJitter;
    mat4 prevViewProj;    // previous frame, no jitter
    vec4 cameraPos;
    vec4 ambient;         // unused here; lights live in LightingUBO
    vec4 renderSizeJitter; // xy = render size (pixels), zw = jitter (pixels)
    vec4 clipPlane;    // planar mirror plane (xyz normal toward the camera, w offset)
    mat4 reflViewProj; // mirrored-view view-projection (main view, reflParams.x = 1)
    vec4 reflParams;   // x = reflection image available, y = clip behind clipPlane,
                       // zw = mirrored proj m[10]/m[14] (viewZ = w / (depth + z))
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
layout(location = 4) out vec4 vCurrentClip;
layout(location = 5) out vec4 vPreviousClip;

void main() {
    vec4 world = pc.model * vec4(aPos, 1.0);
    vWorldPos = world.xyz;
    vNormal = normalize(mat3(pc.normalModel) * aNormal);
    vUV = aUV;
    vTangent = vec4(normalize(mat3(pc.model) * aTangent.xyz), aTangent.w);

    gl_Position = ubo.viewProj * world;

    // Perspective division is deferred until the fragment stage so motion is
    // exact across large triangles. Both projections deliberately omit jitter.
    vCurrentClip = ubo.viewProjNoJitter * world;
    vPreviousClip = ubo.prevViewProj * (pc.prevModel * vec4(aPos, 1.0));
}
