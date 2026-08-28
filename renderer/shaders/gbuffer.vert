#version 450
// Deferred GBuffer vertex shader: transforms to current (possibly jittered)
// clip space and passes the current/previous un-jittered clip positions to the
// fragment shader.  The perspective divides must happen per fragment: dividing
// per vertex and interpolating the result is not exact on large triangles.
// Same UBO contract as the legacy forward scene.vert; per-instance
// transforms come from the instance SSBO (Phase 7a) at gl_InstanceIndex (the
// indirect command's firstInstance) instead of push constants, so static
// draws can be issued as vkCmdDrawIndexedIndirect.

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
} ubo;

// std430 mirror of sr::GpuInstance (DeferredCore.h).
struct GpuInstance {
    mat4 model;
    mat4 prevModel;
    mat4 normalModel; // transpose(inverse(mat3(model))) in the upper 3x3
    vec4 aabbMin;     // world-space bounds (used by occlusion_cull.comp only)
    vec4 aabbMax;
    uint materialIndex;
    uint flags;
    uint pad0;
    uint pad1;
};
layout(set = 0, binding = 3) readonly buffer InstanceBuffer {
    GpuInstance gInstances[];
};

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec2 vUV;
layout(location = 3) out vec4 vTangent;
layout(location = 4) out vec4 vCurrentClip;
layout(location = 5) out vec4 vPreviousClip;

void main() {
    GpuInstance inst = gInstances[gl_InstanceIndex];
    vec4 world = inst.model * vec4(aPos, 1.0);
    vWorldPos = world.xyz;
    vNormal = normalize(mat3(inst.normalModel) * aNormal);
    vUV = aUV;
    // Tangents survive non-uniform scale acceptably with the plain model
    // matrix; the fragment shader re-orthogonalizes against the normal.
    vTangent = vec4(normalize(mat3(inst.model) * aTangent.xyz), aTangent.w);

    vec4 clip = ubo.viewProj * world;
    gl_Position = clip;

    // Both projections are unjittered. Temporal jitter is reported separately
    // to each upscaler and must not be baked into object/camera motion.
    vCurrentClip = ubo.viewProjNoJitter * world;
    vPreviousClip = ubo.prevViewProj * (inst.prevModel * vec4(aPos, 1.0));
}
