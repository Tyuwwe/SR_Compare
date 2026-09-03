#version 450
// Deferred GBuffer vertex shader, skinned variant: positions come from the
// joint palette (palette = jointGlobal * inverseBind, so no model matrix is
// applied — glTF skinned meshes ignore their node's transform).  Motion is
// computed from the CURRENT and PREVIOUS frame palettes, both skinned here,
// so dynamic meshes get correct per-pixel motion vectors.

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;
layout(location = 3) in vec4 aTangent;
layout(location = 4) in uvec4 aJoints;
layout(location = 5) in vec4 aWeights;

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

// Both palettes live in one buffer: [current joints][previous joints].
layout(set = 0, binding = 2) readonly buffer JointPalette {
    mat4 bones[];
} palette;

layout(push_constant) uniform Push {
    mat4 model;       // unused by skinned draws (kept for layout compatibility)
    mat4 prevModel;   // unused
    mat4 normalModel; // unused
    uint paletteCur;  // first current-frame joint of this draw's skin
    uint palettePrev; // first previous-frame joint
    uvec2 pad;
} pc;

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec2 vUV;
layout(location = 3) out vec4 vTangent;
layout(location = 4) out vec4 vCurrentClip;
layout(location = 5) out vec4 vPreviousClip;

void main() {
    mat4 skin = mat4(0.0);
    mat4 skinPrev = mat4(0.0);
    for (int i = 0; i < 4; ++i) {
        skin += aWeights[i] * palette.bones[pc.paletteCur + aJoints[i]];
        skinPrev += aWeights[i] * palette.bones[pc.palettePrev + aJoints[i]];
    }

    vec4 world = skin * vec4(aPos, 1.0);
    vec4 prevWorld = skinPrev * vec4(aPos, 1.0);
    vWorldPos = world.xyz;
    // Linear-blend skinning of normals with the blended matrix is approximate
    // (no inverse-transpose per joint); fine for near-rigid bone transforms.
    vNormal = normalize(mat3(skin) * aNormal);
    vUV = aUV;
    vTangent = vec4(normalize(mat3(skin) * aTangent.xyz), aTangent.w);

    vec4 clip = ubo.viewProj * world;
    gl_Position = clip;

    // Both projections are unjittered. Temporal jitter is reported separately
    // to each upscaler and must not be baked into object/camera motion.
    // Perspective division is deferred until the fragment stage so motion is
    // exact across large triangles.
    vCurrentClip = ubo.viewProjNoJitter * world;
    vPreviousClip = ubo.prevViewProj * prevWorld;
}
