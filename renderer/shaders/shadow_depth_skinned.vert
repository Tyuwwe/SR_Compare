#version 450
// CSM shadow depth vertex shader, skinned variant: current-frame joint palette
// only (the shadow pass writes no motion vectors).  Reuses the scene pipeline
// layout like the static shadow_depth.vert.

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;  // unused; declared to match the skinned vertex layout
layout(location = 2) in vec2 aUV;
layout(location = 3) in vec4 aTangent; // unused
layout(location = 4) in uvec4 aJoints;
layout(location = 5) in vec4 aWeights;

layout(set = 0, binding = 2) readonly buffer JointPalette {
    mat4 bones[];
} palette;

layout(push_constant) uniform Push {
    mat4 model;   // unused (palette already carries the node transform)
    mat4 lightVP; // current cascade's light view-projection
    uint paletteCur;
    uvec3 pad;
} pc;

layout(location = 0) out vec2 vUV;

void main() {
    mat4 skin = mat4(0.0);
    for (int i = 0; i < 4; ++i) {
        skin += aWeights[i] * palette.bones[pc.paletteCur + aJoints[i]];
    }
    vUV = aUV;
    gl_Position = pc.lightVP * (skin * vec4(aPos, 1.0));
}
