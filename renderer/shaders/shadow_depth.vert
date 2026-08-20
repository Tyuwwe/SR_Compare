#version 450
// CSM shadow depth vertex shader: transforms each caster into the current
// cascade's light clip space.  Depth-only pass; the fragment shader only
// alpha-discards MASK materials.  Reuses the scene pipeline layout (set0 =
// scene/material UBO, set1 = texture array), but only the push constants are
// read here.

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;  // unused; declared to match the scene vertex layout
layout(location = 2) in vec2 aUV;
layout(location = 3) in vec4 aTangent; // unused

layout(push_constant) uniform Push {
    mat4 model;
    mat4 lightVP; // current cascade's light view-projection
} pc;

layout(location = 0) out vec2 vUV;

void main() {
    vUV = aUV;
    gl_Position = pc.lightVP * (pc.model * vec4(aPos, 1.0));
}
