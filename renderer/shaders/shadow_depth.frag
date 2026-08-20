#version 450
// CSM shadow depth fragment shader.  Opaque fragments write depth implicitly
// (no color attachments); alphaMode MASK materials sample the base-color
// alpha and discard below the cutoff so foliage/fences punch holes in the
// shadow map exactly as they do in the GBuffer pass.  BLEND materials are
// skipped on the CPU (glass does not occlude the sun).

layout(set = 0, binding = 1) uniform MaterialUBO {
    vec4 baseColor; // rgb + alpha factor
    vec4 factors;   // x = metallic, y = roughness, z = occlusionStrength, w = alphaCutoff
    vec4 emissive;  // rgb factor
    vec4 tex0;      // texture indices: baseColor, normal, mr, ao (-1 = none)
    vec4 tex1;      // x = emissive texture index
} material;

layout(set = 1, binding = 0) uniform sampler2D uTextures[1024];

layout(location = 0) in vec2 vUV;

void main() {
    // factors.w > 0 marks alphaMode MASK; everything else is fully opaque.
    if (material.factors.w > 0.0) {
        float alpha = material.baseColor.a;
        const int baseTex = int(floor(material.tex0.x + 0.5));
        if (baseTex >= 0) alpha *= texture(uTextures[baseTex], vUV).a;
        if (alpha < material.factors.w) discard;
    }
}
