#version 450
// Deferred GBuffer fragment shader for the native ground-truth path.
// Identical to gbuffer.frag except it has no motion output (the GT GBuffer has
// four color attachments), which keeps dynamic rendering validation clean.

layout(set = 0, binding = 1) uniform MaterialUBO {
    vec4 baseColor;
    vec4 factors;   // x = metallic, y = roughness, z = occlusionStrength, w = alphaCutoff
    vec4 emissive;
    vec4 tex0;      // baseColor, normal, mr, ao
    vec4 tex1;      // x = emissive
} material;

layout(set = 1, binding = 0) uniform sampler2D uTextures[1024];

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vUV;
layout(location = 3) in vec4 vTangent;
layout(location = 4) in vec2 vMotion; // unused (GT has no motion attachment)

layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outMaterial;
layout(location = 3) out vec4 outEmissive;

int texIndex(float f) { return int(floor(f + 0.5)); }

void main() {
    vec4 base = material.baseColor;
    const int baseTex = texIndex(material.tex0.x);
    if (baseTex >= 0) base *= texture(uTextures[baseTex], vUV);

    if (material.factors.w > 0.0 && base.a < material.factors.w) discard;

    float metallic = material.factors.x;
    float roughness = material.factors.y;
    const int mrTex = texIndex(material.tex0.z);
    if (mrTex >= 0) {
        const vec4 mr = texture(uTextures[mrTex], vUV);
        roughness *= mr.g;
        metallic *= mr.b;
    }

    float ao = 1.0;
    const int aoTex = texIndex(material.tex0.w);
    if (aoTex >= 0) {
        ao = mix(1.0, texture(uTextures[aoTex], vUV).r, material.factors.z);
    }

    vec3 N = normalize(vNormal);
    const int normalTex = texIndex(material.tex0.y);
    if (normalTex >= 0) {
        vec3 T = normalize(vTangent.xyz - N * dot(N, vTangent.xyz));
        vec3 B = cross(N, T) * vTangent.w;
        vec3 n = texture(uTextures[normalTex], vUV).xyz * 2.0 - 1.0;
        N = normalize(mat3(T, B, N) * n);
    }

    vec3 emissive = material.emissive.rgb;
    const int emTex = texIndex(material.tex1.x);
    if (emTex >= 0) emissive *= texture(uTextures[emTex], vUV).rgb;

    outAlbedo = vec4(base.rgb, base.a);
    outNormal = vec4(N, 0.0);
    outMaterial = vec4(clamp(metallic, 0.0, 1.0), clamp(roughness, 0.0, 1.0),
                       clamp(ao, 0.0, 1.0), 1.0);
    outEmissive = vec4(emissive, 1.0);
}
