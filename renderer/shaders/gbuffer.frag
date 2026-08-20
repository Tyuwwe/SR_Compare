#version 450
// Deferred GBuffer fragment shader (upscaler input path).  Writes the material
// properties; lighting happens in lighting.frag.
//   RT0 albedo   RGBA8_SRGB  rgb = base color (linear->sRGB on write), a = alpha
//   RT1 normal   A2B10G10R10 xyz = world normal * 0.5 + 0.5 (packed [0,1])
//   RT2 material RGBA8_UNORM r = metallic, g = roughness, b = AO
//   RT3 emissive B10G11R11UF rgb = emissive (unsigned float, no alpha)
//   RT4 motion   RG16F       pixel units, cur - prev, no jitter (unchanged
//                            upscaler contract)

layout(set = 0, binding = 1) uniform MaterialUBO {
    vec4 baseColor; // rgb + alpha factor
    vec4 factors;   // x = metallic, y = roughness, z = occlusionStrength, w = alphaCutoff (0 = opaque)
    vec4 emissive;  // rgb factor
    vec4 tex0;      // texture indices: baseColor, normal, mr, ao (-1 = none)
    vec4 tex1;      // x = emissive texture index
} material;

layout(set = 1, binding = 0) uniform sampler2D uTextures[1024];

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vUV;
layout(location = 3) in vec4 vTangent;
layout(location = 4) in vec2 vMotion;

layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outMaterial;
layout(location = 3) out vec3 outEmissive; // B10G11R11_UFLOAT: no alpha channel
layout(location = 4) out vec2 outMotion;

// floor() so that texIndex == -1 (untextured) rounds to -1, not 0.
int texIndex(float f) { return int(floor(f + 0.5)); }

void main() {
    vec4 base = material.baseColor;
    const int baseTex = texIndex(material.tex0.x);
    if (baseTex >= 0) base *= texture(uTextures[baseTex], vUV);

    // alphaMode MASK (alpha cutout: foliage, fences).
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
    outNormal = vec4(N * 0.5 + 0.5, 1.0); // A2B10G10R10 is unsigned: remap [-1,1] to [0,1]
    outMaterial = vec4(clamp(metallic, 0.0, 1.0), clamp(roughness, 0.0, 1.0),
                       clamp(ao, 0.0, 1.0), 1.0);
    outEmissive = emissive;
    outMotion = vMotion;
}
