#version 450
// GT-path transparency fragment shader: identical shading to transparent.frag
// but only the blended color output (the GT path has no motion/mask targets).
// Keep the bodies in sync.

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 viewProj;
    mat4 viewProjNoJitter;
    mat4 prevViewProj;
    vec4 cameraPos;
    vec4 light0Pos;
    vec4 light0Color;
    vec4 light1Pos;
    vec4 light1Color;
    vec4 ambient;
    vec4 renderSizeJitter;
} ubo;

layout(set = 0, binding = 1) uniform MaterialUBO {
    vec4 baseColor;
    vec4 factors;
    vec4 emissive;
    vec4 tex0;
    vec4 tex1;
} material;

layout(set = 1, binding = 0) uniform sampler2D uTextures[1024];

layout(set = 2, binding = 0) uniform LightingUBO {
    mat4 invViewProj;
    vec4 cameraPos;
    vec4 light0Pos;
    vec4 light0Color;
    vec4 light1Pos;
    vec4 light1Color;
    vec4 ambient;
    vec4 iblParams;
} lighting;
layout(set = 2, binding = 1) uniform samplerCube iblIrradiance;
layout(set = 2, binding = 2) uniform samplerCube iblPrefilter;
layout(set = 2, binding = 3) uniform sampler2D iblBrdfLut;
// Screen-space AO (R16F, same resolution as this path's GBuffer).
layout(set = 2, binding = 4) uniform sampler2D ssaoTex;

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vUV;
layout(location = 3) in vec4 vTangent;
layout(location = 4) in vec2 vMotion; // unused in the GT path

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

int texIndex(float f) { return int(floor(f + 0.5)); }

float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdH = max(dot(N, H), 0.0);
    float d = NdH * NdH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float geometrySchlickGGX(float NdX, float roughness) {
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    return NdX / (NdX * (1.0 - k) + k);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    return geometrySchlickGGX(max(dot(N, V), 0.0), roughness) *
           geometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    vec3 F1 = max(vec3(1.0 - roughness), F0);
    return F0 + (F1 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Returns the point-light BRDF split into its diffuse and specular lobes.
// The glass model weights them differently: specular is surface reflection,
// whose energy does not depend on transmission (opacity), while the diffuse
// lobe approximates the transmitted/tinted term and scales with opacity.
void shadePointLight(vec3 N, vec3 V, vec3 worldPos, vec3 albedo, float metallic,
                     float roughness, vec3 F0, vec3 lightPos, vec3 lightColor,
                     out vec3 diffuse, out vec3 specular) {
    diffuse = vec3(0.0);
    specular = vec3(0.0);

    vec3 toLight = lightPos - worldPos;
    float dist = length(toLight);
    vec3 L = toLight / dist;
    vec3 radiance = lightColor / (dist * dist);

    vec3 H = normalize(V + L);
    float NdL = max(dot(N, L), 0.0);
    if (NdL <= 0.0) return;

    float D = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
    specular = (D * G * F) / max(4.0 * max(dot(N, V), 0.0) * NdL, 1e-4) * radiance * NdL;

    vec3 kd = (vec3(1.0) - F) * (1.0 - metallic);
    diffuse = kd * albedo / PI * radiance * NdL;
}

void main() {
    vec4 base = material.baseColor;
    const int baseTex = texIndex(material.tex0.x);
    if (baseTex >= 0) base *= texture(uTextures[baseTex], vUV);

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

    vec3 albedo = base.rgb;
    vec3 V = normalize(ubo.cameraPos.xyz - vWorldPos);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    roughness = max(roughness, 0.04);

    vec3 lightDiffuse = vec3(0.0);
    vec3 lightSpecular = vec3(0.0);
    vec3 ld, ls;
    shadePointLight(N, V, vWorldPos, albedo, metallic, roughness, F0, ubo.light0Pos.xyz,
                    ubo.light0Color.rgb * ubo.light0Color.w, ld, ls);
    lightDiffuse += ld; lightSpecular += ls;
    shadePointLight(N, V, vWorldPos, albedo, metallic, roughness, F0, ubo.light1Pos.xyz,
                    ubo.light1Color.rgb * ubo.light1Color.w, ld, ls);
    lightDiffuse += ld; lightSpecular += ls;

    float NdV = max(dot(N, V), 0.0);
    vec3 F = fresnelSchlickRoughness(NdV, F0, roughness);
    vec3 kd = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuseIbl = texture(iblIrradiance, N).rgb * albedo;
    vec3 R = reflect(-V, N);
    vec3 prefiltered = textureLod(iblPrefilter, R, roughness * lighting.iblParams.y).rgb;
    vec2 brdf = texture(iblBrdfLut, vec2(NdV, roughness)).rg;
    vec3 specularIbl = prefiltered * (F * brdf.x + brdf.y);

    float ssao = texelFetch(ssaoTex, ivec2(gl_FragCoord.xy), 0).r;
    float ambientScale = ao * ssao * lighting.iblParams.x;

    // Energy-conserving dielectric glass model (same as transparent.frag —
    // keep in sync):
    //  - Specular lobes (direct + IBL) are surface reflections: their energy
    //    is independent of transmission, so they are NOT scaled by opacity.
    //    Scaling them by base.a (~0.1-0.4 for glass) would crush point-light
    //    highlights and leave only the IBL mirror, reading as metal.
    //  - specularIbl is weighted by the un-roughened Fresnel Fg so head-on
    //    reflections stay near F0 (~4%) and only grow toward grazing angles;
    //    an unweighted full-strength sample of the sharpest prefilter mip
    //    (roughness ~0.04) would mirror the sky's sun disk almost losslessly.
    //  - The diffuse/emissive terms approximate the transmitted, tinted part
    //    (glass albedo is a tint, not a milky diffuse lobe) and keep the
    //    opacity scale.
    //  - Opacity is max(base.a, Fg): Fresnel reflection still turns panes
    //    more opaque at grazing angles, but moderate angles no longer
    //    saturate to a fully opaque mirror.
    vec3 Fg = fresnelSchlick(NdV, F0);
    float alpha = clamp(max(base.a, Fg.r), 0.0, 1.0);
    vec3 glassColor = (lightDiffuse + emissive + kd * diffuseIbl * ambientScale) * base.a +
                      lightSpecular + specularIbl * Fg * ambientScale;

    outColor = vec4(glassColor, alpha);
}
