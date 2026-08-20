#version 450
// Forward transparency pass (upscaler input path).  Shades alpha-blended
// surfaces (glass, bottles) with the same Cook-Torrance GGX + point lights +
// IBL split-sum as the deferred lighting pass, combined through an
// energy-conserving dielectric glass model (see main): specular lobes keep
// full strength while only the transmitted/tinted terms scale with opacity.
// Outputs:
//   RT0 color   RGBA16F  blended over the lit scene (srcAlpha, 1-srcAlpha)
//   RT1 motion  RG16F    overwrites the background MV (blend disabled);
//                        static glass -> camera motion, matching what UE calls
//                        "Output Velocity" for translucency
//   RT2 mask    R16F     translucent coverage accumulated additively
//                        (ONE, ONE); consumed by upscalers as the
//                        reactive / transparency-composition / bias mask
// Depth is tested (opaque GBuffer depth) but never written.

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 viewProj;
    mat4 viewProjNoJitter;
    mat4 prevViewProj;
    vec4 cameraPos;
    vec4 ambient;          // unused here; lights live in LightingUBO
    vec4 renderSizeJitter;
} ubo;

layout(set = 0, binding = 1) uniform MaterialUBO {
    vec4 baseColor; // rgb + alpha factor
    vec4 factors;   // x = metallic, y = roughness, z = occlusionStrength, w = alphaCutoff
    vec4 emissive;  // rgb factor
    vec4 tex0;      // texture indices: baseColor, normal, mr, ao (-1 = none)
    vec4 tex1;      // x = emissive texture index
} material;

layout(set = 1, binding = 0) uniform sampler2D uTextures[1024];

// IBL + light inputs; the UBO matches LightingUBO in DeferredCore.h.  Same
// LightGPU layout as lighting.frag.
struct LightGPU {
    vec4 posOrDir; // xyz = position (point) / direction-to-light (directional), w = type (0 = dir, 1 = point)
    vec4 color;    // rgb + w = intensity (PI-scaled on the CPU)
    vec4 params;   // x = range (0 = infinite), y = castShadow (reserved, C2), zw = reserved
};
layout(set = 2, binding = 0) uniform LightingUBO {
    mat4 invViewProj;
    vec4 cameraPos;
    LightGPU lights[8];
    vec4 lightCounts; // x = active light count, yzw reserved
    vec4 ambient;
    vec4 iblParams;   // x = env intensity, y = prefilter max lod
    mat4 cascadeVp[4];  // CSM light view-projections (one per cascade layer)
    vec4 cascadeSplits; // view-space depth (positive metres) of each cascade's far plane
    vec4 shadowParams;  // x = rasterizer constant bias, y = slope bias, z = shadows enabled,
                        // w = debug cascade tint
    vec4 viewForward;   // xyz = camera forward (world); w = shadowed sun light index (-1 = none)
} lighting;
layout(set = 2, binding = 1) uniform samplerCube iblIrradiance;
layout(set = 2, binding = 2) uniform samplerCube iblPrefilter;
layout(set = 2, binding = 3) uniform sampler2D iblBrdfLut;
// Screen-space AO (R16F, same resolution as this path's GBuffer).
layout(set = 2, binding = 4) uniform sampler2D ssaoTex;
layout(set = 2, binding = 5) uniform sampler2DArrayShadow shadowMap; // CSM, comparison sampler

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vUV;
layout(location = 3) in vec4 vTangent;
layout(location = 4) in vec2 vMotion;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outMotion;
layout(location = 2) out float outMask;

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

// Returns one punctual light's BRDF split into its diffuse and specular lobes.
// The glass model weights them differently: specular is surface reflection,
// whose energy does not depend on transmission (opacity), while the diffuse
// lobe approximates the transmitted/tinted term and scales with opacity.
// Directional lights have no falloff; point lights use the windowed
// inverse-square falloff of lighting.frag (range = 0 -> pure inverse-square).
void shadeLight(vec3 N, vec3 V, vec3 worldPos, vec3 albedo, float metallic,
                float roughness, vec3 F0, LightGPU light,
                out vec3 diffuse, out vec3 specular) {
    diffuse = vec3(0.0);
    specular = vec3(0.0);

    vec3 radiance = light.color.rgb * light.color.w;
    vec3 L;
    if (light.posOrDir.w < 0.5) {
        L = light.posOrDir.xyz;
    } else {
        vec3 toLight = light.posOrDir.xyz - worldPos;
        float dist = length(toLight);
        L = toLight / dist;
        float range = light.params.x;
        float atten = 1.0 / max(dist * dist, 1e-4);
        if (range > 0.0) {
            float w = clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0);
            atten *= w * w;
        }
        radiance *= atten;
    }

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

// --- CSM sun shadow sampling (keep in sync with lighting.frag) --------------
const float kShadowMapTexel = 1.0 / 2048.0;
const float kShadowDepthEpsilon = 0.0002; // receiver epsilon atop the rasterizer bias

int selectCascade(float viewDepth) {
    for (int i = 0; i < 4; ++i)
        if (viewDepth <= lighting.cascadeSplits[i]) return i;
    return 3;
}

float sampleCascade(int c, vec3 worldPos) {
    vec4 sc = lighting.cascadeVp[c] * vec4(worldPos, 1.0);
    vec3 p = sc.xyz / sc.w;
    vec2 uv = p.xy * 0.5 + 0.5;
    float ref = p.z - kShadowDepthEpsilon;
    float sum = 0.0;
    for (int y = -1; y <= 2; ++y)
        for (int x = -1; x <= 2; ++x)
            sum += texture(shadowMap, vec4(uv + (vec2(x, y) - 0.5) * kShadowMapTexel, float(c), ref));
    return sum / 16.0;
}

float sunShadow(vec3 worldPos, float viewDepth) {
    const int cascade = selectCascade(viewDepth);
    float s = sampleCascade(cascade, worldPos);
    if (cascade < 3) {
        const float split = lighting.cascadeSplits[cascade];
        const float blend = clamp((viewDepth - split * 0.9) / (split * 0.1), 0.0, 1.0);
        if (blend > 0.0)
            s = mix(s, sampleCascade(cascade + 1, worldPos), blend);
    }
    return s;
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
    // Glass direct light is occluded by the same CSM sun shadow as the opaque
    // pass (both diffuse and specular lobes).
    const float viewDepth = dot(vWorldPos - lighting.cameraPos.xyz, lighting.viewForward.xyz);
    const int sunIndex = int(floor(lighting.viewForward.w + 0.5));
    int lightCount = int(lighting.lightCounts.x + 0.5);
    for (int i = 0; i < lightCount; ++i) {
        vec3 ld, ls;
        shadeLight(N, V, vWorldPos, albedo, metallic, roughness, F0, lighting.lights[i], ld, ls);
        if (lighting.shadowParams.z > 0.5 && i == sunIndex) {
            const float shadow = sunShadow(vWorldPos, viewDepth);
            ld *= shadow;
            ls *= shadow;
        }
        lightDiffuse += ld; lightSpecular += ls;
    }

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

    // Energy-conserving dielectric glass model:
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
    outMotion = vMotion;
    outMask = alpha; // accumulated additively across layers
}
