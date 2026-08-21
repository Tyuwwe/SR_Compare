#version 450
#extension GL_GOOGLE_include_directive : require
#include "brdf.glsl"
// Deferred lighting pass: reconstructs world position from depth, shades with
// Heitz height-correlated GGX + Hammon 2017 diffuse (no Lambert) for punctual
// lights, plus IBL split-sum and emissive.  Far-plane pixels are the skybox.
// Output is HDR linear color (R16G16B16A16F); present.frag tone-maps.

// GPU mirror of scene::Light; must match LightGPU in DeferredCore.h.
struct LightGPU {
    vec4 posOrDir; // xyz = position (point) / direction-to-light (directional), w = type (0 = dir, 1 = point)
    vec4 color;    // rgb + w = intensity (PI-scaled on the CPU)
    vec4 params;   // x = range (0 = infinite), y = castShadow (reserved, C2), zw = reserved
};

layout(set = 0, binding = 0) uniform LightingUBO {
    mat4 invViewProj;   // matches the GBuffer pass (jittered for the low-res path)
    vec4 cameraPos;
    LightGPU lights[16]; // must match kMaxLights in DeferredCore.h
    vec4 lightCounts;   // x = active light count, yzw reserved
    vec4 ambient;       // unused when IBL is active (kept for reference)
    vec4 iblParams;     // x = env intensity, y = prefilter max lod, z = skybox enabled, w = unused
    mat4 cascadeVp[4];  // CSM light view-projections (one per cascade layer)
    vec4 cascadeSplits; // view-space depth (positive metres) of each cascade's far plane
    vec4 shadowParams;  // x = rasterizer constant bias, y = slope bias (reference),
                        // z = shadows enabled, w = debug cascade tint
    vec4 viewForward;   // xyz = camera forward (world); w = shadowed sun light index (-1 = none)
} u;

layout(set = 0, binding = 1) uniform sampler2D gbAlbedo;
layout(set = 0, binding = 2) uniform sampler2D gbNormal;
layout(set = 0, binding = 3) uniform sampler2D gbMaterial;
layout(set = 0, binding = 4) uniform sampler2D gbEmissive;
layout(set = 0, binding = 5) uniform sampler2D gbDepth;
layout(set = 0, binding = 6) uniform samplerCube iblIrradiance;
layout(set = 0, binding = 7) uniform samplerCube iblPrefilter;
layout(set = 0, binding = 8) uniform sampler2D iblBrdfLut;
layout(set = 0, binding = 9) uniform samplerCube envCube; // base environment (skybox)
layout(set = 0, binding = 10) uniform sampler2D ssaoTex;  // GTAO (ssao.comp + bilateral denoise)
layout(set = 0, binding = 11) uniform sampler2DArrayShadow shadowMap; // CSM, comparison sampler

layout(location = 0) out vec4 outColor;

// Shades one punctual light.  Directional: no falloff, L = posOrDir (unit
// direction towards the light).  Point: inverse-square falloff with a
// Frostbite-style window (saturate(1 - (d/range)^4)^2) that reaches zero
// exactly at range; range = 0 keeps the legacy pure inverse-square falloff.
vec3 shadeLight(vec3 N, vec3 V, vec3 worldPos, vec3 albedo, float metallic,
                float roughness, vec3 F0, LightGPU light) {
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

    float NdL = max(dot(N, L), 0.0);
    if (NdL <= 0.0) return vec3(0.0);

    vec3 diffuse, specular;
    evalBrdf(N, V, L, albedo, metallic, roughness, F0, diffuse, specular);
    return (diffuse + specular) * radiance * NdL;
}

// --- CSM sun shadow sampling (keep in sync with transparent*.frag) ----------
const float kShadowMapTexel = 1.0 / 2048.0;
// Receiver-side epsilon on top of the rasterizer depth bias (caster side,
// vkCmdSetDepthBias): guards against depth-precision acne on surfaces nearly
// perpendicular to the sun.
const float kShadowDepthEpsilon = 0.0002;

// First cascade whose far boundary covers the view-space depth; fragments
// beyond the last split clamp to cascade 3 (its map is border-white outside).
int selectCascade(float viewDepth) {
    for (int i = 0; i < 4; ++i)
        if (viewDepth <= u.cascadeSplits[i]) return i;
    return 3;
}

// 4x4 PCF via the comparison sampler (each texture() call is a hardware
// filtered depth compare).  xy * 0.5 + 0.5 maps the y-flipped NDC to UV.
float sampleCascade(int c, vec3 worldPos) {
    vec4 sc = u.cascadeVp[c] * vec4(worldPos, 1.0);
    vec3 p = sc.xyz / sc.w;
    vec2 uv = p.xy * 0.5 + 0.5;
    float ref = p.z - kShadowDepthEpsilon;
    float sum = 0.0;
    for (int y = -1; y <= 2; ++y)
        for (int x = -1; x <= 2; ++x)
            sum += texture(shadowMap, vec4(uv + (vec2(x, y) - 0.5) * kShadowMapTexel, float(c), ref));
    return sum / 16.0;
}

// Shadow factor for the CSM sun; blends into the next cascade over the outer
// 10% of the current one to hide the resolution transition.
float sunShadow(vec3 worldPos, float viewDepth, out int cascade) {
    cascade = selectCascade(viewDepth);
    float s = sampleCascade(cascade, worldPos);
    if (cascade < 3) {
        const float split = u.cascadeSplits[cascade];
        const float blend = clamp((viewDepth - split * 0.9) / (split * 0.1), 0.0, 1.0);
        if (blend > 0.0)
            s = mix(s, sampleCascade(cascade + 1, worldPos), blend);
    }
    return s;
}

void main() {
    ivec2 pix = ivec2(gl_FragCoord.xy);
    float depth = texelFetch(gbDepth, pix, 0).r;

    // World position from the inverted view-projection (NDC: y down, z [0,1]).
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(gbDepth, 0));
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 wp = u.invViewProj * clip;
    wp /= wp.w;
    vec3 viewDir = normalize(wp.xyz - u.cameraPos.xyz);

    if (depth == 1.0) {
        // Skybox: sample the base environment map along the view ray.
        vec3 sky = u.iblParams.z > 0.5
                       ? textureLod(envCube, viewDir, 0.0).rgb * u.iblParams.x
                       : u.ambient.rgb * u.ambient.w;
        outColor = vec4(sky, 1.0);
        return;
    }

    vec4 albedoA = texelFetch(gbAlbedo, pix, 0);
    vec3 albedo = albedoA.rgb;
    // A2B10G10R10 stores the normal remapped to [0,1]; unpack back to [-1,1].
    vec3 N = normalize(texelFetch(gbNormal, pix, 0).xyz * 2.0 - 1.0);
    vec4 matParams = texelFetch(gbMaterial, pix, 0);
    float metallic = matParams.r;
    float roughness = max(matParams.g, 0.04); // floor: avoids GGX singularity
    float ao = matParams.b;
    // Screen-space AO (R16F, same resolution as this GBuffer path) multiplies
    // the IBL environment term together with the material AO texture.
    float ssao = texelFetch(ssaoTex, pix, 0).r;
    vec3 emissive = texelFetch(gbEmissive, pix, 0).rgb;

    vec3 V = -viewDir;
    // Two-sided foliage (cull is none): backfaces store inverted GBuffer
    // normals, which zero NdV and blow up Hammon / correlated-Smith.
    if (dot(N, V) < 0.0) N = -N;
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    vec3 color = vec3(0.0);
    // View-space depth of this fragment for cascade selection.
    const float viewDepth = dot(wp.xyz - u.cameraPos.xyz, u.viewForward.xyz);
    const int sunIndex = int(floor(u.viewForward.w + 0.5));
    int debugCascade = -1;
    int lightCount = int(u.lightCounts.x + 0.5);
    for (int i = 0; i < lightCount; ++i) {
        // Only the CSM sun (first shadow-casting directional light) is
        // shadow-mapped; every other light stays unshadowed.
        float shadow = 1.0;
        if (u.shadowParams.z > 0.5 && i == sunIndex) {
            int c;
            shadow = sunShadow(wp.xyz, viewDepth, c);
            debugCascade = c;
        }
        color += shadeLight(N, V, wp.xyz, albedo, metallic, roughness, F0, u.lights[i]) * shadow;
    }

    // IBL (split-sum): cosine irradiance * Hammon multi-scatter albedo +
    // prefiltered GGX * LUT (LUT integrates the correlated Smith V).
    float NdV = max(dot(N, V), 0.0);
    vec3 F = F_SchlickRoughness(NdV, F0, roughness);
    vec3 kd = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 irradiance = texture(iblIrradiance, N).rgb;
    vec3 diffuseIbl = irradiance * albedo * (1.0 + albedo * (0.1159 * roughness));

    vec3 R = reflect(-V, N);
    vec3 prefiltered = textureLod(iblPrefilter, R, roughness * u.iblParams.y).rgb;
    vec2 brdf = texture(iblBrdfLut, vec2(NdV, roughness)).rg;
    vec3 specularIbl = prefiltered * (F * brdf.x + brdf.y);

    color += (kd * diffuseIbl + specularIbl) * ao * ssao * u.iblParams.x;
    color += emissive;

    // Debug cascade tint (red/green/blue/yellow) over the lit result.
    if (u.shadowParams.w > 0.5 && debugCascade >= 0) {
        const vec3 tints[4] = {vec3(1.0, 0.15, 0.15), vec3(0.15, 1.0, 0.15),
                               vec3(0.25, 0.45, 1.0), vec3(1.0, 1.0, 0.2)};
        color = mix(color, tints[debugCascade], 0.25);
    }

    outColor = vec4(color, 1.0);
}
