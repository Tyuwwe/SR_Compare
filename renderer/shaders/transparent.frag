#version 450
#extension GL_GOOGLE_include_directive : require
#include "brdf.glsl"
#include "ssr.glsl"
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
    vec4 posOrDir; // xyz = position (point/spot) / direction-to-light (directional), w = type (0 = dir, 1 = point, 2 = spot)
    vec4 color;    // rgb + w = intensity (PI-scaled on the CPU)
    vec4 params;   // x = range (0 = infinite), y = castShadow,
                   // z = shadowIndex (spot atlas tile, -1 = unshadowed), w = spot cos(inner)
    vec4 spotDir;  // xyz = spot cone direction (unit, world), w = spot cos(outer)
};
layout(set = 2, binding = 0) uniform LightingUBO {
    mat4 invViewProj;
    vec4 cameraPos;
    LightGPU lights[16]; // must match kMaxLights in DeferredCore.h
    vec4 lightCounts; // x = active light count, yzw reserved
    vec4 ambient;
    vec4 iblParams;   // x = env intensity, y = prefilter max lod
    mat4 cascadeVp[4];  // CSM light view-projections (one per cascade layer)
    vec4 cascadeSplits; // view-space depth (positive metres) of each cascade's far plane
    vec4 shadowParams;  // x = rasterizer constant bias, y = slope bias, z = shadows enabled,
                        // w = debug cascade tint
    vec4 viewForward;   // xyz = camera forward (world); w = shadowed sun light index (-1 = none)
    vec4 clusterDepth;  // clustered shading slicing range (unused in this pass)
    // Spot shadow atlas state (Phase 4b), declared for layout parity with
    // lighting.frag; the forward pass does not sample spot shadows.
    mat4 shadowTileVp[16];
    vec4 shadowAtlasParams;
    // Forward view-projection for the lighting pass's contact-shadow march
    // (Phase 4c); layout parity, unused here.
    mat4 viewProj;
} lighting;
layout(set = 2, binding = 1) uniform samplerCube iblIrradiance;
layout(set = 2, binding = 2) uniform samplerCube iblPrefilter;
layout(set = 2, binding = 3) uniform sampler2D iblBrdfLut;
// Screen-space AO (R16F, same resolution as this path's GBuffer).
layout(set = 2, binding = 4) uniform sampler2D ssaoTex;
layout(set = 2, binding = 5) uniform sampler2DArrayShadow shadowMap; // CSM, comparison sampler
// Opaque HDR color mip chain (RGBA16F) + depth pyramid (Hi-Z, R32F), both
// captured before this pass writes color (SSR).
layout(set = 2, binding = 6) uniform sampler2D ssrColor;
layout(set = 2, binding = 7) uniform sampler2D ssrHiZ;

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vUV;
layout(location = 3) in vec4 vTangent;
layout(location = 4) in vec2 vMotion;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outMotion;
layout(location = 2) out float outMask;

int texIndex(float f) { return int(floor(f + 0.5)); }

// Returns one punctual light's BRDF split into its diffuse and specular lobes.
// The glass model weights them differently: specular is surface reflection,
// whose energy does not depend on transmission (opacity), while the diffuse
// lobe approximates the transmitted/tinted term and scales with opacity.
// Directional lights have no falloff; point lights use the windowed
// inverse-square falloff of lighting.frag (range = 0 -> pure inverse-square);
// spots add the same smoothed cone ramp (cos(inner) -> cos(outer)).
// Note: this forward pass still iterates the legacy 16-slot UBO array (scene
// order, sun first); the full light set is only in the clustered deferred
// pass.
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
        if (light.posOrDir.w > 1.5) {
            float cosTheta = dot(-L, light.spotDir.xyz);
            float cone = clamp((cosTheta - light.spotDir.w) /
                                   max(light.params.w - light.spotDir.w, 1e-5),
                               0.0, 1.0);
            atten *= cone * cone;
        }
        radiance *= atten;
    }

    float NdL = max(dot(N, L), 0.0);
    if (NdL <= 0.0) return;

    evalBrdf(N, V, L, albedo, metallic, roughness, F0, diffuse, specular);
    diffuse *= radiance * NdL;
    specular *= radiance * NdL;
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
    // Two-sided glass (cull is none): the far pane of a shop is drawn with
    // inverted geometric normals, which made NdV = 0 and lit as black.
    if (dot(N, V) < 0.0) N = -N;
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    roughness = max(roughness, 0.04);
    // Shop windows: Bistro ORM is too rough, and uncoated F0=0.04 lets the
    // dark interior punch through.  A mild coating + sharp roughness matches
    // the NVIDIA exterior reference better (more "mirror", less "hole").
    const bool shopGlass = base.a < 0.40 && metallic < 0.2;
    if (shopGlass) {
        roughness = min(roughness, 0.05);
        F0 = mix(vec3(0.04), vec3(0.22), 0.75);
    }

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
    vec3 F = F_SchlickRoughness(NdV, F0, roughness);
    vec3 kd = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuseIbl = texture(iblIrradiance, N).rgb * albedo * (1.0 + albedo * (0.1159 * roughness));
    vec3 R = reflect(-V, N);
    vec3 prefiltered = textureLod(iblPrefilter, R, roughness * lighting.iblParams.y).rgb;
    vec2 brdf = texture(iblBrdfLut, vec2(NdV, roughness)).rg;
    vec3 envBrdf = specularIblMultiScatter(F, F0, brdf); // IBL specular weight incl.
    // multi-scatter (same helper as lighting.frag / ssr_opaque.comp, applied at the
    // SSR composite below)
    vec3 specularIbl = prefiltered * envBrdf;

    float ssao = texelFetch(ssaoTex, ivec2(gl_FragCoord.xy), 0).r;
    float ambientScale = ao * ssao * lighting.iblParams.x;

    // Dielectric shop-window model:
    //  - Specular lobes (direct + IBL/SSR) are surface reflections and are
    //    NOT scaled by opacity.
    //  - SSR hits are the reflected scene; UE composites them with EnvBRDF
    //    (not a second Fresnel multiply).  Misses fall back to IBL.
    //  - Transmission scales with opacity and is attenuated by SSR confidence
    //    so dark furniture does not silhouette through a successful hit.
    //  - Opacity is max(base.a, Fg, ssrHit).
    vec3 Fg = F_Schlick(NdV, F0);

    vec3 specSsr = specularIbl * ambientScale;
    // SSR covers the full roughness range: traceSsr reads the hit colour from
    // the box-filtered colour mip chain at lod = roughness * (mipCount - 1),
    // so rough glass gets a blurred reflection instead of the old hard cutoff
    // that fell back to IBL above roughness 0.45.
    // Unlike the opaque path (Phase 2d: trace -> RT -> ssr_temporal.comp EMA),
    // this inline trace deliberately stays single-frame: the pass has no
    // opaque-only RT to accumulate into, glass already stabilises temporally
    // through the reactive/TC mask the upscalers get, and reprojection of a
    // transmissive surface would key on the wrong (front-most) depth layer.
    vec4 ssr = traceSsr(ssrColor, textureQueryLevels(ssrColor), ssrHiZ,
                        textureQueryLevels(ssrHiZ), ubo.viewProj,
                        lighting.invViewProj, ubo.cameraPos.xyz, vWorldPos, N, R, roughness,
                        ubo.renderSizeJitter.xy);
    float ssrHit = clamp(ssr.a, 0.0, 1.0);
    // UE applies EnvBRDF on the hit (not D*G*F).  Shop glass is a
    // coated mirror: lerp EnvBRDF toward 1 with hit confidence so a
    // solid trace is not crushed back to F0 (~0.22).
    specSsr = mix(specSsr, ssr.rgb * mix(envBrdf, vec3(1.0), ssrHit * 0.8), ssrHit);

    // Hide the dark interior: SSR confidence and Fresnel both raise opacity.
    // Head-on panes reflect behind the camera (not in the colour buffer), so
    // a miss still has to read as a dark mirror (IBL) rather than a hole.
    float alpha = clamp(max(base.a, max(Fg.r, ssrHit * 0.88)), 0.0, 1.0);
    if (shopGlass && ssrHit < 0.35)
        alpha = max(alpha, mix(0.84, 1.0, Fg.r));
    float transmit = base.a * (1.0 - max(ssrHit, shopGlass ? 0.75 : 0.0) * 0.92);

    // Premultiplied: transmissive terms scale with opacity, specular does not
    // (blend uses ONE, ONE_MINUS_SRC_ALPHA).
    vec3 glassColor = (lightDiffuse + emissive + kd * diffuseIbl * ambientScale) * transmit +
                      lightSpecular + specSsr;

    outColor = vec4(glassColor, alpha);
    outMotion = vMotion;
    outMask = alpha; // accumulated additively across layers
}
