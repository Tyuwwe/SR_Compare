#version 450
#extension GL_GOOGLE_include_directive : require
#include "brdf.glsl"
#include "ssr.glsl"
#include "volfog.glsl"
// GT-path transparency fragment shader: identical shading to transparent.frag
// but only the blended color output (the GT path has no motion/mask targets).
// Keep the bodies in sync.

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 viewProj;
    mat4 viewProjNoJitter;
    mat4 prevViewProj;
    vec4 cameraPos;
    vec4 ambient;          // unused here; lights live in LightingUBO
    vec4 renderSizeJitter;
    vec4 clipPlane;    // planar mirror plane (xyz normal toward the camera, w offset)
    mat4 reflViewProj; // mirrored-view view-projection (main view, reflParams.x = 1)
    vec4 reflParams;   // x = reflection image available, y = clip behind clipPlane,
                       // zw = mirrored proj m[10]/m[14] (viewZ = w / (depth + z))
} ubo;

layout(set = 0, binding = 1) uniform MaterialUBO {
    vec4 baseColor;
    vec4 factors;
    vec4 emissive;
    vec4 tex0;
    vec4 tex1;
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
    vec4 lightCounts; // x = active light count, y = glass inline-SSR enabled, zw reserved
    vec4 ambient;
    vec4 iblParams;
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
// Screen-space AO (R16F, same resolution as this path's GBuffer).  Bound for
// layout parity but deliberately NOT sampled (see transparent.frag).
layout(set = 2, binding = 4) uniform sampler2D ssaoTex;
layout(set = 2, binding = 5) uniform sampler2DArrayShadow shadowMap; // CSM, comparison sampler
layout(set = 2, binding = 6) uniform sampler2D ssrColor; // opaque HDR color mip chain (RGBA16F)
layout(set = 2, binding = 7) uniform sampler2D ssrHiZ; // opaque depth pyramid (Hi-Z, R32F)
// Ray-integrated froxel volume (same as transparent.frag — keep in sync).
layout(set = 2, binding = 8) uniform sampler3D fogVolume;
// Baked local reflection probes (same as transparent.frag — keep in sync):
// the glass inline-SSR miss fallback chains SSR hit -> local probe -> global.
#define SR_PROBE_SET 2
#define SR_PROBE_UBO_BINDING 9
#define SR_PROBE_SPEC_BINDING 10
#define SR_PROBE_DIFF_BINDING 11
#include "probes.glsl"
// Planar mirror reflection of the frame's active mirror plane
// (PlanarReflection.h): the mirrored view's HDR radiance + depth, sampled at
// this fragment's projection through SceneUBO.reflViewProj.  Stand-in 1x1
// images on paths without the pass (never sampled: reflParams.x = 0).
layout(set = SR_PROBE_SET, binding = 12) uniform sampler2D reflColor;
layout(set = SR_PROBE_SET, binding = 13) uniform sampler2D reflDepth;

// Fragment-stage push block.  Member offsets in SPIR-V are ABSOLUTE within
// the pipeline's push-constant memory, so params sits explicitly at byte 208,
// right after the vertex-stage SkinnedScenePush range
// (sizeof(SkinnedScenePush), see DeferredCore::createPipelines).
layout(push_constant) uniform TransparentFogPush {
    layout(offset = 208) vec4 params; // x = near, y = fog far, z = enabled (1/0), w unused
} fogPc;

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vUV;
layout(location = 3) in vec4 vTangent;
layout(location = 4) in vec4 vCurrentClip;  // unused in the GT path
layout(location = 5) in vec4 vPreviousClip; // unused

layout(location = 0) out vec4 outColor;

int texIndex(float f) { return int(floor(f + 0.5)); }

// Returns one punctual light's BRDF split into its diffuse and specular lobes.
// The glass model weights them differently: specular is surface reflection,
// whose energy does not depend on transmission (opacity), while the diffuse
// lobe approximates the transmitted/tinted term and scales with opacity.
// Directional lights have no falloff; point lights use the windowed
// inverse-square falloff of lighting.frag (range = 0 -> pure inverse-square);
// spots add the same smoothed cone ramp (cos(inner) -> cos(outer)).
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
    if (dot(N, V) < 0.0) N = -N;
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    roughness = max(roughness, 0.04);
    const bool shopGlass = base.a < 0.40 && metallic < 0.2;
    if (shopGlass) {
        roughness = min(roughness, 0.05);
        F0 = mix(vec3(0.04), vec3(0.22), 0.75);
    }
    // Mirror-mode panes (same as transparent.frag — keep in sync).
    const bool mirror = material.emissive.w > 0.5;
    if (mirror) {
        roughness = min(roughness, 0.04);
        // Plain (uncoated) glass: F0 = 0.04 with Schlick Fresnel.  The old
        // 0.30 coating (plus the EnvBRDF->1 boost below) reflected ~85% of
        // the scene radiance, so the panes read as a bright duplicate of the
        // terrace instead of a dark tinted mirror; the reference render
        // (path traced) shows the reflected street at ~6% of its direct
        // brightness at this incidence, exactly the dielectric Fresnel curve.
#ifndef SR_MIRROR_F0
#define SR_MIRROR_F0 0.04
#endif
        F0 = vec3(SR_MIRROR_F0);
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
    vec3 irradiance = texture(iblIrradiance, N).rgb;
    vec3 R = reflect(-V, N);
    vec3 prefiltered = textureLod(iblPrefilter, R, roughness * lighting.iblParams.y).rgb;
    // Phase 4c-2 (same as transparent.frag — keep in sync): baked reflection
    // probes replace the global environment where a probe box contains the
    // glass fragment; the SSR composite below mixes its hit over this
    // probe-aware specularIbl (chain: SSR hit -> probe -> global).
    int pi0, pi1;
    float pw0, pw1;
    const float probeW = probeSelect(vWorldPos, pi0, pi1, pw0, pw1);
    if (probeW > 0.0) {
        irradiance = mix(irradiance, probeDiffuse(N, pi0, pi1, pw0, pw1), probeW);
        prefiltered = mix(prefiltered,
                          probeSpecular(vWorldPos, R, roughness, pi0, pi1, pw0, pw1), probeW);
    }
    vec3 diffuseIbl = irradiance * albedo * (1.0 + albedo * (0.1159 * roughness));
    vec2 brdf = texture(iblBrdfLut, vec2(NdV, roughness)).rg;
    vec3 envBrdf = specularIblMultiScatter(F, F0, brdf); // same as transparent.frag
    vec3 specularIbl = prefiltered * envBrdf;

    // No SSAO on translucency (same as transparent.frag — keep in sync):
    // ssaoTex describes the opaque surface behind the pane, not the glass.
    float ambientScale = ao * lighting.iblParams.x;

    // Dielectric shop-window model (same as transparent.frag — keep in sync).
    vec3 Fg = F_Schlick(NdV, F0);

    vec3 specSsr = specularIbl * ambientScale;
    // Full roughness range (same as transparent.frag — keep in sync): the hit
    // colour comes from the colour mip chain at lod = roughness * (mips - 1).
    // Single-frame on purpose: only the opaque path accumulates SSR temporally
    // (Phase 2d) — see the note in transparent.frag.
    // Iteration budget: the near-field lock (cfg.x = 40 m) keeps the march at
    // mip 0, i.e. 1 px of screen segment per iteration, so the budget must
    // cover the longest clipped on-screen segment — the render-target
    // diagonal — plus descent/reject overhead (~1.5x).  A fixed 384 exhausted
    // mid-screen on long grazing rays; those false misses fell back to the
    // sky IBL and painted hard-edged white regions into the pane.
    // Thickness window (cfg.yz, same as transparent.frag — keep in sync):
    // deliberately tight in the near field.  Terrace furniture reflects at
    // <3 m, where the loose (0.18 + 0.02·z) window accepted rays landing
    // 6-18 cm behind silhouette texels — the march had tunneled past the
    // camera-occluded furniture flank (no depth for it) and the accepted
    // texel sampled the SUNLIT pavement behind/under the tables, painting
    // hard-edged bright patches into the reflected tabletop.  Clean bisected
    // crossings overshoot by <2 cm, so 0.05 + 0.005·z keeps them (and every
    // far-field case in the regression suite) while rejecting the tunneling
    // landings.
    // The inline glass trace follows the same global SSR toggle as the opaque
    // pass (lightCounts.y, same as transparent.frag — keep in sync): off ->
    // probe -> global env fallback only, ssrHit stays 0.
    float ssrHit = 0.0;
    float ssrHitDist = 0.0; // unused: the GT path has no motion target
    bool planar = false;
    // Planar mirror reflection: a mirror fragment lying on the frame's active
    // plane samples the mirrored view at its own projection — exact for a
    // planar reflector (back faces, off-screen content), no depth guesswork.
    // The reflected depth gives the virtual image point for the motion
    // vectors below.  Panes on other planes keep the SSR fallback.
    if (mirror && ubo.reflParams.x > 0.5) {
        const vec3 pn = ubo.clipPlane.xyz;
        const float pd = dot(pn, vWorldPos) + ubo.clipPlane.w;
        if (abs(pd) < 0.05 && dot(N, pn) > 0.95) {
            const vec4 rc = ubo.reflViewProj * vec4(vWorldPos, 1.0);
            if (rc.w > 0.0) {
                const vec2 ruv = rc.xy / rc.w * 0.5 + 0.5;
                if (all(greaterThanEqual(ruv, vec2(0.0))) && all(lessThanEqual(ruv, vec2(1.0)))) {
                    planar = true;
                    const vec3 rcol = textureLod(reflColor, ruv, 0.0).rgb;
                    specSsr = rcol * envBrdf; // EnvBRDF-weighted, like an SSR hit
                    ssrHit = 1.0;
                    // Reflected point: view Z from the mirrored camera along
                    // the ray through this fragment, minus the leg to the pane.
                    const float rd = textureLod(reflDepth, ruv, 0.0).r;
                    const float rz = ubo.reflParams.w / (rd + ubo.reflParams.z);
                    const vec3 camM = ubo.cameraPos.xyz -
                                      pn * (2.0 * (dot(pn, ubo.cameraPos.xyz) + ubo.clipPlane.w));
                    const vec3 fwdM = normalize(lighting.viewForward.xyz -
                                                pn * (2.0 * dot(pn, lighting.viewForward.xyz)));
                    const vec3 toP = vWorldPos - camM;
                    const float distR = rz / max(dot(normalize(toP), fwdM), 1e-3);
                    ssrHitDist = max(distR - length(toP), 0.0);
                }
            }
        }
    }
    if (!planar && lighting.lightCounts.y > 0.5) {
#ifndef SR_SSR_EXTENT_K
#define SR_SSR_EXTENT_K 1.0
#endif
        vec4 ssr = traceSsrEx(ssrColor, textureQueryLevels(ssrColor), ssrHiZ,
                              textureQueryLevels(ssrHiZ), ubo.viewProj,
                              lighting.invViewProj, ubo.cameraPos.xyz, vWorldPos, N, R, roughness,
                              ubo.renderSizeJitter.xy, 1, 1, vec4(40.0, 0.05, 0.005, 0.0),
                              int(length(ubo.renderSizeJitter.xy) * 1.5) + 8, SR_SSR_EXTENT_K,
                              ssrHitDist);
        ssrHit = clamp(ssr.a, 0.0, 1.0);
        // Mirror panes keep the physically based EnvBRDF weight (hit radiance
        // times the Fresnel/geometry term, UE composite); the boost stays for
        // the legacy transmissive shop-glass look only.
        const vec3 hitWeight = mirror ? envBrdf : mix(envBrdf, vec3(1.0), ssrHit * 0.8);
        specSsr = mix(specSsr, ssr.rgb * hitWeight, ssrHit);
    }

    float alpha = clamp(max(base.a, max(Fg.r, ssrHit * 0.88)), 0.0, 1.0);
    if (shopGlass && ssrHit < 0.35)
        alpha = max(alpha, mix(0.84, 1.0, Fg.r));
    float transmit = base.a * (1.0 - max(ssrHit, shopGlass ? 0.75 : 0.0) * 0.92);
    if (mirror) {
        // Pure mirror (same as transparent.frag — keep in sync).
        alpha = 1.0;
        transmit = 0.0;
    }

    vec3 glassColor = (lightDiffuse + emissive + kd * diffuseIbl * ambientScale) * transmit +
                      lightSpecular + specSsr;

    // Volumetric fog on translucency (same as transparent.frag — keep in
    // sync): fog the glass contribution with the ray-integrated volume at the
    // fragment's view depth; the background was fogged by volfog_composite.
    if (fogPc.params.z > 0.5 && viewDepth > fogPc.params.x) {
        const vec2 fuv = gl_FragCoord.xy / ubo.renderSizeJitter.xy;
        const vec4 fog = texture(fogVolume, vec3(fuv, froxelW(viewDepth, fogPc.params.x,
                                                              fogPc.params.y)));
        glassColor = glassColor * fog.a + fog.rgb;
    }

    outColor = vec4(glassColor, alpha);
}
