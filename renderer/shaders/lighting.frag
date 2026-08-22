#version 450
#extension GL_GOOGLE_include_directive : require
#include "brdf.glsl"
#include "ssr.glsl"
#define SR_PROBE_UBO_BINDING 15
#define SR_PROBE_SPEC_BINDING 16
#define SR_PROBE_DIFF_BINDING 17
#include "probes.glsl"
// Deferred lighting pass: reconstructs world position from depth, shades with
// Heitz height-correlated GGX + Hammon 2017 diffuse (no Lambert) for punctual
// lights, plus IBL split-sum and emissive.  Far-plane pixels are the skybox.
// Output is HDR linear color (R16G16B16A16F); present.frag tone-maps.
//
// Point/spot lights are evaluated via clustered shading (Olsson et al., HPG
// 2012; DOOM 2016/Eternal SIGGRAPH): cluster_assign.comp binning pass writes
// per-cluster light index lists; this shader iterates only its own cluster's
// list.  Directional lights (the CSM sun) bypass the clusters and are shaded
// straight from the UBO's legacy 16-slot array, which also still feeds the
// forward transparency pass.
//
// Shadows (Phase 4b): the sun uses 4 CSM cascades with a temporally dithered
// cascade transition; shadow-casting spot lights selected into the shadow
// atlas this frame (LightGPU.params.z = tile) sample their 1024^2 atlas tile
// with the same 16-tap PCF.  Point lights are unshadowed this phase.
// Phase 4c: the sun additionally gets a screen-space contact shadow
// (contactShadow() below), and the IBL specular term carries the Fdez-Agüera
// 2019 multi-scatter compensation (specularIblMultiScatter, brdf.glsl).

// GPU mirror of scene::Light; must match LightGPU in DeferredCore.h.
struct LightGPU {
    vec4 posOrDir; // xyz = position (point/spot) / direction-to-light (directional), w = type (0 = dir, 1 = point, 2 = spot)
    vec4 color;    // rgb + w = intensity (PI-scaled on the CPU)
    vec4 params;   // x = range (0 = infinite), y = castShadow,
                   // z = shadowIndex (spot atlas tile, -1 = unshadowed), w = spot cos(inner)
    vec4 spotDir;  // xyz = spot cone direction (unit, world), w = spot cos(outer)
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
    vec4 clusterDepth;  // x = near, y = far (exponential cluster slicing), zw unused
    mat4 shadowTileVp[16]; // spot shadow atlas per-tile light VPs (must match
                           // kShadowAtlasTiles in DeferredCore.h); only the
                           // first shadowAtlasParams.x entries are valid
    vec4 shadowAtlasParams; // x = tiles rendered this frame, y = 1/atlasSize
                            // (PCF texel step), z = frame index (CSM cascade
                            // dither), w = contact shadows enabled
    mat4 viewProj;          // forward view-projection of this path (jittered for
                            // the low-res path); the contact-shadow march
                            // reprojects its world-space samples with it
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
// Clustered shading inputs (cluster_assign.comp output + the full lights SSBO).
layout(std430, set = 0, binding = 12) readonly buffer ClusterLightsSSBO {
    uvec4 header;    // x = packed point/spot light count
    LightGPU lights[];
} clusterLights;
layout(std430, set = 0, binding = 13) readonly buffer ClusterGridSSBO {
    uvec4 gridHeader; // gridX, gridY, gridZ, maxPerCluster
    uint data[];      // counts[N] then indices[N * maxPerCluster]
} clusterGrid;
// Spot shadow atlas (Phase 4b): 4096^2 D32, 16 row-major 1024^2 tiles,
// comparison sampler shared with the CSM map.
layout(set = 0, binding = 14) uniform sampler2DShadow shadowAtlas;

// Screen-tile edge of the cluster grid; must match kClusterTileSize in
// DeferredCore.h.
const uint kClusterTileSize = 64;

layout(location = 0) out vec4 outColor;

// Shades one punctual light.  Directional: no falloff, L = posOrDir (unit
// direction towards the light).  Point/spot: inverse-square falloff with a
// Frostbite-style window (saturate(1 - (d/range)^4)^2) that reaches zero
// exactly at range; range = 0 keeps the legacy pure inverse-square falloff.
// Spots additionally fade by the cone: a smoothed (squared) ramp from
// cos(inner) down to cos(outer), per KHR_lights_punctual's umbra/penumbra.
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
        if (light.posOrDir.w > 1.5) {
            // Spot cone: cos of the angle between the spot axis and the
            // direction from the light to the fragment.
            float cosTheta = dot(-L, light.spotDir.xyz);
            float cone = clamp((cosTheta - light.spotDir.w) /
                                   max(light.params.w - light.spotDir.w, 1e-5),
                               0.0, 1.0);
            atten *= cone * cone;
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

// Temporal dither for the CSM cascade transition: interleaved gradient noise
// (Jimenez 2014) offset by the frame index, so the cascade chosen in the 10%
// blend zone varies per pixel AND per frame (deterministic — same frame index
// always reproduces the same pattern).  Dithering picks ONE cascade instead of
// PCF-sampling and blending both maps: cheaper, and the seam reads as
// blue-ish noise that TAA/temporal passes absorb rather than a resolution ramp.
float cascadeDither(ivec2 pix) {
    const float f = mod(u.shadowAtlasParams.z, 64.0);
    const vec2 p = vec2(pix) + vec2(5.588238) * f;
    return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}

// Shadow factor for the CSM sun.  Within the outer 10% of a cascade the
// per-pixel dithered threshold switches to the next cascade, hiding the
// resolution transition.
float sunShadow(vec3 worldPos, float viewDepth, ivec2 pix, out int cascade) {
    cascade = selectCascade(viewDepth);
    if (cascade < 3) {
        const float split = u.cascadeSplits[cascade];
        const float blend = clamp((viewDepth - split * 0.9) / (split * 0.1), 0.0, 1.0);
        if (blend > 0.0 && cascadeDither(pix) < blend) ++cascade;
    }
    return sampleCascade(cascade, worldPos);
}

// --- Spot shadow atlas sampling (Phase 4b) ------------------------------------
// Tile UV rect is derived from the tile index (row-major 4x4 grid).  PCF taps
// must never cross a tile boundary (the atlas is one texture and the shared
// border-white sampler only guards the outer edge), so the projected UV is
// clamped with a 2-texel inset — that covers the 16-tap kernel footprint plus
// the comparison sampler's bilinear footprint.  PCSS-style variable penumbra
// was considered and skipped: a distance-widened kernel injects noise that
// fights TAA and the GT/upscaler pixel parity this tool exists to measure;
// the fixed 16-tap kernel matches the CSM quality level.
float spotShadow(int tile, vec3 worldPos) {
    vec4 sc = u.shadowTileVp[tile] * vec4(worldPos, 1.0);
    vec3 p = sc.xyz / sc.w;
    // Behind the light or past the far plane: unshadowed (outside the cone
    // frustum the cone falloff in shadeLight zeroes the contribution anyway).
    if (p.z <= 0.0 || p.z >= 1.0) return 1.0;
    vec2 ndc = p.xy * 0.5 + 0.5; // y-flipped NDC -> tile-local UV (same as CSM)
    if (any(lessThan(ndc, vec2(0.0))) || any(greaterThan(ndc, vec2(1.0)))) return 1.0;
    const float inset = 2.0 / 1024.0; // PCF kernel + bilinear footprint, tile UV
    ndc = clamp(ndc, vec2(inset), vec2(1.0 - inset));
    const vec2 tileMin = vec2(float(tile % 4), float(tile / 4)) * 0.25;
    const vec2 uv = tileMin + ndc * 0.25;
    const float ref = p.z - kShadowDepthEpsilon;
    const float texel = u.shadowAtlasParams.y; // 1/atlasSize
    float sum = 0.0;
    for (int y = -1; y <= 2; ++y)
        for (int x = -1; x <= 2; ++x)
            sum += texture(shadowAtlas, vec3(uv + (vec2(x, y) - 0.5) * texel, ref));
    return sum / 16.0;
}

// --- Screen-space contact shadows (UE4.19+, Phase 4c), sun only ---------------
// Short march from the surface towards the sun against the opaque depth
// buffer (binding 5).  Covers the two gaps the 4-cascade CSM leaves:
// contact-scale detail its 2048^2 texels blur away, and everything past the
// last split (~200 m, where cascade 3's border-white leaves the sun fully
// lit).  Sun only, like the 2020-era engines the feature comes from: a
// per-light screen march does not pay off, and shadowed local lights already
// have the spot atlas.
//
// Tradeoff vs. the SSR marcher (ssr.glsl): rays are ~2 m and the step count
// is small, so the Hi-Z cell-skip machinery costs more setup than it saves;
// fixed world-space steps with a per-sample view-Z compare stay deterministic
// (no frame jitter) and cheap.  The occlusion test mirrors SSR's acceptance
// model: a step counts when the ray point sits behind the visible surface
// inside a distance-scaled thickness window; the darkening is strongest at
// the contact and fades to fully lit at the ray end so the length cutoff
// does not pop.
const float kContactMaxLen = 2.0; // world metres (UE ContactShadowLength)
const float kContactBias = 0.03;  // origin offset along N, self-hit guard
const int kContactSteps = 12;
// Upper half of the acceptance window (view Z, metres).  The depth buffer
// only stores the front-most surface, so "ray point behind it" is ambiguous:
// the point may sit inside the occluder volume (a real shadow) or metres
// behind a foreground object that merely crosses the ray's screen path.  The
// latter used to score full-strength hits and projected object-shaped ghost
// shadows onto anything behind them, swimming with the camera.  Capping the
// depth gap limits hits to volumes the ray actually enters; CSM covers the
// large occluders this rejects.
const float kContactThickness = 0.5; // +2% of ray distance, below

float contactShadow(vec3 worldPos, vec3 N, vec3 L) {
    const vec2 renderSize = vec2(textureSize(gbDepth, 0));
    const vec3 origin = worldPos + N * kContactBias;
    float shadow = 1.0;
    for (int i = 1; i <= kContactSteps; ++i) {
        const float t = float(i) / float(kContactSteps);
        const vec4 clip = u.viewProj * vec4(origin + L * (kContactMaxLen * t), 1.0);
        if (clip.w < 0.02) break; // crossed the near plane; later steps too
        const vec2 suv = clip.xy / clip.w * 0.5 + 0.5;
        if (any(lessThan(suv, vec2(0.0))) || any(greaterThan(suv, vec2(1.0)))) continue;
        const ivec2 spix = clamp(ivec2(suv * renderSize), ivec2(0), ivec2(renderSize) - 1);
        const float bufDepth = texelFetch(gbDepth, spix, 0).r;
        if (bufDepth >= 0.9999) continue; // sky: no occluder
        // View-Z of the visible surface at that pixel (the SSR marcher's
        // helper; both matrices come from this path's lighting UBO).
        const vec2 cuv = (vec2(spix) + 0.5) / renderSize;
        const float bufZ = ssrViewZ(u.viewProj, u.invViewProj, cuv, bufDepth);
        const float rayZ = abs(clip.w);
        const float over = rayZ - bufZ; // >0: ray point behind the visible surface
        if (over <= 0.02 + 0.01 * bufZ || over >= kContactThickness + 0.02 * rayZ)
            continue;
        // Attenuation grows with hit distance along the ray: strongest at the
        // contact, fully lit at the length cutoff, so the cutoff does not pop.
        // (The phase-4c-1 code had this inverted — 1 - smoothstep(0.75,1,t) —
        // which zeroed every hit closer than 1.5 m, i.e. all real contacts,
        // and gave full weight to far hits, i.e. the screen-space ghosts.)
        // The 5%-of-screen edge fade softens the clip when the march crosses
        // the screen border (off-screen samples count as unoccluded).
        const float edge = smoothstep(0.0, 0.05, suv.x) * smoothstep(0.0, 0.05, suv.y) *
                           smoothstep(0.0, 0.05, 1.0 - suv.x) *
                           smoothstep(0.0, 0.05, 1.0 - suv.y);
        shadow = min(shadow, mix(1.0, smoothstep(0.0, 1.0, t), edge));
    }
    return shadow;
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
    // View-space depth of this fragment for cascade selection and cluster slicing.
    const float viewDepth = dot(wp.xyz - u.cameraPos.xyz, u.viewForward.xyz);
    const int sunIndex = int(floor(u.viewForward.w + 0.5));
    int debugCascade = -1;
    int lightCount = int(u.lightCounts.x + 0.5);
    // Directional lights only: the sun (shadow-mapped via CSM) plus any
    // leftover directionals.  Point/spot lights come from the cluster list
    // below; the UBO's point/spot entries exist for the transparency pass.
    for (int i = 0; i < lightCount; ++i) {
        if (u.lights[i].posOrDir.w >= 0.5) continue;
        // Only the CSM sun (first shadow-casting directional light) is
        // shadow-mapped; every other light stays unshadowed.
        float shadow = 1.0;
        if (u.shadowParams.z > 0.5 && i == sunIndex) {
            int c;
            shadow = sunShadow(wp.xyz, viewDepth, pix, c);
            debugCascade = c;
            // Screen-space contact shadow for the sun (UE4.19+): hardens the
            // contacts the CSM texels blur and fills in past the last split.
            // Key light only — a per-light march does not pay off.  Skipped
            // when the surface already faces away (shadeLight zeroes NdL).
            if (u.shadowAtlasParams.w > 0.5 && dot(N, u.lights[i].posOrDir.xyz) > 0.0)
                shadow = min(shadow, contactShadow(wp.xyz, N, u.lights[i].posOrDir.xyz));
        }
        color += shadeLight(N, V, wp.xyz, albedo, metallic, roughness, F0, u.lights[i]) * shadow;
    }

    // Clustered point/spot lights: locate this pixel's cluster (screen tile +
    // exponential depth slice, mirroring cluster_assign.comp) and iterate its
    // light list.  The depth slice uses the fragment's view Z reconstructed
    // from the GBuffer, so thin geometry straddling a slice boundary still
    // picks a cluster that conservatively contains the light (the assignment
    // AABB test is conservative by construction).
    {
        const uvec3 gridDim = clusterGrid.gridHeader.xyz;
        const uint maxPerCluster = clusterGrid.gridHeader.w;
        const uint clusterCount = gridDim.x * gridDim.y * gridDim.z;
        const uvec2 tile = min(uvec2(pix) / kClusterTileSize, gridDim.xy - 1);
        const float nearZ = u.clusterDepth.x;
        const float sliceF = log(max(viewDepth, nearZ) / nearZ) *
                             float(gridDim.z) / log(u.clusterDepth.y / nearZ);
        const uint slice = min(uint(max(sliceF, 0.0)), gridDim.z - 1);
        const uint clusterIdx = tile.x + tile.y * gridDim.x + slice * gridDim.x * gridDim.y;
        const uint count = clusterGrid.data[clusterIdx];
        const uint base = clusterCount + clusterIdx * maxPerCluster;
        for (uint j = 0; j < count; ++j) {
            const LightGPU l = clusterLights.lights[clusterGrid.data[base + j]];
            // Spot shadow atlas: shadowIndex (params.z) >= 0 selects the
            // light's atlas tile.  Points keep -1 this phase (no map).
            float shadow = 1.0;
            const int tile = int(floor(l.params.z + 0.5));
            if (u.shadowParams.z > 0.5 && tile >= 0 &&
                tile < int(floor(u.shadowAtlasParams.x + 0.5)))
                shadow = spotShadow(tile, wp.xyz);
            color += shadeLight(N, V, wp.xyz, albedo, metallic, roughness, F0, l) * shadow;
        }
    }

    // IBL (split-sum): cosine irradiance * Hammon multi-scatter albedo +
    // prefiltered GGX * LUT (LUT integrates the correlated Smith V).
    // Phase 4c-2: baked reflection probes replace the global environment where
    // a probe box contains the pixel (fallback chain: SSR -> probe -> global).
    float NdV = max(dot(N, V), 0.0);
    vec3 F = F_SchlickRoughness(NdV, F0, roughness);
    vec3 kd = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 irradiance = texture(iblIrradiance, N).rgb;

    vec3 R = reflect(-V, N);
    vec3 prefiltered = textureLod(iblPrefilter, R, roughness * u.iblParams.y).rgb;
    int pi0, pi1;
    float pw0, pw1;
    const float probeW = probeSelect(wp.xyz, pi0, pi1, pw0, pw1);
    if (probeW > 0.0) {
        irradiance = mix(irradiance, probeDiffuse(N, pi0, pi1, pw0, pw1), probeW);
        prefiltered = mix(prefiltered,
                          probeSpecular(wp.xyz, R, roughness, pi0, pi1, pw0, pw1), probeW);
    }
    vec3 diffuseIbl = irradiance * albedo * (1.0 + albedo * (0.1159 * roughness));

    vec2 brdf = texture(iblBrdfLut, vec2(NdV, roughness)).rg;
    vec3 specularIbl = prefiltered * specularIblMultiScatter(F, F0, brdf);

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
