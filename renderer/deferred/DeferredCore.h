#pragma once
// ============================================================================
// DeferredCore — the shared deferred GBuffer + PBR/IBL lighting pipeline used
// by the viewer (Renderer), compare mode (CompareApp) and the GUI (GuiApp).
//
// Owns everything that is scene- and resolution-independent: shader modules,
// descriptor set layouts, pipeline layouts, the GBuffer/GT/lighting pipelines,
// the texture/GBuffer samplers and the IBL maps (built once per instance).
// Hosts keep their own render targets, descriptor pools, per-frame UBOs and
// descriptor sets; this header also provides the std140 UBO layouts, the UBO
// fillers (shared light/ambient defaults) and the pass-recording helpers so
// the three front ends stay pixel-identical.
// ============================================================================
#include "renderer/core/Vk.h"
#include "renderer/core/VulkanContext.h"
#include "renderer/ibl/Ibl.h"
#include "renderer/ibl/Probes.h"
#include "renderer/ibl/SkyAtmosphere.h"
#include "renderer/math/Math.h"

#include <cstdint>
#include <vector>

namespace sr {

class Scene;
class Camera;
struct Light;
struct VolFogParams;

// Bundled equirect HDR environment map (Bistro san_giuseppe_bridge).  An
// explicit static-env choice for --env-map / preset envFile; the default is
// the procedural sky atmosphere (ibl/SkyAtmosphere.h).
inline const char* kDefaultEnvMapPath = "assets/env/san_giuseppe_bridge_4k.hdr";

// Attachment formats of the deferred pipeline (identical everywhere).
namespace deferred {
constexpr VkFormat kHdrColorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat kMotionFormat = VK_FORMAT_R16G16_SFLOAT;
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr VkFormat kAlbedoFormat = VK_FORMAT_R8G8B8A8_SRGB;   // rgb sRGB-encoded, a = alpha
constexpr VkFormat kNormalFormat = VK_FORMAT_A2B10G10R10_UNORM_PACK32; // xyz = world normal * 0.5 + 0.5 (unsigned: remap on read)
constexpr VkFormat kMaterialFormat = VK_FORMAT_R8G8B8A8_UNORM;    // r/g/b = metallic/rough/AO
constexpr VkFormat kEmissiveFormat = VK_FORMAT_B10G11R11_UFLOAT_PACK32; // rgb = emissive (no alpha; max ~65004 covers emissive strength)
constexpr VkFormat kReactiveFormat = VK_FORMAT_R16_SFLOAT; // translucent coverage mask
constexpr uint32_t kMaxTextures = 1024;
} // namespace deferred

// std140, matches the SceneUBO block in gbuffer.vert / transparent.vert.
// Light data lives in LightingUBO (shared by the deferred and the forward
// transparency passes); this block carries only camera/transform state.
struct SceneUBO {
    float viewProj[16];
    float viewProjNoJitter[16];
    float prevViewProj[16];
    float cameraPos[4];
    float ambient[4];
    float renderSizeJitter[4];
};
static_assert(sizeof(SceneUBO) == 240, "SceneUBO std140 size mismatch");

// std140, matches the MaterialUBO block in gbuffer.frag / gbuffer_gt.frag.
struct MaterialUBO {
    float baseColor[4];
    float factors[4];  // metallic, roughness, occlusionStrength, alphaCutoff
    float emissive[4]; // rgb factor
    float tex0[4];     // baseColor, normal, mr, ao texture indices
    float tex1[4];     // emissive texture index
};
static_assert(sizeof(MaterialUBO) == 80, "MaterialUBO std140 size mismatch");

// GPU mirror of scene::Light (std430/std140: four vec4s per light).  Must
// match the LightGPU struct in lighting.frag / transparent.frag /
// transparent_gt.frag / cluster_assign.comp.
struct LightGPU {
    float posOrDir[4]; // xyz = position (point/spot) / direction-to-light (directional), w = LightType
    float color[4];    // rgb + w = intensity (PI-scaled, see fillLightingUBO)
    float params[4];   // x = range (0 = infinite), y = castShadow,
                       // z = shadowIndex (spot shadow atlas tile, -1 = unshadowed),
                       // w = spot cos(inner)
    float spotDir[4];  // xyz = spot cone direction (unit, world), w = spot cos(outer)
};
static_assert(sizeof(LightGPU) == 64, "LightGPU layout mismatch");

// Legacy fixed array in LightingUBO: still feeds the forward transparency
// pass (which shades the global list) and carries the CSM sun for
// lighting.frag.  The clustered path reads the full light set from the
// lights SSBO instead.
constexpr uint32_t kMaxLights = 16; // must match lights[] in lighting/transparent shaders

// --- Clustered shading (Olsson et al. 2012; DOOM 2016/Eternal, SIGGRAPH) -----
// The view frustum is split into a screen-tile x exponential-depth grid of
// clusters; a compute pass (cluster_assign.comp) tests every punctual light
// against each cluster's view-space AABB once per frame and writes a
// per-cluster light index list.  lighting.frag then iterates only its own
// cluster's list instead of a globally truncated 16-light array, so the
// kMaxLights cap (and its CPU intensity/distance^2 scoring) is gone:
// up to kMaxSceneLights point/spot lights are considered, directionals
// (the CSM sun) bypass the clusters entirely.
constexpr uint32_t kMaxSceneLights = 1024;  // lights SSBO capacity (point+spot)
constexpr uint32_t kClusterTileSize = 64;   // screen tile edge, px (DOOM 2016 uses 64-128 px tiles)
constexpr uint32_t kClusterSlicesZ = 24;    // exponential depth slices (DOOM Eternal: 24)
constexpr uint32_t kMaxLightsPerCluster = 64; // fixed-capacity lists keep assignment deterministic
// Double-buffered to match every host's kFramesInFlight: the slot's buffers
// are only rewritten once its fence has passed (same rule as the per-slot UBOs).
constexpr uint32_t kClusterSlots = 2;

// Host-owned clustered-shading state (same ownership model as DepthPyramid):
// one per deferred render path (LR / GT / GT-SSAA) because the grid depends
// on the path resolution.  lightsBuffer (std430: uvec4 header +
// LightGPU[kMaxSceneLights]) is host-visible and refilled per frame;
// gridBuffer (uvec4 grid header + counts[N] + indices[N*kMaxLightsPerCluster],
// N = clusterCount) is device-local, its header written once at creation.
struct ClusterGrid {
    VkBuffer lightsBuffer[kClusterSlots] = {};
    VmaAllocation lightsMemory[kClusterSlots] = {};
    void* lightsMapped[kClusterSlots] = {};
    VkBuffer gridBuffer[kClusterSlots] = {};
    VmaAllocation gridMemory[kClusterSlots] = {};
    VkDescriptorSet assignSet[kClusterSlots] = {}; // pool-owned (clusterSetLayout)
    uint32_t gridX = 0;
    uint32_t gridY = 0;
    uint32_t gridZ = kClusterSlicesZ;
    uint32_t clusterCount = 0;
};

// --- CSM sun shadows (AC Unity reference: 4 cascades, 2048^2 each) -----------
constexpr uint32_t kShadowCascadeCount = 4;
constexpr uint32_t kShadowMapSize = 2048; // per cascade, fixed (resolution-independent)
// Practical split blend factor (0 = uniform, 1 = logarithmic).
constexpr float kShadowSplitLambda = 0.5f;
// Shadow coverage is capped at this distance (m); beyond it the sun is
// unshadowed (AC Unity-style range limit keeps cascade texel density useful).
constexpr float kShadowMaxDistance = 200.f;
// Light-space margin (m) pulling each cascade's near plane back towards the
// sun so casters between the slice and the sun still land in the map.
constexpr float kShadowCasterMargin = 150.f;
// CSM stabilization (bounding-sphere cascades, per MSFT "Common Techniques to
// Improve Shadow Depth Maps"): the cascade radius is padded by two texels and
// quantized to a 1/N-metre grid (N = kShadowRadiusSnap) so the world size of a
// shadow texel stays constant for a given split distance regardless of camera
// position, rotation or TAA jitter — a drifting texel size is what makes the
// snap grid itself swim and the shadow edges shimmer.  The padding guarantees
// the snapped ortho window always covers the whole slice sphere (the floor()
// centre snap can shift the window almost a full texel towards -x/-y).
constexpr float kShadowRadiusSnap = 16.f;
// The light-space depth range [minZ, maxZ] is quantized to this fixed step
// (m) so the depth remap does not resample against a drifting near/far range
// frame-to-frame (which would shimmer even with stable XY texels).
constexpr float kShadowZSnap = 0.25f;
// Rasterizer-side depth bias (vkCmdSetDepthBias, dynamic state) of the shadow
// pass; mirrored into LightingUBO::shadowParams.xy for reference.
constexpr float kShadowDepthBiasConstant = 1.25f;
constexpr float kShadowDepthBiasSlope = 1.75f;

// Host-owned render target set for the CSM pass: one D32 texture array with
// one layer per cascade, a per-layer attachment view and an all-layer sampled
// view (bound with DeferredCore's comparison sampler).
struct ShadowTargets {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation memory = VK_NULL_HANDLE;
    VkImageView arrayView = VK_NULL_HANDLE;                       // 4 layers, sampled
    VkImageView layerViews[kShadowCascadeCount] = {};             // per-cascade attachment
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;             // tracked by the host
};

// --- Local-light shadow atlas (Phase 4b: shadow-casting spot lights) ---------
// One D32 atlas per host (shared by the LR/GT/SSAA paths, like the CSM maps),
// split into a row-major grid of square tiles.  Each frame the hosts score
// every spot light with castShadow by intensity / distance^2 to the camera
// (stable sort, ties keep the lower scene index) and assign the top
// kShadowAtlasTiles lights a tile (Light::shadowIndex = tile); each tile's
// spot view-projection goes to LightingUBO::shadowTileVp and the map is
// rendered into the tile rect by recordSpotShadowPass.
// Point lights are NOT shadow-mapped this phase: an omnidirectional map needs
// 6 cube-face tiles per light (6x the tile budget plus per-face selection in
// the fragment shader) or a seam-prone dual-paraboloid pair, neither of which
// pays for itself in the current scenes.  Their shadowIndex stays -1 and they
// shade unshadowed (documented leftover).
constexpr uint32_t kShadowAtlasSize = 4096;     // D32, whole atlas
constexpr uint32_t kShadowAtlasTileSize = 1024; // one tile per shadowed spot
constexpr uint32_t kShadowAtlasGrid = kShadowAtlasSize / kShadowAtlasTileSize; // 4x4
constexpr uint32_t kShadowAtlasTiles = kShadowAtlasGrid * kShadowAtlasGrid;    // 16
// Perspective far plane for a spot with range == 0 (infinite): the shadow map
// only needs to cover a generous fixed reach.
constexpr float kSpotShadowInfiniteRange = 100.f;
// Angular margin (radians) beyond the outer cone so the penumbra band and the
// PCF kernel near the cone rim stay inside the map.
constexpr float kSpotShadowFovMargin = 0.04f;

// Host-owned spot-shadow atlas (same ownership model as ShadowTargets).  The
// single whole-atlas 2D view doubles as the depth attachment (each light is
// restricted to its tile via renderArea + viewport/scissor) and as the
// comparison-sampled view (lighting set binding 14).
struct ShadowAtlas {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED; // tracked by the host
};

// Per-frame cascade data fed into fillLightingUBO.  splitDepth[i] is the far
// boundary of cascade i in view-space depth (positive metres).
struct ShadowFrame {
    Mat4 cascadeVp[kShadowCascadeCount];
    float splitDepth[kShadowCascadeCount] = {};
    int32_t lightIndex = -1;   // index into LightingUBO::lights of the shadowed sun
    bool debugCascades = false;
    // Spot shadow atlas (Phase 4b): per-tile spot view-projections and the
    // number of tiles rendered this frame (0 = no spot shadows).  Tile t
    // covers pixels [(t%kShadowAtlasGrid)*tile, (t/kShadowAtlasGrid)*tile] +
    // kShadowAtlasTileSize^2.
    Mat4 atlasVp[kShadowAtlasTiles];
    uint32_t atlasTileCount = 0;
    // Frame index; drives the deterministic CSM cascade-transition dither in
    // lighting.frag (LightingUBO::shadowAtlasParams.z).
    uint32_t frameIndex = 0;
};

// std140, matches the LightingUBO block in lighting.frag.
struct LightingUBO {
    float invViewProj[16];
    float cameraPos[4];
    LightGPU lights[kMaxLights];
    float lightCounts[4]; // x = active light count, yzw reserved
    float ambient[4];
    float iblParams[4]; // envIntensity, prefilterMaxLod, skyboxEnabled, unused
    float cascadeVp[kShadowCascadeCount][16];
    float cascadeSplits[4]; // view-space depth (positive) of each cascade's far plane
    float shadowParams[4];  // x = depthBiasConstant, y = depthBiasSlope,
                            // z = shadowsEnabled, w = debugCascades
    float viewForward[4];   // xyz = camera forward (world); w = shadowed sun light index (-1 = none)
    float clusterDepth[4];  // x = near, y = far (exponential cluster slicing), zw unused
    // Spot shadow atlas (Phase 4b): per-tile spot light view-projections
    // (perspective, y-flipped Vulkan NDC like the CSM matrices); only the
    // first shadowAtlasParams.x entries are valid.
    float shadowTileVp[kShadowAtlasTiles][16];
    float shadowAtlasParams[4]; // x = tiles rendered this frame,
                                // y = 1 / kShadowAtlasSize (atlas texel, PCF step),
                                // z = frame index (CSM cascade dither),
                                // w = contact shadows enabled (Phase 4c, hosts
                                //     set after fillLightingUBO)
    // Forward view-projection of this path (jittered for the LR path): the
    // lighting pass's screen-space contact-shadow march (Phase 4c) reprojects
    // its world-space samples with it.
    float viewProj[16];
};
static_assert(sizeof(LightingUBO) == 2576, "LightingUBO std140 size mismatch");

// Push constants of the cluster light-assignment pass (cluster_assign.comp),
// 112 bytes.  The per-cluster view-space AABBs are derived in the shader from
// the tile NDC corners and the exponential slice depths (no per-frame CPU
// AABB rebuild, no extra invProj buffer).
struct ClusterAssignPush {
    float view[16];       // world -> view of this frame's camera
    float projParams[4];  // x = proj.m[0], y = proj.m[5], z = near, w = far
    uint32_t grid[4];     // gridX, gridY, gridZ, kMaxLightsPerCluster
    uint32_t misc[4];     // screenW, screenH, kClusterTileSize, unused
};
static_assert(sizeof(ClusterAssignPush) == 112, "ClusterAssignPush size mismatch");

struct ScenePush {
    float model[16];
    float prevModel[16];
    float normalModel[16]; // transpose(inverse(mat3(model))), upper 3x3 used
};
static_assert(sizeof(ScenePush) == 192, "ScenePush size mismatch");

// Skinned GBuffer draws (gbuffer_skinned.vert): ScenePush + the draw's joint
// palette offsets.  The matrix slots are unused (the palette already carries
// the node transform, per the glTF spec) but kept so one push range covers
// both static and skinned draws; paletteCur/palettePrev address the current /
// previous-frame blocks of the joint palette SSBO (scene set binding 2).
struct SkinnedScenePush {
    float model[16];
    float prevModel[16];
    float normalModel[16];
    uint32_t paletteCur = 0;
    uint32_t palettePrev = 0;
    uint32_t pad[2] = {};
};
static_assert(sizeof(SkinnedScenePush) == 208, "SkinnedScenePush size mismatch");

// Push constants of the shadow depth pass (shadow_depth.vert): per-instance
// model + the cascade's light view-projection.  128 bytes fits the guaranteed
// maxPushConstantsSize minimum, and the pass reuses scenePipelineLayout()
// (whose 192-byte range covers it).
struct ShadowPush {
    float model[16];
    float lightVp[16];
};
static_assert(sizeof(ShadowPush) == 128, "ShadowPush size mismatch");

// Skinned shadow depth draws (shadow_depth_skinned.vert): the current-frame
// palette offset only — the shadow pass writes no motion vectors, so no
// previous palette is needed.
struct SkinnedShadowPush {
    float model[16]; // unused (palette carries the node transform)
    float lightVp[16];
    uint32_t paletteCur = 0;
    uint32_t pad[3] = {};
};
static_assert(sizeof(SkinnedShadowPush) == 144, "SkinnedShadowPush size mismatch");

// Push constants of the GTAO compute pass (ssao.comp), 96 bytes.  The
// filename is historical; the shader is a XeGTAO-style GTAO main pass.
// invViewProj is pushed (not inverted per pixel) — same matrix the lighting
// pass carries in its UBO prefix.
struct SsaoPush {
    float invViewProj[16]; // inverse of this path's GBuffer view-projection (jittered for LR)
    float params[4];   // x = radius (m), y = final power, z/w = depth unpack
                       // (viewZ = mul / (add - ndcDepth)); non-reversed Vulkan depth
    float params2[4];  // x = frame index (drives the XeGTAO temporal noise spin),
                       // y = max depth mip LOD, z = far plane (sky test), w = unused
};
static_assert(sizeof(SsaoPush) == 96, "SsaoPush size mismatch");

// Push constants of the temporal AO accumulation pass (ssao_temporal.comp),
// 144 bytes (< the 192 the device check already guarantees for ScenePush).
struct SsaoTemporalPush {
    float invViewProj[16];  // current frame, this path's convention (jittered for LR)
    float prevViewProj[16]; // previous frame, same path/convention
    float params[4];        // x = EMA blend weight, y = reset (1 = pass-through), zw unused
};
static_assert(sizeof(SsaoTemporalPush) == 144, "SsaoTemporalPush size mismatch");

// GTAO defaults (Jimenez 2016 / XeGTAO heuristics).  0.5 m is a near-field
// contact radius at Bistro street scale; FinalValuePower 2.2 is XeGTAO High.
// The RadiusMultiplier/FalloffRange mirror XE_GTAO_DEFAULT_* and are shared
// with the AO depth-chain filter (XeGTAO_DepthMIPFilter, hiz_downsample.comp).
constexpr float kSsaoRadius = 0.5f;
constexpr float kSsaoRadiusMultiplier = 1.457f; // XE_GTAO_DEFAULT_RADIUS_MULTIPLIER
constexpr float kSsaoFalloffRange = 0.615f;     // XE_GTAO_DEFAULT_FALLOFF_RANGE
constexpr float kSsaoDepthMipRangeScale = 0.75f; // XeGTAO_DepthMIPFilter, "found empirically"
constexpr float kSsaoBias = 0.03f;      // unused (kept so SsaoPush comments stay stable)
constexpr float kSsaoIntensity = 1.5f;  // unused
constexpr float kSsaoPower = 2.2f;
// Temporal accumulation (Jimenez 2016 temporal filtering; XeGTAO itself defers
// to TAA): exponential moving average with ~8-frame window, history rejected
// on reprojection depth mismatch.  Frame-index driven, deterministic.
constexpr float kSsaoTemporalBlend = 0.125f;
constexpr float kSsaoTemporalDepthReject = 0.04f; // relative view-Z tolerance

// Working GTAO target: R = visibility, G = view-space |z| for the temporal
// accumulation and the bilateral denoise.  The filtered output (gbAo_) stays
// R16F for lighting/transparent.
constexpr VkFormat kAoRawFormat = VK_FORMAT_R16G16_SFLOAT;

// --- Hi-Z depth pyramid (max-reduced opaque depth mip chain) -----------------
// Shared general-purpose resource in two flavors:
//   * SSR chains (default): R32F NDC depth, 2x2 MAX-reduce — depth semantics
//     are non-reversed Vulkan (0 = near, 1 = far), so the marcher needs the
//     farthest surface per cell.
//   * AO chains (aoFilter): R32F LINEAR view-space Z (mip 0 is a true
//     per-texel NDC->viewZ conversion, not the SSR pseudo-copy), reduced with
//     XeGTAO's DepthMIPFilter — a far-biased weighted average that keeps
//     coherent-surface mips smooth while letting thin near occluders fade out
//     instead of haloing (Intel XeGTAO, vaGTAO/XeGTAO_PrefilterDepths16x16).
// R32F, mip 0 = full-res, mips 1..N down to 1x1.
constexpr VkFormat kDepthPyramidFormat = VK_FORMAT_R32_SFLOAT;

// Push constants of the Hi-Z downsample pass (hiz_downsample.comp), 32 bytes.
struct HiZPush {
    int32_t srcSize[2];   // texel size of the source level (edge clamp)
    int32_t aoFilter;     // 0 = SSR max-reduce, 1 = XeGTAO DepthMIPFilter (view-Z chain)
    int32_t level;        // AO chains: level 0 is the per-texel NDC->viewZ copy
    float depthUnpack[2]; // viewZ = mul / (add - ndcDepth); AO chains only
    float falloff[2];     // x = falloffMul, y = falloffAdd (DepthMIPFilter weights)
};
static_assert(sizeof(HiZPush) == 32, "HiZPush size mismatch");

// Host-owned depth pyramid (same ownership model as ShadowTargets: the host
// holds the struct and the descriptor pool, DeferredCore provides
// create/write/record/destroy).  The image stays in VK_IMAGE_LAYOUT_GENERAL
// for its whole lifetime — every mip ping-pongs between compute storage
// writes and sampled reads — so hosts track no per-frame layout for it.
struct DepthPyramid {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation memory = VK_NULL_HANDLE;
    VkImageView chainView = VK_NULL_HANDLE;   // all mips; bound to the SSR marcher
    std::vector<VkImageView> mipViews;        // per-mip views (downsample src/dst)
    std::vector<VkDescriptorSet> sets;        // per-mip downsample sets (pool-owned)
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mipCount = 0;
    // AO-chain flavor (see the header comment above); zeroed for SSR chains.
    bool aoFilter = false;
    float depthUnpack[2] = {0.f, 0.f};
    float falloff[2] = {0.f, 0.f};
};

// --- GPU occlusion culling + indirect GBuffer draws (Phase 7a) -----------------
// 2020-standard GPU-driven visibility for the static opaque instances:
// per-instance transform/material data lives in a device-local SSBO (scene
// set binding 3, read by gbuffer.vert via gl_InstanceIndex) instead of
// per-draw push constants, and the GBuffer pass issues one
// vkCmdDrawIndexedIndirect per material run over a CPU-filled command buffer.
// A compute pass (occlusion_cull.comp) then tests each candidate's world AABB
// against the PREVIOUS frame's Hi-Z pyramid (the path's max-reduced NDC depth
// chain, reprojected with the previous frame's exact view-projection —
// jittered for the LR path) and zeroes the instanceCount of fully occluded
// commands (1-frame latency; first frame / resize / path switch => cull
// skipped, everything visible).
// CPU frustum culling + LOD selection (Phase 3b) stay on the CPU: they are
// path-independent and already pay for themselves; the GPU pass only re-tests
// the CPU-visible set (lower implementation risk than moving frustum/LOD into
// compute, and the candidate list doubles as the command buffer content).
// Commands stay in place (no compaction): zero-instance indirect draws are
// nearly free, the output keeps the CPU's sorted (material, mesh) order and
// the result is bit-deterministic (no atomics, no prefix sums).  Because the
// CPU always knows the run lengths, plain vkCmdDrawIndexedIndirect is used
// instead of the count variant (core since Vulkan 1.2 — nothing extra to
// enable either way).  Skinned and BLEND instances keep the direct-draw
// paths (few; transparency is order-dependent).
constexpr uint32_t kCullSlots = 2; // per-slot staging, matches every host's kFramesInFlight

// std430, matches gbuffer.vert / occlusion_cull.comp.
struct GpuInstance {
    float model[16];
    float prevModel[16];
    float normalModel[16];
    float aabbMin[4]; // xyz = world AABB min
    float aabbMax[4];
    uint32_t materialIndex = 0;
    uint32_t flags = 0; // bit0: occlusion-exempt (transform changed this frame)
    uint32_t pad[2] = {};
};
static_assert(sizeof(GpuInstance) == 240, "GpuInstance std430 size mismatch");

// Contiguous candidate range sharing one material (one indirect draw call).
struct CullDrawRun {
    uint32_t materialIndex = 0;
    uint32_t firstCommand = 0;
    uint32_t commandCount = 0;
};

// Host-owned per-frame instance data: the CPU fills the slot's mapped staging
// (buildInstanceList), recordInstanceUpload copies it into the device-local
// SSBO the GBuffer vertex shaders and the cull pass read.  One per host; all
// paths share it (the candidate list is resolution-independent).
struct InstanceBuffer {
    VkBuffer staging[kCullSlots] = {};
    VmaAllocation stagingMemory[kCullSlots] = {};
    void* stagingMapped[kCullSlots] = {};
    VkBuffer buffer = VK_NULL_HANDLE; // device-local SSBO (scene set binding 3)
    VmaAllocation memory = VK_NULL_HANDLE;
    uint32_t capacity = 0;
};

// Host-owned per-path indirect/cull state (same ownership model as
// DepthPyramid): one per deferred render path because the cull set binds that
// path's Hi-Z chain and prevViewProj tracks that path's last rendered frame.
// The command buffer contents are CPU-identical across paths; each path gets
// its own buffer because its cull pass zeroes different instanceCounts
// (resolution-dependent Hi-Z; the culled sets agree up to edge texels).
struct CullChannel {
    VkBuffer cmdStaging[kCullSlots] = {};
    VmaAllocation cmdStagingMemory[kCullSlots] = {};
    void* cmdStagingMapped[kCullSlots] = {};
    VkBuffer indirect = VK_NULL_HANDLE; // device-local INDIRECT | STORAGE
    VmaAllocation indirectMemory = VK_NULL_HANDLE;
    VkDescriptorSet cullSet = VK_NULL_HANDLE; // pool-owned (cullSetLayout)
    // Exact view-projection that produced the bound Hi-Z chain (jittered for
    // the LR path); prevValid = false skips the cull pass (first frame of the
    // path / pyramid recreated / camera cut), leaving every command visible.
    Mat4 prevViewProj = Mat4::identity();
    bool prevValid = false;
    uint32_t capacity = 0;
};

// Push constants of occlusion_cull.comp.
struct OcclusionCullPush {
    float prevViewProj[16];
    uint32_t candidateCount = 0;
    uint32_t mipCount = 0;
    int32_t screenW = 0;
    int32_t screenH = 0;
};
static_assert(sizeof(OcclusionCullPush) == 80, "OcclusionCullPush size mismatch");

// --- Temporal AO accumulation (GTAO visibility EMA) ----------------------------
// Host-owned ping-pong state (same ownership/layout model as DepthPyramid):
// two RG16F buffers (R = accumulated visibility, G = CURRENT-frame view |z| —
// never blended, so the denoise weights stay correct), GENERAL for life.
// ssao_temporal.comp reprojects buffer 1-i with the previous frame's
// view-projection, depth-rejects stale texels and EMA-blends into buffer i;
// ssao_blur.comp then denoises buffer i.  Camera-only reprojection (scenes
// are static; a moving occluder would be caught by the depth rejection).
struct AoHistory {
    VkImage image[2] = {};
    VmaAllocation memory[2] = {};
    VkImageView view[2] = {};
    VkDescriptorSet temporalSet[2] = {}; // [i]: reads view[1-i], writes view[i]
    VkDescriptorSet blurSet[2] = {};     // [i]: samples view[i] into the path's AO target
    uint32_t width = 0;
    uint32_t height = 0;
};

// --- Temporal SSR accumulation (Phase 2d) --------------------------------------
// Same ownership/layout model as AoHistory: two RGBA16F ping-pong buffers,
// GENERAL for life.  Layout: rgb = filtered composite delta
// conf*(ssrColor*envBRDF - specIbl) — the confidence is premultiplied, so one
// EMA smooths hit/miss flicker and hit-colour noise together; a = CURRENT
// frame's view |z| (never blended) for the reprojection depth rejection.
// The trace pass (ssr_opaque.comp) writes the raw per-frame delta + view |z|
// into a host-owned full-res trace target; ssr_temporal.comp reprojects
// buffer 1-i, depth-rejects + neighbourhood-clamps + EMA-blends into buffer i
// and fuses the composite (in-place RMW add on the lit HDR target).
// Camera-only reprojection, frame-index driven — deterministic.
// Ref: Stachowiak, Stochastic SSR, SIGGRAPH 2015; UE SSR temporal.
struct SsrHistory {
    VkImage image[2] = {};
    VmaAllocation memory[2] = {};
    VkImageView view[2] = {};
    VkDescriptorSet temporalSet[2] = {}; // [i]: reads view[1-i] + trace, writes view[i], RMW sceneColor
    uint32_t width = 0;
    uint32_t height = 0;
};
constexpr VkFormat kSsrTraceFormat = deferred::kHdrColorFormat; // RGBA16F trace target + history
// Temporal accumulation constants (same relative view-Z tolerance as the
// GTAO temporal pass; see ssao_temporal.comp).  The EMA window is ~12 frames
// (0.08): the 8-frame window let the grazing-angle hit-position jitter of the
// marcher through as per-pixel speckle flicker on ground planes; the
// staleness-driven blend in ssr_temporal.comp still drops moved reflection
// content within a frame or two, so the longer window costs no ghosting and
// static accumulation (the mirror anti-aliasing) only gets sharper.
constexpr float kSsrTemporalBlend = 0.08f;
constexpr float kSsrTemporalDepthReject = 0.04f;

// --- HDR color mip chain (box-filtered lit-color pyramid) --------------------
// Roughness-aware SSR (Phase 1b-2) samples this chain at lod = roughness *
// (mipCount - 1) instead of a single sharp mip-0 read, so one ray
// approximates the widened GGX lobe.  The Phase 6a bloom pyramid is a
// SEPARATE chain (BloomPyramid below) on purpose: this chain is a plain box
// average feeding reflection LOD, while bloom starts from a thresholded
// extract — merging them would couple the bloom threshold to the reflection
// blur.  RGBA16F like the lighting target, mip 0 = straight copy of the lit
// opaque HDR color, mips 1..N = 2x2 box average.  The chain length rule
// matches the depth pyramid (full chain down to 1x1).
constexpr VkFormat kColorPyramidFormat = deferred::kHdrColorFormat;

// Host-owned color pyramid.  Same ownership/layout model as DepthPyramid:
// the host holds the struct and the descriptor pool, the image stays in
// VK_IMAGE_LAYOUT_GENERAL for its whole lifetime (zero layout tracking).
struct ColorPyramid {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation memory = VK_NULL_HANDLE;
    VkImageView chainView = VK_NULL_HANDLE;   // all mips; bound to the SSR marcher
    std::vector<VkImageView> mipViews;        // per-mip views (downsample src/dst)
    std::vector<VkDescriptorSet> sets;        // per-mip downsample sets (pool-owned)
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mipCount = 0;
};

// Push constants of the opaque SSR pass (ssr_opaque.comp): the path's
// view-projection (jittered for the LR path).  Everything else the marcher
// needs (invViewProj, cameraPos, iblParams) comes from the bound LightingUBO
// prefix, so no second UBO per path is required.
struct SsrPush {
    float viewProj[16];
};
static_assert(sizeof(SsrPush) == 64, "SsrPush size mismatch");

// --- Froxel volumetric fog (Phase 5a) ----------------------------------------
// Frostbite 2015 unified participating media (Hillaire, "Physically Based
// Sky, Atmosphere and Cloud Rendering in Frostbite") / UE4.16 Volumetric Fog
// (Wronski, SIGGRAPH 2014 + UE4 docs): a camera-frustum-aligned voxel grid
// ("froxels") fed by four compute passes per frame and path:
//   1. inject   (volfog_inject.comp): exponential height fog + static 3D
//      value noise (frame-index domain offset) -> density + scattering albedo;
//   2. light    (volfog_light.comp): CSM sun (shadow-map visibility -> god
//      rays) + the froxel's cluster light list (Phase 4a SSBO reuse) +
//      isotropic ambient; single scattering with a Schlick phase function;
//   3. temporal (volfog_temporal.comp): previous-frame volume reprojection +
//      EMA — the same pattern as the GTAO/SSR temporal filters;
//   4. march    (volfog_march.comp): front-to-back analytic per-slice
//      integration -> per-froxel accumulated inscatter + transmittance.
// volfog_composite.comp then applies color * transmittance + inscatter to the
// lit HDR target right after the lighting pass (before tonemapping).
// All volumes are RGBA16F 3D, GENERAL for life (same ownership/layout model
// as AoHistory/SsrHistory).  All inputs derive from the frame index:
// deterministic, no wall clock.  The fog passes use the UN-JITTERED camera
// matrices for both paths so the volume does not swim under TAA jitter.
constexpr uint32_t kFroxelTileSize = 12; // px per froxel XY at path res (160x90 at 1080p)
constexpr uint32_t kFroxelSlices = 64;   // exponential Z slices (UE4 volumetric fog default)
constexpr VkFormat kFroxelFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
// Temporal EMA window (~8 frames), same convention as GTAO/SSR.
constexpr float kVolFogTemporalBlend = 0.125f;

// Push blocks of the froxel passes; all carry the un-jittered view inverse +
// projection parameters so the shader derives each froxel's world position
// analytically (volfog.glsl froxelWorldPos).
struct VolFogInjectPush {
    float invView[16];
    float projParams[4];  // x = proj.m[0], y = proj.m[5], z = near, w = fog far
    float fogA[4];        // density, heightFalloff, baseHeight, noiseScale
    float fogAlbedo[4];   // rgb = albedo, w = noiseStrength
    float fogB[4];        // x = frame index, yzw unused
    uint32_t grid[4];     // dimX, dimY, dimZ, unused
};
static_assert(sizeof(VolFogInjectPush) == 144, "VolFogInjectPush size mismatch");

struct VolFogLightPush {
    float invView[16];
    float projParams[4];  // x = proj.m[0], y = proj.m[5], z = near, w = fog far
    float fogA[4];        // x = anisotropy g, y = ambient scale, zw unused
    uint32_t grid[4];     // dimX, dimY, dimZ, unused
    uint32_t misc[4];     // screenW, screenH, kClusterTileSize, unused
};
static_assert(sizeof(VolFogLightPush) == 128, "VolFogLightPush size mismatch");

struct VolFogTemporalPush {
    float invView[16];      // current frame (un-jittered)
    float prevViewProj[16]; // previous frame (un-jittered)
    float projParams[4];    // x = proj.m[0], y = proj.m[5], z = near, w = fog far
    float params[4];        // x = EMA blend, y = reset (1 = pass-through)
    uint32_t grid[4];       // dimX, dimY, dimZ, unused
};
static_assert(sizeof(VolFogTemporalPush) == 176, "VolFogTemporalPush size mismatch");

struct VolFogMarchPush {
    float invView[16];
    float projParams[4];  // x = proj.m[0], y = proj.m[5], z = near, w = fog far
    uint32_t grid[4];     // dimX, dimY, dimZ, unused
};
static_assert(sizeof(VolFogMarchPush) == 96, "VolFogMarchPush size mismatch");

struct VolFogCompositePush {
    float depthParams[4]; // x = proj.m[10], y = proj.m[14] (NDC -> view Z),
                          // z = near, w = fog far
};
static_assert(sizeof(VolFogCompositePush) == 16, "VolFogCompositePush size mismatch");

// Fragment-stage push block of the transparency pipelines
// (transparent.frag / transparent_gt.frag), appended AFTER the vertex-stage
// SkinnedScenePush range (offset = sizeof(SkinnedScenePush)).  Carries the
// froxel volume depth range so the shader can sample the ray-integrated
// inscatter/transmittance at the fragment's view depth (volumetric fog on
// translucency, Phase 5a fix).
struct TransparentFogPush {
    float params[4]; // x = near, y = fog far, z = enabled (1/0), w unused
};
static_assert(sizeof(TransparentFogPush) == 16, "TransparentFogPush size mismatch");

// Host-owned froxel volume set, one per deferred render path (grid resolution
// scales with the path resolution — same rule as ClusterGrid).  Five volumes:
// the inject target (per-frame media properties), the raw lit volume (light
// pass output), the temporal ping-pong pair and the ray-integrated result.
// Sets are pool-owned; the light sets are per-slot because they bind the
// per-slot LightingUBO + ClusterGrid buffers.
struct VolFogVolume {
    VkImage injectImage = VK_NULL_HANDLE;
    VmaAllocation injectMemory = VK_NULL_HANDLE;
    VkImageView injectView = VK_NULL_HANDLE;
    VkImage rawImage = VK_NULL_HANDLE;
    VmaAllocation rawMemory = VK_NULL_HANDLE;
    VkImageView rawView = VK_NULL_HANDLE;
    VkImage histImage[2] = {};
    VmaAllocation histMemory[2] = {};
    VkImageView histView[2] = {};
    VkImage intImage = VK_NULL_HANDLE;
    VmaAllocation intMemory = VK_NULL_HANDLE;
    VkImageView intView = VK_NULL_HANDLE;
    VkDescriptorSet injectSet = VK_NULL_HANDLE;
    VkDescriptorSet lightSet[kClusterSlots] = {}; // per-slot (UBO + cluster SSBOs)
    VkDescriptorSet temporalSet[2] = {};          // [i]: read raw + hist[1-i], write hist[i]
    VkDescriptorSet marchSet[2] = {};             // [i]: read hist[i], write intImage
    VkDescriptorSet compositeSet = VK_NULL_HANDLE;
    uint32_t dimX = 0;
    uint32_t dimY = 0;
    uint32_t dimZ = kFroxelSlices;
};

// --- Bloom pyramid (Phase 6a) --------------------------------------------------
// COD:AW / Unity-style bloom (Jimenez, "Next Generation Post Processing in
// Call of Duty: Advanced Warfare", SIGGRAPH 2014; Unity HDRP/URP bloom):
//   1. extract (bloom_extract.comp): quadratic soft-knee threshold into mip 0
//      at half resolution (threshold/knee semantics unchanged from the old
//      single-level bloom);
//   2. downsample (bloom_downsample.comp): 13-tap energy-preserving filter,
//      kBloomMipCount - 1 levels (1080p: 960x540 -> 60x34);
//   3. upsample (bloom_upsample.comp): 3x3 tent filter of the lower mip
//      accumulated back into each level (widest radius per cost — one
//      thresholded chain blurred at every scale instead of one half-res
//      Gaussian);
//   4. composite (bloom_composite.comp): accumulated mip 0 added onto the
//      lit HDR target, still before tonemapping (HDR domain, as before).
// The chain is independent of ColorPyramid (see its comment).  Host-owned,
// GENERAL-for-life resource model matching ColorPyramid.
constexpr uint32_t kBloomMipCount = 5; // mip 0 = half-res extract, 1..4 = downsamples
struct BloomPush {
    float params[4]; // extract: threshold, knee; composite: strength
};
static_assert(sizeof(BloomPush) == 16, "BloomPush size mismatch");
constexpr float kBloomThreshold = 1.0f;
constexpr float kBloomKnee = 0.5f;
constexpr float kBloomStrength = 0.15f;

// Host-owned bloom pyramid (one per deferred path).  Sets are pool-owned;
// destroyBloomPyramid only releases the image and views.
struct BloomPyramid {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation memory = VK_NULL_HANDLE;
    std::vector<VkImageView> mipViews;        // kBloomMipCount single-mip views
    VkDescriptorSet extractSet = VK_NULL_HANDLE;   // HDR src -> mip 0
    std::vector<VkDescriptorSet> downSets;         // [i]: mip i -> mip i+1
    std::vector<VkDescriptorSet> upSets;           // [i]: mip i+1 -> accumulate into mip i
    VkDescriptorSet compositeSet = VK_NULL_HANDLE; // mip 0 -> HDR color RMW
    uint32_t width = 0;  // mip 0 extent (half res)
    uint32_t height = 0;
};

// --- Motion blur + depth of field (Phase 6b) -----------------------------------
// Reconstruction-filter motion blur (McGuire et al., "A Fast and Stable
// Feature-Aware Motion Blur Filter", HPG 2012; same tile-max/gather structure
// as DOOM Eternal's motion blur, SIGGRAPH 2020):
//   1. tile max   (motion_blur_tilemax.comp): max-magnitude velocity per
//      kMotionBlurTileSize^2 tile of the path's motion RT (pixel units);
//   2. neighbour  (motion_blur_neighborhood.comp): 3x3 dilation of the tile
//      grid so edge pixels search far enough for fast neighbours;
//   3. gather     (motion_blur_gather.comp): full-res N-tap line gather along
//      the dominant velocity (own vs tile max), coverage x depth (foreground
//      priority) x velocity-difference weights, Poisson jitter rotated by
//      frame index (deterministic).
// Depth of field (UE4 scatter-as-gather CoC; Guertin, GDC 2013):
//   1. coc        (dof_coc.comp): half-res RGBA16F, rgb = downsampled color,
//      a = signed CoC / maxRadius (thin-lens approximation, manual focus
//      distance or auto-focus on the screen-centre depth texel —
//      deterministic, no CPU readback);
//   2. gather     (dof_gather.comp): half-res Poisson-disk gather with
//      cylinder coverage into premultiplied foreground/background layers;
//   3. composite  (dof_composite.comp): full-res, background layer behind the
//      sharp image, foreground on top, written back into the lit HDR target.
// Both run in the HDR domain after lighting+bloom and before upscale/present,
// once per deferred path (LR at render res feeding the upscaler, GT at native
// res, GT-SSAA at 2x before the downsample) with the same algorithm and the
// same parameters; the blur-radius clamps scale with path height so the
// display-space blur is resolution-independent.
constexpr uint32_t kMotionBlurTileSize = 20;  // px per tile (McGuire 2012 tile size)
constexpr float kMotionBlurShutter = 0.5f;    // 180-degree shutter
// Blur-length clamp at 1080p, scaled by path height (display-space constant).
constexpr float kMotionBlurMaxPixels = 32.f;
constexpr uint32_t kMotionBlurGatherTaps = 12;
constexpr float kDofMaxCoC = 12.f;   // max bokeh radius at 1080p, scaled by path height
constexpr float kDofAperture = 1.5f; // CoC scale of the (focus - z) / z thin-lens term
// f-stop front end for the aperture scale: aperture = kDofAperture *
// (kDofDefaultFstop / fstop), so the default f/4 reproduces kDofAperture and
// smaller f-stops widen the bokeh (CLI --dof-fstop, GUI slider).
constexpr float kDofDefaultFstop = 4.f;
constexpr float kDofSkyFocus = 20.f; // focus fallback (m) when the screen centre is sky
constexpr uint32_t kDofGatherTaps = 24;

struct MotionBlurTilePush {
    int32_t srcSize[2]; // full-res motion size
    int32_t tileSize;
    int32_t pad;
};
static_assert(sizeof(MotionBlurTilePush) == 16, "MotionBlurTilePush size mismatch");

struct MotionBlurGatherPush {
    float params[4];  // x = shutter, y = max blur (px), z = frame index, w = tile size
    float params2[4]; // xy = full-res size, zw unused
};
static_assert(sizeof(MotionBlurGatherPush) == 32, "MotionBlurGatherPush size mismatch");

struct DofCocPush {
    float depthParams[4]; // x = proj m[10], y = proj m[14] (NDC -> view Z),
                          // z = far, w = max CoC (half-res px)
    float params[4];      // x = aperture scale, y = sky focus fallback (m),
                          // z = manual focus (m; <= 0 = auto-focus), w unused
};
static_assert(sizeof(DofCocPush) == 32, "DofCocPush size mismatch");

struct DofGatherPush {
    float params[4]; // x = max CoC (half-res px), y = frame index, zw unused
};
static_assert(sizeof(DofGatherPush) == 16, "DofGatherPush size mismatch");

struct DofCompositePush {
    float params[4]; // x = 1 / kDofGatherTaps (coverage normalization), yzw unused
};
static_assert(sizeof(DofCompositePush) == 16, "DofCompositePush size mismatch");

// Host-owned post-fx working targets, one per deferred path (same
// ownership/layout model as BloomPyramid: host holds the struct + descriptor
// pool, all images are GENERAL for life).  cocSet/compositeSet come in two
// flavours: reading the motion-blurred intermediate (Mb suffix) or the lit
// HDR target directly (Lit suffix, when motion blur is off but DOF is on).
struct PostFxTargets {
    VkImage tileMaxImage = VK_NULL_HANDLE;       // RG16F, tile grid
    VmaAllocation tileMaxMemory = VK_NULL_HANDLE;
    VkImageView tileMaxView = VK_NULL_HANDLE;
    VkImage neighborMaxImage = VK_NULL_HANDLE;   // RG16F, tile grid
    VmaAllocation neighborMaxMemory = VK_NULL_HANDLE;
    VkImageView neighborMaxView = VK_NULL_HANDLE;
    VkImage mbOutImage = VK_NULL_HANDLE;         // RGBA16F, full path res
    VmaAllocation mbOutMemory = VK_NULL_HANDLE;
    VkImageView mbOutView = VK_NULL_HANDLE;
    VkImage cocColorImage = VK_NULL_HANDLE;      // RGBA16F, half res (rgb + signed CoC)
    VmaAllocation cocColorMemory = VK_NULL_HANDLE;
    VkImageView cocColorView = VK_NULL_HANDLE;
    VkImage bgImage = VK_NULL_HANDLE;            // RGBA16F, half res (premultiplied layer)
    VmaAllocation bgMemory = VK_NULL_HANDLE;
    VkImageView bgView = VK_NULL_HANDLE;
    VkImage fgImage = VK_NULL_HANDLE;            // RGBA16F, half res (premultiplied layer)
    VmaAllocation fgMemory = VK_NULL_HANDLE;
    VkImageView fgView = VK_NULL_HANDLE;
    VkDescriptorSet tileSet = VK_NULL_HANDLE;        // motion -> tileMax
    VkDescriptorSet neighborSet = VK_NULL_HANDLE;    // tileMax -> neighborMax
    VkDescriptorSet gatherSet = VK_NULL_HANDLE;      // color+motion+neighbor+depth -> mbOut
    VkDescriptorSet cocSetMb = VK_NULL_HANDLE;       // mbOut+depth -> cocColor
    VkDescriptorSet cocSetLit = VK_NULL_HANDLE;      // lit color+depth -> cocColor
    VkDescriptorSet dofGatherSet = VK_NULL_HANDLE;   // cocColor -> bg/fg layers
    VkDescriptorSet compositeSetMb = VK_NULL_HANDLE; // sharp=mbOut -> lit color
    VkDescriptorSet compositeSetLit = VK_NULL_HANDLE;// sharp=lit color -> lit color
    VkDescriptorSet copybackSet = VK_NULL_HANDLE;    // mbOut -> lit color (MB on, DOF off)
    uint32_t width = 0;   // full path res
    uint32_t height = 0;
    uint32_t tilesX = 0;
    uint32_t tilesY = 0;
};

// Per-frame post-fx parameters handed to recordPostFxPass.  depthM10/depthM14
// are the projection entries used to unpack view Z (m14 / (ndc + m10));
// maxBlurPx/maxCocPx are the path-resolution-scaled clamps (host scales
// kMotionBlurMaxPixels / kDofMaxCoC by height / 1080).
struct PostFxParams {
    float depthM10 = 0.f;
    float depthM14 = 0.f;
    float farPlane = 1000.f;
    float maxBlurPx = kMotionBlurMaxPixels;
    float maxCocPx = kDofMaxCoC;
    float aperture = kDofAperture;
    // Manual DOF focus distance (m); <= 0 = auto-focus on the screen-centre
    // depth texel (dof_coc.comp).  CLI --dof-focus / GUI slider.
    float focusDistance = 0.f;
    bool motionBlur = true;
    bool dof = true;
};


// --- Terminal lens-effects chain (Phase 6a) ------------------------------------
// Shared by the viewer present pass (present.frag) and the compare/GUI column
// compose (compare_compose.frag).  Defaults are deliberately weak — the
// effects should texture the image, not announce themselves.  Every effect
// keys off its own strength; a zero strength skips it (GUI checkboxes / CLI
// --no-lens-fx).  Lens dirt is viewer-only: it modulates the HDR bloom
// pyramid, which the compare/GUI paths do not build.
constexpr float kLensCaStrength = 0.003f;       // radial UV split scale (corner ~2px @1080p)
constexpr float kLensVignetteStrength = 0.15f;  // corner darkening factor
constexpr float kLensGrainStrength = 0.008f;    // display-domain noise amplitude
constexpr float kLensDirtStrength = 0.5f;       // dirt mask x accumulated bloom (viewer)

// --- Auto exposure (UE4 AutoExposure / Frostbite histogram + EV solver) -------
// Two compute passes per frame and path: exposure_histogram.comp builds a
// 256-bin log2-luminance histogram of the lit HDR color (half-res sampling),
// exposure_solve.comp reduces it to an average log luminance, derives a
// target EV (geometric mean mapped onto the middle-grey key), and advances a
// smoothed EV with exponential up/down rates.  The smoothing step uses the
// FIXED kExposureFixedDt per frame (never wall-clock), so a fixed camera path
// produces the same exposure at the same frame index on every run — bench
// determinism depends on this.
constexpr uint32_t kExposureHistogramBins = 256;
constexpr float kExposureKeyValue = 0.18f;    // middle grey the average maps to
constexpr float kExposureSpeedUp = 3.f;       // EV/s towards a brighter target
constexpr float kExposureSpeedDown = 1.f;     // EV/s towards a darker target
constexpr float kExposureFixedDt = 1.f / 60.f; // per-frame smoothing step

// Push constants of exposure_solve.comp (ExposureSolvePush) and
// exposure_histogram.comp (HistogramPush).
struct ExposureSolvePush {
    float minEV = 0.f;      // clamp range for the target/smoothed EV
    float maxEV = 0.f;
    float evOffset = 0.f;   // exposure compensation (EV units)
    float resetState = 0.f; // != 0: snap the smoothed EV to the target (first frame)
};
static_assert(sizeof(ExposureSolvePush) == 16, "ExposureSolvePush size mismatch");

struct HistogramPush {
    int32_t sampleDims[2]; // half-resolution sample grid (ceil(src / 2))
};
static_assert(sizeof(HistogramPush) == 8, "HistogramPush size mismatch");

// GPU state record of the solve pass (std430 vec4).  Also the readback layout:
// hosts copy it to a per-slot staging buffer and read it after the slot's
// fence, so the CPU-side exposure used for present/preExposure/screenshots
// lags the GPU by kFramesInFlight frames (engine-style: last frames'
// luminance drives this frame's exposure).
struct ExposureState {
    float exposure;  // display-domain multiplier (exp2(evOffset - ev))
    float avgLogLum; // average log2 luminance of the measured frame
    float ev;        // smoothed EV
    float targetEV;  // unsmoothed target EV (debug)
};
static_assert(sizeof(ExposureState) == 16, "ExposureState size mismatch");

// Host-owned auto-exposure state (same ownership model as DepthPyramid: the
// host holds the struct and the descriptor pool, DeferredCore provides
// create/write/record/destroy).  One instance per HDR source path (LR / GT /
// GT-SSAA) — each path auto-exposes from its own HDR target, so compare
// columns legitimately differ (independent pipelines, engine behaviour).
struct AutoExposure {
    VkBuffer histogram = VK_NULL_HANDLE;          // kExposureHistogramBins uints
    VmaAllocation histogramMemory = VK_NULL_HANDLE;
    VkBuffer state = VK_NULL_HANDLE;              // ExposureState
    VmaAllocation stateMemory = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;         // sampler + histogram + state
    uint32_t srcWidth = 0;
    uint32_t srcHeight = 0;
};

// Host-side auto-exposure bundle: the GPU state (AutoExposure) plus the
// per-slot staging buffers the solved ExposureState is copied into each frame,
// plus the latest harvested CPU-side value.  The slot count matches every
// host's kFramesInFlight (2); the harvested value lags the GPU by that many
// frames (engine convention: earlier frames' luminance drives the current
// frame's exposure) and is deterministic under a fixed camera path.
constexpr uint32_t kExposureSlots = 2;
struct ExposureChannel {
    AutoExposure gpu;
    VkBuffer staging[kExposureSlots] = {};
    VmaAllocation stagingMemory[kExposureSlots] = {};
    void* stagingMapped[kExposureSlots] = {};
    bool pending[kExposureSlots] = {};
    float value = 1.f; // latest harvested exposure multiplier
};


class DeferredCore {
public:
    // Builds the IBL maps and loads the deferred shaders/layouts/pipelines.
    // envMapPath non-empty: static equirect HDR (unreadable file -> procedural
    // gradient fallback inside IblMaps).  envMapPath empty: procedural sky
    // atmosphere (Hillaire 2020, ibl/SkyAtmosphere.h) rendered for skySunDir;
    // the sky + IBL maps then follow updateAtmosphereSky().
    bool init(const VulkanContext& ctx, const char* envMapPath, const Vec3& skySunDir);
    void destroy(const VulkanContext& ctx);

    // Atmosphere mode only (init with an empty envMapPath): re-renders the
    // sky and regenerates the sun-dependent IBL maps for a new sun direction.
    // Blocking one-shot submission — call on sun changes, not per frame.
    // Returns false in static-env mode.
    bool updateAtmosphereSky(const VulkanContext& ctx, const Vec3& sunDir);
    bool atmosphereSky() const { return atmosphereSky_; }

    // --- UBO fillers (shared defaults: fallback lights, ambient, PI scaling) ---
    void fillSceneUBO(SceneUBO& out, const Scene& scene, const Camera& camera,
                      const Mat4& view, const Mat4& proj, const Mat4& projJittered,
                      const Mat4& prevViewProj, uint32_t renderW, uint32_t renderH,
                      float jitterX, float jitterY, bool jitter) const;
    // overrideLights (optional): pack this list instead of scene.lights — the
    // GUI sun controls rebuild the list per frame without touching the scene.
    // The fallback-to-defaultLights() rule applies only when no override is
    // given.  Packed order: shadowed sun first (slot 0, CSM lightIndex remaps
    // to it), then the remaining lights in scene order, truncated at
    // kMaxLights.  This legacy array feeds only the forward transparency pass
    // and the sun; the deferred lighting pass iterates the cluster lists
    // (fillClusterLights) which carry the full, untruncated light set.
    // shadow (optional): non-null enables CSM sampling in the shaders
    // (shadowParams.z = 1); null writes identity cascades with shadows off.
    // viewProj is the forward matrix invViewProj inverts (jittered for the LR
    // path); only the contact-shadow march reads it.
    void fillLightingUBO(LightingUBO& out, const Scene& scene, const Camera& camera,
                         const Mat4& viewProj, const Mat4& invViewProj,
                         const std::vector<Light>* overrideLights = nullptr,
                         const ShadowFrame* shadow = nullptr,
                         float iblIntensity = 1.f) const;

    // The light list fillLightingUBO/fillClusterLights would use: the override
    // when given, else scene.lights, else defaultLights().  Hosts filling a
    // ClusterGrid must pass the same list they pass to fillLightingUBO.
    static const std::vector<Light>& effectiveLights(const Scene& scene,
                                                     const std::vector<Light>* overrideLights);

    // --- clustered shading (see the constants above) ---------------------------
    // Packs every point/spot light of the list (scene order, capped at
    // kMaxSceneLights; directionals bypass the clusters) into the mapped
    // lights SSBO (uvec4 count header + LightGPU[]).  Returns the packed count.
    uint32_t fillClusterLights(void* mappedLightsSsbo, const std::vector<Light>& lights) const;
    // Creates the per-slot lights/grid buffers of one path (w x h = that
    // path's render resolution) and writes the grid header via staging.
    bool createClusterGrid(const VulkanContext& ctx, uint32_t w, uint32_t h,
                           ClusterGrid& out) const;
    // Allocates the per-slot assignment sets from the caller's pool and binds
    // the slot's lights (binding 0) + grid (binding 1) buffers.
    bool writeClusterGridSets(const VulkanContext& ctx, VkDescriptorPool pool,
                              ClusterGrid& grid) const;
    void destroyClusterGrid(const VulkanContext& ctx, ClusterGrid& grid) const;

    // Uploads the dynamic-offset material UBO array (one entry per material).
    bool createMaterialUbo(const VulkanContext& ctx, const Scene& scene, VkBuffer& buffer,
                           VmaAllocation& memory, uint32_t& stride) const;

    // --- descriptor writers (sets are allocated from the caller's pool) --------
    void writeTextureSet(const VulkanContext& ctx, VkDescriptorSet set, const Scene& scene) const;
    // Scene set binding 2: the joint palette SSBO for this frame slot
    // (Scene::skinPalette(slot)).  Call once per scene set after scene load;
    // only needed when the scene has skinned meshes (the static shaders never
    // statically use binding 2, so an unwritten binding is valid for them).
    void writeSceneSkinBinding(const VulkanContext& ctx, VkDescriptorSet set,
                               VkBuffer palette) const;
    // shadow = ShadowTargets::arrayView, or VK_NULL_HANDLE to leave binding 11
    // unwritten (hosts without a shadow pass; the shaders must run with
    // shadowParams.z == 0 then, which fillLightingUBO guarantees by default).
    // shadowAtlas = ShadowAtlas::view (binding 14), same VK_NULL_HANDLE
    // convention (safe while shadowAtlasParams.x == 0).
    // clusterLights/clusterGrid bind this slot's ClusterGrid buffers (bindings
    // 12/13: full lights SSBO + this path's per-cluster light lists).
    void writeLightingSet(const VulkanContext& ctx, VkDescriptorSet set, VkBuffer lightingUbo,
                          VkImageView albedo, VkImageView normal, VkImageView material,
                          VkImageView emissive, VkImageView depth, VkImageView ssao,
                          VkImageView shadow, VkImageView shadowAtlas,
                          VkBuffer clusterLights, VkBuffer clusterGrid) const;

    // --- pass recording (the caller owns all layout transitions) ---------------
    // Draws the scene into the already-begun GBuffer rendering block: static
    // opaque instances as one vkCmdDrawIndexedIndirect per material run over
    // the channel's command buffer (transforms from the scene-set instance
    // SSBO via gl_InstanceIndex), then the skinned instances on the direct
    // push-constant path (CPU frustum-culled with the un-jittered cullViewProj,
    // as before — skinned casts stay off the indirect path this phase).
    // runs/runCount come from this frame's
    // buildInstanceList; the channel's upload (and optional cull pass) must
    // already be recorded.  Descriptor rebinds collapse to one per run.
    void recordGBufferDraws(VkCommandBuffer cmd, const Scene& scene, bool gtPass,
                            VkDescriptorSet sceneSet, VkDescriptorSet textureSet,
                            uint32_t materialStride, uint32_t width, uint32_t height,
                            const Mat4& cullViewProj, const CullChannel& channel,
                            const CullDrawRun* runs, uint32_t runCount) const;

    // --- GPU occlusion culling (Phase 7a; see the CullChannel comment) ---------
    // capacity = upper bound of static opaque instances (scene.instances.size()).
    bool createInstanceBuffer(const VulkanContext& ctx, uint32_t capacity, InstanceBuffer& out) const;
    void destroyInstanceBuffer(const VulkanContext& ctx, InstanceBuffer& buf) const;
    bool createCullChannel(const VulkanContext& ctx, uint32_t capacity, CullChannel& out) const;
    void destroyCullChannel(const VulkanContext& ctx, CullChannel& ch) const;
    // Allocates the channel's cull set from the caller's pool: binding 0 = the
    // shared instance SSBO, 1 = the channel's indirect command buffer, 2 = this
    // path's Hi-Z chain (all mips, GENERAL).  Re-run when the pyramid is
    // recreated (resize) and reset prevValid afterwards.
    bool writeCullSet(const VulkanContext& ctx, VkDescriptorPool pool, const InstanceBuffer& inst,
                      VkImageView hizChain, CullChannel& out) const;
    // Scene set binding 3: the device-local instance SSBO (gbuffer.vert fetches
    // gInstances[gl_InstanceIndex]).  Call once per scene set after the
    // instance buffer exists (same convention as writeSceneSkinBinding).
    void writeSceneInstanceBinding(const VulkanContext& ctx, VkDescriptorSet set,
                                   VkBuffer instances) const;
    // Fills this frame's static-opaque candidate list: CPU frustum cull with
    // the un-jittered cullViewProj plus the LOD state of updateLodSelection
    // (Phase 3b rules, unchanged), grouped into material runs in the scene's
    // (material, mesh) sort order.  Writes up to `capacity` entries into
    // instOut (GpuInstance) and cmdOut (VkDrawIndexedIndirectCommand,
    // instanceCount = 1, firstInstance = candidate index) and returns the
    // candidate count.  Instances whose transform changed this frame
    // (model != prevModel) are flagged occlusion-exempt in GpuInstance::flags.
    uint32_t buildInstanceList(const Scene& scene, const Mat4& cullViewProj, uint32_t capacity,
                               GpuInstance* instOut, VkDrawIndexedIndirectCommand* cmdOut,
                               std::vector<CullDrawRun>& runs) const;
    // Records the slot's staging -> device copy of the instance SSBO plus the
    // barrier making it visible to the cull pass (compute) and the GBuffer
    // vertex stage.  Once per frame, before any path's cull pass or draws.
    void recordInstanceUpload(VkCommandBuffer cmd, uint32_t slot, const InstanceBuffer& buf,
                              uint32_t count) const;
    // Records the slot's staging -> device copy of the channel's indirect
    // command buffer.  culled=true leaves the buffer compute-writable for
    // recordOcclusionCull (which carries the final -> indirect-read barrier);
    // culled=false barriers straight to the indirect-draw stage.
    void recordCommandUpload(VkCommandBuffer cmd, uint32_t slot, const CullChannel& ch,
                             uint32_t count, bool culled) const;
    // Dispatches occlusion_cull.comp on the channel: reprojects each
    // candidate's world AABB with prevViewProj (the exact VP that produced the
    // bound Hi-Z chain) and zeroes the instanceCount of fully occluded
    // commands.  Ends with a compute-write -> indirect-read barrier.
    void recordOcclusionCull(VkCommandBuffer cmd, const CullChannel& ch, uint32_t candidateCount,
                             const Mat4& prevViewProj, uint32_t mipCount, uint32_t width,
                             uint32_t height) const;
    // Fullscreen deferred lighting: GBuffer + IBL -> HDR target.  First runs
    // the cluster light-assignment compute pass (cluster_assign.comp on
    // grid.assignSet[slot], view-space AABB tests) with a
    // compute-write -> fragment-read barrier, then the fullscreen draw.
    // view/proj are this frame's camera matrices (proj may be the jittered
    // variant; the jitter only touches m[8]/m[9], which the pass ignores).
    void recordLightingPass(VkCommandBuffer cmd, VkDescriptorSet lightingSet, ClusterGrid& grid,
                            uint32_t slot, const Mat4& view, const Mat4& proj,
                            VkImageView target, uint32_t width, uint32_t height) const;

    // --- transparency pass (alpha-blended surfaces, after lighting) -----------
    // True when any material in the scene is alphaMode BLEND.
    bool sceneHasTransparency(const Scene& scene) const;
    // Binds the LightingUBO (lights + iblParams are read) + IBL maps + the
    // SSAO texture of this path into a transparentSetLayout() descriptor set.
    // shadow follows the same VK_NULL_HANDLE convention as writeLightingSet
    // (binding 5 left unwritten).  ssrColor is the chain view (all mips) of
    // this path's ColorPyramid and depthPyramid the chain view of its
    // DepthPyramid; ssr.glsl marches the latter and samples the former at a
    // roughness-driven LOD.  fogVolume is the ray-integrated froxel volume of
    // this path (binding 8); pass VK_NULL_HANDLE when volumetric fog is off —
    // an internal identity volume (T=1, I=0) is bound instead so the
    // descriptor is always valid, and the push-constant enable flag gates the
    // sample.
    void writeTransparentSet(const VulkanContext& ctx, VkDescriptorSet set,
                             VkBuffer lightingUbo, VkImageView ssao, VkImageView shadow,
                             VkImageView ssrColor, VkImageView depthPyramid,
                             VkImageView fogVolume) const;
    // Draws all BLEND-material instances back-to-front over the lit scene.
    // LR path (gtPass=false): 3 attachments = color (alpha blend) + motion
    // (overwrite) + reactive mask (additive).  GT path: color only.  The
    // caller begins/ends the rendering block and owns all layout transitions.
    // fogNear/fogFar/fogOn feed the fragment-stage TransparentFogPush block
    // (volumetric fog on translucency; ignored when fogOn is false).
    void recordTransparentDraws(VkCommandBuffer cmd, const Scene& scene, bool gtPass,
                                VkDescriptorSet sceneSet, VkDescriptorSet textureSet,
                                VkDescriptorSet transparentSet, uint32_t materialStride,
                                uint32_t width, uint32_t height, const Mat4& cullViewProj,
                                const Vec3& cameraPos, float fogNear, float fogFar,
                                bool fogOn) const;

    const IblMaps& ibl() const { return ibl_; }

    // Baked local reflection probes (Phase 4c-2; see ibl/Probes.h).  The empty
    // volume (count 0) is created in init() so the lighting/SSR descriptor
    // bindings are always written; hosts call this once after scene load with
    // the registry placements + bake file path (probeFilePathForScene).
    // Without a matching bake file the probe count stays 0 and the shaders
    // fall back to the global environment, unchanged.
    bool loadProbes(const VulkanContext& ctx, const std::vector<ReflectionProbe>& defs,
                    const std::string& filePath) {
        return probes_.load(ctx, defs, filePath);
    }
    const ReflectionProbes& probes() const { return probes_; }

    // --- GTAO pass (between the GBuffer and the lighting pass) ----------------
    // ssao set: AO depth-chain sampler (view-Z mips, GENERAL) + world-normal
    // sampler + working RG16F storage (R=AO, G=|z|).
    void writeSsaoSet(const VulkanContext& ctx, VkDescriptorSet set, VkImageView depthChain,
                      VkImageView normal, VkImageView aoRaw) const;
    // Dispatches ssao.comp (GTAO main pass, XeGTAO High: 3x3 slices/steps with
    // depth-mip step sampling).  The caller owns all layout transitions:
    // normal in SHADER_READ_ONLY, aoRaw in GENERAL; the AO depth chain must
    // already be rebuilt this frame.  invViewProj is the inverse of the exact
    // view-projection used by this path's GBuffer pass (jittered for LR);
    // nearZ/farZ unpack the chain's view Z, maxMipLod caps the step LOD
    // (min(chainMipCount-1, 4), XeGTAO's 5-level working depth).
    void recordSsaoPass(VkCommandBuffer cmd, VkDescriptorSet ssaoSet, const Mat4& invViewProj,
                        uint32_t frameIndex, float nearZ, float farZ, float maxMipLod,
                        uint32_t width, uint32_t height) const;
    // Dispatches ssao_temporal.comp (reprojection + depth rejection + EMA) on
    // the path's AoHistory.  aoRaw must be SHADER_READ_ONLY (main pass done),
    // the GBuffer depth SHADER_READ_ONLY.  Both history barriers (last frame's
    // readers -> storage write -> this frame's blur read) are handled inside.
    // reset = first frame of this path: history is bypassed.  Hosts track
    // prevViewProj per path (jittered for LR, matching invViewProj).
    void recordSsaoTemporalPass(VkCommandBuffer cmd, const AoHistory& history,
                                uint32_t writeIndex, const Mat4& invViewProj,
                                const Mat4& prevViewProj, uint32_t width, uint32_t height,
                                bool reset) const;
    // Dispatches ssao_blur.comp (5x5 depth-aware denoise) on blurSet
    // (AoHistory::blurSet[writeIndex] of this frame): history GENERAL-sampled,
    // ao GENERAL storage.
    void recordSsaoBlurPass(VkCommandBuffer cmd, VkDescriptorSet blurSet, uint32_t width,
                            uint32_t height) const;

    // --- Translucent coverage mask conditioning (after transparency, before upscale)
    // 3x3 max dilate + motion gate (reactive_dilate.comp): the dilated
    // plateau absorbs the sub-pixel straddle of consumers sampling the mask
    // at the jittered coordinate; the motion gate keeps the mask from
    // dropping history where the pixel does not move (reprojection is exact
    // there, and the aliased current frame would otherwise pass through
    // unfiltered and shimmer).  Uses its own 3-binding set layout
    // (src mask sampler, dst storage, motion sampler); the pipeline is
    // created in createPipelines.  The host owns the dilated R16F target.
    // Allocates the dilate set from the caller's pool: srcMask = raw coverage
    // mask (SHADER_READ_ONLY), dstMask = dilated target (GENERAL storage),
    // motion = the path's RG16F motion buffer (SHADER_READ_ONLY).
    bool writeReactiveDilateSet(const VulkanContext& ctx, VkDescriptorPool pool,
                                VkImageView srcMask, VkImageView dstMask, VkImageView motion,
                                VkDescriptorSet& out) const;
    void recordReactiveDilatePass(VkCommandBuffer cmd, VkDescriptorSet set, uint32_t width,
                                  uint32_t height) const;
    // Temporal accumulation state (host-owned; see AoHistory).  Creates the
    // two RG16F ping-pong images (GENERAL for life); descriptor sets are
    // written by writeAoHistorySets once the host's pool exists.
    bool createAoHistory(const VulkanContext& ctx, uint32_t w, uint32_t h, AoHistory& out) const;
    // Allocates temporalSet[2] + blurSet[2] from the caller's pool: temporal
    // binds aoRaw (SHADER_READ_ONLY), the two history buffers and the path's
    // GBuffer depth (SHADER_READ_ONLY); blur binds history + the path's
    // filtered R16F AO target.  The sets are pool-owned; destroyAoHistory
    // only releases the images and views.
    bool writeAoHistorySets(const VulkanContext& ctx, VkDescriptorPool pool, VkImageView aoRaw,
                            VkImageView depth, VkImageView aoOut, AoHistory& out) const;
    void destroyAoHistory(const VulkanContext& ctx, AoHistory& history) const;

    // --- Bloom pyramid (after lighting+transparency, before upscale) ----------
    // Reuses ssaoBlurSetLayout (sampler + storage) for every set.  Creates the
    // kBloomMipCount-level RGBA16F pyramid (mip 0 = half res) and transitions
    // it to GENERAL for life.  Sets are allocated by writeBloomPyramidSets
    // once the host's pool exists.
    bool createBloomPyramid(const VulkanContext& ctx, uint32_t fullW, uint32_t fullH,
                            BloomPyramid& out) const;
    // Allocates extract/downsample/upsample/composite sets from the caller's
    // pool; extract samples srcColor (the lit HDR target, read while in
    // SHADER_READ_ONLY), all mips are bound in GENERAL.
    bool writeBloomPyramidSets(const VulkanContext& ctx, VkDescriptorPool pool,
                               VkImageView srcColor, BloomPyramid& out) const;
    void destroyBloomPyramid(const VulkanContext& ctx, BloomPyramid& pyramid) const;
    // Extract -> 4x downsample -> 4x upsample-accumulate -> composite.  color
    // must be SHADER_READ_ONLY on entry and is left SHADER_READ_ONLY; the
    // pyramid itself needs no layout tracking (GENERAL for life).  No-op when
    // strength <= 0.  The pass barriers against last frame's readers
    // (including the present pass's lens-dirt sample of mip 0).
    void recordBloomPyramidPass(VkCommandBuffer cmd, const BloomPyramid& pyramid, VkImage color,
                                VkImageLayout& colorLayout, uint32_t fullW, uint32_t fullH,
                                float strength = kBloomStrength) const;

    // --- Motion blur + depth of field (Phase 6b; see the constants above) -----
    // Creates the per-path working targets (tile max grid, full-res MB output,
    // half-res CoC color + fg/bg bokeh layers); GENERAL for life via a
    // one-shot transition.  Descriptor sets are written by writePostFxSets
    // once the host's pool exists.
    bool createPostFxTargets(const VulkanContext& ctx, uint32_t w, uint32_t h,
                             PostFxTargets& out) const;
    // Allocates all sets from the caller's pool.  srcColor is the path's lit
    // HDR target (SHADER_READ_ONLY when sampled; the DOF composite also binds
    // it as a storage image for the write-back).  motion/depth are the path's
    // GBuffer motion RT + depth (SHADER_READ_ONLY).  The sets are pool-owned;
    // destroyPostFxTargets only releases images/views.
    bool writePostFxSets(const VulkanContext& ctx, VkDescriptorPool pool, VkImageView srcColor,
                         VkImageView motion, VkImageView depth, PostFxTargets& out) const;
    void destroyPostFxTargets(const VulkanContext& ctx, PostFxTargets& fx) const;
    // Records the enabled post chain on the path's lit HDR target: tile max ->
    // neighbourhood max -> MB gather (into the full-res intermediate) -> DOF
    // coc -> DOF gather -> DOF composite (back into the lit target; when DOF is
    // off but MB ran, a straight copy-back of the MB intermediate instead).  color is
    // SHADER_READ_ONLY on entry and exit (colorLayout tracked by the host);
    // the motion RT and depth must be SHADER_READ_ONLY with compute in the
    // dst scope.  No-op (no barriers, no writes) when both effects are off.
    void recordPostFxPass(VkCommandBuffer cmd, const PostFxTargets& fx, VkImage color,
                          VkImageLayout& colorLayout, const PostFxParams& params,
                          uint32_t frameIndex) const;

    // --- Hi-Z depth pyramid (between the GBuffer and the lighting pass) ------
    // Full mip chain length for w x h (down to 1x1): 1 + floor(log2(max)).
    static uint32_t depthPyramidMipCount(uint32_t w, uint32_t h);
    // Creates the pyramid image + per-mip views and transitions it to GENERAL
    // (one-shot submit).  Descriptor sets are NOT allocated here — call
    // writeDepthPyramidSets once the host's descriptor pool exists.
    // aoFilter=true builds a GTAO chain (linear view-Z, XeGTAO DepthMIPFilter)
    // instead of the default SSR chain (NDC depth, max-reduce); nearZ/farZ
    // feed the NDC->viewZ unpack and are ignored for SSR chains.
    bool createDepthPyramid(const VulkanContext& ctx, uint32_t w, uint32_t h, DepthPyramid& out,
                            bool aoFilter = false, float nearZ = 1.f, float farZ = 100.f) const;
    // Allocates mipCount downsample sets from the caller's pool and writes
    // them: set 0 samples srcDepth (the opaque D32 depth view, read while in
    // SHADER_READ_ONLY), set i>0 samples mip i-1.  The sets are pool-owned;
    // destroyDepthPyramid only releases the image and views.
    bool writeDepthPyramidSets(const VulkanContext& ctx, VkDescriptorPool pool,
                               VkImageView srcDepth, DepthPyramid& out) const;
    void destroyDepthPyramid(const VulkanContext& ctx, DepthPyramid& pyramid) const;
    // Fills mip 0 from the bound source depth, then reduces each mip from the
    // previous (one 8x8 dispatch per level, barriers between; SSR chains
    // max-reduce NDC depth, AO chains run XeGTAO's far-biased DepthMIPFilter
    // on view Z).  The source depth must already be shader-readable by compute
    // (SHADER_READ_ONLY with compute in the dst scope); the pyramid stays
    // GENERAL throughout and the pass ends with a barrier making the writes
    // visible to fragment reads.
    void recordDepthPyramidPass(VkCommandBuffer cmd, const DepthPyramid& pyramid) const;

    // --- HDR color mip chain (after the lighting pass, before transparency) ---
    // Replaces the old full-res transfer copy of the lit HDR target: mip 0 of
    // the chain IS the opaque color copy, so glass gets the sharp reflection
    // and every blurred level from one pass.  The downsample reuses the Hi-Z
    // descriptor set layout / pipeline layout (identical binding shape and
    // HiZPush), only the shader differs.
    bool createColorPyramid(const VulkanContext& ctx, uint32_t w, uint32_t h,
                            ColorPyramid& out) const;
    // Allocates mipCount downsample sets from the caller's pool: set 0 samples
    // srcColor (the lit HDR target, read while in SHADER_READ_ONLY), set i>0
    // samples mip i-1.  The sets are pool-owned; destroyColorPyramid only
    // releases the image and views.
    bool writeColorPyramidSets(const VulkanContext& ctx, VkDescriptorPool pool,
                               VkImageView srcColor, ColorPyramid& out) const;
    void destroyColorPyramid(const VulkanContext& ctx, ColorPyramid& pyramid) const;
    // Copies the bound source color into mip 0, then box-averages each mip
    // from the previous (one 8x8 dispatch per level, barriers between).  Same
    // host contract as recordDepthPyramidPass: the source must already be
    // shader-readable by compute (SHADER_READ_ONLY with compute in the dst
    // scope); the chain stays GENERAL throughout and the pass ends with a
    // barrier making the writes visible to fragment reads.
    void recordColorPyramidPass(VkCommandBuffer cmd, const ColorPyramid& pyramid) const;

    // --- Opaque SSR (after lighting + color pyramid, before transparency) -----
    // Phase 2d: two passes instead of the old single in-place pass.
    //   1. Trace (ssr_opaque.comp): full-screen Hi-Z march per opaque pixel;
    //      writes the composite delta conf*(ssrColor*envBRDF - specIbl) plus
    //      the pixel's view |z| into a full-res RGBA16F trace target.
    //   2. Temporal + composite (ssr_temporal.comp, see SsrHistory): EMA-
    //      accumulates the delta and adds the filtered result to the lit HDR
    //      target (energy-conserving IBL-specular replacement, same maths as
    //      the old in-place pass on the first frame).
    // ssr set: binding 0 = the path's LightingUBO (only the invViewProj /
    // cameraPos / iblParams prefix is read); 1-4 = GBuffer albedo/normal/
    // material/depth; 5 = SSAO; 6-7 = IBL prefilter + BRDF LUT; 8 = color
    // pyramid chain (GENERAL); 9 = depth pyramid chain (GENERAL); 10 = the
    // path's trace target as a write-only storage image (GENERAL).
    void writeSsrSet(const VulkanContext& ctx, VkDescriptorSet set, VkBuffer lightingUbo,
                     VkImageView albedo, VkImageView normal, VkImageView material,
                     VkImageView depth, VkImageView ssao, VkImageView ssrColor,
                     VkImageView depthPyramid, VkImageView ssrTraceOut) const;
    // Dispatches ssr_opaque.comp (trace stage).  viewProj must be the exact
    // view-projection of this path's GBuffer pass (jittered for LR).  The
    // caller owns all layout transitions: the trace target in GENERAL on
    // entry, the GBuffer + SSAO textures shader-readable by compute, both
    // pyramids already rebuilt.
    void recordSsrPass(VkCommandBuffer cmd, VkDescriptorSet ssrSet, const Mat4& viewProj,
                       uint32_t width, uint32_t height) const;
    // Temporal accumulation state (host-owned; see SsrHistory).  Creates the
    // two RGBA16F ping-pong images (GENERAL for life); descriptor sets are
    // written by writeSsrHistorySets once the host's pool exists.
    bool createSsrHistory(const VulkanContext& ctx, uint32_t w, uint32_t h, SsrHistory& out) const;
    // Allocates temporalSet[2] from the caller's pool: binds the path's trace
    // target (SHADER_READ_ONLY), the two history buffers, the path's GBuffer
    // depth (SHADER_READ_ONLY) and the lit HDR target (GENERAL storage RMW).
    // The sets are pool-owned; destroySsrHistory only releases images/views.
    bool writeSsrHistorySets(const VulkanContext& ctx, VkDescriptorPool pool, VkImageView ssrTrace,
                             VkImageView depth, VkImageView sceneColor, SsrHistory& out) const;
    void destroySsrHistory(const VulkanContext& ctx, SsrHistory& history) const;
    // Dispatches ssr_temporal.comp (reprojection + depth rejection + 3x3
    // neighbourhood clamp + EMA, fused composite RMW on the lit HDR target).
    // The trace target must be SHADER_READ_ONLY (trace pass done), the GBuffer
    // depth SHADER_READ_ONLY, sceneColor GENERAL.  Both history barriers
    // (last frame's readers -> storage write -> next frame's temporal read)
    // are handled inside.  reset = first frame of this path: history is
    // bypassed.  Hosts track prevViewProj per path (jittered for LR, matching
    // invViewProj).  Reuses the SsaoTemporalPush layout.
    void recordSsrTemporalPass(VkCommandBuffer cmd, const SsrHistory& history,
                               uint32_t writeIndex, const Mat4& invViewProj,
                               const Mat4& prevViewProj, uint32_t width, uint32_t height,
                               bool reset) const;

    // --- Froxel volumetric fog (Phase 5a; see the constants above) ------------
    // Creates the five RGBA16F 3D volumes (inject / raw-lit / history x2 /
    // integrated) for a path of resolution w x h; GENERAL for life via a
    // one-shot transition.  Descriptor sets are written by writeVolFogSets
    // once the host's pool exists.
    bool createVolFogVolume(const VulkanContext& ctx, uint32_t w, uint32_t h,
                            VolFogVolume& out) const;
    // Allocates all sets from the caller's pool: inject (storage only); light
    // per slot (LightingUBO + this path's ClusterGrid SSBOs + CSM/spot shadow
    // maps + inject volume + raw-lit out); temporal x2; march x2; composite
    // (lit HDR target RMW + GBuffer depth + integrated volume).  shadowMap /
    // shadowAtlas follow the writeLightingSet VK_NULL_HANDLE convention (the
    // binding stays unwritten; shadowParams.z == 0 short-circuits the sample).
    bool writeVolFogSets(const VulkanContext& ctx, VkDescriptorPool pool, VolFogVolume& fog,
                         const ClusterGrid& cluster, const VkBuffer* lightingUbos,
                         VkImageView shadowMap, VkImageView shadowAtlas, VkImageView depth,
                         VkImageView sceneColor) const;
    void destroyVolFogVolume(const VulkanContext& ctx, VolFogVolume& fog) const;
    // Records inject -> light -> temporal -> march with internal barriers.
    // view/proj are the UN-JITTERED camera matrices (the volume stays stable
    // under TAA jitter); prevViewProj is last frame's proj*view of this path.
    // The cluster assignment for this frame/slot must already be recorded
    // (recordLightingPass does it) and the CSM/spot maps must be
    // shader-readable.  writeIndex is the temporal ping-pong index (the
    // host's per-path fog frame counter & 1); reset = first frame of this
    // path (history bypassed).
    void recordVolFogAccumulate(VkCommandBuffer cmd, VolFogVolume& fog,
                                const ClusterGrid& cluster, uint32_t slot, const Mat4& view,
                                const Mat4& proj, const Mat4& prevViewProj,
                                const VolFogParams& params, uint32_t frameIndex,
                                uint32_t writeIndex, bool reset) const;
    // In-place composite on the lit HDR target (per-texel RMW, same contract
    // as ssr_temporal.comp): color * transmittance + inscatter.  The lit
    // target must be GENERAL (storage RMW) and the GBuffer depth
    // SHADER_READ_ONLY on entry; proj supplies the depth unpack (m[10]/m[14])
    // of this path's projection, fogFar the volume's far range.
    void recordVolFogComposite(VkCommandBuffer cmd, const VolFogVolume& fog, const Mat4& proj,
                               float fogFar, uint32_t width, uint32_t height) const;

    // --- Auto exposure (after lighting + bloom, before upscale/present) -------
    // Creates the histogram + state buffers; initialEV seeds the smoothed EV
    // (pick -log2(initialExposure) so the first frames match the manual look
    // until the first readback arrives).
    bool createAutoExposure(const VulkanContext& ctx, float initialEV, AutoExposure& out) const;
    // Allocates the shared set from the caller's pool and binds srcColor (the
    // lit HDR target, sampled in SHADER_READ_ONLY) + the two buffers.
    bool writeAutoExposureSet(const VulkanContext& ctx, VkDescriptorPool pool,
                              VkImageView srcColor, AutoExposure& out) const;
    void destroyAutoExposure(const VulkanContext& ctx, AutoExposure& ae) const;
    // Histogram pass + solve pass, then copies the solved ExposureState into
    // readbackDst (TRANSFER_DST staging, may be VK_NULL_HANDLE).  The source
    // must already be shader-readable by compute (SHADER_READ_ONLY with
    // compute in the dst scope); the caller sets solve.push fields and marks
    // the readback pending for its slot.
    void recordAutoExposurePass(VkCommandBuffer cmd, const AutoExposure& ae,
                                const ExposureSolvePush& solve, VkBuffer readbackDst) const;

    // ExposureChannel helpers (all hosts share this exact plumbing).
    // createExposureChannel = createAutoExposure + writeAutoExposureSet +
    // per-slot readback staging; initialEV seeds the smoothed EV (pass
    // -log2(initialExposure) so early frames match the manual look).
    bool createExposureChannel(const VulkanContext& ctx, VkDescriptorPool pool,
                               VkImageView srcColor, uint32_t srcW, uint32_t srcH,
                               float initialEV, ExposureChannel& out) const;
    void destroyExposureChannel(const VulkanContext& ctx, ExposureChannel& channel) const;
    // Call after the slot's fence: adopts the solved exposure if this slot's
    // readback was recorded.
    void harvestExposureChannel(ExposureChannel& channel, uint32_t slot) const;

    // --- CSM sun shadow pass (between the GBuffer and the lighting pass) ------
    // Creates the 2048^2 x kShadowCascadeCount D32 array + views.  The
    // comparison sampler is shared (shadowSampler()); it is created in init().
    bool createShadowTargets(const VulkanContext& ctx, ShadowTargets& out) const;
    void destroyShadowTargets(const VulkanContext& ctx, ShadowTargets& targets) const;
    // Computes the 4 cascade view-projections for the given camera/sun.
    // Practical split (lambda = kShadowSplitLambda) over [near, min(far,
    // kShadowMaxDistance)]; each cascade covers the bounding sphere of its
    // frustum slice with a quantized radius, and the sphere centre is snapped
    // to a constant light-space texel grid (kShadowRadiusSnap / kShadowZSnap)
    // so the shadow map does not shimmer when the camera moves.  outSplitViewDepth[i]
    // = far boundary of cascade i (positive view-space metres).  Vulkan
    // conventions throughout (y-flipped projection, [0,1] depth).
    static void computeCascadeVPs(const Camera& cam, float aspect, const Vec3& sunDirTowardLight,
                                  Mat4 outVP[kShadowCascadeCount],
                                  float outSplitViewDepth[kShadowCascadeCount]);
    // Renders the scene depth into each cascade layer (one dynamic-rendering
    // block per cascade, cleared to 1.0).  BLEND materials are skipped (they
    // do not occlude); MASK materials alpha-discard in shadow_depth.frag.
    // Reuses the scene pipeline layout: sceneSet (material UBO) + textureSet.
    // The caller owns the image layout transitions (DEPTH_STENCIL_ATTACHMENT
    // before, SHADER_READ_ONLY after) — record after the GBuffer pass and
    // before the lighting pass of the active path.
    void recordShadowPass(VkCommandBuffer cmd, const ShadowTargets& targets, const Scene& scene,
                          const Mat4 cascadeVp[kShadowCascadeCount], VkDescriptorSet sceneSet,
                          VkDescriptorSet textureSet, uint32_t materialStride) const;

    // --- Spot light shadow atlas (Phase 4b; see the constants above) -----------
    // Creates the 4096^2 D32 atlas + the whole-atlas view.  The comparison
    // sampler is shared with the CSM pass (shadowSampler()).
    bool createShadowAtlas(const VulkanContext& ctx, ShadowAtlas& out) const;
    void destroyShadowAtlas(const VulkanContext& ctx, ShadowAtlas& atlas) const;
    // Deterministic per-frame tile assignment: scores every spot light with
    // castShadow by intensity / distance^2 to the camera, stable-sorts by
    // score descending (ties keep the lower scene index) and writes
    // shadowIndex = tile for the top kShadowAtlasTiles lights; every other
    // light's shadowIndex is reset to -1 (point lights are never selected
    // this phase — see the atlas comment above).  Mutates the passed list, so
    // hosts copy the scene list first; returns the number of tiles assigned.
    // The same frame state always produces the same assignment on every host.
    static uint32_t selectSpotShadowLights(std::vector<Light>& lights, const Vec3& cameraPos);
    // Spot shadow view-projection of one light: perspective from the light
    // position along spotDirection, fov = 2 * outerConeAngle + margin, far =
    // range (kSpotShadowInfiniteRange when 0).  Camera-independent, so the
    // map needs no texel snapping to stay stable frame-to-frame.
    static Mat4 computeSpotShadowVp(const Light& light);
    // Renders the scene depth of every selected light into its atlas tile:
    // one dynamic-rendering block per tile (cleared to 1.0), renderArea +
    // viewport/scissor restricted to the tile, per-tile frustum cull.  Same
    // pipelines, depth bias and material rules as recordShadowPass (BLEND
    // skipped, MASK alpha-discards, skinned casters included).  The caller
    // owns the atlas layout transitions (DEPTH_STENCIL_ATTACHMENT before,
    // SHADER_READ_ONLY after).
    void recordSpotShadowPass(VkCommandBuffer cmd, const ShadowAtlas& atlas, const Scene& scene,
                              const Mat4* tileVp, uint32_t tileCount, VkDescriptorSet sceneSet,
                              VkDescriptorSet textureSet, uint32_t materialStride) const;

    VkDescriptorSetLayout sceneSetLayout() const { return sceneSetLayout_; }
    VkDescriptorSetLayout textureSetLayout() const { return textureSetLayout_; }
    VkDescriptorSetLayout lightingSetLayout() const { return lightingSetLayout_; }
    VkDescriptorSetLayout transparentSetLayout() const { return transparentSetLayout_; }
    VkDescriptorSetLayout ssaoSetLayout() const { return ssaoSetLayout_; }
    VkDescriptorSetLayout ssaoBlurSetLayout() const { return ssaoBlurSetLayout_; }
    VkDescriptorSetLayout hizSetLayout() const { return hizSetLayout_; }
    VkDescriptorSetLayout ssrSetLayout() const { return ssrSetLayout_; }
    VkDescriptorSetLayout ssrTemporalSetLayout() const { return ssrTemporalSetLayout_; }
    VkDescriptorSetLayout clusterSetLayout() const { return clusterSetLayout_; }
    VkDescriptorSetLayout exposureSetLayout() const { return exposureSetLayout_; }
    VkPipelineLayout scenePipelineLayout() const { return scenePipelineLayout_; }
    VkPipelineLayout lightingPipelineLayout() const { return lightingPipelineLayout_; }
    VkPipeline gbufferPipeline() const { return gbufferPipeline_; }
    VkPipeline gbufferGtPipeline() const { return gbufferGtPipeline_; }
    VkPipeline lightingPipeline() const { return lightingPipeline_; }
    VkShaderModule fullscreenVert() const { return fullscreenVert_; }
    VkSampler textureSampler() const { return textureSampler_; }   // aniso + mipmaps
    VkSampler gbufferSampler() const { return gbufferSampler_; }   // linear clamp
    VkSampler shadowSampler() const { return shadowSampler_; }     // depth compare (LESS_OR_EQUAL)

private:
    bool loadShader(const VulkanContext& ctx, const char* name, VkShaderModule& out);
    bool createLayouts(const VulkanContext& ctx);
    bool createPipelines(const VulkanContext& ctx);
    // Shared draw loop of the shadow depth passes (CSM cascades and spot
    // atlas tiles): draws every frustum-visible opaque/skinned caster with the
    // given light view-projection into the currently open rendering block.
    void recordShadowDraws(VkCommandBuffer cmd, const Scene& scene, const Mat4& lightVp,
                           VkDescriptorSet sceneSet, VkDescriptorSet textureSet,
                           uint32_t materialStride) const;

    IblMaps ibl_;
    ReflectionProbes probes_; // baked local reflection captures (empty until loadProbes)
    // Procedural sky atmosphere (Hillaire 2020): LUTs baked once when init()
    // runs without an env map file; ibl_ then renders + re-renders the sky
    // from them.  atmosphereSky_ mirrors ibl().fromAtmosphere.
    SkyAtmosphere sky_;
    bool atmosphereSky_ = false;

    VkDescriptorSetLayout sceneSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout textureSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout lightingSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout transparentSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout ssaoSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout ssaoBlurSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout ssaoTemporalSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout hizSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout ssrSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout ssrTemporalSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout clusterSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout clusterPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline clusterPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout scenePipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout lightingPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout transparentPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout ssaoPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout ssaoBlurPipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout reactiveDilateSetLayout_ = VK_NULL_HANDLE; // mask + dst + motion
    VkPipelineLayout reactiveDilatePipelineLayout_ = VK_NULL_HANDLE; // uses reactiveDilateSetLayout_
    VkPipelineLayout ssaoTemporalPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline gbufferPipeline_ = VK_NULL_HANDLE;
    VkPipeline gbufferGtPipeline_ = VK_NULL_HANDLE;
    VkPipeline gbufferSkinnedPipeline_ = VK_NULL_HANDLE;   // gbuffer_skinned.vert, motion output
    VkPipeline gbufferSkinnedGtPipeline_ = VK_NULL_HANDLE; // GT variant (no motion attachment)
    VkPipeline lightingPipeline_ = VK_NULL_HANDLE;
    VkPipeline transparentPipeline_ = VK_NULL_HANDLE;
    VkPipeline transparentGtPipeline_ = VK_NULL_HANDLE;
    VkPipeline ssaoPipeline_ = VK_NULL_HANDLE;
    VkPipeline ssaoBlurPipeline_ = VK_NULL_HANDLE;
    VkPipeline reactiveDilatePipeline_ = VK_NULL_HANDLE;
    VkPipeline ssaoTemporalPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout hizPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline hizPipeline_ = VK_NULL_HANDLE;
    VkPipeline colorDownsamplePipeline_ = VK_NULL_HANDLE; // reuses hizPipelineLayout_
    VkPipelineLayout ssrPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline ssrPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout ssrTemporalPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline ssrTemporalPipeline_ = VK_NULL_HANDLE;
    // Froxel volumetric fog (Phase 5a): one set layout + pipeline per pass.
    VkDescriptorSetLayout volfogInjectSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout volfogLightSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout volfogTemporalSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout volfogMarchSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout volfogCompositeSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout volfogInjectPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout volfogLightPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout volfogTemporalPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout volfogMarchPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout volfogCompositePipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline volfogInjectPipeline_ = VK_NULL_HANDLE;
    VkPipeline volfogLightPipeline_ = VK_NULL_HANDLE;
    VkPipeline volfogTemporalPipeline_ = VK_NULL_HANDLE;
    VkPipeline volfogMarchPipeline_ = VK_NULL_HANDLE;
    VkPipeline volfogCompositePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout bloomPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline bloomExtractPipeline_ = VK_NULL_HANDLE;
    VkPipeline bloomDownsamplePipeline_ = VK_NULL_HANDLE;
    VkPipeline bloomUpsamplePipeline_ = VK_NULL_HANDLE;
    VkPipeline bloomCompositePipeline_ = VK_NULL_HANDLE;
    // Motion blur + DOF (Phase 6b): one set layout / pipeline per pass.
    VkDescriptorSetLayout mbTileSetLayout_ = VK_NULL_HANDLE;      // also neighbour max
    VkDescriptorSetLayout mbGatherSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout dofCocSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout dofGatherSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout dofCompositeSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout postFxCopybackSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout mbTilePipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout mbGatherPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout dofCocPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout dofGatherPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout dofCompositePipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout postFxCopybackPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline mbTilePipeline_ = VK_NULL_HANDLE;
    VkPipeline mbNeighborPipeline_ = VK_NULL_HANDLE;
    VkPipeline mbGatherPipeline_ = VK_NULL_HANDLE;
    VkPipeline dofCocPipeline_ = VK_NULL_HANDLE;
    VkPipeline dofGatherPipeline_ = VK_NULL_HANDLE;
    VkPipeline dofCompositePipeline_ = VK_NULL_HANDLE;
    VkPipeline postFxCopybackPipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout exposureSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout cullSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout cullPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline cullPipeline_ = VK_NULL_HANDLE;
    uint32_t maxDrawIndirectCount_ = 0xFFFFu; // capped from device limits in init()
    VkPipelineLayout histogramPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout exposureSolvePipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline histogramPipeline_ = VK_NULL_HANDLE;
    VkPipeline exposureSolvePipeline_ = VK_NULL_HANDLE;
    VkPipeline shadowPipeline_ = VK_NULL_HANDLE;
    VkPipeline shadowSkinnedPipeline_ = VK_NULL_HANDLE;
    VkShaderModule gbufferVert_ = VK_NULL_HANDLE;
    VkShaderModule gbufferSkinnedVert_ = VK_NULL_HANDLE;
    VkShaderModule gbufferFrag_ = VK_NULL_HANDLE;
    VkShaderModule gbufferGtFrag_ = VK_NULL_HANDLE;
    VkShaderModule lightingFrag_ = VK_NULL_HANDLE;
    VkShaderModule fullscreenVert_ = VK_NULL_HANDLE;
    VkShaderModule transparentVert_ = VK_NULL_HANDLE;
    VkShaderModule transparentFrag_ = VK_NULL_HANDLE;
    VkShaderModule transparentGtFrag_ = VK_NULL_HANDLE;
    VkShaderModule ssaoComp_ = VK_NULL_HANDLE;
    VkShaderModule ssaoBlurComp_ = VK_NULL_HANDLE;
    VkShaderModule reactiveDilateComp_ = VK_NULL_HANDLE;
    VkShaderModule ssaoTemporalComp_ = VK_NULL_HANDLE;
    VkShaderModule hizDownsampleComp_ = VK_NULL_HANDLE;
    VkShaderModule colorDownsampleComp_ = VK_NULL_HANDLE;
    VkShaderModule ssrOpaqueComp_ = VK_NULL_HANDLE;
    VkShaderModule ssrTemporalComp_ = VK_NULL_HANDLE;
    VkShaderModule volfogInjectComp_ = VK_NULL_HANDLE;
    VkShaderModule volfogLightComp_ = VK_NULL_HANDLE;
    VkShaderModule volfogTemporalComp_ = VK_NULL_HANDLE;
    VkShaderModule volfogMarchComp_ = VK_NULL_HANDLE;
    VkShaderModule volfogCompositeComp_ = VK_NULL_HANDLE;
    VkShaderModule bloomExtractComp_ = VK_NULL_HANDLE;
    VkShaderModule bloomDownsampleComp_ = VK_NULL_HANDLE;
    VkShaderModule bloomUpsampleComp_ = VK_NULL_HANDLE;
    VkShaderModule bloomCompositeComp_ = VK_NULL_HANDLE;
    VkShaderModule motionBlurTilemaxComp_ = VK_NULL_HANDLE;
    VkShaderModule motionBlurNeighborhoodComp_ = VK_NULL_HANDLE;
    VkShaderModule motionBlurGatherComp_ = VK_NULL_HANDLE;
    VkShaderModule dofCocComp_ = VK_NULL_HANDLE;
    VkShaderModule dofGatherComp_ = VK_NULL_HANDLE;
    VkShaderModule dofCompositeComp_ = VK_NULL_HANDLE;
    VkShaderModule postFxCopybackComp_ = VK_NULL_HANDLE;
    VkShaderModule exposureHistogramComp_ = VK_NULL_HANDLE;
    VkShaderModule exposureSolveComp_ = VK_NULL_HANDLE;
    VkShaderModule shadowDepthVert_ = VK_NULL_HANDLE;
    VkShaderModule shadowDepthSkinnedVert_ = VK_NULL_HANDLE;
    VkShaderModule shadowDepthFrag_ = VK_NULL_HANDLE;
    VkShaderModule clusterAssignComp_ = VK_NULL_HANDLE;
    VkShaderModule occlusionCullComp_ = VK_NULL_HANDLE;
    VkSampler textureSampler_ = VK_NULL_HANDLE;
    VkSampler gbufferSampler_ = VK_NULL_HANDLE;
    VkSampler shadowSampler_ = VK_NULL_HANDLE;
    VkSampler hizSampler_ = VK_NULL_HANDLE; // nearest + clamp (texelFetch pyramid reads)
    VkSampler colorPyramidSampler_ = VK_NULL_HANDLE; // trilinear + clamp (SSR roughness LOD reads)
    // 1x1x1 identity froxel volume (T=1, I=0), GENERAL for life: fallback for
    // the transparency set's binding 8 when volumetric fog is off, so the
    // statically-used sampler always has a valid descriptor.
    VkImage fogFallbackImage_ = VK_NULL_HANDLE;
    VmaAllocation fogFallbackMemory_ = VK_NULL_HANDLE;
    VkImageView fogFallbackView_ = VK_NULL_HANDLE;
};

} // namespace sr
