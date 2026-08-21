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
#include "renderer/math/Math.h"

#include <cstdint>
#include <vector>

namespace sr {

class Scene;
class Camera;
struct Light;

// Default equirect HDR environment map (Bistro san_giuseppe_bridge, bundled
// in the project's assets; override with --env-map).
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

// GPU mirror of scene::Light (std140: three vec4s per light).  Must match the
// LightGPU struct in lighting.frag / transparent.frag / transparent_gt.frag.
struct LightGPU {
    float posOrDir[4]; // xyz = position (point) / direction-to-light (directional), w = LightType
    float color[4];    // rgb + w = intensity (PI-scaled, see fillLightingUBO)
    float params[4];   // x = range (0 = infinite), y = castShadow (C2), zw = reserved
};
static_assert(sizeof(LightGPU) == 48, "LightGPU std140 size mismatch");

constexpr uint32_t kMaxLights = 16; // must match lights[] in lighting/transparent shaders

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

// Per-frame cascade data fed into fillLightingUBO.  splitDepth[i] is the far
// boundary of cascade i in view-space depth (positive metres).
struct ShadowFrame {
    Mat4 cascadeVp[kShadowCascadeCount];
    float splitDepth[kShadowCascadeCount] = {};
    int32_t lightIndex = -1;   // index into LightingUBO::lights of the shadowed sun
    bool debugCascades = false;
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
};
static_assert(sizeof(LightingUBO) == 1200, "LightingUBO std140 size mismatch");

struct ScenePush {
    float model[16];
    float prevModel[16];
    float normalModel[16]; // transpose(inverse(mat3(model))), upper 3x3 used
};
static_assert(sizeof(ScenePush) == 192, "ScenePush size mismatch");

// Push constants of the shadow depth pass (shadow_depth.vert): per-instance
// model + the cascade's light view-projection.  128 bytes fits the guaranteed
// maxPushConstantsSize minimum, and the pass reuses scenePipelineLayout()
// (whose 192-byte range covers it).
struct ShadowPush {
    float model[16];
    float lightVp[16];
};
static_assert(sizeof(ShadowPush) == 128, "ShadowPush size mismatch");

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

// --- HDR color mip chain (box-filtered lit-color pyramid) --------------------
// Roughness-aware SSR (Phase 1b-2) samples this chain at lod = roughness *
// (mipCount - 1) instead of a single sharp mip-0 read, so one ray
// approximates the widened GGX lobe; Phase 6's bloom pyramid is expected to
// reuse the same resource.  RGBA16F like the lighting target, mip 0 =
// straight copy of the lit opaque HDR color, mips 1..N = 2x2 box average.
// The chain length rule matches the depth pyramid (full chain down to 1x1).
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

// Bloom (half-res extract + separable Gaussian, added back before upscale).
struct BloomPush {
    float params[4]; // extract: threshold, knee; blur: dir.xy; composite: strength
};
static_assert(sizeof(BloomPush) == 16, "BloomPush size mismatch");
constexpr float kBloomThreshold = 1.0f;
constexpr float kBloomKnee = 0.5f;
constexpr float kBloomStrength = 0.15f;

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
    // Builds the IBL maps (envMapPath empty/unreadable -> procedural gradient
    // fallback inside IblMaps), loads the deferred shaders and creates all
    // layouts/pipelines/samplers.
    bool init(const VulkanContext& ctx, const char* envMapPath);
    void destroy(const VulkanContext& ctx);

    // --- UBO fillers (shared defaults: fallback lights, ambient, PI scaling) ---
    void fillSceneUBO(SceneUBO& out, const Scene& scene, const Camera& camera,
                      const Mat4& view, const Mat4& proj, const Mat4& projJittered,
                      const Mat4& prevViewProj, uint32_t renderW, uint32_t renderH,
                      float jitterX, float jitterY, bool jitter) const;
    // overrideLights (optional): pack this list instead of scene.lights — the
    // GUI sun controls rebuild the list per frame without touching the scene.
    // The fallback-to-defaultLights() rule applies only when no override is
    // given.  Packed order: shadowed sun first, then other lights scored by
    // intensity/distance² (truncated at kMaxLights).  The GPU sun index is
    // remapped to slot 0.
    // shadow (optional): non-null enables CSM sampling in the shaders
    // (shadowParams.z = 1); null writes identity cascades with shadows off.
    void fillLightingUBO(LightingUBO& out, const Scene& scene, const Camera& camera,
                         const Mat4& invViewProj,
                         const std::vector<Light>* overrideLights = nullptr,
                         const ShadowFrame* shadow = nullptr,
                         float iblIntensity = 1.f) const;

    // Uploads the dynamic-offset material UBO array (one entry per material).
    bool createMaterialUbo(const VulkanContext& ctx, const Scene& scene, VkBuffer& buffer,
                           VmaAllocation& memory, uint32_t& stride) const;

    // --- descriptor writers (sets are allocated from the caller's pool) --------
    void writeTextureSet(const VulkanContext& ctx, VkDescriptorSet set, const Scene& scene) const;
    // shadow = ShadowTargets::arrayView, or VK_NULL_HANDLE to leave binding 11
    // unwritten (hosts without a shadow pass; the shaders must run with
    // shadowParams.z == 0 then, which fillLightingUBO guarantees by default).
    void writeLightingSet(const VulkanContext& ctx, VkDescriptorSet set, VkBuffer lightingUbo,
                          VkImageView albedo, VkImageView normal, VkImageView material,
                          VkImageView emissive, VkImageView depth, VkImageView ssao,
                          VkImageView shadow) const;

    // --- pass recording (the caller owns all layout transitions) ---------------
    // Draws the scene into the already-begun GBuffer rendering block (merged
    // scene-wide buffers, frustum culled with the un-jittered view-projection,
    // descriptor rebinds collapsed per material run).
    void recordGBufferDraws(VkCommandBuffer cmd, const Scene& scene, bool gtPass,
                            VkDescriptorSet sceneSet, VkDescriptorSet textureSet,
                            uint32_t materialStride, uint32_t width, uint32_t height,
                            const Mat4& cullViewProj) const;
    // Fullscreen deferred lighting: GBuffer + IBL -> HDR target.
    void recordLightingPass(VkCommandBuffer cmd, VkDescriptorSet lightingSet, VkImageView target,
                            uint32_t width, uint32_t height) const;

    // --- transparency pass (alpha-blended surfaces, after lighting) -----------
    // True when any material in the scene is alphaMode BLEND.
    bool sceneHasTransparency(const Scene& scene) const;
    // Binds the LightingUBO (lights + iblParams are read) + IBL maps + the
    // SSAO texture of this path into a transparentSetLayout() descriptor set.
    // shadow follows the same VK_NULL_HANDLE convention as writeLightingSet
    // (binding 5 left unwritten).  ssrColor is the chain view (all mips) of
    // this path's ColorPyramid and depthPyramid the chain view of its
    // DepthPyramid; ssr.glsl marches the latter and samples the former at a
    // roughness-driven LOD.
    void writeTransparentSet(const VulkanContext& ctx, VkDescriptorSet set,
                             VkBuffer lightingUbo, VkImageView ssao, VkImageView shadow,
                             VkImageView ssrColor, VkImageView depthPyramid) const;
    // Draws all BLEND-material instances back-to-front over the lit scene.
    // LR path (gtPass=false): 3 attachments = color (alpha blend) + motion
    // (overwrite) + reactive mask (additive).  GT path: color only.  The
    // caller begins/ends the rendering block and owns all layout transitions.
    void recordTransparentDraws(VkCommandBuffer cmd, const Scene& scene, bool gtPass,
                                VkDescriptorSet sceneSet, VkDescriptorSet textureSet,
                                VkDescriptorSet transparentSet, uint32_t materialStride,
                                uint32_t width, uint32_t height, const Mat4& cullViewProj,
                                const Vec3& cameraPos) const;

    const IblMaps& ibl() const { return ibl_; }

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

    // --- Bloom (after lighting+transparency, before upscale) ------------------
    // Reuses ssaoBlurSetLayout (sampler + storage).  writeBloomSet binds src
    // (sampled) and dst (GENERAL storage).
    void writeBloomSet(const VulkanContext& ctx, VkDescriptorSet set, VkImageView src,
                       VkImageView dst) const;
    // Extract + H blur + V blur + composite.  colorLayout must be
    // SHADER_READ_ONLY on entry; it is left SHADER_READ_ONLY.  bloomA/B layouts
    // are tracked across frames.  No-op when strength <= 0.
    void recordBloomPass(VkCommandBuffer cmd, VkDescriptorSet extractSet, VkDescriptorSet blurHSet,
                         VkDescriptorSet blurVSet, VkDescriptorSet compositeSet, VkImage bloomA,
                         VkImage bloomB, VkImage color, VkImageLayout& bloomALayout,
                         VkImageLayout& bloomBLayout, VkImageLayout& colorLayout, uint32_t fullW,
                         uint32_t fullH, float strength = kBloomStrength) const;

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
    // Full-screen compute pass (ssr_opaque.comp): Hi-Z march per opaque pixel,
    // composited energy-conservingly by replacing the IBL specular term the
    // lighting pass already wrote (mix(iblSpec, ssrColor*envBRDF, conf)).
    // ssr set: binding 0 = the path's LightingUBO (only the invViewProj /
    // cameraPos / iblParams prefix is read); 1-4 = GBuffer albedo/normal/
    // material/depth; 5 = SSAO; 6-7 = IBL prefilter + BRDF LUT; 8 = color
    // pyramid chain (GENERAL); 9 = depth pyramid chain (GENERAL); 10 = the lit
    // HDR target as a storage image (in-place read-modify-write, GENERAL).
    void writeSsrSet(const VulkanContext& ctx, VkDescriptorSet set, VkBuffer lightingUbo,
                     VkImageView albedo, VkImageView normal, VkImageView material,
                     VkImageView depth, VkImageView ssao, VkImageView ssrColor,
                     VkImageView depthPyramid, VkImageView sceneColor) const;
    // Dispatches ssr_opaque.comp.  viewProj must be the exact view-projection
    // of this path's GBuffer pass (jittered for LR).  The caller owns all
    // layout transitions: sceneColor in GENERAL on entry, the GBuffer + SSAO
    // textures shader-readable by compute, both pyramids already rebuilt.
    void recordSsrPass(VkCommandBuffer cmd, VkDescriptorSet ssrSet, const Mat4& viewProj,
                       uint32_t width, uint32_t height) const;

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

    VkDescriptorSetLayout sceneSetLayout() const { return sceneSetLayout_; }
    VkDescriptorSetLayout textureSetLayout() const { return textureSetLayout_; }
    VkDescriptorSetLayout lightingSetLayout() const { return lightingSetLayout_; }
    VkDescriptorSetLayout transparentSetLayout() const { return transparentSetLayout_; }
    VkDescriptorSetLayout ssaoSetLayout() const { return ssaoSetLayout_; }
    VkDescriptorSetLayout ssaoBlurSetLayout() const { return ssaoBlurSetLayout_; }
    VkDescriptorSetLayout hizSetLayout() const { return hizSetLayout_; }
    VkDescriptorSetLayout ssrSetLayout() const { return ssrSetLayout_; }
    VkDescriptorSetLayout bloomSetLayout() const { return ssaoBlurSetLayout_; }
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

    IblMaps ibl_;

    VkDescriptorSetLayout sceneSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout textureSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout lightingSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout transparentSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout ssaoSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout ssaoBlurSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout ssaoTemporalSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout hizSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout ssrSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout scenePipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout lightingPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout transparentPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout ssaoPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout ssaoBlurPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout ssaoTemporalPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline gbufferPipeline_ = VK_NULL_HANDLE;
    VkPipeline gbufferGtPipeline_ = VK_NULL_HANDLE;
    VkPipeline lightingPipeline_ = VK_NULL_HANDLE;
    VkPipeline transparentPipeline_ = VK_NULL_HANDLE;
    VkPipeline transparentGtPipeline_ = VK_NULL_HANDLE;
    VkPipeline ssaoPipeline_ = VK_NULL_HANDLE;
    VkPipeline ssaoBlurPipeline_ = VK_NULL_HANDLE;
    VkPipeline ssaoTemporalPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout hizPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline hizPipeline_ = VK_NULL_HANDLE;
    VkPipeline colorDownsamplePipeline_ = VK_NULL_HANDLE; // reuses hizPipelineLayout_
    VkPipelineLayout ssrPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline ssrPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout bloomPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline bloomExtractPipeline_ = VK_NULL_HANDLE;
    VkPipeline bloomBlurPipeline_ = VK_NULL_HANDLE;
    VkPipeline bloomCompositePipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout exposureSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout histogramPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout exposureSolvePipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline histogramPipeline_ = VK_NULL_HANDLE;
    VkPipeline exposureSolvePipeline_ = VK_NULL_HANDLE;
    VkPipeline shadowPipeline_ = VK_NULL_HANDLE;
    VkShaderModule gbufferVert_ = VK_NULL_HANDLE;
    VkShaderModule gbufferFrag_ = VK_NULL_HANDLE;
    VkShaderModule gbufferGtFrag_ = VK_NULL_HANDLE;
    VkShaderModule lightingFrag_ = VK_NULL_HANDLE;
    VkShaderModule fullscreenVert_ = VK_NULL_HANDLE;
    VkShaderModule transparentVert_ = VK_NULL_HANDLE;
    VkShaderModule transparentFrag_ = VK_NULL_HANDLE;
    VkShaderModule transparentGtFrag_ = VK_NULL_HANDLE;
    VkShaderModule ssaoComp_ = VK_NULL_HANDLE;
    VkShaderModule ssaoBlurComp_ = VK_NULL_HANDLE;
    VkShaderModule ssaoTemporalComp_ = VK_NULL_HANDLE;
    VkShaderModule hizDownsampleComp_ = VK_NULL_HANDLE;
    VkShaderModule colorDownsampleComp_ = VK_NULL_HANDLE;
    VkShaderModule ssrOpaqueComp_ = VK_NULL_HANDLE;
    VkShaderModule bloomExtractComp_ = VK_NULL_HANDLE;
    VkShaderModule bloomBlurComp_ = VK_NULL_HANDLE;
    VkShaderModule bloomCompositeComp_ = VK_NULL_HANDLE;
    VkShaderModule exposureHistogramComp_ = VK_NULL_HANDLE;
    VkShaderModule exposureSolveComp_ = VK_NULL_HANDLE;
    VkShaderModule shadowDepthVert_ = VK_NULL_HANDLE;
    VkShaderModule shadowDepthFrag_ = VK_NULL_HANDLE;
    VkSampler textureSampler_ = VK_NULL_HANDLE;
    VkSampler gbufferSampler_ = VK_NULL_HANDLE;
    VkSampler shadowSampler_ = VK_NULL_HANDLE;
    VkSampler hizSampler_ = VK_NULL_HANDLE; // nearest + clamp (texelFetch pyramid reads)
    VkSampler colorPyramidSampler_ = VK_NULL_HANDLE; // trilinear + clamp (SSR roughness LOD reads)
};

} // namespace sr
