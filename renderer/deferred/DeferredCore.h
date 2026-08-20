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

constexpr uint32_t kMaxLights = 8;

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
    VkDeviceMemory memory = VK_NULL_HANDLE;
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
static_assert(sizeof(LightingUBO) == 816, "LightingUBO std140 size mismatch");

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

// Push constants of the SSAO compute pass (ssao.comp), 96 bytes.
struct SsaoPush {
    float viewProj[16]; // matches the GBuffer pass of this path (jittered for LR)
    float params[4];    // radius (m), bias (m), intensity, power
    float params2[4];   // frame index (per-frame noise rotation), unused x3
};
static_assert(sizeof(SsaoPush) == 96, "SsaoPush size mismatch");

// SSAO defaults (scene scale: Bistro street ~180 m, sponza ~20 m).  0.5 m is
// a contact-shadow radius; bias pushes samples off the surface along the
// normal to avoid self-occlusion acne.
constexpr float kSsaoRadius = 0.5f;
constexpr float kSsaoBias = 0.03f;
constexpr float kSsaoIntensity = 1.5f;
constexpr float kSsaoPower = 1.0f;

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
    // given; an override is used as-is (still truncated at kMaxLights).
    // shadow (optional): non-null enables CSM sampling in the shaders
    // (shadowParams.z = 1); null writes identity cascades with shadows off.
    void fillLightingUBO(LightingUBO& out, const Scene& scene, const Camera& camera,
                         const Mat4& invViewProj,
                         const std::vector<Light>* overrideLights = nullptr,
                         const ShadowFrame* shadow = nullptr) const;

    // Uploads the dynamic-offset material UBO array (one entry per material).
    bool createMaterialUbo(const VulkanContext& ctx, const Scene& scene, VkBuffer& buffer,
                           VkDeviceMemory& memory, uint32_t& stride) const;

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
    // (binding 5 left unwritten).
    void writeTransparentSet(const VulkanContext& ctx, VkDescriptorSet set,
                             VkBuffer lightingUbo, VkImageView ssao, VkImageView shadow) const;
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

    // --- SSAO pass (between the GBuffer and the lighting pass) ----------------
    // ssao set: depth + normal samplers + raw-AO storage image (R16_SFLOAT).
    void writeSsaoSet(const VulkanContext& ctx, VkDescriptorSet set, VkImageView depth,
                      VkImageView normal, VkImageView aoRaw) const;
    // ssao-blur set: raw-AO sampler + blurred-AO storage image.
    void writeSsaoBlurSet(const VulkanContext& ctx, VkDescriptorSet set, VkImageView aoRaw,
                          VkImageView ao) const;
    // Dispatches ssao.comp.  The caller owns all layout transitions: depth and
    // normal in SHADER_READ_ONLY, aoRaw in GENERAL.  viewProj must match the
    // GBuffer pass of this path (jittered for the low-res path).
    void recordSsaoPass(VkCommandBuffer cmd, VkDescriptorSet ssaoSet, const Mat4& viewProj,
                        uint32_t frameIndex, uint32_t width, uint32_t height) const;
    // Dispatches ssao_blur.comp (cross box filter): aoRaw SHADER_READ_ONLY,
    // ao GENERAL.
    void recordSsaoBlurPass(VkCommandBuffer cmd, VkDescriptorSet blurSet, uint32_t width,
                            uint32_t height) const;

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
    VkPipelineLayout scenePipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout lightingPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout transparentPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout ssaoPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout ssaoBlurPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline gbufferPipeline_ = VK_NULL_HANDLE;
    VkPipeline gbufferGtPipeline_ = VK_NULL_HANDLE;
    VkPipeline lightingPipeline_ = VK_NULL_HANDLE;
    VkPipeline transparentPipeline_ = VK_NULL_HANDLE;
    VkPipeline transparentGtPipeline_ = VK_NULL_HANDLE;
    VkPipeline ssaoPipeline_ = VK_NULL_HANDLE;
    VkPipeline ssaoBlurPipeline_ = VK_NULL_HANDLE;
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
    VkShaderModule shadowDepthVert_ = VK_NULL_HANDLE;
    VkShaderModule shadowDepthFrag_ = VK_NULL_HANDLE;
    VkSampler textureSampler_ = VK_NULL_HANDLE;
    VkSampler gbufferSampler_ = VK_NULL_HANDLE;
    VkSampler shadowSampler_ = VK_NULL_HANDLE;
};

} // namespace sr
