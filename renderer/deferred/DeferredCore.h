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

namespace sr {

class Scene;
class Camera;

// Default equirect HDR environment map (Bistro san_giuseppe_bridge, bundled
// in the project's assets; override with --env-map).
inline const char* kDefaultEnvMapPath = "assets/env/san_giuseppe_bridge_4k.hdr";

// Attachment formats of the deferred pipeline (identical everywhere).
namespace deferred {
constexpr VkFormat kHdrColorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat kMotionFormat = VK_FORMAT_R16G16_SFLOAT;
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr VkFormat kAlbedoFormat = VK_FORMAT_R8G8B8A8_SRGB;   // rgb sRGB-encoded, a = alpha
constexpr VkFormat kNormalFormat = VK_FORMAT_R16G16B16A16_SFLOAT; // xyz = world normal
constexpr VkFormat kMaterialFormat = VK_FORMAT_R8G8B8A8_UNORM;    // r/g/b = metallic/rough/AO
constexpr VkFormat kEmissiveFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat kReactiveFormat = VK_FORMAT_R16_SFLOAT; // translucent coverage mask
constexpr uint32_t kMaxTextures = 1024;
} // namespace deferred

// std140, matches the SceneUBO block in gbuffer.vert.
struct SceneUBO {
    float viewProj[16];
    float viewProjNoJitter[16];
    float prevViewProj[16];
    float cameraPos[4];
    float light0Pos[4];
    float light0Color[4];
    float light1Pos[4];
    float light1Color[4];
    float ambient[4];
    float renderSizeJitter[4];
};
static_assert(sizeof(SceneUBO) == 304, "SceneUBO std140 size mismatch");

// std140, matches the MaterialUBO block in gbuffer.frag / gbuffer_gt.frag.
struct MaterialUBO {
    float baseColor[4];
    float factors[4];  // metallic, roughness, occlusionStrength, alphaCutoff
    float emissive[4]; // rgb factor
    float tex0[4];     // baseColor, normal, mr, ao texture indices
    float tex1[4];     // emissive texture index
};
static_assert(sizeof(MaterialUBO) == 80, "MaterialUBO std140 size mismatch");

// std140, matches the LightingUBO block in lighting.frag.
struct LightingUBO {
    float invViewProj[16];
    float cameraPos[4];
    float light0Pos[4];
    float light0Color[4];
    float light1Pos[4];
    float light1Color[4];
    float ambient[4];
    float iblParams[4]; // envIntensity, prefilterMaxLod, skyboxEnabled, unused
};
static_assert(sizeof(LightingUBO) == 176, "LightingUBO std140 size mismatch");

struct ScenePush {
    float model[16];
    float prevModel[16];
    float normalModel[16]; // transpose(inverse(mat3(model))), upper 3x3 used
};
static_assert(sizeof(ScenePush) == 192, "ScenePush size mismatch");

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
    void fillLightingUBO(LightingUBO& out, const Scene& scene, const Camera& camera,
                         const Mat4& invViewProj) const;

    // Uploads the dynamic-offset material UBO array (one entry per material).
    bool createMaterialUbo(const VulkanContext& ctx, const Scene& scene, VkBuffer& buffer,
                           VkDeviceMemory& memory, uint32_t& stride) const;

    // --- descriptor writers (sets are allocated from the caller's pool) --------
    void writeTextureSet(const VulkanContext& ctx, VkDescriptorSet set, const Scene& scene) const;
    void writeLightingSet(const VulkanContext& ctx, VkDescriptorSet set, VkBuffer lightingUbo,
                          VkImageView albedo, VkImageView normal, VkImageView material,
                          VkImageView emissive, VkImageView depth, VkImageView ssao) const;

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
    // Binds the LightingUBO (only iblParams is read) + IBL maps + the SSAO
    // texture of this path into a transparentSetLayout() descriptor set.
    void writeTransparentSet(const VulkanContext& ctx, VkDescriptorSet set,
                             VkBuffer lightingUbo, VkImageView ssao) const;
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
    VkSampler textureSampler_ = VK_NULL_HANDLE;
    VkSampler gbufferSampler_ = VK_NULL_HANDLE;
};

} // namespace sr
