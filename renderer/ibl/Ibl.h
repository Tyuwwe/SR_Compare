#pragma once
// ============================================================================
// IblMaps — IBL preprocessing for the deferred renderer.
// Two environment sources, both feeding the same split-sum chain:
//   - static equirect HDR file (stb_image) -> cubemap, with a small procedural
//     gradient fallback when the file is missing/unreadable;
//   - procedural sky atmosphere (Hillaire 2020, see ibl/SkyAtmosphere.h):
//     sky_render.comp shades the env cubemap from the transmittance +
//     multi-scatter LUTs for a given sun direction, and updateAtmosphereSky()
//     re-runs it (plus irradiance/prefilter) when the sun moves.
// Downstream chain (identical for both sources):
//   - diffuse irradiance cubemap (cosine convolution)
//   - prefiltered specular cubemap (GGX importance sampling, one mip per
//     roughness bucket)
//   - BRDF integration LUT (scale/bias for F0)
// ============================================================================
#include "renderer/core/Vk.h"
#include "renderer/core/VulkanContext.h"
#include "renderer/math/Math.h"

namespace sr {

class SkyAtmosphere;

class IblMaps {
public:
    static constexpr uint32_t kEnvSize = 512;
    static constexpr uint32_t kIrradianceSize = 32;
    static constexpr uint32_t kPrefilterSize = 128;
    static constexpr uint32_t kPrefilterMips = 5;
    static constexpr uint32_t kLutSize = 256;

    bool build(const VulkanContext& ctx, const char* hdrPath);
    // Procedural atmosphere entry point: sky's LUTs must already be baked.
    bool buildAtmosphere(const VulkanContext& ctx, const SkyAtmosphere& sky,
                         const Vec3& sunDir);
    // Re-renders the sky + regenerates the sun-dependent IBL maps for a new
    // sun direction (atmosphere mode only; returns false otherwise).  A
    // blocking one-shot submission: call on sun-direction changes, not per
    // frame.  The BRDF LUT is sun-independent and is not rebuilt.
    bool updateAtmosphereSky(const VulkanContext& ctx, const SkyAtmosphere& sky,
                             const Vec3& sunDir);
    void destroy(const VulkanContext& ctx);

    VkImageView envView = VK_NULL_HANDLE;        // base env cube, full mip chain (skybox)
    VkImageView irradianceView = VK_NULL_HANDLE; // 32^2 cube, cosine irradiance
    VkImageView prefilterView = VK_NULL_HANDLE;  // 128^2 cube, kPrefilterMips levels
    VkImageView brdfLutView = VK_NULL_HANDLE;    // 256^2 2D (NdV, roughness) -> (scale, bias)
    VkSampler cubeSampler = VK_NULL_HANDLE;      // trilinear, clamp (all cubemaps)
    VkSampler lutSampler = VK_NULL_HANDLE;       // bilinear, clamp (LUT)
    uint32_t prefilterMaxLod = kPrefilterMips - 1;
    bool fromFile = false;       // false = procedural fallback gradient or atmosphere
    bool fromAtmosphere = false; // sky driven by the sun direction; updatable

    struct ComputeStage {
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        void destroy(const VulkanContext& ctx);
    };

private:
    // Compute stages for the env->chain passes.  The equirect stage is only
    // built for the static-file path, the sky stage only for the atmosphere
    // path.
    bool createStages(const VulkanContext& ctx, bool withEquirect, bool withSky);
    bool createTargets(const VulkanContext& ctx); // images, views, samplers, pool, sets
    // Records: sky/equirect pass must already have written env mip 0 (GENERAL);
    // generates the remaining mips and lands the whole cube in SHADER_READ_ONLY.
    // initialLayout is the layout of mips 1+ on entry (UNDEFINED on first
    // build, SHADER_READ_ONLY on updates).
    void recordEnvMipChain(VkCommandBuffer cmd, VkImageLayout initialLayout) const;
    // Records the irradiance + prefilter dispatches (+ BRDF LUT when
    // withBrdfLut).  initialLayout is the incoming layout of the target
    // images (UNDEFINED on first build, SHADER_READ_ONLY on updates).
    void recordDownstream(VkCommandBuffer cmd, VkImageLayout initialLayout,
                          bool withBrdfLut) const;
    // Records the sky_render dispatch into env cube mip 0 (ends with mip 0 in
    // GENERAL, ready for recordEnvMipChain).
    void recordSkyRender(VkCommandBuffer cmd, const Vec3& sunDir,
                         VkImageLayout initialLayout) const;

    VkImage envImage_ = VK_NULL_HANDLE;
    VmaAllocation envMemory_ = VK_NULL_HANDLE;
    VkImage irrImage_ = VK_NULL_HANDLE;
    VmaAllocation irrMemory_ = VK_NULL_HANDLE;
    VkImage preImage_ = VK_NULL_HANDLE;
    VmaAllocation preMemory_ = VK_NULL_HANDLE;
    VkImage lutImage_ = VK_NULL_HANDLE;
    VmaAllocation lutMemory_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    uint32_t envMips_ = 1;

    // Persistent storage views + sets kept alive so updateAtmosphereSky can
    // re-run the chain without rebuilding descriptors.
    VkImageView envStoreMip0_ = VK_NULL_HANDLE;
    VkImageView irrStore_ = VK_NULL_HANDLE;
    VkImageView preStore_[kPrefilterMips] = {};
    VkDescriptorSet irradianceSet_ = VK_NULL_HANDLE;
    VkDescriptorSet prefilterSets_[kPrefilterMips] = {};
    VkDescriptorSet brdfLutSet_ = VK_NULL_HANDLE;
    VkDescriptorSet skySet_ = VK_NULL_HANDLE;

    ComputeStage equirectStage_;
    ComputeStage irradianceStage_;
    ComputeStage prefilterStage_;
    ComputeStage brdfLutStage_;
    ComputeStage skyStage_;
};

} // namespace sr
