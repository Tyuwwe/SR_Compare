#pragma once
// ============================================================================
// IblMaps — one-shot IBL preprocessing for the deferred renderer.
// Loads an equirectangular HDR environment map (stb_image), converts it to a
// cubemap (compute), then precomputes the split-sum IBL maps:
//   - diffuse irradiance cubemap (cosine convolution)
//   - prefiltered specular cubemap (GGX importance sampling, one mip per
//     roughness bucket)
//   - BRDF integration LUT (scale/bias for F0)
// When the file is missing/unreadable a small procedural gradient is used, so
// the renderer degrades to a constant ambient-ish environment.
// ============================================================================
#include "renderer/core/Vk.h"
#include "renderer/core/VulkanContext.h"

namespace sr {

class IblMaps {
public:
    static constexpr uint32_t kEnvSize = 512;
    static constexpr uint32_t kIrradianceSize = 32;
    static constexpr uint32_t kPrefilterSize = 128;
    static constexpr uint32_t kPrefilterMips = 5;
    static constexpr uint32_t kLutSize = 256;

    bool build(const VulkanContext& ctx, const char* hdrPath);
    void destroy(const VulkanContext& ctx);

    VkImageView envView = VK_NULL_HANDLE;        // base env cube, full mip chain (skybox)
    VkImageView irradianceView = VK_NULL_HANDLE; // 32^2 cube, cosine irradiance
    VkImageView prefilterView = VK_NULL_HANDLE;  // 128^2 cube, kPrefilterMips levels
    VkImageView brdfLutView = VK_NULL_HANDLE;    // 256^2 2D (NdV, roughness) -> (scale, bias)
    VkSampler cubeSampler = VK_NULL_HANDLE;      // trilinear, clamp (all cubemaps)
    VkSampler lutSampler = VK_NULL_HANDLE;       // bilinear, clamp (LUT)
    uint32_t prefilterMaxLod = kPrefilterMips - 1;
    bool fromFile = false; // false = procedural fallback gradient

private:
    VkImage envImage_ = VK_NULL_HANDLE;
    VkDeviceMemory envMemory_ = VK_NULL_HANDLE;
    VkImage irrImage_ = VK_NULL_HANDLE;
    VkDeviceMemory irrMemory_ = VK_NULL_HANDLE;
    VkImage preImage_ = VK_NULL_HANDLE;
    VkDeviceMemory preMemory_ = VK_NULL_HANDLE;
    VkImage lutImage_ = VK_NULL_HANDLE;
    VkDeviceMemory lutMemory_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;

    struct ComputeStage {
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        void destroy(const VulkanContext& ctx);
    };
    ComputeStage equirectStage_;
    ComputeStage irradianceStage_;
    ComputeStage prefilterStage_;
    ComputeStage brdfLutStage_;
};

} // namespace sr
