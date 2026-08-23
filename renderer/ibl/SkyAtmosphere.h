#pragma once
// ============================================================================
// SkyAtmosphere — Hillaire 2020 (EGSR, "A Scalable and Production Ready Sky
// and Atmosphere Rendering Technique"; reference implementation:
// sebh/UnrealEngineSkyAtmosphere, UE4 SkyAtmosphere.usf).
//
// Owns the two atmosphere LUTs, baked once at init with fixed Earth
// parameters (see shaders/atmosphere.glsl):
//   - transmittance LUT 256x64: ray optical depth -> transmittance, used by
//     the sky render and (via CPU mirror) preset sun colour derivation;
//   - multi-scatter LUT 32x32: 2nd-order luminance + infinite-order geometric
//     series (paper eq. 10), sampled by the sky render.
// The actual sky image is rendered by IblMaps (sky_render.comp) straight into
// the env cubemap, so the existing irradiance/prefilter chain is reused
// unchanged and IBL follows the sun direction.
//
// Deterministic: fixed sample counts, no random or wall-clock input.
//
// Note (Phase 5b scope): the froxel volumetric fog (Phase 5a) stays a
// separate medium; a Frostbite-style unified atmosphere+fog integration is
// out of scope here.
// ============================================================================
#include "renderer/core/Vk.h"
#include "renderer/core/VulkanContext.h"
#include "renderer/math/Math.h"

namespace sr {

class SkyAtmosphere {
public:
    static constexpr uint32_t kTransmittanceWidth = 256;
    static constexpr uint32_t kTransmittanceHeight = 64;
    static constexpr uint32_t kMultiScatterSize = 32;
    // Relative sun illuminance driving the sky render (LUTs are unit-illuminance
    // transfer functions).  Tuned so the noon sky sits in the same few-unit
    // luminance range as the bundled HDR environments under display exposure 1.
    static constexpr float kSunIlluminance = 30.f;

    // Creates + bakes both LUTs (two one-shot compute dispatches).
    bool init(const VulkanContext& ctx);
    void destroy(const VulkanContext& ctx);

    VkImageView transmittanceView() const { return transView_; }
    VkImageView multiScatterView() const { return msView_; }
    VkSampler lutSampler() const { return sampler_; }

    // CPU mirror of the transmittance LUT march (same constants as
    // shaders/atmosphere.glsl — keep in sync): transmittance from ground
    // level towards the sun.  Used to derive warm low-sun colours for the
    // scene's directional light from the same atmosphere model.  Zero when
    // the sun is below the horizon.
    static Vec3 sunTransmittanceFromGround(const Vec3& sunDir);

    struct Stage {
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
    };

private:
    VkImage transImage_ = VK_NULL_HANDLE;
    VmaAllocation transMemory_ = VK_NULL_HANDLE;
    VkImageView transView_ = VK_NULL_HANDLE;
    VkImage msImage_ = VK_NULL_HANDLE;
    VmaAllocation msMemory_ = VK_NULL_HANDLE;
    VkImageView msView_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    Stage transStage_;
    Stage msStage_;
};

} // namespace sr
