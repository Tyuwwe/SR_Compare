#pragma once
// ============================================================================
// ReflectionProbes — baked local reflection captures (UE4 reflection-capture
// style, Phase 4c-2).  Up to kMaxProbes axis-aligned box probes per scene.
// Offline (--bake-probes, Renderer::bakeProbes) each probe renders the scene
// into a 128^2 cubemap from its position; the raw cubemaps are written to a
// .probes file next to the scene.  At load time this class uploads the raw
// cubes, runs the same irradiance + GGX prefilter compute chain as the global
// IBL (ibl_irradiance.comp / ibl_prefilter.comp) into two cube-ARRAY textures
// (layer group i*6..i*6+5 = probe i), and fills a small UBO with the boxes.
// lighting.frag / ssr_opaque.comp then blend between the best two containing
// probes (weighted by distance to the box edge) with parallax-corrected box
// projection, falling back to the global environment outside every box.
// Without a bake file the probe count stays 0 and rendering is unchanged.
// ============================================================================
#include "renderer/core/Vk.h"
#include "renderer/core/VulkanContext.h"
#include "renderer/scene/Scene.h"

#include <cstdint>
#include <string>
#include <vector>

namespace sr {

// --- .probes bake file (v1, little-endian) -------------------------------------
//   char     magic[4] = "SRP1"
//   uint32   version = 1
//   uint32   probeCount
//   uint32   faceSize          (kBakeSize)
//   per probe: float position[3], boxMin[3], boxMax[3]
//   pixel data: probeCount * 6 * faceSize^2 * RGBA16F (face-major, row-major,
//   Vulkan cube face order — the horizontal flip needed by the raster
//   convention is applied by the baker before writing).
//
// Writes the bake file for `probes` (rgba = probeCount * 6 * faceSize^2 * 4
// half floats).  Creates the parent directory if needed.
bool saveProbeFile(const char* path, const std::vector<ReflectionProbe>& probes,
                   uint32_t faceSize, const std::vector<uint16_t>& rgba);

// Reads + validates a bake file against the scene's current probe placements
// (exact position/box match, so a stale bake after re-placement is rejected
// and the probes stay inactive).  On success rgbaOut receives the texels.
bool loadProbeFile(const char* path, const std::vector<ReflectionProbe>& defs,
                   uint32_t expectedFaceSize, std::vector<uint16_t>& rgbaOut);

// std140 mirror of the ProbeUBO block in shaders/probes.glsl.
constexpr uint32_t kMaxProbes = kMaxReflectionProbes;
struct ProbeUBO {
    float info[4]; // x = active count, y = prefilter max lod, z = blend distance (m), w unused
    float boxMin[kMaxProbes][4];
    float boxMax[kMaxProbes][4];
    float position[kMaxProbes][4]; // xyz = capture position, w unused
};
static_assert(sizeof(ProbeUBO) == 400, "ProbeUBO std140 size mismatch");

class ReflectionProbes {
public:
    static constexpr uint32_t kBakeSize = 128;       // baked cube face resolution
    static constexpr uint32_t kPrefilterMips = 5;    // same buckets as IblMaps
    static constexpr uint32_t kIrradianceSize = 32;  // same as IblMaps
    static constexpr float kBlendDistance = 0.75f;   // box-edge fade width (m)

    // Creates the (empty) cube arrays, the UBO (count = 0) and the two
    // prefilter pipelines.  Called once from DeferredCore::init so the
    // descriptor bindings are always writable.
    bool create(const VulkanContext& ctx);
    // Uploads + prefilters the baked probes matching `defs` (see
    // loadProbeFile).  No-op (count stays 0) when the bake file is missing or
    // stale.  Safe to call after create(); not reloadable (host restarts).
    bool load(const VulkanContext& ctx, const std::vector<ReflectionProbe>& defs,
              const std::string& filePath);
    void destroy(const VulkanContext& ctx);

    uint32_t count() const { return count_; }
    VkBuffer uboBuffer() const { return ubo_; }
    VkImageView prefilterView() const { return prefilterView_; }   // cube array, 5 mips
    VkImageView irradianceView() const { return irradianceView_; } // cube array, 1 mip
    VkSampler sampler() const { return sampler_; }                 // trilinear clamp

private:
    struct ComputeStage {
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        void destroy(const VulkanContext& ctx);
    };

    uint32_t count_ = 0;
    VkImage prefilterImage_ = VK_NULL_HANDLE;   // 128^2 x 5 mips x kMaxProbes*6 layers
    VmaAllocation prefilterMemory_ = VK_NULL_HANDLE;
    VkImage irradianceImage_ = VK_NULL_HANDLE;  // 32^2 x 1 mip x kMaxProbes*6 layers
    VmaAllocation irradianceMemory_ = VK_NULL_HANDLE;
    VkImageView prefilterView_ = VK_NULL_HANDLE;
    VkImageView irradianceView_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;
    VkBuffer ubo_ = VK_NULL_HANDLE;
    VmaAllocation uboMemory_ = VK_NULL_HANDLE;
    void* uboMapped_ = nullptr;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    ComputeStage irradianceStage_;
    ComputeStage prefilterStage_;
};

} // namespace sr
