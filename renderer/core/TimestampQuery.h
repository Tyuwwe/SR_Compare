#pragma once
// ============================================================================
// GPU timestamp query wrapper.  Records coarse per-frame timings (whole frame,
// scene pass, upscaler dispatch) for later use by the benchmark module.
// ============================================================================
#include "renderer/core/Vk.h"
#include "renderer/core/VulkanContext.h"

namespace sr {

class TimestampQuery {
public:
    struct Timings {
        double frameMs = 0.0;
        double sceneMs = 0.0;
        double upscaleMs = 0.0;
    };

    bool create(const VulkanContext& ctx, uint32_t framesInFlight);
    void destroy(const VulkanContext& ctx);

    void resetForFrame(VkCommandBuffer cmd, uint32_t slot);
    void frameBegin(VkCommandBuffer cmd, uint32_t slot);
    void sceneBegin(VkCommandBuffer cmd, uint32_t slot);
    void sceneEnd(VkCommandBuffer cmd, uint32_t slot);
    void upscaleBegin(VkCommandBuffer cmd, uint32_t slot);
    void upscaleEnd(VkCommandBuffer cmd, uint32_t slot);
    void frameEnd(VkCommandBuffer cmd, uint32_t slot);

    // Blocking readback of one frame slot (call after the fence for that slot
    // has been signaled).
    Timings read(const VulkanContext& ctx, uint32_t slot) const;

private:
    VkQueryPool pool_ = VK_NULL_HANDLE;
    uint32_t framesInFlight_ = 0;
    float periodNs_ = 0.f;
    static constexpr uint32_t kQueriesPerSlot = 6;
};

} // namespace sr
