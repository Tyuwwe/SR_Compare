#pragma once
// ============================================================================
// Swapchain wrapper: creation, image acquisition, presentation and resize.
// Uses a plain BGRA8 UNORM format; tone mapping + sRGB gamma are applied in
// the present shader (and mirrored in the CPU screenshot path) for consistency.
// ============================================================================
#include "renderer/core/Vk.h"
#include "renderer/core/VulkanContext.h"

#include <vector>

namespace sr {

class Swapchain {
public:
    // allowMailbox: vsync on prefers MAILBOX then FIFO.  Viewer frame-gen
    // dual-present needs FIFO so both the interpolated and true presents
    // hit the display (MAILBOX would drop the first of the pair).
    bool create(const VulkanContext& ctx, uint32_t width, uint32_t height, bool vsync,
                bool allowMailbox = true);
    void destroy(const VulkanContext& ctx);

    VkResult acquireNext(const VulkanContext& ctx, VkSemaphore signal, uint32_t& imageIndex);
    VkResult present(const VulkanContext& ctx, uint32_t imageIndex, VkSemaphore wait);

    VkFormat format() const { return format_; }
    VkExtent2D extent() const { return extent_; }
    uint32_t imageCount() const { return static_cast<uint32_t>(images_.size()); }
    VkImage image(uint32_t i) const { return images_[i]; }
    VkImageView view(uint32_t i) const { return views_[i]; }
    bool valid() const { return swapchain_ != VK_NULL_HANDLE; }

private:
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    std::vector<VkImage> images_;
    std::vector<VkImageView> views_;
    VkFormat format_ = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D extent_{};
};

} // namespace sr
