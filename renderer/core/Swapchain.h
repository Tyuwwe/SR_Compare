#pragma once
// ============================================================================
// Swapchain wrapper: creation, image acquisition, presentation and resize.
// SDR uses a plain BGRA8 UNORM format; tone mapping + sRGB gamma are applied
// in the present shader (and mirrored in the CPU screenshot path) for
// consistency.  With preferHdr the format probe prefers HDR10 (A2B10G10R10 +
// ST2084 PQ), then scRGB (RGBA16F + extended sRGB), falling back to SDR with
// a stderr note when the surface exposes neither (Phase 6c).
// ============================================================================
#include "renderer/core/Vk.h"
#include "renderer/core/VulkanContext.h"

#include <vector>

namespace sr {

// Display output mode, derived from the swapchain surface format.
enum class HdrMode { Sdr = 0, Hdr10 = 1, ScRgb = 2 };

class Swapchain {
public:
    // allowMailbox: vsync on prefers MAILBOX then FIFO.  Viewer frame-gen
    // dual-present needs FIFO so both the interpolated and true presents
    // hit the display (MAILBOX would drop the first of the pair).
    // preferHdr: probe for HDR10 then scRGB surface formats (SDR fallback).
    bool create(const VulkanContext& ctx, uint32_t width, uint32_t height, bool vsync,
                bool allowMailbox = true, bool preferHdr = false);
    void destroy(const VulkanContext& ctx);

    VkResult acquireNext(const VulkanContext& ctx, VkSemaphore signal, uint32_t& imageIndex);
    VkResult present(const VulkanContext& ctx, uint32_t imageIndex, VkSemaphore wait);

    // Surface HDR capability probe (no swapchain created).
    static void queryHdrSupport(const VulkanContext& ctx, bool& hdr10, bool& scRgb);

    VkFormat format() const { return format_; }
    VkColorSpaceKHR colorSpace() const { return colorSpace_; }
    HdrMode hdrMode() const { return hdrMode_; }
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
    VkColorSpaceKHR colorSpace_ = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    HdrMode hdrMode_ = HdrMode::Sdr;
    VkExtent2D extent_{};
};

} // namespace sr
