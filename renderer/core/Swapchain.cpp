#include "renderer/core/Swapchain.h"

#include "renderer/core/VkUtil.h"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace sr {

namespace {

// HDR10 (A2B10G10R10 + ST2084 PQ, ITU-R BT.2100) first, then scRGB (RGBA16F
// + extended linear sRGB, 1.0 = 80 nits), then the existing SDR BGRA8.  The
// HDR encodes live in grading.glsl (present.frag).
VkSurfaceFormatKHR chooseFormat(const std::vector<VkSurfaceFormatKHR>& formats, bool preferHdr,
                                HdrMode& mode) {
    if (formats.size() == 1 && formats[0].format == VK_FORMAT_UNDEFINED) {
        mode = HdrMode::Sdr;
        return {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    }
    if (preferHdr) {
        for (const auto& f : formats) {
            if (f.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 &&
                f.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT) {
                mode = HdrMode::Hdr10;
                return f;
            }
        }
        for (const auto& f : formats) {
            if (f.format == VK_FORMAT_R16G16B16A16_SFLOAT &&
                f.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT) {
                mode = HdrMode::ScRgb;
                return f;
            }
        }
        std::fprintf(stderr, "swapchain: HDR requested but the surface exposes neither HDR10 nor"
                             " scRGB; falling back to SDR\n");
    }
    mode = HdrMode::Sdr;
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return f;
    }
    return formats[0];
}

VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& modes, bool vsync,
                                   bool allowMailbox) {
    const auto has = [&](VkPresentModeKHR m) {
        return std::find(modes.begin(), modes.end(), m) != modes.end();
    };
    if (!vsync) {
        if (has(VK_PRESENT_MODE_IMMEDIATE_KHR)) return VK_PRESENT_MODE_IMMEDIATE_KHR;
        if (allowMailbox && has(VK_PRESENT_MODE_MAILBOX_KHR)) return VK_PRESENT_MODE_MAILBOX_KHR;
    } else if (allowMailbox && has(VK_PRESENT_MODE_MAILBOX_KHR)) {
        return VK_PRESENT_MODE_MAILBOX_KHR;
    }
    return VK_PRESENT_MODE_FIFO_KHR; // always available
}

} // namespace

bool Swapchain::create(const VulkanContext& ctx, uint32_t width, uint32_t height, bool vsync,
                       bool allowMailbox, bool preferHdr) {
    VkSurfaceCapabilitiesKHR caps;
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx.physicalDevice, ctx.surface, &caps) != VK_SUCCESS)
        return false;

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx.physicalDevice, ctx.surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx.physicalDevice, ctx.surface, &formatCount, formats.data());

    uint32_t modeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(ctx.physicalDevice, ctx.surface, &modeCount, nullptr);
    std::vector<VkPresentModeKHR> modes(modeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(ctx.physicalDevice, ctx.surface, &modeCount, modes.data());

    const VkSurfaceFormatKHR format = chooseFormat(formats, preferHdr, hdrMode_);
    format_ = format.format;
    colorSpace_ = format.colorSpace;

    extent_.width = std::clamp(width, caps.minImageExtent.width, caps.maxImageExtent.width);
    extent_.height = std::clamp(height, caps.minImageExtent.height, caps.maxImageExtent.height);

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0) imageCount = std::min(imageCount, caps.maxImageCount);

    VkSwapchainCreateInfoKHR ci = {};
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface = ctx.surface;
    ci.minImageCount = imageCount;
    ci.imageFormat = format.format;
    ci.imageColorSpace = format.colorSpace;
    ci.imageExtent = extent_;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = choosePresentMode(modes, vsync, allowMailbox);
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = swapchain_;

    VkSwapchainKHR newSwapchain = VK_NULL_HANDLE;
    if (vkCreateSwapchainKHR(ctx.device, &ci, nullptr, &newSwapchain) != VK_SUCCESS) return false;

    // In-flight frames may still reference the old swapchain images, so wait
    // for all queued work to finish before destroying the previous handles.
    if (swapchain_ != VK_NULL_HANDLE) vkDeviceWaitIdle(ctx.device);

    // Destroy the previous swapchain images/views before replacing the handle.
    for (auto& v : views_) vkDestroyImageView(ctx.device, v, nullptr);
    views_.clear();
    images_.clear();
    if (swapchain_) vkDestroySwapchainKHR(ctx.device, swapchain_, nullptr);
    swapchain_ = newSwapchain;

    uint32_t count = 0;
    vkGetSwapchainImagesKHR(ctx.device, swapchain_, &count, nullptr);
    images_.resize(count);
    vkGetSwapchainImagesKHR(ctx.device, swapchain_, &count, images_.data());
    views_.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        views_[i] = createImageView(ctx, images_[i], format_, VK_IMAGE_ASPECT_COLOR_BIT);
    }
    return true;
}

void Swapchain::destroy(const VulkanContext& ctx) {
    for (auto& v : views_) vkDestroyImageView(ctx.device, v, nullptr);
    views_.clear();
    images_.clear();
    if (swapchain_) vkDestroySwapchainKHR(ctx.device, swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
}

void Swapchain::queryHdrSupport(const VulkanContext& ctx, bool& hdr10, bool& scRgb) {
    hdr10 = false;
    scRgb = false;
    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx.physicalDevice, ctx.surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx.physicalDevice, ctx.surface, &formatCount,
                                         formats.data());
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 &&
            f.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT)
            hdr10 = true;
        if (f.format == VK_FORMAT_R16G16B16A16_SFLOAT &&
            f.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT)
            scRgb = true;
    }
}

VkResult Swapchain::acquireNext(const VulkanContext& ctx, VkSemaphore signal, uint32_t& imageIndex) {
    return vkAcquireNextImageKHR(ctx.device, swapchain_, UINT64_MAX, signal, VK_NULL_HANDLE, &imageIndex);
}

VkResult Swapchain::present(const VulkanContext& ctx, uint32_t imageIndex, VkSemaphore wait) {
    VkPresentInfoKHR info = {};
    info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &wait;
    info.swapchainCount = 1;
    info.pSwapchains = &swapchain_;
    info.pImageIndices = &imageIndex;
    // presentQueue aliases graphicsQueue when the families coincide, and the
    // async loader may be submitting to it from the worker thread.
    std::lock_guard<std::mutex> lk(ctx.queueMutex);
    return vkQueuePresentKHR(ctx.presentQueue, &info);
}

} // namespace sr
