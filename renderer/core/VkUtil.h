#pragma once
// ============================================================================
// Small reusable Vulkan helpers: memory type selection, buffer/image creation,
// image views/samplers, layout transitions and one-time command submission.
// ============================================================================
#include "renderer/core/Vk.h"
#include "renderer/core/VulkanContext.h"

#include <functional>

namespace sr {

uint32_t findMemoryType(const VulkanContext& ctx, uint32_t typeBits, VkMemoryPropertyFlags required);

VkResult createBuffer(const VulkanContext& ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags props, VkBuffer& buffer, VkDeviceMemory& memory);

VkResult createImage(const VulkanContext& ctx, uint32_t width, uint32_t height, VkFormat format,
                     VkImageUsageFlags usage, VkImage& image, VkDeviceMemory& memory,
                     uint32_t mipLevels = 1, uint32_t arrayLayers = 1,
                     VkImageCreateFlags flags = 0);

VkImageView createImageView(const VulkanContext& ctx, VkImage image, VkFormat format,
                            VkImageAspectFlags aspect, uint32_t baseMip = 0, uint32_t levelCount = 1,
                            VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D,
                            uint32_t layerCount = 1);

// maxLod > 0 enables trilinear mipmapping (mipmapMode LINEAR).
VkSampler createSampler(const VulkanContext& ctx, VkFilter filter, VkSamplerAddressMode addressMode,
                        float maxAnisotropy = 0.f, float maxLod = 0.f);

// Insert an image memory barrier.  Stage/access masks are derived from the two
// layouts (coarse but correct); aspect defaults to COLOR.
void imageBarrier(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout,
                  VkImageLayout newLayout, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                  uint32_t baseMip = 0, uint32_t levelCount = 1, uint32_t baseLayer = 0,
                  uint32_t layerCount = 1);

// Copy a host-visible buffer into a 2D image and leave it SHADER_READ_ONLY.
void copyBufferToImage(VkCommandBuffer cmd, VkBuffer src, VkImage dst, uint32_t width,
                       uint32_t height, VkFormat format);

// Allocate a one-time command buffer, record fn, submit to the graphics queue,
// wait for completion and free the buffer.  pool != VK_NULL_HANDLE selects the
// command pool (must be owned by the calling thread); the queue submit itself
// is serialized through VulkanContext::queueMutex.
void submitOneShot(const VulkanContext& ctx, const std::function<void(VkCommandBuffer)>& fn,
                   VkCommandPool pool = VK_NULL_HANDLE);

} // namespace sr
