#pragma once
// ============================================================================
// Small reusable Vulkan helpers: buffer/image creation (backed by the
// VulkanContext VMA allocator), image views/samplers, layout transitions and
// one-time command submission.
// ============================================================================
#include "renderer/core/Vk.h"
#include "renderer/core/VulkanContext.h"

#include <functional>

namespace sr {

// Buffers/images are allocated through ctx.allocator (VMA).  Destroy them
// with vmaDestroyBuffer / vmaDestroyImage, map with vmaMapMemory.
VkResult createBuffer(const VulkanContext& ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags props, VkBuffer& buffer, VmaAllocation& allocation);

VkResult createImage(const VulkanContext& ctx, uint32_t width, uint32_t height, VkFormat format,
                     VkImageUsageFlags usage, VkImage& image, VmaAllocation& allocation,
                     uint32_t mipLevels = 1, uint32_t arrayLayers = 1,
                     VkImageCreateFlags flags = 0);

// 3D volume variant (froxel fog grids); single mip, no layers.
VkResult createImage3D(const VulkanContext& ctx, uint32_t width, uint32_t height, uint32_t depth,
                       VkFormat format, VkImageUsageFlags usage, VkImage& image,
                       VmaAllocation& allocation);

VkImageView createImageView(const VulkanContext& ctx, VkImage image, VkFormat format,
                            VkImageAspectFlags aspect, uint32_t baseMip = 0, uint32_t levelCount = 1,
                            VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D,
                            uint32_t layerCount = 1);

// maxLod > 0 enables trilinear mipmapping (mipmapMode LINEAR).
VkSampler createSampler(const VulkanContext& ctx, VkFilter filter, VkSamplerAddressMode addressMode,
                        float maxAnisotropy = 0.f, float maxLod = 0.f);

// vkCreateGraphicsPipelines / vkCreateComputePipelines through the context's
// persistent pipeline cache; host access to the cache is serialized through
// ctx.pipelineMutex (the GUI async loader creates pipelines off-thread).
VkResult createGraphicsPipeline(const VulkanContext& ctx, const VkGraphicsPipelineCreateInfo& ci,
                                VkPipeline& out);
VkResult createComputePipeline(const VulkanContext& ctx, const VkComputePipelineCreateInfo& ci,
                               VkPipeline& out);

// Insert an image layout transition via synchronization2.  The caller names
// the exact producing stage/access (work that last touched the image in its
// old layout) and consuming stage/access.  oldLayout == UNDEFINED discards
// the contents, so the source scope is forced to NONE.
void imageBarrier(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout,
                  VkImageLayout newLayout, VkPipelineStageFlags2 srcStage,
                  VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage,
                  VkAccessFlags2 dstAccess,
                  VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                  uint32_t baseMip = 0, uint32_t levelCount = 1, uint32_t baseLayer = 0,
                  uint32_t layerCount = 1);

// Recurring producer/consumer scopes for imageBarrier call sites (sync2).
// They keep the frame-transition call sites readable; every transition still
// spells out the exact stage/access pair it needs.
namespace sync {
// Sampled-read stages reachable from our textures: fragment (lighting,
// transparent, present, compose) and compute (upscalers, GTAO, bloom, IBL).
inline constexpr VkPipelineStageFlags2 kSampleStages =
    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
inline constexpr VkPipelineStageFlags2 kColorAttach =
    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
inline constexpr VkPipelineStageFlags2 kDepthTests =
    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
inline constexpr VkPipelineStageFlags2 kCompute = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
inline constexpr VkPipelineStageFlags2 kFragment = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
inline constexpr VkPipelineStageFlags2 kBlit = VK_PIPELINE_STAGE_2_BLIT_BIT;
inline constexpr VkPipelineStageFlags2 kCopy = VK_PIPELINE_STAGE_2_COPY_BIT;
inline constexpr VkAccessFlags2 kSampled = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
inline constexpr VkAccessFlags2 kStorageWrite = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
// Opaque SSR reads its own texel back via imageLoad before rewriting it.
inline constexpr VkAccessFlags2 kStorageReadWrite =
    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
inline constexpr VkAccessFlags2 kColorWrite = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
inline constexpr VkAccessFlags2 kColorReadWrite =
    VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
inline constexpr VkAccessFlags2 kDepthWrite = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
inline constexpr VkAccessFlags2 kDepthReadWrite =
    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
inline constexpr VkAccessFlags2 kDepthRead = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
inline constexpr VkAccessFlags2 kTransferRead = VK_ACCESS_2_TRANSFER_READ_BIT;
inline constexpr VkAccessFlags2 kTransferWrite = VK_ACCESS_2_TRANSFER_WRITE_BIT;
} // namespace sync

inline void copyColorImage(VkCommandBuffer cmd, VkImage src, VkImageLayout srcLayout, VkImage dst,
                           VkImageLayout dstLayout, uint32_t width, uint32_t height) {
    VkImageCopy region = {};
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.layerCount = 1;
    region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.dstSubresource.layerCount = 1;
    region.extent = {width, height, 1};
    vkCmdCopyImage(cmd, src, srcLayout, dst, dstLayout, 1, &region);
}

// Copy a host-visible buffer into a 2D image and leave it SHADER_READ_ONLY.
void copyBufferToImage(VkCommandBuffer cmd, VkBuffer src, VkImage dst, uint32_t width,
                       uint32_t height, VkFormat format);

// copyBufferToImage split across the queue handoff for submitUploadOneShot:
// the first stage uses only transfer-legal stages (valid in a transfer-queue
// command buffer); the second transitions TRANSFER_DST -> SHADER_READ_ONLY
// and must run on the graphics queue (shader stages are not legal in
// transfer-family barriers).
void copyBufferToImageTransferStage(VkCommandBuffer cmd, VkBuffer src, VkImage dst, uint32_t width,
                                    uint32_t height);
void transitionImageToShaderRead(VkCommandBuffer cmd, VkImage image);

// Allocate a one-time command buffer, record fn, submit to the graphics queue,
// wait for completion and free the buffer.  pool != VK_NULL_HANDLE selects the
// command pool (must be owned by the calling thread); the queue submit itself
// is serialized through VulkanContext::queueMutex.
void submitOneShot(const VulkanContext& ctx, const std::function<void(VkCommandBuffer)>& fn,
                   VkCommandPool pool = VK_NULL_HANDLE);

// Upload variant for pure transfer work (buffer/image copies).  fnCopy may use
// only transfer-queue-legal commands and barrier stages (no blits/clears, no
// shader stages in barrier masks); fnPost, if non-empty, records the consumer
// transition (e.g. TRANSFER_DST -> SHADER_READ_ONLY) and runs on the graphics
// queue.  When the context has a dedicated transfer queue the copy runs there
// (on the DMA engine, overlapping in-flight rendering instead of queueing
// behind it) using the shared VulkanContext::transferPool; the transfer submit
// signals a binary semaphore that the graphics-queue submit (fnPost) waits on,
// so every later graphics submission sees the uploaded data and final layouts
// without queue-family ownership transfers (upload targets are created with
// CONCURRENT sharing across the two families; see createBuffer/createImage).
// Without a transfer queue both parts record into one graphics command buffer,
// exactly like submitOneShot; pool then selects the command pool as there, and
// on the transfer path it also provides the graphics-family pool for fnPost.
void submitUploadOneShot(const VulkanContext& ctx,
                         const std::function<void(VkCommandBuffer)>& fnCopy,
                         const std::function<void(VkCommandBuffer)>& fnPost = {},
                         VkCommandPool pool = VK_NULL_HANDLE);

} // namespace sr
