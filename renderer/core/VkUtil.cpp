#include "renderer/core/VkUtil.h"

#include "renderer/core/Vma.h"

#include <cstdio>

namespace sr {

VkResult createBuffer(const VulkanContext& ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags props, VkBuffer& buffer, VmaAllocation& allocation) {
    VkBufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc = {};
    alloc.usage = VMA_MEMORY_USAGE_AUTO;
    alloc.requiredFlags = props;
    if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        alloc.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

    return vmaCreateBuffer(ctx.allocator, &info, &alloc, &buffer, &allocation, nullptr);
}

VkResult createImage(const VulkanContext& ctx, uint32_t width, uint32_t height, VkFormat format,
                     VkImageUsageFlags usage, VkImage& image, VmaAllocation& allocation,
                     uint32_t mipLevels, uint32_t arrayLayers, VkImageCreateFlags flags) {
    VkImageCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.flags = flags;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.extent = {width, height, 1};
    info.mipLevels = mipLevels;
    info.arrayLayers = arrayLayers;
    info.format = format;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    info.usage = usage;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc = {};
    alloc.usage = VMA_MEMORY_USAGE_AUTO;
    alloc.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    return vmaCreateImage(ctx.allocator, &info, &alloc, &image, &allocation, nullptr);
}

VkResult createImage3D(const VulkanContext& ctx, uint32_t width, uint32_t height, uint32_t depth,
                       VkFormat format, VkImageUsageFlags usage, VkImage& image,
                       VmaAllocation& allocation) {
    VkImageCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_3D;
    info.extent = {width, height, depth};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.format = format;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    info.usage = usage;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc = {};
    alloc.usage = VMA_MEMORY_USAGE_AUTO;
    alloc.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    return vmaCreateImage(ctx.allocator, &info, &alloc, &image, &allocation, nullptr);
}

VkImageView createImageView(const VulkanContext& ctx, VkImage image, VkFormat format,
                            VkImageAspectFlags aspect, uint32_t baseMip, uint32_t levelCount,
                            VkImageViewType viewType, uint32_t layerCount) {
    VkImageViewCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    info.image = image;
    info.viewType = viewType;
    info.format = format;
    info.subresourceRange.aspectMask = aspect;
    info.subresourceRange.baseMipLevel = baseMip;
    info.subresourceRange.levelCount = levelCount;
    info.subresourceRange.baseArrayLayer = 0;
    info.subresourceRange.layerCount = layerCount;
    VkImageView view = VK_NULL_HANDLE;
    vkCreateImageView(ctx.device, &info, nullptr, &view);
    return view;
}

VkSampler createSampler(const VulkanContext& ctx, VkFilter filter, VkSamplerAddressMode addressMode,
                        float maxAnisotropy, float maxLod) {
    VkSamplerCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.magFilter = filter;
    info.minFilter = filter;
    info.mipmapMode = maxLod > 0.f ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    info.addressModeU = addressMode;
    info.addressModeV = addressMode;
    info.addressModeW = addressMode;
    info.mipLodBias = 0.f;
    info.minLod = 0.f;
    info.maxLod = maxLod;
    info.anisotropyEnable = (maxAnisotropy > 1.f && ctx.features.samplerAnisotropy) ? VK_TRUE : VK_FALSE;
    info.maxAnisotropy = info.anisotropyEnable ? maxAnisotropy : 1.f;
    VkSampler sampler = VK_NULL_HANDLE;
    vkCreateSampler(ctx.device, &info, nullptr, &sampler);
    return sampler;
}

VkResult createGraphicsPipeline(const VulkanContext& ctx, const VkGraphicsPipelineCreateInfo& ci,
                                VkPipeline& out) {
    std::lock_guard<std::mutex> lk(ctx.pipelineMutex);
    return vkCreateGraphicsPipelines(ctx.device, ctx.pipelineCache, 1, &ci, nullptr, &out);
}

VkResult createComputePipeline(const VulkanContext& ctx, const VkComputePipelineCreateInfo& ci,
                               VkPipeline& out) {
    std::lock_guard<std::mutex> lk(ctx.pipelineMutex);
    return vkCreateComputePipelines(ctx.device, ctx.pipelineCache, 1, &ci, nullptr, &out);
}

void imageBarrier(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout,
                  VkImageLayout newLayout, VkPipelineStageFlags2 srcStage,
                  VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage,
                  VkAccessFlags2 dstAccess, VkImageAspectFlags aspect, uint32_t baseMip,
                  uint32_t levelCount, uint32_t baseLayer, uint32_t layerCount) {
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
        // Contents are discarded; there is no producer to synchronize with.
        srcStage = VK_PIPELINE_STAGE_2_NONE;
        srcAccess = VK_ACCESS_2_NONE;
    }

    VkImageMemoryBarrier2 barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = aspect;
    barrier.subresourceRange.baseMipLevel = baseMip;
    barrier.subresourceRange.levelCount = levelCount;
    barrier.subresourceRange.baseArrayLayer = baseLayer;
    barrier.subresourceRange.layerCount = layerCount;

    VkDependencyInfo dependency = {};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dependency);
}

void copyBufferToImage(VkCommandBuffer cmd, VkBuffer src, VkImage dst, uint32_t width,
                       uint32_t height, VkFormat format) {
    (void)format; // format is implicit in the source data; kept for API clarity
    imageBarrier(cmd, dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                 VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(cmd, src, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Consumers: fragment (font atlas) and compute (IBL equirect) sampling.
    imageBarrier(cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
}

void submitOneShot(const VulkanContext& ctx, const std::function<void(VkCommandBuffer)>& fn,
                   VkCommandPool pool) {
    const VkCommandPool cmdPool = pool ? pool : ctx.oneShotPool;
    VkCommandBufferAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = cmdPool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(ctx.device, &alloc, &cmd) != VK_SUCCESS || !cmd) {
        std::fprintf(stderr, "submitOneShot: vkAllocateCommandBuffers failed\n");
        return;
    }

    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS) {
        std::fprintf(stderr, "submitOneShot: vkBeginCommandBuffer failed\n");
        vkFreeCommandBuffers(ctx.device, cmdPool, 1, &cmd);
        return;
    }
    fn(cmd);
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        std::fprintf(stderr, "submitOneShot: vkEndCommandBuffer failed\n");
        vkFreeCommandBuffers(ctx.device, cmdPool, 1, &cmd);
        return;
    }

    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    if (vkCreateFence(ctx.device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
        std::fprintf(stderr, "submitOneShot: vkCreateFence failed\n");
        vkFreeCommandBuffers(ctx.device, cmdPool, 1, &cmd);
        return;
    }
    VkResult submitResult = VK_SUCCESS;
    {
        std::lock_guard<std::mutex> lk(ctx.queueMutex);
        submitResult = vkQueueSubmit(ctx.graphicsQueue, 1, &submit, fence);
    }
    if (submitResult != VK_SUCCESS) {
        std::fprintf(stderr, "submitOneShot: vkQueueSubmit failed\n");
        vkDestroyFence(ctx.device, fence, nullptr);
        vkFreeCommandBuffers(ctx.device, cmdPool, 1, &cmd);
        return;
    }
    if (vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        // Log and fall through to the cleanup below; the fence/cmd are torn down
        // unconditionally here regardless of the wait result, matching the prior
        // behavior (a failed wait is fatal and leaves the device unusable).
        std::fprintf(stderr, "submitOneShot: vkWaitForFences failed\n");
    }
    vkDestroyFence(ctx.device, fence, nullptr);
    vkFreeCommandBuffers(ctx.device, cmdPool, 1, &cmd);
}

} // namespace sr
