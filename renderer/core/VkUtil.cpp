#include "renderer/core/VkUtil.h"

namespace sr {

uint32_t findMemoryType(const VulkanContext& ctx, uint32_t typeBits, VkMemoryPropertyFlags required) {
    for (uint32_t i = 0; i < ctx.memoryProperties.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) &&
            (ctx.memoryProperties.memoryTypes[i].propertyFlags & required) == required)
            return i;
    }
    return 0xFFFFFFFFu;
}

VkResult createBuffer(const VulkanContext& ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags props, VkBuffer& buffer, VkDeviceMemory& memory) {
    VkBufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult res = vkCreateBuffer(ctx.device, &info, nullptr, &buffer);
    if (res != VK_SUCCESS) return res;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(ctx.device, buffer, &req);
    const uint32_t type = findMemoryType(ctx, req.memoryTypeBits, props);
    if (type == 0xFFFFFFFFu) {
        vkDestroyBuffer(ctx.device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkMemoryAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type;
    res = vkAllocateMemory(ctx.device, &alloc, nullptr, &memory);
    if (res != VK_SUCCESS) {
        vkDestroyBuffer(ctx.device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        return res;
    }
    vkBindBufferMemory(ctx.device, buffer, memory, 0);
    return VK_SUCCESS;
}

VkResult createImage(const VulkanContext& ctx, uint32_t width, uint32_t height, VkFormat format,
                     VkImageUsageFlags usage, VkImage& image, VkDeviceMemory& memory,
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
    VkResult res = vkCreateImage(ctx.device, &info, nullptr, &image);
    if (res != VK_SUCCESS) return res;

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(ctx.device, image, &req);
    const uint32_t type = findMemoryType(ctx, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == 0xFFFFFFFFu) {
        vkDestroyImage(ctx.device, image, nullptr);
        image = VK_NULL_HANDLE;
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkMemoryAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type;
    res = vkAllocateMemory(ctx.device, &alloc, nullptr, &memory);
    if (res != VK_SUCCESS) {
        vkDestroyImage(ctx.device, image, nullptr);
        image = VK_NULL_HANDLE;
        return res;
    }
    vkBindImageMemory(ctx.device, image, memory, 0);
    return VK_SUCCESS;
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

namespace {

VkAccessFlags accessForLayout(VkImageLayout layout, bool isSource) {
    (void)isSource;
    switch (layout) {
    case VK_IMAGE_LAYOUT_UNDEFINED:
    case VK_IMAGE_LAYOUT_PREINITIALIZED:
    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
        return 0;
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
        return VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        return VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
        return VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        return VK_ACCESS_SHADER_READ_BIT;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        return VK_ACCESS_TRANSFER_READ_BIT;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        return VK_ACCESS_TRANSFER_WRITE_BIT;
    case VK_IMAGE_LAYOUT_GENERAL:
    default:
        return VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    }
}

} // namespace

void imageBarrier(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout,
                  VkImageLayout newLayout, VkImageAspectFlags aspect, uint32_t baseMip,
                  uint32_t levelCount, uint32_t baseLayer, uint32_t layerCount) {
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
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
    barrier.srcAccessMask = accessForLayout(oldLayout, true);
    barrier.dstAccessMask = accessForLayout(newLayout, false);

    VkPipelineStageFlags srcStage = barrier.srcAccessMask == 0 ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                                                               : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkPipelineStageFlags dstStage = barrier.dstAccessMask == 0 ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
                                                               : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void copyBufferToImage(VkCommandBuffer cmd, VkBuffer src, VkImage dst, uint32_t width,
                       uint32_t height, VkFormat format) {
    (void)format; // format is implicit in the source data; kept for API clarity
    imageBarrier(cmd, dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

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

    imageBarrier(cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
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
    if (vkAllocateCommandBuffers(ctx.device, &alloc, &cmd) != VK_SUCCESS || !cmd) return;

    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);
    fn(cmd);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(ctx.device, &fenceInfo, nullptr, &fence);
    {
        std::lock_guard<std::mutex> lk(ctx.queueMutex);
        vkQueueSubmit(ctx.graphicsQueue, 1, &submit, fence);
    }
    vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(ctx.device, fence, nullptr);
    vkFreeCommandBuffers(ctx.device, cmdPool, 1, &cmd);
}

} // namespace sr
