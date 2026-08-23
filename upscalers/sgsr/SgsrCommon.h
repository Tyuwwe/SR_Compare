#pragma once
// ============================================================================
// SgsrCommon — small Vulkan helpers shared by the SGSR1 / SGSR2 plugins.
// ============================================================================
#include "upscalers/IUpscaler.h"
#include "upscalers/VkHelpers.h"

#include <cstdio>
#include <fstream>
#include <vector>

namespace sr {
namespace sgsr {

struct Image2D {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize allocationSize = 0;
};

inline uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeBits,
                               VkMemoryPropertyFlags required) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & required) == required)
            return i;
    }
    return 0xFFFFFFFFu;
}

inline bool createImage2D(const VulkanEnv& env, uint32_t width, uint32_t height, VkFormat format,
                          VkImageUsageFlags usage, Image2D& out) {
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.extent = {width, height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.format = format;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ci.usage = usage;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateImage(env.device, &ci, nullptr, &out.image) != VK_SUCCESS) return false;

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(env.device, out.image, &req);
    const uint32_t type = findMemoryType(env.physicalDevice, req.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == 0xFFFFFFFFu) {
        vkDestroyImage(env.device, out.image, nullptr);
        out.image = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryAllocateInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    if (vkAllocateMemory(env.device, &ai, nullptr, &out.memory) != VK_SUCCESS) {
        vkDestroyImage(env.device, out.image, nullptr);
        out.image = VK_NULL_HANDLE;
        return false;
    }
    vkBindImageMemory(env.device, out.image, out.memory, 0);
    out.allocationSize = req.size;

    VkImageViewCreateInfo vi = {};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = out.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    if (vkCreateImageView(env.device, &vi, nullptr, &out.view) != VK_SUCCESS) {
        vkDestroyImage(env.device, out.image, nullptr);
        vkFreeMemory(env.device, out.memory, nullptr);
        out.image = VK_NULL_HANDLE;
        out.memory = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

inline void destroyImage2D(VkDevice device, Image2D& img) {
    if (img.view) { vkDestroyImageView(device, img.view, nullptr); img.view = VK_NULL_HANDLE; }
    if (img.image) { vkDestroyImage(device, img.image, nullptr); img.image = VK_NULL_HANDLE; }
    if (img.memory) { vkFreeMemory(device, img.memory, nullptr); img.memory = VK_NULL_HANDLE; }
}

inline VkSampler createSampler(VkDevice device, VkFilter filter) {
    VkSamplerCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    ci.magFilter = filter;
    ci.minFilter = filter;
    ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ci.maxLod = 0.f;
    VkSampler sampler = VK_NULL_HANDLE;
    vkCreateSampler(device, &ci, nullptr, &sampler);
    return sampler;
}

inline VkShaderModule loadShaderModule(const VulkanEnv& env, const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "SGSR: failed to open shader %s\n", path);
        return VK_NULL_HANDLE;
    }
    const std::streamoff size = file.tellg();
    if (size <= 0 || size % 4 != 0) return VK_NULL_HANDLE;
    file.seekg(0);
    std::vector<uint32_t> code(static_cast<size_t>(size) / 4);
    file.read(reinterpret_cast<char*>(code.data()), size);

    VkShaderModuleCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = static_cast<size_t>(size);
    ci.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(env.device, &ci, nullptr, &module) != VK_SUCCESS) return VK_NULL_HANDLE;
    return module;
}

// One-shot layout transition for internal images (UNDEFINED -> GENERAL at first use).
inline void transitionImage(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout,
                            VkImageLayout newLayout) {
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                         &barrier);
}

// Execution+memory barrier between two compute passes.
inline void computeBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0,
                         nullptr);
}

inline VkPipeline createComputePipeline(const VulkanEnv& env, VkPipelineLayout layout,
                                        VkShaderModule module) {
    VkComputePipelineCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = module;
    ci.stage.pName = "main";
    ci.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (sr::createComputePipeline(env, ci, pipeline) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return pipeline;
}

} // namespace sgsr
} // namespace sr
