#pragma once
// ============================================================================
// VkHelpers — small Vulkan helpers shared by the upscaler plugins.
// ============================================================================
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <vector>

#include <vulkan/vulkan.h>

#include "upscalers/IUpscaler.h"

namespace sr {

// vkCreateGraphicsPipelines / vkCreateComputePipelines through the renderer's
// persistent pipeline cache (env.pipelineCache).  env.pipelineMutex (when set)
// serializes host access: the GUI async loader creates pipelines off-thread.
inline VkResult createGraphicsPipeline(const VulkanEnv& env,
                                       const VkGraphicsPipelineCreateInfo& ci, VkPipeline& out) {
    std::unique_lock<std::mutex> lk;
    if (env.pipelineMutex) lk = std::unique_lock<std::mutex>(*env.pipelineMutex);
    return vkCreateGraphicsPipelines(env.device, env.pipelineCache, 1, &ci, nullptr, &out);
}

inline VkResult createComputePipeline(const VulkanEnv& env,
                                      const VkComputePipelineCreateInfo& ci, VkPipeline& out) {
    std::unique_lock<std::mutex> lk;
    if (env.pipelineMutex) lk = std::unique_lock<std::mutex>(*env.pipelineMutex);
    return vkCreateComputePipelines(env.device, env.pipelineCache, 1, &ci, nullptr, &out);
}

// Finds the memory type index matching the required property flags from the
// bits allowed for a resource.  Returns 0xFFFFFFFF when no type matches.
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

// Loads a SPIR-V shader module from a file.  Returns VK_NULL_HANDLE when the
// file cannot be opened, its size is invalid, or module creation fails.
inline VkShaderModule loadShader(VkDevice device, const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "failed to open shader %s\n", path);
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
    if (vkCreateShaderModule(device, &ci, nullptr, &module) != VK_SUCCESS) return VK_NULL_HANDLE;
    return module;
}

} // namespace sr
