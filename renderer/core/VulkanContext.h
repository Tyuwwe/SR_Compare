#pragma once
// ============================================================================
// VulkanContext — owns instance / device / queues / command pools and exports
// the subset of itself that upscaler plugins are allowed to use (sr::VulkanEnv).
// ============================================================================
#include "renderer/core/Vk.h"
#include "upscalers/IUpscaler.h"

#include <mutex>
#include <vector>

namespace sr {

class Window;

struct VulkanContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    uint32_t graphicsQueueFamily = 0;
    VkQueue presentQueue = VK_NULL_HANDLE;
    uint32_t presentQueueFamily = 0;
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    VkCommandPool oneShotPool = VK_NULL_HANDLE;  // one-time submits (upscalers)
    VkCommandPool framePool = VK_NULL_HANDLE;    // per-frame recording

    // VMA allocator — all device memory owned by VkUtil helpers goes through
    // this instead of per-resource vkAllocateMemory (Bistro's texture count
    // otherwise approaches the driver's allocation limit).
    VmaAllocator allocator = VK_NULL_HANDLE;

    // Guards every host access to graphicsQueue/presentQueue (submit, present,
    // queue-wait-idle).  The GUI async loader submits uploads from a worker
    // thread while the main thread keeps rendering, so queue access needs
    // external synchronization; command pools stay per-thread instead.
    mutable std::mutex queueMutex;

    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceFeatures features{};
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    VkPhysicalDeviceMemoryBudgetPropertiesEXT memoryBudget{};
    bool hasMemoryBudget = false;
    VkDeviceSize minUniformBufferOffsetAlignment = 256;

    bool create(Window& window);
    void destroy();

    // Plugin-facing subset.  getInstanceProcAddr lets plugins resolve extension
    // entry points (they must not create their own device/instance).
    sr::VulkanEnv toEnv() const {
        sr::VulkanEnv env;
        env.instance = instance;
        env.physicalDevice = physicalDevice;
        env.device = device;
        env.graphicsQueue = graphicsQueue;
        env.graphicsQueueFamily = graphicsQueueFamily;
        env.commandPool = oneShotPool;
        env.queueMutex = &queueMutex;
        env.getInstanceProcAddr = vkGetInstanceProcAddr;
        return env;
    }

    // Variant with an explicit command pool: the async loader hands plugins a
    // worker-private pool so pool access stays single-threaded.
    sr::VulkanEnv toEnv(VkCommandPool commandPool) const {
        sr::VulkanEnv env = toEnv();
        env.commandPool = commandPool;
        return env;
    }
};

} // namespace sr
