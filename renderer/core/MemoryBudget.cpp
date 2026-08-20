#include "renderer/core/MemoryBudget.h"

namespace sr {

MemoryBudgetInfo queryMemoryBudget(const VulkanContext& ctx) {
    MemoryBudgetInfo out;
    if (!ctx.hasMemoryBudget) return out;

    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = {};
    budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    VkPhysicalDeviceMemoryProperties2 props2 = {};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    props2.pNext = &budget;
    vkGetPhysicalDeviceMemoryProperties2(ctx.physicalDevice, &props2);

    out.heapCount = ctx.memoryProperties.memoryHeapCount;
    for (uint32_t i = 0; i < out.heapCount && i < VK_MAX_MEMORY_HEAPS; ++i) {
        out.heapBudget[i] = budget.heapBudget[i];
        out.heapUsage[i] = budget.heapUsage[i];
    }
    return out;
}

} // namespace sr
