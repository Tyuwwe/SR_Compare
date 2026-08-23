#include "renderer/core/MemoryBudget.h"

#include "renderer/core/Vma.h"

namespace sr {

MemoryBudgetInfo queryMemoryBudget(const VulkanContext& ctx) {
    MemoryBudgetInfo out;
    out.heapCount = ctx.memoryProperties.memoryHeapCount;

    if (ctx.hasMemoryBudget) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = {};
        budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        VkPhysicalDeviceMemoryProperties2 props2 = {};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
        props2.pNext = &budget;
        vkGetPhysicalDeviceMemoryProperties2(ctx.physicalDevice, &props2);

        for (uint32_t i = 0; i < out.heapCount && i < VK_MAX_MEMORY_HEAPS; ++i) {
            out.heapBudget[i] = budget.heapBudget[i];
            out.heapUsage[i] = budget.heapUsage[i];
        }
    }

    if (ctx.allocator) {
        VmaBudget budgets[VK_MAX_MEMORY_HEAPS] = {};
        vmaGetHeapBudgets(ctx.allocator, budgets);
        for (uint32_t i = 0; i < out.heapCount && i < VK_MAX_MEMORY_HEAPS; ++i) {
            out.vmaBlockBytes[i] = budgets[i].statistics.blockBytes;
            out.vmaAllocationBytes[i] = budgets[i].statistics.allocationBytes;
        }
    }
    return out;
}

} // namespace sr
