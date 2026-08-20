#pragma once
// ============================================================================
// VK_EXT_memory_budget query helper.  Returns heap budget/usage so the
// benchmark module can report per-algorithm VRAM pressure.
// ============================================================================
#include "renderer/core/Vk.h"
#include "renderer/core/VulkanContext.h"

#include <cstdint>

namespace sr {

struct MemoryBudgetInfo {
    VkDeviceSize heapBudget[VK_MAX_MEMORY_HEAPS] = {};
    VkDeviceSize heapUsage[VK_MAX_MEMORY_HEAPS] = {};
    uint32_t heapCount = 0;
};

// Live query.  Requires the VK_EXT_memory_budget device extension; if it was
// not enabled the budget entries stay zero (usage is still reported).
MemoryBudgetInfo queryMemoryBudget(const VulkanContext& ctx);

} // namespace sr
