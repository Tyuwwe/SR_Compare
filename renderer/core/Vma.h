#pragma once
// ============================================================================
// Single include point for VMA (Vulkan Memory Allocator).  The header lives
// under third_party/vma (SYSTEM include, warnings silenced via /external:W0);
// this wrapper only ensures Vulkan types are declared first.
// ============================================================================
#include "renderer/core/Vk.h"

#include <vk_mem_alloc.h>
