#pragma once
// ============================================================================
// Single include point for Vulkan; silences warnings from the SDK headers
// under MSVC /W4.  The platform surface is created through SDL
// (SDL_Vulkan_CreateSurface), so no VK_USE_PLATFORM_* macro is needed here.
// Include this BEFORE upscalers/IUpscaler.h (which also includes vulkan.h).
// ============================================================================
#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#include <vulkan/vulkan.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

// VMA handle forward declarations.  vk_mem_alloc.h itself is only included
// (via renderer/core/Vma.h) in translation units that call VMA functions.
struct VmaAllocator_T;
typedef struct VmaAllocator_T* VmaAllocator;
struct VmaAllocation_T;
typedef struct VmaAllocation_T* VmaAllocation;
