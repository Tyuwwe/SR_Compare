#pragma once
// ============================================================================
// Single include point for Vulkan.  Defines the Win32 platform surface macro
// and silences warnings from the SDK headers under MSVC /W4.
// Include this BEFORE upscalers/IUpscaler.h (which also includes vulkan.h).
// ============================================================================
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif

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
