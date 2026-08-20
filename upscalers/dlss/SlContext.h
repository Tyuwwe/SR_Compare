#pragma once
// ============================================================================
// SlContext — process-wide Streamline (sl.interposer) lifecycle manager.
//
// Both DLSS plugins ("dlss-k" / "dlss-m") share a single slInit/slShutdown
// pair: slInit runs at most once per process (lazily, on first use) and
// slShutdown runs when the last DLSS upscaler instance is destroyed (while
// the Vulkan device is still alive, as required by the SL programming guide).
//
// This module also implements the VulkanDeviceNeeds hooks registered with
// SR_REGISTER_VULKAN_DEVICE_NEEDS: the renderer invokes them before
// vkCreateInstance/vkCreateDevice, which is where we perform slInit (before
// any vkCreate* call, as Streamline requires).
//
// Integration mode: the final executable resolves the hooked Vulkan entry
// points (vkCreateInstance/vkCreateDevice/...) to sl.interposer.dll (checked
// via `dumpbin /imports`), so the SL proxies handle device association and
// inject all required extensions/features at creation time.  The device-needs
// hooks therefore only trigger slInit and contribute no extensions; the
// requirements reported by slGetFeatureRequirements are logged for reference.
// ============================================================================
#include "upscalers/IUpscaler.h"

#include <vector>

namespace sl_dlss {

// Runs slInit once (idempotent).  Safe to call before Vulkan instance
// creation (from the device-needs hooks) or any time after.
bool ensureInitialized();
bool initialized();

// slIsFeatureSupported(kFeatureDLSS) for the given physical device.
// Returns false (graceful degradation) when SL/DLSS is unavailable.
bool dlssSupported(VkPhysicalDevice physicalDevice);

// Records the created Vulkan device.  With proxy-based linking the proxies
// already associate the device at vkCreateDevice time, so this only tracks
// the binding; idempotent per device.
bool bindDevice(const sr::VulkanEnv& env);

// Reference counting of live DLSS upscaler instances; the last release
// shuts Streamline down (the Vulkan device must still be alive).
void addRef();
void release();
// slShutdown when initialized but no live upscaler holds a reference
// (used by instances whose init() failed halfway).
void shutdownIfIdle();

// Hooks registered with SR_REGISTER_VULKAN_DEVICE_NEEDS (function pointers
// must have static storage; defined in SlContext.cpp).
void appendInstanceExtensionsHook(std::vector<const char*>& instanceExts);
void appendDeviceExtensionsHook(std::vector<const char*>& deviceExts);
const void* featureChainHook();

} // namespace sl_dlss
