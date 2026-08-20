#pragma once
// ============================================================================
// Upscaler registry — self-registration based plugin discovery.
//
// Each upscaler module (upscalers/<name>/) registers itself at static-init
// time with SR_REGISTER_UPSCALER("name", &createFn) in one of its .cpp files.
// Modules are static libraries linked whole-archive into the `upscalers` lib
// (see upscalers/CMakeLists.txt), so their static initializers always run.
//
// Plugins must NOT edit this file or UpscalerFactory.cpp — registration is
// automatic.  To add a new algorithm, create upscalers/<name>/ with a
// CMakeLists.txt defining a STATIC target named `upscaler_<name>` and call
// SR_REGISTER_UPSCALER in one of its sources.
// ============================================================================
#include "upscalers/IUpscaler.h"

#include <memory>
#include <string>
#include <vector>

namespace sr {

using UpscalerCreateFn = std::unique_ptr<IUpscaler> (*)();

// Called by SR_REGISTER_UPSCALER static initializers.  Returns false if the
// name is already taken (later registration is ignored).
bool registerUpscaler(const char* name, UpscalerCreateFn fn);

// Create an upscaler by registered name; nullptr if unknown.
std::unique_ptr<IUpscaler> createUpscaler(const char* name);

// All registered names, in registration order.
std::vector<std::string> listUpscalers();

// ---------------------------------------------------------------------------
// Global plugin gate override.  The dlss/xess/nss modules gate their device
// requirement hooks (and slInit) on command-line substrings so unified
// builds stay clean for runs that do not use them.  Interactive hosts that
// select plugins at runtime (the `gui` mode) instead call
// setAllPluginsEnabled(true) BEFORE any Vulkan object is created, which
// makes every gate evaluate as requested.  Default: false.
// ---------------------------------------------------------------------------
void setAllPluginsEnabled(bool enabled);
bool allPluginsEnabled();

// ---------------------------------------------------------------------------
// Vulkan device requirements: plugins that need extra device extensions or
// feature structs at vkCreateDevice time (e.g. XeSS needs
// shaderIntegerDotProduct + VK_EXT_mutable_descriptor_type) register a hook
// with SR_REGISTER_VULKAN_DEVICE_NEEDS.  The renderer consumes these once,
// before creating the device.  Unsupported extensions are skipped with a
// warning (the plugin's isAvailable() must then fail gracefully).
// ---------------------------------------------------------------------------
struct VulkanDeviceNeeds {
    // Append required device extension names (only those strictly required).
    void (*appendExtensions)(std::vector<const char*>& deviceExts) = nullptr;
    // Return the head of a plugin-owned (static storage) pNext feature chain
    // to append to VkDeviceCreateInfo, or nullptr.
    const void* (*featureChain)() = nullptr;
    // Append required instance extensions / layers (e.g. a software-emulation
    // layer providing vendor extension implementations).  Layers are skipped
    // with a warning when the loader does not provide them.
    void (*appendInstanceExtensions)(std::vector<const char*>& instanceExts) = nullptr;
    void (*appendInstanceLayers)(std::vector<const char*>& instanceLayers) = nullptr;
};

bool registerVulkanDeviceNeeds(const VulkanDeviceNeeds& needs);
std::vector<VulkanDeviceNeeds> collectVulkanDeviceNeeds();

} // namespace sr

// Register a plugin: SR_REGISTER_UPSCALER("fsr2", &sr::CreateFsr2Upscaler);
#define SR_CONCAT_INNER(a, b) a##b
#define SR_CONCAT(a, b) SR_CONCAT_INNER(a, b)
#define SR_REGISTER_UPSCALER(pluginName, createFn)                                     \
    namespace {                                                                        \
    const bool SR_CONCAT(kSrRegistered_, __LINE__) =                                   \
        ::sr::registerUpscaler(pluginName, createFn);                                  \
    }

// Register device-creation requirements: SR_REGISTER_VULKAN_DEVICE_NEEDS(needs);
// where `needs` is a ::sr::VulkanDeviceNeeds value (function pointers must
// point to static-storage data).
#define SR_REGISTER_VULKAN_DEVICE_NEEDS(needsValue)                                    \
    namespace {                                                                        \
    const bool SR_CONCAT(kSrNeeds_, __LINE__) =                                        \
        ::sr::registerVulkanDeviceNeeds(needsValue);                                   \
    }
