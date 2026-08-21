#include "renderer/core/VulkanContext.h"

#include "renderer/core/PathUtil.h"
#include "renderer/core/Vma.h"
#include "renderer/core/Window.h"
#include "upscalers/UpscalerFactory.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace sr {

namespace {

constexpr uint32_t kApiVersion = VK_API_VERSION_1_3;

bool isDiscrete(VkPhysicalDeviceType type) {
    return type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
           type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
}

} // namespace

bool VulkanContext::create(Window& window) {
    std::vector<const char*> instanceExt = {VK_KHR_SURFACE_EXTENSION_NAME,
                                            VK_KHR_WIN32_SURFACE_EXTENSION_NAME};

    // Optional validation layer (only if the loader actually provides it).
    const char* validationLayer = "VK_LAYER_KHRONOS_validation";
    bool useValidation = false;
    std::vector<VkLayerProperties> availableLayers;
    {
        uint32_t layerCount = 0;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        availableLayers.resize(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
        for (const auto& l : availableLayers) {
            if (std::strcmp(l.layerName, validationLayer) == 0) { useValidation = true; break; }
        }
    }
    if (std::getenv("SR_NO_VALIDATION")) useValidation = false;  // opt-out for end users
    std::vector<const char*> enabledLayers;
    if (useValidation) enabledLayers.push_back(validationLayer);

    // Plugin-declared instance requirements (registered via
    // SR_REGISTER_VULKAN_DEVICE_NEEDS): instance extensions and layers.
    // Hooks run FIRST (they may set VK_ADD_LAYER_PATH for their own layer
    // manifests); only afterwards do we enumerate what the loader offers.
    {
        std::vector<const char*> wantedExts;
        std::vector<const char*> wantedLayers;
        for (const sr::VulkanDeviceNeeds& needs : sr::collectVulkanDeviceNeeds()) {
            if (needs.appendInstanceExtensions) needs.appendInstanceExtensions(wantedExts);
            if (needs.appendInstanceLayers) needs.appendInstanceLayers(wantedLayers);
        }

        uint32_t extCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> supportedExts(extCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extCount, supportedExts.data());
        uint32_t layerCount = 0;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayersNow(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayersNow.data());

        for (const char* w : wantedExts) {
            bool supported = false;
            for (const auto& e : supportedExts) {
                if (std::strcmp(e.extensionName, w) == 0) { supported = true; break; }
            }
            bool already = false;
            for (const char* existing : instanceExt) {
                if (std::strcmp(existing, w) == 0) { already = true; break; }
            }
            if (!supported) {
                std::fprintf(stderr, "warning: plugin-required instance extension %s unsupported\n", w);
            } else if (!already) {
                instanceExt.push_back(w);
            }
        }
        for (const char* w : wantedLayers) {
            bool supported = false;
            for (const auto& l : availableLayersNow) {
                if (std::strcmp(l.layerName, w) == 0) { supported = true; break; }
            }
            bool already = false;
            for (const char* existing : enabledLayers) {
                if (std::strcmp(existing, w) == 0) { already = true; break; }
            }
            if (!supported) {
                std::fprintf(stderr, "warning: plugin-required instance layer %s unavailable\n", w);
            } else if (!already) {
                enabledLayers.push_back(w);
            }
        }
    }

    VkApplicationInfo app = {};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "sr_compare";
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.pEngineName = "sr_compare";
    app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app.apiVersion = kApiVersion;

    VkInstanceCreateInfo ici = {};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = static_cast<uint32_t>(instanceExt.size());
    ici.ppEnabledExtensionNames = instanceExt.data();
    ici.enabledLayerCount = static_cast<uint32_t>(enabledLayers.size());
    ici.ppEnabledLayerNames = enabledLayers.empty() ? nullptr : enabledLayers.data();
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) return false;

    VkWin32SurfaceCreateInfoKHR sci = {};
    sci.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    sci.hwnd = window.hwnd();
    sci.hinstance = window.hinstance();
    if (vkCreateWin32SurfaceKHR(instance, &sci, nullptr, &surface) != VK_SUCCESS) {
        destroy();
        return false;
    }

    // Pick a physical device with graphics + present support and the features we need.
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) { destroy(); return false; }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    int chosen = -1;
    VkPhysicalDeviceProperties chosenProps{};
    for (uint32_t i = 0; i < deviceCount; ++i) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        if (props.apiVersion < kApiVersion) continue;

        uint32_t qfCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &qfCount, qfs.data());

        uint32_t gfx = 0xFFFFFFFFu, present = 0xFFFFFFFFu;
        for (uint32_t q = 0; q < qfCount; ++q) {
            if ((qfs[q].queueFlags & VK_QUEUE_GRAPHICS_BIT) && gfx == 0xFFFFFFFFu) gfx = q;
            VkBool32 supportsPresent = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(devices[i], q, surface, &supportsPresent);
            if (supportsPresent && present == 0xFFFFFFFFu) present = q;
        }
        if (gfx == 0xFFFFFFFFu || present == 0xFFFFFFFFu) continue;

        // Prefer discrete GPUs, fall back to any suitable one.
        if (chosen == -1 || (isDiscrete(props.deviceType) && !isDiscrete(chosenProps.deviceType))) {
            chosen = static_cast<int>(i);
            chosenProps = props;
            graphicsQueueFamily = gfx;
            presentQueueFamily = present;
        }
        if (isDiscrete(props.deviceType)) break;
    }
    if (chosen < 0) { destroy(); return false; }
    physicalDevice = devices[static_cast<uint32_t>(chosen)];

    vkGetPhysicalDeviceProperties(physicalDevice, &properties);
    vkGetPhysicalDeviceFeatures(physicalDevice, &features);
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
    minUniformBufferOffsetAlignment = properties.limits.minUniformBufferOffsetAlignment;

    // Query memory budget (VK_EXT_memory_budget) if supported, and remember
    // the full supported-extension list for plugin requirement filtering.
    std::vector<VkExtensionProperties> supportedExts;
    {
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, nullptr);
        supportedExts.resize(extCount);
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, supportedExts.data());
        for (const auto& e : supportedExts) {
            if (std::strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) {
                hasMemoryBudget = true;
                break;
            }
        }
        memoryBudget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        VkPhysicalDeviceMemoryProperties2 memProps2 = {};
        memProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
        memProps2.pNext = &memoryBudget;
        vkGetPhysicalDeviceMemoryProperties2(physicalDevice, &memProps2);
    }

    // Feature chain root: VkPhysicalDeviceFeatures2 (so pEnabledFeatures can
    // stay nullptr — required once any feature struct appears in pNext).
    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.features.samplerAnisotropy = features.samplerAnisotropy;
    features2.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;  // XeSS
    features2.features.independentBlend = features.independentBlend;    // transparency pass (per-attachment blend)

    // Vulkan 1.2 features shared by plugins (each sType may appear only once
    // in the chain, so plugins must not add their own v12/v13 nodes).
    VkPhysicalDeviceVulkan12Features v12 = {};
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.shaderInt8 = VK_TRUE;                      // XeSS DP4a, Arm NSS int8 graph
    v12.vulkanMemoryModel = VK_TRUE;               // Arm NSS SDK shaders
    v12.vulkanMemoryModelDeviceScope = VK_TRUE;    // Arm NSS device-scope atomics

    // Enable dynamic rendering (core in Vulkan 1.3) plus the swapchain/budget extensions.
    VkPhysicalDeviceVulkan13Features v13 = {};
    v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    v13.dynamicRendering = VK_TRUE;
    v13.synchronization2 = VK_TRUE;  // Arm NSS emulation layer uses barrier2 calls
    v13.privateData = VK_TRUE;       // Streamline/DLSS allocates private data slots
    v13.shaderIntegerDotProduct = VK_TRUE;  // XeSS DP4a path (core in 1.3)
    v13.shaderDemoteToHelperInvocation = VK_TRUE; // GLSL discard in GBuffer shaders
    features2.pNext = &v12;
    v12.pNext = &v13;

    std::vector<const char*> deviceExt = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    if (hasMemoryBudget) deviceExt.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);

    // Plugin-declared device requirements (registered via
    // SR_REGISTER_VULKAN_DEVICE_NEEDS): extra extensions (skipped with a
    // warning when unsupported) and pNext feature chains.
    VkBaseOutStructure* featureTail = reinterpret_cast<VkBaseOutStructure*>(&v13);
    for (const sr::VulkanDeviceNeeds& needs : sr::collectVulkanDeviceNeeds()) {
        if (needs.appendExtensions) {
            std::vector<const char*> wanted;
            needs.appendExtensions(wanted);
            for (const char* w : wanted) {
                bool supported = false;
                for (const auto& e : supportedExts) {
                    if (std::strcmp(e.extensionName, w) == 0) { supported = true; break; }
                }
                bool already = false;
                for (const char* existing : deviceExt) {
                    if (std::strcmp(existing, w) == 0) { already = true; break; }
                }
                if (!supported) {
                    std::fprintf(stderr, "warning: plugin-required device extension %s unsupported\n", w);
                } else if (!already) {
                    deviceExt.push_back(w);
                }
            }
        }
        if (needs.featureChain) {
            if (const void* head = needs.featureChain()) {
                featureTail->pNext = reinterpret_cast<VkBaseOutStructure*>(const_cast<void*>(head));
                while (featureTail->pNext) featureTail = featureTail->pNext;
            }
        }
    }

    float queuePriority = 1.f;
    const bool sameFamily = graphicsQueueFamily == presentQueueFamily;
    std::vector<VkDeviceQueueCreateInfo> qcis;
    std::vector<uint32_t> uniqueFamilies = {graphicsQueueFamily};
    if (!sameFamily) uniqueFamilies.push_back(presentQueueFamily);
    for (uint32_t f : uniqueFamilies) {
        VkDeviceQueueCreateInfo q = {};
        q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        q.queueFamilyIndex = f;
        q.queueCount = 1;
        q.pQueuePriorities = &queuePriority;
        qcis.push_back(q);
    }

    VkDeviceCreateInfo dci = {};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &features2;  // features2 -> v12 -> v13 -> plugin chains
    dci.queueCreateInfoCount = static_cast<uint32_t>(qcis.size());
    dci.pQueueCreateInfos = qcis.data();
    dci.enabledExtensionCount = static_cast<uint32_t>(deviceExt.size());
    dci.ppEnabledExtensionNames = deviceExt.data();
    dci.pEnabledFeatures = nullptr;  // features live in the pNext chain (VUID-00373)
    if (vkCreateDevice(physicalDevice, &dci, nullptr, &device) != VK_SUCCESS) {
        destroy();
        return false;
    }

    vkGetDeviceQueue(device, graphicsQueueFamily, 0, &graphicsQueue);
    vkGetDeviceQueue(device, presentQueueFamily, 0, &presentQueue);

    VkCommandPoolCreateInfo pool = {};
    pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool.queueFamilyIndex = graphicsQueueFamily;
    vkCreateCommandPool(device, &pool, nullptr, &oneShotPool);
    vkCreateCommandPool(device, &pool, nullptr, &framePool);

    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = physicalDevice;
    allocatorInfo.device = device;
    allocatorInfo.instance = instance;
    allocatorInfo.vulkanApiVersion = kApiVersion;
    if (vmaCreateAllocator(&allocatorInfo, &allocator) != VK_SUCCESS) {
        destroy();
        return false;
    }

    // Persistent pipeline cache next to the exe.  Incompatible or corrupt
    // initial data is silently discarded by the driver (empty cache).
    {
        const std::string cachePath = exeDir() + "/pipeline.cache";
        std::vector<char> initialData;
        if (FILE* f = std::fopen(cachePath.c_str(), "rb")) {
            std::fseek(f, 0, SEEK_END);
            const long len = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            if (len > 0) {
                initialData.resize(static_cast<size_t>(len));
                const size_t got = std::fread(initialData.data(), 1, initialData.size(), f);
                initialData.resize(got);
            }
            std::fclose(f);
        }
        VkPipelineCacheCreateInfo cacheCi = {};
        cacheCi.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        cacheCi.initialDataSize = initialData.size();
        cacheCi.pInitialData = initialData.empty() ? nullptr : initialData.data();
        if (vkCreatePipelineCache(device, &cacheCi, nullptr, &pipelineCache) != VK_SUCCESS)
            std::fprintf(stderr, "warning: vkCreatePipelineCache failed, caching disabled\n");
    }

    return true;
}

void VulkanContext::destroy() {
    if (!device) {
        // Instance-only partial state.
        if (surface) { vkDestroySurfaceKHR(instance, surface, nullptr); surface = VK_NULL_HANDLE; }
        if (instance) { vkDestroyInstance(instance, nullptr); instance = VK_NULL_HANDLE; }
        return;
    }
    if (oneShotPool) { vkDestroyCommandPool(device, oneShotPool, nullptr); oneShotPool = VK_NULL_HANDLE; }
    if (framePool) { vkDestroyCommandPool(device, framePool, nullptr); framePool = VK_NULL_HANDLE; }
    if (pipelineCache) {
        // Persist the merged cache next to the exe for the next run.
        std::lock_guard<std::mutex> lk(pipelineMutex);
        size_t size = 0;
        if (vkGetPipelineCacheData(device, pipelineCache, &size, nullptr) == VK_SUCCESS && size > 0) {
            std::vector<char> data(size);
            if (vkGetPipelineCacheData(device, pipelineCache, &size, data.data()) == VK_SUCCESS) {
                const std::string cachePath = exeDir() + "/pipeline.cache";
                if (FILE* f = std::fopen(cachePath.c_str(), "wb")) {
                    std::fwrite(data.data(), 1, size, f);
                    std::fclose(f);
                }
            }
        }
        vkDestroyPipelineCache(device, pipelineCache, nullptr);
        pipelineCache = VK_NULL_HANDLE;
    }
    if (allocator) { vmaDestroyAllocator(allocator); allocator = VK_NULL_HANDLE; }
    vkDestroyDevice(device, nullptr);
    device = VK_NULL_HANDLE;
    if (surface) { vkDestroySurfaceKHR(instance, surface, nullptr); surface = VK_NULL_HANDLE; }
    if (instance) { vkDestroyInstance(instance, nullptr); instance = VK_NULL_HANDLE; }
}

} // namespace sr
