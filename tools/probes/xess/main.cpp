// Probe: query XeSS Vulkan requirements and validate full context init on this machine.
#include <vulkan/vulkan.h>
#include <xess/xess.h>
#include <xess/xess_vk.h>

#include <cstdio>
#include <cstring>
#include <vector>

static const char* resStr(xess_result_t r) {
    switch (r) {
        case XESS_RESULT_SUCCESS: return "SUCCESS";
        case XESS_RESULT_WARNING_OLD_DRIVER: return "WARNING_OLD_DRIVER";
        case XESS_RESULT_ERROR_UNSUPPORTED_DEVICE: return "ERROR_UNSUPPORTED_DEVICE";
        case XESS_RESULT_ERROR_UNSUPPORTED_DRIVER: return "ERROR_UNSUPPORTED_DRIVER";
        case XESS_RESULT_ERROR_UNINITIALIZED: return "ERROR_UNINITIALIZED";
        case XESS_RESULT_ERROR_INVALID_ARGUMENT: return "ERROR_INVALID_ARGUMENT";
        case XESS_RESULT_ERROR_DEVICE_OUT_OF_MEMORY: return "ERROR_DEVICE_OUT_OF_MEMORY";
        case XESS_RESULT_ERROR_DEVICE: return "ERROR_DEVICE";
        case XESS_RESULT_ERROR_NOT_IMPLEMENTED: return "ERROR_NOT_IMPLEMENTED";
        case XESS_RESULT_ERROR_INVALID_CONTEXT: return "ERROR_INVALID_CONTEXT";
        case XESS_RESULT_ERROR_UNSUPPORTED: return "ERROR_UNSUPPORTED";
        case XESS_RESULT_ERROR_CANT_LOAD_LIBRARY: return "ERROR_CANT_LOAD_LIBRARY";
        case XESS_RESULT_ERROR_WRONG_CALL_ORDER: return "ERROR_WRONG_CALL_ORDER";
        default: return "UNKNOWN";
    }
}

static const char* sTypeName(VkStructureType t) {
    switch (t) {
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2: return "PhysicalDeviceFeatures2";
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES: return "Vulkan11Features";
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES: return "Vulkan12Features";
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: return "Vulkan13Features";
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT: return "MutableDescriptorTypeFeaturesEXT";
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES: return "ShaderIntegerDotProductFeatures";
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES: return "ShaderFloat16Int8Features";
        default: return "?";
    }
}

int main() {
    xess_version_t ver{};
    std::printf("xessGetVersion: %s -> %u.%u.%u\n", resStr(xessGetVersion(&ver)), ver.major, ver.minor, ver.patch);

    // --- instance requirements ---
    uint32_t instExtCount = 0;
    const char* const* instExts = nullptr;
    uint32_t minApi = 0;
    xess_result_t r = xessVKGetRequiredInstanceExtensions(&instExtCount, &instExts, &minApi);
    std::printf("xessVKGetRequiredInstanceExtensions: %s, count=%u, minApi=%u.%u.%u\n", resStr(r),
                instExtCount, VK_API_VERSION_MAJOR(minApi), VK_API_VERSION_MINOR(minApi), VK_API_VERSION_PATCH(minApi));
    std::vector<const char*> instanceExt = {VK_KHR_SURFACE_EXTENSION_NAME, "VK_KHR_win32_surface"};
    for (uint32_t i = 0; i < instExtCount; ++i) {
        std::printf("  instExt: %s\n", instExts[i]);
        instanceExt.push_back(instExts[i]);
    }

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "xess_probe";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = (uint32_t)instanceExt.size();
    ici.ppEnabledExtensionNames = instanceExt.data();
    VkInstance instance = VK_NULL_HANDLE;
    VkResult vr = vkCreateInstance(&ici, nullptr, &instance);
    std::printf("vkCreateInstance: %d\n", (int)vr);
    if (vr != VK_SUCCESS) return 1;

    uint32_t pdCount = 0;
    vkEnumeratePhysicalDevices(instance, &pdCount, nullptr);
    std::vector<VkPhysicalDevice> pds(pdCount);
    vkEnumeratePhysicalDevices(instance, &pdCount, pds.data());
    VkPhysicalDevice pd = pds[0];
    for (auto p : pds) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(p, &props);
        std::printf("  device: %s type=%d\n", props.deviceName, (int)props.deviceType);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) pd = p;
    }
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    std::printf("selected: %s\n", props.deviceName);

    // --- device extension requirements ---
    uint32_t devExtCount = 0;
    const char* const* devExts = nullptr;
    r = xessVKGetRequiredDeviceExtensions(instance, pd, &devExtCount, &devExts);
    std::printf("xessVKGetRequiredDeviceExtensions: %s, count=%u\n", resStr(r), devExtCount);
    std::vector<const char*> deviceExt;
    for (uint32_t i = 0; i < devExtCount; ++i) {
        std::printf("  devExt: %s\n", devExts[i]);
        deviceExt.push_back(devExts[i]);
    }

    // --- feature requirements ---
    void* features = nullptr;
    r = xessVKGetRequiredDeviceFeatures(instance, pd, &features);
    std::printf("xessVKGetRequiredDeviceFeatures: %s, chain=%p\n", resStr(r), features);
    for (const VkBaseInStructure* s = (const VkBaseInStructure*)features; s; s = s->pNext) {
        std::printf("  chain node: sType=%d (%s)\n", (int)s->sType, sTypeName(s->sType));
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) {
            auto* f = (const VkPhysicalDeviceFeatures2*)s;
            std::printf("    shaderStorageImageWriteWithoutFormat=%d\n",
                        (int)f->features.shaderStorageImageWriteWithoutFormat);
        } else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT) {
            auto* f = (const VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT*)s;
            std::printf("    mutableDescriptorType=%d\n", (int)f->mutableDescriptorType);
        } else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES) {
            auto* f = (const VkPhysicalDeviceShaderIntegerDotProductFeatures*)s;
            std::printf("    shaderIntegerDotProduct=%d\n", (int)f->shaderIntegerDotProduct);
        } else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES) {
            auto* f = (const VkPhysicalDeviceShaderFloat16Int8Features*)s;
            std::printf("    shaderFloat16=%d shaderInt8=%d\n", (int)f->shaderFloat16, (int)f->shaderInt8);
        } else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) {
            auto* f = (const VkPhysicalDeviceVulkan12Features*)s;
            std::printf("    shaderInt8=%d shaderFloat16=%d\n", (int)f->shaderInt8, (int)f->shaderFloat16);
        } else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES) {
            auto* f = (const VkPhysicalDeviceVulkan13Features*)s;
            std::printf("    shaderIntegerDotProduct=%d dynamicRendering=%d\n",
                        (int)f->shaderIntegerDotProduct, (int)f->dynamicRendering);
        }
    }

    // --- create device mimicking the renderer (pEnabledFeatures + pNext chain) ---
    float prio = 1.f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = 0;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkPhysicalDeviceVulkan13Features v13{};
    v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    v13.dynamicRendering = VK_TRUE;
    // append XeSS chain after v13, like the renderer does
    VkBaseOutStructure* tail = (VkBaseOutStructure*)&v13;
    if (features) {
        tail->pNext = (VkBaseOutStructure*)features;
    }

    VkPhysicalDeviceFeatures enabled{};
    enabled.samplerAnisotropy = VK_TRUE;

    deviceExt.insert(deviceExt.begin(), VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &v13;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)deviceExt.size();
    dci.ppEnabledExtensionNames = deviceExt.data();
    dci.pEnabledFeatures = &enabled;
    VkDevice device = VK_NULL_HANDLE;
    vr = vkCreateDevice(pd, &dci, nullptr, &device);
    std::printf("vkCreateDevice (renderer-style): %d\n", (int)vr);
    if (vr != VK_SUCCESS) { vkDestroyInstance(instance, nullptr); return 1; }

    // --- XeSS context ---
    xess_context_handle_t ctx = nullptr;
    r = xessVKCreateContext(instance, pd, device, &ctx);
    std::printf("xessVKCreateContext: %s\n", resStr(r));
    if (r != XESS_RESULT_SUCCESS && r != XESS_RESULT_WARNING_OLD_DRIVER) {
        vkDestroyDevice(device, nullptr); vkDestroyInstance(instance, nullptr); return 1;
    }

    r = xessIsOptimalDriver(ctx);
    std::printf("xessIsOptimalDriver: %s\n", resStr(r));

    xess_2d_t outRes{1920, 1080};
    xess_properties_t xprops{};
    r = xessGetProperties(ctx, &outRes, &xprops);
    std::printf("xessGetProperties: %s, tempBuffer=%llu tempTexture=%llu descriptors=%u\n", resStr(r),
                (unsigned long long)xprops.tempBufferHeapSize, (unsigned long long)xprops.tempTextureHeapSize,
                xprops.requiredDescriptorCount);

    xess_2d_t opt{}, mn{}, mx{};
    r = xessGetOptimalInputResolution(ctx, &outRes, XESS_QUALITY_SETTING_QUALITY, &opt, &mn, &mx);
    std::printf("QUALITY: %s optimal=%ux%u min=%ux%u max=%ux%u\n", resStr(r), opt.x, opt.y, mn.x, mn.y, mx.x, mx.y);
    r = xessGetOptimalInputResolution(ctx, &outRes, XESS_QUALITY_SETTING_BALANCED, &opt, &mn, &mx);
    std::printf("BALANCED: %s optimal=%ux%u min=%ux%u max=%ux%u\n", resStr(r), opt.x, opt.y, mn.x, mn.y, mx.x, mx.y);

    const uint32_t initFlags = XESS_INIT_FLAG_NONE;
    r = xessVKBuildPipelines(ctx, VK_NULL_HANDLE, true, initFlags);
    std::printf("xessVKBuildPipelines: %s\n", resStr(r));

    xess_vk_init_params_t ip{};
    ip.outputResolution = outRes;
    ip.qualitySetting = XESS_QUALITY_SETTING_QUALITY;
    ip.initFlags = initFlags;
    ip.creationNodeMask = 1;
    ip.visibleNodeMask = 1;
    r = xessVKInit(ctx, &ip);
    std::printf("xessVKInit: %s\n", resStr(r));

    if (r == XESS_RESULT_SUCCESS) {
        float jx = 0, jy = 0, vx = 0, vy = 0;
        xessGetJitterScale(ctx, &jx, &jy);
        xessGetVelocityScale(ctx, &vx, &vy);
        std::printf("jitterScale=(%f,%f) velocityScale=(%f,%f)\n", jx, jy, vx, vy);
        xessDestroyContext(ctx);
    }

    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
}
