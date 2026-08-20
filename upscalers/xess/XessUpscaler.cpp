// ============================================================================
// XeSS — Intel XeSS Super Resolution (Vulkan path) plugin.
//
// Conventions (verified against the XeSS 2.x developer guide and the official
// basic_sample_super_resolution_vk sample):
//   * Motion vectors: pixel units at input resolution, current frame ->
//     previous frame, no jitter (2.x guide).  Our renderer produces
//     previous -> current, so we flip the sign via xessSetVelocityScale(-1,-1).
//   * Jitter: pixel units in [-0.5, 0.5].  The official VK sample applies
//     clip.xy += 2*jitter/res (y-down NDC, same as us) and passes
//     jitterOffsetX = +jitter.x, jitterOffsetY = -jitter.y — we mirror that.
//   * Depth: non-inverted, low-res MV path (XeSS dilates/up-samples MVs
//     internally using depth), XESS_INIT_FLAG_RESPONSIVE_PIXEL_MASK (the
//     renderer's translucent coverage mask feeds responsivePixelMaskTexture;
//     a cleared render-res fallback is bound when the scene has none).
//   * Layouts are guaranteed by the renderer: inputs SHADER_READ_ONLY_OPTIMAL,
//     output GENERAL — exactly what xessVKExecute expects.
//
// Device requirements (probed on RTX 4070 SUPER via xessVKGetRequiredDevice*):
//   extension VK_EXT_mutable_descriptor_type; features
//   shaderStorageImageWriteWithoutFormat (core), shaderInt8 (Vulkan 1.2),
//   shaderIntegerDotProduct (1.3), mutableDescriptorType (EXT).
// ============================================================================
#include "upscalers/xess/XessUpscaler.h"

#include "upscalers/UpscalerFactory.h"

#include <xess/xess.h>
#include <xess/xess_vk.h>

#include <cstdio>
#include <cstring>
#include <vector>
#include <windows.h>

#include "upscalers/Cmdline.h"

namespace sr {

namespace {

const char* xessResultStr(xess_result_t r) {
    switch (r) {
        case XESS_RESULT_SUCCESS: return "SUCCESS";
        case XESS_RESULT_WARNING_NONEXISTING_FOLDER: return "WARNING_NONEXISTING_FOLDER";
        case XESS_RESULT_WARNING_OLD_DRIVER: return "WARNING_OLD_DRIVER";
        case XESS_RESULT_ERROR_UNSUPPORTED_DEVICE: return "ERROR_UNSUPPORTED_DEVICE";
        case XESS_RESULT_ERROR_UNSUPPORTED_DRIVER: return "ERROR_UNSUPPORTED_DRIVER";
        case XESS_RESULT_ERROR_UNINITIALIZED: return "ERROR_UNINITIALIZED";
        case XESS_RESULT_ERROR_INVALID_ARGUMENT: return "ERROR_INVALID_ARGUMENT";
        case XESS_RESULT_ERROR_DEVICE_OUT_OF_MEMORY: return "ERROR_DEVICE_OUT_OF_MEMORY";
        case XESS_RESULT_ERROR_DEVICE: return "ERROR_DEVICE";
        case XESS_RESULT_ERROR_NOT_IMPLEMENTED: return "ERROR_NOT_IMPLEMENTED";
        case XESS_RESULT_ERROR_INVALID_CONTEXT: return "ERROR_INVALID_CONTEXT";
        case XESS_RESULT_ERROR_OPERATION_IN_PROGRESS: return "ERROR_OPERATION_IN_PROGRESS";
        case XESS_RESULT_ERROR_UNSUPPORTED: return "ERROR_UNSUPPORTED";
        case XESS_RESULT_ERROR_CANT_LOAD_LIBRARY: return "ERROR_CANT_LOAD_LIBRARY";
        case XESS_RESULT_ERROR_WRONG_CALL_ORDER: return "ERROR_WRONG_CALL_ORDER";
        default: return "UNKNOWN";
    }
}

void xessLogCallback(const char* message, xess_logging_level_t level) {
    std::fprintf(stderr, "XeSS [%d]: %s", static_cast<int>(level), message);
}

// --- Vulkan device creation requirements (consumed by the renderer via
// SR_REGISTER_VULKAN_DEVICE_NEEDS before vkCreateDevice) ---

// Gate: only contribute device requirements when the run actually involves
// xess (mirrors the dlss/nss gating).  Without this, the XeSS feature chain
// (which triggers non-fatal VUID-00373/06532 warnings) would be injected
// into every process of a unified build.
bool xessRequestedOnCommandLine() {
    static const bool requested = sr::cmdline::pluginRequested("xess");
    // GUI mode flips the global override before device creation; the cached
    // command-line result alone would leave xess unavailable there.
    return allPluginsEnabled() || requested;
}

void xessAppendDeviceExtensions(std::vector<const char*>& deviceExts) {
    if (!xessRequestedOnCommandLine()) return;
    deviceExts.push_back(VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME);
}

// Static pNext feature chain appended after the renderer's own chain.
// The renderer already enables shaderStorageImageWriteWithoutFormat (core),
// shaderInt8 (its v12 node), and shaderIntegerDotProduct (its v13 node) —
// each sType may appear only once in the chain, so only the XeSS-specific
// mutable-descriptor feature remains here.
struct XessFeatureChain {
    VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT mutableDesc;

    XessFeatureChain() {
        mutableDesc = {};
        mutableDesc.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT;
        mutableDesc.mutableDescriptorType = VK_TRUE;
        mutableDesc.pNext = nullptr;
    }
};

const void* xessFeatureChain() {
    if (!xessRequestedOnCommandLine()) return nullptr;
    static XessFeatureChain chain;
    return &chain.mutableDesc;
}

VulkanDeviceNeeds makeXessDeviceNeeds() {
    VulkanDeviceNeeds needs;
    needs.appendExtensions = &xessAppendDeviceExtensions;
    needs.featureChain = &xessFeatureChain;
    return needs;
}

bool deviceExtensionSupported(VkPhysicalDevice physicalDevice, const char* name) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> props(count);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, props.data());
    for (const auto& p : props) {
        if (std::strcmp(p.extensionName, name) == 0) return true;
    }
    return false;
}

xess_vk_image_view_info makeTextureInfo(VkImageView view, VkImage image, VkFormat format,
                                        VkImageAspectFlags aspect, uint32_t width, uint32_t height) {
    xess_vk_image_view_info info = {};
    info.imageView = view;
    info.image = image;
    info.subresourceRange.aspectMask = aspect;
    info.subresourceRange.baseMipLevel = 0;
    info.subresourceRange.levelCount = 1;
    info.subresourceRange.baseArrayLayer = 0;
    info.subresourceRange.layerCount = 1;
    info.format = format;
    info.width = width;
    info.height = height;
    return info;
}

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeBits,
                        VkMemoryPropertyFlags required) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & required) == required)
            return i;
    }
    return 0xFFFFFFFFu;
}

// Fallback responsive pixel mask: a render-resolution R16_SFLOAT image cleared
// to 0 (no responsive pixels), left in SHADER_READ_ONLY_OPTIMAL.  Bound when
// the scene produces no translucency coverage mask, because the mask input is
// mandatory once XESS_INIT_FLAG_RESPONSIVE_PIXEL_MASK is set.
bool createFallbackMask(const VulkanEnv& env, uint32_t width, uint32_t height, VkImage& image,
                        VkDeviceMemory& memory, VkImageView& view) {
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.extent = {width, height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.format = VK_FORMAT_R16_SFLOAT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateImage(env.device, &ci, nullptr, &image) != VK_SUCCESS) return false;

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(env.device, image, &req);
    const uint32_t type = findMemoryType(env.physicalDevice, req.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == 0xFFFFFFFFu) {
        vkDestroyImage(env.device, image, nullptr);
        image = VK_NULL_HANDLE;
        return false;
    }
    VkMemoryAllocateInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    if (vkAllocateMemory(env.device, &ai, nullptr, &memory) != VK_SUCCESS) {
        vkDestroyImage(env.device, image, nullptr);
        image = VK_NULL_HANDLE;
        return false;
    }
    vkBindImageMemory(env.device, image, memory, 0);

    VkImageViewCreateInfo vi = {};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = VK_FORMAT_R16_SFLOAT;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    if (vkCreateImageView(env.device, &vi, nullptr, &view) != VK_SUCCESS) {
        vkFreeMemory(env.device, memory, nullptr);
        vkDestroyImage(env.device, image, nullptr);
        image = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        return false;
    }

    // One-shot clear to 0 + transition to SHADER_READ_ONLY_OPTIMAL.
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = env.commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(env.device, &allocInfo, &cmd) != VK_SUCCESS) return false;

    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    VkImageMemoryBarrier b = {};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.srcAccessMask = 0;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &b);
    const VkClearColorValue zero = {{0.f, 0.f, 0.f, 0.f}};
    vkCmdClearColorImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1,
                         &b.subresourceRange);
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &b);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    if (env.queueMutex) env.queueMutex->lock();
    vkQueueSubmit(env.graphicsQueue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(env.graphicsQueue);
    if (env.queueMutex) env.queueMutex->unlock();
    vkFreeCommandBuffers(env.device, env.commandPool, 1, &cmd);
    return true;
}

} // namespace

struct XessUpscaler::Impl {
    xess_context_handle_t context = nullptr;
    UpscalerDesc desc;
    xess_quality_settings_t quality = XESS_QUALITY_SETTING_BALANCED;
    uint64_t memoryBytes = 0;
    // Cleared render-res fallback for the responsive pixel mask (bound when
    // the scene has no translucency coverage mask).
    VkDevice device = VK_NULL_HANDLE;
    VkImage fallbackMask = VK_NULL_HANDLE;
    VkDeviceMemory fallbackMaskMemory = VK_NULL_HANDLE;
    VkImageView fallbackMaskView = VK_NULL_HANDLE;
};

XessUpscaler::~XessUpscaler() {
    shutdown();
}

const char* XessUpscaler::name() const {
    return "XeSS";
}

uint32_t XessUpscaler::capabilities() const {
    return Cap_Temporal | Cap_ML;
}

bool XessUpscaler::isAvailable(const VulkanEnv& env) {
    xess_version_t ver = {};
    const xess_result_t vr = xessGetVersion(&ver);
    if (vr != XESS_RESULT_SUCCESS) {
        std::fprintf(stderr, "XeSS: xessGetVersion failed (%s) — libxess.dll missing?\n",
                     xessResultStr(vr));
        return false;
    }

    // Cross-check the requirements we registered statically against what the
    // SDK actually wants and what the selected GPU supports.
    uint32_t extCount = 0;
    const char* const* exts = nullptr;
    const xess_result_t er =
        xessVKGetRequiredDeviceExtensions(env.instance, env.physicalDevice, &extCount, &exts);
    if (er != XESS_RESULT_SUCCESS) {
        std::fprintf(stderr, "XeSS: xessVKGetRequiredDeviceExtensions failed (%s)\n",
                     xessResultStr(er));
        return false;
    }
    for (uint32_t i = 0; i < extCount; ++i) {
        if (!deviceExtensionSupported(env.physicalDevice, exts[i])) {
            std::fprintf(stderr, "XeSS: required device extension %s unsupported\n", exts[i]);
            return false;
        }
    }

    VkPhysicalDeviceVulkan13Features v13 = {};
    v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    VkPhysicalDeviceVulkan12Features v12 = {};
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.pNext = &v13;
    VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT mutableDesc = {};
    mutableDesc.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT;
    mutableDesc.pNext = &v12;
    VkPhysicalDeviceFeatures2 supported = {};
    supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    supported.pNext = &mutableDesc;
    vkGetPhysicalDeviceFeatures2(env.physicalDevice, &supported);

    struct Requirement {
        VkBool32 value;
        const char* name;
    };
    const Requirement reqs[] = {
        {supported.features.shaderStorageImageWriteWithoutFormat, "shaderStorageImageWriteWithoutFormat"},
        {v12.shaderInt8, "shaderInt8"},
        {v13.shaderIntegerDotProduct, "shaderIntegerDotProduct"},
        {mutableDesc.mutableDescriptorType, "mutableDescriptorType"},
    };
    for (const Requirement& req : reqs) {
        if (!req.value) {
            std::fprintf(stderr, "XeSS: required device feature %s unsupported\n", req.name);
            return false;
        }
    }
    return true;
}

bool XessUpscaler::init(const VulkanEnv& env, const UpscalerDesc& desc) {
    shutdown();
    impl_ = new Impl();
    impl_->desc = desc;
    impl_->device = env.device;

    xess_result_t r = xessVKCreateContext(env.instance, env.physicalDevice, env.device, &impl_->context);
    if (r != XESS_RESULT_SUCCESS) {
        std::fprintf(stderr, "XeSS: xessVKCreateContext failed (%s)\n", xessResultStr(r));
        shutdown();
        return false;
    }
    xessSetLoggingCallback(impl_->context, XESS_LOGGING_LEVEL_WARNING, &xessLogCallback);

    if (xessIsOptimalDriver(impl_->context) == XESS_RESULT_WARNING_OLD_DRIVER) {
        std::fprintf(stderr, "XeSS: warning — driver is not optimal for XeSS\n");
    }

    const xess_2d_t outputRes = {desc.displayWidth, desc.displayHeight};

    xess_properties_t props = {};
    r = xessGetProperties(impl_->context, &outputRes, &props);
    if (r != XESS_RESULT_SUCCESS) {
        std::fprintf(stderr, "XeSS: xessGetProperties failed (%s)\n", xessResultStr(r));
        shutdown();
        return false;
    }
    impl_->memoryBytes = props.tempBufferHeapSize + props.tempTextureHeapSize;

    // Pick the highest-quality preset whose input resolution range covers our
    // render resolution (e.g. 960x540 -> 1920x1080 fits BALANCED; QUALITY
    // requires >= 1130x636 for 1080p output).
    const xess_quality_settings_t candidates[] = {
        XESS_QUALITY_SETTING_ULTRA_QUALITY_PLUS,
        XESS_QUALITY_SETTING_ULTRA_QUALITY,
        XESS_QUALITY_SETTING_QUALITY,
        XESS_QUALITY_SETTING_BALANCED,
        XESS_QUALITY_SETTING_PERFORMANCE,
        XESS_QUALITY_SETTING_ULTRA_PERFORMANCE,
    };
    const char* const qualityNames[] = {
        "ULTRA_QUALITY_PLUS", "ULTRA_QUALITY", "QUALITY", "BALANCED", "PERFORMANCE",
        "ULTRA_PERFORMANCE",
    };
    bool picked = false;
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        xess_2d_t optimal = {}, minRes = {}, maxRes = {};
        r = xessGetOptimalInputResolution(impl_->context, &outputRes, candidates[i], &optimal,
                                          &minRes, &maxRes);
        if (r != XESS_RESULT_SUCCESS) continue;
        if (desc.renderWidth >= minRes.x && desc.renderWidth <= maxRes.x &&
            desc.renderHeight >= minRes.y && desc.renderHeight <= maxRes.y) {
            impl_->quality = candidates[i];
            std::printf("XeSS: %ux%u -> %ux%u, preset %s (optimal input %ux%u)\n",
                        desc.renderWidth, desc.renderHeight, desc.displayWidth, desc.displayHeight,
                        qualityNames[i], optimal.x, optimal.y);
            picked = true;
            break;
        }
    }
    if (!picked) {
        std::fprintf(stderr, "XeSS: no quality preset accepts input %ux%u for output %ux%u\n",
                     desc.renderWidth, desc.renderHeight, desc.displayWidth, desc.displayHeight);
        shutdown();
        return false;
    }

    // Low-res MV + depth, non-inverted depth, responsive pixel mask input
    // (translucent coverage; fallback black mask bound when absent).
    const uint32_t initFlags = XESS_INIT_FLAG_RESPONSIVE_PIXEL_MASK;
    r = xessVKBuildPipelines(impl_->context, VK_NULL_HANDLE, true, initFlags);
    if (r != XESS_RESULT_SUCCESS) {
        std::fprintf(stderr, "XeSS: xessVKBuildPipelines failed (%s)\n", xessResultStr(r));
        shutdown();
        return false;
    }

    xess_vk_init_params_t initParams = {};
    initParams.outputResolution = outputRes;
    initParams.qualitySetting = impl_->quality;
    initParams.initFlags = initFlags;
    initParams.creationNodeMask = 1;
    initParams.visibleNodeMask = 1;
    initParams.tempBufferHeap = VK_NULL_HANDLE;  // internal allocation
    initParams.bufferHeapOffset = 0;
    initParams.tempTextureHeap = VK_NULL_HANDLE; // internal allocation
    initParams.textureHeapOffset = 0;
    initParams.pipelineCache = VK_NULL_HANDLE;
    r = xessVKInit(impl_->context, &initParams);
    if (r != XESS_RESULT_SUCCESS) {
        std::fprintf(stderr, "XeSS: xessVKInit failed (%s)\n", xessResultStr(r));
        shutdown();
        return false;
    }

    if (!createFallbackMask(env, desc.renderWidth, desc.renderHeight, impl_->fallbackMask,
                            impl_->fallbackMaskMemory, impl_->fallbackMaskView)) {
        std::fprintf(stderr, "XeSS: failed to create fallback responsive mask\n");
        shutdown();
        return false;
    }

    // Our MV buffer stores previous -> current motion; XeSS 2.x expects
    // current -> previous, so flip both axes (velocity is already in
    // input-resolution pixels, y down).
    r = xessSetVelocityScale(impl_->context, -1.f, -1.f);
    if (r != XESS_RESULT_SUCCESS) {
        std::fprintf(stderr, "XeSS: xessSetVelocityScale failed (%s)\n", xessResultStr(r));
        shutdown();
        return false;
    }

    // Our coverage mask accumulates alpha and can exceed 1; make the clip at
    // 1.0 explicit (header documents no default for this).
    r = xessSetMaxResponsiveMaskValue(impl_->context, 1.0f);
    if (r != XESS_RESULT_SUCCESS) {
        std::fprintf(stderr, "XeSS: xessSetMaxResponsiveMaskValue failed (%s)\n", xessResultStr(r));
        shutdown();
        return false;
    }
    return true;
}

void XessUpscaler::dispatch(VkCommandBuffer cmd, const UpscalerResources& res, const CameraParams& cam,
                            const FrameParams& frame) {
    if (!impl_ || !impl_->context) return;

    xess_vk_execute_params_t params = {};
    params.colorTexture =
        makeTextureInfo(res.colorView, res.color, VK_FORMAT_R16G16B16A16_SFLOAT,
                        VK_IMAGE_ASPECT_COLOR_BIT, impl_->desc.renderWidth, impl_->desc.renderHeight);
    params.velocityTexture =
        makeTextureInfo(res.motionView, res.motion, VK_FORMAT_R16G16_SFLOAT,
                        VK_IMAGE_ASPECT_COLOR_BIT, impl_->desc.renderWidth, impl_->desc.renderHeight);
    params.depthTexture =
        makeTextureInfo(res.depthView, res.depth, VK_FORMAT_D32_SFLOAT,
                        VK_IMAGE_ASPECT_DEPTH_BIT, impl_->desc.renderWidth, impl_->desc.renderHeight);
    // Translucent coverage mask (R16_SFLOAT, render res, 0 = no translucency)
    // -> XeSS responsive pixel mask (clipped at 1.0, set at init).
    const bool hasMask = res.reactive != VK_NULL_HANDLE && res.reactiveView != VK_NULL_HANDLE;
    params.responsivePixelMaskTexture =
        makeTextureInfo(hasMask ? res.reactiveView : impl_->fallbackMaskView,
                        hasMask ? res.reactive : impl_->fallbackMask,
                        VK_FORMAT_R16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT,
                        impl_->desc.renderWidth, impl_->desc.renderHeight);
    params.outputTexture =
        makeTextureInfo(res.outputView, res.output, VK_FORMAT_R16G16B16A16_SFLOAT,
                        VK_IMAGE_ASPECT_COLOR_BIT, impl_->desc.displayWidth, impl_->desc.displayHeight);
    params.jitterOffsetX = cam.jitterX;
    params.jitterOffsetY = -cam.jitterY; // official VK sample convention (y-down clip space)
    params.exposureScale = frame.preExposure;
    params.resetHistory = frame.resetHistory ? 1u : 0u;
    params.inputWidth = impl_->desc.renderWidth;
    params.inputHeight = impl_->desc.renderHeight;

    const xess_result_t r = xessVKExecute(impl_->context, cmd, &params);
    if (r != XESS_RESULT_SUCCESS) {
        std::fprintf(stderr, "XeSS: xessVKExecute failed (%s)\n", xessResultStr(r));
    }
}

void XessUpscaler::shutdown() {
    if (!impl_) return;
    if (impl_->context) {
        xessDestroyContext(impl_->context);
        impl_->context = nullptr;
    }
    if (impl_->device) {
        if (impl_->fallbackMaskView) { vkDestroyImageView(impl_->device, impl_->fallbackMaskView, nullptr); impl_->fallbackMaskView = VK_NULL_HANDLE; }
        if (impl_->fallbackMask) { vkDestroyImage(impl_->device, impl_->fallbackMask, nullptr); impl_->fallbackMask = VK_NULL_HANDLE; }
        if (impl_->fallbackMaskMemory) { vkFreeMemory(impl_->device, impl_->fallbackMaskMemory, nullptr); impl_->fallbackMaskMemory = VK_NULL_HANDLE; }
    }
    delete impl_;
    impl_ = nullptr;
}

uint64_t XessUpscaler::gpuMemoryBytes() const {
    return impl_ ? impl_->memoryBytes : 0;
}

std::unique_ptr<IUpscaler> createXessUpscaler() { return std::make_unique<XessUpscaler>(); }

} // namespace sr

SR_REGISTER_UPSCALER("xess", &sr::createXessUpscaler);
SR_REGISTER_VULKAN_DEVICE_NEEDS(sr::makeXessDeviceNeeds());
