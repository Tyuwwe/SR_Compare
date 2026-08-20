// ============================================================================
// AMD FSR1 / FSR2 / FSR3.1 upscalers via the FidelityFX SDK host components
// (ffx_fsr1 / ffx_fsr2 / ffx_fsr3upscaler) and the SDK's Vulkan backend
// (ffx_vk).  Shader permutation headers are pre-generated into
// ffx_permutations/ by gen_permutations.sh.
//
// Input conventions (see upscalers/InputAdapter.h):
//   * motion vectors are render-resolution pixels, y down, no jitter, but ours
//     point cur->prev while FSR2/FSR3 reproject with prev = uv + MV, so
//     motionVectorScale negates them ({-1,-1}).
//   * depth is D32_SFLOAT, non-inverted, infinite far plane.
//   * no exposure input: preExposure = 1 and the effects fall back to their
//     internal default exposure resource.
//   * when UpscalerResources::reactive is non-null (translucency coverage mask,
//     R16_SFLOAT), the same mask feeds both the reactive and the
//     transparencyAndComposition inputs of FSR2/FSR3.
//
// FP16 permutations are disabled by wrapping fpGetDeviceCapabilities: the
// renderer's device does not enable shaderFloat16/shaderInt16 (the latter is
// a core Vulkan 1.0 feature that cannot be requested through the plugin
// device-needs feature chain).
// ============================================================================
#include "upscalers/fsr/FsrUpscaler.h"

#include "upscalers/UpscalerFactory.h"
#include "upscalers/VkHelpers.h"

#include <FidelityFX/host/backends/vk/ffx_vk.h>
#include <FidelityFX/host/ffx_fsr1.h>
#include <FidelityFX/host/ffx_fsr2.h>
#include <FidelityFX/host/ffx_fsr3upscaler.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// The VK backend's ffxGetInterfaceVK references the frame-interpolation
// swapchain entry point.  We do not use frame interpolation, so stub it out
// instead of pulling in the FrameInterpolationSwapchain sources.
extern "C" FfxErrorCode ffxSetFrameGenerationConfigToSwapchainVK(FfxFrameGenerationConfig const*) {
    return FFX_ERROR_INVALID_ARGUMENT;
}

namespace sr {

namespace {

constexpr size_t kMaxFfxContexts = 8;    // effect contexts sharing one backend
constexpr float  kSharpness = 0.5f;      // RCAS sharpness, [0..1]

// ---------------------------------------------------------------------------
// fpGetDeviceCapabilities wrapper: report fp16 as unsupported so the FSR
// components select their FP32 shader permutations (see file header).
// ---------------------------------------------------------------------------
FfxGetDeviceCapabilitiesFunc g_origGetDeviceCaps = nullptr;

// vkGetDeviceProcAddr trampoline: on this driver the KHR names of commands
// promoted to core Vulkan are not queryable even when the extension is
// enabled (core name works).  The FFX VK backend only queries KHR names, so
// fall back to the core alias on NULL.
PFN_vkVoidFunction VKAPI_CALL srVkGetDeviceProcAddr(VkDevice device, const char* pName) {
    PFN_vkVoidFunction fp = vkGetDeviceProcAddr(device, pName);
    if (!fp && std::strcmp(pName, "vkGetBufferMemoryRequirements2KHR") == 0)
        fp = vkGetDeviceProcAddr(device, "vkGetBufferMemoryRequirements2");
    return fp;
}

FfxErrorCode getDeviceCapabilitiesFp32(FfxInterface* iface, FfxDeviceCapabilities* caps) {
    FfxErrorCode err = g_origGetDeviceCaps(iface, caps);
    caps->fp16Supported = false;
    return err;
}

// ---------------------------------------------------------------------------
// Shared FFX Vulkan backend: one scratch buffer + interface per plugin
// instance.  Effect contexts must be destroyed before destroy().
// ---------------------------------------------------------------------------
struct FfxVkBackend {
    VkDeviceContext deviceContext = {};  // must outlive ffxGetDeviceVK users
    FfxInterface    iface = {};
    void*           scratch = nullptr;

    bool init(const VulkanEnv& env) {
        deviceContext.vkDevice = env.device;
        deviceContext.vkPhysicalDevice = env.physicalDevice;
        deviceContext.vkDeviceProcAddr = &srVkGetDeviceProcAddr;
        FfxDevice ffxDevice = ffxGetDeviceVK(&deviceContext);

        const size_t scratchSize = ffxGetScratchMemorySizeVK(env.physicalDevice, kMaxFfxContexts);
        scratch = std::malloc(scratchSize);
        if (!scratch) return false;

        if (ffxGetInterfaceVK(&iface, ffxDevice, scratch, scratchSize, kMaxFfxContexts) != FFX_OK) {
            destroy();
            return false;
        }
        // Force the FP32 permutation path: the renderer enables neither
        // shaderFloat16 nor shaderInt16 on the device.
        if (!g_origGetDeviceCaps) g_origGetDeviceCaps = iface.fpGetDeviceCapabilities;
        iface.fpGetDeviceCapabilities = &getDeviceCapabilitiesFp32;
        return true;
    }

    void destroy() {
        std::free(scratch);
        scratch = nullptr;
    }
};

FfxResourceDescription imageDesc(FfxSurfaceFormat format, uint32_t width, uint32_t height,
                                 FfxResourceUsage usage) {
    FfxResourceDescription d = {};
    d.type = FFX_RESOURCE_TYPE_TEXTURE2D;
    d.format = format;
    d.width = width;
    d.height = height;
    d.depth = 1;
    d.mipCount = 1;
    d.flags = FFX_RESOURCE_FLAGS_NONE;
    d.usage = usage;
    return d;
}

// Small Vulkan image owned by the plugin (used for FSR3 shared resources).
struct OwnedImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint64_t bytes = 0;
};

bool createStorageImage(const VulkanEnv& env, uint32_t width, uint32_t height, VkFormat format,
                        OwnedImage& out) {
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.extent = {width, height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.format = format;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateImage(env.device, &ci, nullptr, &out.image) != VK_SUCCESS) return false;

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(env.device, out.image, &req);
    const uint32_t type = findMemoryType(env.physicalDevice, req.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == 0xFFFFFFFFu) {
        vkDestroyImage(env.device, out.image, nullptr);
        out.image = VK_NULL_HANDLE;
        return false;
    }
    VkMemoryAllocateInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    if (vkAllocateMemory(env.device, &ai, nullptr, &out.memory) != VK_SUCCESS) {
        vkDestroyImage(env.device, out.image, nullptr);
        out.image = VK_NULL_HANDLE;
        return false;
    }
    vkBindImageMemory(env.device, out.image, out.memory, 0);
    out.bytes = req.size;
    return true;
}

void destroyStorageImage(const VulkanEnv& env, OwnedImage& img) {
    if (img.image) { vkDestroyImage(env.device, img.image, nullptr); img.image = VK_NULL_HANDLE; }
    if (img.memory) { vkFreeMemory(env.device, img.memory, nullptr); img.memory = VK_NULL_HANDLE; }
}

// One-shot UNDEFINED -> GENERAL transitions via the renderer's command pool.
void transitionToGeneral(const VulkanEnv& env, const OwnedImage* images, uint32_t count) {
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = env.commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(env.device, &allocInfo, &cmd) != VK_SUCCESS) return;

    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS) {
        std::fprintf(stderr, "FSR3.1: vkBeginCommandBuffer failed\n");
        vkFreeCommandBuffers(env.device, env.commandPool, 1, &cmd);
        return;
    }

    VkImageMemoryBarrier barriers[8];
    for (uint32_t i = 0; i < count && i < 8; ++i) {
        VkImageMemoryBarrier& b = barriers[i];
        b = {};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = images[i].image;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        b.srcAccessMask = 0;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    }
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                         nullptr, 0, nullptr, count, barriers);
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        std::fprintf(stderr, "FSR3.1: vkEndCommandBuffer failed\n");
        vkFreeCommandBuffers(env.device, env.commandPool, 1, &cmd);
        return;
    }

    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    if (env.queueMutex) {
        std::lock_guard<std::mutex> lk(*env.queueMutex);
        if (vkQueueSubmit(env.graphicsQueue, 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS)
            std::fprintf(stderr, "FSR3.1: vkQueueSubmit failed\n");
        vkQueueWaitIdle(env.graphicsQueue);
    } else {
        if (vkQueueSubmit(env.graphicsQueue, 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS)
            std::fprintf(stderr, "FSR3.1: vkQueueSubmit failed\n");
        vkQueueWaitIdle(env.graphicsQueue);
    }
    vkFreeCommandBuffers(env.device, env.commandPool, 1, &cmd);
}

// Shared per-frame FfxResource wrappers for the renderer-provided inputs.
struct FrameFfxResources {
    FfxResource color;
    FfxResource depth;
    FfxResource motion;
    FfxResource output;
    FfxResource reactive;  // null resource when the scene has no translucency
};

FrameFfxResources wrapFrameResources(const UpscalerResources& res, const UpscalerDesc& desc) {
    FrameFfxResources f = {};
    if (res.reactive != VK_NULL_HANDLE) {
        // Translucency coverage mask (R16_SFLOAT, render resolution, 0..1+);
        // already in SHADER_READ_ONLY at dispatch time.
        f.reactive = ffxGetResourceVK(res.reactive,
            imageDesc(FFX_SURFACE_FORMAT_R16_FLOAT, desc.renderWidth, desc.renderHeight,
                      FFX_RESOURCE_USAGE_READ_ONLY),
            L"SR_InputReactive", FFX_RESOURCE_STATE_COMPUTE_READ);
    }
    f.color = ffxGetResourceVK(res.color,
        imageDesc(FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT, desc.renderWidth, desc.renderHeight,
                  FFX_RESOURCE_USAGE_READ_ONLY),
        L"SR_InputColor", FFX_RESOURCE_STATE_COMPUTE_READ);
    f.depth = ffxGetResourceVK(res.depth,
        imageDesc(FFX_SURFACE_FORMAT_R32_FLOAT, desc.renderWidth, desc.renderHeight,
                  (FfxResourceUsage)(FFX_RESOURCE_USAGE_READ_ONLY | FFX_RESOURCE_USAGE_DEPTHTARGET)),
        L"SR_InputDepth", FFX_RESOURCE_STATE_COMPUTE_READ);
    f.motion = ffxGetResourceVK(res.motion,
        imageDesc(FFX_SURFACE_FORMAT_R16G16_FLOAT, desc.renderWidth, desc.renderHeight,
                  FFX_RESOURCE_USAGE_READ_ONLY),
        L"SR_InputMotion", FFX_RESOURCE_STATE_COMPUTE_READ);
    f.output = ffxGetResourceVK(res.output,
        imageDesc(FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT, desc.displayWidth, desc.displayHeight,
                  FFX_RESOURCE_USAGE_UAV),
        L"SR_Output", FFX_RESOURCE_STATE_UNORDERED_ACCESS);
    return f;
}

FfxFloatCoords2D jitterOf(const CameraParams& cam) { return {cam.jitterX, cam.jitterY}; }

float cameraFarOf(const UpscalerDesc& desc, const CameraParams& cam) {
    // FSR expects FLT_MAX for a non-inverted infinite far plane.
    return desc.infiniteFarPlane ? FLT_MAX : cam.cameraFar;
}

} // namespace

// ============================================================================
// FSR1 (spatial)
// ============================================================================
struct Fsr1Upscaler::Impl {
    VulkanEnv env;
    UpscalerDesc desc;
    FfxVkBackend backend;
    FfxFsr1Context context = {};
    bool contextCreated = false;
    uint64_t memoryBytes = 0;
};

Fsr1Upscaler::~Fsr1Upscaler() { shutdown(); }
const char* Fsr1Upscaler::name() const { return "FSR1"; }
uint32_t Fsr1Upscaler::capabilities() const { return Cap_Spatial; }
bool Fsr1Upscaler::isAvailable(const VulkanEnv& env) { return env.device != VK_NULL_HANDLE; }

bool Fsr1Upscaler::init(const VulkanEnv& env, const UpscalerDesc& desc) {
    shutdown();
    impl_ = new Impl();
    impl_->env = env;
    impl_->desc = desc;

    if (!impl_->backend.init(env)) { shutdown(); return false; }

    FfxFsr1ContextDescription ctxDesc = {};
    ctxDesc.flags = FFX_FSR1_ENABLE_RCAS;
    if (desc.hdr) ctxDesc.flags |= FFX_FSR1_ENABLE_HIGH_DYNAMIC_RANGE;
    ctxDesc.outputFormat = FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT;
    ctxDesc.maxRenderSize = {desc.renderWidth, desc.renderHeight};
    ctxDesc.displaySize = {desc.displayWidth, desc.displayHeight};
    ctxDesc.backendInterface = impl_->backend.iface;
    if (ffxFsr1ContextCreate(&impl_->context, &ctxDesc) != FFX_OK) {
        std::fprintf(stderr, "FSR1: context creation failed\n");
        shutdown();
        return false;
    }
    impl_->contextCreated = true;

    FfxEffectMemoryUsage usage = {};
    if (ffxFsr1ContextGetGpuMemoryUsage(&impl_->context, &usage) == FFX_OK)
        impl_->memoryBytes = usage.totalUsageInBytes;
    return true;
}

void Fsr1Upscaler::dispatch(VkCommandBuffer cmd, const UpscalerResources& res, const CameraParams& cam,
                            const FrameParams& frame) {
    (void)cam;
    (void)frame;
    if (!impl_ || !impl_->contextCreated) return;

    FrameFfxResources f = wrapFrameResources(res, impl_->desc);

    FfxFsr1DispatchDescription dd = {};
    dd.commandList = ffxGetCommandListVK(cmd);
    dd.color = f.color;
    dd.output = f.output;
    dd.renderSize = {impl_->desc.renderWidth, impl_->desc.renderHeight};
    dd.enableSharpening = true;
    dd.sharpness = kSharpness;
    ffxFsr1ContextDispatch(&impl_->context, &dd);
}

void Fsr1Upscaler::shutdown() {
    if (!impl_) return;
    if (impl_->contextCreated) {
        ffxFsr1ContextDestroy(&impl_->context);
        impl_->contextCreated = false;
    }
    impl_->backend.destroy();
    delete impl_;
    impl_ = nullptr;
}

uint64_t Fsr1Upscaler::gpuMemoryBytes() const { return impl_ ? impl_->memoryBytes : 0; }

// ============================================================================
// FSR2 (temporal)
// ============================================================================
struct Fsr2Upscaler::Impl {
    VulkanEnv env;
    UpscalerDesc desc;
    FfxVkBackend backend;
    FfxFsr2Context context = {};
    bool contextCreated = false;
    uint64_t memoryBytes = 0;
};

Fsr2Upscaler::~Fsr2Upscaler() { shutdown(); }
const char* Fsr2Upscaler::name() const { return "FSR2"; }
uint32_t Fsr2Upscaler::capabilities() const { return Cap_Temporal; }
bool Fsr2Upscaler::isAvailable(const VulkanEnv& env) { return env.device != VK_NULL_HANDLE; }

bool Fsr2Upscaler::init(const VulkanEnv& env, const UpscalerDesc& desc) {
    shutdown();
    impl_ = new Impl();
    impl_->env = env;
    impl_->desc = desc;

    if (!impl_->backend.init(env)) { shutdown(); return false; }

    FfxFsr2ContextDescription ctxDesc = {};
    ctxDesc.flags = FFX_FSR2_ENABLE_HIGH_DYNAMIC_RANGE;
    if (desc.infiniteFarPlane) ctxDesc.flags |= FFX_FSR2_ENABLE_DEPTH_INFINITE;
    if (desc.invertedDepth) ctxDesc.flags |= FFX_FSR2_ENABLE_DEPTH_INVERTED;
    ctxDesc.maxRenderSize = {desc.renderWidth, desc.renderHeight};
    ctxDesc.displaySize = {desc.displayWidth, desc.displayHeight};
    ctxDesc.fpMessage = nullptr;
    ctxDesc.backendInterface = impl_->backend.iface;
    if (ffxFsr2ContextCreate(&impl_->context, &ctxDesc) != FFX_OK) {
        std::fprintf(stderr, "FSR2: context creation failed\n");
        shutdown();
        return false;
    }
    impl_->contextCreated = true;

    FfxEffectMemoryUsage usage = {};
    if (ffxFsr2ContextGetGpuMemoryUsage(&impl_->context, &usage) == FFX_OK)
        impl_->memoryBytes = usage.totalUsageInBytes;
    return true;
}

void Fsr2Upscaler::dispatch(VkCommandBuffer cmd, const UpscalerResources& res, const CameraParams& cam,
                            const FrameParams& frame) {
    if (!impl_ || !impl_->contextCreated) return;

    FrameFfxResources f = wrapFrameResources(res, impl_->desc);

    FfxFsr2DispatchDescription dd = {};
    dd.commandList = ffxGetCommandListVK(cmd);
    dd.color = f.color;
    dd.depth = f.depth;
    dd.motionVectors = f.motion;
    dd.exposure = {};              // null -> internal default exposure (1.0)
    dd.reactive = f.reactive;      // null when scene has no translucency -> internal default (zero) mask
    dd.transparencyAndComposition = f.reactive;  // same coverage mask feeds TC
    dd.output = f.output;
    dd.jitterOffset = jitterOf(cam);
    dd.motionVectorScale = {-1.f, -1.f};  // our MVs are cur->prev; FSR wants prev = uv + MV (cur->prev negated)
    dd.renderSize = {impl_->desc.renderWidth, impl_->desc.renderHeight};
    dd.enableSharpening = true;
    dd.sharpness = kSharpness;
    dd.frameTimeDelta = frame.deltaTime * 1000.f;  // milliseconds
    dd.preExposure = frame.preExposure;
    dd.reset = frame.resetHistory;
    dd.cameraNear = cam.cameraNear;
    dd.cameraFar = cameraFarOf(impl_->desc, cam);
    dd.cameraFovAngleVertical = cam.fovY;
    dd.viewSpaceToMetersFactor = 1.f;
    dd.enableAutoReactive = false;
    dd.colorOpaqueOnly = {};
    ffxFsr2ContextDispatch(&impl_->context, &dd);
}

void Fsr2Upscaler::shutdown() {
    if (!impl_) return;
    if (impl_->contextCreated) {
        ffxFsr2ContextDestroy(&impl_->context);
        impl_->contextCreated = false;
    }
    impl_->backend.destroy();
    delete impl_;
    impl_ = nullptr;
}

uint64_t Fsr2Upscaler::gpuMemoryBytes() const { return impl_ ? impl_->memoryBytes : 0; }

// ============================================================================
// FSR3.1 upscaler (temporal)
// ============================================================================
struct Fsr3Upscaler::Impl {
    VulkanEnv env;
    UpscalerDesc desc;
    FfxVkBackend backend;
    FfxFsr3UpscalerContext context = {};
    bool contextCreated = false;
    // Shared resources the app must provide to the FSR3 upscaler (see
    // ffxFsr3UpscalerGetSharedResourceDescriptions).
    OwnedImage dilatedDepth;                    // R32_SFLOAT, render size
    OwnedImage dilatedMotionVectors;            // R16G16_SFLOAT, render size
    OwnedImage reconstructedPrevNearestDepth;   // R32_UINT, render size
    uint64_t memoryBytes = 0;
};

Fsr3Upscaler::~Fsr3Upscaler() { shutdown(); }
const char* Fsr3Upscaler::name() const { return "FSR3.1"; }
uint32_t Fsr3Upscaler::capabilities() const { return Cap_Temporal; }
bool Fsr3Upscaler::isAvailable(const VulkanEnv& env) { return env.device != VK_NULL_HANDLE; }

bool Fsr3Upscaler::init(const VulkanEnv& env, const UpscalerDesc& desc) {
    shutdown();
    impl_ = new Impl();
    impl_->env = env;
    impl_->desc = desc;

    if (!impl_->backend.init(env)) { shutdown(); return false; }

    FfxFsr3UpscalerContextDescription ctxDesc = {};
    ctxDesc.flags = FFX_FSR3UPSCALER_ENABLE_HIGH_DYNAMIC_RANGE;
    if (desc.infiniteFarPlane) ctxDesc.flags |= FFX_FSR3UPSCALER_ENABLE_DEPTH_INFINITE;
    if (desc.invertedDepth) ctxDesc.flags |= FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED;
    ctxDesc.maxRenderSize = {desc.renderWidth, desc.renderHeight};
    ctxDesc.maxUpscaleSize = {desc.displayWidth, desc.displayHeight};
    ctxDesc.fpMessage = nullptr;
    ctxDesc.backendInterface = impl_->backend.iface;
    if (ffxFsr3UpscalerContextCreate(&impl_->context, &ctxDesc) != FFX_OK) {
        std::fprintf(stderr, "FSR3.1: context creation failed\n");
        shutdown();
        return false;
    }
    impl_->contextCreated = true;

    // Allocate the shared resources the FSR3 upscaler writes to and exposes
    // for follow-up effects.  They stay in VK_IMAGE_LAYOUT_GENERAL forever.
    const uint32_t rw = desc.renderWidth, rh = desc.renderHeight;
    if (!createStorageImage(env, rw, rh, VK_FORMAT_R32_SFLOAT, impl_->dilatedDepth) ||
        !createStorageImage(env, rw, rh, VK_FORMAT_R16G16_SFLOAT, impl_->dilatedMotionVectors) ||
        !createStorageImage(env, rw, rh, VK_FORMAT_R32_UINT, impl_->reconstructedPrevNearestDepth)) {
        std::fprintf(stderr, "FSR3.1: shared resource allocation failed\n");
        shutdown();
        return false;
    }
    const OwnedImage sharedImages[3] = {impl_->dilatedDepth, impl_->dilatedMotionVectors,
                                        impl_->reconstructedPrevNearestDepth};
    transitionToGeneral(env, sharedImages, 3);

    FfxEffectMemoryUsage usage = {};
    if (ffxFsr3UpscalerContextGetGpuMemoryUsage(&impl_->context, &usage) == FFX_OK)
        impl_->memoryBytes = usage.totalUsageInBytes;
    impl_->memoryBytes += impl_->dilatedDepth.bytes + impl_->dilatedMotionVectors.bytes +
                          impl_->reconstructedPrevNearestDepth.bytes;
    return true;
}

void Fsr3Upscaler::dispatch(VkCommandBuffer cmd, const UpscalerResources& res, const CameraParams& cam,
                            const FrameParams& frame) {
    if (!impl_ || !impl_->contextCreated) return;

    FrameFfxResources f = wrapFrameResources(res, impl_->desc);
    const uint32_t rw = impl_->desc.renderWidth, rh = impl_->desc.renderHeight;

    FfxResource dilatedDepth = ffxGetResourceVK(
        impl_->dilatedDepth.image,
        imageDesc(FFX_SURFACE_FORMAT_R32_FLOAT, rw, rh,
                  (FfxResourceUsage)(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV)),
        L"SR_FSR3_DilatedDepth", FFX_RESOURCE_STATE_UNORDERED_ACCESS);
    FfxResource dilatedMotion = ffxGetResourceVK(
        impl_->dilatedMotionVectors.image,
        imageDesc(FFX_SURFACE_FORMAT_R16G16_FLOAT, rw, rh,
                  (FfxResourceUsage)(FFX_RESOURCE_USAGE_RENDERTARGET | FFX_RESOURCE_USAGE_UAV)),
        L"SR_FSR3_DilatedMotionVectors", FFX_RESOURCE_STATE_UNORDERED_ACCESS);
    FfxResource reconDepth = ffxGetResourceVK(
        impl_->reconstructedPrevNearestDepth.image,
        imageDesc(FFX_SURFACE_FORMAT_R32_UINT, rw, rh, FFX_RESOURCE_USAGE_UAV),
        L"SR_FSR3_ReconstructedPrevNearestDepth", FFX_RESOURCE_STATE_UNORDERED_ACCESS);

    FfxFsr3UpscalerDispatchDescription dd = {};
    dd.commandList = ffxGetCommandListVK(cmd);
    dd.color = f.color;
    dd.depth = f.depth;
    dd.motionVectors = f.motion;
    dd.exposure = {};
    dd.reactive = f.reactive;      // null when scene has no translucency -> internal default (zero) mask
    dd.transparencyAndComposition = f.reactive;  // same coverage mask feeds TC
    dd.dilatedDepth = dilatedDepth;
    dd.dilatedMotionVectors = dilatedMotion;
    dd.reconstructedPrevNearestDepth = reconDepth;
    dd.output = f.output;
    dd.jitterOffset = jitterOf(cam);
    dd.motionVectorScale = {-1.f, -1.f};  // our MVs are cur->prev; FSR wants prev = uv + MV (cur->prev negated)
    dd.renderSize = {rw, rh};
    dd.upscaleSize = {impl_->desc.displayWidth, impl_->desc.displayHeight};
    dd.enableSharpening = true;
    dd.sharpness = kSharpness;
    dd.frameTimeDelta = frame.deltaTime * 1000.f;  // milliseconds
    dd.preExposure = frame.preExposure;
    dd.reset = frame.resetHistory;
    dd.cameraNear = cam.cameraNear;
    dd.cameraFar = cameraFarOf(impl_->desc, cam);
    dd.cameraFovAngleVertical = cam.fovY;
    dd.viewSpaceToMetersFactor = 1.f;
    dd.flags = 0;
    ffxFsr3UpscalerContextDispatch(&impl_->context, &dd);
}

void Fsr3Upscaler::shutdown() {
    if (!impl_) return;
    if (impl_->contextCreated) {
        ffxFsr3UpscalerContextDestroy(&impl_->context);
        impl_->contextCreated = false;
    }
    destroyStorageImage(impl_->env, impl_->dilatedDepth);
    destroyStorageImage(impl_->env, impl_->dilatedMotionVectors);
    destroyStorageImage(impl_->env, impl_->reconstructedPrevNearestDepth);
    impl_->backend.destroy();
    delete impl_;
    impl_ = nullptr;
}

uint64_t Fsr3Upscaler::gpuMemoryBytes() const { return impl_ ? impl_->memoryBytes : 0; }

// ============================================================================
// Registration
// ============================================================================
std::unique_ptr<IUpscaler> createFsr1Upscaler() { return std::make_unique<Fsr1Upscaler>(); }
std::unique_ptr<IUpscaler> createFsr2Upscaler() { return std::make_unique<Fsr2Upscaler>(); }
std::unique_ptr<IUpscaler> createFsr3Upscaler() { return std::make_unique<Fsr3Upscaler>(); }

} // namespace sr

SR_REGISTER_UPSCALER("fsr1", &sr::createFsr1Upscaler);
SR_REGISTER_UPSCALER("fsr2", &sr::createFsr2Upscaler);
SR_REGISTER_UPSCALER("fsr3", &sr::createFsr3Upscaler);

// The FidelityFX VK backend checks *supported* (enumerated) extensions, then
// calls extension entry points unconditionally — e.g. the breadcrumbs setup
// calls vkGetBufferMemoryRequirements2KHR when VK_KHR_dedicated_allocation is
// supported.  vkGetDeviceProcAddr only returns extension entry points when the
// extension is enabled, so make sure it is enabled on the device.
namespace {
void appendFsrDeviceExtensions(std::vector<const char*>& deviceExts) {
    deviceExts.push_back(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
}
const ::sr::VulkanDeviceNeeds kFsrDeviceNeeds = {&appendFsrDeviceExtensions, nullptr, nullptr, nullptr};
} // namespace
SR_REGISTER_VULKAN_DEVICE_NEEDS(kFsrDeviceNeeds);
