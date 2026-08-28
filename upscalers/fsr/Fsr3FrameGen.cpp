// ============================================================================
// FSR3 frame interpolation as IFrameGen: optical flow + FI, dispatch to a
// texture (no FrameInterpolationSwapchain).
// ============================================================================
#include "upscalers/IFrameGen.h"
#include "upscalers/InputAdapter.h"
#include "upscalers/UpscalerFactory.h"
#include "upscalers/VkHelpers.h"
#include "renderer/math/Math.h"

#include <FidelityFX/host/backends/vk/ffx_vk.h>
#include <FidelityFX/host/ffx_frameinterpolation.h>
#include <FidelityFX/host/ffx_opticalflow.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace sr {
namespace {

constexpr size_t kMaxFfxContexts = 8;

FfxGetDeviceCapabilitiesFunc g_origGetDeviceCaps = nullptr;

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

struct FfxVkBackend {
    VkDeviceContext deviceContext = {};
    FfxInterface iface = {};
    void* scratch = nullptr;

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
    if (img.image) {
        vkDestroyImage(env.device, img.image, nullptr);
        img.image = VK_NULL_HANDLE;
    }
    if (img.memory) {
        vkFreeMemory(env.device, img.memory, nullptr);
        img.memory = VK_NULL_HANDLE;
    }
}

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
    vkBeginCommandBuffer(cmd, &begin);
    for (uint32_t i = 0; i < count; ++i) {
        if (!images[i].image) continue;
        VkImageMemoryBarrier b = {};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = images[i].image;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                             &b);
    }
    vkEndCommandBuffer(cmd);
    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    if (env.queueMutex) {
        std::lock_guard<std::mutex> lk(*env.queueMutex);
        vkQueueSubmit(env.graphicsQueue, 1, &submit, VK_NULL_HANDLE);
        vkQueueWaitIdle(env.graphicsQueue);
    } else {
        vkQueueSubmit(env.graphicsQueue, 1, &submit, VK_NULL_HANDLE);
        vkQueueWaitIdle(env.graphicsQueue);
    }
    vkFreeCommandBuffers(env.device, env.commandPool, 1, &cmd);
}

} // namespace

class Fsr3FrameGen : public IFrameGen {
public:
    ~Fsr3FrameGen() override { shutdown(); }
    const char* name() const override { return "FSR3-FG"; }
    bool isAvailable(const VulkanEnv& env) override { return env.device != VK_NULL_HANDLE; }

    bool init(const VulkanEnv& env, const FrameGenDesc& desc) override {
        shutdown();
        env_ = env;
        desc_ = desc;
        if (!backend_.init(env)) {
            shutdown();
            return false;
        }

        FfxOpticalflowContextDescription ofDesc = {};
        ofDesc.backendInterface = backend_.iface;
        ofDesc.resolution = {desc.displayWidth, desc.displayHeight};
        if (ffxOpticalflowContextCreate(&ofCtx_, &ofDesc) != FFX_OK) {
            std::fprintf(stderr, "FSR3-FG: optical flow context failed\n");
            shutdown();
            return false;
        }
        ofOk_ = true;

        FfxFrameInterpolationContextDescription fiDesc = {};
        fiDesc.backendInterface = backend_.iface;
        fiDesc.flags = FFX_FRAMEINTERPOLATION_ENABLE_HDR_COLOR_INPUT;
        if (desc.infiniteFarPlane) fiDesc.flags |= FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INFINITE;
        if (desc.invertedDepth) fiDesc.flags |= FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INVERTED;
        fiDesc.maxRenderSize = {desc.renderWidth, desc.renderHeight};
        fiDesc.displaySize = {desc.displayWidth, desc.displayHeight};
        fiDesc.backBufferFormat = FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT;
        fiDesc.previousInterpolationSourceFormat = FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT;
        if (ffxFrameInterpolationContextCreate(&fiCtx_, &fiDesc) != FFX_OK) {
            std::fprintf(stderr, "FSR3-FG: frame interpolation context failed\n");
            shutdown();
            return false;
        }
        fiOk_ = true;

        FfxOpticalflowSharedResourceDescriptions ofShared = {};
        ffxOpticalflowGetSharedResourceDescriptions(&ofCtx_, &ofShared);
        const uint32_t ofW = ofShared.opticalFlowVector.resourceDescription.width;
        const uint32_t ofH = ofShared.opticalFlowVector.resourceDescription.height;
        if (!createStorageImage(env, ofW, ofH, VK_FORMAT_R16G16_SINT, ofVector_) ||
            !createStorageImage(env, ofShared.opticalFlowSCD.resourceDescription.width,
                                ofShared.opticalFlowSCD.resourceDescription.height,
                                VK_FORMAT_R32_UINT, ofScd_)) {
            shutdown();
            return false;
        }
        ofW_ = ofW;
        ofH_ = ofH;

        const uint32_t rw = desc.renderWidth, rh = desc.renderHeight;
        if (!createStorageImage(env, rw, rh, VK_FORMAT_R32_SFLOAT, dilatedDepth_) ||
            !createStorageImage(env, rw, rh, VK_FORMAT_R16G16_SFLOAT, dilatedMv_) ||
            !createStorageImage(env, rw, rh, VK_FORMAT_R32_UINT, reconDepth_)) {
            shutdown();
            return false;
        }

        const OwnedImage imgs[] = {ofVector_, ofScd_, dilatedDepth_, dilatedMv_, reconDepth_};
        transitionToGeneral(env, imgs, 5);
        memoryBytes_ = ofVector_.bytes + ofScd_.bytes + dilatedDepth_.bytes + dilatedMv_.bytes +
                       reconDepth_.bytes;
        return true;
    }

    void dispatch(VkCommandBuffer cmd, const FrameGenResources& res, const CameraParams& cam,
                  const FrameParams& frame) override {
        if (!fiOk_ || !ofOk_) return;
        const uint32_t rw = desc_.renderWidth, rh = desc_.renderHeight;
        const uint32_t dw = desc_.displayWidth, dh = desc_.displayHeight;
        const FfxCommandList cl = ffxGetCommandListVK(cmd);

        Mat4 view;
        std::memcpy(view.m, cam.view, sizeof(view.m));
        const Mat4 inv = Mat4::inverse(view);

        FfxFrameInterpolationPrepareDescription prep = {};
        prep.commandList = cl;
        prep.renderSize = {rw, rh};
        prep.jitterOffset = {cam.jitterX, cam.jitterY};
        // Same canonical UV motion as the SR path: scale to input pixels, the
        // SDK normalizes internally (see InputAdapter.h).
        const MotionScale motionScale = fsrMotionVectorScale(rw, rh);
        prep.motionVectorScale = {motionScale.x, motionScale.y};
        prep.frameTimeDelta = std::max(frame.deltaTime * 1000.f, 1.f);
        prep.cameraNear = cam.cameraNear;
        prep.cameraFar = cam.cameraFar;
        prep.viewSpaceToMetersFactor = 1.f;
        prep.cameraFovAngleVertical = cam.fovY;
        prep.depth = ffxGetResourceVK(
            res.depth,
            imageDesc(FFX_SURFACE_FORMAT_R32_FLOAT, rw, rh,
                      (FfxResourceUsage)(FFX_RESOURCE_USAGE_READ_ONLY | FFX_RESOURCE_USAGE_DEPTHTARGET)),
            L"SR_FG_Depth", FFX_RESOURCE_STATE_COMPUTE_READ);
        prep.motionVectors = ffxGetResourceVK(
            res.motion, imageDesc(FFX_SURFACE_FORMAT_R16G16_FLOAT, rw, rh, FFX_RESOURCE_USAGE_READ_ONLY),
            L"SR_FG_Motion", FFX_RESOURCE_STATE_COMPUTE_READ);
        prep.frameID = static_cast<uint64_t>(std::max(frame.frameIndex, 0));
        prep.dilatedDepth = ffxGetResourceVK(
            dilatedDepth_.image,
            imageDesc(FFX_SURFACE_FORMAT_R32_FLOAT, rw, rh,
                      (FfxResourceUsage)(FFX_RESOURCE_USAGE_UAV | FFX_RESOURCE_USAGE_RENDERTARGET)),
            L"SR_FG_DilatedDepth", FFX_RESOURCE_STATE_UNORDERED_ACCESS);
        prep.dilatedMotionVectors = ffxGetResourceVK(
            dilatedMv_.image,
            imageDesc(FFX_SURFACE_FORMAT_R16G16_FLOAT, rw, rh,
                      (FfxResourceUsage)(FFX_RESOURCE_USAGE_UAV | FFX_RESOURCE_USAGE_RENDERTARGET)),
            L"SR_FG_DilatedMV", FFX_RESOURCE_STATE_UNORDERED_ACCESS);
        prep.reconstructedPrevDepth = ffxGetResourceVK(
            reconDepth_.image, imageDesc(FFX_SURFACE_FORMAT_R32_UINT, rw, rh, FFX_RESOURCE_USAGE_UAV),
            L"SR_FG_ReconDepth", FFX_RESOURCE_STATE_UNORDERED_ACCESS);
        prep.cameraPosition[0] = inv.m[12];
        prep.cameraPosition[1] = inv.m[13];
        prep.cameraPosition[2] = inv.m[14];
        prep.cameraRight[0] = inv.m[0];
        prep.cameraRight[1] = inv.m[1];
        prep.cameraRight[2] = inv.m[2];
        prep.cameraUp[0] = inv.m[4];
        prep.cameraUp[1] = inv.m[5];
        prep.cameraUp[2] = inv.m[6];
        prep.cameraForward[0] = -inv.m[8];
        prep.cameraForward[1] = -inv.m[9];
        prep.cameraForward[2] = -inv.m[10];
        ffxFrameInterpolationPrepare(&fiCtx_, &prep);

        FfxResource ofVec = ffxGetResourceVK(
            ofVector_.image, imageDesc(FFX_SURFACE_FORMAT_R16G16_SINT, ofW_, ofH_, FFX_RESOURCE_USAGE_UAV),
            L"SR_FG_OFVec", FFX_RESOURCE_STATE_UNORDERED_ACCESS);
        FfxResource ofScd = ffxGetResourceVK(
            ofScd_.image, imageDesc(FFX_SURFACE_FORMAT_R32_UINT, 3, 1, FFX_RESOURCE_USAGE_UAV),
            L"SR_FG_OFScd", FFX_RESOURCE_STATE_UNORDERED_ACCESS);

        FfxOpticalflowDispatchDescription ofd = {};
        ofd.commandList = cl;
        ofd.color = ffxGetResourceVK(
            res.color,
            imageDesc(FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT, dw, dh, FFX_RESOURCE_USAGE_READ_ONLY),
            L"SR_FG_Color", FFX_RESOURCE_STATE_COMPUTE_READ);
        ofd.opticalFlowVector = ofVec;
        ofd.opticalFlowSCD = ofScd;
        ofd.reset = frame.resetHistory;
        ofd.backbufferTransferFunction = FFX_BACKBUFFER_TRANSFER_FUNCTION_SCRGB;
        ofd.minMaxLuminance = {0.f, 1000.f};
        ffxOpticalflowContextDispatch(&ofCtx_, &ofd);

        FfxFrameInterpolationDispatchDescription fid = {};
        fid.commandList = cl;
        fid.displaySize = {dw, dh};
        fid.renderSize = {rw, rh};
        fid.currentBackBuffer = ofd.color;
        fid.output = ffxGetResourceVK(
            res.output,
            imageDesc(FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT, dw, dh, FFX_RESOURCE_USAGE_UAV),
            L"SR_FG_Output", FFX_RESOURCE_STATE_UNORDERED_ACCESS);
        fid.interpolationRect = {0, 0, static_cast<int>(dw), static_cast<int>(dh)};
        fid.opticalFlowVector = ofVec;
        fid.opticalFlowSceneChangeDetection = ofScd;
        fid.opticalFlowBufferSize = {ofW_, ofH_};
        fid.opticalFlowScale = {1.f / static_cast<float>(dw), 1.f / static_cast<float>(dh)};
        fid.opticalFlowBlockSize = 8;
        fid.cameraNear = cam.cameraNear;
        fid.cameraFar = cam.cameraFar;
        fid.cameraFovAngleVertical = cam.fovY;
        fid.viewSpaceToMetersFactor = 1.f;
        fid.frameTimeDelta = prep.frameTimeDelta;
        fid.reset = frame.resetHistory;
        fid.backBufferTransferFunction = FFX_BACKBUFFER_TRANSFER_FUNCTION_SCRGB;
        fid.minMaxLuminance[0] = 0.f;
        fid.minMaxLuminance[1] = 1000.f;
        fid.frameID = prep.frameID;
        fid.dilatedDepth = prep.dilatedDepth;
        fid.dilatedMotionVectors = prep.dilatedMotionVectors;
        fid.reconstructedPrevDepth = prep.reconstructedPrevDepth;
        ffxFrameInterpolationDispatch(&fiCtx_, &fid);
    }

    void shutdown() override {
        if (fiOk_) {
            ffxFrameInterpolationContextDestroy(&fiCtx_);
            fiOk_ = false;
        }
        if (ofOk_) {
            ffxOpticalflowContextDestroy(&ofCtx_);
            ofOk_ = false;
        }
        destroyStorageImage(env_, ofVector_);
        destroyStorageImage(env_, ofScd_);
        destroyStorageImage(env_, dilatedDepth_);
        destroyStorageImage(env_, dilatedMv_);
        destroyStorageImage(env_, reconDepth_);
        backend_.destroy();
        memoryBytes_ = 0;
    }

    uint64_t gpuMemoryBytes() const override { return memoryBytes_; }

private:
    VulkanEnv env_{};
    FrameGenDesc desc_{};
    FfxVkBackend backend_;
    FfxOpticalflowContext ofCtx_{};
    FfxFrameInterpolationContext fiCtx_{};
    bool ofOk_ = false;
    bool fiOk_ = false;
    OwnedImage ofVector_, ofScd_, dilatedDepth_, dilatedMv_, reconDepth_;
    uint32_t ofW_ = 0, ofH_ = 0;
    uint64_t memoryBytes_ = 0;
};

std::unique_ptr<IFrameGen> createFsr3FrameGen() { return std::make_unique<Fsr3FrameGen>(); }

} // namespace sr

SR_REGISTER_FRAMEGEN("fsr3", &sr::createFsr3FrameGen);
