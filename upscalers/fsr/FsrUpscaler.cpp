// ============================================================================
// AMD FSR1 / FSR2 / FSR3.1 upscalers via the FidelityFX SDK host components
// (ffx_fsr1 / ffx_fsr2 / ffx_fsr3upscaler) and the SDK's Vulkan backend
// (ffx_vk).  Shader permutation headers are pre-generated into
// ffx_permutations/ by gen_permutations.sh.
//
// Input conventions (see upscalers/InputAdapter.h):
//   * motion vectors are current->previous framebuffer UV offsets, y down and
//     no jitter. motionVectorScale converts them to input pixels; the SDK then
//     normalizes them internally for reprojection.
//   * depth is D32_SFLOAT, non-inverted, with the host-selected far-plane mode.
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

#include "renderer/core/PathUtil.h"
#include "upscalers/InputAdapter.h"
#include "upscalers/UpscalerFactory.h"
#include "upscalers/VkHelpers.h"

#include <FidelityFX/host/backends/vk/ffx_vk.h>
#include <FidelityFX/host/ffx_fsr1.h>
#include <FidelityFX/host/ffx_fsr2.h>
#include <FidelityFX/host/ffx_fsr3upscaler.h>

#include <cfloat>
#include <cmath>
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

void transitionToGeneral(const VulkanEnv& env, const OwnedImage* images, uint32_t count);

// Spatial FSR1 has no history, so a shared Halton-jittered GBuffer would
// shake.  This pass resamples color at uv + jitter/size into a stable copy.
struct UnjitterPass {
    OwnedImage color;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    bool inited = false;
};

void destroyUnjitter(const VulkanEnv& env, UnjitterPass& p) {
    if (p.pipeline) { vkDestroyPipeline(env.device, p.pipeline, nullptr); p.pipeline = VK_NULL_HANDLE; }
    if (p.pipeLayout) { vkDestroyPipelineLayout(env.device, p.pipeLayout, nullptr); p.pipeLayout = VK_NULL_HANDLE; }
    if (p.pool) { vkDestroyDescriptorPool(env.device, p.pool, nullptr); p.pool = VK_NULL_HANDLE; }
    if (p.setLayout) { vkDestroyDescriptorSetLayout(env.device, p.setLayout, nullptr); p.setLayout = VK_NULL_HANDLE; }
    if (p.sampler) { vkDestroySampler(env.device, p.sampler, nullptr); p.sampler = VK_NULL_HANDLE; }
    if (p.view) { vkDestroyImageView(env.device, p.view, nullptr); p.view = VK_NULL_HANDLE; }
    destroyStorageImage(env, p.color);
    p.inited = false;
}

bool initUnjitter(const VulkanEnv& env, uint32_t w, uint32_t h, UnjitterPass& p) {
    destroyUnjitter(env, p);
    if (!createStorageImage(env, w, h, VK_FORMAT_R16G16B16A16_SFLOAT, p.color)) return false;
    VkImageViewCreateInfo vi = {};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = p.color.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    if (vkCreateImageView(env.device, &vi, nullptr, &p.view) != VK_SUCCESS) return false;

    VkSamplerCreateInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(env.device, &si, nullptr, &p.sampler) != VK_SUCCESS) return false;

    VkDescriptorSetLayoutBinding b[2] = {};
    b[0].binding = 0;
    b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[0].descriptorCount = 1;
    b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b[1].binding = 1;
    b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b[1].descriptorCount = 1;
    b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo lci = {};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 2;
    lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(env.device, &lci, nullptr, &p.setLayout) != VK_SUCCESS)
        return false;

    VkPushConstantRange pc = {};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.size = 16;
    VkPipelineLayoutCreateInfo pl = {};
    pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &p.setLayout;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(env.device, &pl, nullptr, &p.pipeLayout) != VK_SUCCESS)
        return false;

    VkShaderModule mod = loadShader(env.device,
        resolveShaderPath(SR_SHADER_DIR, "fsr1_unjitter.comp.spv").c_str());
    if (!mod) return false;
    VkPipelineShaderStageCreateInfo st = {};
    st.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    st.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    st.module = mod;
    st.pName = "main";
    VkComputePipelineCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.stage = st;
    ci.layout = p.pipeLayout;
    const VkResult pr = createComputePipeline(env, ci, p.pipeline);
    vkDestroyShaderModule(env.device, mod, nullptr);
    if (pr != VK_SUCCESS) return false;

    VkDescriptorPoolSize ps[2] = {};
    ps[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ps[0].descriptorCount = 1;
    ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ps[1].descriptorCount = 1;
    VkDescriptorPoolCreateInfo pci = {};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 1;
    pci.poolSizeCount = 2;
    pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(env.device, &pci, nullptr, &p.pool) != VK_SUCCESS) return false;
    VkDescriptorSetAllocateInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = p.pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &p.setLayout;
    if (vkAllocateDescriptorSets(env.device, &ai, &p.set) != VK_SUCCESS) return false;

    transitionToGeneral(env, &p.color, 1);
    p.inited = true;
    return true;
}

void recordUnjitter(VkCommandBuffer cmd, VkDevice device, UnjitterPass& p, VkImageView srcView,
                    float jx, float jy, uint32_t w, uint32_t h) {
    VkDescriptorImageInfo in = {};
    in.sampler = p.sampler;
    in.imageView = srcView;
    in.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo out = {};
    out.imageView = p.view;
    out.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet wr[2] = {};
    wr[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wr[0].dstSet = p.set;
    wr[0].dstBinding = 0;
    wr[0].descriptorCount = 1;
    wr[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr[0].pImageInfo = &in;
    wr[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wr[1].dstSet = p.set;
    wr[1].dstBinding = 1;
    wr[1].descriptorCount = 1;
    wr[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    wr[1].pImageInfo = &out;
    vkUpdateDescriptorSets(device, 2, wr, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeLayout, 0, 1, &p.set, 0,
                            nullptr);
    const float push[4] = {jx, jy, static_cast<float>(w), static_cast<float>(h)};
    vkCmdPushConstants(cmd, p.pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
    vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);

    VkImageMemoryBarrier b = {};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = p.color.image;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
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
    UnjitterPass unjitter;
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
    if (!initUnjitter(env, desc.renderWidth, desc.renderHeight, impl_->unjitter)) {
        std::fprintf(stderr, "FSR1: unjitter pass init failed\n");
        shutdown();
        return false;
    }
    impl_->memoryBytes += impl_->unjitter.color.bytes;
    return true;
}

void Fsr1Upscaler::dispatch(VkCommandBuffer cmd, const UpscalerResources& res, const CameraParams& cam,
                            const FrameParams& frame) {
    (void)frame;
    if (!impl_ || !impl_->contextCreated) return;

    FrameFfxResources f = wrapFrameResources(res, impl_->desc);
    const bool needUnjitter = impl_->unjitter.inited &&
                              (std::fabs(cam.jitterX) > 1e-6f || std::fabs(cam.jitterY) > 1e-6f);
    if (needUnjitter) {
        recordUnjitter(cmd, impl_->env.device, impl_->unjitter, res.colorView, cam.jitterX,
                       cam.jitterY, impl_->desc.renderWidth, impl_->desc.renderHeight);
        f.color = ffxGetResourceVK(
            impl_->unjitter.color.image,
            imageDesc(FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT, impl_->desc.renderWidth,
                      impl_->desc.renderHeight, FFX_RESOURCE_USAGE_READ_ONLY),
            L"SR_FSR1_UnjitteredColor", FFX_RESOURCE_STATE_COMPUTE_READ);
    }

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
    destroyUnjitter(impl_->env, impl_->unjitter);
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
    const MotionScale motionScale =
        fsrMotionVectorScale(impl_->desc.renderWidth, impl_->desc.renderHeight);
    dd.motionVectorScale = {motionScale.x, motionScale.y};
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
    const MotionScale motionScale = fsrMotionVectorScale(rw, rh);
    dd.motionVectorScale = {motionScale.x, motionScale.y};
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
