// ============================================================================
// Arm NFRU as IFrameGen.  Same ngsdk DLL / ML emu layer as NSS.  Color is
// converted to R11G11B10 because the NFRU optical-flow path does not accept
// R16G16B16A16.  Dispatch uses NO_SWAPCHAIN so we write a texture.
// ============================================================================
#include "upscalers/IFrameGen.h"
#include "upscalers/UpscalerFactory.h"
#include "renderer/core/PathUtil.h"
#include "renderer/math/Math.h"
#include "upscalers/VkHelpers.h"
#include "upscalers/Cmdline.h"

#define FFX_CPU
#include <ffx_api/ffx_api.h>
#include <ffx_api/ffx_api_types.h>
#include <ffx_api/ffx_framegeneration.h>
#include <ffx_api/vk/ffx_api_vk.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace sr {
namespace {

constexpr VkFormat kNfruColorFormat = VK_FORMAT_B10G11R11_UFLOAT_PACK32;

using PFN_ffxCreateContextDyn = ffxReturnCode_t (*)(ffxContext*, ffxCreateContextDescHeader*,
                                                    const ffxAllocationCallbacks*);
using PFN_ffxDestroyContextDyn = ffxReturnCode_t (*)(ffxContext*, const ffxAllocationCallbacks*);
using PFN_ffxDispatchDyn = ffxReturnCode_t (*)(ffxContext*, const ffxDispatchDescHeader*);
using PFN_ffxConfigureDyn = ffxReturnCode_t (*)(ffxContext*, const ffxConfigureDescHeader*);

HMODULE loadNgsdk() {
    if (HMODULE m = LoadLibraryA("ngsdk_windows_x64.dll")) return m;
#ifdef SR_NSS_RUNTIME_DIR
    return LoadLibraryA(SR_NSS_RUNTIME_DIR "\\ngsdk_windows_x64.dll");
#else
    return nullptr;
#endif
}

void nfruMsg(uint32_t type, const char* message) {
    std::fprintf(stderr, "[NFRU] %s: %s\n",
                 type == FFX_API_MESSAGE_TYPE_ERROR ? "error" : "warning",
                 message ? message : "(null)");
}

FfxApiResource wrapImage(VkImage image, VkFormat format, uint32_t width, uint32_t height,
                         uint32_t usage, uint32_t state) {
    FfxApiResource res = {};
    res.resource = reinterpret_cast<void*>(image);
    res.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    res.description.format = ffxApiGetSurfaceFormatVK(format);
    res.description.width = width;
    res.description.height = height;
    res.description.depth = 1;
    res.description.mipCount = 1;
    res.description.flags = FFX_API_RESOURCE_FLAGS_NONE;
    res.description.usage = usage;
    res.state = state;
    return res;
}

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize allocationSize = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
};

bool createImage2D(const VulkanEnv& env, uint32_t width, uint32_t height, VkFormat format,
                   VkImageUsageFlags usage, Image& out) {
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.extent = {width, height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.format = format;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ci.usage = usage;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateImage(env.device, &ci, nullptr, &out.image) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(env.device, out.image, &req);
    const uint32_t type = findMemoryType(env.physicalDevice, req.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == 0xFFFFFFFFu) {
        vkDestroyImage(env.device, out.image, nullptr);
        return false;
    }
    VkMemoryAllocateInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    if (vkAllocateMemory(env.device, &ai, nullptr, &out.memory) != VK_SUCCESS) {
        vkDestroyImage(env.device, out.image, nullptr);
        return false;
    }
    vkBindImageMemory(env.device, out.image, out.memory, 0);
    VkImageViewCreateInfo vi = {};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = out.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    if (vkCreateImageView(env.device, &vi, nullptr, &out.view) != VK_SUCCESS) return false;
    out.allocationSize = req.size;
    return true;
}

void destroyImage(const VulkanEnv& env, Image& img) {
    if (img.view) vkDestroyImageView(env.device, img.view, nullptr);
    if (img.image) vkDestroyImage(env.device, img.image, nullptr);
    if (img.memory) vkFreeMemory(env.device, img.memory, nullptr);
    img = {};
}

void transitionImage(VkCommandBuffer cmd, Image& img, VkImageLayout newLayout) {
    if (img.layout == newLayout) return;
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = img.layout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = img.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (img.layout == VK_IMAGE_LAYOUT_UNDEFINED) srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    if (newLayout == VK_IMAGE_LAYOUT_GENERAL)
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    else
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    img.layout = newLayout;
}

} // namespace

class NfruFrameGen : public IFrameGen {
public:
    ~NfruFrameGen() override { shutdown(); }
    const char* name() const override { return "NFRU"; }

    bool isAvailable(const VulkanEnv& env) override {
        if (env.device == VK_NULL_HANDLE) return false;
        uint32_t count = 0;
        vkEnumerateDeviceExtensionProperties(env.physicalDevice, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> exts(count);
        vkEnumerateDeviceExtensionProperties(env.physicalDevice, nullptr, &count, exts.data());
        bool hasTensors = false, hasGraph = false, hasOf = false;
        for (const auto& e : exts) {
            if (std::strcmp(e.extensionName, VK_ARM_TENSORS_EXTENSION_NAME) == 0) hasTensors = true;
            if (std::strcmp(e.extensionName, VK_ARM_DATA_GRAPH_EXTENSION_NAME) == 0) hasGraph = true;
            if (std::strcmp(e.extensionName, VK_ARM_DATA_GRAPH_OPTICAL_FLOW_EXTENSION_NAME) == 0)
                hasOf = true;
        }
        if (!hasTensors || !hasGraph) {
            std::fprintf(stderr, "NFRU: ARM ML extensions missing (use run_with_nss.bat)\n");
            return false;
        }
        if (!hasOf)
            std::fprintf(stderr,
                         "NFRU: VK_ARM_data_graph_optical_flow not enumerated; emu layer may "
                         "still inject it at create-device time.\n");
        HMODULE probe = loadNgsdk();
        if (!probe) {
            std::fprintf(stderr, "NFRU: ngsdk_windows_x64.dll not found\n");
            return false;
        }
        FreeLibrary(probe);
        return true;
    }

    bool init(const VulkanEnv& env, const FrameGenDesc& desc) override {
        shutdown();
        env_ = env;
        desc_ = desc;
        if (!createImage2D(env, desc.displayWidth, desc.displayHeight, kNfruColorFormat,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, colorIn_) ||
            !createImage2D(env, desc.displayWidth, desc.displayHeight, kNfruColorFormat,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, colorOut_)) {
            shutdown();
            return false;
        }
        memoryBytes_ = colorIn_.allocationSize + colorOut_.allocationSize;
        if (!initConvert(env)) {
            shutdown();
            return false;
        }

        sdk_ = loadNgsdk();
        if (!sdk_) {
            shutdown();
            return false;
        }
        ffxCreate_ = reinterpret_cast<PFN_ffxCreateContextDyn>(GetProcAddress(sdk_, "ffxCreateContext"));
        ffxDestroy_ = reinterpret_cast<PFN_ffxDestroyContextDyn>(GetProcAddress(sdk_, "ffxDestroyContext"));
        ffxDispatch_ = reinterpret_cast<PFN_ffxDispatchDyn>(GetProcAddress(sdk_, "ffxDispatch"));
        ffxConfigure_ = reinterpret_cast<PFN_ffxConfigureDyn>(GetProcAddress(sdk_, "ffxConfigure"));
        if (!ffxCreate_ || !ffxDestroy_ || !ffxDispatch_ || !ffxConfigure_) {
            std::fprintf(stderr, "NFRU: missing FFX entry points in ngsdk DLL\n");
            shutdown();
            return false;
        }

        ffxCreateBackendVKDesc backendDesc = {};
        backendDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
        backendDesc.vkDevice = env.device;
        backendDesc.vkPhysicalDevice = env.physicalDevice;
        backendDesc.vkInstance = env.instance;
        backendDesc.vkDeviceProcAddr = vkGetDeviceProcAddr;
        backendDesc.vkGetInstanceProcAddr = env.getInstanceProcAddr;

        ffxApiCreateContextDescFrameGeneration fgDesc = {};
        fgDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
        fgDesc.header.pNext = &backendDesc.header;
        fgDesc.displaySize = {desc.displayWidth, desc.displayHeight};
        fgDesc.renderSize = {desc.renderWidth, desc.renderHeight};
        fgDesc.backBufferFormat = ffxApiGetSurfaceFormatVK(kNfruColorFormat);
        fgDesc.fpMessage = &nfruMsg;
        fgDesc.flags = FFX_API_FG_CONTEXT_FLAG_ENABLE_HIGH_DYNAMIC_RANGE |
                       FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_DEPTH |
                       FFX_API_FG_CONTEXT_FLAG_MANAGE_PREVIOUS_COLOR;
        if (desc.infiniteFarPlane) fgDesc.flags |= FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INFINITE;
        if (desc.invertedDepth) fgDesc.flags |= FFX_API_FG_CONTEXT_FLAG_ENABLE_DEPTH_INVERTED;
        for (int i = 0; i < 16; ++i) fgDesc.initialViewProjection[i] = (i % 5 == 0) ? 1.f : 0.f;

        const ffxReturnCode_t rc = ffxCreate_(&ctx_, &fgDesc.header, nullptr);
        if (rc != FFX_API_RETURN_OK || !ctx_) {
            std::fprintf(stderr, "NFRU: ffxCreateContext failed (rc=%u)\n", rc);
            ctx_ = nullptr;
            shutdown();
            return false;
        }

        ffxApiConfigureDescFrameGeneration cfg = {};
        cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
        cfg.swapChain = nullptr;
        cfg.frameGenerationEnabled = true;
        cfg.flags = FFX_API_FG_DISPATCH_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY;
        const ffxReturnCode_t crc = ffxConfigure_(&ctx_, &cfg.header);
        if (crc != FFX_API_RETURN_OK)
            std::fprintf(stderr, "NFRU: ffxConfigure failed (rc=%u)\n", crc);
        return true;
    }

    void dispatch(VkCommandBuffer cmd, const FrameGenResources& res, const CameraParams& cam,
                  const FrameParams& frame) override {
        if (!ctx_) return;
        const uint32_t dw = desc_.displayWidth, dh = desc_.displayHeight;
        const uint32_t rw = desc_.renderWidth, rh = desc_.renderHeight;
        const int slot = (std::max)(frame.frameIndex, 0) % 2;

        transitionImage(cmd, colorIn_, VK_IMAGE_LAYOUT_GENERAL);
        recordConvert(cmd, slot, true, res.colorView, colorIn_.view, dw, dh);
        transitionImage(cmd, colorIn_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        transitionImage(cmd, colorOut_, VK_IMAGE_LAYOUT_GENERAL);

        Mat4 view, proj;
        std::memcpy(view.m, cam.view, sizeof(view.m));
        std::memcpy(proj.m, cam.proj, sizeof(proj.m));
        const Mat4 vp = Mat4::multiply(proj, view);

        ffxApiDispatchDescFrameGenerationPrepare prep = {};
        prep.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
        prep.frameID = static_cast<uint64_t>((std::max)(frame.frameIndex, 0));
        prep.commandList = reinterpret_cast<void*>(cmd);
        prep.jitterOffset.x = -cam.jitterX;
        prep.jitterOffset.y = -cam.jitterY;
        prep.motionVectorScale.x = -1.f;
        prep.motionVectorScale.y = -1.f;
        prep.frameTimeDelta = (std::max)(frame.deltaTime * 1000.f, 1.f);
        prep.cameraNear = cam.cameraNear;
        prep.cameraFar = cam.cameraFar;
        prep.cameraFovAngleVertical = cam.fovY;
        prep.viewSpaceToMetersFactor = 1.f;
        prep.depth = wrapImage(res.depth, VK_FORMAT_D32_SFLOAT, rw, rh,
                               FFX_API_RESOURCE_USAGE_READ_ONLY,
                               FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        prep.motionVectors = wrapImage(res.motion, VK_FORMAT_R16G16_SFLOAT, rw, rh,
                                       FFX_API_RESOURCE_USAGE_READ_ONLY,
                                       FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        std::memcpy(prep.viewProjection, vp.m, sizeof(prep.viewProjection));
        ffxReturnCode_t rc = ffxDispatch_(&ctx_, &prep.header);
        if (rc != FFX_API_RETURN_OK)
            std::fprintf(stderr, "NFRU: prepare failed (rc=%u)\n", rc);

        ffxApiDispatchDescFrameGeneration disp = {};
        disp.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
        disp.commandList = reinterpret_cast<void*>(cmd);
        disp.presentColor = wrapImage(colorIn_.image, kNfruColorFormat, dw, dh,
                                      FFX_API_RESOURCE_USAGE_READ_ONLY,
                                      FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        disp.outputs[0] = wrapImage(colorOut_.image, kNfruColorFormat, dw, dh,
                                    FFX_API_RESOURCE_USAGE_UAV, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        disp.numGeneratedFrames = 1;
        disp.reset = frame.resetHistory;
        disp.frameID = prep.frameID;
        rc = ffxDispatch_(&ctx_, &disp.header);
        if (rc != FFX_API_RETURN_OK)
            std::fprintf(stderr, "NFRU: dispatch failed (rc=%u)\n", rc);

        transitionImage(cmd, colorOut_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        recordConvert(cmd, slot, false, colorOut_.view, res.outputView, dw, dh);
    }

    void shutdown() override {
        if (ctx_ && ffxDestroy_) {
            ffxDestroy_(&ctx_, nullptr);
            ctx_ = nullptr;
        }
        if (convertInPipe_) vkDestroyPipeline(env_.device, convertInPipe_, nullptr);
        if (convertOutPipe_) vkDestroyPipeline(env_.device, convertOutPipe_, nullptr);
        if (convertLayout_) vkDestroyPipelineLayout(env_.device, convertLayout_, nullptr);
        if (pool_) vkDestroyDescriptorPool(env_.device, pool_, nullptr);
        if (setLayout_) vkDestroyDescriptorSetLayout(env_.device, setLayout_, nullptr);
        if (sampler_) vkDestroySampler(env_.device, sampler_, nullptr);
        convertInPipe_ = convertOutPipe_ = VK_NULL_HANDLE;
        convertLayout_ = VK_NULL_HANDLE;
        pool_ = VK_NULL_HANDLE;
        setLayout_ = VK_NULL_HANDLE;
        sampler_ = VK_NULL_HANDLE;
        destroyImage(env_, colorIn_);
        destroyImage(env_, colorOut_);
        if (sdk_) {
            FreeLibrary(sdk_);
            sdk_ = nullptr;
        }
        ffxCreate_ = nullptr;
        ffxDestroy_ = nullptr;
        ffxDispatch_ = nullptr;
        ffxConfigure_ = nullptr;
        memoryBytes_ = 0;
    }

    uint64_t gpuMemoryBytes() const override { return memoryBytes_; }

private:
    bool initConvert(const VulkanEnv& env) {
        VkSamplerCreateInfo sci = {};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = VK_FILTER_NEAREST;
        sci.minFilter = VK_FILTER_NEAREST;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(env.device, &sci, nullptr, &sampler_) != VK_SUCCESS) return false;
        VkDescriptorSetLayoutBinding bindings[2] = {};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo lci = {};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 2;
        lci.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(env.device, &lci, nullptr, &setLayout_) != VK_SUCCESS)
            return false;
        VkDescriptorPoolSize poolSizes[2] = {};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[0].descriptorCount = 4;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        poolSizes[1].descriptorCount = 4;
        VkDescriptorPoolCreateInfo pci = {};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 4;
        pci.poolSizeCount = 2;
        pci.pPoolSizes = poolSizes;
        if (vkCreateDescriptorPool(env.device, &pci, nullptr, &pool_) != VK_SUCCESS) return false;
        VkDescriptorSetLayout layouts[2] = {setLayout_, setLayout_};
        VkDescriptorSetAllocateInfo sai = {};
        sai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        sai.descriptorPool = pool_;
        sai.descriptorSetCount = 2;
        sai.pSetLayouts = layouts;
        if (vkAllocateDescriptorSets(env.device, &sai, convertInSet_) != VK_SUCCESS ||
            vkAllocateDescriptorSets(env.device, &sai, convertOutSet_) != VK_SUCCESS)
            return false;
        VkPipelineLayoutCreateInfo plci = {};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &setLayout_;
        if (vkCreatePipelineLayout(env.device, &plci, nullptr, &convertLayout_) != VK_SUCCESS)
            return false;
        auto makePipe = [&](const char* shader, VkPipeline* out) {
            VkShaderModule module = loadShader(env.device, resolveShaderPath(SR_SHADER_DIR, shader).c_str());
            if (!module) return false;
            VkComputePipelineCreateInfo cpci = {};
            cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cpci.stage.module = module;
            cpci.stage.pName = "main";
            cpci.layout = convertLayout_;
            const VkResult r = createComputePipeline(env, cpci, *out);
            vkDestroyShaderModule(env.device, module, nullptr);
            return r == VK_SUCCESS;
        };
        return makePipe("nss_convert_in.comp.spv", &convertInPipe_) &&
               makePipe("nss_convert_out.comp.spv", &convertOutPipe_);
    }

    void recordConvert(VkCommandBuffer cmd, int slot, bool toR11, VkImageView src, VkImageView dst,
                       uint32_t w, uint32_t h) {
        VkDescriptorSet set = toR11 ? convertInSet_[slot] : convertOutSet_[slot];
        VkDescriptorImageInfo infos[2] = {};
        infos[0].sampler = sampler_;
        infos[0].imageView = src;
        infos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        infos[1].imageView = dst;
        infos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet writes[2] = {};
        for (int i = 0; i < 2; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType =
                i == 0 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[i].pImageInfo = &infos[i];
        }
        vkUpdateDescriptorSets(env_.device, 2, writes, 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          toR11 ? convertInPipe_ : convertOutPipe_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, convertLayout_, 0, 1, &set, 0,
                                nullptr);
        vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
    }

    VulkanEnv env_{};
    FrameGenDesc desc_{};
    Image colorIn_, colorOut_;
    VkSampler sampler_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet convertInSet_[2] = {};
    VkDescriptorSet convertOutSet_[2] = {};
    VkPipelineLayout convertLayout_ = VK_NULL_HANDLE;
    VkPipeline convertInPipe_ = VK_NULL_HANDLE;
    VkPipeline convertOutPipe_ = VK_NULL_HANDLE;
    HMODULE sdk_ = nullptr;
    PFN_ffxCreateContextDyn ffxCreate_ = nullptr;
    PFN_ffxDestroyContextDyn ffxDestroy_ = nullptr;
    PFN_ffxDispatchDyn ffxDispatch_ = nullptr;
    PFN_ffxConfigureDyn ffxConfigure_ = nullptr;
    ffxContext ctx_ = nullptr;
    uint64_t memoryBytes_ = 0;
};

std::unique_ptr<IFrameGen> createNfruFrameGen() { return std::make_unique<NfruFrameGen>(); }

} // namespace sr

SR_REGISTER_FRAMEGEN("nfru", &sr::createNfruFrameGen);
