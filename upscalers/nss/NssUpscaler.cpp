#include "upscalers/nss/NssUpscaler.h"

#include "upscalers/InputAdapter.h"
#include "upscalers/UpscalerFactory.h"
#include "renderer/core/PathUtil.h"
#include "upscalers/VkHelpers.h"

// Arm Neural Graphics SDK public headers (FFX-style API).
#define FFX_CPU
#include <ffx_api/ffx_api.h>
#include <ffx_api/ffx_api_types.h>
#include <ffx_api/ffx_nss.h>
#include <ffx_api/vk/ffx_api_vk.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "upscalers/Cmdline.h"

namespace sr {

namespace {

constexpr VkFormat kNssColorFormat = VK_FORMAT_B10G11R11_UFLOAT_PACK32;

// ---------------------------------------------------------------------------
// Dynamically loaded Neural Graphics SDK entry points.  The SDK runtime
// (ngsdk_windows_x64.dll, prebuilt by Arm) is loaded at init time so the
// executable still runs (with NSS unavailable) when the DLL is not on PATH.
// ---------------------------------------------------------------------------
using PFN_ffxCreateContextDyn  = ffxReturnCode_t (*)(ffxContext*, ffxCreateContextDescHeader*,
                                                     const ffxAllocationCallbacks*);
using PFN_ffxDestroyContextDyn = ffxReturnCode_t (*)(ffxContext*, const ffxAllocationCallbacks*);
using PFN_ffxDispatchDyn       = ffxReturnCode_t (*)(ffxContext*, const ffxDispatchDescHeader*);

// ---------------------------------------------------------------------------
// Device requirements, declared to the renderer before vkCreateDevice.
// Static storage as required by SR_REGISTER_VULKAN_DEVICE_NEEDS.
// The renderer's own v12 node already enables shaderInt8 / vulkanMemoryModel
// (each sType may appear only once), so the chain starts at the ARM tensors
// feature.
// ---------------------------------------------------------------------------
VkPhysicalDeviceTensorFeaturesARM g_tensorFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM};
VkPhysicalDeviceDataGraphFeaturesARM g_dataGraphFeatures = {
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM};
VkPhysicalDeviceDataGraphOpticalFlowFeaturesARM g_opticalFlowFeatures = {
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_OPTICAL_FLOW_FEATURES_ARM};
VkPhysicalDeviceShaderReplicatedCompositesFeaturesEXT g_replicatedCompositesFeatures = {
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_REPLICATED_COMPOSITES_FEATURES_EXT};
bool g_featureChainInit = false;

// Gate: only contribute device/instance requirements when the run actually
// involves nss (mirrors the dlss module's command-line gating).  Without
// this, the emulation layers + ARM feature chain would be injected into
// every process of a unified build and break other upscalers.
bool nssRequestedOnCommandLine() {
    static const bool requested = sr::cmdline::pluginRequested("nss");
    // GUI mode flips the global override before device creation; the cached
    // command-line result alone would leave nss unavailable there.
    return allPluginsEnabled() || requested || sr::cmdline::pluginRequested("nfru");
}

const void* nssFeatureChain() {
    if (!nssRequestedOnCommandLine()) return nullptr;
    if (!g_featureChainInit) {
        g_tensorFeatures.tensors            = VK_TRUE;
        g_tensorFeatures.shaderTensorAccess = VK_TRUE;
        g_tensorFeatures.pNext              = &g_dataGraphFeatures;

        g_dataGraphFeatures.dataGraph             = VK_TRUE;
        g_dataGraphFeatures.dataGraphShaderModule = VK_TRUE;  // SDK ships the VGF as a shader module
        g_dataGraphFeatures.pNext                 = &g_opticalFlowFeatures;

        g_opticalFlowFeatures.dataGraphOpticalFlow = VK_TRUE;
        g_opticalFlowFeatures.pNext                = &g_replicatedCompositesFeatures;

        g_replicatedCompositesFeatures.shaderReplicatedComposites = VK_TRUE;
        g_replicatedCompositesFeatures.pNext                      = nullptr;

        g_featureChainInit = true;
    }
    return &g_tensorFeatures;
}

// Forward declarations (definitions below, next to g_nssNeeds).
void nssEnsureLayerPath();
HMODULE loadNgsdk();

void nssAppendDeviceExtensions(std::vector<const char*>& exts) {
    if (!nssRequestedOnCommandLine()) return;
    exts.push_back(VK_ARM_TENSORS_EXTENSION_NAME);
    exts.push_back(VK_ARM_DATA_GRAPH_EXTENSION_NAME);
    exts.push_back(VK_ARM_DATA_GRAPH_OPTICAL_FLOW_EXTENSION_NAME);
    // Dependencies of VK_ARM_data_graph (dynamic rendering is core 1.3 and
    // already enabled by the renderer).
    exts.push_back(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
    exts.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    // SDK shaders declare the SPV_EXT_replicated_composites extension.
    exts.push_back(VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME);
}

void nssAppendInstanceLayers(std::vector<const char*>& layers) {
    if (!nssRequestedOnCommandLine()) return;
    nssEnsureLayerPath();  // runs before vkCreateInstance; loader sees it
    // Order matters: the graph emulation layer must sit above the tensor
    // emulation layer (Arm developer guide).
    layers.push_back("VK_LAYER_ML_Graph_Emulation");
    layers.push_back("VK_LAYER_ML_Tensor_Emulation");
}

// ---------------------------------------------------------------------------
// Self-contained runtime setup (paths baked at build time, see CMakeLists):
// the emulation-layer manifests are appended to VK_ADD_LAYER_PATH before
// vkCreateInstance, and the SDK runtime is loaded by absolute path, so
// `--upscaler nss` works without run_with_nss.bat.
// ---------------------------------------------------------------------------
void nssEnsureLayerPath() {
#ifdef SR_NSS_EMU_GRAPH_DIR
    static const bool done = [] {
        const char* cur = std::getenv("VK_ADD_LAYER_PATH");
        std::string value;
        if (cur && *cur) value = std::string(cur) + ";";
        // Packaged layout (<exe>/nss-emu/{graph,tensor}) takes precedence;
        // the baked absolute dirs cover the dev machine.  (Probe for the
        // manifest files since resolveAssetPath checks regular files.)
        const std::string pkgGraph = resolveAssetPath("nss-emu/graph/VkLayer_Graph.json");
        const std::string pkgTensor = resolveAssetPath("nss-emu/tensor/VkLayer_Tensor.json");
        const bool havePkg = pkgGraph.find("VkLayer_Graph.json") != std::string::npos &&
                             assetFileExists(pkgGraph);
        const bool havePkgT = assetFileExists(pkgTensor);
        if (havePkg && havePkgT) {
            value += pkgGraph.substr(0, pkgGraph.find_last_of("\\/")) + ";" +
                     pkgTensor.substr(0, pkgTensor.find_last_of("\\/"));
        } else {
            value += SR_NSS_EMU_GRAPH_DIR ";" SR_NSS_EMU_TENSOR_DIR;
        }
        _putenv_s("VK_ADD_LAYER_PATH", value.c_str());
        return true;
    }();
    (void)done;
#endif
}

HMODULE loadNgsdk() {
    if (HMODULE m = LoadLibraryA("ngsdk_windows_x64.dll")) return m;  // PATH (run_with_nss.bat)
#ifdef SR_NSS_RUNTIME_DIR
    return LoadLibraryA(SR_NSS_RUNTIME_DIR "\\ngsdk_windows_x64.dll");
#else
    return nullptr;
#endif
}

VulkanDeviceNeeds g_nssNeeds = {&nssAppendDeviceExtensions, &nssFeatureChain, nullptr,
                                &nssAppendInstanceLayers};
// ---------------------------------------------------------------------------
// Small Vulkan helpers (same patterns as the TAA baseline).
// ---------------------------------------------------------------------------
struct Image {
    VkImage        image          = VK_NULL_HANDLE;
    VkImageView    view           = VK_NULL_HANDLE;
    VkDeviceMemory memory         = VK_NULL_HANDLE;
    VkDeviceSize   allocationSize = 0;
    VkImageLayout  layout         = VK_IMAGE_LAYOUT_UNDEFINED;
};

bool createImage2D(const VulkanEnv& env, uint32_t width, uint32_t height, VkFormat format,
                   VkImageUsageFlags usage, Image& out) {
    VkImageCreateInfo ci = {};
    ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType     = VK_IMAGE_TYPE_2D;
    ci.extent        = {width, height, 1};
    ci.mipLevels     = 1;
    ci.arrayLayers   = 1;
    ci.format        = format;
    ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ci.usage         = usage;
    ci.samples       = VK_SAMPLE_COUNT_1_BIT;
    ci.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateImage(env.device, &ci, nullptr, &out.image) != VK_SUCCESS) return false;

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(env.device, out.image, &req);
    const uint32_t type =
        findMemoryType(env.physicalDevice, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == 0xFFFFFFFFu) {
        vkDestroyImage(env.device, out.image, nullptr);
        out.image = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryAllocateInfo ai = {};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = type;
    if (vkAllocateMemory(env.device, &ai, nullptr, &out.memory) != VK_SUCCESS) {
        vkDestroyImage(env.device, out.image, nullptr);
        out.image = VK_NULL_HANDLE;
        return false;
    }
    vkBindImageMemory(env.device, out.image, out.memory, 0);
    out.allocationSize = req.size;

    VkImageViewCreateInfo vi = {};
    vi.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image    = out.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format   = format;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    if (vkCreateImageView(env.device, &vi, nullptr, &out.view) != VK_SUCCESS) {
        vkFreeMemory(env.device, out.memory, nullptr);
        vkDestroyImage(env.device, out.image, nullptr);
        out.image = VK_NULL_HANDLE;
        out.memory = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

void destroyImage(VkDevice device, Image& img) {
    if (img.view) { vkDestroyImageView(device, img.view, nullptr); img.view = VK_NULL_HANDLE; }
    if (img.image) { vkDestroyImage(device, img.image, nullptr); img.image = VK_NULL_HANDLE; }
    if (img.memory) { vkFreeMemory(device, img.memory, nullptr); img.memory = VK_NULL_HANDLE; }
}

void transitionImage(VkCommandBuffer cmd, Image& img, VkImageLayout newLayout) {
    if (img.layout == newLayout) return;
    VkImageMemoryBarrier barrier = {};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout           = img.layout;
    barrier.newLayout           = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = img.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    if (img.layout == VK_IMAGE_LAYOUT_GENERAL) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        srcStage              = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (newLayout == VK_IMAGE_LAYOUT_GENERAL)
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    else if (newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    img.layout = newLayout;
}

void nssMessageCallback(uint32_t type, const char* message) {
    std::fprintf(stderr, "[NSS] %s: %s\n",
                 type == FFX_API_MESSAGE_TYPE_ERROR ? "error" : "warning",
                 message ? message : "(null)");
}

// Wrap a VkImage as an FfxApiResource without needing its original create info.
FfxApiResource wrapImage(VkImage image, VkFormat format, uint32_t width, uint32_t height,
                         uint32_t usage, uint32_t state) {
    FfxApiResource res           = {};
    res.resource                 = reinterpret_cast<void*>(image);
    res.description.type         = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    res.description.format       = ffxApiGetSurfaceFormatVK(format);
    res.description.width        = width;
    res.description.height       = height;
    res.description.depth        = 1;
    res.description.mipCount     = 1;
    res.description.flags        = FFX_API_RESOURCE_FLAGS_NONE;
    res.description.usage        = usage;
    res.state                    = state;
    return res;
}

} // namespace

struct NssUpscaler::Impl {
    VulkanEnv   env;
    UpscalerDesc desc;

    // Format conversion passes (R16G16B16A16 <-> R11G11B10).
    VkSampler            sampler           = VK_NULL_HANDLE;
    VkDescriptorSetLayout convertSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool     descriptorPool    = VK_NULL_HANDLE;
    VkDescriptorSet      convertInSet[2]   = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSet      convertOutSet[2]  = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkPipelineLayout     convertLayout     = VK_NULL_HANDLE;
    VkPipeline           convertInPipeline  = VK_NULL_HANDLE;
    VkPipeline           convertOutPipeline = VK_NULL_HANDLE;

    Image   nssColor;   // render resolution,  R11G11B10 (SDK color input)
    Image   nssOutput;  // display resolution, R11G11B10 (SDK output)
    uint64_t memoryBytes = 0;

    // Neural Graphics SDK runtime (dll + context).
    HMODULE                 sdkModule       = nullptr;
    PFN_ffxCreateContextDyn  ffxCreateCtx   = nullptr;
    PFN_ffxDestroyContextDyn ffxDestroyCtx  = nullptr;
    PFN_ffxDispatchDyn       ffxDispatchFn  = nullptr;
    ffxContext               context        = nullptr;
};

NssUpscaler::~NssUpscaler() {
    shutdown();
}

const char* NssUpscaler::name() const {
    return "NSS";
}

uint32_t NssUpscaler::capabilities() const {
    return Cap_Temporal | Cap_ML;
}

bool NssUpscaler::isAvailable(const VulkanEnv& env) {
    if (env.device == VK_NULL_HANDLE || env.physicalDevice == VK_NULL_HANDLE) return false;

    // The VK_ARM ML extensions are injected by the emulation layers, so they
    // only enumerate when the loader found the layers (VK_ADD_LAYER_PATH set
    // before process start, e.g. via run_with_nss.bat).
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(env.physicalDevice, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> exts(count);
    vkEnumerateDeviceExtensionProperties(env.physicalDevice, nullptr, &count, exts.data());
    bool hasTensors = false, hasDataGraph = false;
    for (const auto& e : exts) {
        if (std::strcmp(e.extensionName, VK_ARM_TENSORS_EXTENSION_NAME) == 0) hasTensors = true;
        if (std::strcmp(e.extensionName, VK_ARM_DATA_GRAPH_EXTENSION_NAME) == 0) hasDataGraph = true;
    }
    if (!hasTensors || !hasDataGraph) {
        std::fprintf(stderr,
                     "NSS: %s/%s not exposed by the device. On PC these require Arm's ML "
                     "Emulation Layer - start the app via upscalers/nss/run_with_nss.bat "
                     "(sets VK_ADD_LAYER_PATH before process start).\n",
                     VK_ARM_TENSORS_EXTENSION_NAME, VK_ARM_DATA_GRAPH_EXTENSION_NAME);
        return false;
    }

    VkPhysicalDeviceTensorFeaturesARM tensorFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM};
    VkPhysicalDeviceDataGraphFeaturesARM graphFeatures = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM};
    tensorFeatures.pNext = &graphFeatures;
    VkPhysicalDeviceFeatures2 features2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features2.pNext = &tensorFeatures;
    vkGetPhysicalDeviceFeatures2(env.physicalDevice, &features2);
    if (!tensorFeatures.tensors || !graphFeatures.dataGraph) {
        std::fprintf(stderr, "NSS: ML features not supported (tensors=%d, dataGraph=%d)\n",
                     static_cast<int>(tensorFeatures.tensors), static_cast<int>(graphFeatures.dataGraph));
        return false;
    }

    HMODULE probe = loadNgsdk();
    if (!probe) {
        std::fprintf(stderr,
                     "NSS: ngsdk_windows_x64.dll not found (PATH or baked SR_NSS_RUNTIME_DIR)\n");
        return false;
    }
    FreeLibrary(probe);
    return true;
}

bool NssUpscaler::init(const VulkanEnv& env, const UpscalerDesc& desc) {
    shutdown();

    // Only exact 2x upscaling is supported by the NSS model.
    if (desc.renderWidth * 2 != desc.displayWidth || desc.renderHeight * 2 != desc.displayHeight) {
        std::fprintf(stderr,
                     "NSS: only 2x upscaling is supported (render %ux%u -> display %ux%u). "
                     "Use --render-scale 0.5.\n",
                     desc.renderWidth, desc.renderHeight, desc.displayWidth, desc.displayHeight);
        return false;
    }
    if (!desc.hdr) {
        std::fprintf(stderr, "NSS: HDR input is required by the SDK.\n");
        return false;
    }

    impl_       = new Impl();
    impl_->env  = env;
    impl_->desc = desc;

    // --- internal images ----------------------------------------------------
    if (!createImage2D(env, desc.renderWidth, desc.renderHeight, kNssColorFormat,
                       VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, impl_->nssColor)) {
        shutdown();
        return false;
    }
    if (!createImage2D(env, desc.displayWidth, desc.displayHeight, kNssColorFormat,
                       VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, impl_->nssOutput)) {
        shutdown();
        return false;
    }
    impl_->memoryBytes += impl_->nssColor.allocationSize + impl_->nssOutput.allocationSize;

    // --- conversion pipelines ------------------------------------------------
    {
        VkSamplerCreateInfo sci = {};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_NEAREST;
        sci.minFilter    = VK_FILTER_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(env.device, &sci, nullptr, &impl_->sampler) != VK_SUCCESS) {
            shutdown();
            return false;
        }

        VkDescriptorSetLayoutBinding bindings[2] = {};
        bindings[0].binding         = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].binding         = 1;
        bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo lci = {};
        lci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 2;
        lci.pBindings    = bindings;
        if (vkCreateDescriptorSetLayout(env.device, &lci, nullptr, &impl_->convertSetLayout) != VK_SUCCESS) {
            shutdown();
            return false;
        }

        VkDescriptorPoolSize poolSizes[2] = {};
        poolSizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[0].descriptorCount = 4;  // 2 passes * 2 slots
        poolSizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        poolSizes[1].descriptorCount = 4;
        VkDescriptorPoolCreateInfo pci = {};
        pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets       = 4;
        pci.poolSizeCount = 2;
        pci.pPoolSizes    = poolSizes;
        if (vkCreateDescriptorPool(env.device, &pci, nullptr, &impl_->descriptorPool) != VK_SUCCESS) {
            shutdown();
            return false;
        }

        VkDescriptorSetAllocateInfo sai = {};
        sai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        sai.descriptorPool     = impl_->descriptorPool;
        sai.descriptorSetCount = 2;
        VkDescriptorSetLayout layouts[2] = {impl_->convertSetLayout, impl_->convertSetLayout};
        sai.pSetLayouts = layouts;
        if (vkAllocateDescriptorSets(env.device, &sai, impl_->convertInSet) != VK_SUCCESS ||
            vkAllocateDescriptorSets(env.device, &sai, impl_->convertOutSet) != VK_SUCCESS) {
            shutdown();
            return false;
        }

        VkPipelineLayoutCreateInfo plci = {};
        plci.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts    = &impl_->convertSetLayout;
        if (vkCreatePipelineLayout(env.device, &plci, nullptr, &impl_->convertLayout) != VK_SUCCESS) {
            shutdown();
            return false;
        }

        struct {
            std::string path;
            VkPipeline* out;
        } passes[2] = {
            {sr::resolveShaderPath(SR_SHADER_DIR, "nss_convert_in.comp.spv"), &impl_->convertInPipeline},
            {sr::resolveShaderPath(SR_SHADER_DIR, "nss_convert_out.comp.spv"), &impl_->convertOutPipeline},
        };
        for (auto& pass : passes) {
            VkShaderModule module = loadShader(env.device, pass.path.c_str());
            if (!module) {
                shutdown();
                return false;
            }
            VkComputePipelineCreateInfo cpci = {};
            cpci.sType              = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpci.stage.sType        = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cpci.stage.stage        = VK_SHADER_STAGE_COMPUTE_BIT;
            cpci.stage.module       = module;
            cpci.stage.pName        = "main";
            cpci.layout             = impl_->convertLayout;
            const VkResult res = createComputePipeline(env, cpci, *pass.out);
            vkDestroyShaderModule(env.device, module, nullptr);
            if (res != VK_SUCCESS) {
                shutdown();
                return false;
            }
        }
    }

    // --- Neural Graphics SDK runtime -----------------------------------------
    impl_->sdkModule = loadNgsdk();
    if (!impl_->sdkModule) {
        std::fprintf(stderr, "NSS: failed to load ngsdk_windows_x64.dll\n");
        shutdown();
        return false;
    }
    impl_->ffxCreateCtx =
        reinterpret_cast<PFN_ffxCreateContextDyn>(GetProcAddress(impl_->sdkModule, "ffxCreateContext"));
    impl_->ffxDestroyCtx =
        reinterpret_cast<PFN_ffxDestroyContextDyn>(GetProcAddress(impl_->sdkModule, "ffxDestroyContext"));
    impl_->ffxDispatchFn =
        reinterpret_cast<PFN_ffxDispatchDyn>(GetProcAddress(impl_->sdkModule, "ffxDispatch"));
    if (!impl_->ffxCreateCtx || !impl_->ffxDestroyCtx || !impl_->ffxDispatchFn) {
        std::fprintf(stderr, "NSS: ngsdk_windows_x64.dll is missing FFX entry points\n");
        shutdown();
        return false;
    }

    ffxCreateBackendVKDesc backendDesc = {};
    backendDesc.header.type          = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    backendDesc.header.pNext         = nullptr;
    backendDesc.vkDevice             = env.device;
    backendDesc.vkPhysicalDevice     = env.physicalDevice;
    backendDesc.vkInstance           = env.instance;
    backendDesc.vkDeviceProcAddr     = vkGetDeviceProcAddr;
    backendDesc.vkGetInstanceProcAddr = env.getInstanceProcAddr;

    ffxApiCreateContextDescNss nssDesc = {};
    nssDesc.header.type          = FFX_API_CREATE_CONTEXT_DESC_TYPE_NSS;
    nssDesc.header.pNext         = &backendDesc.header;
    nssDesc.maxRenderSize.width  = desc.renderWidth;
    nssDesc.maxRenderSize.height = desc.renderHeight;
    nssDesc.maxUpscaleSize.width  = desc.displayWidth;
    nssDesc.maxUpscaleSize.height = desc.displayHeight;
    nssDesc.flags = FFX_API_NSS_CONTEXT_FLAG_QUANTIZED |          // required (int8 graphs only)
                    FFX_API_NSS_CONTEXT_FLAG_HIGH_DYNAMIC_RANGE | // required
                    FFX_API_NSS_CONTEXT_FLAG_MANAGE_HISTORY;      // SDK ping-pongs outputTm1
    if (desc.infiniteFarPlane) nssDesc.flags |= FFX_API_NSS_CONTEXT_FLAG_DEPTH_INFINITE;
    if (desc.invertedDepth) nssDesc.flags |= FFX_API_NSS_CONTEXT_FLAG_DEPTH_INVERTED;
    nssDesc.fpMessage = &nssMessageCallback;

    // The SDK embeds the VGF models at build time; SR_NSS_MODEL only selects
    // the quality mode (the high model maps to QUALITY, mid_low to BALANCED).
    nssDesc.qualityMode = FFX_API_NSS_SHADER_QUALITY_MODE_QUALITY;
    char* modelEnv = nullptr;
    size_t modelEnvLen = 0;
    if (_dupenv_s(&modelEnv, &modelEnvLen, "SR_NSS_MODEL") == 0 && modelEnv != nullptr) {
        if (std::strstr(modelEnv, "mid_low") != nullptr)
            nssDesc.qualityMode = FFX_API_NSS_SHADER_QUALITY_MODE_BALANCED;
        std::fprintf(stderr, "NSS: SR_NSS_MODEL=%s (model embedded in SDK; selecting %s mode)\n",
                     modelEnv, nssDesc.qualityMode == FFX_API_NSS_SHADER_QUALITY_MODE_QUALITY ? "QUALITY"
                                                                                              : "BALANCED");
        free(modelEnv);
    }

    std::fprintf(stderr, "NSS: calling ffxCreateContext...\n");
    const ffxReturnCode_t rc = impl_->ffxCreateCtx(&impl_->context, &nssDesc.header, nullptr);
    if (rc != FFX_API_RETURN_OK || impl_->context == nullptr) {
        std::fprintf(stderr, "NSS: ffxCreateContext failed (rc=%u)\n", rc);
        impl_->context = nullptr;
        shutdown();
        return false;
    }
    std::fprintf(stderr, "NSS: context created (%ux%u -> %ux%u)\n", desc.renderWidth,
                 desc.renderHeight, desc.displayWidth, desc.displayHeight);
    return true;
}

void NssUpscaler::dispatch(VkCommandBuffer cmd, const UpscalerResources& res, const CameraParams& cam,
                           const FrameParams& frame) {
    if (!impl_ || !impl_->context) return;
    Impl* impl = impl_;
    const int slot = frame.frameIndex % 2;

    // 1) Convert the R16G16B16A16 render-res color into R11G11B10.
    transitionImage(cmd, impl->nssColor, VK_IMAGE_LAYOUT_GENERAL);
    {
        VkDescriptorImageInfo infos[2] = {};
        infos[0].sampler     = impl->sampler;
        infos[0].imageView   = res.colorView;
        infos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        infos[1].imageView   = impl->nssColor.view;
        infos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet writes[2] = {};
        for (int i = 0; i < 2; ++i) {
            writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet          = impl->convertInSet[slot];
            writes[i].dstBinding      = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType  = i == 0 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                               : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[i].pImageInfo      = &infos[i];
        }
        vkUpdateDescriptorSets(impl->env.device, 2, writes, 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl->convertInPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl->convertLayout, 0, 1,
                                &impl->convertInSet[slot], 0, nullptr);
        vkCmdDispatch(cmd, (impl->desc.renderWidth + 7) / 8, (impl->desc.renderHeight + 7) / 8, 1);
    }
    transitionImage(cmd, impl->nssColor, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // 2) Run NSS.  The SDK manages layout transitions for registered resources
    //    and history ping-pong (FFX_API_NSS_CONTEXT_FLAG_MANAGE_HISTORY).
    transitionImage(cmd, impl->nssOutput, VK_IMAGE_LAYOUT_GENERAL);
    {
        ffxApiDispatchDescNss desc = {};
        desc.header.type = FFX_API_DISPATCH_DESC_TYPE_NSS;
        desc.commandList = reinterpret_cast<void*>(cmd);
        desc.color = wrapImage(impl->nssColor.image, kNssColorFormat, impl->desc.renderWidth,
                               impl->desc.renderHeight, FFX_API_RESOURCE_USAGE_READ_ONLY,
                               FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        desc.depth = wrapImage(res.depth, VK_FORMAT_D32_SFLOAT, impl->desc.renderWidth,
                               impl->desc.renderHeight, FFX_API_RESOURCE_USAGE_DEPTHTARGET,
                               FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        desc.motionVectors = wrapImage(res.motion, VK_FORMAT_R16G16_SFLOAT, impl->desc.renderWidth,
                                       impl->desc.renderHeight, FFX_API_RESOURCE_USAGE_READ_ONLY,
                                       FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        desc.output = wrapImage(impl->nssOutput.image, kNssColorFormat, impl->desc.displayWidth,
                                impl->desc.displayHeight, FFX_API_RESOURCE_USAGE_UAV,
                                FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);

        // NSS jitterOffset convention is the NEGATIVE of the actual
        // framebuffer pixel displacement (the sampling offset that undoes
        // the jitter) — opposite sign to FSR2/XeSS.  Derivation: Arm's
        // reference sample applies jitter with projection[2][0/1] += ndc on
        // a clip.w = -z view space, which displaces content by -jitter
        // pixels, then reports +jitter (temporal_forward_subpass.cpp:171,
        // nss.cpp:701).  Our projection (m[8]/m[9] -= jitter*2/size, same
        // clip.w = -z convention) displaces content by +jitter pixels, so
        // the value reported here must be negated.  Empirical evidence: with
        // the un-negated value the converged output showed phase-doubled
        // edges and sat ~3.8 dB PSNR below FSR2 vs GT on a static camera.
        desc.jitterOffset.x = -cam.jitterX;
        desc.jitterOffset.y = -cam.jitterY;
        desc.renderSize.width    = impl->desc.renderWidth;
        desc.renderSize.height   = impl->desc.renderHeight;
        desc.upscaleSize.width   = impl->desc.displayWidth;
        desc.upscaleSize.height  = impl->desc.displayHeight;
        desc.cameraNear = cam.cameraNear;
        desc.cameraFar  = cam.cameraFar;
        desc.cameraFovAngleVertical = cam.fovY;
        desc.exposure = frame.preExposure != 0.f ? frame.preExposure : 1.f;  // must not be 0
        // Canonical motion is already backward and stored in framebuffer UV.
        // NSS consumes backward input-pixel offsets after this scale is applied.
        const MotionScale motionScale =
            nssMotionVectorScale(impl->desc.renderWidth, impl->desc.renderHeight);
        desc.motionVectorScale.x = motionScale.x;
        desc.motionVectorScale.y = motionScale.y;
        desc.frameTimeDelta = frame.deltaTime * 1000.f;
        if (desc.frameTimeDelta < 1.f) desc.frameTimeDelta = 1.f;
        desc.reset = frame.resetHistory;
        desc.flags = 0;

        const ffxReturnCode_t rc = impl->ffxDispatchFn(&impl->context, &desc.header);
        if (rc != FFX_API_RETURN_OK)
            std::fprintf(stderr, "NSS: ffxDispatch failed (rc=%u)\n", rc);
    }

    // 3) Convert the R11G11B10 upscaled result back into the renderer output.
    transitionImage(cmd, impl->nssOutput, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    {
        VkDescriptorImageInfo infos[2] = {};
        infos[0].sampler     = impl->sampler;
        infos[0].imageView   = impl->nssOutput.view;
        infos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        infos[1].imageView   = res.outputView;
        infos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet writes[2] = {};
        for (int i = 0; i < 2; ++i) {
            writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet          = impl->convertOutSet[slot];
            writes[i].dstBinding      = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType  = i == 0 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                               : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[i].pImageInfo      = &infos[i];
        }
        vkUpdateDescriptorSets(impl->env.device, 2, writes, 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl->convertOutPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl->convertLayout, 0, 1,
                                &impl->convertOutSet[slot], 0, nullptr);
        vkCmdDispatch(cmd, (impl->desc.displayWidth + 7) / 8, (impl->desc.displayHeight + 7) / 8, 1);
    }
}

void NssUpscaler::shutdown() {
    if (!impl_) return;
    Impl* impl = impl_;
    if (impl->context && impl->ffxDestroyCtx) {
        impl->ffxDestroyCtx(&impl->context, nullptr);
        impl->context = nullptr;
    }
    const VkDevice device = impl->env.device;
    if (device) {
        if (impl->convertInPipeline) vkDestroyPipeline(device, impl->convertInPipeline, nullptr);
        if (impl->convertOutPipeline) vkDestroyPipeline(device, impl->convertOutPipeline, nullptr);
        if (impl->convertLayout) vkDestroyPipelineLayout(device, impl->convertLayout, nullptr);
        if (impl->descriptorPool) vkDestroyDescriptorPool(device, impl->descriptorPool, nullptr);
        if (impl->convertSetLayout) vkDestroyDescriptorSetLayout(device, impl->convertSetLayout, nullptr);
        if (impl->sampler) vkDestroySampler(device, impl->sampler, nullptr);
        destroyImage(device, impl->nssColor);
        destroyImage(device, impl->nssOutput);
    }
    if (impl->sdkModule) {
        FreeLibrary(impl->sdkModule);
        impl->sdkModule = nullptr;
    }
    delete impl_;
    impl_ = nullptr;
}

uint64_t NssUpscaler::gpuMemoryBytes() const {
    if (!impl_) return 0;
    // Own images plus a rough estimate of SDK-internal allocations (upscaled
    // history ping-pong, tensors, LUTs, feedback); the FFX NSS API exposes no
    // memory-usage query.
    const uint64_t sdkEstimate =
        2ull * impl_->desc.displayWidth * impl_->desc.displayHeight * 4 +  // history ping-pong
        48ull * 1024 * 1024;                                               // tensors/LUTs/feedback
    return impl_->memoryBytes + sdkEstimate;
}

std::unique_ptr<IUpscaler> createNssUpscaler() {
    return std::make_unique<NssUpscaler>();
}

} // namespace sr

SR_REGISTER_UPSCALER("nss", &sr::createNssUpscaler);
SR_REGISTER_VULKAN_DEVICE_NEEDS(sr::g_nssNeeds);
