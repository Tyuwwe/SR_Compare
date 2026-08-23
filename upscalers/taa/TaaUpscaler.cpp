#include "renderer/core/Vk.h"
#include "upscalers/InputAdapter.h"
#include "upscalers/UpscalerFactory.h"
#include "renderer/core/PathUtil.h"
#include "upscalers/taa/TaaUpscaler.h"
#include "upscalers/VkHelpers.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace sr {

namespace {

constexpr VkFormat kHistoryFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
// History depth ping-pong mirrors history color.  R32 (not R16) because the
// hyperbolic depth range bunches scene content near 1.0 and the disocclusion
// threshold in taa.comp needs better than half-float precision there.
constexpr VkFormat kHistoryDepthFormat = VK_FORMAT_R32_SFLOAT;
// Fixed RCAS sharpness: 0 is the CAS default (lowest ringing), 1 maximum.
// 0.2 recovers the high frequencies TAA accumulation smooths out without
// visible overshoot; baked into the constants buffer at init.
constexpr float kCasSharpness = 0.2f;

struct HistoryImage {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize allocationSize = 0;
};

struct TaaPush {
    float renderSize[2];
    float displaySize[2];
    float alpha;
    float reset;
    float useReactive;
    float pad0;
    float jitter[2]; // current-frame sub-pixel jitter (render pixels)
    float pad1[2];
};
static_assert(sizeof(TaaPush) == 48, "TaaPush size mismatch");

// CAS constants, mirrors ffxCasSetup() (cas/ffx_cas.h) with input == output
// resolution (sharpen-only): the scaling terms collapse to 1.0 / 0.0.
// const1[1] feeds only the 16-bit packed path (FFX_HALF = 0), so it stays 0.
struct CasConstants {
    uint32_t const0[4];
    uint32_t const1[4];
};

CasConstants makeCasConstants(float sharpness) {
    const auto asUint = [](float f) {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        return u;
    };
    const float s = sharpness < 0.f ? 0.f : (sharpness > 1.f ? 1.f : sharpness);
    const float sharp = -1.f / (8.f - 3.f * s); // -1 / lerp(8, 5, sharpness)
    CasConstants c = {};
    c.const0[0] = asUint(1.f);
    c.const0[1] = asUint(1.f);
    c.const0[2] = asUint(0.f);
    c.const0[3] = asUint(0.f);
    c.const1[0] = asUint(sharp);
    c.const1[1] = 0;
    c.const1[2] = asUint(8.f);
    c.const1[3] = 0;
    return c;
}

bool createHistoryImage(const VulkanEnv& env, uint32_t width, uint32_t height, VkFormat format,
                        HistoryImage& out) {
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.extent = {width, height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.format = format;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
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
    out.allocationSize = req.size;

    VkImageViewCreateInfo vi = {};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = out.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    if (vkCreateImageView(env.device, &vi, nullptr, &out.view) != VK_SUCCESS) {
        vkDestroyImage(env.device, out.image, nullptr);
        vkFreeMemory(env.device, out.memory, nullptr);
        out.image = VK_NULL_HANDLE;
        out.memory = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

VkSampler createSamplerLocal(VkDevice device, VkFilter filter) {
    VkSamplerCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    ci.magFilter = filter;
    ci.minFilter = filter;
    ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ci.maxLod = 0.f;
    VkSampler sampler = VK_NULL_HANDLE;
    vkCreateSampler(device, &ci, nullptr, &sampler);
    return sampler;
}

void historyBarrier(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkAccessFlags src = 0, dst = 0;
    if (oldLayout == VK_IMAGE_LAYOUT_GENERAL) src = VK_ACCESS_SHADER_WRITE_BIT;
    else if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) src = VK_ACCESS_SHADER_READ_BIT;
    if (newLayout == VK_IMAGE_LAYOUT_GENERAL) dst = VK_ACCESS_SHADER_WRITE_BIT;
    else if (newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) dst = VK_ACCESS_SHADER_READ_BIT;

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = src;
    barrier.dstAccessMask = dst;
    const VkPipelineStageFlags srcStage = src == 0 ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                                                   : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    vkCmdPipelineBarrier(cmd, srcStage, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                         &barrier);
}

void destroyImage(const VulkanEnv& env, HistoryImage& h) {
    if (h.view) { vkDestroyImageView(env.device, h.view, nullptr); h.view = VK_NULL_HANDLE; }
    if (h.image) { vkDestroyImage(env.device, h.image, nullptr); h.image = VK_NULL_HANDLE; }
    if (h.memory) { vkFreeMemory(env.device, h.memory, nullptr); h.memory = VK_NULL_HANDLE; }
}

} // namespace

struct TaaUpscaler::Impl {
    VulkanEnv env;
    UpscalerDesc desc;
    VkSampler sampler = VK_NULL_HANDLE;        // linear clamp (color/history)
    VkSampler samplerNearest = VK_NULL_HANDLE; // point clamp (history depth)
    HistoryImage history[2];
    HistoryImage historyDepth[2];
    HistoryImage resolved; // resolve output; CAS reads it, writes res.output
    int read = 0;
    int write = 1;
    VkImageLayout historyLayout[2] = {VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED};
    VkImageLayout historyDepthLayout[2] = {VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED};
    VkImageLayout resolvedLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout casSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSet casSet[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout casPipelineLayout = VK_NULL_HANDLE;
    VkPipeline casPipeline = VK_NULL_HANDLE;
    VkBuffer casConstants = VK_NULL_HANDLE;
    VkDeviceMemory casConstantsMemory = VK_NULL_HANDLE;
    uint64_t memoryBytes = 0;
};

TaaUpscaler::~TaaUpscaler() {
    shutdown();
}

const char* TaaUpscaler::name() const {
    return "TAA";
}

uint32_t TaaUpscaler::capabilities() const {
    return Cap_Temporal;
}

bool TaaUpscaler::isAvailable(const VulkanEnv& env) {
    return env.device != VK_NULL_HANDLE;
}

bool TaaUpscaler::init(const VulkanEnv& env, const UpscalerDesc& desc) {
    shutdown();
    impl_ = new Impl();
    impl_->env = env;
    impl_->desc = desc;

    impl_->sampler = createSamplerLocal(env.device, VK_FILTER_LINEAR);
    impl_->samplerNearest = createSamplerLocal(env.device, VK_FILTER_NEAREST);
    if (!impl_->sampler || !impl_->samplerNearest) { shutdown(); return false; }

    for (int i = 0; i < 2; ++i) {
        if (!createHistoryImage(env, desc.displayWidth, desc.displayHeight, kHistoryFormat,
                                impl_->history[i]) ||
            !createHistoryImage(env, desc.displayWidth, desc.displayHeight, kHistoryDepthFormat,
                                impl_->historyDepth[i])) {
            shutdown();
            return false;
        }
        impl_->memoryBytes += impl_->history[i].allocationSize + impl_->historyDepth[i].allocationSize;
    }
    if (!createHistoryImage(env, desc.displayWidth, desc.displayHeight, kHistoryFormat,
                            impl_->resolved)) {
        shutdown();
        return false;
    }
    impl_->memoryBytes += impl_->resolved.allocationSize;

    // CAS constants: written once (sizes and sharpness are fixed at init).
    {
        VkBufferCreateInfo bci = {};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = sizeof(CasConstants);
        bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(env.device, &bci, nullptr, &impl_->casConstants) != VK_SUCCESS) {
            shutdown();
            return false;
        }
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(env.device, impl_->casConstants, &req);
        const uint32_t type = findMemoryType(
            env.physicalDevice, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (type == 0xFFFFFFFFu) { shutdown(); return false; }
        VkMemoryAllocateInfo ai = {};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = type;
        if (vkAllocateMemory(env.device, &ai, nullptr, &impl_->casConstantsMemory) != VK_SUCCESS) {
            shutdown();
            return false;
        }
        vkBindBufferMemory(env.device, impl_->casConstants, impl_->casConstantsMemory, 0);
        impl_->memoryBytes += req.size;
        const CasConstants constants = makeCasConstants(kCasSharpness);
        void* mapped = nullptr;
        if (vkMapMemory(env.device, impl_->casConstantsMemory, 0, sizeof(constants), 0, &mapped) !=
            VK_SUCCESS) {
            shutdown();
            return false;
        }
        std::memcpy(mapped, &constants, sizeof(constants));
        vkUnmapMemory(env.device, impl_->casConstantsMemory);
    }

    // Resolve set: 0-3,6,7 combined image samplers; 4,5,8 storage images.
    {
        VkDescriptorSetLayoutBinding bindings[9] = {};
        const uint32_t samplerBindings[] = {0, 1, 2, 3, 6, 7};
        for (uint32_t b : samplerBindings) {
            bindings[b].binding = b;
            bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[b].descriptorCount = 1;
            bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        const uint32_t storageBindings[] = {4, 5, 8};
        for (uint32_t b : storageBindings) {
            bindings[b].binding = b;
            bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[b].descriptorCount = 1;
            bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo layoutCi = {};
        layoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutCi.bindingCount = 9;
        layoutCi.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(env.device, &layoutCi, nullptr, &impl_->setLayout) !=
            VK_SUCCESS) {
            shutdown();
            return false;
        }
    }

    // CAS set: 0 sampled image, 1 storage image, 2 constants uniform buffer.
    {
        VkDescriptorSetLayoutBinding bindings[3] = {};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo layoutCi = {};
        layoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutCi.bindingCount = 3;
        layoutCi.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(env.device, &layoutCi, nullptr, &impl_->casSetLayout) !=
            VK_SUCCESS) {
            shutdown();
            return false;
        }
    }

    VkDescriptorPoolSize poolSizes[4] = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = 12; // 2 sets * 6 combined-image-sampler bindings
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = 8; // 2 resolve sets * 3 + 2 CAS sets * 1
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    poolSizes[2].descriptorCount = 2; // 2 CAS sets
    poolSizes[3].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[3].descriptorCount = 2; // 2 CAS sets
    VkDescriptorPoolCreateInfo poolCi = {};
    poolCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCi.maxSets = 4;
    poolCi.poolSizeCount = 4;
    poolCi.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(env.device, &poolCi, nullptr, &impl_->pool) != VK_SUCCESS) {
        shutdown();
        return false;
    }

    VkDescriptorSetAllocateInfo setAlloc = {};
    setAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setAlloc.descriptorPool = impl_->pool;
    setAlloc.descriptorSetCount = 2;
    VkDescriptorSetLayout setLayouts[2] = {impl_->setLayout, impl_->setLayout};
    setAlloc.pSetLayouts = setLayouts;
    if (vkAllocateDescriptorSets(env.device, &setAlloc, impl_->set) != VK_SUCCESS) {
        shutdown();
        return false;
    }
    VkDescriptorSetLayout casSetLayouts[2] = {impl_->casSetLayout, impl_->casSetLayout};
    setAlloc.pSetLayouts = casSetLayouts;
    if (vkAllocateDescriptorSets(env.device, &setAlloc, impl_->casSet) != VK_SUCCESS) {
        shutdown();
        return false;
    }

    VkShaderModule module = loadShader(env.device, sr::resolveShaderPath(SR_SHADER_DIR, "taa.comp.spv").c_str());
    if (!module) { shutdown(); return false; }

    VkPushConstantRange pushRange = {};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(TaaPush);

    VkPipelineLayoutCreateInfo plCi = {};
    plCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plCi.setLayoutCount = 1;
    plCi.pSetLayouts = &impl_->setLayout;
    plCi.pushConstantRangeCount = 1;
    plCi.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(env.device, &plCi, nullptr, &impl_->pipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(env.device, module, nullptr);
        shutdown();
        return false;
    }

    VkComputePipelineCreateInfo pipeCi = {};
    pipeCi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeCi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeCi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeCi.stage.module = module;
    pipeCi.stage.pName = "main";
    pipeCi.layout = impl_->pipelineLayout;
    if (createComputePipeline(env, pipeCi, impl_->pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(env.device, module, nullptr);
        shutdown();
        return false;
    }
    vkDestroyShaderModule(env.device, module, nullptr);

    // CAS sharpen pipeline (no push constants).
    VkShaderModule casModule =
        loadShader(env.device, sr::resolveShaderPath(SR_SHADER_DIR, "taa_cas.comp.spv").c_str());
    if (!casModule) { shutdown(); return false; }
    VkPipelineLayoutCreateInfo casPlCi = {};
    casPlCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    casPlCi.setLayoutCount = 1;
    casPlCi.pSetLayouts = &impl_->casSetLayout;
    if (vkCreatePipelineLayout(env.device, &casPlCi, nullptr, &impl_->casPipelineLayout) !=
        VK_SUCCESS) {
        vkDestroyShaderModule(env.device, casModule, nullptr);
        shutdown();
        return false;
    }
    VkComputePipelineCreateInfo casPipeCi = {};
    casPipeCi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    casPipeCi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    casPipeCi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    casPipeCi.stage.module = casModule;
    casPipeCi.stage.pName = "main";
    casPipeCi.layout = impl_->casPipelineLayout;
    if (createComputePipeline(env, casPipeCi, impl_->casPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(env.device, casModule, nullptr);
        shutdown();
        return false;
    }
    vkDestroyShaderModule(env.device, casModule, nullptr);

    impl_->read = 0;
    impl_->write = 1;
    impl_->historyLayout[0] = VK_IMAGE_LAYOUT_UNDEFINED;
    impl_->historyLayout[1] = VK_IMAGE_LAYOUT_UNDEFINED;
    impl_->historyDepthLayout[0] = VK_IMAGE_LAYOUT_UNDEFINED;
    impl_->historyDepthLayout[1] = VK_IMAGE_LAYOUT_UNDEFINED;
    impl_->resolvedLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    return true;
}

void TaaUpscaler::dispatch(VkCommandBuffer cmd, const UpscalerResources& res, const CameraParams& cam,
                           const FrameParams& frame) {
    if (!impl_) return;

    const int read = impl_->read;
    const int write = impl_->write;
    // Per-frame-in-flight descriptor set: frame.frameIndex % 2 matches the
    // renderer's frame slot, so we never update a set while it is pending.
    const int slot = frame.frameIndex % 2;

    // Transition internal images: history color/depth ping-pong plus the
    // resolve intermediate (written here, sampled by the CAS pass below).
    historyBarrier(cmd, impl_->history[read].image, impl_->historyLayout[read],
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    historyBarrier(cmd, impl_->history[write].image, impl_->historyLayout[write],
                   VK_IMAGE_LAYOUT_GENERAL);
    historyBarrier(cmd, impl_->historyDepth[read].image, impl_->historyDepthLayout[read],
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    historyBarrier(cmd, impl_->historyDepth[write].image, impl_->historyDepthLayout[write],
                   VK_IMAGE_LAYOUT_GENERAL);
    historyBarrier(cmd, impl_->resolved.image, impl_->resolvedLayout, VK_IMAGE_LAYOUT_GENERAL);
    impl_->historyLayout[read] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    impl_->historyLayout[write] = VK_IMAGE_LAYOUT_GENERAL;
    impl_->historyDepthLayout[read] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    impl_->historyDepthLayout[write] = VK_IMAGE_LAYOUT_GENERAL;
    impl_->resolvedLayout = VK_IMAGE_LAYOUT_GENERAL;

    // Refresh descriptor bindings (input/output views may change on resize).
    VkDescriptorImageInfo infos[9] = {};
    infos[0].sampler = impl_->sampler;
    infos[0].imageView = res.colorView;
    infos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    infos[1].sampler = impl_->sampler;
    infos[1].imageView = res.motionView;
    infos[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    infos[2].sampler = impl_->sampler;
    infos[2].imageView = res.depthView;
    infos[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    infos[3].sampler = impl_->sampler;
    infos[3].imageView = impl_->history[read].view;
    infos[3].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    infos[4].imageView = impl_->resolved.view;
    infos[4].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    infos[5].imageView = impl_->history[write].view;
    infos[5].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    // Reactive mask; when absent, bind any valid view (color) — the shader
    // skips the sample via the useReactive push flag.
    infos[6].sampler = impl_->sampler;
    infos[6].imageView = res.reactiveView != VK_NULL_HANDLE ? res.reactiveView : res.colorView;
    infos[6].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    infos[7].sampler = impl_->samplerNearest;
    infos[7].imageView = impl_->historyDepth[read].view;
    infos[7].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    infos[8].imageView = impl_->historyDepth[write].view;
    infos[8].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[9] = {};
    for (int i = 0; i < 9; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = impl_->set[slot];
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = (i < 4 || i == 6 || i == 7) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                               : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i].pImageInfo = &infos[i];
    }
    vkUpdateDescriptorSets(impl_->env.device, 9, writes, 0, nullptr);

    TaaPush push;
    push.renderSize[0] = static_cast<float>(impl_->desc.renderWidth);
    push.renderSize[1] = static_cast<float>(impl_->desc.renderHeight);
    push.displaySize[0] = static_cast<float>(impl_->desc.displayWidth);
    push.displaySize[1] = static_cast<float>(impl_->desc.displayHeight);
    push.alpha = 0.9f;
    push.reset = frame.resetHistory ? 1.f : 0.f;
    push.useReactive = res.reactiveView != VK_NULL_HANDLE ? 1.f : 0.f;
    push.pad0 = 0.f;
    push.jitter[0] = cam.jitterX;
    push.jitter[1] = cam.jitterY;
    push.pad1[0] = push.pad1[1] = 0.f;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->pipelineLayout, 0, 1,
                            &impl_->set[slot], 0, nullptr);
    vkCmdPushConstants(cmd, impl_->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    const uint32_t gx = (impl_->desc.displayWidth + 7) / 8;
    const uint32_t gy = (impl_->desc.displayHeight + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);

    // Resolve -> CAS: the intermediate becomes the sharpen pass input.
    historyBarrier(cmd, impl_->resolved.image, VK_IMAGE_LAYOUT_GENERAL,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    impl_->resolvedLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo casInfo[2] = {};
    casInfo[0].imageView = impl_->resolved.view;
    casInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    casInfo[1].imageView = res.outputView;
    casInfo[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorBufferInfo casBufferInfo = {};
    casBufferInfo.buffer = impl_->casConstants;
    casBufferInfo.offset = 0;
    casBufferInfo.range = sizeof(CasConstants);

    VkWriteDescriptorSet casWrites[3] = {};
    casWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    casWrites[0].dstSet = impl_->casSet[slot];
    casWrites[0].dstBinding = 0;
    casWrites[0].descriptorCount = 1;
    casWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    casWrites[0].pImageInfo = &casInfo[0];
    casWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    casWrites[1].dstSet = impl_->casSet[slot];
    casWrites[1].dstBinding = 1;
    casWrites[1].descriptorCount = 1;
    casWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    casWrites[1].pImageInfo = &casInfo[1];
    casWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    casWrites[2].dstSet = impl_->casSet[slot];
    casWrites[2].dstBinding = 2;
    casWrites[2].descriptorCount = 1;
    casWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    casWrites[2].pBufferInfo = &casBufferInfo;
    vkUpdateDescriptorSets(impl_->env.device, 3, casWrites, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->casPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->casPipelineLayout, 0, 1,
                            &impl_->casSet[slot], 0, nullptr);
    // One 64-thread workgroup sharpens a 16x16 tile (see taa_cas.comp).
    vkCmdDispatch(cmd, (impl_->desc.displayWidth + 15) / 16,
                  (impl_->desc.displayHeight + 15) / 16, 1);

    impl_->read = write;
    impl_->write = read;
}

void TaaUpscaler::shutdown() {
    if (!impl_) return;
    const VkDevice device = impl_->env.device;
    if (device) {
        if (impl_->casPipeline) { vkDestroyPipeline(device, impl_->casPipeline, nullptr); impl_->casPipeline = VK_NULL_HANDLE; }
        if (impl_->casPipelineLayout) { vkDestroyPipelineLayout(device, impl_->casPipelineLayout, nullptr); impl_->casPipelineLayout = VK_NULL_HANDLE; }
        if (impl_->pipeline) { vkDestroyPipeline(device, impl_->pipeline, nullptr); impl_->pipeline = VK_NULL_HANDLE; }
        if (impl_->pipelineLayout) { vkDestroyPipelineLayout(device, impl_->pipelineLayout, nullptr); impl_->pipelineLayout = VK_NULL_HANDLE; }
        if (impl_->casConstants) { vkDestroyBuffer(device, impl_->casConstants, nullptr); impl_->casConstants = VK_NULL_HANDLE; }
        if (impl_->casConstantsMemory) { vkFreeMemory(device, impl_->casConstantsMemory, nullptr); impl_->casConstantsMemory = VK_NULL_HANDLE; }
        if (impl_->pool) { vkDestroyDescriptorPool(device, impl_->pool, nullptr); impl_->pool = VK_NULL_HANDLE; }
        if (impl_->casSetLayout) { vkDestroyDescriptorSetLayout(device, impl_->casSetLayout, nullptr); impl_->casSetLayout = VK_NULL_HANDLE; }
        if (impl_->setLayout) { vkDestroyDescriptorSetLayout(device, impl_->setLayout, nullptr); impl_->setLayout = VK_NULL_HANDLE; }
        if (impl_->samplerNearest) { vkDestroySampler(device, impl_->samplerNearest, nullptr); impl_->samplerNearest = VK_NULL_HANDLE; }
        if (impl_->sampler) { vkDestroySampler(device, impl_->sampler, nullptr); impl_->sampler = VK_NULL_HANDLE; }
        for (int i = 0; i < 2; ++i) {
            destroyImage(impl_->env, impl_->history[i]);
            destroyImage(impl_->env, impl_->historyDepth[i]);
        }
        destroyImage(impl_->env, impl_->resolved);
    }
    delete impl_;
    impl_ = nullptr;
}

uint64_t TaaUpscaler::gpuMemoryBytes() const {
    return impl_ ? impl_->memoryBytes : 0;
}

std::unique_ptr<IUpscaler> createTaaUpscaler() { return std::make_unique<TaaUpscaler>(); }

} // namespace sr

SR_REGISTER_UPSCALER("taa", &sr::createTaaUpscaler);
