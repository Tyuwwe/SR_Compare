#include "renderer/core/Vk.h"
#include "upscalers/InputAdapter.h"
#include "upscalers/UpscalerFactory.h"
#include "renderer/core/PathUtil.h"
#include "upscalers/taa/TaaUpscaler.h"
#include "upscalers/VkHelpers.h"

#include <cstdio>
#include <fstream>
#include <vector>

namespace sr {

namespace {

constexpr VkFormat kHistoryFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

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

bool createHistoryImage(const VulkanEnv& env, uint32_t width, uint32_t height, HistoryImage& out) {
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.extent = {width, height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.format = kHistoryFormat;
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
    vi.format = kHistoryFormat;
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

VkSampler createSamplerLocal(VkDevice device) {
    VkSamplerCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    ci.magFilter = VK_FILTER_LINEAR;
    ci.minFilter = VK_FILTER_LINEAR;
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

} // namespace

struct TaaUpscaler::Impl {
    VulkanEnv env;
    UpscalerDesc desc;
    VkSampler sampler = VK_NULL_HANDLE;
    HistoryImage history[2];
    int read = 0;
    int write = 1;
    VkImageLayout historyLayout[2] = {VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED};
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
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

    impl_->sampler = createSamplerLocal(env.device);
    if (!impl_->sampler) { shutdown(); return false; }

    for (int i = 0; i < 2; ++i) {
        if (!createHistoryImage(env, desc.displayWidth, desc.displayHeight, impl_->history[i])) {
            shutdown();
            return false;
        }
        impl_->memoryBytes += impl_->history[i].allocationSize;
    }

    VkDescriptorSetLayoutBinding bindings[7] = {};
    for (int i = 0; i < 4; ++i) {
        bindings[i].binding = static_cast<uint32_t>(i);
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    // binding 6: optional translucent coverage mask (reactive) sampler.
    bindings[6].binding = 6;
    bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[6].descriptorCount = 1;
    bindings[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    for (int i = 4; i < 6; ++i) {
        bindings[i].binding = static_cast<uint32_t>(i);
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutCi = {};
    layoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCi.bindingCount = 7;
    layoutCi.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(env.device, &layoutCi, nullptr, &impl_->setLayout) != VK_SUCCESS) {
        shutdown();
        return false;
    }

    VkDescriptorPoolSize poolSizes[2] = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = 10; // 2 sets * 5 combined-image-sampler bindings
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = 4; // 2 sets * 2 storage-image bindings
    VkDescriptorPoolCreateInfo poolCi = {};
    poolCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCi.maxSets = 2;
    poolCi.poolSizeCount = 2;
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
    if (vkCreateComputePipelines(env.device, VK_NULL_HANDLE, 1, &pipeCi, nullptr, &impl_->pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(env.device, module, nullptr);
        shutdown();
        return false;
    }
    vkDestroyShaderModule(env.device, module, nullptr);

    impl_->read = 0;
    impl_->write = 1;
    impl_->historyLayout[0] = VK_IMAGE_LAYOUT_UNDEFINED;
    impl_->historyLayout[1] = VK_IMAGE_LAYOUT_UNDEFINED;
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

    // Transition internal history images.
    historyBarrier(cmd, impl_->history[read].image, impl_->historyLayout[read],
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    historyBarrier(cmd, impl_->history[write].image, impl_->historyLayout[write],
                   VK_IMAGE_LAYOUT_GENERAL);
    impl_->historyLayout[read] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    impl_->historyLayout[write] = VK_IMAGE_LAYOUT_GENERAL;

    // Refresh descriptor bindings (input/output views may change on resize).
    VkDescriptorImageInfo infos[7] = {};
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
    infos[4].imageView = res.outputView;
    infos[4].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    infos[5].imageView = impl_->history[write].view;
    infos[5].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    // Reactive mask; when absent, bind any valid view (color) — the shader
    // skips the sample via the useReactive push flag.
    infos[6].sampler = impl_->sampler;
    infos[6].imageView = res.reactiveView != VK_NULL_HANDLE ? res.reactiveView : res.colorView;
    infos[6].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet writes[7] = {};
    for (int i = 0; i < 7; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = impl_->set[slot];
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = (i < 4 || i == 6) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                     : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i].pImageInfo = &infos[i];
    }
    vkUpdateDescriptorSets(impl_->env.device, 7, writes, 0, nullptr);

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

    impl_->read = write;
    impl_->write = read;
}

void TaaUpscaler::shutdown() {
    if (!impl_) return;
    const VkDevice device = impl_->env.device;
    if (device) {
        if (impl_->pipeline) { vkDestroyPipeline(device, impl_->pipeline, nullptr); impl_->pipeline = VK_NULL_HANDLE; }
        if (impl_->pipelineLayout) { vkDestroyPipelineLayout(device, impl_->pipelineLayout, nullptr); impl_->pipelineLayout = VK_NULL_HANDLE; }
        if (impl_->pool) { vkDestroyDescriptorPool(device, impl_->pool, nullptr); impl_->pool = VK_NULL_HANDLE; }
        if (impl_->setLayout) { vkDestroyDescriptorSetLayout(device, impl_->setLayout, nullptr); impl_->setLayout = VK_NULL_HANDLE; }
        if (impl_->sampler) { vkDestroySampler(device, impl_->sampler, nullptr); impl_->sampler = VK_NULL_HANDLE; }
        for (auto& h : impl_->history) {
            if (h.view) { vkDestroyImageView(device, h.view, nullptr); h.view = VK_NULL_HANDLE; }
            if (h.image) { vkDestroyImage(device, h.image, nullptr); h.image = VK_NULL_HANDLE; }
            if (h.memory) { vkFreeMemory(device, h.memory, nullptr); h.memory = VK_NULL_HANDLE; }
        }
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
