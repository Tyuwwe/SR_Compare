#include "upscalers/UpscalerFactory.h"
#include "upscalers/sgsr/Sgsr1Upscaler.h"
#include "renderer/core/PathUtil.h"
#include "upscalers/sgsr/SgsrCommon.h"

#include <cstring>

namespace sr {

namespace {

constexpr VkFormat kHdrColorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr int kFrameSlots = 2; // matches the renderer's frames-in-flight

struct UboBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize allocationSize = 0;
};

bool createUbo(const VulkanEnv& env, VkDeviceSize size, UboBuffer& out) {
    VkBufferCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size = size;
    ci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(env.device, &ci, nullptr, &out.buffer) != VK_SUCCESS) return false;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(env.device, out.buffer, &req);
    const uint32_t type = sgsr::findMemoryType(
        env.physicalDevice, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type == 0xFFFFFFFFu) {
        vkDestroyBuffer(env.device, out.buffer, nullptr);
        out.buffer = VK_NULL_HANDLE;
        return false;
    }
    VkMemoryAllocateInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    if (vkAllocateMemory(env.device, &ai, nullptr, &out.memory) != VK_SUCCESS) {
        vkDestroyBuffer(env.device, out.buffer, nullptr);
        out.buffer = VK_NULL_HANDLE;
        return false;
    }
    vkBindBufferMemory(env.device, out.buffer, out.memory, 0);
    vkMapMemory(env.device, out.memory, 0, size, 0, &out.mapped);
    out.allocationSize = req.size;
    return true;
}

void destroyUbo(VkDevice device, UboBuffer& ubo) {
    if (ubo.buffer) { vkDestroyBuffer(device, ubo.buffer, nullptr); ubo.buffer = VK_NULL_HANDLE; }
    if (ubo.memory) { vkFreeMemory(device, ubo.memory, nullptr); ubo.memory = VK_NULL_HANDLE; }
    ubo.mapped = nullptr;
}

} // namespace

struct Sgsr1Upscaler::Impl {
    VulkanEnv env;
    UpscalerDesc desc;
    VkSampler sampler = VK_NULL_HANDLE;
    UboBuffer ubo[kFrameSlots];
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set[kFrameSlots] = {};
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    uint64_t memoryBytes = 0;
};

Sgsr1Upscaler::~Sgsr1Upscaler() {
    shutdown();
}

const char* Sgsr1Upscaler::name() const {
    return "SGSR1";
}

uint32_t Sgsr1Upscaler::capabilities() const {
    return Cap_Spatial;
}

bool Sgsr1Upscaler::isAvailable(const VulkanEnv& env) {
    return env.device != VK_NULL_HANDLE;
}

bool Sgsr1Upscaler::init(const VulkanEnv& env, const UpscalerDesc& desc) {
    shutdown();
    impl_ = new Impl();
    impl_->env = env;
    impl_->desc = desc;

    impl_->sampler = sgsr::createSampler(env.device, VK_FILTER_LINEAR);
    if (!impl_->sampler) { shutdown(); return false; }

    for (int i = 0; i < kFrameSlots; ++i) {
        if (!createUbo(env, 32 /* two vec4 ViewportInfo */, impl_->ubo[i])) {
            shutdown();
            return false;
        }
        impl_->memoryBytes += impl_->ubo[i].allocationSize;
    }

    // Descriptors: binding 0 = ViewportInfo UBO, binding 1 = input color sampler.
    VkDescriptorSetLayoutBinding bindings[2] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo layoutCi = {};
    layoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCi.bindingCount = 2;
    layoutCi.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(env.device, &layoutCi, nullptr, &impl_->setLayout) != VK_SUCCESS) {
        shutdown();
        return false;
    }

    VkDescriptorPoolSize poolSizes[2] = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = kFrameSlots;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = kFrameSlots;
    VkDescriptorPoolCreateInfo poolCi = {};
    poolCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCi.maxSets = kFrameSlots;
    poolCi.poolSizeCount = 2;
    poolCi.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(env.device, &poolCi, nullptr, &impl_->pool) != VK_SUCCESS) {
        shutdown();
        return false;
    }

    VkDescriptorSetLayout setLayouts[kFrameSlots] = {impl_->setLayout, impl_->setLayout};
    VkDescriptorSetAllocateInfo setAlloc = {};
    setAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setAlloc.descriptorPool = impl_->pool;
    setAlloc.descriptorSetCount = kFrameSlots;
    setAlloc.pSetLayouts = setLayouts;
    if (vkAllocateDescriptorSets(env.device, &setAlloc, impl_->set) != VK_SUCCESS) {
        shutdown();
        return false;
    }
    for (int i = 0; i < kFrameSlots; ++i) {
        VkDescriptorBufferInfo bi = {};
        bi.buffer = impl_->ubo[i].buffer;
        bi.offset = 0;
        bi.range = 32;
        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = impl_->set[i];
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.pBufferInfo = &bi;
        vkUpdateDescriptorSets(env.device, 1, &write, 0, nullptr);
    }

    VkShaderModule vert = sgsr::loadShaderModule(env, sr::resolveShaderPath(SR_SHADER_DIR, "sgsr1.vert.spv").c_str());
    VkShaderModule frag = sgsr::loadShaderModule(env, sr::resolveShaderPath(SR_SHADER_DIR, "sgsr1.frag.spv").c_str());
    if (!vert || !frag) {
        if (vert) vkDestroyShaderModule(env.device, vert, nullptr);
        if (frag) vkDestroyShaderModule(env.device, frag, nullptr);
        shutdown();
        return false;
    }

    VkPipelineLayoutCreateInfo plCi = {};
    plCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plCi.setLayoutCount = 1;
    plCi.pSetLayouts = &impl_->setLayout;
    if (vkCreatePipelineLayout(env.device, &plCi, nullptr, &impl_->pipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(env.device, vert, nullptr);
        vkDestroyShaderModule(env.device, frag, nullptr);
        shutdown();
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vi = {};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo ia = {};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp = {};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs = {};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.f;
    VkPipelineMultisampleStateCreateInfo ms = {};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds = {};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    VkPipelineColorBlendAttachmentState blendAtt = {};
    blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb = {};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &blendAtt;
    const VkDynamicState dynStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn = {};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = dynStates;

    VkPipelineRenderingCreateInfo rendering = {};
    rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &kHdrColorFormat;

    VkGraphicsPipelineCreateInfo pipeCi = {};
    pipeCi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeCi.pNext = &rendering;
    pipeCi.stageCount = 2;
    pipeCi.pStages = stages;
    pipeCi.pVertexInputState = &vi;
    pipeCi.pInputAssemblyState = &ia;
    pipeCi.pViewportState = &vp;
    pipeCi.pRasterizationState = &rs;
    pipeCi.pMultisampleState = &ms;
    pipeCi.pDepthStencilState = &ds;
    pipeCi.pColorBlendState = &cb;
    pipeCi.pDynamicState = &dyn;
    pipeCi.layout = impl_->pipelineLayout;
    const VkResult res = vkCreateGraphicsPipelines(env.device, VK_NULL_HANDLE, 1, &pipeCi, nullptr,
                                                   &impl_->pipeline);
    vkDestroyShaderModule(env.device, vert, nullptr);
    vkDestroyShaderModule(env.device, frag, nullptr);
    if (res != VK_SUCCESS) {
        shutdown();
        return false;
    }
    return true;
}

void Sgsr1Upscaler::dispatch(VkCommandBuffer cmd, const UpscalerResources& res,
                             const CameraParams& cam, const FrameParams& frame) {
    if (!impl_) return;
    const int slot = frame.frameIndex % kFrameSlots;

    // ViewportInfo[0]: xy = 1/renderSize, zw = renderSize.
    // ViewportInfo[1].xy: Halton jitter in pixels (undo a shared temporal GBuffer).
    float viewportInfo[8] = {1.f / static_cast<float>(impl_->desc.renderWidth),
                             1.f / static_cast<float>(impl_->desc.renderHeight),
                             static_cast<float>(impl_->desc.renderWidth),
                             static_cast<float>(impl_->desc.renderHeight),
                             cam.jitterX, cam.jitterY, 0.f, 0.f};
    std::memcpy(impl_->ubo[slot].mapped, viewportInfo, sizeof(viewportInfo));

    // Refresh the input binding (the view may change across resizes).
    VkDescriptorImageInfo ii = {};
    ii.sampler = impl_->sampler;
    ii.imageView = res.colorView;
    ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = impl_->set[slot];
    write.dstBinding = 1;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &ii;
    vkUpdateDescriptorSets(impl_->env.device, 1, &write, 0, nullptr);

    // Render directly into the output image (kept in GENERAL by the renderer;
    // GENERAL is a valid attachment layout for dynamic rendering).
    VkRenderingAttachmentInfo colorAtt = {};
    colorAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAtt.imageView = res.outputView;
    colorAtt.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri = {};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea = {{0, 0}, {impl_->desc.displayWidth, impl_->desc.displayHeight}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &colorAtt;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport viewport = {0.f, 0.f, static_cast<float>(impl_->desc.displayWidth),
                           static_cast<float>(impl_->desc.displayHeight), 0.f, 1.f};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor = {{0, 0}, {impl_->desc.displayWidth, impl_->desc.displayHeight}};
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, impl_->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, impl_->pipelineLayout, 0, 1,
                            &impl_->set[slot], 0, nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);

    // Make the attachment writes visible to the renderer's later SHADER_READ transition.
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void Sgsr1Upscaler::shutdown() {
    if (!impl_) return;
    const VkDevice device = impl_->env.device;
    if (device) {
        if (impl_->pipeline) { vkDestroyPipeline(device, impl_->pipeline, nullptr); impl_->pipeline = VK_NULL_HANDLE; }
        if (impl_->pipelineLayout) { vkDestroyPipelineLayout(device, impl_->pipelineLayout, nullptr); impl_->pipelineLayout = VK_NULL_HANDLE; }
        if (impl_->pool) { vkDestroyDescriptorPool(device, impl_->pool, nullptr); impl_->pool = VK_NULL_HANDLE; }
        if (impl_->setLayout) { vkDestroyDescriptorSetLayout(device, impl_->setLayout, nullptr); impl_->setLayout = VK_NULL_HANDLE; }
        if (impl_->sampler) { vkDestroySampler(device, impl_->sampler, nullptr); impl_->sampler = VK_NULL_HANDLE; }
        for (auto& u : impl_->ubo) destroyUbo(device, u);
    }
    delete impl_;
    impl_ = nullptr;
}

uint64_t Sgsr1Upscaler::gpuMemoryBytes() const {
    return impl_ ? impl_->memoryBytes : 0;
}

std::unique_ptr<IUpscaler> createSgsr1Upscaler() { return std::make_unique<Sgsr1Upscaler>(); }

} // namespace sr

SR_REGISTER_UPSCALER("sgsr1", &sr::createSgsr1Upscaler);
