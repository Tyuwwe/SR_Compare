#include "renderer/Renderer.h"

#include "app/CliUtils.h"

#include "renderer/Screenshot.h"
#include "renderer/core/MemoryBudget.h"
#include "renderer/core/PathUtil.h"
#include "renderer/core/VkUtil.h"
#include "renderer/core/Vma.h"
#include "renderer/scene/SceneRegistry.h"
#include "upscalers/UpscalerFactory.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace sr {

namespace {

VkRenderingAttachmentInfo makeColorAttachment(VkImageView view, VkImageLayout layout,
                                              VkAttachmentLoadOp loadOp,
                                              float r = 0.f, float g = 0.f, float b = 0.f) {
    VkRenderingAttachmentInfo a = {};
    a.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    a.imageView = view;
    a.imageLayout = layout;
    a.loadOp = loadOp;
    a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    if (loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
        a.clearValue.color = {{r, g, b, 1.f}};
    }
    return a;
}

VkRenderingAttachmentInfo makeDepthAttachment(VkImageView view, VkImageLayout layout,
                                              VkAttachmentLoadOp loadOp) {
    VkRenderingAttachmentInfo a = {};
    a.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    a.imageView = view;
    a.imageLayout = layout;
    a.loadOp = loadOp;
    a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    if (loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
        a.clearValue.depthStencil = {1.f, 0};
    }
    return a;
}

void beginRendering(VkCommandBuffer cmd, uint32_t width, uint32_t height, uint32_t colorCount,
                    const VkRenderingAttachmentInfo* colors, const VkRenderingAttachmentInfo* depth) {
    VkRenderingInfo ri = {};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea = {{0, 0}, {width, height}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = colorCount;
    ri.pColorAttachments = colors;
    ri.pDepthAttachment = depth;
    vkCmdBeginRendering(cmd, &ri);
}

} // namespace

void Renderer::ImageResource::destroy(const VulkanContext& ctx) {
    if (view) { vkDestroyImageView(ctx.device, view, nullptr); view = VK_NULL_HANDLE; }
    if (image) { vmaDestroyImage(ctx.allocator, image, memory); image = VK_NULL_HANDLE; memory = VK_NULL_HANDLE; }
}

bool Renderer::init(const RendererOptions& opts) {
    opts_ = opts;
    diagNoJitter_ = sr::envFlag("SR_NO_JITTER");

    if (!window_.create("sr_compare", static_cast<int>(opts.displayWidth),
                        static_cast<int>(opts.displayHeight)))
        return false;
    if (!ctx_.create(window_)) return false;
    if (!swapchain_.create(ctx_, opts.displayWidth, opts.displayHeight, opts.vsync)) return false;

    renderWidth_ = std::max(1u, static_cast<uint32_t>(static_cast<float>(opts.displayWidth) * opts.renderScale));
    renderHeight_ = std::max(1u, static_cast<uint32_t>(static_cast<float>(opts.displayHeight) * opts.renderScale));

    bool sceneOk = false;
    if (!opts.scenePath.empty()) sceneOk = scene_.loadGltf(ctx_, opts.scenePath.c_str());
    if (!sceneOk) sceneOk = scene_.loadProcedural(ctx_);
    if (!sceneOk) return false;
    hasTransparency_ = deferred_.sceneHasTransparency(scene_);
    iblIntensity_ = lightingPresetForScene(opts.scenePath).iblIntensity;

    if (scene_.materials.empty()) {
        Material fallback;
        fallback.baseColor = {0.8f, 0.8f, 0.8f, 1.f};
        fallback.roughness = 0.6f;
        scene_.materials.push_back(fallback);
    }
    if (scene_.textures.empty()) {
        const uint8_t white[4] = {255, 255, 255, 255};
        Texture dummy;
        if (!scene_.uploadTexture(ctx_, 1, 1, white, dummy)) return false;
        scene_.textures.push_back(dummy);
    }

    if (opts.upscalerName != "none") {
        upscaler_ = createUpscaler(opts.upscalerName.c_str());
        if (upscaler_) {
            UpscalerDesc desc;
            desc.renderWidth = renderWidth_;
            desc.renderHeight = renderHeight_;
            desc.displayWidth = opts.displayWidth;
            desc.displayHeight = opts.displayHeight;
            desc.hdr = true;
            desc.invertedDepth = false;
            desc.infiniteFarPlane = true;
            if (!upscaler_->init(ctx_.toEnv(), desc)) upscaler_.reset();
        }
        if (!upscaler_) {
            std::fprintf(stderr, "warning: upscaler '%s' unavailable, falling back to native GT\n",
                         opts.upscalerName.c_str());
        }
    }

    if (!createRenderTargets()) return false;
    // Shared deferred pipeline: IBL maps + shaders + layouts + pipelines.
    if (!deferred_.init(ctx_, opts_.envMapPath.c_str())) return false;
    // CSM shadow targets (fixed size; a failure degrades to no shadows).
    if (opts_.shadows) {
        shadowsActive_ = deferred_.createShadowTargets(ctx_, shadow_);
        if (!shadowsActive_)
            std::fprintf(stderr, "warning: shadow target creation failed, shadows disabled\n");
    }
    if (!createShaders()) return false;
    if (!createSceneDescriptors()) return false;
    if (!createPipelines()) return false;
    if (!createSyncResources()) return false;
    if (!createScreenshotStaging()) return false;

    if (!opts.cameraPath.empty()) {
        if (!loadCameraPath(opts.cameraPath.c_str(), path_)) {
            std::fprintf(stderr, "warning: failed to load camera path %s\n", opts.cameraPath.c_str());
            path_.clear();
        }
    } else if (opts.frames >= 0) {
        // Orbit stays inside the procedural room (walls at |x| = 10, z = -10).
        path_ = generateOrbitPath(opts.frames, {0.f, 2.f, 0.f}, 6.5f, 2.f, 5.5f, 1.f);
    }

    if (path_.empty()) {
        // Interactive free-fly start: per-scene pose when registered (the
        // generic default sits inside Bistro's outer wall), else the Camera
        // constructor default.
        Vec3 pos, fwd;
        if (initialCameraPose(opts_.scenePath, pos, fwd))
            camera_.setPose(pos, fwd, {0.f, 1.f, 0.f});
    }

    return true;
}

bool Renderer::createRenderTargets() {
    const uint32_t dw = opts_.displayWidth;
    const uint32_t dh = opts_.displayHeight;

    auto createRT = [&](ImageResource& rt, uint32_t w, uint32_t h, VkFormat format,
                        VkImageUsageFlags usage, VkImageAspectFlags aspect) {
        rt.width = w;
        rt.height = h;
        rt.format = format;
        if (createImage(ctx_, w, h, format, usage, rt.image, rt.memory) != VK_SUCCESS) return false;
        rt.view = createImageView(ctx_, rt.image, format, aspect);
        return rt.view != VK_NULL_HANDLE;
    };

    // TRANSFER_SRC/DST: some upscalers (DLSS via Streamline) copy the input
    // color/depth/motion into internal buffers instead of just sampling them.
    if (!createRT(gbColor_, renderWidth_, renderHeight_, deferred::kHdrColorFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gbMotion_, renderWidth_, renderHeight_, deferred::kMotionFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gbDepth_, renderWidth_, renderHeight_, deferred::kDepthFormat,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_DEPTH_BIT))
        return false;
    // Translucent coverage mask (reactive/TC mask for upscalers).
    if (!createRT(gbReactive_, renderWidth_, renderHeight_, deferred::kReactiveFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    // Deferred GBuffer attachments (low-res input path / full-res GT path).
    const VkImageUsageFlags gbUsage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (!createRT(gbAlbedo_, renderWidth_, renderHeight_, deferred::kAlbedoFormat, gbUsage,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gbNormal_, renderWidth_, renderHeight_, deferred::kNormalFormat, gbUsage,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gbMaterial_, renderWidth_, renderHeight_, deferred::kMaterialFormat, gbUsage,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gbEmissive_, renderWidth_, renderHeight_, deferred::kEmissiveFormat, gbUsage,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtAlbedo_, dw, dh, deferred::kAlbedoFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtNormal_, dw, dh, deferred::kNormalFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtMaterial_, dw, dh, deferred::kMaterialFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtEmissive_, dw, dh, deferred::kEmissiveFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtDepth_, dw, dh, deferred::kDepthFormat,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_DEPTH_BIT))
        return false;
    // GTAO targets (working RG16F + filtered R16F) for both GBuffer paths.
    const VkImageUsageFlags aoUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (!createRT(gbAoRaw_, renderWidth_, renderHeight_, kAoRawFormat, aoUsage,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gbAo_, renderWidth_, renderHeight_, VK_FORMAT_R16_SFLOAT, aoUsage,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtAoRaw_, dw, dh, kAoRawFormat, aoUsage, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtAo_, dw, dh, VK_FORMAT_R16_SFLOAT, aoUsage, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    const VkImageUsageFlags bloomUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    const uint32_t gbHalfW = std::max(1u, renderWidth_ / 2);
    const uint32_t gbHalfH = std::max(1u, renderHeight_ / 2);
    const uint32_t gtHalfW = std::max(1u, dw / 2);
    const uint32_t gtHalfH = std::max(1u, dh / 2);
    if (!createRT(gbBloomA_, gbHalfW, gbHalfH, deferred::kHdrColorFormat, bloomUsage,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gbBloomB_, gbHalfW, gbHalfH, deferred::kHdrColorFormat, bloomUsage,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtBloomA_, gtHalfW, gtHalfH, deferred::kHdrColorFormat, bloomUsage,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtBloomB_, gtHalfW, gtHalfH, deferred::kHdrColorFormat, bloomUsage,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    const VkImageUsageFlags ssrUsage =
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (!createRT(gbSsrSrc_, renderWidth_, renderHeight_, deferred::kHdrColorFormat, ssrUsage,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtSsrSrc_, dw, dh, deferred::kHdrColorFormat, ssrUsage, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!deferred_.createDepthPyramid(ctx_, renderWidth_, renderHeight_, gbPyramid_))
        return false;
    if (!deferred_.createDepthPyramid(ctx_, dw, dh, gtPyramid_))
        return false;
    if (!createRT(finalImage_, dw, dh, deferred::kHdrColorFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    return true;
}

bool Renderer::loadShader(const char* name, VkShaderModule& out) {
    const std::string path = sr::resolveShaderPath(SR_SHADER_DIR, name);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "failed to open shader: %s\n", path.c_str());
        return false;
    }
    const std::streamoff size = file.tellg();
    if (size <= 0 || size % 4 != 0) return false;
    file.seekg(0);
    std::vector<uint32_t> code(static_cast<size_t>(size) / 4);
    file.read(reinterpret_cast<char*>(code.data()), size);

    VkShaderModuleCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = static_cast<size_t>(size);
    ci.pCode = code.data();
    return vkCreateShaderModule(ctx_.device, &ci, nullptr, &out) == VK_SUCCESS;
}

bool Renderer::createShaders() {
    // Deferred shaders live in DeferredCore; the viewer only needs present.frag.
    return loadShader("present.frag.spv", presentFrag_);
}

bool Renderer::createSceneDescriptors() {
    VkDescriptorSetLayoutBinding presentBinding = {};
    presentBinding.binding = 0;
    presentBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    presentBinding.descriptorCount = 1;
    presentBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo presentLayoutCi = {};
    presentLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    presentLayoutCi.bindingCount = 1;
    presentLayoutCi.pBindings = &presentBinding;
    if (vkCreateDescriptorSetLayout(ctx_.device, &presentLayoutCi, nullptr, &presentSetLayout_) != VK_SUCCESS)
        return false;

    VkDescriptorPoolSize sizes[4] = {};
    // Hi-Z downsample sets: one per mip per pyramid (sampler + storage image).
    const uint32_t hizSets = gbPyramid_.mipCount + gtPyramid_.mipCount;
    sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = deferred::kMaxTextures + 1 + // texture array + present
                               11 * kFramesInFlight * 2 + // lighting sets (GB/GT), +1 shadow map
                               7 * kFramesInFlight * 2 +  // transparent sets (GB/GT), +SSR+shadow
                               2 * 2 + 1 * 2 +            // ssao + blur samplers
                               8 +                        // bloom extract/blur/comp (GB/GT)
                               hizSets;
    sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[1].descriptorCount = kFramesInFlight * 4; // scene + lighting + 2 transparent UBOs
    sizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    sizes[2].descriptorCount = kFramesInFlight;
    sizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    sizes[3].descriptorCount = 4 + 8 + hizSets; // ssao + bloom (GB/GT) + Hi-Z mips
    VkDescriptorPoolCreateInfo poolCi = {};
    poolCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCi.maxSets = kFramesInFlight * 5 + 6 + 8 + hizSets; // + 8 bloom sets
    poolCi.poolSizeCount = 4;
    poolCi.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(ctx_.device, &poolCi, nullptr, &descriptorPool_) != VK_SUCCESS)
        return false;

    if (!deferred_.writeDepthPyramidSets(ctx_, descriptorPool_, gbDepth_.view, gbPyramid_))
        return false;
    if (!deferred_.writeDepthPyramidSets(ctx_, descriptorPool_, gtDepth_.view, gtPyramid_))
        return false;

    if (!deferred_.createMaterialUbo(ctx_, scene_, materialUbo_, materialUboMemory_,
                                     materialStride_))
        return false;

    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = descriptorPool_;
        alloc.descriptorSetCount = 1;
        const VkDescriptorSetLayout sceneLayout = deferred_.sceneSetLayout();
        alloc.pSetLayouts = &sceneLayout;
        if (vkAllocateDescriptorSets(ctx_.device, &alloc, &frames_[i].sceneSet) != VK_SUCCESS) return false;
    }

    {
        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = descriptorPool_;
        alloc.descriptorSetCount = 1;
        const VkDescriptorSetLayout texLayout = deferred_.textureSetLayout();
        alloc.pSetLayouts = &texLayout;
        if (vkAllocateDescriptorSets(ctx_.device, &alloc, &textureSet_) != VK_SUCCESS) return false;
        deferred_.writeTextureSet(ctx_, textureSet_, scene_);
    }

    {
        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = descriptorPool_;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &presentSetLayout_;
        if (vkAllocateDescriptorSets(ctx_.device, &alloc, &presentSet_) != VK_SUCCESS) return false;

        VkDescriptorImageInfo info = {};
        info.sampler = deferred_.textureSampler();
        info.imageView = finalImage_.view;
        info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = presentSet_;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &info;
        vkUpdateDescriptorSets(ctx_.device, 1, &write, 0, nullptr);
    }

    return true;
}

bool Renderer::createPipelines() {
    // The GBuffer/GT/lighting pipelines live in DeferredCore; only the
    // swapchain present pipeline (fullscreen triangle + tonemap) remains here.
    VkPushConstantRange presentPush = {};
    presentPush.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    presentPush.offset = 0;
    presentPush.size = 16; // vec4: exposure.x
    VkPipelineLayoutCreateInfo presentLayoutCi = {};
    presentLayoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    presentLayoutCi.setLayoutCount = 1;
    presentLayoutCi.pSetLayouts = &presentSetLayout_;
    presentLayoutCi.pushConstantRangeCount = 1;
    presentLayoutCi.pPushConstantRanges = &presentPush;
    if (vkCreatePipelineLayout(ctx_.device, &presentLayoutCi, nullptr, &presentPipelineLayout_) != VK_SUCCESS)
        return false;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.f;

    VkPipelineMultisampleStateCreateInfo multisample = {};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineVertexInputStateCreateInfo emptyVertexInput = {};
    emptyVertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineDepthStencilStateCreateInfo noDepth = {};
    noDepth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

    VkDynamicState dynamicStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState = {};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineShaderStageCreateInfo presentStages[2] = {};
    presentStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    presentStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    presentStages[0].module = deferred_.fullscreenVert();
    presentStages[0].pName = "main";
    presentStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    presentStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    presentStages[1].module = presentFrag_;
    presentStages[1].pName = "main";

    VkPipelineColorBlendAttachmentState presentBlend = {};
    presentBlend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo presentColorBlend = {};
    presentColorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    presentColorBlend.attachmentCount = 1;
    presentColorBlend.pAttachments = &presentBlend;

    VkGraphicsPipelineCreateInfo presentCi = {};
    presentCi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    presentCi.stageCount = 2;
    presentCi.pStages = presentStages;
    presentCi.pVertexInputState = &emptyVertexInput;
    presentCi.pInputAssemblyState = &inputAssembly;
    presentCi.pViewportState = &viewportState;
    presentCi.pRasterizationState = &rasterizer;
    presentCi.pMultisampleState = &multisample;
    presentCi.pDepthStencilState = &noDepth;
    presentCi.pColorBlendState = &presentColorBlend;
    presentCi.pDynamicState = &dynamicState;
    presentCi.layout = presentPipelineLayout_;

    VkPipelineRenderingCreateInfo presentRendering = {};
    presentRendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    presentRendering.colorAttachmentCount = 1;
    const VkFormat presentFormat = swapchain_.format();
    presentRendering.pColorAttachmentFormats = &presentFormat;
    presentCi.pNext = &presentRendering;
    if (createGraphicsPipeline(ctx_, presentCi, presentPipeline_) != VK_SUCCESS)
        return false;

    return true;
}

bool Renderer::createSyncResources() {
    VkCommandBufferAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = ctx_.framePool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = kFramesInFlight;
    VkCommandBuffer cmds[kFramesInFlight] = {};
    if (vkAllocateCommandBuffers(ctx_.device, &alloc, cmds) != VK_SUCCESS) return false;
    for (uint32_t i = 0; i < kFramesInFlight; ++i) frames_[i].cmd = cmds[i];

    VkSemaphoreCreateInfo semCi = {};
    semCi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fenceCi = {};
    fenceCi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCi.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    const VkDeviceSize uboSize = sizeof(SceneUBO);
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        if (vkCreateSemaphore(ctx_.device, &semCi, nullptr, &frames_[i].imageAvailable) != VK_SUCCESS) return false;
        if (vkCreateFence(ctx_.device, &fenceCi, nullptr, &frames_[i].fence) != VK_SUCCESS) return false;

        if (createBuffer(ctx_, uboSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         frames_[i].ubo, frames_[i].uboMemory) != VK_SUCCESS)
            return false;
        vmaMapMemory(ctx_.allocator, frames_[i].uboMemory, &frames_[i].uboMapped);

        if (createBuffer(ctx_, sizeof(LightingUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         frames_[i].lightingUbo, frames_[i].lightingUboMemory) != VK_SUCCESS)
            return false;
        vmaMapMemory(ctx_.allocator, frames_[i].lightingUboMemory,
                    &frames_[i].lightingUboMapped);

        VkDescriptorBufferInfo sceneBuf = {};
        sceneBuf.buffer = frames_[i].ubo;
        sceneBuf.offset = 0;
        sceneBuf.range = uboSize;
        VkDescriptorBufferInfo materialBuf = {};
        materialBuf.buffer = materialUbo_;
        materialBuf.offset = 0;
        materialBuf.range = sizeof(MaterialUBO);

        VkWriteDescriptorSet writes[2] = {};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = frames_[i].sceneSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo = &sceneBuf;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = frames_[i].sceneSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        writes[1].pBufferInfo = &materialBuf;
        vkUpdateDescriptorSets(ctx_.device, 2, writes, 0, nullptr);

        // Lighting sets: one per frame slot per path (low-res GB / full-res GT).
        const VkDescriptorSetLayout lightingLayout = deferred_.lightingSetLayout();
        VkDescriptorSetAllocateInfo lightAlloc = {};
        lightAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        lightAlloc.descriptorPool = descriptorPool_;
        lightAlloc.descriptorSetCount = 1;
        lightAlloc.pSetLayouts = &lightingLayout;
        if (vkAllocateDescriptorSets(ctx_.device, &lightAlloc, &frames_[i].lightingSetGb) != VK_SUCCESS)
            return false;
        if (vkAllocateDescriptorSets(ctx_.device, &lightAlloc, &frames_[i].lightingSetGt) != VK_SUCCESS)
            return false;

        const VkImageView shadowView = shadowsActive_ ? shadow_.arrayView : VK_NULL_HANDLE;
        deferred_.writeLightingSet(ctx_, frames_[i].lightingSetGb, frames_[i].lightingUbo,
                                   gbAlbedo_.view, gbNormal_.view, gbMaterial_.view,
                                   gbEmissive_.view, gbDepth_.view, gbAo_.view, shadowView);
        deferred_.writeLightingSet(ctx_, frames_[i].lightingSetGt, frames_[i].lightingUbo,
                                   gtAlbedo_.view, gtNormal_.view, gtMaterial_.view,
                                   gtEmissive_.view, gtDepth_.view, gtAo_.view, shadowView);

        // Per-path transparent sets (each binds its own path's SSAO texture).
        const VkDescriptorSetLayout transparentLayout = deferred_.transparentSetLayout();
        VkDescriptorSetAllocateInfo transparentAlloc = {};
        transparentAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        transparentAlloc.descriptorPool = descriptorPool_;
        transparentAlloc.descriptorSetCount = 1;
        transparentAlloc.pSetLayouts = &transparentLayout;
        if (vkAllocateDescriptorSets(ctx_.device, &transparentAlloc,
                                     &frames_[i].transparentSetGb) != VK_SUCCESS)
            return false;
        if (vkAllocateDescriptorSets(ctx_.device, &transparentAlloc,
                                     &frames_[i].transparentSetGt) != VK_SUCCESS)
            return false;
        deferred_.writeTransparentSet(ctx_, frames_[i].transparentSetGb, frames_[i].lightingUbo,
                                      gbAo_.view, shadowView, gbSsrSrc_.view, gbPyramid_.chainView);
        deferred_.writeTransparentSet(ctx_, frames_[i].transparentSetGt, frames_[i].lightingUbo,
                                      gtAo_.view, shadowView, gtSsrSrc_.view, gtPyramid_.chainView);
    }

    // SSAO sets (static bindings; per-frame data goes through push constants).
    {
        VkDescriptorSetAllocateInfo ssaoAlloc = {};
        ssaoAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ssaoAlloc.descriptorPool = descriptorPool_;
        ssaoAlloc.descriptorSetCount = 1;
        const VkDescriptorSetLayout ssaoLayout = deferred_.ssaoSetLayout();
        const VkDescriptorSetLayout blurLayout = deferred_.ssaoBlurSetLayout();
        ssaoAlloc.pSetLayouts = &ssaoLayout;
        if (vkAllocateDescriptorSets(ctx_.device, &ssaoAlloc, &ssaoSetGb_) != VK_SUCCESS) return false;
        if (vkAllocateDescriptorSets(ctx_.device, &ssaoAlloc, &ssaoSetGt_) != VK_SUCCESS) return false;
        ssaoAlloc.pSetLayouts = &blurLayout;
        if (vkAllocateDescriptorSets(ctx_.device, &ssaoAlloc, &ssaoBlurSetGb_) != VK_SUCCESS) return false;
        if (vkAllocateDescriptorSets(ctx_.device, &ssaoAlloc, &ssaoBlurSetGt_) != VK_SUCCESS) return false;
        deferred_.writeSsaoSet(ctx_, ssaoSetGb_, gbDepth_.view, gbNormal_.view, gbAoRaw_.view);
        deferred_.writeSsaoSet(ctx_, ssaoSetGt_, gtDepth_.view, gtNormal_.view, gtAoRaw_.view);
        deferred_.writeSsaoBlurSet(ctx_, ssaoBlurSetGb_, gbAoRaw_.view, gbAo_.view);
        deferred_.writeSsaoBlurSet(ctx_, ssaoBlurSetGt_, gtAoRaw_.view, gtAo_.view);
    }

    {
        VkDescriptorSetAllocateInfo bloomAlloc = {};
        bloomAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        bloomAlloc.descriptorPool = descriptorPool_;
        bloomAlloc.descriptorSetCount = 1;
        const VkDescriptorSetLayout bloomLayout = deferred_.bloomSetLayout();
        bloomAlloc.pSetLayouts = &bloomLayout;
        auto allocBloom = [&](VkDescriptorSet& set) {
            return vkAllocateDescriptorSets(ctx_.device, &bloomAlloc, &set) == VK_SUCCESS;
        };
        if (!allocBloom(bloomExtractGb_) || !allocBloom(bloomBlurHGb_) ||
            !allocBloom(bloomBlurVGb_) || !allocBloom(bloomCompGb_) ||
            !allocBloom(bloomExtractGt_) || !allocBloom(bloomBlurHGt_) ||
            !allocBloom(bloomBlurVGt_) || !allocBloom(bloomCompGt_))
            return false;
        deferred_.writeBloomSet(ctx_, bloomExtractGb_, gbColor_.view, gbBloomA_.view);
        deferred_.writeBloomSet(ctx_, bloomBlurHGb_, gbBloomA_.view, gbBloomB_.view);
        deferred_.writeBloomSet(ctx_, bloomBlurVGb_, gbBloomB_.view, gbBloomA_.view);
        deferred_.writeBloomSet(ctx_, bloomCompGb_, gbBloomA_.view, gbColor_.view);
        deferred_.writeBloomSet(ctx_, bloomExtractGt_, finalImage_.view, gtBloomA_.view);
        deferred_.writeBloomSet(ctx_, bloomBlurHGt_, gtBloomA_.view, gtBloomB_.view);
        deferred_.writeBloomSet(ctx_, bloomBlurVGt_, gtBloomB_.view, gtBloomA_.view);
        deferred_.writeBloomSet(ctx_, bloomCompGt_, gtBloomA_.view, finalImage_.view);
    }

    // One renderFinished semaphore per swapchain image: the submit signals the
    // semaphore for the acquired image, and present waits on that same image's
    // semaphore.  Reusing per-image semaphores avoids reuse while a previous
    // present is still pending.
    renderFinished_.resize(swapchain_.imageCount(), VK_NULL_HANDLE);
    for (size_t i = 0; i < renderFinished_.size(); ++i) {
        if (vkCreateSemaphore(ctx_.device, &semCi, nullptr, &renderFinished_[i]) != VK_SUCCESS) return false;
    }

    if (!timestamps_.create(ctx_, kFramesInFlight)) return false;
    return true;
}

bool Renderer::recreateSwapchain(uint32_t width, uint32_t height, bool vsync) {
    // In-flight frames may still reference the old swapchain images, so wait
    // for all queued work to finish before destroying the previous semaphores
    // and swapchain (the image count may change after the recreate).
    vkDeviceWaitIdle(ctx_.device);

    for (VkSemaphore sem : renderFinished_) {
        if (sem) vkDestroySemaphore(ctx_.device, sem, nullptr);
    }
    renderFinished_.clear();

    if (!swapchain_.create(ctx_, width, height, vsync)) return false;

    // Rebuild one present semaphore per swapchain image: the new swapchain may
    // have a different image count than the previous one.
    VkSemaphoreCreateInfo semCi = {};
    semCi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    renderFinished_.resize(swapchain_.imageCount(), VK_NULL_HANDLE);
    for (size_t i = 0; i < renderFinished_.size(); ++i) {
        if (vkCreateSemaphore(ctx_.device, &semCi, nullptr, &renderFinished_[i]) != VK_SUCCESS) return false;
    }
    return true;
}

bool Renderer::createScreenshotStaging() {
    screenshotSize_ = static_cast<VkDeviceSize>(opts_.displayWidth) * opts_.displayHeight * 8; // RGBA16F
    if (createBuffer(ctx_, screenshotSize_, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     screenshotStaging_, screenshotStagingMemory_) != VK_SUCCESS)
        return false;
    vmaMapMemory(ctx_.allocator, screenshotStagingMemory_, &screenshotMapped_);
    return true;
}

void Renderer::updateSceneUBO(uint32_t frameIndex, bool jitter, uint32_t renderW, uint32_t renderH,
                              const Mat4& view, const Mat4& proj, const Mat4& projJittered,
                              const Mat4& prevViewProj) {
    FrameResources& fr = frames_[frameIndex % kFramesInFlight];
    SceneUBO ubo;
    deferred_.fillSceneUBO(ubo, scene_, camera_, view, proj, projJittered, prevViewProj,
                           renderW, renderH, jitterX_, jitterY_, jitter);
    std::memcpy(fr.uboMapped, &ubo, sizeof(ubo));
}

void Renderer::updateLightingUBO(uint32_t frameIndex, const Mat4& invViewProj,
                                 const ShadowFrame* shadow) {
    FrameResources& fr = frames_[frameIndex % kFramesInFlight];
    LightingUBO ubo;
    deferred_.fillLightingUBO(ubo, scene_, camera_, invViewProj, nullptr, shadow, iblIntensity_);
    std::memcpy(fr.lightingUboMapped, &ubo, sizeof(ubo));
}

void Renderer::applyCameraKeyframe(uint32_t frameIndex) {
    if (path_.empty()) return;
    const CameraKeyframe& kf = path_[frameIndex % path_.size()];
    camera_.setPose(kf.position, kf.forward, kf.up);
}

void Renderer::updateCamera(uint32_t frameIndex, float dt) {
    if (!path_.empty()) {
        applyCameraKeyframe(frameIndex);
    } else {
        camera_.updateFreeFly(window_.input(), dt);
    }
    window_.clearMouseDelta();
}

void Renderer::captureScreenshotIntoStaging(VkCommandBuffer cmd) {
    // finalImage_ is SHADER_READ_ONLY here (sampled by the present pass).
    imageBarrier(cmd, finalImage_.image, finalImage_.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                 VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferImageCopy region = {};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {opts_.displayWidth, opts_.displayHeight, 1};
    vkCmdCopyImageToBuffer(cmd, finalImage_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           screenshotStaging_, 1, &region);
    imageBarrier(cmd, finalImage_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    finalImage_.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void Renderer::saveScreenshot(const std::string& path) {
    if (!screenshotMapped_) return;
    if (!savePngFromHalfRgba(path.c_str(), static_cast<const uint8_t*>(screenshotMapped_),
                             opts_.displayWidth, opts_.displayHeight, opts_.exposure)) {
        std::fprintf(stderr, "failed to save screenshot %s\n", path.c_str());
    }
}

void Renderer::recordFrame(uint32_t frameIndex, uint32_t swapchainIndex) {
    const uint32_t slot = frameIndex % kFramesInFlight;
    FrameResources& fr = frames_[slot];
    VkCommandBuffer cmd = fr.cmd;

    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    timestamps_.resetForFrame(cmd, slot);
    timestamps_.frameBegin(cmd, slot);

    const bool useUpscaler = opts_.upscalerName != "none" && upscaler_ != nullptr;
    const bool jitter = useUpscaler && upscalerNeedsJitter(upscaler_.get());
    const float aspect = static_cast<float>(opts_.displayWidth) / static_cast<float>(opts_.displayHeight);
    const Mat4 view = camera_.view();
    const Mat4 proj = camera_.proj(aspect);
    Mat4 projJittered = proj;
    prevJitterX_ = jitterX_;
    prevJitterY_ = jitterY_;
    if (jitter) {
        const Vec2 h = halton23(frameIndex + 1);
        // SR_NO_JITTER keeps the low-res path but zeroes the offsets (and the
        // jitter reported to the upscaler) to isolate jitter from resolution.
        jitterX_ = diagNoJitter_ ? 0.f : h.x - 0.5f;
        jitterY_ = diagNoJitter_ ? 0.f : h.y - 0.5f;
        // Sub-pixel jitter must shift NDC by a constant (every temporal
        // upscaler assumes: jittered pixel pos = unjittered pos + jitter),
        // i.e. clip.xy += offset * clip.w.  clip.w = -z_view comes from row 3
        // (m[11] = -1), so the offset lands in column 2 (m[8]/m[9]) with a
        // negative sign.  Writing the translation column (m[12]/m[13])
        // instead would add a constant clip-space term, making the pixel
        // shift depth-dependent (jitterX / viewDepth) and breaking the
        // uniform-jitter contract the upscalers are told about.
        projJittered.m[8] -= jitterX_ * 2.f / static_cast<float>(renderWidth_);
        projJittered.m[9] -= jitterY_ * 2.f / static_cast<float>(renderHeight_);
    } else {
        jitterX_ = 0.f;
        jitterY_ = 0.f;
    }

    updateSceneUBO(frameIndex, jitter, renderWidth_, renderHeight_, view, proj, projJittered, prevViewProj_);

    const bool gbuffer = useUpscaler;

    // Lighting reconstructs world positions with the inverse of the exact
    // view-projection used for this pass (jittered for the low-res path).
    const Mat4 viewProjUsed = Mat4::multiply(gbuffer ? projJittered : proj, view);

    // CSM sun shadows: pick the first shadow-casting directional light (same
    // light-list selection rule as fillLightingUBO) and compute the cascades.
    // The GT path samples the same map — GT is the same lighting at native res.
    ShadowFrame shadowFrame;
    const ShadowFrame* shadow = nullptr;
    if (shadowsActive_) {
        const std::vector<Light>& lights =
            scene_.lights.empty() ? defaultLights() : scene_.lights;
        for (uint32_t i = 0; i < lights.size(); ++i) {
            if (lights[i].type == LightType::Directional && lights[i].castShadow) {
                DeferredCore::computeCascadeVPs(camera_, aspect, lights[i].positionOrDirection,
                                                shadowFrame.cascadeVp, shadowFrame.splitDepth);
                shadowFrame.lightIndex = static_cast<int32_t>(i);
                shadowFrame.debugCascades = opts_.shadowDebug;
                shadow = &shadowFrame;
                break;
            }
        }
    }
    updateLightingUBO(frameIndex, Mat4::inverse(viewProjUsed), shadow);

    ImageResource& tgtAlbedo = gbuffer ? gbAlbedo_ : gtAlbedo_;
    ImageResource& tgtNormal = gbuffer ? gbNormal_ : gtNormal_;
    ImageResource& tgtMaterial = gbuffer ? gbMaterial_ : gtMaterial_;
    ImageResource& tgtEmissive = gbuffer ? gbEmissive_ : gtEmissive_;
    ImageResource& tgtDepth = gbuffer ? gbDepth_ : gtDepth_;
    ImageResource& tgtAoRaw = gbuffer ? gbAoRaw_ : gtAoRaw_;
    ImageResource& tgtAo = gbuffer ? gbAo_ : gtAo_;
    ImageResource& litTarget = gbuffer ? gbColor_ : finalImage_;
    const uint32_t sceneW = gbuffer ? renderWidth_ : opts_.displayWidth;
    const uint32_t sceneH = gbuffer ? renderHeight_ : opts_.displayHeight;

    auto transition = [&](ImageResource& rt, VkImageLayout target,
                          VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                          VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
                          VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
        imageBarrier(cmd, rt.image, rt.layout, target, srcStage, srcAccess, dstStage, dstAccess,
                     aspect);
        rt.layout = target;
    };

    // GBuffer targets: last frame they were sampled by the lighting fragment
    // shader (or the upscaler, which may run compute) -> attachment writes.
    transition(tgtAlbedo, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
               sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite);
    transition(tgtNormal, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
               sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite);
    transition(tgtMaterial, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
               sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite);
    transition(tgtEmissive, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
               sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite);
    transition(tgtDepth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
               sync::kSampleStages, sync::kSampled, sync::kDepthTests, sync::kDepthReadWrite,
               VK_IMAGE_ASPECT_DEPTH_BIT);
    if (gbuffer) {
        transition(gbMotion_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite);
    }

    timestamps_.sceneBegin(cmd, slot);

    // --- GBuffer pass ---------------------------------------------------------
    {
        VkRenderingAttachmentInfo colors[5] = {
            makeColorAttachment(tgtAlbedo.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_ATTACHMENT_LOAD_OP_CLEAR),
            makeColorAttachment(tgtNormal.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_ATTACHMENT_LOAD_OP_CLEAR, 0.f, 0.f, 1.f),
            makeColorAttachment(tgtMaterial.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_ATTACHMENT_LOAD_OP_CLEAR),
            makeColorAttachment(tgtEmissive.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_ATTACHMENT_LOAD_OP_CLEAR),
            makeColorAttachment(gbuffer ? gbMotion_.view : VK_NULL_HANDLE,
                                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_ATTACHMENT_LOAD_OP_CLEAR)};
        const uint32_t colorCount = gbuffer ? 5 : 4;
        VkRenderingAttachmentInfo depth =
            makeDepthAttachment(tgtDepth.view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                VK_ATTACHMENT_LOAD_OP_CLEAR);
        beginRendering(cmd, sceneW, sceneH, colorCount, colors, &depth);
        deferred_.recordGBufferDraws(cmd, scene_, !gbuffer, fr.sceneSet, textureSet_,
                                     materialStride_, sceneW, sceneH,
                                     Mat4::multiply(proj, view));
        vkCmdEndRendering(cmd);
    }

    // --- Shadow pass (sun CSM, one 2048^2 layer per cascade) -----------------
    // Runs once per frame and feeds both the LR and the GT lighting paths.
    if (shadow) {
        // Sampled by the lighting fragment shaders last frame -> depth attach.
        imageBarrier(cmd, shadow_.image, shadow_.layout,
                     VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                     sync::kFragment, sync::kSampled, sync::kDepthTests, sync::kDepthReadWrite,
                     VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, kShadowCascadeCount);
        shadow_.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        deferred_.recordShadowPass(cmd, shadow_, scene_, shadow->cascadeVp, fr.sceneSet,
                                   textureSet_, materialStride_);
        // Depth writes -> shadow-comparison samples in the lighting shaders.
        imageBarrier(cmd, shadow_.image, shadow_.layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, sync::kDepthWrite,
                     sync::kFragment, sync::kSampled,
                     VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, kShadowCascadeCount);
        shadow_.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    // --- Lighting pass (deferred PBR + IBL, skybox on far-plane pixels) --------
    transition(tgtAlbedo, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled);
    transition(tgtNormal, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled);
    transition(tgtMaterial, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled);
    transition(tgtEmissive, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled);
    transition(tgtDepth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               sync::kDepthTests, sync::kDepthWrite, sync::kSampleStages, sync::kSampled,
               VK_IMAGE_ASPECT_DEPTH_BIT);
    // Hi-Z pyramid for the glass SSR marcher (skippable when no BLEND
    // material exists; the pyramid then just keeps its last contents).
    if (hasTransparency_)
        deferred_.recordDepthPyramidPass(cmd, gbuffer ? gbPyramid_ : gtPyramid_);
    if (gbuffer) {
        transition(gbMotion_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled);
    }

    // --- GTAO (depth + normal -> working AO/Z -> 5x5 bilateral denoise) ---------
    // AoRaw ping-pongs between ssao main (storage write) and blur (sampled).
    transition(tgtAoRaw, VK_IMAGE_LAYOUT_GENERAL,
               sync::kCompute, sync::kSampled, sync::kCompute, sync::kStorageWrite);
    deferred_.recordSsaoPass(cmd, gbuffer ? ssaoSetGb_ : ssaoSetGt_, viewProjUsed, frameIndex,
                             sceneW, sceneH);
    transition(tgtAoRaw, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               sync::kCompute, sync::kStorageWrite, sync::kCompute, sync::kSampled);
    // Ao: sampled by the lighting fragment shader, rewritten by the blur pass.
    transition(tgtAo, VK_IMAGE_LAYOUT_GENERAL,
               sync::kFragment, sync::kSampled, sync::kCompute, sync::kStorageWrite);
    deferred_.recordSsaoBlurPass(cmd, gbuffer ? ssaoBlurSetGb_ : ssaoBlurSetGt_, sceneW, sceneH);
    transition(tgtAo, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               sync::kCompute, sync::kStorageWrite, sync::kFragment, sync::kSampled);

    // litTarget was sampled by last frame's upscaler (GB) or present pass (GT).
    transition(litTarget, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
               sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite);

    deferred_.recordLightingPass(cmd, gbuffer ? fr.lightingSetGb : fr.lightingSetGt,
                                 litTarget.view, sceneW, sceneH);

    // --- Transparency pass (alpha-blended surfaces over the lit scene) --------
    // Copy opaque HDR for SSR: the transparent pass writes the same color
    // target, so glass cannot sample it in-place.
    if (hasTransparency_) {
        transition(litTarget, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kCopy, sync::kTransferRead);
        ImageResource& ssrSrc = gbuffer ? gbSsrSrc_ : gtSsrSrc_;
        // ssrSrc was sampled by the transparent fragment shader last frame.
        transition(ssrSrc, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   sync::kFragment, sync::kSampled, sync::kCopy, sync::kTransferWrite);
        copyColorImage(cmd, litTarget.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ssrSrc.image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, sceneW, sceneH);
        transition(ssrSrc, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kCopy, sync::kTransferWrite, sync::kFragment, sync::kSampled);
        transition(litTarget, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kCopy, sync::kTransferRead, sync::kColorAttach, sync::kColorReadWrite);
    }
    if (hasTransparency_) {
        if (gbuffer) {
            transition(gbMotion_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                       sync::kFragment, sync::kSampled, sync::kColorAttach, sync::kColorReadWrite);
            transition(gbReactive_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                       sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite);
        }
        transition(tgtDepth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                   sync::kFragment, sync::kSampled, sync::kDepthTests, sync::kDepthRead,
                   VK_IMAGE_ASPECT_DEPTH_BIT);

        VkRenderingAttachmentInfo tColors[3] = {};
        tColors[0] =
            makeColorAttachment(litTarget.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_ATTACHMENT_LOAD_OP_LOAD);
        if (gbuffer) {
            tColors[1] =
                makeColorAttachment(gbMotion_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_LOAD);
            tColors[2] =
                makeColorAttachment(gbReactive_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR);
        }
        VkRenderingAttachmentInfo tDepth =
            makeDepthAttachment(tgtDepth.view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                                VK_ATTACHMENT_LOAD_OP_LOAD);
        beginRendering(cmd, sceneW, sceneH, gbuffer ? 3 : 1, tColors, &tDepth);
        deferred_.recordTransparentDraws(cmd, scene_, !gbuffer, fr.sceneSet, textureSet_,
                                         gbuffer ? fr.transparentSetGb : fr.transparentSetGt,
                                         materialStride_, sceneW, sceneH,
                                         Mat4::multiply(proj, view), camera_.position);
        vkCmdEndRendering(cmd);

        if (gbuffer) {
            transition(gbMotion_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled);
            transition(gbReactive_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled);
        }
        transition(tgtDepth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kDepthTests, sync::kDepthRead, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    transition(litTarget, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled);
    if (opts_.bloom) {
        if (gbuffer) {
            deferred_.recordBloomPass(cmd, bloomExtractGb_, bloomBlurHGb_, bloomBlurVGb_,
                                      bloomCompGb_, gbBloomA_.image, gbBloomB_.image, gbColor_.image,
                                      gbBloomA_.layout, gbBloomB_.layout, gbColor_.layout,
                                      renderWidth_, renderHeight_);
        } else {
            deferred_.recordBloomPass(cmd, bloomExtractGt_, bloomBlurHGt_, bloomBlurVGt_,
                                      bloomCompGt_, gtBloomA_.image, gtBloomB_.image,
                                      finalImage_.image, gtBloomA_.layout, gtBloomB_.layout,
                                      finalImage_.layout, opts_.displayWidth, opts_.displayHeight);
        }
    }
    timestamps_.sceneEnd(cmd, slot);

    if (!gbuffer) {
        // GT has no upscaler pass; emit a zero-width range so every query slot
        // is written (readback with WAIT_BIT otherwise blocks forever).
        timestamps_.upscaleBegin(cmd, slot);
        timestamps_.upscaleEnd(cmd, slot);
    }

    if (gbuffer) {
        // gbColor_/gbMotion_/gbDepth_ are already SHADER_READ_ONLY (lighting
        // wrote gbColor_ and sampled depth; motion was resolved right after
        // the GBuffer pass).  finalImage_ was sampled by the present pass;
        // the upscaler writes it as a storage image (fragment or compute).
        transition(finalImage_, VK_IMAGE_LAYOUT_GENERAL,
                   sync::kFragment, sync::kSampled, sync::kSampleStages, sync::kStorageWrite);

        UpscalerResources res;
        res.color = gbColor_.image;
        res.colorView = gbColor_.view;
        res.depth = gbDepth_.image;
        res.depthView = gbDepth_.view;
        res.motion = gbMotion_.image;
        res.motionView = gbMotion_.view;
        if (hasTransparency_) {
            res.reactive = gbReactive_.image;
            res.reactiveView = gbReactive_.view;
        }
        res.output = finalImage_.image;
        res.outputView = finalImage_.view;

        CameraParams cam;
        std::memcpy(cam.view, view.m, sizeof(cam.view));
        std::memcpy(cam.proj, proj.m, sizeof(cam.proj));
        std::memcpy(cam.prevViewProj, prevViewProj_.m, sizeof(cam.prevViewProj));
        cam.jitterX = jitterX_;
        cam.jitterY = jitterY_;
        cam.prevJitterX = prevJitterX_;
        cam.prevJitterY = prevJitterY_;
        cam.cameraNear = camera_.nearPlane;
        cam.cameraFar = camera_.farPlane;
        cam.fovY = camera_.fovY;

        FrameParams frame;
        frame.frameIndex = static_cast<int>(frameIndex);
        frame.deltaTime = deltaTime_;
        frame.preExposure = 1.f;
        frame.resetHistory = (frameIndex == 0);

        timestamps_.upscaleBegin(cmd, slot);
        upscaler_->dispatch(cmd, res, cam, frame);
        timestamps_.upscaleEnd(cmd, slot);

        transition(finalImage_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kSampleStages, sync::kStorageWrite, sync::kFragment, sync::kSampled);
    }
    // GT path: finalImage_ is already SHADER_READ_ONLY (post-lighting).

    const VkImage swapImage = swapchain_.image(swapchainIndex);
    const VkImageView swapView = swapchain_.view(swapchainIndex);
    imageBarrier(cmd, swapImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                 sync::kColorAttach, sync::kColorWrite);

    VkRenderingAttachmentInfo presentColor =
        makeColorAttachment(swapView, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ATTACHMENT_LOAD_OP_CLEAR);
    beginRendering(cmd, swapchain_.extent().width, swapchain_.extent().height, 1, &presentColor, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, presentPipeline_);
    VkViewport pvp = {0.f, 0.f, static_cast<float>(swapchain_.extent().width),
                      static_cast<float>(swapchain_.extent().height), 0.f, 1.f};
    vkCmdSetViewport(cmd, 0, 1, &pvp);
    VkRect2D psc = {{0, 0}, {swapchain_.extent().width, swapchain_.extent().height}};
    vkCmdSetScissor(cmd, 0, 1, &psc);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, presentPipelineLayout_, 0, 1,
                            &presentSet_, 0, nullptr);
    const float presentPush[4] = {opts_.exposure, 0.f, 0.f, 0.f};
    vkCmdPushConstants(cmd, presentPipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(presentPush), presentPush);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);

    imageBarrier(cmd, swapImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 sync::kColorAttach, sync::kColorWrite, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE);

    const bool capturing = !opts_.screenshotPath.empty() && static_cast<int>(frameIndex) == opts_.frames - 1;
    if (capturing) {
        captureScreenshotIntoStaging(cmd);
    }

    timestamps_.frameEnd(cmd, slot);
    vkEndCommandBuffer(cmd);

    prevViewProj_ = Mat4::multiply(proj, view);
}

void Renderer::run() {
    uint32_t frameIndex = 0;
    auto lastTime = std::chrono::steady_clock::now();

    while (true) {
        if (!window_.poll()) break;

        // WM_SIZE sets input_.resized; consume it here and rebuild the swapchain
        // against the actual client size (the previous recreate paths used the
        // fixed opts_ display size, so interactive resizes were ignored).
        // TODO: the offscreen targets (finalImage_, screenshot staging, ...) are
        // still allocated from opts_.displayWidth/Height and are not rebuilt
        // here; the present pass scales them to the swapchain extent, so a
        // resize works at the presentation level only.
        if (window_.input().resized) {
            window_.clearMouseDelta();
            recreateSwapchain(static_cast<uint32_t>(window_.width()),
                              static_cast<uint32_t>(window_.height()), opts_.vsync);
        }

        if (opts_.frames >= 0 && static_cast<int>(frameIndex) >= opts_.frames) break;

        const auto now = std::chrono::steady_clock::now();
        const float dt = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;
        // Bench mode (opts_.frames >= 0) keeps the fixed 1/60 default so the
        // upscaler's frame-to-frame integration is reproducible across machines
        // and runs; interactive mode tracks real wall-clock time.
        if (opts_.frames < 0) deltaTime_ = dt;
        updateCamera(frameIndex, dt);

        const uint32_t slot = frameIndex % kFramesInFlight;

        // Ensure the previous work using this frame slot is complete before
        // reusing its fence / image-available semaphore / UBO.
        vkWaitForFences(ctx_.device, 1, &frames_[slot].fence, VK_TRUE, UINT64_MAX);
        vkResetFences(ctx_.device, 1, &frames_[slot].fence);

        // The frame that used this slot (frameIndex - kFramesInFlight) is now
        // complete; harvest its GPU timings before the slot is reused.
        if (!opts_.frameTimesPath.empty() && frameIndex >= kFramesInFlight) {
            frameTimes_.push_back(timestamps_.read(ctx_, slot));
        }

        uint32_t swapIndex = 0;
        VkResult acq = swapchain_.acquireNext(ctx_, frames_[slot].imageAvailable, swapIndex);
        if (acq == VK_ERROR_OUT_OF_DATE_KHR || acq == VK_SUBOPTIMAL_KHR) {
            recreateSwapchain(opts_.displayWidth, opts_.displayHeight, opts_.vsync);
            continue;
        }
        if (acq != VK_SUCCESS) break;

        recordFrame(frameIndex, swapIndex);

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit = {};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &frames_[slot].imageAvailable;
        submit.pWaitDstStageMask = &waitStage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &frames_[slot].cmd;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &renderFinished_[swapIndex];
        if (vkQueueSubmit(ctx_.graphicsQueue, 1, &submit, frames_[slot].fence) != VK_SUCCESS) break;

        VkResult pres = swapchain_.present(ctx_, swapIndex, renderFinished_[swapIndex]);
        if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) {
            recreateSwapchain(opts_.displayWidth, opts_.displayHeight, opts_.vsync);
        } else if (pres != VK_SUCCESS) {
            break;
        }

        if (!opts_.screenshotPath.empty() && static_cast<int>(frameIndex) == opts_.frames - 1) {
            vkWaitForFences(ctx_.device, 1, &frames_[slot].fence, VK_TRUE, UINT64_MAX);
            saveScreenshot(opts_.screenshotPath);
        }

        ++frameIndex;
    }

    vkDeviceWaitIdle(ctx_.device);

    if (!opts_.frameTimesPath.empty()) {
        // Drain the last in-flight frames (all work is complete after WaitIdle).
        for (uint32_t f = static_cast<uint32_t>(frameTimes_.size()); f < frameIndex; ++f) {
            frameTimes_.push_back(timestamps_.read(ctx_, f % kFramesInFlight));
        }
        const MemoryBudgetInfo budget = queryMemoryBudget(ctx_);
        VkDeviceSize heapTotal = 0;
        VkDeviceSize vmaTotal = 0;
        for (uint32_t i = 0; i < budget.heapCount; ++i) {
            heapTotal += budget.heapUsage[i];
            vmaTotal += budget.vmaAllocationBytes[i];
        }
        const uint64_t algoBytes = upscaler_ ? upscaler_->gpuMemoryBytes() : 0;
        std::ofstream csv(opts_.frameTimesPath);
        csv << "frame,frameMs,sceneMs,upscaleMs,vramAlgoBytes,vramTotalBytes,vramVmaBytes\n";
        for (size_t i = 0; i < frameTimes_.size(); ++i) {
            csv << i << ',' << frameTimes_[i].frameMs << ',' << frameTimes_[i].sceneMs << ','
                << frameTimes_[i].upscaleMs << ',' << algoBytes << ',' << heapTotal << ','
                << vmaTotal << '\n';
        }
        std::fprintf(stdout, "frameTimes=%zu written to %s\n", frameTimes_.size(),
                     opts_.frameTimesPath.c_str());
    }

    if (opts_.frames >= 0 && frameIndex > 0) {
        const TimestampQuery::Timings t = timestamps_.read(ctx_, (frameIndex - 1) % kFramesInFlight);
        std::fprintf(stdout, "frames=%u lastFrameMs=%.3f sceneMs=%.3f upscaleMs=%.3f\n",
                     frameIndex, t.frameMs, t.sceneMs, t.upscaleMs);
    }
}

void Renderer::shutdown() {
    if (!ctx_.device) return;
    vkDeviceWaitIdle(ctx_.device);

    if (upscaler_) { upscaler_->shutdown(); upscaler_.reset(); }

    if (screenshotStaging_) { vmaDestroyBuffer(ctx_.allocator, screenshotStaging_, screenshotStagingMemory_); screenshotStaging_ = VK_NULL_HANDLE; screenshotStagingMemory_ = VK_NULL_HANDLE; }

    timestamps_.destroy(ctx_);

    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        FrameResources& fr = frames_[i];
        if (fr.ubo) { vmaDestroyBuffer(ctx_.allocator, fr.ubo, fr.uboMemory); fr.ubo = VK_NULL_HANDLE; fr.uboMemory = VK_NULL_HANDLE; }
        if (fr.lightingUbo) { vmaDestroyBuffer(ctx_.allocator, fr.lightingUbo, fr.lightingUboMemory); fr.lightingUbo = VK_NULL_HANDLE; fr.lightingUboMemory = VK_NULL_HANDLE; }
        if (fr.imageAvailable) { vkDestroySemaphore(ctx_.device, fr.imageAvailable, nullptr); fr.imageAvailable = VK_NULL_HANDLE; }
        if (fr.fence) { vkDestroyFence(ctx_.device, fr.fence, nullptr); fr.fence = VK_NULL_HANDLE; }
    }
    for (VkSemaphore sem : renderFinished_) {
        if (sem) vkDestroySemaphore(ctx_.device, sem, nullptr);
    }
    renderFinished_.clear();

    if (presentPipeline_) { vkDestroyPipeline(ctx_.device, presentPipeline_, nullptr); presentPipeline_ = VK_NULL_HANDLE; }
    if (presentPipelineLayout_) { vkDestroyPipelineLayout(ctx_.device, presentPipelineLayout_, nullptr); presentPipelineLayout_ = VK_NULL_HANDLE; }
    if (presentFrag_) { vkDestroyShaderModule(ctx_.device, presentFrag_, nullptr); presentFrag_ = VK_NULL_HANDLE; }

    if (descriptorPool_) { vkDestroyDescriptorPool(ctx_.device, descriptorPool_, nullptr); descriptorPool_ = VK_NULL_HANDLE; }
    if (presentSetLayout_) { vkDestroyDescriptorSetLayout(ctx_.device, presentSetLayout_, nullptr); presentSetLayout_ = VK_NULL_HANDLE; }

    if (materialUbo_) { vmaDestroyBuffer(ctx_.allocator, materialUbo_, materialUboMemory_); materialUbo_ = VK_NULL_HANDLE; materialUboMemory_ = VK_NULL_HANDLE; }

    gbColor_.destroy(ctx_);
    gbMotion_.destroy(ctx_);
    gbReactive_.destroy(ctx_);
    gbDepth_.destroy(ctx_);
    gbAlbedo_.destroy(ctx_);
    gbNormal_.destroy(ctx_);
    gbMaterial_.destroy(ctx_);
    gbEmissive_.destroy(ctx_);
    gtDepth_.destroy(ctx_);
    gtAlbedo_.destroy(ctx_);
    gtNormal_.destroy(ctx_);
    gtMaterial_.destroy(ctx_);
    gtEmissive_.destroy(ctx_);
    gbAoRaw_.destroy(ctx_);
    gbAo_.destroy(ctx_);
    gtAoRaw_.destroy(ctx_);
    gtAo_.destroy(ctx_);
    gbBloomA_.destroy(ctx_);
    gbBloomB_.destroy(ctx_);
    gtBloomA_.destroy(ctx_);
    gtBloomB_.destroy(ctx_);
    gbSsrSrc_.destroy(ctx_);
    gtSsrSrc_.destroy(ctx_);
    deferred_.destroyDepthPyramid(ctx_, gbPyramid_);
    deferred_.destroyDepthPyramid(ctx_, gtPyramid_);
    finalImage_.destroy(ctx_);

    if (shadowsActive_) { deferred_.destroyShadowTargets(ctx_, shadow_); shadowsActive_ = false; }

    deferred_.destroy(ctx_);

    scene_.destroy(ctx_);
    swapchain_.destroy(ctx_);
    ctx_.destroy();
    window_.destroy();
}

} // namespace sr
