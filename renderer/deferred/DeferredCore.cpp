// ============================================================================
// DeferredCore — see DeferredCore.h for the design overview.
// ============================================================================
#include "renderer/deferred/DeferredCore.h"

#include "renderer/core/PathUtil.h"
#include "renderer/core/VkUtil.h"
#include "renderer/core/Vma.h"
#include "renderer/scene/Camera.h"
#include "renderer/scene/Scene.h"

#include <algorithm>
#include <cmath>
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

uint32_t alignUp(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

// Six frustum planes (xyz = normal, w = offset; inside = dot(plane, p) >= 0)
// extracted from a view-projection matrix with Gribb-Hartmann.  Depth is
// Vulkan-style [0,1], so the near plane is row 2 (not row3 + row2).
struct Frustum {
    Vec4 planes[6];
};

Frustum extractFrustum(const Mat4& vp) {
    auto row = [&](int r) { return Vec4{vp.m[r], vp.m[4 + r], vp.m[8 + r], vp.m[12 + r]}; };
    const Vec4 r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
    auto add = [](const Vec4& a, const Vec4& b) {
        return Vec4{a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
    };
    auto sub = [](const Vec4& a, const Vec4& b) {
        return Vec4{a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
    };
    Frustum f;
    f.planes[0] = add(r3, r0); // left
    f.planes[1] = sub(r3, r0); // right
    f.planes[2] = add(r3, r1); // bottom
    f.planes[3] = sub(r3, r1); // top
    f.planes[4] = r2;          // near (depth [0,1])
    f.planes[5] = sub(r3, r2); // far
    return f;
}

bool aabbIntersectsFrustum(const Frustum& f, const Vec3& mn, const Vec3& mx) {
    for (const Vec4& p : f.planes) {
        // Positive vertex: the corner farthest along the plane normal.
        const float x = p.x >= 0.f ? mx.x : mn.x;
        const float y = p.y >= 0.f ? mx.y : mn.y;
        const float z = p.z >= 0.f ? mx.z : mn.z;
        if (p.x * x + p.y * y + p.z * z + p.w < 0.f) return false;
    }
    return true;
}

} // namespace

bool DeferredCore::init(const VulkanContext& ctx, const char* envMapPath) {
    if (!ibl_.build(ctx, envMapPath)) {
        std::fprintf(stderr, "IBL preprocessing failed\n");
        return false;
    }

    // Scene textures: trilinear mipmapping + anisotropy (grazing-angle fix).
    float maxAniso = 4.f;
    if (ctx.features.samplerAnisotropy)
        maxAniso = std::min(16.f, ctx.properties.limits.maxSamplerAnisotropy);
    textureSampler_ = createSampler(ctx, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                    maxAniso, 32.f);
    gbufferSampler_ = createSampler(ctx, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    if (!textureSampler_ || !gbufferSampler_) return false;

    // Depth comparison sampler for the CSM shadow map.  CLAMP_TO_BORDER with
    // an opaque-white border makes off-map reads compare as "lit" (beyond the
    // last cascade the sun is unshadowed).
    VkSamplerCreateInfo shadowSamplerCi = {};
    shadowSamplerCi.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    shadowSamplerCi.magFilter = VK_FILTER_LINEAR; // hardware 2x2 PCF inside the compare
    shadowSamplerCi.minFilter = VK_FILTER_LINEAR;
    shadowSamplerCi.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    shadowSamplerCi.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    shadowSamplerCi.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    shadowSamplerCi.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    shadowSamplerCi.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    shadowSamplerCi.compareEnable = VK_TRUE;
    shadowSamplerCi.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    if (vkCreateSampler(ctx.device, &shadowSamplerCi, nullptr, &shadowSampler_) != VK_SUCCESS)
        return false;

    if (!loadShader(ctx, "gbuffer.vert.spv", gbufferVert_) ||
        !loadShader(ctx, "gbuffer.frag.spv", gbufferFrag_) ||
        !loadShader(ctx, "gbuffer_gt.frag.spv", gbufferGtFrag_) ||
        !loadShader(ctx, "lighting.frag.spv", lightingFrag_) ||
        !loadShader(ctx, "fullscreen.vert.spv", fullscreenVert_) ||
        !loadShader(ctx, "transparent.vert.spv", transparentVert_) ||
        !loadShader(ctx, "transparent.frag.spv", transparentFrag_) ||
        !loadShader(ctx, "transparent_gt.frag.spv", transparentGtFrag_) ||
        !loadShader(ctx, "ssao.comp.spv", ssaoComp_) ||
        !loadShader(ctx, "ssao_blur.comp.spv", ssaoBlurComp_) ||
        !loadShader(ctx, "bloom_extract.comp.spv", bloomExtractComp_) ||
        !loadShader(ctx, "bloom_blur.comp.spv", bloomBlurComp_) ||
        !loadShader(ctx, "bloom_composite.comp.spv", bloomCompositeComp_) ||
        !loadShader(ctx, "shadow_depth.vert.spv", shadowDepthVert_) ||
        !loadShader(ctx, "shadow_depth.frag.spv", shadowDepthFrag_))
        return false;

    if (!createLayouts(ctx)) return false;
    if (!createPipelines(ctx)) return false;
    return true;
}

bool DeferredCore::loadShader(const VulkanContext& ctx, const char* name, VkShaderModule& out) {
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
    return vkCreateShaderModule(ctx.device, &ci, nullptr, &out) == VK_SUCCESS;
}

bool DeferredCore::createLayouts(const VulkanContext& ctx) {
    VkDescriptorSetLayoutBinding sceneBindings[2] = {};
    sceneBindings[0].binding = 0;
    sceneBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sceneBindings[0].descriptorCount = 1;
    sceneBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    sceneBindings[1].binding = 1;
    sceneBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    sceneBindings[1].descriptorCount = 1;
    sceneBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo sceneLayoutCi = {};
    sceneLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    sceneLayoutCi.bindingCount = 2;
    sceneLayoutCi.pBindings = sceneBindings;
    if (vkCreateDescriptorSetLayout(ctx.device, &sceneLayoutCi, nullptr, &sceneSetLayout_) !=
        VK_SUCCESS)
        return false;

    VkDescriptorSetLayoutBinding texBinding = {};
    texBinding.binding = 0;
    texBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    texBinding.descriptorCount = deferred::kMaxTextures; // matches uTextures[] in the shader
    texBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo texLayoutCi = {};
    texLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    texLayoutCi.bindingCount = 1;
    texLayoutCi.pBindings = &texBinding;
    if (vkCreateDescriptorSetLayout(ctx.device, &texLayoutCi, nullptr, &textureSetLayout_) !=
        VK_SUCCESS)
        return false;

    // Lighting: binding 0 = LightingUBO; 1-5 = GBuffer (albedo/normal/material/
    // emissive/depth); 6-8 = IBL (irradiance/prefilter/LUT); 9 = env (skybox);
    // 10 = SSAO (blurred screen-space AO, R16F); 11 = CSM shadow map array
    // (D32, comparison sampler).
    VkDescriptorSetLayoutBinding lightBindings[12] = {};
    lightBindings[0].binding = 0;
    lightBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    lightBindings[0].descriptorCount = 1;
    lightBindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    for (uint32_t i = 1; i < 12; ++i) {
        lightBindings[i].binding = i;
        lightBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        lightBindings[i].descriptorCount = 1;
        lightBindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lightLayoutCi = {};
    lightLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lightLayoutCi.bindingCount = 12;
    lightLayoutCi.pBindings = lightBindings;
    if (vkCreateDescriptorSetLayout(ctx.device, &lightLayoutCi, nullptr, &lightingSetLayout_) !=
        VK_SUCCESS)
        return false;

    // Transparency pass: binding 0 = LightingUBO; 1-3 = IBL; 4 = SSAO;
    // 5 = CSM shadow map; 6 = opaque HDR copy (SSR); 7 = opaque depth (SSR).
    VkDescriptorSetLayoutBinding transparentBindings[8] = {};
    transparentBindings[0].binding = 0;
    transparentBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    transparentBindings[0].descriptorCount = 1;
    transparentBindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    for (uint32_t i = 1; i < 8; ++i) {
        transparentBindings[i].binding = i;
        transparentBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        transparentBindings[i].descriptorCount = 1;
        transparentBindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo transparentLayoutCi = {};
    transparentLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    transparentLayoutCi.bindingCount = 8;
    transparentLayoutCi.pBindings = transparentBindings;
    if (vkCreateDescriptorSetLayout(ctx.device, &transparentLayoutCi, nullptr,
                                    &transparentSetLayout_) != VK_SUCCESS)
        return false;

    // GTAO: binding 0 = depth, 1 = normal (samplers), 2 = working RG16F (storage).
    VkDescriptorSetLayoutBinding ssaoBindings[3] = {};
    for (uint32_t i = 0; i < 3; ++i) {
        ssaoBindings[i].binding = i;
        ssaoBindings[i].descriptorCount = 1;
        ssaoBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        ssaoBindings[i].descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                               : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    }
    VkDescriptorSetLayoutCreateInfo ssaoLayoutCi = {};
    ssaoLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ssaoLayoutCi.bindingCount = 3;
    ssaoLayoutCi.pBindings = ssaoBindings;
    if (vkCreateDescriptorSetLayout(ctx.device, &ssaoLayoutCi, nullptr, &ssaoSetLayout_) !=
        VK_SUCCESS)
        return false;

    // GTAO denoise / bloom: binding 0 = src (sampler), 1 = dst (storage).
    VkDescriptorSetLayoutBinding ssaoBlurBindings[2] = {};
    ssaoBlurBindings[0].binding = 0;
    ssaoBlurBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ssaoBlurBindings[0].descriptorCount = 1;
    ssaoBlurBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    ssaoBlurBindings[1].binding = 1;
    ssaoBlurBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ssaoBlurBindings[1].descriptorCount = 1;
    ssaoBlurBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo ssaoBlurLayoutCi = {};
    ssaoBlurLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ssaoBlurLayoutCi.bindingCount = 2;
    ssaoBlurLayoutCi.pBindings = ssaoBlurBindings;
    return vkCreateDescriptorSetLayout(ctx.device, &ssaoBlurLayoutCi, nullptr,
                                       &ssaoBlurSetLayout_) == VK_SUCCESS;
}

bool DeferredCore::createPipelines(const VulkanContext& ctx) {
    // ScenePush is 192 bytes; require a device with a large-enough push range.
    if (ctx.properties.limits.maxPushConstantsSize < sizeof(ScenePush)) {
        std::fprintf(stderr, "device maxPushConstantsSize=%u < %zu\n",
                     ctx.properties.limits.maxPushConstantsSize, sizeof(ScenePush));
        return false;
    }

    VkPushConstantRange pushRange = {};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(ScenePush);

    VkDescriptorSetLayout sceneLayouts[2] = {sceneSetLayout_, textureSetLayout_};
    VkPipelineLayoutCreateInfo sceneLayoutCi = {};
    sceneLayoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    sceneLayoutCi.setLayoutCount = 2;
    sceneLayoutCi.pSetLayouts = sceneLayouts;
    sceneLayoutCi.pushConstantRangeCount = 1;
    sceneLayoutCi.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(ctx.device, &sceneLayoutCi, nullptr, &scenePipelineLayout_) !=
        VK_SUCCESS)
        return false;

    VkPipelineLayoutCreateInfo lightingLayoutCi = {};
    lightingLayoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    lightingLayoutCi.setLayoutCount = 1;
    lightingLayoutCi.pSetLayouts = &lightingSetLayout_;
    if (vkCreatePipelineLayout(ctx.device, &lightingLayoutCi, nullptr, &lightingPipelineLayout_) !=
        VK_SUCCESS)
        return false;

    VkVertexInputBindingDescription binding = {};
    binding.binding = 0;
    binding.stride = sizeof(Vertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrs[4] = {};
    attrs[0].location = 0;
    attrs[0].binding = 0;
    attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[0].offset = 0;
    attrs[1].location = 1;
    attrs[1].binding = 0;
    attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[1].offset = sizeof(Vec3);
    attrs[2].location = 2;
    attrs[2].binding = 0;
    attrs[2].format = VK_FORMAT_R32G32_SFLOAT;
    attrs[2].offset = sizeof(Vec3) * 2;
    attrs[3].location = 3;
    attrs[3].binding = 0;
    attrs[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attrs[3].offset = sizeof(Vec3) * 2 + sizeof(Vec2);

    VkPipelineVertexInputStateCreateInfo vertexInput = {};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = 4;
    vertexInput.pVertexAttributeDescriptions = attrs;

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

    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState blendAttachments[5] = {};
    for (auto& b : blendAttachments) {
        b.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    }
    VkPipelineColorBlendStateCreateInfo colorBlend = {};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 5;
    colorBlend.pAttachments = blendAttachments;

    VkDynamicState dynamicStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState = {};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = gbufferVert_;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = gbufferFrag_;
    stages[1].pName = "main";

    VkGraphicsPipelineCreateInfo sceneCi = {};
    sceneCi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    sceneCi.stageCount = 2;
    sceneCi.pStages = stages;
    sceneCi.pVertexInputState = &vertexInput;
    sceneCi.pInputAssemblyState = &inputAssembly;
    sceneCi.pViewportState = &viewportState;
    sceneCi.pRasterizationState = &rasterizer;
    sceneCi.pMultisampleState = &multisample;
    sceneCi.pDepthStencilState = &depthStencil;
    sceneCi.pColorBlendState = &colorBlend;
    sceneCi.pDynamicState = &dynamicState;
    sceneCi.layout = scenePipelineLayout_;

    // GBuffer pipeline (upscaler input path): albedo/normal/material/emissive/motion + depth.
    VkPipelineRenderingCreateInfo gbRendering = {};
    gbRendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    VkFormat gbColorFormats[5] = {deferred::kAlbedoFormat, deferred::kNormalFormat,
                                  deferred::kMaterialFormat, deferred::kEmissiveFormat,
                                  deferred::kMotionFormat};
    gbRendering.colorAttachmentCount = 5;
    gbRendering.pColorAttachmentFormats = gbColorFormats;
    gbRendering.depthAttachmentFormat = deferred::kDepthFormat;
    sceneCi.pNext = &gbRendering;
    if (createGraphicsPipeline(ctx, sceneCi, gbufferPipeline_) != VK_SUCCESS)
        return false;

    // GT GBuffer pipeline: same minus the motion attachment.
    VkPipelineRenderingCreateInfo gtRendering = {};
    gtRendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    gtRendering.colorAttachmentCount = 4;
    gtRendering.pColorAttachmentFormats = gbColorFormats;
    gtRendering.depthAttachmentFormat = deferred::kDepthFormat;
    sceneCi.pNext = &gtRendering;
    colorBlend.attachmentCount = 4;
    stages[1].module = gbufferGtFrag_; // GT shader has no motion output
    if (createGraphicsPipeline(ctx, sceneCi, gbufferGtPipeline_) != VK_SUCCESS)
        return false;

    // Deferred lighting pipeline: fullscreen triangle, GBuffer + IBL -> HDR color.
    VkPipelineShaderStageCreateInfo lightingStages[2] = {};
    lightingStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    lightingStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    lightingStages[0].module = fullscreenVert_;
    lightingStages[0].pName = "main";
    lightingStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    lightingStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    lightingStages[1].module = lightingFrag_;
    lightingStages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo emptyVertexInput = {};
    emptyVertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineDepthStencilStateCreateInfo noDepth = {};
    noDepth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

    VkPipelineColorBlendAttachmentState lightingBlend = {};
    lightingBlend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                   VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo lightingColorBlend = {};
    lightingColorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    lightingColorBlend.attachmentCount = 1;
    lightingColorBlend.pAttachments = &lightingBlend;

    VkGraphicsPipelineCreateInfo lightingCi = {};
    lightingCi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    lightingCi.stageCount = 2;
    lightingCi.pStages = lightingStages;
    lightingCi.pVertexInputState = &emptyVertexInput;
    lightingCi.pInputAssemblyState = &inputAssembly;
    lightingCi.pViewportState = &viewportState;
    lightingCi.pRasterizationState = &rasterizer;
    lightingCi.pMultisampleState = &multisample;
    lightingCi.pDepthStencilState = &noDepth;
    lightingCi.pColorBlendState = &lightingColorBlend;
    lightingCi.pDynamicState = &dynamicState;
    lightingCi.layout = lightingPipelineLayout_;

    VkPipelineRenderingCreateInfo lightingRendering = {};
    lightingRendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    lightingRendering.colorAttachmentCount = 1;
    lightingRendering.pColorAttachmentFormats = &deferred::kHdrColorFormat;
    lightingCi.pNext = &lightingRendering;
    if (createGraphicsPipeline(ctx, lightingCi, lightingPipeline_) != VK_SUCCESS)
        return false;

    // --- Transparency pass ---------------------------------------------------
    VkDescriptorSetLayout transparentLayouts[3] = {sceneSetLayout_, textureSetLayout_,
                                                   transparentSetLayout_};
    VkPipelineLayoutCreateInfo transparentLayoutCi = {};
    transparentLayoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    transparentLayoutCi.setLayoutCount = 3;
    transparentLayoutCi.pSetLayouts = transparentLayouts;
    transparentLayoutCi.pushConstantRangeCount = 1;
    transparentLayoutCi.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(ctx.device, &transparentLayoutCi, nullptr,
                               &transparentPipelineLayout_) != VK_SUCCESS)
        return false;

    // Depth-tested against the opaque GBuffer depth, never written.
    VkPipelineDepthStencilStateCreateInfo transparentDepth = {};
    transparentDepth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    transparentDepth.depthTestEnable = VK_TRUE;
    transparentDepth.depthWriteEnable = VK_FALSE;
    transparentDepth.depthCompareOp = VK_COMPARE_OP_LESS;

    // att0 = alpha over; att1 = motion overwrite; att2 = additive coverage.
    VkPipelineColorBlendAttachmentState transparentBlend[3] = {};
    for (auto& b : transparentBlend) {
        b.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    }
    transparentBlend[0].blendEnable = VK_TRUE;
    // Premultiplied: RGB already contains unscaled specular (SSR / highlights)
    // so SRC_ALPHA would crush shop-window reflections.
    transparentBlend[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    transparentBlend[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    transparentBlend[0].colorBlendOp = VK_BLEND_OP_ADD;
    transparentBlend[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    transparentBlend[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    transparentBlend[0].alphaBlendOp = VK_BLEND_OP_ADD;
    transparentBlend[2].blendEnable = VK_TRUE;
    transparentBlend[2].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    transparentBlend[2].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    transparentBlend[2].colorBlendOp = VK_BLEND_OP_ADD;
    transparentBlend[2].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    transparentBlend[2].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    transparentBlend[2].alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo transparentColorBlend = {};
    transparentColorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    transparentColorBlend.attachmentCount = 3;
    transparentColorBlend.pAttachments = transparentBlend;

    VkPipelineShaderStageCreateInfo transparentStages[2] = {};
    transparentStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    transparentStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    transparentStages[0].module = transparentVert_;
    transparentStages[0].pName = "main";
    transparentStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    transparentStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    transparentStages[1].module = transparentFrag_;
    transparentStages[1].pName = "main";

    VkGraphicsPipelineCreateInfo transparentCi = {};
    transparentCi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    transparentCi.stageCount = 2;
    transparentCi.pStages = transparentStages;
    transparentCi.pVertexInputState = &vertexInput;
    transparentCi.pInputAssemblyState = &inputAssembly;
    transparentCi.pViewportState = &viewportState;
    transparentCi.pRasterizationState = &rasterizer;
    transparentCi.pMultisampleState = &multisample;
    transparentCi.pDepthStencilState = &transparentDepth;
    transparentCi.pColorBlendState = &transparentColorBlend;
    transparentCi.pDynamicState = &dynamicState;
    transparentCi.layout = transparentPipelineLayout_;

    // LR variant: color + motion + reactive mask.
    VkPipelineRenderingCreateInfo transparentRendering = {};
    transparentRendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    VkFormat transparentFormats[3] = {deferred::kHdrColorFormat, deferred::kMotionFormat,
                                      deferred::kReactiveFormat};
    transparentRendering.colorAttachmentCount = 3;
    transparentRendering.pColorAttachmentFormats = transparentFormats;
    transparentRendering.depthAttachmentFormat = deferred::kDepthFormat;
    transparentCi.pNext = &transparentRendering;
    if (createGraphicsPipeline(ctx, transparentCi, transparentPipeline_) != VK_SUCCESS)
        return false;

    // GT variant: blended color only.
    VkPipelineRenderingCreateInfo transparentGtRendering = {};
    transparentGtRendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    transparentGtRendering.colorAttachmentCount = 1;
    transparentGtRendering.pColorAttachmentFormats = &deferred::kHdrColorFormat;
    transparentGtRendering.depthAttachmentFormat = deferred::kDepthFormat;
    transparentCi.pNext = &transparentGtRendering;
    transparentColorBlend.attachmentCount = 1;
    transparentStages[1].module = transparentGtFrag_;
    if (createGraphicsPipeline(ctx, transparentCi, transparentGtPipeline_) != VK_SUCCESS)
        return false;

    // --- GTAO compute passes (main + 5x5 bilateral denoise) --------------------
    VkPushConstantRange ssaoPushRange = {};
    ssaoPushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    ssaoPushRange.offset = 0;
    ssaoPushRange.size = sizeof(SsaoPush);
    VkPipelineLayoutCreateInfo ssaoLayoutCi = {};
    ssaoLayoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ssaoLayoutCi.setLayoutCount = 1;
    ssaoLayoutCi.pSetLayouts = &ssaoSetLayout_;
    ssaoLayoutCi.pushConstantRangeCount = 1;
    ssaoLayoutCi.pPushConstantRanges = &ssaoPushRange;
    if (vkCreatePipelineLayout(ctx.device, &ssaoLayoutCi, nullptr, &ssaoPipelineLayout_) !=
        VK_SUCCESS)
        return false;

    VkPipelineLayoutCreateInfo ssaoBlurLayoutCi = {};
    ssaoBlurLayoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ssaoBlurLayoutCi.setLayoutCount = 1;
    ssaoBlurLayoutCi.pSetLayouts = &ssaoBlurSetLayout_;
    if (vkCreatePipelineLayout(ctx.device, &ssaoBlurLayoutCi, nullptr,
                               &ssaoBlurPipelineLayout_) != VK_SUCCESS)
        return false;

    VkComputePipelineCreateInfo ssaoCi = {};
    ssaoCi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ssaoCi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ssaoCi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ssaoCi.stage.module = ssaoComp_;
    ssaoCi.stage.pName = "main";
    ssaoCi.layout = ssaoPipelineLayout_;
    if (createComputePipeline(ctx, ssaoCi, ssaoPipeline_) != VK_SUCCESS)
        return false;
    ssaoCi.stage.module = ssaoBlurComp_;
    ssaoCi.layout = ssaoBlurPipelineLayout_;
    if (createComputePipeline(ctx, ssaoCi, ssaoBlurPipeline_) != VK_SUCCESS)
        return false;

    // --- Bloom (reuses ssaoBlur set layout: sampler + storage) ----------------
    VkPushConstantRange bloomPushRange = {};
    bloomPushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bloomPushRange.offset = 0;
    bloomPushRange.size = sizeof(BloomPush);
    VkPipelineLayoutCreateInfo bloomLayoutCi = {};
    bloomLayoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    bloomLayoutCi.setLayoutCount = 1;
    bloomLayoutCi.pSetLayouts = &ssaoBlurSetLayout_;
    bloomLayoutCi.pushConstantRangeCount = 1;
    bloomLayoutCi.pPushConstantRanges = &bloomPushRange;
    if (vkCreatePipelineLayout(ctx.device, &bloomLayoutCi, nullptr, &bloomPipelineLayout_) !=
        VK_SUCCESS)
        return false;
    VkComputePipelineCreateInfo bloomCi = {};
    bloomCi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    bloomCi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    bloomCi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    bloomCi.stage.pName = "main";
    bloomCi.layout = bloomPipelineLayout_;
    bloomCi.stage.module = bloomExtractComp_;
    if (createComputePipeline(ctx, bloomCi, bloomExtractPipeline_) != VK_SUCCESS)
        return false;
    bloomCi.stage.module = bloomBlurComp_;
    if (createComputePipeline(ctx, bloomCi, bloomBlurPipeline_) != VK_SUCCESS)
        return false;
    bloomCi.stage.module = bloomCompositeComp_;
    if (createComputePipeline(ctx, bloomCi, bloomCompositePipeline_) != VK_SUCCESS)
        return false;

    // --- CSM shadow depth pass ------------------------------------------------
    // Depth-only rendering into one cascade layer at a time.  Reuses the scene
    // pipeline layout (set0 = scene/material UBO, set1 = texture array); the
    // ShadowPush block (128 B) fits the layout's 192-byte push range.  Depth
    // bias is a dynamic state so the values can come from LightingUBO
    // (shadowParams.xy) per frame without a pipeline rebuild.
    VkDynamicState shadowDynamicStates[3] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                             VK_DYNAMIC_STATE_DEPTH_BIAS};
    VkPipelineDynamicStateCreateInfo shadowDynamicState = {};
    shadowDynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    shadowDynamicState.dynamicStateCount = 3;
    shadowDynamicState.pDynamicStates = shadowDynamicStates;

    VkPipelineRasterizationStateCreateInfo shadowRasterizer = rasterizer;
    shadowRasterizer.depthBiasEnable = VK_TRUE;

    VkPipelineDepthStencilStateCreateInfo shadowDepthStencil = depthStencil; // test+write, LESS

    VkPipelineColorBlendStateCreateInfo shadowColorBlend = {};
    shadowColorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    shadowColorBlend.attachmentCount = 0; // depth-only

    VkPipelineShaderStageCreateInfo shadowStages[2] = {};
    shadowStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shadowStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shadowStages[0].module = shadowDepthVert_;
    shadowStages[0].pName = "main";
    shadowStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shadowStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shadowStages[1].module = shadowDepthFrag_;
    shadowStages[1].pName = "main";

    VkGraphicsPipelineCreateInfo shadowCi = {};
    shadowCi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    shadowCi.stageCount = 2;
    shadowCi.pStages = shadowStages;
    shadowCi.pVertexInputState = &vertexInput;
    shadowCi.pInputAssemblyState = &inputAssembly;
    shadowCi.pViewportState = &viewportState;
    shadowCi.pRasterizationState = &shadowRasterizer;
    shadowCi.pMultisampleState = &multisample;
    shadowCi.pDepthStencilState = &shadowDepthStencil;
    shadowCi.pColorBlendState = &shadowColorBlend;
    shadowCi.pDynamicState = &shadowDynamicState;
    shadowCi.layout = scenePipelineLayout_;

    VkPipelineRenderingCreateInfo shadowRendering = {};
    shadowRendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    shadowRendering.colorAttachmentCount = 0;
    shadowRendering.depthAttachmentFormat = deferred::kDepthFormat;
    shadowCi.pNext = &shadowRendering;
    return createGraphicsPipeline(ctx, shadowCi, shadowPipeline_) == VK_SUCCESS;
}

void DeferredCore::destroy(const VulkanContext& ctx) {
    if (!ctx.device) return;
    if (gbufferPipeline_) { vkDestroyPipeline(ctx.device, gbufferPipeline_, nullptr); gbufferPipeline_ = VK_NULL_HANDLE; }
    if (gbufferGtPipeline_) { vkDestroyPipeline(ctx.device, gbufferGtPipeline_, nullptr); gbufferGtPipeline_ = VK_NULL_HANDLE; }
    if (lightingPipeline_) { vkDestroyPipeline(ctx.device, lightingPipeline_, nullptr); lightingPipeline_ = VK_NULL_HANDLE; }
    if (transparentPipeline_) { vkDestroyPipeline(ctx.device, transparentPipeline_, nullptr); transparentPipeline_ = VK_NULL_HANDLE; }
    if (transparentGtPipeline_) { vkDestroyPipeline(ctx.device, transparentGtPipeline_, nullptr); transparentGtPipeline_ = VK_NULL_HANDLE; }
    if (ssaoPipeline_) { vkDestroyPipeline(ctx.device, ssaoPipeline_, nullptr); ssaoPipeline_ = VK_NULL_HANDLE; }
    if (ssaoBlurPipeline_) { vkDestroyPipeline(ctx.device, ssaoBlurPipeline_, nullptr); ssaoBlurPipeline_ = VK_NULL_HANDLE; }
    if (bloomExtractPipeline_) { vkDestroyPipeline(ctx.device, bloomExtractPipeline_, nullptr); bloomExtractPipeline_ = VK_NULL_HANDLE; }
    if (bloomBlurPipeline_) { vkDestroyPipeline(ctx.device, bloomBlurPipeline_, nullptr); bloomBlurPipeline_ = VK_NULL_HANDLE; }
    if (bloomCompositePipeline_) { vkDestroyPipeline(ctx.device, bloomCompositePipeline_, nullptr); bloomCompositePipeline_ = VK_NULL_HANDLE; }
    if (shadowPipeline_) { vkDestroyPipeline(ctx.device, shadowPipeline_, nullptr); shadowPipeline_ = VK_NULL_HANDLE; }
    if (scenePipelineLayout_) { vkDestroyPipelineLayout(ctx.device, scenePipelineLayout_, nullptr); scenePipelineLayout_ = VK_NULL_HANDLE; }
    if (lightingPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, lightingPipelineLayout_, nullptr); lightingPipelineLayout_ = VK_NULL_HANDLE; }
    if (transparentPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, transparentPipelineLayout_, nullptr); transparentPipelineLayout_ = VK_NULL_HANDLE; }
    if (ssaoPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, ssaoPipelineLayout_, nullptr); ssaoPipelineLayout_ = VK_NULL_HANDLE; }
    if (ssaoBlurPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, ssaoBlurPipelineLayout_, nullptr); ssaoBlurPipelineLayout_ = VK_NULL_HANDLE; }
    if (bloomPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, bloomPipelineLayout_, nullptr); bloomPipelineLayout_ = VK_NULL_HANDLE; }
    if (gbufferVert_) { vkDestroyShaderModule(ctx.device, gbufferVert_, nullptr); gbufferVert_ = VK_NULL_HANDLE; }
    if (gbufferFrag_) { vkDestroyShaderModule(ctx.device, gbufferFrag_, nullptr); gbufferFrag_ = VK_NULL_HANDLE; }
    if (gbufferGtFrag_) { vkDestroyShaderModule(ctx.device, gbufferGtFrag_, nullptr); gbufferGtFrag_ = VK_NULL_HANDLE; }
    if (lightingFrag_) { vkDestroyShaderModule(ctx.device, lightingFrag_, nullptr); lightingFrag_ = VK_NULL_HANDLE; }
    if (fullscreenVert_) { vkDestroyShaderModule(ctx.device, fullscreenVert_, nullptr); fullscreenVert_ = VK_NULL_HANDLE; }
    if (transparentVert_) { vkDestroyShaderModule(ctx.device, transparentVert_, nullptr); transparentVert_ = VK_NULL_HANDLE; }
    if (transparentFrag_) { vkDestroyShaderModule(ctx.device, transparentFrag_, nullptr); transparentFrag_ = VK_NULL_HANDLE; }
    if (transparentGtFrag_) { vkDestroyShaderModule(ctx.device, transparentGtFrag_, nullptr); transparentGtFrag_ = VK_NULL_HANDLE; }
    if (ssaoComp_) { vkDestroyShaderModule(ctx.device, ssaoComp_, nullptr); ssaoComp_ = VK_NULL_HANDLE; }
    if (ssaoBlurComp_) { vkDestroyShaderModule(ctx.device, ssaoBlurComp_, nullptr); ssaoBlurComp_ = VK_NULL_HANDLE; }
    if (bloomExtractComp_) { vkDestroyShaderModule(ctx.device, bloomExtractComp_, nullptr); bloomExtractComp_ = VK_NULL_HANDLE; }
    if (bloomBlurComp_) { vkDestroyShaderModule(ctx.device, bloomBlurComp_, nullptr); bloomBlurComp_ = VK_NULL_HANDLE; }
    if (bloomCompositeComp_) { vkDestroyShaderModule(ctx.device, bloomCompositeComp_, nullptr); bloomCompositeComp_ = VK_NULL_HANDLE; }
    if (shadowDepthVert_) { vkDestroyShaderModule(ctx.device, shadowDepthVert_, nullptr); shadowDepthVert_ = VK_NULL_HANDLE; }
    if (shadowDepthFrag_) { vkDestroyShaderModule(ctx.device, shadowDepthFrag_, nullptr); shadowDepthFrag_ = VK_NULL_HANDLE; }
    if (sceneSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, sceneSetLayout_, nullptr); sceneSetLayout_ = VK_NULL_HANDLE; }
    if (textureSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, textureSetLayout_, nullptr); textureSetLayout_ = VK_NULL_HANDLE; }
    if (lightingSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, lightingSetLayout_, nullptr); lightingSetLayout_ = VK_NULL_HANDLE; }
    if (transparentSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, transparentSetLayout_, nullptr); transparentSetLayout_ = VK_NULL_HANDLE; }
    if (ssaoSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, ssaoSetLayout_, nullptr); ssaoSetLayout_ = VK_NULL_HANDLE; }
    if (ssaoBlurSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, ssaoBlurSetLayout_, nullptr); ssaoBlurSetLayout_ = VK_NULL_HANDLE; }
    if (textureSampler_) { vkDestroySampler(ctx.device, textureSampler_, nullptr); textureSampler_ = VK_NULL_HANDLE; }
    if (gbufferSampler_) { vkDestroySampler(ctx.device, gbufferSampler_, nullptr); gbufferSampler_ = VK_NULL_HANDLE; }
    if (shadowSampler_) { vkDestroySampler(ctx.device, shadowSampler_, nullptr); shadowSampler_ = VK_NULL_HANDLE; }
    ibl_.destroy(ctx);
}

void DeferredCore::fillSceneUBO(SceneUBO& out, const Scene& scene, const Camera& camera,
                                const Mat4& view, const Mat4& proj, const Mat4& projJittered,
                                const Mat4& prevViewProj, uint32_t renderW, uint32_t renderH,
                                float jitterX, float jitterY, bool jitter) const {
    const Mat4 viewProj = Mat4::multiply(projJittered, view);
    const Mat4 viewProjNoJitter = Mat4::multiply(proj, view);
    std::memcpy(out.viewProj, viewProj.m, sizeof(out.viewProj));
    std::memcpy(out.viewProjNoJitter, viewProjNoJitter.m, sizeof(out.viewProjNoJitter));
    std::memcpy(out.prevViewProj, prevViewProj.m, sizeof(out.prevViewProj));

    out.cameraPos[0] = camera.position.x;
    out.cameraPos[1] = camera.position.y;
    out.cameraPos[2] = camera.position.z;
    out.cameraPos[3] = 1.f;

    // Light data no longer lives here; the forward transparency pass reads the
    // shared LightingUBO array filled by fillLightingUBO.
    out.ambient[0] = 0.08f; out.ambient[1] = 0.08f; out.ambient[2] = 0.10f; out.ambient[3] = 1.f;
    out.renderSizeJitter[0] = static_cast<float>(renderW);
    out.renderSizeJitter[1] = static_cast<float>(renderH);
    out.renderSizeJitter[2] = jitter ? jitterX : 0.f;
    out.renderSizeJitter[3] = jitter ? jitterY : 0.f;
}

void DeferredCore::fillLightingUBO(LightingUBO& out, const Scene& scene, const Camera& camera,
                                   const Mat4& invViewProj,
                                   const std::vector<Light>* overrideLights,
                                   const ShadowFrame* shadow, float iblIntensity) const {
    std::memcpy(out.invViewProj, invViewProj.m, sizeof(out.invViewProj));

    out.cameraPos[0] = camera.position.x;
    out.cameraPos[1] = camera.position.y;
    out.cameraPos[2] = camera.position.z;
    out.cameraPos[3] = 1.f;

    // Pack into the fixed-size GPU array: the shadowed sun (if any) is always
    // slot 0 so CSM lightIndex remaps cleanly; remaining lights are scored by
    // intensity / distance² from the camera so the nearest lanterns survive
    // truncation at kMaxLights.
    const std::vector<Light>& lights =
        overrideLights ? *overrideLights
                       : (scene.lights.empty() ? defaultLights() : scene.lights);

    int sunSrc = -1;
    if (shadow && shadow->lightIndex >= 0 &&
        static_cast<size_t>(shadow->lightIndex) < lights.size() &&
        lights[static_cast<size_t>(shadow->lightIndex)].type == LightType::Directional) {
        sunSrc = shadow->lightIndex;
    } else {
        for (int i = 0; i < static_cast<int>(lights.size()); ++i) {
            if (lights[static_cast<size_t>(i)].type == LightType::Directional &&
                lights[static_cast<size_t>(i)].castShadow) {
                sunSrc = i;
                break;
            }
        }
        if (sunSrc < 0) {
            for (int i = 0; i < static_cast<int>(lights.size()); ++i) {
                if (lights[static_cast<size_t>(i)].type == LightType::Directional) {
                    sunSrc = i;
                    break;
                }
            }
        }
    }

    struct Scored {
        int index;
        float score;
    };
    std::vector<Scored> rest;
    rest.reserve(lights.size());
    const Vec3 camPos = camera.position;
    for (int i = 0; i < static_cast<int>(lights.size()); ++i) {
        if (i == sunSrc) continue;
        const Light& l = lights[static_cast<size_t>(i)];
        float score = l.intensity;
        if (l.type == LightType::Point) {
            const Vec3 d = l.positionOrDirection - camPos;
            const float dist2 = std::max(dot(d, d), 1.f);
            score = l.intensity / dist2;
            if (l.range > 0.f && std::sqrt(dist2) > l.range) score *= 0.01f;
        } else {
            score = l.intensity * 1000.f; // leftover directionals stay near the front
        }
        rest.push_back({i, score});
    }
    std::sort(rest.begin(), rest.end(),
              [](const Scored& a, const Scored& b) { return a.score > b.score; });

    std::vector<int> order;
    order.reserve(kMaxLights);
    if (sunSrc >= 0) order.push_back(sunSrc);
    for (const Scored& s : rest) {
        if (order.size() >= kMaxLights) break;
        order.push_back(s.index);
    }

    const uint32_t count = static_cast<uint32_t>(order.size());
    std::memset(out.lights, 0, sizeof(out.lights)); // deterministic unused slots
    for (uint32_t i = 0; i < count; ++i) {
        const Light& l = lights[static_cast<size_t>(order[static_cast<size_t>(i)])];
        LightGPU& g = out.lights[i];
        g.posOrDir[0] = l.positionOrDirection.x;
        g.posOrDir[1] = l.positionOrDirection.y;
        g.posOrDir[2] = l.positionOrDirection.z;
        g.posOrDir[3] = static_cast<float>(l.type);
        g.color[0] = l.color.x;
        g.color[1] = l.color.y;
        g.color[2] = l.color.z;
        // Intensities are scaled by PI so Hammon's single-scatter 1/PI term
        // matches the brightness of the legacy forward pass.
        g.color[3] = l.intensity * 3.14159265f;
        g.params[0] = l.range;
        g.params[1] = l.castShadow ? 1.f : 0.f;
        g.params[2] = 0.f;
        g.params[3] = 0.f;
    }
    out.lightCounts[0] = static_cast<float>(count);
    out.lightCounts[1] = 0.f;
    out.lightCounts[2] = 0.f;
    out.lightCounts[3] = 0.f;

    out.ambient[0] = 0.08f; out.ambient[1] = 0.08f; out.ambient[2] = 0.10f; out.ambient[3] = 1.f;
    out.iblParams[0] = iblIntensity;
    out.iblParams[1] = static_cast<float>(ibl_.prefilterMaxLod);
    out.iblParams[2] = 1.f; // skybox enabled
    out.iblParams[3] = 0.f;

    // Camera forward is needed by the shaders to convert a world position to
    // view-space depth for cascade selection; fill it even with shadows off.
    out.viewForward[0] = camera.forward.x;
    out.viewForward[1] = camera.forward.y;
    out.viewForward[2] = camera.forward.z;

    // The rasterizer depth-bias values are mirrored here so hosts can feed the
    // same constants to vkCmdSetDepthBias in recordShadowPass.
    out.shadowParams[0] = kShadowDepthBiasConstant;
    out.shadowParams[1] = kShadowDepthBiasSlope;
    if (shadow) {
        for (uint32_t i = 0; i < kShadowCascadeCount; ++i) {
            std::memcpy(out.cascadeVp[i], shadow->cascadeVp[i].m, sizeof(out.cascadeVp[i]));
            out.cascadeSplits[i] = shadow->splitDepth[i];
        }
        out.shadowParams[2] = 1.f; // shadows enabled
        out.shadowParams[3] = shadow->debugCascades ? 1.f : 0.f;
        // Packed index: the sun is always slot 0 when present.
        out.viewForward[3] = (sunSrc >= 0) ? 0.f : -1.f;
    } else {
        // Deterministic disabled state: identity VPs, infinite splits; the
        // shaders short-circuit on shadowParams.z before touching the map.
        const Mat4 identity = Mat4::identity();
        for (uint32_t i = 0; i < kShadowCascadeCount; ++i) {
            std::memcpy(out.cascadeVp[i], identity.m, sizeof(out.cascadeVp[i]));
            out.cascadeSplits[i] = 1e9f;
        }
        out.shadowParams[2] = 0.f;
        out.shadowParams[3] = 0.f;
        out.viewForward[3] = -1.f;
    }
}

bool DeferredCore::createMaterialUbo(const VulkanContext& ctx, const Scene& scene,
                                     VkBuffer& buffer, VmaAllocation& memory,
                                     uint32_t& stride) const {
    stride = alignUp(sizeof(MaterialUBO), static_cast<uint32_t>(ctx.minUniformBufferOffsetAlignment));
    const VkDeviceSize size = static_cast<VkDeviceSize>(stride) * scene.materials.size();
    if (createBuffer(ctx, size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     buffer, memory) != VK_SUCCESS)
        return false;
    void* mapped = nullptr;
    vmaMapMemory(ctx.allocator, memory, &mapped);
    auto* bytes = static_cast<uint8_t*>(mapped);
    for (size_t i = 0; i < scene.materials.size(); ++i) {
        MaterialUBO ubo;
        const Material& m = scene.materials[i];
        ubo.baseColor[0] = m.baseColor.x;
        ubo.baseColor[1] = m.baseColor.y;
        ubo.baseColor[2] = m.baseColor.z;
        ubo.baseColor[3] = m.baseColor.w;
        ubo.factors[0] = m.metallic;
        ubo.factors[1] = m.roughness;
        ubo.factors[2] = m.occlusionStrength;
        ubo.factors[3] = m.alphaCutoff;
        ubo.emissive[0] = m.emissiveFactor.x;
        ubo.emissive[1] = m.emissiveFactor.y;
        ubo.emissive[2] = m.emissiveFactor.z;
        ubo.emissive[3] = 0.f;
        ubo.tex0[0] = static_cast<float>(m.texIndex);
        ubo.tex0[1] = static_cast<float>(m.normalTexIndex);
        ubo.tex0[2] = static_cast<float>(m.mrTexIndex);
        ubo.tex0[3] = static_cast<float>(m.aoTexIndex);
        ubo.tex1[0] = static_cast<float>(m.emissiveTexIndex);
        ubo.tex1[1] = ubo.tex1[2] = ubo.tex1[3] = -1.f;
        std::memcpy(bytes + i * stride, &ubo, sizeof(ubo));
    }
    vmaUnmapMemory(ctx.allocator, memory);
    return true;
}

void DeferredCore::writeTextureSet(const VulkanContext& ctx, VkDescriptorSet set,
                                   const Scene& scene) const {
    // A glTF without images produces no textures; every material then has
    // texIndex == -1 and the shaders never sample this array, so nothing to bind.
    if (scene.textures.empty()) return;
    if (scene.textures.size() > deferred::kMaxTextures) {
        std::fprintf(stderr,
                     "warning: scene has %zu textures, exceeding kMaxTextures=%u; "
                     "materials referencing slots beyond the limit will sample texture 0\n",
                     scene.textures.size(), deferred::kMaxTextures);
    }
    const uint32_t textureCount =
        std::min<uint32_t>(deferred::kMaxTextures, static_cast<uint32_t>(scene.textures.size()));
    // Bind the full array; unused slots alias texture 0 (they are never
    // sampled because materials with texIndex == -1 skip sampling).
    std::vector<VkDescriptorImageInfo> infos(deferred::kMaxTextures);
    for (uint32_t i = 0; i < deferred::kMaxTextures; ++i) {
        infos[i].sampler = textureSampler_;
        infos[i].imageView = scene.textures[i < textureCount ? i : 0].view;
        infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = 0;
    write.descriptorCount = deferred::kMaxTextures;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = infos.data();
    vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
}

void DeferredCore::writeLightingSet(const VulkanContext& ctx, VkDescriptorSet set,
                                    VkBuffer lightingUbo, VkImageView albedo, VkImageView normal,
                                    VkImageView material, VkImageView emissive,
                                    VkImageView depth, VkImageView ssao, VkImageView shadow) const {
    VkDescriptorBufferInfo lightBuf = {};
    lightBuf.buffer = lightingUbo;
    lightBuf.offset = 0;
    lightBuf.range = sizeof(LightingUBO);

    VkDescriptorImageInfo img[11] = {};
    const VkImageView gb[5] = {albedo, normal, material, emissive, depth};
    for (int k = 0; k < 5; ++k) {
        img[k].sampler = gbufferSampler_;
        img[k].imageView = gb[k];
        img[k].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    img[5].sampler = ibl_.cubeSampler;
    img[5].imageView = ibl_.irradianceView;
    img[5].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    img[6].sampler = ibl_.cubeSampler;
    img[6].imageView = ibl_.prefilterView;
    img[6].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    img[7].sampler = ibl_.lutSampler;
    img[7].imageView = ibl_.brdfLutView;
    img[7].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    img[8].sampler = ibl_.cubeSampler;
    img[8].imageView = ibl_.envView;
    img[8].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    img[9].sampler = gbufferSampler_;
    img[9].imageView = ssao;
    img[9].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    img[10].sampler = shadowSampler_;
    img[10].imageView = shadow;
    img[10].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet w[12] = {};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = set;
    w[0].dstBinding = 0;
    w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w[0].pBufferInfo = &lightBuf;
    const uint32_t samplerCount = shadow ? 11u : 10u; // skip binding 11 without a shadow map
    for (uint32_t k = 0; k < samplerCount; ++k) {
        w[k + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[k + 1].dstSet = set;
        w[k + 1].dstBinding = k + 1;
        w[k + 1].descriptorCount = 1;
        w[k + 1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[k + 1].pImageInfo = &img[k];
    }
    vkUpdateDescriptorSets(ctx.device, 1 + samplerCount, w, 0, nullptr);
}

void DeferredCore::recordGBufferDraws(VkCommandBuffer cmd, const Scene& scene, bool gtPass,
                                      VkDescriptorSet sceneSet, VkDescriptorSet textureSet,
                                      uint32_t materialStride, uint32_t width, uint32_t height,
                                      const Mat4& cullViewProj) const {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      gtPass ? gbufferGtPipeline_ : gbufferPipeline_);
    VkViewport viewport = {0.f, 0.f, static_cast<float>(width), static_cast<float>(height), 0.f, 1.f};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor = {{0, 0}, {width, height}};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipelineLayout_, 1, 1,
                            &textureSet, 0, nullptr);

    // Frustum for CPU culling (un-jittered; jitter is sub-pixel).
    const Frustum frustum = extractFrustum(cullViewProj);

    // Bind the scene-wide merged buffers once; draws address into them with
    // per-mesh firstIndex/vertexOffset.
    const VkDeviceSize zeroOffset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &scene.mergedVertexBuffer, &zeroOffset);
    vkCmdBindIndexBuffer(cmd, scene.mergedIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

    // Instances are sorted by (material, mesh) at load time, so state changes
    // collapse to one descriptor bind per material run.
    uint32_t lastMaterial = UINT32_MAX;
    for (const auto& inst : scene.instances) {
        if (scene.materials[inst.materialIndex].blend) continue; // transparency pass draws these
        if (!aabbIntersectsFrustum(frustum, inst.aabbMin, inst.aabbMax)) continue;

        ScenePush push;
        std::memcpy(push.model, inst.model.m, sizeof(push.model));
        std::memcpy(push.prevModel, inst.prevModel.m, sizeof(push.prevModel));
        std::memcpy(push.normalModel, inst.normalModel.m, sizeof(push.normalModel));
        vkCmdPushConstants(cmd, scenePipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push),
                           &push);

        if (inst.materialIndex != lastMaterial) {
            lastMaterial = inst.materialIndex;
            const uint32_t dynOffset = inst.materialIndex * materialStride;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipelineLayout_, 0, 1,
                                    &sceneSet, 1, &dynOffset);
        }

        const Mesh& mesh = scene.meshes[inst.meshIndex];
        vkCmdDrawIndexed(cmd, mesh.indexCount, 1, mesh.firstIndex, mesh.vertexOffset, 0);
    }
}

void DeferredCore::recordLightingPass(VkCommandBuffer cmd, VkDescriptorSet lightingSet,
                                      VkImageView target, uint32_t width, uint32_t height) const {
    VkRenderingAttachmentInfo color =
        makeColorAttachment(target, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_ATTACHMENT_LOAD_OP_CLEAR);
    VkRenderingInfo ri = {};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea = {{0, 0}, {width, height}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &color;
    vkCmdBeginRendering(cmd, &ri);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lightingPipeline_);
    VkViewport viewport = {0.f, 0.f, static_cast<float>(width), static_cast<float>(height), 0.f, 1.f};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor = {{0, 0}, {width, height}};
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lightingPipelineLayout_, 0, 1,
                            &lightingSet, 0, nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);
}

bool DeferredCore::sceneHasTransparency(const Scene& scene) const {
    for (const auto& m : scene.materials)
        if (m.blend) return true;
    return false;
}

void DeferredCore::writeTransparentSet(const VulkanContext& ctx, VkDescriptorSet set,
                                       VkBuffer lightingUbo, VkImageView ssao,
                                       VkImageView shadow, VkImageView ssrColor,
                                       VkImageView ssrDepth) const {
    VkDescriptorBufferInfo lightBuf = {};
    lightBuf.buffer = lightingUbo;
    lightBuf.offset = 0;
    lightBuf.range = sizeof(LightingUBO);

    VkDescriptorImageInfo img[7] = {};
    img[0].sampler = ibl_.cubeSampler;
    img[0].imageView = ibl_.irradianceView;
    img[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    img[1].sampler = ibl_.cubeSampler;
    img[1].imageView = ibl_.prefilterView;
    img[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    img[2].sampler = ibl_.lutSampler;
    img[2].imageView = ibl_.brdfLutView;
    img[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    img[3].sampler = gbufferSampler_;
    img[3].imageView = ssao;
    img[3].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    img[4].sampler = shadowSampler_;
    img[4].imageView = shadow;
    img[4].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    img[5].sampler = gbufferSampler_;
    img[5].imageView = ssrColor;
    img[5].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    img[6].sampler = gbufferSampler_;
    img[6].imageView = ssrDepth;
    img[6].imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet w[8] = {};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = set;
    w[0].dstBinding = 0;
    w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w[0].pBufferInfo = &lightBuf;
    // Bindings 1-4 always; 5 (shadow) skipped if no map; 6-7 SSR always.
    uint32_t n = 1;
    for (uint32_t k = 0; k < 4; ++k) {
        w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[n].dstSet = set;
        w[n].dstBinding = k + 1;
        w[n].descriptorCount = 1;
        w[n].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[n].pImageInfo = &img[k];
        ++n;
    }
    if (shadow) {
        w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[n].dstSet = set;
        w[n].dstBinding = 5;
        w[n].descriptorCount = 1;
        w[n].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[n].pImageInfo = &img[4];
        ++n;
    }
    for (uint32_t k = 0; k < 2; ++k) {
        w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[n].dstSet = set;
        w[n].dstBinding = 6 + k;
        w[n].descriptorCount = 1;
        w[n].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[n].pImageInfo = &img[5 + k];
        ++n;
    }
    vkUpdateDescriptorSets(ctx.device, n, w, 0, nullptr);
}

void DeferredCore::writeSsaoSet(const VulkanContext& ctx, VkDescriptorSet set, VkImageView depth,
                                VkImageView normal, VkImageView aoRaw) const {
    VkDescriptorImageInfo img[3] = {};
    img[0].sampler = gbufferSampler_;
    img[0].imageView = depth;
    img[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    img[1].sampler = gbufferSampler_;
    img[1].imageView = normal;
    img[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    img[2].sampler = VK_NULL_HANDLE;
    img[2].imageView = aoRaw;
    img[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet w[3] = {};
    for (uint32_t k = 0; k < 3; ++k) {
        w[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[k].dstSet = set;
        w[k].dstBinding = k;
        w[k].descriptorCount = 1;
        w[k].descriptorType = k < 2 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                    : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[k].pImageInfo = &img[k];
    }
    vkUpdateDescriptorSets(ctx.device, 3, w, 0, nullptr);
}

void DeferredCore::writeSsaoBlurSet(const VulkanContext& ctx, VkDescriptorSet set,
                                    VkImageView aoRaw, VkImageView ao) const {
    VkDescriptorImageInfo img[2] = {};
    img[0].sampler = gbufferSampler_;
    img[0].imageView = aoRaw;
    img[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    img[1].sampler = VK_NULL_HANDLE;
    img[1].imageView = ao;
    img[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet w[2] = {};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = set;
    w[0].dstBinding = 0;
    w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w[0].pImageInfo = &img[0];
    w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[1].dstSet = set;
    w[1].dstBinding = 1;
    w[1].descriptorCount = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w[1].pImageInfo = &img[1];
    vkUpdateDescriptorSets(ctx.device, 2, w, 0, nullptr);
}

void DeferredCore::recordSsaoPass(VkCommandBuffer cmd, VkDescriptorSet ssaoSet,
                                  const Mat4& viewProj, uint32_t frameIndex, uint32_t width,
                                  uint32_t height) const {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ssaoPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ssaoPipelineLayout_, 0, 1,
                            &ssaoSet, 0, nullptr);
    SsaoPush push;
    std::memcpy(push.viewProj, viewProj.m, sizeof(push.viewProj));
    push.params[0] = kSsaoRadius;
    push.params[1] = 0.f;
    push.params[2] = 0.f;
    push.params[3] = kSsaoPower;
    push.params2[0] = static_cast<float>(frameIndex);
    push.params2[1] = push.params2[2] = push.params2[3] = 0.f;
    vkCmdPushConstants(cmd, ssaoPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                       &push);
    vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);
}

void DeferredCore::recordSsaoBlurPass(VkCommandBuffer cmd, VkDescriptorSet blurSet, uint32_t width,
                                      uint32_t height) const {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ssaoBlurPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ssaoBlurPipelineLayout_, 0, 1,
                            &blurSet, 0, nullptr);
    vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);
}

void DeferredCore::writeBloomSet(const VulkanContext& ctx, VkDescriptorSet set, VkImageView src,
                                 VkImageView dst) const {
    writeSsaoBlurSet(ctx, set, src, dst);
}

void DeferredCore::recordBloomPass(VkCommandBuffer cmd, VkDescriptorSet extractSet,
                                   VkDescriptorSet blurHSet, VkDescriptorSet blurVSet,
                                   VkDescriptorSet compositeSet, VkImage bloomA, VkImage bloomB,
                                   VkImage color, VkImageLayout& bloomALayout,
                                   VkImageLayout& bloomBLayout, VkImageLayout& colorLayout,
                                   uint32_t fullW, uint32_t fullH, float strength) const {
    if (strength <= 0.f || fullW == 0 || fullH == 0) return;
    const uint32_t halfW = std::max(1u, fullW / 2);
    const uint32_t halfH = std::max(1u, fullH / 2);
    const uint32_t hx = (halfW + 7) / 8;
    const uint32_t hy = (halfH + 7) / 8;

    auto dispatch = [&](VkPipeline pipe, VkDescriptorSet set, const BloomPush& push, uint32_t gx,
                        uint32_t gy) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bloomPipelineLayout_, 0, 1,
                                &set, 0, nullptr);
        vkCmdPushConstants(cmd, bloomPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                           &push);
        vkCmdDispatch(cmd, gx, gy, 1);
    };

    imageBarrier(cmd, bloomA, bloomALayout, VK_IMAGE_LAYOUT_GENERAL);
    bloomALayout = VK_IMAGE_LAYOUT_GENERAL;
    BloomPush extractPush{};
    extractPush.params[0] = kBloomThreshold;
    extractPush.params[1] = kBloomKnee;
    dispatch(bloomExtractPipeline_, extractSet, extractPush, hx, hy);

    imageBarrier(cmd, bloomA, bloomALayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    bloomALayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageBarrier(cmd, bloomB, bloomBLayout, VK_IMAGE_LAYOUT_GENERAL);
    bloomBLayout = VK_IMAGE_LAYOUT_GENERAL;
    BloomPush blurH{};
    blurH.params[0] = 1.f;
    blurH.params[1] = 0.f;
    dispatch(bloomBlurPipeline_, blurHSet, blurH, hx, hy);

    imageBarrier(cmd, bloomB, bloomBLayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    bloomBLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageBarrier(cmd, bloomA, bloomALayout, VK_IMAGE_LAYOUT_GENERAL);
    bloomALayout = VK_IMAGE_LAYOUT_GENERAL;
    BloomPush blurV{};
    blurV.params[0] = 0.f;
    blurV.params[1] = 1.f;
    dispatch(bloomBlurPipeline_, blurVSet, blurV, hx, hy);

    imageBarrier(cmd, bloomA, bloomALayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    bloomALayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageBarrier(cmd, color, colorLayout, VK_IMAGE_LAYOUT_GENERAL);
    colorLayout = VK_IMAGE_LAYOUT_GENERAL;
    BloomPush comp{};
    comp.params[0] = strength;
    dispatch(bloomCompositePipeline_, compositeSet, comp, (fullW + 7) / 8, (fullH + 7) / 8);

    imageBarrier(cmd, color, colorLayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    colorLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void DeferredCore::recordTransparentDraws(VkCommandBuffer cmd, const Scene& scene, bool gtPass,
                                          VkDescriptorSet sceneSet, VkDescriptorSet textureSet,
                                          VkDescriptorSet transparentSet, uint32_t materialStride,
                                          uint32_t width, uint32_t height,
                                          const Mat4& cullViewProj, const Vec3& cameraPos) const {
    // Collect visible BLEND instances, sorted back-to-front by AABB center.
    const Frustum frustum = extractFrustum(cullViewProj);
    std::vector<uint32_t> order;
    for (uint32_t i = 0; i < static_cast<uint32_t>(scene.instances.size()); ++i) {
        const MeshInstance& inst = scene.instances[i];
        if (!scene.materials[inst.materialIndex].blend) continue;
        if (!aabbIntersectsFrustum(frustum, inst.aabbMin, inst.aabbMax)) continue;
        order.push_back(i);
    }
    if (order.empty()) return;

    auto dist2 = [&](uint32_t i) {
        const MeshInstance& inst = scene.instances[i];
        const float cx = (inst.aabbMin.x + inst.aabbMax.x) * 0.5f - cameraPos.x;
        const float cy = (inst.aabbMin.y + inst.aabbMax.y) * 0.5f - cameraPos.y;
        const float cz = (inst.aabbMin.z + inst.aabbMax.z) * 0.5f - cameraPos.z;
        return cx * cx + cy * cy + cz * cz;
    };
    std::sort(order.begin(), order.end(),
              [&](uint32_t a, uint32_t b) { return dist2(a) > dist2(b); });

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      gtPass ? transparentGtPipeline_ : transparentPipeline_);
    VkViewport viewport = {0.f, 0.f, static_cast<float>(width), static_cast<float>(height), 0.f, 1.f};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor = {{0, 0}, {width, height}};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, transparentPipelineLayout_, 1, 1,
                            &textureSet, 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, transparentPipelineLayout_, 2, 1,
                            &transparentSet, 0, nullptr);

    const VkDeviceSize zeroOffset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &scene.mergedVertexBuffer, &zeroOffset);
    vkCmdBindIndexBuffer(cmd, scene.mergedIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

    uint32_t lastMaterial = UINT32_MAX;
    for (const uint32_t i : order) {
        const MeshInstance& inst = scene.instances[i];
        ScenePush push;
        std::memcpy(push.model, inst.model.m, sizeof(push.model));
        std::memcpy(push.prevModel, inst.prevModel.m, sizeof(push.prevModel));
        std::memcpy(push.normalModel, inst.normalModel.m, sizeof(push.normalModel));
        vkCmdPushConstants(cmd, transparentPipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0,
                           sizeof(push), &push);

        if (inst.materialIndex != lastMaterial) {
            lastMaterial = inst.materialIndex;
            const uint32_t dynOffset = inst.materialIndex * materialStride;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, transparentPipelineLayout_,
                                    0, 1, &sceneSet, 1, &dynOffset);
        }

        const Mesh& mesh = scene.meshes[inst.meshIndex];
        vkCmdDrawIndexed(cmd, mesh.indexCount, 1, mesh.firstIndex, mesh.vertexOffset, 0);
    }
}

bool DeferredCore::createShadowTargets(const VulkanContext& ctx, ShadowTargets& out) const {
    if (createImage(ctx, kShadowMapSize, kShadowMapSize, deferred::kDepthFormat,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    out.image, out.memory, 1, kShadowCascadeCount) != VK_SUCCESS)
        return false;

    // All-layers sampled view (read as sampler2DArrayShadow in the shaders).
    out.arrayView = createImageView(ctx, out.image, deferred::kDepthFormat,
                                    VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, VK_IMAGE_VIEW_TYPE_2D_ARRAY,
                                    kShadowCascadeCount);
    if (!out.arrayView) return false;

    // One single-layer 2D view per cascade for the depth attachment.
    for (uint32_t i = 0; i < kShadowCascadeCount; ++i) {
        VkImageViewCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image = out.image;
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format = deferred::kDepthFormat;
        ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        ci.subresourceRange.baseMipLevel = 0;
        ci.subresourceRange.levelCount = 1;
        ci.subresourceRange.baseArrayLayer = i;
        ci.subresourceRange.layerCount = 1;
        if (vkCreateImageView(ctx.device, &ci, nullptr, &out.layerViews[i]) != VK_SUCCESS)
            return false;
    }
    out.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    return true;
}

void DeferredCore::destroyShadowTargets(const VulkanContext& ctx, ShadowTargets& targets) const {
    if (!ctx.device) return;
    for (VkImageView& v : targets.layerViews) {
        if (v) { vkDestroyImageView(ctx.device, v, nullptr); v = VK_NULL_HANDLE; }
    }
    if (targets.arrayView) { vkDestroyImageView(ctx.device, targets.arrayView, nullptr); targets.arrayView = VK_NULL_HANDLE; }
    if (targets.image) { vmaDestroyImage(ctx.allocator, targets.image, targets.memory); targets.image = VK_NULL_HANDLE; targets.memory = VK_NULL_HANDLE; }
    targets.layout = VK_IMAGE_LAYOUT_UNDEFINED;
}

void DeferredCore::computeCascadeVPs(const Camera& cam, float aspect, const Vec3& sunDirTowardLight,
                                     Mat4 outVP[kShadowCascadeCount],
                                     float outSplitViewDepth[kShadowCascadeCount]) {
    const float nearZ = cam.nearPlane;
    const float farZ = std::min(cam.farPlane, kShadowMaxDistance);

    // Practical split scheme: blend uniform and logarithmic splits so the
    // near cascades stay dense without starving the far ones.
    float splits[kShadowCascadeCount];
    for (uint32_t i = 0; i < kShadowCascadeCount; ++i) {
        const float p = static_cast<float>(i + 1) / static_cast<float>(kShadowCascadeCount);
        const float uniform = nearZ + (farZ - nearZ) * p;
        const float log = nearZ * std::pow(farZ / nearZ, p);
        splits[i] = kShadowSplitLambda * log + (1.f - kShadowSplitLambda) * uniform;
        outSplitViewDepth[i] = splits[i];
    }

    const Mat4 view = cam.view();
    const Vec3 sunDir = normalize(sunDirTowardLight);
    // Light travels from the sun towards the scene, i.e. along -sunDir.
    Vec3 up{0.f, 1.f, 0.f};
    if (std::fabs(dot(sunDir, up)) > 0.99f) up = {1.f, 0.f, 0.f};

    // Light orientation only (pure rotation, gluLookAt basis layout, no
    // translation).  A lookAt(eye = centre + ...) would bake the
    // camera-dependent slice centre into the basis; keeping the frame
    // translation-free is what allows the cascade origin to be snapped to a
    // constant light-space texel grid below.
    const Vec3 lightFwd = sunDir * (-1.f);
    const Vec3 lightRight = normalize(cross(lightFwd, up));
    const Vec3 lightUp = cross(lightRight, lightFwd);
    Mat4 lightRot;
    lightRot.m[0] = lightRight.x; lightRot.m[4] = lightRight.y; lightRot.m[8]  = lightRight.z;
    lightRot.m[1] = lightUp.x;    lightRot.m[5] = lightUp.y;    lightRot.m[9]  = lightUp.z;
    lightRot.m[2] = -lightFwd.x;  lightRot.m[6] = -lightFwd.y;  lightRot.m[10] = -lightFwd.z;

    float sliceNear = nearZ;
    for (uint32_t i = 0; i < kShadowCascadeCount; ++i) {
        // Unproject the 8 corners of this depth slice to world space.
        const Mat4 sliceProj = Mat4::perspective(cam.fovY, aspect, sliceNear, splits[i]);
        const Mat4 invVP = Mat4::inverse(Mat4::multiply(sliceProj, view));
        Vec3 corners[8];
        int n = 0;
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 2; ++x) {
                    corners[n++] = transformPoint(
                        invVP, Vec3{x ? 1.f : -1.f, y ? 1.f : -1.f, z ? 1.f : 0.f});
                }

        Vec3 center{0.f, 0.f, 0.f};
        for (const Vec3& c : corners) center += c;
        center = center / 8.f;

        // Bounding sphere of the slice.  The radius is rotation-invariant (it
        // depends only on the split distances, fov and aspect); rounding it
        // up to a fixed 1/kShadowRadiusSnap-metre grid additionally kills
        // float noise, so the world size of a shadow texel stays constant
        // frame-to-frame for a given split distance.  Coverage is slightly
        // larger than the old tight light-space AABB (a sphere wastes the
        // corners of the map), which is the accepted cost of stability.
        float radius = 0.f;
        for (const Vec3& c : corners) radius = std::max(radius, length(c - center));
        // Pad by two texels before quantizing: the floor() centre snap below
        // shifts the ortho window by almost a full texel towards -x/-y, and
        // without padding the sphere would poke past the +x/+y window edge,
        // where receivers sample the (white) border and lose their shadows in
        // a strip whose width oscillates with the snap phase as the camera
        // moves.
        radius += 2.f * (2.f * radius / static_cast<float>(kShadowMapSize));
        radius = std::ceil(radius * kShadowRadiusSnap) / kShadowRadiusSnap;

        // Sphere centre in light space, snapped to the (now constant) texel
        // grid.  The ortho window below is symmetric around the snapped
        // centre; the two-texel radius pad above keeps the whole sphere
        // covered even when the floor snap shifts the window by nearly a full
        // texel.
        Vec3 centerLS = transformPoint(lightRot, center);
        const float worldPerTexel = 2.f * radius / static_cast<float>(kShadowMapSize);
        if (worldPerTexel > 1e-6f) {
            centerLS.x = std::floor(centerLS.x / worldPerTexel) * worldPerTexel;
            centerLS.y = std::floor(centerLS.y / worldPerTexel) * worldPerTexel;
        }

        // Depth range around the sphere.  The sun side (larger z, closer to
        // the eye) is extended so casters between the slice and the sun still
        // land in the map; the far side needs no margin (a caster deeper than
        // a receiver cannot shadow it).  Both ends are quantized to a fixed
        // step so the depth remap does not drift frame-to-frame.
        float mnZ = centerLS.z - radius;
        float mxZ = centerLS.z + radius + kShadowCasterMargin;
        mnZ = std::floor(mnZ / kShadowZSnap) * kShadowZSnap;
        mxZ = std::ceil(mxZ / kShadowZSnap) * kShadowZSnap;

        // Right-handed Vulkan ortho (y-flipped, depth [0,1]), symmetric
        // around the snapped sphere centre.  In light view space the scene
        // sits at z < 0; n/f are positive distances.
        const float invR = 1.f / radius;
        const float nz = -mxZ, fz = -mnZ;
        Mat4 ortho;
        ortho.m[0] = invR;
        ortho.m[12] = -centerLS.x * invR;
        ortho.m[5] = -invR;                       // y flip for Vulkan NDC
        ortho.m[13] = centerLS.y * invR;
        ortho.m[10] = 1.f / (nz - fz);
        ortho.m[14] = nz / (nz - fz);
        ortho.m[15] = 1.f;

        outVP[i] = Mat4::multiply(ortho, lightRot);
        sliceNear = splits[i];
    }
}

void DeferredCore::recordShadowPass(VkCommandBuffer cmd, const ShadowTargets& targets,
                                    const Scene& scene, const Mat4 cascadeVp[kShadowCascadeCount],
                                    VkDescriptorSet sceneSet, VkDescriptorSet textureSet,
                                    uint32_t materialStride) const {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline_);
    VkViewport viewport = {0.f, 0.f, static_cast<float>(kShadowMapSize),
                           static_cast<float>(kShadowMapSize), 0.f, 1.f};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor = {{0, 0}, {kShadowMapSize, kShadowMapSize}};
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdSetDepthBias(cmd, kShadowDepthBiasConstant, 0.f, kShadowDepthBiasSlope);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipelineLayout_, 1, 1,
                            &textureSet, 0, nullptr);

    const VkDeviceSize zeroOffset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &scene.mergedVertexBuffer, &zeroOffset);
    vkCmdBindIndexBuffer(cmd, scene.mergedIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

    for (uint32_t c = 0; c < kShadowCascadeCount; ++c) {
        VkRenderingAttachmentInfo depth =
            makeDepthAttachment(targets.layerViews[c], VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                VK_ATTACHMENT_LOAD_OP_CLEAR);
        VkRenderingInfo ri = {};
        ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea = {{0, 0}, {kShadowMapSize, kShadowMapSize}};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 0;
        ri.pDepthAttachment = &depth;
        vkCmdBeginRendering(cmd, &ri);

        // Cull against the cascade frustum; the extended near plane keeps
        // casters that sit between the slice and the sun.
        const Frustum frustum = extractFrustum(cascadeVp[c]);

        uint32_t lastMaterial = UINT32_MAX;
        for (const auto& inst : scene.instances) {
            if (scene.materials[inst.materialIndex].blend) continue; // glass does not occlude
            if (!aabbIntersectsFrustum(frustum, inst.aabbMin, inst.aabbMax)) continue;

            ShadowPush push;
            std::memcpy(push.model, inst.model.m, sizeof(push.model));
            std::memcpy(push.lightVp, cascadeVp[c].m, sizeof(push.lightVp));
            vkCmdPushConstants(cmd, scenePipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0,
                               sizeof(push), &push);

            if (inst.materialIndex != lastMaterial) {
                lastMaterial = inst.materialIndex;
                const uint32_t dynOffset = inst.materialIndex * materialStride;
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipelineLayout_,
                                        0, 1, &sceneSet, 1, &dynOffset);
            }

            const Mesh& mesh = scene.meshes[inst.meshIndex];
            vkCmdDrawIndexed(cmd, mesh.indexCount, 1, mesh.firstIndex, mesh.vertexOffset, 0);
        }
        vkCmdEndRendering(cmd);
    }
}

} // namespace sr
