// ============================================================================
// DeferredCore — see DeferredCore.h for the design overview.
// ============================================================================
#include "renderer/deferred/DeferredCore.h"

#include "renderer/core/PathUtil.h"
#include "renderer/core/VkUtil.h"
#include "renderer/scene/Camera.h"
#include "renderer/scene/Scene.h"

#include <algorithm>
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

    if (!loadShader(ctx, "gbuffer.vert.spv", gbufferVert_) ||
        !loadShader(ctx, "gbuffer.frag.spv", gbufferFrag_) ||
        !loadShader(ctx, "gbuffer_gt.frag.spv", gbufferGtFrag_) ||
        !loadShader(ctx, "lighting.frag.spv", lightingFrag_) ||
        !loadShader(ctx, "fullscreen.vert.spv", fullscreenVert_) ||
        !loadShader(ctx, "transparent.vert.spv", transparentVert_) ||
        !loadShader(ctx, "transparent.frag.spv", transparentFrag_) ||
        !loadShader(ctx, "transparent_gt.frag.spv", transparentGtFrag_) ||
        !loadShader(ctx, "ssao.comp.spv", ssaoComp_) ||
        !loadShader(ctx, "ssao_blur.comp.spv", ssaoBlurComp_))
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
    // 10 = SSAO (blurred screen-space AO, R16F).
    VkDescriptorSetLayoutBinding lightBindings[11] = {};
    lightBindings[0].binding = 0;
    lightBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    lightBindings[0].descriptorCount = 1;
    lightBindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    for (uint32_t i = 1; i < 11; ++i) {
        lightBindings[i].binding = i;
        lightBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        lightBindings[i].descriptorCount = 1;
        lightBindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lightLayoutCi = {};
    lightLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lightLayoutCi.bindingCount = 11;
    lightLayoutCi.pBindings = lightBindings;
    if (vkCreateDescriptorSetLayout(ctx.device, &lightLayoutCi, nullptr, &lightingSetLayout_) !=
        VK_SUCCESS)
        return false;

    // Transparency pass: binding 0 = LightingUBO (lights array + iblParams
    // read); 1-3 = IBL (irradiance/prefilter/LUT); 4 = SSAO texture of this
    // path.
    VkDescriptorSetLayoutBinding transparentBindings[5] = {};
    transparentBindings[0].binding = 0;
    transparentBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    transparentBindings[0].descriptorCount = 1;
    transparentBindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    for (uint32_t i = 1; i < 5; ++i) {
        transparentBindings[i].binding = i;
        transparentBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        transparentBindings[i].descriptorCount = 1;
        transparentBindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo transparentLayoutCi = {};
    transparentLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    transparentLayoutCi.bindingCount = 5;
    transparentLayoutCi.pBindings = transparentBindings;
    if (vkCreateDescriptorSetLayout(ctx.device, &transparentLayoutCi, nullptr,
                                    &transparentSetLayout_) != VK_SUCCESS)
        return false;

    // SSAO: binding 0 = depth, 1 = normal (samplers), 2 = raw AO (storage).
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

    // SSAO blur: binding 0 = raw AO (sampler), 1 = blurred AO (storage).
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
    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &sceneCi, nullptr,
                                  &gbufferPipeline_) != VK_SUCCESS)
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
    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &sceneCi, nullptr,
                                  &gbufferGtPipeline_) != VK_SUCCESS)
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
    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &lightingCi, nullptr,
                                  &lightingPipeline_) != VK_SUCCESS)
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
    transparentBlend[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
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
    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &transparentCi, nullptr,
                                  &transparentPipeline_) != VK_SUCCESS)
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
    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &transparentCi, nullptr,
                                  &transparentGtPipeline_) != VK_SUCCESS)
        return false;

    // --- SSAO compute passes (ssao + cross-box blur) ---------------------------
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
    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ssaoCi, nullptr,
                                 &ssaoPipeline_) != VK_SUCCESS)
        return false;
    ssaoCi.stage.module = ssaoBlurComp_;
    ssaoCi.layout = ssaoBlurPipelineLayout_;
    return vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &ssaoCi, nullptr,
                                    &ssaoBlurPipeline_) == VK_SUCCESS;
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
    if (scenePipelineLayout_) { vkDestroyPipelineLayout(ctx.device, scenePipelineLayout_, nullptr); scenePipelineLayout_ = VK_NULL_HANDLE; }
    if (lightingPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, lightingPipelineLayout_, nullptr); lightingPipelineLayout_ = VK_NULL_HANDLE; }
    if (transparentPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, transparentPipelineLayout_, nullptr); transparentPipelineLayout_ = VK_NULL_HANDLE; }
    if (ssaoPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, ssaoPipelineLayout_, nullptr); ssaoPipelineLayout_ = VK_NULL_HANDLE; }
    if (ssaoBlurPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, ssaoBlurPipelineLayout_, nullptr); ssaoBlurPipelineLayout_ = VK_NULL_HANDLE; }
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
    if (sceneSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, sceneSetLayout_, nullptr); sceneSetLayout_ = VK_NULL_HANDLE; }
    if (textureSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, textureSetLayout_, nullptr); textureSetLayout_ = VK_NULL_HANDLE; }
    if (lightingSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, lightingSetLayout_, nullptr); lightingSetLayout_ = VK_NULL_HANDLE; }
    if (transparentSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, transparentSetLayout_, nullptr); transparentSetLayout_ = VK_NULL_HANDLE; }
    if (ssaoSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, ssaoSetLayout_, nullptr); ssaoSetLayout_ = VK_NULL_HANDLE; }
    if (ssaoBlurSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, ssaoBlurSetLayout_, nullptr); ssaoBlurSetLayout_ = VK_NULL_HANDLE; }
    if (textureSampler_) { vkDestroySampler(ctx.device, textureSampler_, nullptr); textureSampler_ = VK_NULL_HANDLE; }
    if (gbufferSampler_) { vkDestroySampler(ctx.device, gbufferSampler_, nullptr); gbufferSampler_ = VK_NULL_HANDLE; }
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
                                   const std::vector<Light>* overrideLights) const {
    std::memcpy(out.invViewProj, invViewProj.m, sizeof(out.invViewProj));

    out.cameraPos[0] = camera.position.x;
    out.cameraPos[1] = camera.position.y;
    out.cameraPos[2] = camera.position.z;
    out.cameraPos[3] = 1.f;

    // Pack the typed scene lights into the fixed-size GPU array.  Scenes
    // without authored lights fall back to the shared default set so all
    // three hosts stay identical; extra lights are dropped (lightCounts gates
    // the shader loop).  An override list (GUI sun controls) replaces the
    // scene/fallback selection entirely.
    const std::vector<Light>& lights =
        overrideLights ? *overrideLights
                       : (scene.lights.empty() ? defaultLights() : scene.lights);
    const uint32_t count =
        static_cast<uint32_t>(std::min(lights.size(), static_cast<size_t>(kMaxLights)));
    std::memset(out.lights, 0, sizeof(out.lights)); // deterministic unused slots
    for (uint32_t i = 0; i < count; ++i) {
        const Light& l = lights[i];
        LightGPU& g = out.lights[i];
        g.posOrDir[0] = l.positionOrDirection.x;
        g.posOrDir[1] = l.positionOrDirection.y;
        g.posOrDir[2] = l.positionOrDirection.z;
        g.posOrDir[3] = static_cast<float>(l.type);
        g.color[0] = l.color.x;
        g.color[1] = l.color.y;
        g.color[2] = l.color.z;
        // Intensities are scaled by PI so the PBR Lambert term (albedo/PI)
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
    out.iblParams[0] = 1.f; // env intensity
    out.iblParams[1] = static_cast<float>(ibl_.prefilterMaxLod);
    out.iblParams[2] = 1.f; // skybox enabled
    out.iblParams[3] = 0.f;
}

bool DeferredCore::createMaterialUbo(const VulkanContext& ctx, const Scene& scene,
                                     VkBuffer& buffer, VkDeviceMemory& memory,
                                     uint32_t& stride) const {
    stride = alignUp(sizeof(MaterialUBO), static_cast<uint32_t>(ctx.minUniformBufferOffsetAlignment));
    const VkDeviceSize size = static_cast<VkDeviceSize>(stride) * scene.materials.size();
    if (createBuffer(ctx, size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     buffer, memory) != VK_SUCCESS)
        return false;
    void* mapped = nullptr;
    vkMapMemory(ctx.device, memory, 0, size, 0, &mapped);
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
    vkUnmapMemory(ctx.device, memory);
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
                                    VkImageView depth, VkImageView ssao) const {
    VkDescriptorBufferInfo lightBuf = {};
    lightBuf.buffer = lightingUbo;
    lightBuf.offset = 0;
    lightBuf.range = sizeof(LightingUBO);

    VkDescriptorImageInfo img[10] = {};
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

    VkWriteDescriptorSet w[11] = {};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = set;
    w[0].dstBinding = 0;
    w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w[0].pBufferInfo = &lightBuf;
    for (uint32_t k = 0; k < 10; ++k) {
        w[k + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[k + 1].dstSet = set;
        w[k + 1].dstBinding = k + 1;
        w[k + 1].descriptorCount = 1;
        w[k + 1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[k + 1].pImageInfo = &img[k];
    }
    vkUpdateDescriptorSets(ctx.device, 11, w, 0, nullptr);
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
                                       VkBuffer lightingUbo, VkImageView ssao) const {
    VkDescriptorBufferInfo lightBuf = {};
    lightBuf.buffer = lightingUbo;
    lightBuf.offset = 0;
    lightBuf.range = sizeof(LightingUBO);

    VkDescriptorImageInfo img[4] = {};
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

    VkWriteDescriptorSet w[5] = {};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = set;
    w[0].dstBinding = 0;
    w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w[0].pBufferInfo = &lightBuf;
    for (uint32_t k = 0; k < 4; ++k) {
        w[k + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[k + 1].dstSet = set;
        w[k + 1].dstBinding = k + 1;
        w[k + 1].descriptorCount = 1;
        w[k + 1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[k + 1].pImageInfo = &img[k];
    }
    vkUpdateDescriptorSets(ctx.device, 5, w, 0, nullptr);
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
    push.params[1] = kSsaoBias;
    push.params[2] = kSsaoIntensity;
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

} // namespace sr
