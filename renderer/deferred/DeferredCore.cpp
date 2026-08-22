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

// scene::Light -> GPU packing, shared by fillLightingUBO (legacy 16-slot UBO
// array) and fillClusterLights (full lights SSBO).  Intensities are scaled by
// PI so Hammon's single-scatter 1/PI term matches the brightness of the
// legacy forward pass.  Spot cone angles are stored precomputed as cosines
// (the shaders never need the radians back).
void packLightGpu(const Light& l, LightGPU& g) {
    g.posOrDir[0] = l.positionOrDirection.x;
    g.posOrDir[1] = l.positionOrDirection.y;
    g.posOrDir[2] = l.positionOrDirection.z;
    g.posOrDir[3] = static_cast<float>(l.type);
    g.color[0] = l.color.x;
    g.color[1] = l.color.y;
    g.color[2] = l.color.z;
    g.color[3] = l.intensity * 3.14159265f;
    g.params[0] = l.range;
    g.params[1] = l.castShadow ? 1.f : 0.f;
    g.params[2] = static_cast<float>(l.shadowIndex); // spot atlas tile, -1 = unshadowed
    g.params[3] = std::cos(l.innerConeAngle);
    g.spotDir[0] = l.spotDirection.x;
    g.spotDir[1] = l.spotDirection.y;
    g.spotDir[2] = l.spotDirection.z;
    g.spotDir[3] = std::cos(l.outerConeAngle);
}

} // namespace

bool DeferredCore::init(const VulkanContext& ctx, const char* envMapPath, const Vec3& skySunDir) {
    atmosphereSky_ = !envMapPath || envMapPath[0] == '\0';
    if (atmosphereSky_) {
        // Procedural sky atmosphere (Hillaire 2020): bake the LUTs once, then
        // render the sky for the initial sun direction into the IBL chain.
        if (!sky_.init(ctx) || !ibl_.buildAtmosphere(ctx, sky_, skySunDir)) {
            std::fprintf(stderr, "sky atmosphere IBL preprocessing failed\n");
            return false;
        }
    } else if (!ibl_.build(ctx, envMapPath)) {
        std::fprintf(stderr, "IBL preprocessing failed\n");
        return false;
    }
    // Empty reflection-probe volume (count 0): the lighting/SSR descriptor
    // bindings 15-17 / 11-13 always point at valid resources; loadProbes()
    // fills them when a bake file exists.
    if (!probes_.create(ctx)) {
        std::fprintf(stderr, "reflection probe resources failed\n");
        return false;
    }

    // Scene textures: trilinear mipmapping + anisotropy (grazing-angle fix).
    float maxAniso = 4.f;
    if (ctx.features.samplerAnisotropy)
        maxAniso = std::min(16.f, ctx.properties.limits.maxSamplerAnisotropy);
    textureSampler_ = createSampler(ctx, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                    maxAniso, 32.f);
    gbufferSampler_ = createSampler(ctx, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    // Hi-Z pyramid reads are texelFetch with an explicit lod; nearest + clamp
    // keeps the untouched sampler state out of the way (and D32 mip-0 sources
    // are not guaranteed to support linear filtering).
    hizSampler_ = createSampler(ctx, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    // SSR hit-colour reads pick a mip by roughness (textureLod), so the
    // colour pyramid chain view needs trilinear filtering with the full LOD
    // range; clamp keeps edge texels from wrapping into the opposite border.
    colorPyramidSampler_ = createSampler(ctx, VK_FILTER_LINEAR,
                                         VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.f, 32.f);
    if (!textureSampler_ || !gbufferSampler_ || !hizSampler_ || !colorPyramidSampler_) return false;

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
        !loadShader(ctx, "gbuffer_skinned.vert.spv", gbufferSkinnedVert_) ||
        !loadShader(ctx, "gbuffer.frag.spv", gbufferFrag_) ||
        !loadShader(ctx, "gbuffer_gt.frag.spv", gbufferGtFrag_) ||
        !loadShader(ctx, "lighting.frag.spv", lightingFrag_) ||
        !loadShader(ctx, "fullscreen.vert.spv", fullscreenVert_) ||
        !loadShader(ctx, "transparent.vert.spv", transparentVert_) ||
        !loadShader(ctx, "transparent.frag.spv", transparentFrag_) ||
        !loadShader(ctx, "transparent_gt.frag.spv", transparentGtFrag_) ||
        !loadShader(ctx, "ssao.comp.spv", ssaoComp_) ||
        !loadShader(ctx, "ssao_blur.comp.spv", ssaoBlurComp_) ||
        !loadShader(ctx, "ssao_temporal.comp.spv", ssaoTemporalComp_) ||
        !loadShader(ctx, "hiz_downsample.comp.spv", hizDownsampleComp_) ||
        !loadShader(ctx, "color_downsample.comp.spv", colorDownsampleComp_) ||
        !loadShader(ctx, "ssr_opaque.comp.spv", ssrOpaqueComp_) ||
        !loadShader(ctx, "ssr_temporal.comp.spv", ssrTemporalComp_) ||
        !loadShader(ctx, "bloom_extract.comp.spv", bloomExtractComp_) ||
        !loadShader(ctx, "bloom_downsample.comp.spv", bloomDownsampleComp_) ||
        !loadShader(ctx, "bloom_upsample.comp.spv", bloomUpsampleComp_) ||
        !loadShader(ctx, "bloom_composite.comp.spv", bloomCompositeComp_) ||
        !loadShader(ctx, "exposure_histogram.comp.spv", exposureHistogramComp_) ||
        !loadShader(ctx, "exposure_solve.comp.spv", exposureSolveComp_) ||
        !loadShader(ctx, "shadow_depth.vert.spv", shadowDepthVert_) ||
        !loadShader(ctx, "shadow_depth_skinned.vert.spv", shadowDepthSkinnedVert_) ||
        !loadShader(ctx, "shadow_depth.frag.spv", shadowDepthFrag_) ||
        !loadShader(ctx, "cluster_assign.comp.spv", clusterAssignComp_) ||
        !loadShader(ctx, "volfog_inject.comp.spv", volfogInjectComp_) ||
        !loadShader(ctx, "volfog_light.comp.spv", volfogLightComp_) ||
        !loadShader(ctx, "volfog_temporal.comp.spv", volfogTemporalComp_) ||
        !loadShader(ctx, "volfog_march.comp.spv", volfogMarchComp_) ||
        !loadShader(ctx, "volfog_composite.comp.spv", volfogCompositeComp_) ||
        !loadShader(ctx, "motion_blur_tilemax.comp.spv", motionBlurTilemaxComp_) ||
        !loadShader(ctx, "motion_blur_neighborhood.comp.spv", motionBlurNeighborhoodComp_) ||
        !loadShader(ctx, "motion_blur_gather.comp.spv", motionBlurGatherComp_) ||
        !loadShader(ctx, "dof_coc.comp.spv", dofCocComp_) ||
        !loadShader(ctx, "dof_gather.comp.spv", dofGatherComp_) ||
        !loadShader(ctx, "dof_composite.comp.spv", dofCompositeComp_) ||
        !loadShader(ctx, "postfx_copyback.comp.spv", postFxCopybackComp_))
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
    VkDescriptorSetLayoutBinding sceneBindings[3] = {};
    sceneBindings[0].binding = 0;
    sceneBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sceneBindings[0].descriptorCount = 1;
    sceneBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    sceneBindings[1].binding = 1;
    sceneBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    sceneBindings[1].descriptorCount = 1;
    sceneBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // Joint palette SSBO, read by the skinned vertex shaders only (static
    // shaders never statically use binding 2, so hosts may leave it unwritten
    // for scenes without skins).
    sceneBindings[2].binding = 2;
    sceneBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sceneBindings[2].descriptorCount = 1;
    sceneBindings[2].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo sceneLayoutCi = {};
    sceneLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    sceneLayoutCi.bindingCount = 3;
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
    // (D32, comparison sampler); 12 = clustered-shading lights SSBO (all
    // point/spot lights, std430); 13 = this path's cluster grid SSBO
    // (per-cluster light index lists); 14 = spot shadow atlas (D32,
    // comparison sampler); 15 = reflection ProbeUBO; 16/17 = probe prefiltered
    // specular + irradiance cube arrays (Phase 4c-2).
    VkDescriptorSetLayoutBinding lightBindings[18] = {};
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
    for (uint32_t i = 12; i < 14; ++i) {
        lightBindings[i].binding = i;
        lightBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        lightBindings[i].descriptorCount = 1;
        lightBindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    lightBindings[14].binding = 14;
    lightBindings[14].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    lightBindings[14].descriptorCount = 1;
    lightBindings[14].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    lightBindings[15].binding = 15; // ProbeUBO
    lightBindings[15].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    lightBindings[15].descriptorCount = 1;
    lightBindings[15].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    for (uint32_t i = 16; i < 18; ++i) { // probe prefilter / irradiance cube arrays
        lightBindings[i].binding = i;
        lightBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        lightBindings[i].descriptorCount = 1;
        lightBindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lightLayoutCi = {};
    lightLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lightLayoutCi.bindingCount = 18;
    lightLayoutCi.pBindings = lightBindings;
    if (vkCreateDescriptorSetLayout(ctx.device, &lightLayoutCi, nullptr, &lightingSetLayout_) !=
        VK_SUCCESS)
        return false;

    // Transparency pass: binding 0 = LightingUBO; 1-3 = IBL; 4 = SSAO;
    // 5 = CSM shadow map; 6 = opaque HDR copy (SSR); 7 = opaque depth pyramid
    // (Hi-Z, R32F, SSR hierarchical march).
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

    // GTAO: binding 0 = AO depth chain (view-Z mips, GENERAL), 1 = normal
    // (samplers), 2 = working RG16F (storage).
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

    // GTAO temporal accumulation: binding 0 = raw AO (sampler), 1 = history
    // read (sampler, GENERAL), 2 = history write (storage), 3 = GBuffer depth
    // (sampler, for the reprojection world-position reconstruction).
    VkDescriptorSetLayoutBinding ssaoTemporalBindings[4] = {};
    for (uint32_t i = 0; i < 4; ++i) {
        ssaoTemporalBindings[i].binding = i;
        ssaoTemporalBindings[i].descriptorCount = 1;
        ssaoTemporalBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        ssaoTemporalBindings[i].descriptorType =
            i == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    }
    VkDescriptorSetLayoutCreateInfo ssaoTemporalLayoutCi = {};
    ssaoTemporalLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ssaoTemporalLayoutCi.bindingCount = 4;
    ssaoTemporalLayoutCi.pBindings = ssaoTemporalBindings;
    if (vkCreateDescriptorSetLayout(ctx.device, &ssaoTemporalLayoutCi, nullptr,
                                    &ssaoTemporalSetLayout_) != VK_SUCCESS)
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
    if (vkCreateDescriptorSetLayout(ctx.device, &ssaoBlurLayoutCi, nullptr, &ssaoBlurSetLayout_) !=
        VK_SUCCESS)
        return false;

    // Hi-Z downsample: binding 0 = source level (sampler; mip 0 binds the D32
    // opaque depth), 1 = destination mip (R32F storage).
    VkDescriptorSetLayoutBinding hizBindings[2] = {};
    hizBindings[0].binding = 0;
    hizBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    hizBindings[0].descriptorCount = 1;
    hizBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    hizBindings[1].binding = 1;
    hizBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    hizBindings[1].descriptorCount = 1;
    hizBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo hizLayoutCi = {};
    hizLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    hizLayoutCi.bindingCount = 2;
    hizLayoutCi.pBindings = hizBindings;
    if (vkCreateDescriptorSetLayout(ctx.device, &hizLayoutCi, nullptr, &hizSetLayout_) !=
        VK_SUCCESS)
        return false;

    // Opaque SSR trace (compute): binding 0 = LightingUBO (only the
    // invViewProj / cameraPos / iblParams prefix is read); 1-4 = GBuffer
    // albedo / normal / material / depth; 5 = SSAO; 6 = IBL prefilter cube;
    // 7 = BRDF LUT; 8 = lit-color pyramid chain; 9 = Hi-Z depth pyramid;
    // 10 = the path's full-res trace target (write-only storage; rgb =
    // composite delta, a = view |z|); 11 = reflection ProbeUBO; 12/13 = probe
    // prefilter / irradiance cube arrays (Phase 4c-2: the recomputed specIbl
    // must use the same probe fallback chain as lighting.frag).
    VkDescriptorSetLayoutBinding ssrBindings[14] = {};
    ssrBindings[0].binding = 0;
    ssrBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ssrBindings[0].descriptorCount = 1;
    ssrBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    for (uint32_t i = 1; i < 10; ++i) {
        ssrBindings[i].binding = i;
        ssrBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ssrBindings[i].descriptorCount = 1;
        ssrBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    ssrBindings[10].binding = 10;
    ssrBindings[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ssrBindings[10].descriptorCount = 1;
    ssrBindings[10].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    ssrBindings[11].binding = 11;
    ssrBindings[11].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ssrBindings[11].descriptorCount = 1;
    ssrBindings[11].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    for (uint32_t i = 12; i < 14; ++i) {
        ssrBindings[i].binding = i;
        ssrBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ssrBindings[i].descriptorCount = 1;
        ssrBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo ssrLayoutCi = {};
    ssrLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ssrLayoutCi.bindingCount = 14;
    ssrLayoutCi.pBindings = ssrBindings;
    if (vkCreateDescriptorSetLayout(ctx.device, &ssrLayoutCi, nullptr, &ssrSetLayout_) !=
        VK_SUCCESS)
        return false;

    // SSR temporal filter + fused composite: binding 0 = trace target
    // (sampler), 1 = history read (sampler, GENERAL), 2 = history write
    // (storage), 3 = GBuffer depth (sampler, reprojection reconstruction),
    // 4 = lit HDR scene colour (storage, in-place RMW composite add).
    VkDescriptorSetLayoutBinding ssrTemporalBindings[5] = {};
    for (uint32_t i = 0; i < 5; ++i) {
        ssrTemporalBindings[i].binding = i;
        ssrTemporalBindings[i].descriptorCount = 1;
        ssrTemporalBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        ssrTemporalBindings[i].descriptorType =
            (i == 2 || i == 4) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                               : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    }
    VkDescriptorSetLayoutCreateInfo ssrTemporalLayoutCi = {};
    ssrTemporalLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ssrTemporalLayoutCi.bindingCount = 5;
    ssrTemporalLayoutCi.pBindings = ssrTemporalBindings;
    if (vkCreateDescriptorSetLayout(ctx.device, &ssrTemporalLayoutCi, nullptr,
                                    &ssrTemporalSetLayout_) != VK_SUCCESS)
        return false;

    // Cluster light assignment (compute): binding 0 = lights SSBO (read),
    // 1 = cluster grid SSBO (counts + index lists, write).
    VkDescriptorSetLayoutBinding clusterBindings[2] = {};
    for (uint32_t i = 0; i < 2; ++i) {
        clusterBindings[i].binding = i;
        clusterBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        clusterBindings[i].descriptorCount = 1;
        clusterBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo clusterLayoutCi = {};
    clusterLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    clusterLayoutCi.bindingCount = 2;
    clusterLayoutCi.pBindings = clusterBindings;
    if (vkCreateDescriptorSetLayout(ctx.device, &clusterLayoutCi, nullptr, &clusterSetLayout_) !=
        VK_SUCCESS)
        return false;

    // Auto exposure: one shared set for both passes.  binding 0 = lit HDR
    // source (sampler, histogram pass); 1 = histogram SSBO; 2 = exposure
    // state SSBO (solve pass).  Unused bindings per pass are legal.
    VkDescriptorSetLayoutBinding exposureBindings[3] = {};
    exposureBindings[0].binding = 0;
    exposureBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    exposureBindings[0].descriptorCount = 1;
    exposureBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    for (uint32_t i = 1; i < 3; ++i) {
        exposureBindings[i].binding = i;
        exposureBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        exposureBindings[i].descriptorCount = 1;
        exposureBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo exposureLayoutCi = {};
    exposureLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    exposureLayoutCi.bindingCount = 3;
    exposureLayoutCi.pBindings = exposureBindings;
    if (vkCreateDescriptorSetLayout(ctx.device, &exposureLayoutCi, nullptr,
                                    &exposureSetLayout_) != VK_SUCCESS)
        return false;

    // --- Froxel volumetric fog (Phase 5a; binding shapes per volfog_*.comp) ---
    // Inject: binding 0 = inject volume (storage image3D).
    VkDescriptorSetLayoutBinding volfogInjectBinding = {};
    volfogInjectBinding.binding = 0;
    volfogInjectBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    volfogInjectBinding.descriptorCount = 1;
    volfogInjectBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo volfogInjectLayoutCi = {};
    volfogInjectLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    volfogInjectLayoutCi.bindingCount = 1;
    volfogInjectLayoutCi.pBindings = &volfogInjectBinding;
    if (vkCreateDescriptorSetLayout(ctx.device, &volfogInjectLayoutCi, nullptr,
                                    &volfogInjectSetLayout_) != VK_SUCCESS)
        return false;

    // Light accumulation: 0 = LightingUBO, 1/2 = cluster lights + grid SSBOs,
    // 3/4 = CSM + spot atlas (comparison samplers), 5 = inject volume
    // (sampler3D), 6 = raw-lit volume out (storage image3D).
    VkDescriptorSetLayoutBinding volfogLightBindings[7] = {};
    volfogLightBindings[0].binding = 0;
    volfogLightBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    volfogLightBindings[0].descriptorCount = 1;
    volfogLightBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    for (uint32_t i = 1; i < 3; ++i) {
        volfogLightBindings[i].binding = i;
        volfogLightBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        volfogLightBindings[i].descriptorCount = 1;
        volfogLightBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    for (uint32_t i = 3; i < 6; ++i) {
        volfogLightBindings[i].binding = i;
        volfogLightBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        volfogLightBindings[i].descriptorCount = 1;
        volfogLightBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    volfogLightBindings[6].binding = 6;
    volfogLightBindings[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    volfogLightBindings[6].descriptorCount = 1;
    volfogLightBindings[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo volfogLightLayoutCi = {};
    volfogLightLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    volfogLightLayoutCi.bindingCount = 7;
    volfogLightLayoutCi.pBindings = volfogLightBindings;
    if (vkCreateDescriptorSetLayout(ctx.device, &volfogLightLayoutCi, nullptr,
                                    &volfogLightSetLayout_) != VK_SUCCESS)
        return false;

    // Temporal: 0 = raw lit (sampler3D), 1 = history read (sampler3D),
    // 2 = history write (storage image3D).
    VkDescriptorSetLayoutBinding volfogTemporalBindings[3] = {};
    for (uint32_t i = 0; i < 3; ++i) {
        volfogTemporalBindings[i].binding = i;
        volfogTemporalBindings[i].descriptorCount = 1;
        volfogTemporalBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        volfogTemporalBindings[i].descriptorType =
            i == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    }
    VkDescriptorSetLayoutCreateInfo volfogTemporalLayoutCi = {};
    volfogTemporalLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    volfogTemporalLayoutCi.bindingCount = 3;
    volfogTemporalLayoutCi.pBindings = volfogTemporalBindings;
    if (vkCreateDescriptorSetLayout(ctx.device, &volfogTemporalLayoutCi, nullptr,
                                    &volfogTemporalSetLayout_) != VK_SUCCESS)
        return false;

    // March: 0 = filtered lit volume (sampler3D), 1 = integrated out (storage).
    VkDescriptorSetLayoutBinding volfogMarchBindings[2] = {};
    volfogMarchBindings[0].binding = 0;
    volfogMarchBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    volfogMarchBindings[0].descriptorCount = 1;
    volfogMarchBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    volfogMarchBindings[1].binding = 1;
    volfogMarchBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    volfogMarchBindings[1].descriptorCount = 1;
    volfogMarchBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo volfogMarchLayoutCi = {};
    volfogMarchLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    volfogMarchLayoutCi.bindingCount = 2;
    volfogMarchLayoutCi.pBindings = volfogMarchBindings;
    if (vkCreateDescriptorSetLayout(ctx.device, &volfogMarchLayoutCi, nullptr,
                                    &volfogMarchSetLayout_) != VK_SUCCESS)
        return false;

    // Composite: 0 = lit HDR scene colour (storage image2D, RMW), 1 = GBuffer
    // depth (sampler2D), 2 = integrated volume (sampler3D, trilinear).
    VkDescriptorSetLayoutBinding volfogCompositeBindings[3] = {};
    volfogCompositeBindings[0].binding = 0;
    volfogCompositeBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    volfogCompositeBindings[0].descriptorCount = 1;
    volfogCompositeBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    for (uint32_t i = 1; i < 3; ++i) {
        volfogCompositeBindings[i].binding = i;
        volfogCompositeBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        volfogCompositeBindings[i].descriptorCount = 1;
        volfogCompositeBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo volfogCompositeLayoutCi = {};
    volfogCompositeLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    volfogCompositeLayoutCi.bindingCount = 3;
    volfogCompositeLayoutCi.pBindings = volfogCompositeBindings;
    if (vkCreateDescriptorSetLayout(ctx.device, &volfogCompositeLayoutCi, nullptr,
                                    &volfogCompositeSetLayout_) != VK_SUCCESS)
        return false;

    // --- Motion blur + DOF (Phase 6b; binding shapes per motion_blur_*/dof_*.comp) ---
    // Tile max + neighbourhood max: 0 = src (sampler), 1 = dst (storage) —
    // one layout shared by both reduce passes.
    VkDescriptorSetLayoutBinding mbTileBindings[2] = {};
    mbTileBindings[0].binding = 0;
    mbTileBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    mbTileBindings[0].descriptorCount = 1;
    mbTileBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    mbTileBindings[1].binding = 1;
    mbTileBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    mbTileBindings[1].descriptorCount = 1;
    mbTileBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo mbTileLayoutCi = {};
    mbTileLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    mbTileLayoutCi.bindingCount = 2;
    mbTileLayoutCi.pBindings = mbTileBindings;
    if (vkCreateDescriptorSetLayout(ctx.device, &mbTileLayoutCi, nullptr, &mbTileSetLayout_) !=
        VK_SUCCESS)
        return false;

    // Gather: 0 = color, 1 = motion, 2 = neighbourhood max, 3 = depth
    // (samplers), 4 = blurred out (storage).
    VkDescriptorSetLayoutBinding mbGatherBindings[5] = {};
    for (uint32_t i = 0; i < 4; ++i) {
        mbGatherBindings[i].binding = i;
        mbGatherBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        mbGatherBindings[i].descriptorCount = 1;
        mbGatherBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    mbGatherBindings[4].binding = 4;
    mbGatherBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    mbGatherBindings[4].descriptorCount = 1;
    mbGatherBindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo mbGatherLayoutCi = {};
    mbGatherLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    mbGatherLayoutCi.bindingCount = 5;
    mbGatherLayoutCi.pBindings = mbGatherBindings;
    if (vkCreateDescriptorSetLayout(ctx.device, &mbGatherLayoutCi, nullptr, &mbGatherSetLayout_) !=
        VK_SUCCESS)
        return false;

    // DOF CoC: 0 = color, 1 = depth (samplers), 2 = half-res cocColor (storage).
    VkDescriptorSetLayoutBinding dofCocBindings[3] = {};
    for (uint32_t i = 0; i < 2; ++i) {
        dofCocBindings[i].binding = i;
        dofCocBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        dofCocBindings[i].descriptorCount = 1;
        dofCocBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    dofCocBindings[2].binding = 2;
    dofCocBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    dofCocBindings[2].descriptorCount = 1;
    dofCocBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dofCocLayoutCi = {};
    dofCocLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dofCocLayoutCi.bindingCount = 3;
    dofCocLayoutCi.pBindings = dofCocBindings;
    if (vkCreateDescriptorSetLayout(ctx.device, &dofCocLayoutCi, nullptr, &dofCocSetLayout_) !=
        VK_SUCCESS)
        return false;

    // DOF gather: 0 = cocColor (sampler), 1/2 = background/foreground layers (storage).
    VkDescriptorSetLayoutBinding dofGatherBindings[3] = {};
    dofGatherBindings[0].binding = 0;
    dofGatherBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    dofGatherBindings[0].descriptorCount = 1;
    dofGatherBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    for (uint32_t i = 1; i < 3; ++i) {
        dofGatherBindings[i].binding = i;
        dofGatherBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        dofGatherBindings[i].descriptorCount = 1;
        dofGatherBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dofGatherLayoutCi = {};
    dofGatherLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dofGatherLayoutCi.bindingCount = 3;
    dofGatherLayoutCi.pBindings = dofGatherBindings;
    if (vkCreateDescriptorSetLayout(ctx.device, &dofGatherLayoutCi, nullptr,
                                    &dofGatherSetLayout_) != VK_SUCCESS)
        return false;

    // DOF composite: 0 = sharp color (storage, imageLoad), 1-3 = cocColor /
    // background / foreground layers (samplers), 4 = HDR out (storage).
    VkDescriptorSetLayoutBinding dofCompositeBindings[5] = {};
    dofCompositeBindings[0].binding = 0;
    dofCompositeBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    dofCompositeBindings[0].descriptorCount = 1;
    dofCompositeBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    for (uint32_t i = 1; i < 4; ++i) {
        dofCompositeBindings[i].binding = i;
        dofCompositeBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        dofCompositeBindings[i].descriptorCount = 1;
        dofCompositeBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    dofCompositeBindings[4].binding = 4;
    dofCompositeBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    dofCompositeBindings[4].descriptorCount = 1;
    dofCompositeBindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dofCompositeLayoutCi = {};
    dofCompositeLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dofCompositeLayoutCi.bindingCount = 5;
    dofCompositeLayoutCi.pBindings = dofCompositeBindings;
    if (vkCreateDescriptorSetLayout(ctx.device, &dofCompositeLayoutCi, nullptr,
                                    &dofCompositeSetLayout_) != VK_SUCCESS)
        return false;

    // MB copy-back (MB on, DOF off): 0 = blurred color (storage, read),
    // 1 = lit HDR out (storage, write).
    VkDescriptorSetLayoutBinding copybackBindings[2] = {};
    for (uint32_t i = 0; i < 2; ++i) {
        copybackBindings[i].binding = i;
        copybackBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        copybackBindings[i].descriptorCount = 1;
        copybackBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo copybackLayoutCi = {};
    copybackLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    copybackLayoutCi.bindingCount = 2;
    copybackLayoutCi.pBindings = copybackBindings;
    return vkCreateDescriptorSetLayout(ctx.device, &copybackLayoutCi, nullptr,
                                       &postFxCopybackSetLayout_) == VK_SUCCESS;
}

bool DeferredCore::createPipelines(const VulkanContext& ctx) {
    // The scene push range must cover the largest block: SkinnedScenePush
    // (208 B).  Static draws push fewer bytes, which is always legal.
    if (ctx.properties.limits.maxPushConstantsSize < sizeof(SkinnedScenePush)) {
        std::fprintf(stderr, "device maxPushConstantsSize=%u < %zu\n",
                     ctx.properties.limits.maxPushConstantsSize, sizeof(SkinnedScenePush));
        return false;
    }

    VkPushConstantRange pushRange = {};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(SkinnedScenePush);

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

    // GT GBuffer pipeline: same five attachments (Phase 6b: the GT path keeps
    // a motion RT so GT motion blur matches the LR path's).
    VkPipelineRenderingCreateInfo gtRendering = {};
    gtRendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    gtRendering.colorAttachmentCount = 5;
    gtRendering.pColorAttachmentFormats = gbColorFormats;
    gtRendering.depthAttachmentFormat = deferred::kDepthFormat;
    sceneCi.pNext = &gtRendering;
    colorBlend.attachmentCount = 5;
    stages[1].module = gbufferGtFrag_;
    if (createGraphicsPipeline(ctx, sceneCi, gbufferGtPipeline_) != VK_SUCCESS)
        return false;

    // --- Skinned GBuffer pipelines (gbuffer_skinned.vert) ----------------------
    // Same fragment shaders and pipeline layout; only the vertex input grows
    // by JOINTS_0 (u16x4) / WEIGHTS_0 (f32x4).
    VkVertexInputBindingDescription skinnedBinding = {};
    skinnedBinding.binding = 0;
    skinnedBinding.stride = sizeof(SkinnedVertex);
    skinnedBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription skinnedAttrs[6] = {};
    for (uint32_t k = 0; k < 4; ++k) skinnedAttrs[k] = attrs[k]; // pos/normal/uv/tangent
    skinnedAttrs[4].location = 4;
    skinnedAttrs[4].binding = 0;
    skinnedAttrs[4].format = VK_FORMAT_R16G16B16A16_UINT;
    skinnedAttrs[4].offset = sizeof(Vec3) * 2 + sizeof(Vec2) + sizeof(Vec4);
    skinnedAttrs[5].location = 5;
    skinnedAttrs[5].binding = 0;
    skinnedAttrs[5].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    skinnedAttrs[5].offset = sizeof(Vec3) * 2 + sizeof(Vec2) + sizeof(Vec4) + sizeof(uint16_t) * 4;

    VkPipelineVertexInputStateCreateInfo skinnedVertexInput = {};
    skinnedVertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    skinnedVertexInput.vertexBindingDescriptionCount = 1;
    skinnedVertexInput.pVertexBindingDescriptions = &skinnedBinding;
    skinnedVertexInput.vertexAttributeDescriptionCount = 6;
    skinnedVertexInput.pVertexAttributeDescriptions = skinnedAttrs;

    stages[0].module = gbufferSkinnedVert_;
    stages[1].module = gbufferFrag_;
    colorBlend.attachmentCount = 5;
    sceneCi.pNext = &gbRendering;
    sceneCi.pVertexInputState = &skinnedVertexInput;
    if (createGraphicsPipeline(ctx, sceneCi, gbufferSkinnedPipeline_) != VK_SUCCESS)
        return false;
    stages[1].module = gbufferGtFrag_;
    colorBlend.attachmentCount = 5;
    sceneCi.pNext = &gtRendering;
    if (createGraphicsPipeline(ctx, sceneCi, gbufferSkinnedGtPipeline_) != VK_SUCCESS)
        return false;
    stages[0].module = gbufferVert_;
    sceneCi.pVertexInputState = &vertexInput;
    colorBlend.attachmentCount = 5;

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

    // --- GTAO compute passes (main + temporal accumulation + 5x5 denoise) ------
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

    VkPushConstantRange ssaoTemporalPushRange = {};
    ssaoTemporalPushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    ssaoTemporalPushRange.offset = 0;
    ssaoTemporalPushRange.size = sizeof(SsaoTemporalPush);
    VkPipelineLayoutCreateInfo ssaoTemporalLayoutCi = {};
    ssaoTemporalLayoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ssaoTemporalLayoutCi.setLayoutCount = 1;
    ssaoTemporalLayoutCi.pSetLayouts = &ssaoTemporalSetLayout_;
    ssaoTemporalLayoutCi.pushConstantRangeCount = 1;
    ssaoTemporalLayoutCi.pPushConstantRanges = &ssaoTemporalPushRange;
    if (vkCreatePipelineLayout(ctx.device, &ssaoTemporalLayoutCi, nullptr,
                               &ssaoTemporalPipelineLayout_) != VK_SUCCESS)
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
    ssaoCi.stage.module = ssaoTemporalComp_;
    ssaoCi.layout = ssaoTemporalPipelineLayout_;
    if (createComputePipeline(ctx, ssaoCi, ssaoTemporalPipeline_) != VK_SUCCESS)
        return false;

    // --- Hi-Z downsample (per-mip reduce; max for SSR, XeGTAO filter for AO) ---
    VkPushConstantRange hizPushRange = {};
    hizPushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    hizPushRange.offset = 0;
    hizPushRange.size = sizeof(HiZPush);
    VkPipelineLayoutCreateInfo hizLayoutCi = {};
    hizLayoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    hizLayoutCi.setLayoutCount = 1;
    hizLayoutCi.pSetLayouts = &hizSetLayout_;
    hizLayoutCi.pushConstantRangeCount = 1;
    hizLayoutCi.pPushConstantRanges = &hizPushRange;
    if (vkCreatePipelineLayout(ctx.device, &hizLayoutCi, nullptr, &hizPipelineLayout_) !=
        VK_SUCCESS)
        return false;
    VkComputePipelineCreateInfo hizCi = {};
    hizCi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    hizCi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    hizCi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    hizCi.stage.module = hizDownsampleComp_;
    hizCi.stage.pName = "main";
    hizCi.layout = hizPipelineLayout_;
    if (createComputePipeline(ctx, hizCi, hizPipeline_) != VK_SUCCESS)
        return false;

    // HDR color mip chain: same set layout / pipeline layout / push constants
    // as Hi-Z (one combined sampler + one storage image, HiZPush), only the
    // reduce op differs (box average instead of max).
    hizCi.stage.module = colorDownsampleComp_;
    if (createComputePipeline(ctx, hizCi, colorDownsamplePipeline_) != VK_SUCCESS)
        return false;

    // --- Opaque SSR trace (fullscreen compute -> trace target) ----------------
    VkPushConstantRange ssrPushRange = {};
    ssrPushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    ssrPushRange.offset = 0;
    ssrPushRange.size = sizeof(SsrPush);
    VkPipelineLayoutCreateInfo ssrLayoutCi = {};
    ssrLayoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ssrLayoutCi.setLayoutCount = 1;
    ssrLayoutCi.pSetLayouts = &ssrSetLayout_;
    ssrLayoutCi.pushConstantRangeCount = 1;
    ssrLayoutCi.pPushConstantRanges = &ssrPushRange;
    if (vkCreatePipelineLayout(ctx.device, &ssrLayoutCi, nullptr, &ssrPipelineLayout_) !=
        VK_SUCCESS)
        return false;
    VkComputePipelineCreateInfo ssrCi = {};
    ssrCi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ssrCi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ssrCi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ssrCi.stage.module = ssrOpaqueComp_;
    ssrCi.stage.pName = "main";
    ssrCi.layout = ssrPipelineLayout_;
    if (createComputePipeline(ctx, ssrCi, ssrPipeline_) != VK_SUCCESS)
        return false;

    // --- SSR temporal filter + fused composite (ssr_temporal.comp) ------------
    // Same push layout as the GTAO temporal pass (SsaoTemporalPush, 144 B).
    VkPushConstantRange ssrTemporalPushRange = {};
    ssrTemporalPushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    ssrTemporalPushRange.offset = 0;
    ssrTemporalPushRange.size = sizeof(SsaoTemporalPush);
    VkPipelineLayoutCreateInfo ssrTemporalLayoutCi = {};
    ssrTemporalLayoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ssrTemporalLayoutCi.setLayoutCount = 1;
    ssrTemporalLayoutCi.pSetLayouts = &ssrTemporalSetLayout_;
    ssrTemporalLayoutCi.pushConstantRangeCount = 1;
    ssrTemporalLayoutCi.pPushConstantRanges = &ssrTemporalPushRange;
    if (vkCreatePipelineLayout(ctx.device, &ssrTemporalLayoutCi, nullptr,
                               &ssrTemporalPipelineLayout_) != VK_SUCCESS)
        return false;
    ssrCi.stage.module = ssrTemporalComp_;
    ssrCi.layout = ssrTemporalPipelineLayout_;
    if (createComputePipeline(ctx, ssrCi, ssrTemporalPipeline_) != VK_SUCCESS)
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
    bloomCi.stage.module = bloomDownsampleComp_;
    if (createComputePipeline(ctx, bloomCi, bloomDownsamplePipeline_) != VK_SUCCESS)
        return false;
    bloomCi.stage.module = bloomUpsampleComp_;
    if (createComputePipeline(ctx, bloomCi, bloomUpsamplePipeline_) != VK_SUCCESS)
        return false;
    bloomCi.stage.module = bloomCompositeComp_;
    if (createComputePipeline(ctx, bloomCi, bloomCompositePipeline_) != VK_SUCCESS)
        return false;

    // --- Motion blur + DOF (Phase 6b) ----------------------------------------
    // One pipeline layout per pass (each carries its own push range); the
    // tile-max and neighbourhood-max passes share mbTileSetLayout_.
    {
        VkPushConstantRange mbTilePushRange = {};
        mbTilePushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        mbTilePushRange.offset = 0;
        mbTilePushRange.size = sizeof(MotionBlurTilePush);
        VkPipelineLayoutCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        ci.setLayoutCount = 1;
        ci.pSetLayouts = &mbTileSetLayout_;
        ci.pushConstantRangeCount = 1;
        ci.pPushConstantRanges = &mbTilePushRange;
        if (vkCreatePipelineLayout(ctx.device, &ci, nullptr, &mbTilePipelineLayout_) != VK_SUCCESS)
            return false;

        VkPushConstantRange mbGatherPushRange = {};
        mbGatherPushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        mbGatherPushRange.offset = 0;
        mbGatherPushRange.size = sizeof(MotionBlurGatherPush);
        VkPipelineLayoutCreateInfo gci = {};
        gci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        gci.setLayoutCount = 1;
        gci.pSetLayouts = &mbGatherSetLayout_;
        gci.pushConstantRangeCount = 1;
        gci.pPushConstantRanges = &mbGatherPushRange;
        if (vkCreatePipelineLayout(ctx.device, &gci, nullptr, &mbGatherPipelineLayout_) !=
            VK_SUCCESS)
            return false;

        VkPushConstantRange dofCocPushRange = {};
        dofCocPushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        dofCocPushRange.offset = 0;
        dofCocPushRange.size = sizeof(DofCocPush);
        VkPipelineLayoutCreateInfo cci = {};
        cci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        cci.setLayoutCount = 1;
        cci.pSetLayouts = &dofCocSetLayout_;
        cci.pushConstantRangeCount = 1;
        cci.pPushConstantRanges = &dofCocPushRange;
        if (vkCreatePipelineLayout(ctx.device, &cci, nullptr, &dofCocPipelineLayout_) != VK_SUCCESS)
            return false;

        VkPushConstantRange dofGatherPushRange = {};
        dofGatherPushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        dofGatherPushRange.offset = 0;
        dofGatherPushRange.size = sizeof(DofGatherPush);
        VkPipelineLayoutCreateInfo dgci = {};
        dgci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        dgci.setLayoutCount = 1;
        dgci.pSetLayouts = &dofGatherSetLayout_;
        dgci.pushConstantRangeCount = 1;
        dgci.pPushConstantRanges = &dofGatherPushRange;
        if (vkCreatePipelineLayout(ctx.device, &dgci, nullptr, &dofGatherPipelineLayout_) !=
            VK_SUCCESS)
            return false;

        VkPushConstantRange dofCompositePushRange = {};
        dofCompositePushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        dofCompositePushRange.offset = 0;
        dofCompositePushRange.size = sizeof(DofCompositePush);
        VkPipelineLayoutCreateInfo dcci = {};
        dcci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        dcci.setLayoutCount = 1;
        dcci.pSetLayouts = &dofCompositeSetLayout_;
        dcci.pushConstantRangeCount = 1;
        dcci.pPushConstantRanges = &dofCompositePushRange;
        if (vkCreatePipelineLayout(ctx.device, &dcci, nullptr, &dofCompositePipelineLayout_) !=
            VK_SUCCESS)
            return false;

        // Copy-back: no push constants.
        VkPipelineLayoutCreateInfo kci = {};
        kci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        kci.setLayoutCount = 1;
        kci.pSetLayouts = &postFxCopybackSetLayout_;
        if (vkCreatePipelineLayout(ctx.device, &kci, nullptr, &postFxCopybackPipelineLayout_) !=
            VK_SUCCESS)
            return false;
    }
    VkComputePipelineCreateInfo postFxCi = {};
    postFxCi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    postFxCi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    postFxCi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    postFxCi.stage.pName = "main";
    postFxCi.layout = mbTilePipelineLayout_;
    postFxCi.stage.module = motionBlurTilemaxComp_;
    if (createComputePipeline(ctx, postFxCi, mbTilePipeline_) != VK_SUCCESS)
        return false;
    postFxCi.stage.module = motionBlurNeighborhoodComp_;
    if (createComputePipeline(ctx, postFxCi, mbNeighborPipeline_) != VK_SUCCESS)
        return false;
    postFxCi.layout = mbGatherPipelineLayout_;
    postFxCi.stage.module = motionBlurGatherComp_;
    if (createComputePipeline(ctx, postFxCi, mbGatherPipeline_) != VK_SUCCESS)
        return false;
    postFxCi.layout = dofCocPipelineLayout_;
    postFxCi.stage.module = dofCocComp_;
    if (createComputePipeline(ctx, postFxCi, dofCocPipeline_) != VK_SUCCESS)
        return false;
    postFxCi.layout = dofGatherPipelineLayout_;
    postFxCi.stage.module = dofGatherComp_;
    if (createComputePipeline(ctx, postFxCi, dofGatherPipeline_) != VK_SUCCESS)
        return false;
    postFxCi.layout = dofCompositePipelineLayout_;
    postFxCi.stage.module = dofCompositeComp_;
    if (createComputePipeline(ctx, postFxCi, dofCompositePipeline_) != VK_SUCCESS)
        return false;
    postFxCi.layout = postFxCopybackPipelineLayout_;
    postFxCi.stage.module = postFxCopybackComp_;
    if (createComputePipeline(ctx, postFxCi, postFxCopybackPipeline_) != VK_SUCCESS)
        return false;

    // --- Auto exposure (histogram + EV solver) --------------------------------
    // Both passes share exposureSetLayout_ (bindings 0-2); each pipeline
    // layout carries its own push range.
    VkPushConstantRange histogramPushRange = {};
    histogramPushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    histogramPushRange.offset = 0;
    histogramPushRange.size = sizeof(HistogramPush);
    VkPipelineLayoutCreateInfo histogramLayoutCi = {};
    histogramLayoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    histogramLayoutCi.setLayoutCount = 1;
    histogramLayoutCi.pSetLayouts = &exposureSetLayout_;
    histogramLayoutCi.pushConstantRangeCount = 1;
    histogramLayoutCi.pPushConstantRanges = &histogramPushRange;
    if (vkCreatePipelineLayout(ctx.device, &histogramLayoutCi, nullptr,
                               &histogramPipelineLayout_) != VK_SUCCESS)
        return false;

    VkPushConstantRange solvePushRange = {};
    solvePushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    solvePushRange.offset = 0;
    solvePushRange.size = sizeof(ExposureSolvePush) + 16; // + tuning vec4
    VkPipelineLayoutCreateInfo solveLayoutCi = {};
    solveLayoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    solveLayoutCi.setLayoutCount = 1;
    solveLayoutCi.pSetLayouts = &exposureSetLayout_;
    solveLayoutCi.pushConstantRangeCount = 1;
    solveLayoutCi.pPushConstantRanges = &solvePushRange;
    if (vkCreatePipelineLayout(ctx.device, &solveLayoutCi, nullptr,
                               &exposureSolvePipelineLayout_) != VK_SUCCESS)
        return false;

    VkComputePipelineCreateInfo exposureCi = {};
    exposureCi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    exposureCi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    exposureCi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    exposureCi.stage.pName = "main";
    exposureCi.stage.module = exposureHistogramComp_;
    exposureCi.layout = histogramPipelineLayout_;
    if (createComputePipeline(ctx, exposureCi, histogramPipeline_) != VK_SUCCESS)
        return false;
    exposureCi.stage.module = exposureSolveComp_;
    exposureCi.layout = exposureSolvePipelineLayout_;
    if (createComputePipeline(ctx, exposureCi, exposureSolvePipeline_) != VK_SUCCESS)
        return false;

    // --- Clustered shading: light-assignment compute pass -----------------------
    VkPushConstantRange clusterPushRange = {};
    clusterPushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    clusterPushRange.offset = 0;
    clusterPushRange.size = sizeof(ClusterAssignPush);
    VkPipelineLayoutCreateInfo clusterLayoutCi = {};
    clusterLayoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    clusterLayoutCi.setLayoutCount = 1;
    clusterLayoutCi.pSetLayouts = &clusterSetLayout_;
    clusterLayoutCi.pushConstantRangeCount = 1;
    clusterLayoutCi.pPushConstantRanges = &clusterPushRange;
    if (vkCreatePipelineLayout(ctx.device, &clusterLayoutCi, nullptr, &clusterPipelineLayout_) !=
        VK_SUCCESS)
        return false;
    VkComputePipelineCreateInfo clusterCi = {};
    clusterCi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    clusterCi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    clusterCi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    clusterCi.stage.module = clusterAssignComp_;
    clusterCi.stage.pName = "main";
    clusterCi.layout = clusterPipelineLayout_;
    if (createComputePipeline(ctx, clusterCi, clusterPipeline_) != VK_SUCCESS)
        return false;

    // --- Froxel volumetric fog (Phase 5a): one compute pipeline per pass ------
    // A small helper builds each set-layout + push-range pipeline layout pair;
    // the push sizes differ per pass (VolFog*Push).
    struct VolFogPipeSpec {
        VkDescriptorSetLayout setLayout;
        uint32_t pushSize;
        VkShaderModule module;
        VkPipelineLayout* layoutOut;
        VkPipeline* pipelineOut;
    };
    const VolFogPipeSpec volfogSpecs[5] = {
        {volfogInjectSetLayout_, sizeof(VolFogInjectPush), volfogInjectComp_,
         &volfogInjectPipelineLayout_, &volfogInjectPipeline_},
        {volfogLightSetLayout_, sizeof(VolFogLightPush), volfogLightComp_,
         &volfogLightPipelineLayout_, &volfogLightPipeline_},
        {volfogTemporalSetLayout_, sizeof(VolFogTemporalPush), volfogTemporalComp_,
         &volfogTemporalPipelineLayout_, &volfogTemporalPipeline_},
        {volfogMarchSetLayout_, sizeof(VolFogMarchPush), volfogMarchComp_,
         &volfogMarchPipelineLayout_, &volfogMarchPipeline_},
        {volfogCompositeSetLayout_, sizeof(VolFogCompositePush), volfogCompositeComp_,
         &volfogCompositePipelineLayout_, &volfogCompositePipeline_},
    };
    for (const VolFogPipeSpec& spec : volfogSpecs) {
        VkPushConstantRange volfogPush = {};
        volfogPush.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        volfogPush.offset = 0;
        volfogPush.size = spec.pushSize;
        VkPipelineLayoutCreateInfo layoutCi = {};
        layoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutCi.setLayoutCount = 1;
        layoutCi.pSetLayouts = &spec.setLayout;
        layoutCi.pushConstantRangeCount = 1;
        layoutCi.pPushConstantRanges = &volfogPush;
        if (vkCreatePipelineLayout(ctx.device, &layoutCi, nullptr, spec.layoutOut) != VK_SUCCESS)
            return false;
        VkComputePipelineCreateInfo pipeCi = {};
        pipeCi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeCi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeCi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeCi.stage.module = spec.module;
        pipeCi.stage.pName = "main";
        pipeCi.layout = *spec.layoutOut;
        if (createComputePipeline(ctx, pipeCi, *spec.pipelineOut) != VK_SUCCESS)
            return false;
    }

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
    if (createGraphicsPipeline(ctx, shadowCi, shadowPipeline_) != VK_SUCCESS)
        return false;

    // Skinned variant: same depth-only state, skinned vertex input + shader.
    shadowStages[0].module = shadowDepthSkinnedVert_;
    shadowCi.pVertexInputState = &skinnedVertexInput;
    return createGraphicsPipeline(ctx, shadowCi, shadowSkinnedPipeline_) == VK_SUCCESS;
}

void DeferredCore::destroy(const VulkanContext& ctx) {
    if (!ctx.device) return;
    if (gbufferPipeline_) { vkDestroyPipeline(ctx.device, gbufferPipeline_, nullptr); gbufferPipeline_ = VK_NULL_HANDLE; }
    if (gbufferGtPipeline_) { vkDestroyPipeline(ctx.device, gbufferGtPipeline_, nullptr); gbufferGtPipeline_ = VK_NULL_HANDLE; }
    if (gbufferSkinnedPipeline_) { vkDestroyPipeline(ctx.device, gbufferSkinnedPipeline_, nullptr); gbufferSkinnedPipeline_ = VK_NULL_HANDLE; }
    if (gbufferSkinnedGtPipeline_) { vkDestroyPipeline(ctx.device, gbufferSkinnedGtPipeline_, nullptr); gbufferSkinnedGtPipeline_ = VK_NULL_HANDLE; }
    if (lightingPipeline_) { vkDestroyPipeline(ctx.device, lightingPipeline_, nullptr); lightingPipeline_ = VK_NULL_HANDLE; }
    if (transparentPipeline_) { vkDestroyPipeline(ctx.device, transparentPipeline_, nullptr); transparentPipeline_ = VK_NULL_HANDLE; }
    if (transparentGtPipeline_) { vkDestroyPipeline(ctx.device, transparentGtPipeline_, nullptr); transparentGtPipeline_ = VK_NULL_HANDLE; }
    if (ssaoPipeline_) { vkDestroyPipeline(ctx.device, ssaoPipeline_, nullptr); ssaoPipeline_ = VK_NULL_HANDLE; }
    if (ssaoBlurPipeline_) { vkDestroyPipeline(ctx.device, ssaoBlurPipeline_, nullptr); ssaoBlurPipeline_ = VK_NULL_HANDLE; }
    if (ssaoTemporalPipeline_) { vkDestroyPipeline(ctx.device, ssaoTemporalPipeline_, nullptr); ssaoTemporalPipeline_ = VK_NULL_HANDLE; }
    if (hizPipeline_) { vkDestroyPipeline(ctx.device, hizPipeline_, nullptr); hizPipeline_ = VK_NULL_HANDLE; }
    if (colorDownsamplePipeline_) { vkDestroyPipeline(ctx.device, colorDownsamplePipeline_, nullptr); colorDownsamplePipeline_ = VK_NULL_HANDLE; }
    if (ssrPipeline_) { vkDestroyPipeline(ctx.device, ssrPipeline_, nullptr); ssrPipeline_ = VK_NULL_HANDLE; }
    if (ssrTemporalPipeline_) { vkDestroyPipeline(ctx.device, ssrTemporalPipeline_, nullptr); ssrTemporalPipeline_ = VK_NULL_HANDLE; }
    if (bloomExtractPipeline_) { vkDestroyPipeline(ctx.device, bloomExtractPipeline_, nullptr); bloomExtractPipeline_ = VK_NULL_HANDLE; }
    if (bloomDownsamplePipeline_) { vkDestroyPipeline(ctx.device, bloomDownsamplePipeline_, nullptr); bloomDownsamplePipeline_ = VK_NULL_HANDLE; }
    if (bloomUpsamplePipeline_) { vkDestroyPipeline(ctx.device, bloomUpsamplePipeline_, nullptr); bloomUpsamplePipeline_ = VK_NULL_HANDLE; }
    if (bloomCompositePipeline_) { vkDestroyPipeline(ctx.device, bloomCompositePipeline_, nullptr); bloomCompositePipeline_ = VK_NULL_HANDLE; }
    if (histogramPipeline_) { vkDestroyPipeline(ctx.device, histogramPipeline_, nullptr); histogramPipeline_ = VK_NULL_HANDLE; }
    if (exposureSolvePipeline_) { vkDestroyPipeline(ctx.device, exposureSolvePipeline_, nullptr); exposureSolvePipeline_ = VK_NULL_HANDLE; }
    if (shadowPipeline_) { vkDestroyPipeline(ctx.device, shadowPipeline_, nullptr); shadowPipeline_ = VK_NULL_HANDLE; }
    if (shadowSkinnedPipeline_) { vkDestroyPipeline(ctx.device, shadowSkinnedPipeline_, nullptr); shadowSkinnedPipeline_ = VK_NULL_HANDLE; }
    if (clusterPipeline_) { vkDestroyPipeline(ctx.device, clusterPipeline_, nullptr); clusterPipeline_ = VK_NULL_HANDLE; }
    if (volfogInjectPipeline_) { vkDestroyPipeline(ctx.device, volfogInjectPipeline_, nullptr); volfogInjectPipeline_ = VK_NULL_HANDLE; }
    if (volfogLightPipeline_) { vkDestroyPipeline(ctx.device, volfogLightPipeline_, nullptr); volfogLightPipeline_ = VK_NULL_HANDLE; }
    if (volfogTemporalPipeline_) { vkDestroyPipeline(ctx.device, volfogTemporalPipeline_, nullptr); volfogTemporalPipeline_ = VK_NULL_HANDLE; }
    if (volfogMarchPipeline_) { vkDestroyPipeline(ctx.device, volfogMarchPipeline_, nullptr); volfogMarchPipeline_ = VK_NULL_HANDLE; }
    if (volfogCompositePipeline_) { vkDestroyPipeline(ctx.device, volfogCompositePipeline_, nullptr); volfogCompositePipeline_ = VK_NULL_HANDLE; }
    if (mbTilePipeline_) { vkDestroyPipeline(ctx.device, mbTilePipeline_, nullptr); mbTilePipeline_ = VK_NULL_HANDLE; }
    if (mbNeighborPipeline_) { vkDestroyPipeline(ctx.device, mbNeighborPipeline_, nullptr); mbNeighborPipeline_ = VK_NULL_HANDLE; }
    if (mbGatherPipeline_) { vkDestroyPipeline(ctx.device, mbGatherPipeline_, nullptr); mbGatherPipeline_ = VK_NULL_HANDLE; }
    if (dofCocPipeline_) { vkDestroyPipeline(ctx.device, dofCocPipeline_, nullptr); dofCocPipeline_ = VK_NULL_HANDLE; }
    if (dofGatherPipeline_) { vkDestroyPipeline(ctx.device, dofGatherPipeline_, nullptr); dofGatherPipeline_ = VK_NULL_HANDLE; }
    if (dofCompositePipeline_) { vkDestroyPipeline(ctx.device, dofCompositePipeline_, nullptr); dofCompositePipeline_ = VK_NULL_HANDLE; }
    if (postFxCopybackPipeline_) { vkDestroyPipeline(ctx.device, postFxCopybackPipeline_, nullptr); postFxCopybackPipeline_ = VK_NULL_HANDLE; }
    if (mbTilePipelineLayout_) { vkDestroyPipelineLayout(ctx.device, mbTilePipelineLayout_, nullptr); mbTilePipelineLayout_ = VK_NULL_HANDLE; }
    if (mbGatherPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, mbGatherPipelineLayout_, nullptr); mbGatherPipelineLayout_ = VK_NULL_HANDLE; }
    if (dofCocPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, dofCocPipelineLayout_, nullptr); dofCocPipelineLayout_ = VK_NULL_HANDLE; }
    if (dofGatherPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, dofGatherPipelineLayout_, nullptr); dofGatherPipelineLayout_ = VK_NULL_HANDLE; }
    if (dofCompositePipelineLayout_) { vkDestroyPipelineLayout(ctx.device, dofCompositePipelineLayout_, nullptr); dofCompositePipelineLayout_ = VK_NULL_HANDLE; }
    if (postFxCopybackPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, postFxCopybackPipelineLayout_, nullptr); postFxCopybackPipelineLayout_ = VK_NULL_HANDLE; }
    if (mbTileSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, mbTileSetLayout_, nullptr); mbTileSetLayout_ = VK_NULL_HANDLE; }
    if (mbGatherSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, mbGatherSetLayout_, nullptr); mbGatherSetLayout_ = VK_NULL_HANDLE; }
    if (dofCocSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, dofCocSetLayout_, nullptr); dofCocSetLayout_ = VK_NULL_HANDLE; }
    if (dofGatherSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, dofGatherSetLayout_, nullptr); dofGatherSetLayout_ = VK_NULL_HANDLE; }
    if (dofCompositeSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, dofCompositeSetLayout_, nullptr); dofCompositeSetLayout_ = VK_NULL_HANDLE; }
    if (postFxCopybackSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, postFxCopybackSetLayout_, nullptr); postFxCopybackSetLayout_ = VK_NULL_HANDLE; }
    if (scenePipelineLayout_) { vkDestroyPipelineLayout(ctx.device, scenePipelineLayout_, nullptr); scenePipelineLayout_ = VK_NULL_HANDLE; }
    if (lightingPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, lightingPipelineLayout_, nullptr); lightingPipelineLayout_ = VK_NULL_HANDLE; }
    if (transparentPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, transparentPipelineLayout_, nullptr); transparentPipelineLayout_ = VK_NULL_HANDLE; }
    if (ssaoPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, ssaoPipelineLayout_, nullptr); ssaoPipelineLayout_ = VK_NULL_HANDLE; }
    if (ssaoBlurPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, ssaoBlurPipelineLayout_, nullptr); ssaoBlurPipelineLayout_ = VK_NULL_HANDLE; }
    if (ssaoTemporalPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, ssaoTemporalPipelineLayout_, nullptr); ssaoTemporalPipelineLayout_ = VK_NULL_HANDLE; }
    if (hizPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, hizPipelineLayout_, nullptr); hizPipelineLayout_ = VK_NULL_HANDLE; }
    if (ssrPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, ssrPipelineLayout_, nullptr); ssrPipelineLayout_ = VK_NULL_HANDLE; }
    if (ssrTemporalPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, ssrTemporalPipelineLayout_, nullptr); ssrTemporalPipelineLayout_ = VK_NULL_HANDLE; }
    if (bloomPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, bloomPipelineLayout_, nullptr); bloomPipelineLayout_ = VK_NULL_HANDLE; }
    if (histogramPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, histogramPipelineLayout_, nullptr); histogramPipelineLayout_ = VK_NULL_HANDLE; }
    if (exposureSolvePipelineLayout_) { vkDestroyPipelineLayout(ctx.device, exposureSolvePipelineLayout_, nullptr); exposureSolvePipelineLayout_ = VK_NULL_HANDLE; }
    if (clusterPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, clusterPipelineLayout_, nullptr); clusterPipelineLayout_ = VK_NULL_HANDLE; }
    if (volfogInjectPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, volfogInjectPipelineLayout_, nullptr); volfogInjectPipelineLayout_ = VK_NULL_HANDLE; }
    if (volfogLightPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, volfogLightPipelineLayout_, nullptr); volfogLightPipelineLayout_ = VK_NULL_HANDLE; }
    if (volfogTemporalPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, volfogTemporalPipelineLayout_, nullptr); volfogTemporalPipelineLayout_ = VK_NULL_HANDLE; }
    if (volfogMarchPipelineLayout_) { vkDestroyPipelineLayout(ctx.device, volfogMarchPipelineLayout_, nullptr); volfogMarchPipelineLayout_ = VK_NULL_HANDLE; }
    if (volfogCompositePipelineLayout_) { vkDestroyPipelineLayout(ctx.device, volfogCompositePipelineLayout_, nullptr); volfogCompositePipelineLayout_ = VK_NULL_HANDLE; }
    if (gbufferVert_) { vkDestroyShaderModule(ctx.device, gbufferVert_, nullptr); gbufferVert_ = VK_NULL_HANDLE; }
    if (gbufferSkinnedVert_) { vkDestroyShaderModule(ctx.device, gbufferSkinnedVert_, nullptr); gbufferSkinnedVert_ = VK_NULL_HANDLE; }
    if (gbufferFrag_) { vkDestroyShaderModule(ctx.device, gbufferFrag_, nullptr); gbufferFrag_ = VK_NULL_HANDLE; }
    if (gbufferGtFrag_) { vkDestroyShaderModule(ctx.device, gbufferGtFrag_, nullptr); gbufferGtFrag_ = VK_NULL_HANDLE; }
    if (lightingFrag_) { vkDestroyShaderModule(ctx.device, lightingFrag_, nullptr); lightingFrag_ = VK_NULL_HANDLE; }
    if (fullscreenVert_) { vkDestroyShaderModule(ctx.device, fullscreenVert_, nullptr); fullscreenVert_ = VK_NULL_HANDLE; }
    if (transparentVert_) { vkDestroyShaderModule(ctx.device, transparentVert_, nullptr); transparentVert_ = VK_NULL_HANDLE; }
    if (transparentFrag_) { vkDestroyShaderModule(ctx.device, transparentFrag_, nullptr); transparentFrag_ = VK_NULL_HANDLE; }
    if (transparentGtFrag_) { vkDestroyShaderModule(ctx.device, transparentGtFrag_, nullptr); transparentGtFrag_ = VK_NULL_HANDLE; }
    if (ssaoComp_) { vkDestroyShaderModule(ctx.device, ssaoComp_, nullptr); ssaoComp_ = VK_NULL_HANDLE; }
    if (ssaoBlurComp_) { vkDestroyShaderModule(ctx.device, ssaoBlurComp_, nullptr); ssaoBlurComp_ = VK_NULL_HANDLE; }
    if (ssaoTemporalComp_) { vkDestroyShaderModule(ctx.device, ssaoTemporalComp_, nullptr); ssaoTemporalComp_ = VK_NULL_HANDLE; }
    if (hizDownsampleComp_) { vkDestroyShaderModule(ctx.device, hizDownsampleComp_, nullptr); hizDownsampleComp_ = VK_NULL_HANDLE; }
    if (colorDownsampleComp_) { vkDestroyShaderModule(ctx.device, colorDownsampleComp_, nullptr); colorDownsampleComp_ = VK_NULL_HANDLE; }
    if (ssrOpaqueComp_) { vkDestroyShaderModule(ctx.device, ssrOpaqueComp_, nullptr); ssrOpaqueComp_ = VK_NULL_HANDLE; }
    if (ssrTemporalComp_) { vkDestroyShaderModule(ctx.device, ssrTemporalComp_, nullptr); ssrTemporalComp_ = VK_NULL_HANDLE; }
    if (bloomExtractComp_) { vkDestroyShaderModule(ctx.device, bloomExtractComp_, nullptr); bloomExtractComp_ = VK_NULL_HANDLE; }
    if (bloomDownsampleComp_) { vkDestroyShaderModule(ctx.device, bloomDownsampleComp_, nullptr); bloomDownsampleComp_ = VK_NULL_HANDLE; }
    if (bloomUpsampleComp_) { vkDestroyShaderModule(ctx.device, bloomUpsampleComp_, nullptr); bloomUpsampleComp_ = VK_NULL_HANDLE; }
    if (bloomCompositeComp_) { vkDestroyShaderModule(ctx.device, bloomCompositeComp_, nullptr); bloomCompositeComp_ = VK_NULL_HANDLE; }
    if (exposureHistogramComp_) { vkDestroyShaderModule(ctx.device, exposureHistogramComp_, nullptr); exposureHistogramComp_ = VK_NULL_HANDLE; }
    if (exposureSolveComp_) { vkDestroyShaderModule(ctx.device, exposureSolveComp_, nullptr); exposureSolveComp_ = VK_NULL_HANDLE; }
    if (shadowDepthVert_) { vkDestroyShaderModule(ctx.device, shadowDepthVert_, nullptr); shadowDepthVert_ = VK_NULL_HANDLE; }
    if (shadowDepthSkinnedVert_) { vkDestroyShaderModule(ctx.device, shadowDepthSkinnedVert_, nullptr); shadowDepthSkinnedVert_ = VK_NULL_HANDLE; }
    if (shadowDepthFrag_) { vkDestroyShaderModule(ctx.device, shadowDepthFrag_, nullptr); shadowDepthFrag_ = VK_NULL_HANDLE; }
    if (clusterAssignComp_) { vkDestroyShaderModule(ctx.device, clusterAssignComp_, nullptr); clusterAssignComp_ = VK_NULL_HANDLE; }
    if (volfogInjectComp_) { vkDestroyShaderModule(ctx.device, volfogInjectComp_, nullptr); volfogInjectComp_ = VK_NULL_HANDLE; }
    if (volfogLightComp_) { vkDestroyShaderModule(ctx.device, volfogLightComp_, nullptr); volfogLightComp_ = VK_NULL_HANDLE; }
    if (volfogTemporalComp_) { vkDestroyShaderModule(ctx.device, volfogTemporalComp_, nullptr); volfogTemporalComp_ = VK_NULL_HANDLE; }
    if (volfogMarchComp_) { vkDestroyShaderModule(ctx.device, volfogMarchComp_, nullptr); volfogMarchComp_ = VK_NULL_HANDLE; }
    if (volfogCompositeComp_) { vkDestroyShaderModule(ctx.device, volfogCompositeComp_, nullptr); volfogCompositeComp_ = VK_NULL_HANDLE; }
    if (motionBlurTilemaxComp_) { vkDestroyShaderModule(ctx.device, motionBlurTilemaxComp_, nullptr); motionBlurTilemaxComp_ = VK_NULL_HANDLE; }
    if (motionBlurNeighborhoodComp_) { vkDestroyShaderModule(ctx.device, motionBlurNeighborhoodComp_, nullptr); motionBlurNeighborhoodComp_ = VK_NULL_HANDLE; }
    if (motionBlurGatherComp_) { vkDestroyShaderModule(ctx.device, motionBlurGatherComp_, nullptr); motionBlurGatherComp_ = VK_NULL_HANDLE; }
    if (dofCocComp_) { vkDestroyShaderModule(ctx.device, dofCocComp_, nullptr); dofCocComp_ = VK_NULL_HANDLE; }
    if (dofGatherComp_) { vkDestroyShaderModule(ctx.device, dofGatherComp_, nullptr); dofGatherComp_ = VK_NULL_HANDLE; }
    if (dofCompositeComp_) { vkDestroyShaderModule(ctx.device, dofCompositeComp_, nullptr); dofCompositeComp_ = VK_NULL_HANDLE; }
    if (postFxCopybackComp_) { vkDestroyShaderModule(ctx.device, postFxCopybackComp_, nullptr); postFxCopybackComp_ = VK_NULL_HANDLE; }
    if (sceneSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, sceneSetLayout_, nullptr); sceneSetLayout_ = VK_NULL_HANDLE; }
    if (textureSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, textureSetLayout_, nullptr); textureSetLayout_ = VK_NULL_HANDLE; }
    if (lightingSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, lightingSetLayout_, nullptr); lightingSetLayout_ = VK_NULL_HANDLE; }
    if (transparentSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, transparentSetLayout_, nullptr); transparentSetLayout_ = VK_NULL_HANDLE; }
    if (ssaoSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, ssaoSetLayout_, nullptr); ssaoSetLayout_ = VK_NULL_HANDLE; }
    if (ssaoBlurSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, ssaoBlurSetLayout_, nullptr); ssaoBlurSetLayout_ = VK_NULL_HANDLE; }
    if (ssaoTemporalSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, ssaoTemporalSetLayout_, nullptr); ssaoTemporalSetLayout_ = VK_NULL_HANDLE; }
    if (hizSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, hizSetLayout_, nullptr); hizSetLayout_ = VK_NULL_HANDLE; }
    if (ssrSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, ssrSetLayout_, nullptr); ssrSetLayout_ = VK_NULL_HANDLE; }
    if (ssrTemporalSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, ssrTemporalSetLayout_, nullptr); ssrTemporalSetLayout_ = VK_NULL_HANDLE; }
    if (exposureSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, exposureSetLayout_, nullptr); exposureSetLayout_ = VK_NULL_HANDLE; }
    if (clusterSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, clusterSetLayout_, nullptr); clusterSetLayout_ = VK_NULL_HANDLE; }
    if (volfogInjectSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, volfogInjectSetLayout_, nullptr); volfogInjectSetLayout_ = VK_NULL_HANDLE; }
    if (volfogLightSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, volfogLightSetLayout_, nullptr); volfogLightSetLayout_ = VK_NULL_HANDLE; }
    if (volfogTemporalSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, volfogTemporalSetLayout_, nullptr); volfogTemporalSetLayout_ = VK_NULL_HANDLE; }
    if (volfogMarchSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, volfogMarchSetLayout_, nullptr); volfogMarchSetLayout_ = VK_NULL_HANDLE; }
    if (volfogCompositeSetLayout_) { vkDestroyDescriptorSetLayout(ctx.device, volfogCompositeSetLayout_, nullptr); volfogCompositeSetLayout_ = VK_NULL_HANDLE; }
    if (textureSampler_) { vkDestroySampler(ctx.device, textureSampler_, nullptr); textureSampler_ = VK_NULL_HANDLE; }
    if (gbufferSampler_) { vkDestroySampler(ctx.device, gbufferSampler_, nullptr); gbufferSampler_ = VK_NULL_HANDLE; }
    if (shadowSampler_) { vkDestroySampler(ctx.device, shadowSampler_, nullptr); shadowSampler_ = VK_NULL_HANDLE; }
    if (hizSampler_) { vkDestroySampler(ctx.device, hizSampler_, nullptr); hizSampler_ = VK_NULL_HANDLE; }
    if (colorPyramidSampler_) { vkDestroySampler(ctx.device, colorPyramidSampler_, nullptr); colorPyramidSampler_ = VK_NULL_HANDLE; }
    ibl_.destroy(ctx);
    sky_.destroy(ctx);
    probes_.destroy(ctx);
    atmosphereSky_ = false;
}

bool DeferredCore::updateAtmosphereSky(const VulkanContext& ctx, const Vec3& sunDir) {
    if (!atmosphereSky_) return false;
    return ibl_.updateAtmosphereSky(ctx, sky_, sunDir);
}

void DeferredCore::fillSceneUBO(SceneUBO& out, const Scene& scene, const Camera& camera,
                                const Mat4& view, const Mat4& proj, const Mat4& projJittered,
                                const Mat4& prevViewProj, uint32_t renderW, uint32_t renderH,
                                float jitterX, float jitterY, bool jitter) const {
    const Mat4 viewProj = Mat4::multiply(projJittered, view);
    const Mat4 viewProjNoJitter = Mat4::multiply(proj, view);
    (void)scene; // UBO contents derive from camera/matrices only; scene unused.
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
                                   const Mat4& viewProj, const Mat4& invViewProj,
                                   const std::vector<Light>* overrideLights,
                                   const ShadowFrame* shadow, float iblIntensity) const {
    std::memcpy(out.invViewProj, invViewProj.m, sizeof(out.invViewProj));
    std::memcpy(out.viewProj, viewProj.m, sizeof(out.viewProj));

    out.cameraPos[0] = camera.position.x;
    out.cameraPos[1] = camera.position.y;
    out.cameraPos[2] = camera.position.z;
    out.cameraPos[3] = 1.f;

    // Pack into the fixed-size legacy GPU array: the shadowed sun (if any) is
    // always slot 0 so CSM lightIndex remaps cleanly; the remaining lights
    // follow scene order (no scoring — the clustered path consumes the full
    // light set from the SSBO, so this array only feeds the forward
    // transparency pass and the sun; see fillClusterLights).
    const std::vector<Light>& lights = effectiveLights(scene, overrideLights);

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

    std::vector<int> order;
    order.reserve(std::min<size_t>(kMaxLights, lights.size()));
    if (sunSrc >= 0) order.push_back(sunSrc);
    for (int i = 0; i < static_cast<int>(lights.size()) && order.size() < kMaxLights; ++i) {
        if (i != sunSrc) order.push_back(i);
    }

    const uint32_t count = static_cast<uint32_t>(order.size());
    std::memset(out.lights, 0, sizeof(out.lights)); // deterministic unused slots
    for (uint32_t i = 0; i < count; ++i) {
        const Light& l = lights[static_cast<size_t>(order[static_cast<size_t>(i)])];
        packLightGpu(l, out.lights[i]);
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
    // view-space depth for cascade selection and cluster slicing; fill it even
    // with shadows off.
    out.viewForward[0] = camera.forward.x;
    out.viewForward[1] = camera.forward.y;
    out.viewForward[2] = camera.forward.z;

    // Exponential cluster depth slicing range (cluster_assign.comp builds the
    // same slice bounds from the near/far it gets via push constants).
    out.clusterDepth[0] = camera.nearPlane;
    out.clusterDepth[1] = camera.farPlane;
    out.clusterDepth[2] = 0.f;
    out.clusterDepth[3] = 0.f;

    // The rasterizer depth-bias values are mirrored here so hosts can feed the
    // same constants to vkCmdSetDepthBias in recordShadowPass.
    out.shadowParams[0] = kShadowDepthBiasConstant;
    out.shadowParams[1] = kShadowDepthBiasSlope;
    // The sun shadow index is armed only when the packed slot-0 sun actually
    // casts (a frame with spot-atlas tiles but no casting sun must not enable
    // CSM sampling on a non-casting directional).
    const bool sunShadowed =
        shadow && sunSrc >= 0 && lights[static_cast<size_t>(sunSrc)].castShadow;
    out.shadowAtlasParams[1] = 1.f / static_cast<float>(kShadowAtlasSize);
    // Contact-shadow enable (w) is a host option, not scene state: the hosts
    // overwrite it after this call (Renderer/CompareApp/GuiApp wrappers).
    out.shadowAtlasParams[3] = 0.f;
    if (shadow) {
        for (uint32_t i = 0; i < kShadowCascadeCount; ++i) {
            std::memcpy(out.cascadeVp[i], shadow->cascadeVp[i].m, sizeof(out.cascadeVp[i]));
            out.cascadeSplits[i] = shadow->splitDepth[i];
        }
        for (uint32_t i = 0; i < kShadowAtlasTiles; ++i)
            std::memcpy(out.shadowTileVp[i], shadow->atlasVp[i].m, sizeof(out.shadowTileVp[i]));
        out.shadowAtlasParams[0] = static_cast<float>(shadow->atlasTileCount);
        out.shadowAtlasParams[2] = static_cast<float>(shadow->frameIndex);
        out.shadowParams[2] = 1.f; // shadows enabled
        out.shadowParams[3] = shadow->debugCascades ? 1.f : 0.f;
        // Packed index: the sun is always slot 0 when present.
        out.viewForward[3] = sunShadowed ? 0.f : -1.f;
    } else {
        // Deterministic disabled state: identity VPs, infinite splits; the
        // shaders short-circuit on shadowParams.z before touching the map.
        const Mat4 identity = Mat4::identity();
        for (uint32_t i = 0; i < kShadowCascadeCount; ++i) {
            std::memcpy(out.cascadeVp[i], identity.m, sizeof(out.cascadeVp[i]));
            out.cascadeSplits[i] = 1e9f;
        }
        for (uint32_t i = 0; i < kShadowAtlasTiles; ++i)
            std::memcpy(out.shadowTileVp[i], identity.m, sizeof(out.shadowTileVp[i]));
        out.shadowAtlasParams[0] = 0.f;
        out.shadowAtlasParams[2] = 0.f;
        out.shadowParams[2] = 0.f;
        out.shadowParams[3] = 0.f;
        out.viewForward[3] = -1.f;
    }
}

const std::vector<Light>& DeferredCore::effectiveLights(const Scene& scene,
                                                        const std::vector<Light>* overrideLights) {
    return overrideLights ? *overrideLights
                          : (scene.lights.empty() ? defaultLights() : scene.lights);
}

uint32_t DeferredCore::fillClusterLights(void* mappedLightsSsbo,
                                         const std::vector<Light>& lights) const {
    // std430: uvec4 header (x = packed point/spot count) + LightGPU[].
    // Directionals bypass the clusters (lighting.frag shades them from the
    // LightingUBO directly); the full untruncated point/spot set goes here —
    // per-cluster culling in cluster_assign.comp replaces the old CPU-side
    // intensity/distance^2 top-16 scoring.
    auto* header = static_cast<uint32_t*>(mappedLightsSsbo);
    auto* gpu = reinterpret_cast<LightGPU*>(header + 4);
    uint32_t count = 0;
    for (const Light& l : lights) {
        if (l.type == LightType::Directional) continue;
        if (count >= kMaxSceneLights) break;
        packLightGpu(l, gpu[count]);
        ++count;
    }
    header[0] = count;
    header[1] = header[2] = header[3] = 0;
    return count;
}

bool DeferredCore::createClusterGrid(const VulkanContext& ctx, uint32_t w, uint32_t h,
                                     ClusterGrid& out) const {
    out.gridX = (w + kClusterTileSize - 1) / kClusterTileSize;
    out.gridY = (h + kClusterTileSize - 1) / kClusterTileSize;
    out.gridZ = kClusterSlicesZ;
    out.clusterCount = out.gridX * out.gridY * out.gridZ;
    const VkDeviceSize lightsSize = 16 + static_cast<VkDeviceSize>(kMaxSceneLights) * sizeof(LightGPU);
    // Grid buffer: uvec4 header + counts[N] + indices[N * kMaxLightsPerCluster].
    const VkDeviceSize gridSize =
        16 + static_cast<VkDeviceSize>(out.clusterCount) * (1 + kMaxLightsPerCluster) * 4;
    for (uint32_t slot = 0; slot < kClusterSlots; ++slot) {
        if (createBuffer(ctx, lightsSize,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         out.lightsBuffer[slot], out.lightsMemory[slot]) != VK_SUCCESS)
            return false;
        vmaMapMemory(ctx.allocator, out.lightsMemory[slot], &out.lightsMapped[slot]);
        if (createBuffer(ctx, gridSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out.gridBuffer[slot],
                         out.gridMemory[slot]) != VK_SUCCESS)
            return false;

        // Grid header (dimensions + list capacity) never changes per frame;
        // upload it once through a staging buffer.  The compute pass writes
        // only the counts/indices region.
        const uint32_t headerData[4] = {out.gridX, out.gridY, out.gridZ, kMaxLightsPerCluster};
        VkBuffer staging = VK_NULL_HANDLE;
        VmaAllocation stagingMemory = VK_NULL_HANDLE;
        if (createBuffer(ctx, sizeof(headerData), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         staging, stagingMemory) != VK_SUCCESS)
            return false;
        void* mapped = nullptr;
        vmaMapMemory(ctx.allocator, stagingMemory, &mapped);
        std::memcpy(mapped, headerData, sizeof(headerData));
        vmaUnmapMemory(ctx.allocator, stagingMemory);
        VkBuffer gridBuffer = out.gridBuffer[slot];
        submitOneShot(ctx, [&](VkCommandBuffer cmd) {
            VkBufferCopy region = {};
            region.size = sizeof(headerData);
            vkCmdCopyBuffer(cmd, staging, gridBuffer, 1, &region);
            // Transfer write -> the assignment pass's storage read/write and
            // the lighting fragment shader's storage read.
            VkBufferMemoryBarrier2 bar = {};
            bar.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            bar.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            bar.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            bar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            bar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bar.buffer = gridBuffer;
            bar.offset = 0;
            bar.size = VK_WHOLE_SIZE;
            VkDependencyInfo dep = {};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.bufferMemoryBarrierCount = 1;
            dep.pBufferMemoryBarriers = &bar;
            vkCmdPipelineBarrier2(cmd, &dep);
        });
        vkDestroyBuffer(ctx.device, staging, nullptr);
        vmaFreeMemory(ctx.allocator, stagingMemory);
    }
    return true;
}

bool DeferredCore::writeClusterGridSets(const VulkanContext& ctx, VkDescriptorPool pool,
                                        ClusterGrid& grid) const {
    for (uint32_t slot = 0; slot < kClusterSlots; ++slot) {
        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = pool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &clusterSetLayout_;
        if (vkAllocateDescriptorSets(ctx.device, &alloc, &grid.assignSet[slot]) != VK_SUCCESS)
            return false;
        VkDescriptorBufferInfo bufs[2] = {};
        bufs[0].buffer = grid.lightsBuffer[slot];
        bufs[0].range = VK_WHOLE_SIZE;
        bufs[1].buffer = grid.gridBuffer[slot];
        bufs[1].range = VK_WHOLE_SIZE;
        VkWriteDescriptorSet writes[2] = {};
        for (uint32_t k = 0; k < 2; ++k) {
            writes[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[k].dstSet = grid.assignSet[slot];
            writes[k].dstBinding = k;
            writes[k].descriptorCount = 1;
            writes[k].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[k].pBufferInfo = &bufs[k];
        }
        vkUpdateDescriptorSets(ctx.device, 2, writes, 0, nullptr);
    }
    return true;
}

void DeferredCore::destroyClusterGrid(const VulkanContext& ctx, ClusterGrid& grid) const {
    for (uint32_t slot = 0; slot < kClusterSlots; ++slot) {
        if (grid.lightsBuffer[slot]) {
            vmaUnmapMemory(ctx.allocator, grid.lightsMemory[slot]);
            vkDestroyBuffer(ctx.device, grid.lightsBuffer[slot], nullptr);
            vmaFreeMemory(ctx.allocator, grid.lightsMemory[slot]);
            grid.lightsBuffer[slot] = VK_NULL_HANDLE;
            grid.lightsMemory[slot] = VK_NULL_HANDLE;
            grid.lightsMapped[slot] = nullptr;
        }
        if (grid.gridBuffer[slot]) {
            vkDestroyBuffer(ctx.device, grid.gridBuffer[slot], nullptr);
            vmaFreeMemory(ctx.allocator, grid.gridMemory[slot]);
            grid.gridBuffer[slot] = VK_NULL_HANDLE;
            grid.gridMemory[slot] = VK_NULL_HANDLE;
        }
        grid.assignSet[slot] = VK_NULL_HANDLE; // pool-owned
    }
    grid.clusterCount = 0;
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

void DeferredCore::writeSceneSkinBinding(const VulkanContext& ctx, VkDescriptorSet set,
                                         VkBuffer palette) const {
    VkDescriptorBufferInfo buf = {};
    buf.buffer = palette;
    buf.offset = 0;
    buf.range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet w = {};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = set;
    w.dstBinding = 2;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &buf;
    vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);
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
                                    VkImageView depth, VkImageView ssao, VkImageView shadow,
                                    VkImageView shadowAtlas,
                                    VkBuffer clusterLights, VkBuffer clusterGrid) const {
    VkDescriptorBufferInfo lightBuf = {};
    lightBuf.buffer = lightingUbo;
    lightBuf.offset = 0;
    lightBuf.range = sizeof(LightingUBO);

    VkDescriptorBufferInfo ssboBufs[2] = {};
    ssboBufs[0].buffer = clusterLights; // binding 12: all point/spot lights
    ssboBufs[0].range = VK_WHOLE_SIZE;
    ssboBufs[1].buffer = clusterGrid;   // binding 13: per-cluster light lists
    ssboBufs[1].range = VK_WHOLE_SIZE;

    VkDescriptorImageInfo img[12] = {};
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
    img[11].sampler = shadowSampler_;
    img[11].imageView = shadowAtlas;
    img[11].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet w[19] = {};
    uint32_t writeCount = 0;
    w[writeCount].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[writeCount].dstSet = set;
    w[writeCount].dstBinding = 0;
    w[writeCount].descriptorCount = 1;
    w[writeCount].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w[writeCount].pBufferInfo = &lightBuf;
    ++writeCount;
    // Bindings 1-10 are always written; 11 (CSM array) only with a shadow map.
    const uint32_t samplerCount = shadow ? 11u : 10u;
    for (uint32_t k = 0; k < samplerCount; ++k) {
        VkWriteDescriptorSet& s = w[writeCount++];
        s.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        s.dstSet = set;
        s.dstBinding = k + 1;
        s.descriptorCount = 1;
        s.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        s.pImageInfo = &img[k];
    }
    for (uint32_t k = 0; k < 2; ++k) {
        VkWriteDescriptorSet& s = w[writeCount++];
        s.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        s.dstSet = set;
        s.dstBinding = 12 + k;
        s.descriptorCount = 1;
        s.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        s.pBufferInfo = &ssboBufs[k];
    }
    // Binding 14 (spot shadow atlas) only with an atlas.
    if (shadowAtlas) {
        VkWriteDescriptorSet& s = w[writeCount++];
        s.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        s.dstSet = set;
        s.dstBinding = 14;
        s.descriptorCount = 1;
        s.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        s.pImageInfo = &img[11];
    }
    // Bindings 15-17: reflection probes (Phase 4c-2), always written — the
    // empty volume (count 0) keeps probe-less scenes on the global env path.
    VkDescriptorBufferInfo probeBuf = {};
    probeBuf.buffer = probes_.uboBuffer();
    probeBuf.offset = 0;
    probeBuf.range = sizeof(ProbeUBO);
    VkDescriptorImageInfo probeImg[2] = {};
    probeImg[0].sampler = probes_.sampler();
    probeImg[0].imageView = probes_.prefilterView();
    probeImg[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    probeImg[1].sampler = probes_.sampler();
    probeImg[1].imageView = probes_.irradianceView();
    probeImg[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    {
        VkWriteDescriptorSet& s = w[writeCount++];
        s.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        s.dstSet = set;
        s.dstBinding = 15;
        s.descriptorCount = 1;
        s.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        s.pBufferInfo = &probeBuf;
    }
    for (uint32_t k = 0; k < 2; ++k) {
        VkWriteDescriptorSet& s = w[writeCount++];
        s.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        s.dstSet = set;
        s.dstBinding = 16 + k;
        s.descriptorCount = 1;
        s.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        s.pImageInfo = &probeImg[k];
    }
    vkUpdateDescriptorSets(ctx.device, writeCount, w, 0, nullptr);
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
    // per-mesh firstIndex/vertexOffset.  (Null in fully-skinned scenes, which
    // only ever take the skinned branch below.)
    const VkDeviceSize zeroOffset = 0;
    if (scene.mergedVertexBuffer) {
        vkCmdBindVertexBuffers(cmd, 0, 1, &scene.mergedVertexBuffer, &zeroOffset);
        vkCmdBindIndexBuffer(cmd, scene.mergedIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
    }

    // Instances are sorted by (material, mesh) at load time, so state changes
    // collapse to one descriptor bind per material run.
    uint32_t lastMaterial = UINT32_MAX;
    bool skinnedBound = false;
    for (const auto& inst : scene.instances) {
        if (scene.materials[inst.materialIndex].blend) continue; // transparency pass draws these
        if (!aabbIntersectsFrustum(frustum, inst.aabbMin, inst.aabbMax)) continue;
        if (inst.lodCulled) continue; // LOD screen-size cull (updateLodSelection)

        if (inst.skinIndex >= 0) {
            // Skinned draw: own vertex buffers + palette offsets; the push
            // matrices are unused (the palette carries the node transform).
            if (!skinnedBound) {
                skinnedBound = true;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  gtPass ? gbufferSkinnedGtPipeline_ : gbufferSkinnedPipeline_);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        scenePipelineLayout_, 1, 1, &textureSet, 0, nullptr);
                lastMaterial = UINT32_MAX; // descriptor offsets were bound for the static pipeline
            }
            const Skin& skin = scene.skins[static_cast<size_t>(inst.skinIndex)];
            SkinnedScenePush push;
            push.paletteCur = skin.paletteCur;
            push.palettePrev = skin.palettePrev;
            vkCmdPushConstants(cmd, scenePipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0,
                               sizeof(push), &push);

            if (inst.materialIndex != lastMaterial) {
                lastMaterial = inst.materialIndex;
                const uint32_t dynOffset = inst.materialIndex * materialStride;
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipelineLayout_,
                                        0, 1, &sceneSet, 1, &dynOffset);
            }

            const Mesh& mesh = scene.skinnedMeshes[inst.meshIndex];
            vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertexBuffer, &zeroOffset);
            vkCmdBindIndexBuffer(cmd, mesh.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, mesh.indexCount, 1, 0, 0, 0);
            continue;
        }

        if (skinnedBound) {
            skinnedBound = false;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              gtPass ? gbufferGtPipeline_ : gbufferPipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipelineLayout_, 1,
                                    1, &textureSet, 0, nullptr);
            vkCmdBindVertexBuffers(cmd, 0, 1, &scene.mergedVertexBuffer, &zeroOffset);
            vkCmdBindIndexBuffer(cmd, scene.mergedIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
            lastMaterial = UINT32_MAX;
        }

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

        // LOD level chosen by Scene::updateLodSelection (lodDraws[0] = full mesh).
        const LodDraw& draw = inst.lodDraws[inst.lodLevel];
        vkCmdDrawIndexed(cmd, draw.indexCount, 1, draw.firstIndex, draw.vertexOffset, 0);
    }
}

void DeferredCore::recordLightingPass(VkCommandBuffer cmd, VkDescriptorSet lightingSet,
                                      ClusterGrid& grid, uint32_t slot, const Mat4& view,
                                      const Mat4& proj, VkImageView target, uint32_t width,
                                      uint32_t height) const {
    // --- Cluster light assignment (compute): per-cluster view-space AABB vs
    // every point/spot light, once per frame (Olsson et al. 2012 clustered
    // shading; DOOM 2016/Eternal SIGGRAPH course).  One invocation per cluster
    // loops the lights in SSBO order, so the lists are bit-deterministic.
    const uint32_t s = slot % kClusterSlots;
    ClusterAssignPush push = {};
    std::memcpy(push.view, view.m, sizeof(push.view));
    push.projParams[0] = proj.m[0]; // f / aspect
    push.projParams[1] = proj.m[5]; // -f (Vulkan y-flip; the shader only divides by it)
    // Recover near/far from the perspective projection (Mat4::perspective):
    // m[10] = f/(n-f), m[14] = n*f/(n-f)  =>  n = m[14]/m[10], f = n*m[10]/(1+m[10]).
    push.projParams[2] = proj.m[14] / proj.m[10];
    push.projParams[3] = push.projParams[2] * proj.m[10] / (1.f + proj.m[10]);
    push.grid[0] = grid.gridX;
    push.grid[1] = grid.gridY;
    push.grid[2] = grid.gridZ;
    push.grid[3] = kMaxLightsPerCluster;
    push.misc[0] = width;
    push.misc[1] = height;
    push.misc[2] = kClusterTileSize;
    push.misc[3] = 0;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, clusterPipeline_);
    vkCmdPushConstants(cmd, clusterPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(push), &push);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, clusterPipelineLayout_, 0, 1,
                            &grid.assignSet[s], 0, nullptr);
    vkCmdDispatch(cmd, (grid.clusterCount + 63) / 64, 1, 1);

    // Compute writes -> fragment storage reads in the lighting pass below.
    VkBufferMemoryBarrier2 clusterBar = {};
    clusterBar.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    clusterBar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    clusterBar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    clusterBar.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    clusterBar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    clusterBar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    clusterBar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    clusterBar.buffer = grid.gridBuffer[s];
    clusterBar.offset = 0;
    clusterBar.size = VK_WHOLE_SIZE;
    VkDependencyInfo clusterDep = {};
    clusterDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    clusterDep.bufferMemoryBarrierCount = 1;
    clusterDep.pBufferMemoryBarriers = &clusterBar;
    vkCmdPipelineBarrier2(cmd, &clusterDep);

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
                                       VkImageView depthPyramid) const {
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
    img[5].sampler = colorPyramidSampler_; // trilinear: roughness picks the mip
    img[5].imageView = ssrColor;
    img[5].imageLayout = VK_IMAGE_LAYOUT_GENERAL; // color pyramid lives in GENERAL
    img[6].sampler = hizSampler_;
    img[6].imageView = depthPyramid;
    img[6].imageLayout = VK_IMAGE_LAYOUT_GENERAL; // pyramid lives in GENERAL

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

void DeferredCore::writeSsaoSet(const VulkanContext& ctx, VkDescriptorSet set,
                                VkImageView depthChain, VkImageView normal,
                                VkImageView aoRaw) const {
    VkDescriptorImageInfo img[3] = {};
    img[0].sampler = hizSampler_; // texelFetch with explicit lod on the view-Z chain
    img[0].imageView = depthChain;
    img[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL; // pyramid images are GENERAL for life
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

void DeferredCore::recordSsaoPass(VkCommandBuffer cmd, VkDescriptorSet ssaoSet,
                                  const Mat4& invViewProj, uint32_t frameIndex, float nearZ,
                                  float farZ, float maxMipLod, uint32_t width,
                                  uint32_t height) const {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ssaoPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ssaoPipelineLayout_, 0, 1,
                            &ssaoSet, 0, nullptr);
    SsaoPush push;
    std::memcpy(push.invViewProj, invViewProj.m, sizeof(push.invViewProj));
    push.params[0] = kSsaoRadius;
    push.params[1] = kSsaoPower;
    // Non-reversed perspective unpack: viewZ = mul / (add - ndcDepth), the
    // inverse of d = far*(z-near) / (z*(far-near)).
    push.params[2] = nearZ * farZ / (farZ - nearZ);
    push.params[3] = farZ / (farZ - nearZ);
    push.params2[0] = static_cast<float>(frameIndex);
    push.params2[1] = maxMipLod;
    push.params2[2] = farZ;
    push.params2[3] = 0.f;
    vkCmdPushConstants(cmd, ssaoPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                       &push);
    vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);
}

void DeferredCore::recordSsaoTemporalPass(VkCommandBuffer cmd, const AoHistory& history,
                                          uint32_t writeIndex, const Mat4& invViewProj,
                                          const Mat4& prevViewProj, uint32_t width,
                                          uint32_t height, bool reset) const {
    if (!history.image[0]) return;
    const uint32_t i = writeIndex & 1u;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ssaoTemporalPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ssaoTemporalPipelineLayout_, 0, 1,
                            &history.temporalSet[i], 0, nullptr);
    // WAR: last frame's blur (and the temporal read two frames ago) sampled
    // this buffer before it becomes the storage target again.  Same-layout
    // barrier; the buffers are GENERAL for life.
    imageBarrier(cmd, history.image[i], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                 sync::kCompute, sync::kSampled, sync::kCompute, sync::kStorageWrite,
                 VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);
    SsaoTemporalPush push;
    std::memcpy(push.invViewProj, invViewProj.m, sizeof(push.invViewProj));
    std::memcpy(push.prevViewProj, prevViewProj.m, sizeof(push.prevViewProj));
    push.params[0] = kSsaoTemporalBlend;
    push.params[1] = reset ? 1.f : 0.f;
    push.params[2] = push.params[3] = 0.f;
    vkCmdPushConstants(cmd, ssaoTemporalPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(push), &push);
    vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);
    // This frame's blur samples the just-written buffer.
    imageBarrier(cmd, history.image[i], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                 sync::kCompute, sync::kStorageWrite, sync::kCompute, sync::kSampled,
                 VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);
}

void DeferredCore::recordSsaoBlurPass(VkCommandBuffer cmd, VkDescriptorSet blurSet, uint32_t width,
                                      uint32_t height) const {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ssaoBlurPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ssaoBlurPipelineLayout_, 0, 1,
                            &blurSet, 0, nullptr);
    vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);
}

bool DeferredCore::createAoHistory(const VulkanContext& ctx, uint32_t w, uint32_t h,
                                   AoHistory& out) const {
    out.width = w;
    out.height = h;
    for (uint32_t i = 0; i < 2; ++i) {
        if (createImage(ctx, w, h, kAoRawFormat,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, out.image[i],
                        out.memory[i], 1) != VK_SUCCESS)
            return false;
        out.view[i] = createImageView(ctx, out.image[i], kAoRawFormat, VK_IMAGE_ASPECT_COLOR_BIT,
                                      0, 1);
        if (!out.view[i]) return false;
    }
    // GENERAL for life (same model as the depth pyramids): each buffer
    // ping-pongs between temporal storage writes and temporal/blur reads.
    submitOneShot(ctx, [&](VkCommandBuffer cmd) {
        for (uint32_t i = 0; i < 2; ++i)
            imageBarrier(cmd, out.image[i], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);
    });
    return true;
}

bool DeferredCore::writeAoHistorySets(const VulkanContext& ctx, VkDescriptorPool pool,
                                      VkImageView aoRaw, VkImageView depth, VkImageView aoOut,
                                      AoHistory& out) const {
    for (uint32_t i = 0; i < 2; ++i) {
        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = pool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &ssaoTemporalSetLayout_;
        if (vkAllocateDescriptorSets(ctx.device, &alloc, &out.temporalSet[i]) != VK_SUCCESS)
            return false;
        alloc.pSetLayouts = &ssaoBlurSetLayout_;
        if (vkAllocateDescriptorSets(ctx.device, &alloc, &out.blurSet[i]) != VK_SUCCESS)
            return false;

        VkDescriptorImageInfo img[6] = {};
        // temporal: raw AO + history read (the OTHER buffer) + GBuffer depth
        img[0].sampler = gbufferSampler_;
        img[0].imageView = aoRaw;
        img[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        img[1].sampler = gbufferSampler_;
        img[1].imageView = out.view[1 - i];
        img[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        img[2].imageView = out.view[i]; // history write (storage)
        img[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        img[3].sampler = gbufferSampler_;
        img[3].imageView = depth;
        img[3].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        // blur: accumulated history -> the path's filtered AO target
        img[4].sampler = gbufferSampler_;
        img[4].imageView = out.view[i];
        img[4].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        img[5].imageView = aoOut;
        img[5].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet w[6] = {};
        for (uint32_t k = 0; k < 6; ++k) {
            w[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[k].dstSet = k < 4 ? out.temporalSet[i] : out.blurSet[i];
            w[k].dstBinding = k < 4 ? k : k - 4;
            w[k].descriptorCount = 1;
            w[k].descriptorType = (k == 2 || k == 5) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                     : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[k].pImageInfo = &img[k];
        }
        vkUpdateDescriptorSets(ctx.device, 6, w, 0, nullptr);
    }
    return true;
}

void DeferredCore::destroyAoHistory(const VulkanContext& ctx, AoHistory& history) const {
    for (uint32_t i = 0; i < 2; ++i) {
        if (history.view[i]) {
            vkDestroyImageView(ctx.device, history.view[i], nullptr);
            history.view[i] = VK_NULL_HANDLE;
        }
        if (history.image[i]) {
            vmaDestroyImage(ctx.allocator, history.image[i], history.memory[i]);
            history.image[i] = VK_NULL_HANDLE;
            history.memory[i] = VK_NULL_HANDLE;
        }
    }
    history.temporalSet[0] = history.temporalSet[1] = VK_NULL_HANDLE; // pool-owned
    history.blurSet[0] = history.blurSet[1] = VK_NULL_HANDLE;
    history.width = history.height = 0;
}

// --- Hi-Z depth pyramid -------------------------------------------------------

uint32_t DeferredCore::depthPyramidMipCount(uint32_t w, uint32_t h) {
    // Vulkan full-chain convention: level i extent is max(1, extent >> i), so
    // 1 + floor(log2(max(w, h))) levels end exactly at 1x1.
    uint32_t mips = 1;
    for (uint32_t d = std::max(w, h); d > 1; d >>= 1)
        ++mips;
    return mips;
}

bool DeferredCore::createDepthPyramid(const VulkanContext& ctx, uint32_t w, uint32_t h,
                                      DepthPyramid& out, bool aoFilter, float nearZ,
                                      float farZ) const {
    out.width = w;
    out.height = h;
    out.mipCount = depthPyramidMipCount(w, h);
    out.aoFilter = aoFilter;
    if (aoFilter) {
        // viewZ = mul / (add - ndcDepth); same unpack as SsaoPush.
        out.depthUnpack[0] = nearZ * farZ / (farZ - nearZ);
        out.depthUnpack[1] = farZ / (farZ - nearZ);
        // XeGTAO_DepthMIPFilter falloff (depthRangeScaleFactor 0.75, "found
        // empirically"): weights fade out texels closer than the far surface
        // by more than the effect falloff.
        const float effectRadius =
            kSsaoDepthMipRangeScale * kSsaoRadius * kSsaoRadiusMultiplier;
        const float falloffRange = kSsaoFalloffRange * effectRadius;
        const float falloffFrom = effectRadius * (1.0f - kSsaoFalloffRange);
        out.falloff[0] = -1.0f / falloffRange;
        out.falloff[1] = falloffFrom / falloffRange + 1.0f;
    }
    if (createImage(ctx, w, h, kDepthPyramidFormat,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, out.image,
                    out.memory, out.mipCount) != VK_SUCCESS)
        return false;
    out.chainView =
        createImageView(ctx, out.image, kDepthPyramidFormat, VK_IMAGE_ASPECT_COLOR_BIT, 0,
                        out.mipCount);
    if (!out.chainView) return false;
    out.mipViews.resize(out.mipCount, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < out.mipCount; ++i) {
        out.mipViews[i] = createImageView(ctx, out.image, kDepthPyramidFormat,
                                          VK_IMAGE_ASPECT_COLOR_BIT, i, 1);
        if (!out.mipViews[i]) return false;
    }
    // GENERAL for life: every mip is both a compute storage target and a
    // sampled source, so a fixed layout beats per-frame transitions.
    submitOneShot(ctx, [&](VkCommandBuffer cmd) {
        imageBarrier(cmd, out.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_IMAGE_ASPECT_COLOR_BIT, 0, out.mipCount);
    });
    return true;
}

bool DeferredCore::writeDepthPyramidSets(const VulkanContext& ctx, VkDescriptorPool pool,
                                         VkImageView srcDepth, DepthPyramid& out) const {
    out.sets.resize(out.mipCount, VK_NULL_HANDLE);
    for (uint32_t level = 0; level < out.mipCount; ++level) {
        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = pool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &hizSetLayout_;
        if (vkAllocateDescriptorSets(ctx.device, &alloc, &out.sets[level]) != VK_SUCCESS)
            return false;

        VkDescriptorImageInfo src = {};
        src.sampler = hizSampler_;
        src.imageView = level == 0 ? srcDepth : out.mipViews[level - 1];
        // The D32 source depth is sampled in SHADER_READ_ONLY; pyramid mips
        // are GENERAL.
        src.imageLayout = level == 0 ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                     : VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo dst = {};
        dst.imageView = out.mipViews[level];
        dst.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet w[2] = {};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = out.sets[level];
        w[0].dstBinding = 0;
        w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[0].pImageInfo = &src;
        w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].dstSet = out.sets[level];
        w[1].dstBinding = 1;
        w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[1].pImageInfo = &dst;
        vkUpdateDescriptorSets(ctx.device, 2, w, 0, nullptr);
    }
    return true;
}

void DeferredCore::destroyDepthPyramid(const VulkanContext& ctx, DepthPyramid& pyramid) const {
    for (VkImageView v : pyramid.mipViews)
        if (v) vkDestroyImageView(ctx.device, v, nullptr);
    pyramid.mipViews.clear();
    pyramid.sets.clear(); // pool-owned; freed with the host's descriptor pool
    if (pyramid.chainView) {
        vkDestroyImageView(ctx.device, pyramid.chainView, nullptr);
        pyramid.chainView = VK_NULL_HANDLE;
    }
    if (pyramid.image) {
        vmaDestroyImage(ctx.allocator, pyramid.image, pyramid.memory);
        pyramid.image = VK_NULL_HANDLE;
        pyramid.memory = VK_NULL_HANDLE;
    }
    pyramid.width = pyramid.height = pyramid.mipCount = 0;
}

void DeferredCore::recordDepthPyramidPass(VkCommandBuffer cmd, const DepthPyramid& pyramid) const {
    if (!pyramid.image || pyramid.sets.empty()) return;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hizPipeline_);

    // WAR: last frame's marcher (fragment) and downsample reads finish before
    // this frame rewrites mip 0.  Same-layout barrier; the image is GENERAL.
    imageBarrier(cmd, pyramid.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                 sync::kSampleStages, sync::kSampled, sync::kCompute, sync::kStorageWrite,
                 VK_IMAGE_ASPECT_COLOR_BIT, 0, pyramid.mipCount);

    uint32_t srcW = pyramid.width;
    uint32_t srcH = pyramid.height;
    for (uint32_t level = 0; level < pyramid.mipCount; ++level) {
        if (level > 0) {
            // Level-1 write must be visible before it is sampled as source.
            imageBarrier(cmd, pyramid.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                         sync::kCompute, sync::kStorageWrite, sync::kCompute, sync::kSampled,
                         VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 1);
        }
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hizPipelineLayout_, 0, 1,
                                &pyramid.sets[level], 0, nullptr);
        HiZPush push = {};
        push.srcSize[0] = static_cast<int32_t>(srcW);
        push.srcSize[1] = static_cast<int32_t>(srcH);
        push.aoFilter = pyramid.aoFilter ? 1 : 0;
        push.level = static_cast<int32_t>(level);
        push.depthUnpack[0] = pyramid.depthUnpack[0];
        push.depthUnpack[1] = pyramid.depthUnpack[1];
        push.falloff[0] = pyramid.falloff[0];
        push.falloff[1] = pyramid.falloff[1];
        vkCmdPushConstants(cmd, hizPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                           &push);
        const uint32_t dstW = std::max(1u, pyramid.width >> level);
        const uint32_t dstH = std::max(1u, pyramid.height >> level);
        vkCmdDispatch(cmd, (dstW + 7) / 8, (dstH + 7) / 8, 1);
        srcW = dstW;
        srcH = dstH;
    }

    // The opaque-SSR (compute) and transparent-pass (fragment) marchers sample
    // the chain later this frame.
    imageBarrier(cmd, pyramid.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                 sync::kCompute, sync::kStorageWrite, sync::kSampleStages, sync::kSampled,
                 VK_IMAGE_ASPECT_COLOR_BIT, 0, pyramid.mipCount);
}

// --- HDR color mip chain ------------------------------------------------------
// Same resource model as the depth pyramid above; only the format, the
// mip-0 semantics (copy instead of max-reduce of an external D32 source) and
// the pipeline differ.

bool DeferredCore::createColorPyramid(const VulkanContext& ctx, uint32_t w, uint32_t h,
                                      ColorPyramid& out) const {
    out.width = w;
    out.height = h;
    out.mipCount = depthPyramidMipCount(w, h); // same full-chain rule (down to 1x1)
    if (createImage(ctx, w, h, kColorPyramidFormat,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, out.image,
                    out.memory, out.mipCount) != VK_SUCCESS)
        return false;
    out.chainView =
        createImageView(ctx, out.image, kColorPyramidFormat, VK_IMAGE_ASPECT_COLOR_BIT, 0,
                        out.mipCount);
    if (!out.chainView) return false;
    out.mipViews.resize(out.mipCount, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < out.mipCount; ++i) {
        out.mipViews[i] = createImageView(ctx, out.image, kColorPyramidFormat,
                                          VK_IMAGE_ASPECT_COLOR_BIT, i, 1);
        if (!out.mipViews[i]) return false;
    }
    // GENERAL for life: every mip is both a compute storage target and a
    // sampled source, so a fixed layout beats per-frame transitions.
    submitOneShot(ctx, [&](VkCommandBuffer cmd) {
        imageBarrier(cmd, out.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_IMAGE_ASPECT_COLOR_BIT, 0, out.mipCount);
    });
    return true;
}

bool DeferredCore::writeColorPyramidSets(const VulkanContext& ctx, VkDescriptorPool pool,
                                         VkImageView srcColor, ColorPyramid& out) const {
    out.sets.resize(out.mipCount, VK_NULL_HANDLE);
    for (uint32_t level = 0; level < out.mipCount; ++level) {
        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = pool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &hizSetLayout_; // same binding shape as the Hi-Z downsample
        if (vkAllocateDescriptorSets(ctx.device, &alloc, &out.sets[level]) != VK_SUCCESS)
            return false;

        VkDescriptorImageInfo src = {};
        src.sampler = hizSampler_; // texelFetch only; filter state is irrelevant
        src.imageView = level == 0 ? srcColor : out.mipViews[level - 1];
        // The lit HDR source color is sampled in SHADER_READ_ONLY; pyramid
        // mips are GENERAL.
        src.imageLayout = level == 0 ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                     : VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo dst = {};
        dst.imageView = out.mipViews[level];
        dst.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet w[2] = {};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = out.sets[level];
        w[0].dstBinding = 0;
        w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[0].pImageInfo = &src;
        w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].dstSet = out.sets[level];
        w[1].dstBinding = 1;
        w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[1].pImageInfo = &dst;
        vkUpdateDescriptorSets(ctx.device, 2, w, 0, nullptr);
    }
    return true;
}

void DeferredCore::destroyColorPyramid(const VulkanContext& ctx, ColorPyramid& pyramid) const {
    for (VkImageView v : pyramid.mipViews)
        if (v) vkDestroyImageView(ctx.device, v, nullptr);
    pyramid.mipViews.clear();
    pyramid.sets.clear(); // pool-owned; freed with the host's descriptor pool
    if (pyramid.chainView) {
        vkDestroyImageView(ctx.device, pyramid.chainView, nullptr);
        pyramid.chainView = VK_NULL_HANDLE;
    }
    if (pyramid.image) {
        vmaDestroyImage(ctx.allocator, pyramid.image, pyramid.memory);
        pyramid.image = VK_NULL_HANDLE;
        pyramid.memory = VK_NULL_HANDLE;
    }
    pyramid.width = pyramid.height = pyramid.mipCount = 0;
}

void DeferredCore::recordColorPyramidPass(VkCommandBuffer cmd, const ColorPyramid& pyramid) const {
    if (!pyramid.image || pyramid.sets.empty()) return;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, colorDownsamplePipeline_);

    // WAR: last frame's marcher (fragment) and downsample reads finish before
    // this frame rewrites mip 0.  Same-layout barrier; the image is GENERAL.
    imageBarrier(cmd, pyramid.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                 sync::kSampleStages, sync::kSampled, sync::kCompute, sync::kStorageWrite,
                 VK_IMAGE_ASPECT_COLOR_BIT, 0, pyramid.mipCount);

    uint32_t srcW = pyramid.width;
    uint32_t srcH = pyramid.height;
    for (uint32_t level = 0; level < pyramid.mipCount; ++level) {
        if (level > 0) {
            // Level-1 write must be visible before it is sampled as source.
            imageBarrier(cmd, pyramid.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                         sync::kCompute, sync::kStorageWrite, sync::kCompute, sync::kSampled,
                         VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 1);
        }
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hizPipelineLayout_, 0, 1,
                                &pyramid.sets[level], 0, nullptr);
        // Only srcSize is read by color_downsample.comp; zero the rest of the
        // (now wider) HiZPush so no uninitialised bytes are pushed.
        HiZPush push = {};
        push.srcSize[0] = static_cast<int32_t>(srcW);
        push.srcSize[1] = static_cast<int32_t>(srcH);
        vkCmdPushConstants(cmd, hizPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                           &push);
        const uint32_t dstW = std::max(1u, pyramid.width >> level);
        const uint32_t dstH = std::max(1u, pyramid.height >> level);
        vkCmdDispatch(cmd, (dstW + 7) / 8, (dstH + 7) / 8, 1);
        srcW = dstW;
        srcH = dstH;
    }

    // The opaque-SSR (compute) and transparent-pass (fragment) marchers sample
    // the chain later this frame.
    imageBarrier(cmd, pyramid.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                 sync::kCompute, sync::kStorageWrite, sync::kSampleStages, sync::kSampled,
                 VK_IMAGE_ASPECT_COLOR_BIT, 0, pyramid.mipCount);
}

// --- Opaque SSR (fullscreen compute, after lighting + color pyramid) ----------

void DeferredCore::writeSsrSet(const VulkanContext& ctx, VkDescriptorSet set,
                               VkBuffer lightingUbo, VkImageView albedo, VkImageView normal,
                               VkImageView material, VkImageView depth, VkImageView ssao,
                               VkImageView ssrColor, VkImageView depthPyramid,
                               VkImageView ssrTraceOut) const {
    VkDescriptorBufferInfo ubo = {};
    ubo.buffer = lightingUbo;
    ubo.offset = 0;
    ubo.range = sizeof(LightingUBO);

    VkDescriptorImageInfo img[9] = {};
    img[0].sampler = gbufferSampler_;
    img[0].imageView = albedo;
    img[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    img[1].sampler = gbufferSampler_;
    img[1].imageView = normal;
    img[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    img[2].sampler = gbufferSampler_;
    img[2].imageView = material;
    img[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    img[3].sampler = gbufferSampler_;
    img[3].imageView = depth;
    img[3].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    img[4].sampler = gbufferSampler_;
    img[4].imageView = ssao;
    img[4].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    img[5].sampler = ibl_.cubeSampler;
    img[5].imageView = ibl_.prefilterView;
    img[5].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    img[6].sampler = ibl_.lutSampler;
    img[6].imageView = ibl_.brdfLutView;
    img[6].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    img[7].sampler = colorPyramidSampler_; // trilinear: roughness picks the mip
    img[7].imageView = ssrColor;
    img[7].imageLayout = VK_IMAGE_LAYOUT_GENERAL; // color pyramid lives in GENERAL
    img[8].sampler = hizSampler_;
    img[8].imageView = depthPyramid;
    img[8].imageLayout = VK_IMAGE_LAYOUT_GENERAL; // pyramid lives in GENERAL

    VkDescriptorImageInfo storage = {};
    storage.imageView = ssrTraceOut;
    storage.imageLayout = VK_IMAGE_LAYOUT_GENERAL; // write-only trace target

    // Bindings 11-13: reflection probes (Phase 4c-2), always written; the
    // recomputed specIbl must use the same probe fallback chain as
    // lighting.frag or the delta composite would double-count / leak energy.
    VkDescriptorBufferInfo probeBuf = {};
    probeBuf.buffer = probes_.uboBuffer();
    probeBuf.offset = 0;
    probeBuf.range = sizeof(ProbeUBO);
    VkDescriptorImageInfo probeImg[2] = {};
    probeImg[0].sampler = probes_.sampler();
    probeImg[0].imageView = probes_.prefilterView();
    probeImg[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    probeImg[1].sampler = probes_.sampler();
    probeImg[1].imageView = probes_.irradianceView();
    probeImg[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet w[14] = {};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = set;
    w[0].dstBinding = 0;
    w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w[0].pBufferInfo = &ubo;
    for (uint32_t k = 0; k < 9; ++k) {
        w[k + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[k + 1].dstSet = set;
        w[k + 1].dstBinding = k + 1;
        w[k + 1].descriptorCount = 1;
        w[k + 1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[k + 1].pImageInfo = &img[k];
    }
    w[10].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[10].dstSet = set;
    w[10].dstBinding = 10;
    w[10].descriptorCount = 1;
    w[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w[10].pImageInfo = &storage;
    w[11].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[11].dstSet = set;
    w[11].dstBinding = 11;
    w[11].descriptorCount = 1;
    w[11].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w[11].pBufferInfo = &probeBuf;
    for (uint32_t k = 0; k < 2; ++k) {
        w[12 + k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[12 + k].dstSet = set;
        w[12 + k].dstBinding = 12 + k;
        w[12 + k].descriptorCount = 1;
        w[12 + k].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[12 + k].pImageInfo = &probeImg[k];
    }
    vkUpdateDescriptorSets(ctx.device, 14, w, 0, nullptr);
}

void DeferredCore::recordSsrPass(VkCommandBuffer cmd, VkDescriptorSet ssrSet,
                                 const Mat4& viewProj, uint32_t width, uint32_t height) const {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ssrPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ssrPipelineLayout_, 0, 1,
                            &ssrSet, 0, nullptr);
    SsrPush push;
    std::memcpy(push.viewProj, viewProj.m, sizeof(push.viewProj));
    vkCmdPushConstants(cmd, ssrPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                       &push);
    vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);
}

// --- SSR temporal accumulation (Phase 2d; same model as the GTAO history) -----

bool DeferredCore::createSsrHistory(const VulkanContext& ctx, uint32_t w, uint32_t h,
                                    SsrHistory& out) const {
    out.width = w;
    out.height = h;
    for (uint32_t i = 0; i < 2; ++i) {
        if (createImage(ctx, w, h, kSsrTraceFormat,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, out.image[i],
                        out.memory[i], 1) != VK_SUCCESS)
            return false;
        out.view[i] = createImageView(ctx, out.image[i], kSsrTraceFormat,
                                      VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);
        if (!out.view[i]) return false;
    }
    // GENERAL for life (same model as the depth pyramids): each buffer
    // ping-pongs between temporal storage writes and temporal reads.
    submitOneShot(ctx, [&](VkCommandBuffer cmd) {
        for (uint32_t i = 0; i < 2; ++i)
            imageBarrier(cmd, out.image[i], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);
    });
    return true;
}

bool DeferredCore::writeSsrHistorySets(const VulkanContext& ctx, VkDescriptorPool pool,
                                       VkImageView ssrTrace, VkImageView depth,
                                       VkImageView sceneColor, SsrHistory& out) const {
    for (uint32_t i = 0; i < 2; ++i) {
        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = pool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &ssrTemporalSetLayout_;
        if (vkAllocateDescriptorSets(ctx.device, &alloc, &out.temporalSet[i]) != VK_SUCCESS)
            return false;

        VkDescriptorImageInfo img[5] = {};
        img[0].sampler = gbufferSampler_;
        img[0].imageView = ssrTrace;
        img[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        img[1].sampler = gbufferSampler_;
        img[1].imageView = out.view[1 - i]; // history read (the OTHER buffer)
        img[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        img[2].imageView = out.view[i]; // history write (storage)
        img[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        img[3].sampler = gbufferSampler_;
        img[3].imageView = depth;
        img[3].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        img[4].imageView = sceneColor; // lit HDR target (storage RMW composite)
        img[4].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet w[5] = {};
        for (uint32_t k = 0; k < 5; ++k) {
            w[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[k].dstSet = out.temporalSet[i];
            w[k].dstBinding = k;
            w[k].descriptorCount = 1;
            w[k].descriptorType = (k == 2 || k == 4) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                     : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[k].pImageInfo = &img[k];
        }
        vkUpdateDescriptorSets(ctx.device, 5, w, 0, nullptr);
    }
    return true;
}

void DeferredCore::destroySsrHistory(const VulkanContext& ctx, SsrHistory& history) const {
    for (uint32_t i = 0; i < 2; ++i) {
        if (history.view[i]) {
            vkDestroyImageView(ctx.device, history.view[i], nullptr);
            history.view[i] = VK_NULL_HANDLE;
        }
        if (history.image[i]) {
            vmaDestroyImage(ctx.allocator, history.image[i], history.memory[i]);
            history.image[i] = VK_NULL_HANDLE;
            history.memory[i] = VK_NULL_HANDLE;
        }
    }
    history.temporalSet[0] = history.temporalSet[1] = VK_NULL_HANDLE; // pool-owned
    history.width = history.height = 0;
}

void DeferredCore::recordSsrTemporalPass(VkCommandBuffer cmd, const SsrHistory& history,
                                         uint32_t writeIndex, const Mat4& invViewProj,
                                         const Mat4& prevViewProj, uint32_t width,
                                         uint32_t height, bool reset) const {
    if (!history.image[0]) return;
    const uint32_t i = writeIndex & 1u;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ssrTemporalPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ssrTemporalPipelineLayout_, 0, 1,
                            &history.temporalSet[i], 0, nullptr);
    // WAR: last frame's temporal pass sampled this buffer before it becomes
    // the storage target again.  Same-layout barrier; GENERAL for life.
    imageBarrier(cmd, history.image[i], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                 sync::kCompute, sync::kSampled, sync::kCompute, sync::kStorageWrite,
                 VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);
    SsaoTemporalPush push; // identical layout: invViewProj + prevViewProj + params
    std::memcpy(push.invViewProj, invViewProj.m, sizeof(push.invViewProj));
    std::memcpy(push.prevViewProj, prevViewProj.m, sizeof(push.prevViewProj));
    push.params[0] = kSsrTemporalBlend;
    push.params[1] = reset ? 1.f : 0.f;
    push.params[2] = push.params[3] = 0.f;
    vkCmdPushConstants(cmd, ssrTemporalPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(push), &push);
    vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);
    // Next frame's temporal pass samples the just-written buffer.
    imageBarrier(cmd, history.image[i], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                 sync::kCompute, sync::kStorageWrite, sync::kCompute, sync::kSampled,
                 VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);
}

// --- Froxel volumetric fog (Phase 5a) -----------------------------------------

bool DeferredCore::createVolFogVolume(const VulkanContext& ctx, uint32_t w, uint32_t h,
                                      VolFogVolume& out) const {
    out.dimX = (w + kFroxelTileSize - 1) / kFroxelTileSize;
    out.dimY = (h + kFroxelTileSize - 1) / kFroxelTileSize;
    out.dimZ = kFroxelSlices;
    const VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    struct Slot {
        VkImage* image;
        VmaAllocation* memory;
        VkImageView* view;
    };
    const Slot slots[5] = {
        {&out.injectImage, &out.injectMemory, &out.injectView},
        {&out.rawImage, &out.rawMemory, &out.rawView},
        {&out.histImage[0], &out.histMemory[0], &out.histView[0]},
        {&out.histImage[1], &out.histMemory[1], &out.histView[1]},
        {&out.intImage, &out.intMemory, &out.intView},
    };
    for (const Slot& s : slots) {
        if (createImage3D(ctx, out.dimX, out.dimY, out.dimZ, kFroxelFormat, usage, *s.image,
                          *s.memory) != VK_SUCCESS)
            return false;
        *s.view = createImageView(ctx, *s.image, kFroxelFormat, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1,
                                  VK_IMAGE_VIEW_TYPE_3D);
        if (!*s.view) return false;
    }
    // GENERAL for life (same model as the AO/SSR histories): every volume
    // ping-pongs between compute storage writes and compute sampled reads.
    submitOneShot(ctx, [&](VkCommandBuffer cmd) {
        for (const Slot& s : slots)
            imageBarrier(cmd, *s.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);
    });
    return true;
}

bool DeferredCore::writeVolFogSets(const VulkanContext& ctx, VkDescriptorPool pool,
                                   VolFogVolume& fog, const ClusterGrid& cluster,
                                   const VkBuffer* lightingUbos, VkImageView shadowMap,
                                   VkImageView shadowAtlas, VkImageView depth,
                                   VkImageView sceneColor) const {
    auto allocSet = [&](VkDescriptorSetLayout layout, VkDescriptorSet& out) {
        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = pool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &layout;
        return vkAllocateDescriptorSets(ctx.device, &alloc, &out) == VK_SUCCESS;
    };
    auto imageWrite = [](VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
                         VkSampler sampler, VkImageView view, VkImageLayout layout,
                         VkWriteDescriptorSet& w, VkDescriptorImageInfo& img) {
        img.sampler = sampler;
        img.imageView = view;
        img.imageLayout = layout;
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = set;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = type;
        w.pImageInfo = &img;
    };

    // Inject: storage-only set.
    if (!allocSet(volfogInjectSetLayout_, fog.injectSet)) return false;
    {
        VkDescriptorImageInfo img = {};
        VkWriteDescriptorSet w = {};
        imageWrite(fog.injectSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_NULL_HANDLE,
                   fog.injectView, VK_IMAGE_LAYOUT_GENERAL, w, img);
        vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);
    }

    // Light accumulation, per slot (per-slot LightingUBO + ClusterGrid SSBOs).
    for (uint32_t slot = 0; slot < kClusterSlots; ++slot) {
        if (!allocSet(volfogLightSetLayout_, fog.lightSet[slot])) return false;
        VkDescriptorBufferInfo ubo = {};
        ubo.buffer = lightingUbos[slot];
        ubo.range = sizeof(LightingUBO);
        VkDescriptorBufferInfo ssbo[2] = {};
        ssbo[0].buffer = cluster.lightsBuffer[slot];
        ssbo[0].range = VK_WHOLE_SIZE;
        ssbo[1].buffer = cluster.gridBuffer[slot];
        ssbo[1].range = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w[7] = {};
        for (uint32_t k = 0; k < 3; ++k) {
            w[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[k].dstSet = fog.lightSet[slot];
            w[k].dstBinding = k;
            w[k].descriptorCount = 1;
        }
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w[0].pBufferInfo = &ubo;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[1].pBufferInfo = &ssbo[0];
        w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[2].pBufferInfo = &ssbo[1];
        // Shadow maps follow the writeLightingSet convention: unwritten
        // binding when the host has no map (shadowParams.z == 0 then).
        uint32_t writeCount = 3;
        VkDescriptorImageInfo img[4] = {};
        if (shadowMap) {
            imageWrite(fog.lightSet[slot], 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                       shadowSampler_, shadowMap, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       w[writeCount], img[0]);
            ++writeCount;
        }
        if (shadowAtlas) {
            imageWrite(fog.lightSet[slot], 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                       shadowSampler_, shadowAtlas, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       w[writeCount], img[1]);
            ++writeCount;
        }
        imageWrite(fog.lightSet[slot], 5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                   gbufferSampler_, fog.injectView, VK_IMAGE_LAYOUT_GENERAL, w[writeCount],
                   img[2]);
        ++writeCount;
        imageWrite(fog.lightSet[slot], 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_NULL_HANDLE,
                   fog.rawView, VK_IMAGE_LAYOUT_GENERAL, w[writeCount], img[3]);
        ++writeCount;
        vkUpdateDescriptorSets(ctx.device, writeCount, w, 0, nullptr);
    }

    // Temporal ping-pong: set i reads raw + hist[1-i], writes hist[i].
    for (uint32_t i = 0; i < 2; ++i) {
        if (!allocSet(volfogTemporalSetLayout_, fog.temporalSet[i])) return false;
        VkDescriptorImageInfo img[3] = {};
        VkWriteDescriptorSet w[3] = {};
        imageWrite(fog.temporalSet[i], 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                   gbufferSampler_, fog.rawView, VK_IMAGE_LAYOUT_GENERAL, w[0], img[0]);
        imageWrite(fog.temporalSet[i], 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                   gbufferSampler_, fog.histView[1 - i], VK_IMAGE_LAYOUT_GENERAL, w[1], img[1]);
        imageWrite(fog.temporalSet[i], 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_NULL_HANDLE,
                   fog.histView[i], VK_IMAGE_LAYOUT_GENERAL, w[2], img[2]);
        vkUpdateDescriptorSets(ctx.device, 3, w, 0, nullptr);

        // March: reads the filtered hist[i], writes the integrated volume.
        if (!allocSet(volfogMarchSetLayout_, fog.marchSet[i])) return false;
        VkDescriptorImageInfo mimg[2] = {};
        VkWriteDescriptorSet mw[2] = {};
        imageWrite(fog.marchSet[i], 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                   gbufferSampler_, fog.histView[i], VK_IMAGE_LAYOUT_GENERAL, mw[0], mimg[0]);
        imageWrite(fog.marchSet[i], 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_NULL_HANDLE,
                   fog.intView, VK_IMAGE_LAYOUT_GENERAL, mw[1], mimg[1]);
        vkUpdateDescriptorSets(ctx.device, 2, mw, 0, nullptr);
    }

    // Composite: lit HDR target RMW + GBuffer depth + integrated volume.
    if (!allocSet(volfogCompositeSetLayout_, fog.compositeSet)) return false;
    {
        VkDescriptorImageInfo img[3] = {};
        VkWriteDescriptorSet w[3] = {};
        imageWrite(fog.compositeSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_NULL_HANDLE,
                   sceneColor, VK_IMAGE_LAYOUT_GENERAL, w[0], img[0]);
        imageWrite(fog.compositeSet, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                   gbufferSampler_, depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, w[1], img[1]);
        imageWrite(fog.compositeSet, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                   gbufferSampler_, fog.intView, VK_IMAGE_LAYOUT_GENERAL, w[2], img[2]);
        vkUpdateDescriptorSets(ctx.device, 3, w, 0, nullptr);
    }
    return true;
}

void DeferredCore::destroyVolFogVolume(const VulkanContext& ctx, VolFogVolume& fog) const {
    struct Slot {
        VkImage* image;
        VmaAllocation* memory;
        VkImageView* view;
    };
    const Slot slots[5] = {
        {&fog.injectImage, &fog.injectMemory, &fog.injectView},
        {&fog.rawImage, &fog.rawMemory, &fog.rawView},
        {&fog.histImage[0], &fog.histMemory[0], &fog.histView[0]},
        {&fog.histImage[1], &fog.histMemory[1], &fog.histView[1]},
        {&fog.intImage, &fog.intMemory, &fog.intView},
    };
    for (const Slot& s : slots) {
        if (*s.view) {
            vkDestroyImageView(ctx.device, *s.view, nullptr);
            *s.view = VK_NULL_HANDLE;
        }
        if (*s.image) {
            vmaDestroyImage(ctx.allocator, *s.image, *s.memory);
            *s.image = VK_NULL_HANDLE;
            *s.memory = VK_NULL_HANDLE;
        }
    }
    // Sets are pool-owned.
    fog.injectSet = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < kClusterSlots; ++i) fog.lightSet[i] = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < 2; ++i) {
        fog.temporalSet[i] = VK_NULL_HANDLE;
        fog.marchSet[i] = VK_NULL_HANDLE;
    }
    fog.compositeSet = VK_NULL_HANDLE;
    fog.dimX = fog.dimY = 0;
}

void DeferredCore::recordVolFogAccumulate(VkCommandBuffer cmd, VolFogVolume& fog,
                                          const ClusterGrid& cluster, uint32_t slot,
                                          const Mat4& view, const Mat4& proj,
                                          const Mat4& prevViewProj, const VolFogParams& params,
                                          uint32_t frameIndex, uint32_t writeIndex,
                                          bool reset) const {
    if (!fog.injectImage) return;
    const uint32_t s = slot % kClusterSlots;
    const Mat4 invView = Mat4::inverse(view);
    const float nearZ = proj.m[14] / proj.m[10]; // see recordLightingPass
    const float fogFar = params.maxDistance > nearZ ? params.maxDistance : nearZ * 2.f;
    // Shared grid dims / proj params for every pass (fog far range, not the
    // camera far plane: the volume covers [near, maxDistance]).
    auto fillProj = [&](float* projParams) {
        projParams[0] = proj.m[0];
        projParams[1] = proj.m[5];
        projParams[2] = nearZ;
        projParams[3] = fogFar;
    };
    auto volumeBarrier = [&](VkImage image, VkAccessFlags2 srcAccess,
                             VkAccessFlags2 dstAccess) {
        imageBarrier(cmd, image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                     sync::kCompute, srcAccess, sync::kCompute, dstAccess,
                     VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);
    };
    const uint32_t gx = (fog.dimX + 3) / 4;
    const uint32_t gy = (fog.dimY + 3) / 4;
    const uint32_t gz = (fog.dimZ + 3) / 4;

    // --- 1) inject (height fog + static noise -> density/albedo) -------------
    // WAR: last frame's light pass sampled the inject volume.
    volumeBarrier(fog.injectImage, sync::kSampled, sync::kStorageWrite);
    {
        VolFogInjectPush push = {};
        std::memcpy(push.invView, invView.m, sizeof(push.invView));
        fillProj(push.projParams);
        push.fogA[0] = params.density;
        push.fogA[1] = params.heightFalloff;
        push.fogA[2] = params.baseHeight;
        push.fogA[3] = params.noiseScale;
        push.fogAlbedo[0] = params.albedo.x;
        push.fogAlbedo[1] = params.albedo.y;
        push.fogAlbedo[2] = params.albedo.z;
        push.fogAlbedo[3] = params.noiseStrength;
        push.fogB[0] = static_cast<float>(frameIndex);
        push.grid[0] = fog.dimX;
        push.grid[1] = fog.dimY;
        push.grid[2] = fog.dimZ;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, volfogInjectPipeline_);
        vkCmdPushConstants(cmd, volfogInjectPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push), &push);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, volfogInjectPipelineLayout_,
                                0, 1, &fog.injectSet, 0, nullptr);
        vkCmdDispatch(cmd, gx, gy, gz);
    }

    // --- 2) light accumulation (sun CSM + cluster lights + ambient) ----------
    volumeBarrier(fog.injectImage, sync::kStorageWrite, sync::kSampled);
    // WAR: last frame's temporal pass sampled the raw-lit volume.
    volumeBarrier(fog.rawImage, sync::kSampled, sync::kStorageWrite);
    {
        // The cluster assignment for this frame/slot ran inside
        // recordLightingPass (compute write -> fragment read barrier); extend
        // the read scope to this compute pass.
        VkBufferMemoryBarrier2 clusterBar = {};
        clusterBar.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        clusterBar.srcStageMask =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        clusterBar.srcAccessMask =
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        clusterBar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        clusterBar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        clusterBar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        clusterBar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        clusterBar.buffer = cluster.gridBuffer[s];
        clusterBar.offset = 0;
        clusterBar.size = VK_WHOLE_SIZE;
        VkDependencyInfo dep = {};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.bufferMemoryBarrierCount = 1;
        dep.pBufferMemoryBarriers = &clusterBar;
        vkCmdPipelineBarrier2(cmd, &dep);

        VolFogLightPush push = {};
        std::memcpy(push.invView, invView.m, sizeof(push.invView));
        fillProj(push.projParams);
        push.fogA[0] = params.anisotropy;
        push.fogA[1] = params.ambient;
        push.grid[0] = fog.dimX;
        push.grid[1] = fog.dimY;
        push.grid[2] = fog.dimZ;
        push.misc[0] = cluster.gridX * kClusterTileSize; // screen px covered by the grid
        push.misc[1] = cluster.gridY * kClusterTileSize;
        push.misc[2] = kClusterTileSize;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, volfogLightPipeline_);
        vkCmdPushConstants(cmd, volfogLightPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push), &push);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, volfogLightPipelineLayout_,
                                0, 1, &fog.lightSet[s], 0, nullptr);
        vkCmdDispatch(cmd, gx, gy, gz);
    }

    // --- 3) temporal reprojection + EMA (ping-pong history) ------------------
    const uint32_t i = writeIndex & 1u;
    // Raw-lit writes -> temporal reads; WAR: the march pass of the previous
    // accumulate sampled hist[i] (two frames of ping-pong ago it was also the
    // temporal write target).
    volumeBarrier(fog.rawImage, sync::kStorageWrite, sync::kSampled);
    volumeBarrier(fog.histImage[i], sync::kSampled, sync::kStorageWrite);
    {
        VolFogTemporalPush push = {};
        std::memcpy(push.invView, invView.m, sizeof(push.invView));
        std::memcpy(push.prevViewProj, prevViewProj.m, sizeof(push.prevViewProj));
        fillProj(push.projParams);
        push.params[0] = kVolFogTemporalBlend;
        push.params[1] = reset ? 1.f : 0.f;
        push.grid[0] = fog.dimX;
        push.grid[1] = fog.dimY;
        push.grid[2] = fog.dimZ;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, volfogTemporalPipeline_);
        vkCmdPushConstants(cmd, volfogTemporalPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push), &push);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, volfogTemporalPipelineLayout_,
                                0, 1, &fog.temporalSet[i], 0, nullptr);
        vkCmdDispatch(cmd, gx, gy, gz);
    }

    // --- 4) front-to-back ray integration (per column) -------------------------
    // Filtered writes -> march reads; WAR: last frame's composite sampled the
    // integrated volume.
    volumeBarrier(fog.histImage[i], sync::kStorageWrite, sync::kSampled);
    volumeBarrier(fog.intImage, sync::kSampled, sync::kStorageWrite);
    {
        VolFogMarchPush push = {};
        std::memcpy(push.invView, invView.m, sizeof(push.invView));
        fillProj(push.projParams);
        push.grid[0] = fog.dimX;
        push.grid[1] = fog.dimY;
        push.grid[2] = fog.dimZ;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, volfogMarchPipeline_);
        vkCmdPushConstants(cmd, volfogMarchPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push), &push);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, volfogMarchPipelineLayout_,
                                0, 1, &fog.marchSet[i], 0, nullptr);
        vkCmdDispatch(cmd, (fog.dimX + 7) / 8, (fog.dimY + 7) / 8, 1);
    }
    // The march -> composite barrier lives at the top of recordVolFogComposite
    // (the host may record other passes in between).
}

void DeferredCore::recordVolFogComposite(VkCommandBuffer cmd, const VolFogVolume& fog,
                                         const Mat4& proj, float fogFar, uint32_t width,
                                         uint32_t height) const {
    if (!fog.intImage) return;
    // The march pass just wrote the integrated volume; make the writes
    // visible to this pass's trilinear samples.
    imageBarrier(cmd, fog.intImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                 sync::kCompute, sync::kStorageWrite, sync::kCompute, sync::kSampled,
                 VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);
    VolFogCompositePush push = {};
    push.depthParams[0] = proj.m[10];
    push.depthParams[1] = proj.m[14];
    push.depthParams[2] = proj.m[14] / proj.m[10]; // near (see recordLightingPass)
    push.depthParams[3] = fogFar;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, volfogCompositePipeline_);
    vkCmdPushConstants(cmd, volfogCompositePipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(push), &push);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, volfogCompositePipelineLayout_,
                            0, 1, &fog.compositeSet, 0, nullptr);
    vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);
}

// --- Bloom pyramid (Phase 6a) --------------------------------------------------
// Same resource model as the color pyramid above: one GENERAL-for-life image
// with per-mip views, pool-owned descriptor sets.  Every set shares the
// ssaoBlur binding shape (0 = sampler src, 1 = storage dst).

bool DeferredCore::createBloomPyramid(const VulkanContext& ctx, uint32_t fullW, uint32_t fullH,
                                      BloomPyramid& out) const {
    out.width = std::max(1u, fullW / 2);  // mip 0 = half-res extract
    out.height = std::max(1u, fullH / 2);
    if (createImage(ctx, out.width, out.height, deferred::kHdrColorFormat,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, out.image,
                    out.memory, kBloomMipCount) != VK_SUCCESS)
        return false;
    out.mipViews.resize(kBloomMipCount, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < kBloomMipCount; ++i) {
        out.mipViews[i] = createImageView(ctx, out.image, deferred::kHdrColorFormat,
                                          VK_IMAGE_ASPECT_COLOR_BIT, i, 1);
        if (!out.mipViews[i]) return false;
    }
    // GENERAL for life: every mip is both a compute storage target and a
    // sampled source, so a fixed layout beats per-frame transitions.
    submitOneShot(ctx, [&](VkCommandBuffer cmd) {
        imageBarrier(cmd, out.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_IMAGE_ASPECT_COLOR_BIT, 0, kBloomMipCount);
    });
    return true;
}

bool DeferredCore::writeBloomPyramidSets(const VulkanContext& ctx, VkDescriptorPool pool,
                                         VkImageView srcColor, BloomPyramid& out) const {
    // Helper: one ssaoBlur-layout set, binding 0 = src sampler, 1 = dst storage.
    auto writeSet = [&](VkDescriptorSet set, VkImageView src, VkImageLayout srcLayout,
                        VkImageView dst) {
        VkDescriptorImageInfo s = {};
        s.sampler = gbufferSampler_; // linear clamp (extract/downsample/upsample taps)
        s.imageView = src;
        s.imageLayout = srcLayout;
        VkDescriptorImageInfo d = {};
        d.imageView = dst;
        d.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet w[2] = {};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = set;
        w[0].dstBinding = 0;
        w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[0].pImageInfo = &s;
        w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[1].dstSet = set;
        w[1].dstBinding = 1;
        w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[1].pImageInfo = &d;
        vkUpdateDescriptorSets(ctx.device, 2, w, 0, nullptr);
    };
    auto allocSet = [&](VkDescriptorSet& set) {
        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = pool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &ssaoBlurSetLayout_;
        return vkAllocateDescriptorSets(ctx.device, &alloc, &set) == VK_SUCCESS;
    };

    // Extract: full-res HDR (SHADER_READ_ONLY) -> mip 0.
    if (!allocSet(out.extractSet)) return false;
    writeSet(out.extractSet, srcColor, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, out.mipViews[0]);
    // Downsample [i]: mip i -> mip i+1.
    out.downSets.resize(kBloomMipCount - 1, VK_NULL_HANDLE);
    for (uint32_t i = 0; i + 1 < kBloomMipCount; ++i) {
        if (!allocSet(out.downSets[i])) return false;
        writeSet(out.downSets[i], out.mipViews[i], VK_IMAGE_LAYOUT_GENERAL, out.mipViews[i + 1]);
    }
    // Upsample [i]: accumulated mip i+1 -> tent-add into mip i (RMW).
    out.upSets.resize(kBloomMipCount - 1, VK_NULL_HANDLE);
    for (uint32_t i = 0; i + 1 < kBloomMipCount; ++i) {
        if (!allocSet(out.upSets[i])) return false;
        writeSet(out.upSets[i], out.mipViews[i + 1], VK_IMAGE_LAYOUT_GENERAL, out.mipViews[i]);
    }
    // Composite: accumulated mip 0 -> full-res HDR RMW.  srcColor is bound as
    // storage here; the host transitions it SHADER_READ_ONLY -> GENERAL ->
    // SHADER_READ_ONLY around the pass.
    if (!allocSet(out.compositeSet)) return false;
    writeSet(out.compositeSet, out.mipViews[0], VK_IMAGE_LAYOUT_GENERAL, srcColor);
    return true;
}

void DeferredCore::destroyBloomPyramid(const VulkanContext& ctx, BloomPyramid& pyramid) const {
    for (VkImageView v : pyramid.mipViews)
        if (v) vkDestroyImageView(ctx.device, v, nullptr);
    pyramid.mipViews.clear();
    pyramid.downSets.clear(); // pool-owned; freed with the host's descriptor pool
    pyramid.upSets.clear();
    pyramid.extractSet = pyramid.compositeSet = VK_NULL_HANDLE;
    if (pyramid.image) {
        vmaDestroyImage(ctx.allocator, pyramid.image, pyramid.memory);
        pyramid.image = VK_NULL_HANDLE;
        pyramid.memory = VK_NULL_HANDLE;
    }
    pyramid.width = pyramid.height = 0;
}

void DeferredCore::recordBloomPyramidPass(VkCommandBuffer cmd, const BloomPyramid& pyramid,
                                          VkImage color, VkImageLayout& colorLayout,
                                          uint32_t fullW, uint32_t fullH, float strength) const {
    if (strength <= 0.f || !pyramid.image || fullW == 0 || fullH == 0) return;

    auto dispatch = [&](VkPipeline pipe, VkDescriptorSet set, const BloomPush& push, uint32_t gx,
                        uint32_t gy) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bloomPipelineLayout_, 0, 1,
                                &set, 0, nullptr);
        vkCmdPushConstants(cmd, bloomPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                           &push);
        vkCmdDispatch(cmd, gx, gy, 1);
    };
    // All pyramid passes are compute on one GENERAL image: a whole-chain
    // same-layout barrier between steps keeps the bookkeeping trivial (per-mip
    // subresource tracking buys nothing at 5 levels / ~10 barriers).
    auto chainBarrier = [&](VkAccessFlags2 srcAccess, VkAccessFlags2 dstAccess) {
        imageBarrier(cmd, pyramid.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                     sync::kCompute, srcAccess, sync::kCompute, dstAccess,
                     VK_IMAGE_ASPECT_COLOR_BIT, 0, kBloomMipCount);
    };
    auto mipGroups = [&](uint32_t level, uint32_t& gx, uint32_t& gy) {
        const uint32_t w = std::max(1u, pyramid.width >> level);
        const uint32_t h = std::max(1u, pyramid.height >> level);
        gx = (w + 7) / 8;
        gy = (h + 7) / 8;
    };

    // WAR: last frame's readers — this pass's downsample/upsample/composite
    // and the present pass's lens-dirt sample of mip 0 (fragment).
    imageBarrier(cmd, pyramid.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                 sync::kSampleStages, sync::kSampled, sync::kCompute, sync::kStorageWrite,
                 VK_IMAGE_ASPECT_COLOR_BIT, 0, kBloomMipCount);

    // 1) threshold extract (full-res HDR is SHADER_READ_ONLY) -> mip 0.
    BloomPush extractPush{};
    extractPush.params[0] = kBloomThreshold;
    extractPush.params[1] = kBloomKnee;
    uint32_t gx, gy;
    mipGroups(0, gx, gy);
    dispatch(bloomExtractPipeline_, pyramid.extractSet, extractPush, gx, gy);

    // 2) 13-tap downsample mip i -> mip i+1.
    const BloomPush noPush{};
    for (uint32_t i = 0; i + 1 < kBloomMipCount; ++i) {
        chainBarrier(sync::kStorageWrite, sync::kSampled);
        mipGroups(i + 1, gx, gy);
        dispatch(bloomDownsamplePipeline_, pyramid.downSets[i], noPush, gx, gy);
    }

    // 3) tent upsample: accumulate mip i+1 into mip i (in-place RMW; the WAR
    //    against this level's downsample reads is covered by the barrier).
    for (uint32_t i = kBloomMipCount - 1; i-- > 0;) {
        chainBarrier(sync::kStorageWrite, sync::kSampled | sync::kStorageReadWrite);
        mipGroups(i, gx, gy);
        dispatch(bloomUpsamplePipeline_, pyramid.upSets[i], noPush, gx, gy);
    }

    // 4) composite accumulated mip 0 onto the lit HDR target (in-place RMW;
    //    the previous users are the upscaler/compose/present shader reads).
    chainBarrier(sync::kStorageWrite, sync::kSampled);
    imageBarrier(cmd, color, colorLayout, VK_IMAGE_LAYOUT_GENERAL,
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    colorLayout = VK_IMAGE_LAYOUT_GENERAL;
    BloomPush comp{};
    comp.params[0] = strength;
    dispatch(bloomCompositePipeline_, pyramid.compositeSet, comp, (fullW + 7) / 8,
             (fullH + 7) / 8);

    imageBarrier(cmd, color, colorLayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    colorLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

// --- Motion blur + depth of field (Phase 6b) ------------------------------------
// Resource model matches BloomPyramid: host-owned GENERAL-for-life images,
// pool-owned descriptor sets, DeferredCore provides create/write/record/
// destroy.

bool DeferredCore::createPostFxTargets(const VulkanContext& ctx, uint32_t w, uint32_t h,
                                       PostFxTargets& out) const {
    out.width = w;
    out.height = h;
    out.tilesX = (w + kMotionBlurTileSize - 1) / kMotionBlurTileSize;
    out.tilesY = (h + kMotionBlurTileSize - 1) / kMotionBlurTileSize;
    const uint32_t halfW = std::max(1u, w / 2);
    const uint32_t halfH = std::max(1u, h / 2);
    struct Spec {
        uint32_t w, h;
        VkFormat format;
        VkImage* image;
        VmaAllocation* memory;
        VkImageView* view;
    };
    const Spec specs[] = {
        {out.tilesX, out.tilesY, deferred::kMotionFormat, &out.tileMaxImage, &out.tileMaxMemory, &out.tileMaxView},
        {out.tilesX, out.tilesY, deferred::kMotionFormat, &out.neighborMaxImage, &out.neighborMaxMemory, &out.neighborMaxView},
        {w, h, deferred::kHdrColorFormat, &out.mbOutImage, &out.mbOutMemory, &out.mbOutView},
        {halfW, halfH, deferred::kHdrColorFormat, &out.cocColorImage, &out.cocColorMemory, &out.cocColorView},
        {halfW, halfH, deferred::kHdrColorFormat, &out.bgImage, &out.bgMemory, &out.bgView},
        {halfW, halfH, deferred::kHdrColorFormat, &out.fgImage, &out.fgMemory, &out.fgView},
    };
    for (const Spec& s : specs) {
        if (createImage(ctx, s.w, s.h, s.format,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, *s.image,
                        *s.memory) != VK_SUCCESS)
            return false;
        *s.view = createImageView(ctx, *s.image, s.format, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);
        if (!*s.view) return false;
        // GENERAL for life: every image ping-pongs between compute storage
        // writes and sampled reads within the pass chain.
        VkImage img = *s.image;
        submitOneShot(ctx, [&](VkCommandBuffer cmd) {
            imageBarrier(cmd, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        });
    }
    return true;
}

bool DeferredCore::writePostFxSets(const VulkanContext& ctx, VkDescriptorPool pool,
                                   VkImageView srcColor, VkImageView motion, VkImageView depth,
                                   PostFxTargets& out) const {
    auto allocSet = [&](VkDescriptorSetLayout layout, VkDescriptorSet& set) {
        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = pool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &layout;
        return vkAllocateDescriptorSets(ctx.device, &alloc, &set) == VK_SUCCESS;
    };
    auto samplerInfo = [&](VkImageView view, VkImageLayout layout, VkSampler sampler) {
        VkDescriptorImageInfo info = {};
        info.sampler = sampler;
        info.imageView = view;
        info.imageLayout = layout;
        return info;
    };
    auto storageInfo = [](VkImageView view) {
        VkDescriptorImageInfo info = {};
        info.imageView = view;
        info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        return info;
    };
    auto write = [&](VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
                     const VkDescriptorImageInfo& info) {
        VkWriteDescriptorSet w = {};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = set;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = type;
        w.pImageInfo = &info;
        vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);
    };
    const VkDescriptorType kSam = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    const VkDescriptorType kStor = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    const VkImageLayout kSro = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    const VkImageLayout kGen = VK_IMAGE_LAYOUT_GENERAL;

    // Tile max: motion RT (SHADER_READ_ONLY) -> tileMax.
    if (!allocSet(mbTileSetLayout_, out.tileSet)) return false;
    write(out.tileSet, 0, kSam, samplerInfo(motion, kSro, hizSampler_)); // texelFetch
    write(out.tileSet, 1, kStor, storageInfo(out.tileMaxView));
    // Neighbourhood max: tileMax -> neighborMax (both GENERAL).
    if (!allocSet(mbTileSetLayout_, out.neighborSet)) return false;
    write(out.neighborSet, 0, kSam, samplerInfo(out.tileMaxView, kGen, hizSampler_));
    write(out.neighborSet, 1, kStor, storageInfo(out.neighborMaxView));
    // Gather: lit color + motion + neighbourhood max + depth -> mbOut.
    if (!allocSet(mbGatherSetLayout_, out.gatherSet)) return false;
    write(out.gatherSet, 0, kSam, samplerInfo(srcColor, kSro, gbufferSampler_));
    write(out.gatherSet, 1, kSam, samplerInfo(motion, kSro, gbufferSampler_));
    write(out.gatherSet, 2, kSam, samplerInfo(out.neighborMaxView, kGen, hizSampler_));
    write(out.gatherSet, 3, kSam, samplerInfo(depth, kSro, hizSampler_));
    write(out.gatherSet, 4, kStor, storageInfo(out.mbOutView));
    // DOF CoC: color (mbOut GENERAL or lit SHADER_READ_ONLY) + depth -> cocColor.
    if (!allocSet(dofCocSetLayout_, out.cocSetMb)) return false;
    write(out.cocSetMb, 0, kSam, samplerInfo(out.mbOutView, kGen, gbufferSampler_));
    write(out.cocSetMb, 1, kSam, samplerInfo(depth, kSro, hizSampler_));
    write(out.cocSetMb, 2, kStor, storageInfo(out.cocColorView));
    if (!allocSet(dofCocSetLayout_, out.cocSetLit)) return false;
    write(out.cocSetLit, 0, kSam, samplerInfo(srcColor, kSro, gbufferSampler_));
    write(out.cocSetLit, 1, kSam, samplerInfo(depth, kSro, hizSampler_));
    write(out.cocSetLit, 2, kStor, storageInfo(out.cocColorView));
    // DOF gather: cocColor -> bg/fg layers.
    if (!allocSet(dofGatherSetLayout_, out.dofGatherSet)) return false;
    write(out.dofGatherSet, 0, kSam, samplerInfo(out.cocColorView, kGen, gbufferSampler_));
    write(out.dofGatherSet, 1, kStor, storageInfo(out.bgView));
    write(out.dofGatherSet, 2, kStor, storageInfo(out.fgView));
    // DOF composite: sharp (storage imageLoad) + cocColor + layers -> lit color
    // (storage write; srcColor must be bound for both roles across two sets).
    if (!allocSet(dofCompositeSetLayout_, out.compositeSetMb)) return false;
    write(out.compositeSetMb, 0, kStor, storageInfo(out.mbOutView));
    write(out.compositeSetMb, 1, kSam, samplerInfo(out.cocColorView, kGen, gbufferSampler_));
    write(out.compositeSetMb, 2, kSam, samplerInfo(out.bgView, kGen, gbufferSampler_));
    write(out.compositeSetMb, 3, kSam, samplerInfo(out.fgView, kGen, gbufferSampler_));
    write(out.compositeSetMb, 4, kStor, storageInfo(srcColor));
    if (!allocSet(dofCompositeSetLayout_, out.compositeSetLit)) return false;
    write(out.compositeSetLit, 0, kStor, storageInfo(srcColor));
    write(out.compositeSetLit, 1, kSam, samplerInfo(out.cocColorView, kGen, gbufferSampler_));
    write(out.compositeSetLit, 2, kSam, samplerInfo(out.bgView, kGen, gbufferSampler_));
    write(out.compositeSetLit, 3, kSam, samplerInfo(out.fgView, kGen, gbufferSampler_));
    write(out.compositeSetLit, 4, kStor, storageInfo(srcColor));
    // MB copy-back (MB on, DOF off): mbOut -> lit color (both storage).
    if (!allocSet(postFxCopybackSetLayout_, out.copybackSet)) return false;
    write(out.copybackSet, 0, kStor, storageInfo(out.mbOutView));
    write(out.copybackSet, 1, kStor, storageInfo(srcColor));
    return true;
}

void DeferredCore::destroyPostFxTargets(const VulkanContext& ctx, PostFxTargets& fx) const {
    struct Res {
        VkImage* image;
        VmaAllocation* memory;
        VkImageView* view;
    };
    const Res res[] = {
        {&fx.tileMaxImage, &fx.tileMaxMemory, &fx.tileMaxView},
        {&fx.neighborMaxImage, &fx.neighborMaxMemory, &fx.neighborMaxView},
        {&fx.mbOutImage, &fx.mbOutMemory, &fx.mbOutView},
        {&fx.cocColorImage, &fx.cocColorMemory, &fx.cocColorView},
        {&fx.bgImage, &fx.bgMemory, &fx.bgView},
        {&fx.fgImage, &fx.fgMemory, &fx.fgView},
    };
    for (const Res& r : res) {
        if (*r.view) { vkDestroyImageView(ctx.device, *r.view, nullptr); *r.view = VK_NULL_HANDLE; }
        if (*r.image) {
            vmaDestroyImage(ctx.allocator, *r.image, *r.memory);
            *r.image = VK_NULL_HANDLE;
            *r.memory = VK_NULL_HANDLE;
        }
    }
    // Descriptor sets are pool-owned; freed with the host's descriptor pool.
    fx.tileSet = fx.neighborSet = fx.gatherSet = VK_NULL_HANDLE;
    fx.cocSetMb = fx.cocSetLit = fx.dofGatherSet = VK_NULL_HANDLE;
    fx.compositeSetMb = fx.compositeSetLit = VK_NULL_HANDLE;
    fx.copybackSet = VK_NULL_HANDLE;
    fx.width = fx.height = fx.tilesX = fx.tilesY = 0;
}

void DeferredCore::recordPostFxPass(VkCommandBuffer cmd, const PostFxTargets& fx, VkImage color,
                                    VkImageLayout& colorLayout, const PostFxParams& params,
                                    uint32_t frameIndex) const {
    if ((!params.motionBlur && !params.dof) || !fx.mbOutImage || fx.width == 0 || fx.height == 0)
        return;

    auto dispatch = [&](VkPipeline pipe, VkPipelineLayout layout, VkDescriptorSet set,
                        const void* push, uint32_t pushSize, uint32_t w, uint32_t h) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0,
                                nullptr);
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, push);
        vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
    };
    // GENERAL-to-GENERAL same-image barrier between chain steps (write ->
    // read; readers sample or imageLoad, hence both access bits).
    constexpr VkAccessFlags2 kFxRead =
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    auto chainBarrier = [&](VkImage image) {
        imageBarrier(cmd, image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                     sync::kCompute, sync::kStorageWrite, sync::kCompute,
                     kFxRead, VK_IMAGE_ASPECT_COLOR_BIT);
    };
    // WAR against last frame's (and this pass's earlier) readers before a
    // working target is rewritten.
    auto warBarrier = [&](VkImage image) {
        imageBarrier(cmd, image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                     sync::kCompute, kFxRead, sync::kCompute, sync::kStorageWrite,
                     VK_IMAGE_ASPECT_COLOR_BIT);
    };

    const bool mb = params.motionBlur;
    const bool dof = params.dof;

    if (mb) {
        // WAR: last frame's post-fx readers of the working targets.
        warBarrier(fx.tileMaxImage);
        // 1) tile max over the motion RT (SHADER_READ_ONLY, compute in scope).
        MotionBlurTilePush tilePush = {};
        tilePush.srcSize[0] = static_cast<int32_t>(fx.width);
        tilePush.srcSize[1] = static_cast<int32_t>(fx.height);
        tilePush.tileSize = static_cast<int32_t>(kMotionBlurTileSize);
        dispatch(mbTilePipeline_, mbTilePipelineLayout_, fx.tileSet, &tilePush, sizeof(tilePush),
                 fx.tilesX, fx.tilesY);

        // 2) 3x3 neighbourhood max (tileMax write -> sampled, neighborMax write).
        chainBarrier(fx.tileMaxImage);
        warBarrier(fx.neighborMaxImage);
        dispatch(mbNeighborPipeline_, mbTilePipelineLayout_, fx.neighborSet, &tilePush,
                 sizeof(tilePush), fx.tilesX, fx.tilesY);

        // 3) full-res gather into the intermediate (lit color is
        //    SHADER_READ_ONLY; motion + depth likewise, compute in scope).
        chainBarrier(fx.neighborMaxImage);
        warBarrier(fx.mbOutImage);
        MotionBlurGatherPush gatherPush = {};
        gatherPush.params[0] = kMotionBlurShutter;
        gatherPush.params[1] = params.maxBlurPx;
        gatherPush.params[2] = static_cast<float>(frameIndex);
        gatherPush.params[3] = static_cast<float>(kMotionBlurTileSize);
        gatherPush.params2[0] = static_cast<float>(fx.width);
        gatherPush.params2[1] = static_cast<float>(fx.height);
        dispatch(mbGatherPipeline_, mbGatherPipelineLayout_, fx.gatherSet, &gatherPush,
                 sizeof(gatherPush), fx.width, fx.height);
        chainBarrier(fx.mbOutImage); // gather writes -> coc/composite reads
    }

    if (dof) {
        const uint32_t halfW = std::max(1u, fx.width / 2);
        const uint32_t halfH = std::max(1u, fx.height / 2);
        // 1) CoC setup: color (mbOut when MB ran, else the lit target) + depth
        //    -> half-res cocColor.
        warBarrier(fx.cocColorImage);
        DofCocPush cocPush = {};
        cocPush.depthParams[0] = params.depthM10;
        cocPush.depthParams[1] = params.depthM14;
        cocPush.depthParams[2] = params.farPlane;
        // CoC lives in half-res pixels on both sides (coc stores coc / max,
        // the gather re-multiplies by it).
        cocPush.depthParams[3] = params.maxCocPx * 0.5f;
        cocPush.params[0] = params.aperture;
        cocPush.params[1] = kDofSkyFocus;
        dispatch(dofCocPipeline_, dofCocPipelineLayout_, mb ? fx.cocSetMb : fx.cocSetLit,
                 &cocPush, sizeof(cocPush), halfW, halfH);

        // 2) half-res foreground/background bokeh gather.
        chainBarrier(fx.cocColorImage);
        warBarrier(fx.bgImage);
        warBarrier(fx.fgImage);
        DofGatherPush dofGatherPush = {};
        dofGatherPush.params[0] = params.maxCocPx * 0.5f; // cocColor alpha is in half-res px
        dofGatherPush.params[1] = static_cast<float>(frameIndex);
        dispatch(dofGatherPipeline_, dofGatherPipelineLayout_, fx.dofGatherSet, &dofGatherPush,
                 sizeof(dofGatherPush), halfW, halfH);

        // 3) full-res composite back into the lit HDR target.  When MB is off
        //    the sharp source IS the lit target: same-texel imageLoad/store
        //    per invocation (race-free, same pattern as bloom composite).
        chainBarrier(fx.bgImage);
        chainBarrier(fx.fgImage);
        imageBarrier(cmd, color, colorLayout, VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, sync::kCompute,
                     sync::kStorageReadWrite);
        colorLayout = VK_IMAGE_LAYOUT_GENERAL;
        DofCompositePush compPush = {};
        compPush.params[0] = 1.f / static_cast<float>(kDofGatherTaps);
        dispatch(dofCompositePipeline_, dofCompositePipelineLayout_,
                 mb ? fx.compositeSetMb : fx.compositeSetLit, &compPush, sizeof(compPush),
                 fx.width, fx.height);
        imageBarrier(cmd, color, colorLayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     sync::kCompute, sync::kStorageReadWrite,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     sync::kSampled);
        colorLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    } else if (mb) {
        // DOF off: nothing else writes the MB gather result back into the lit
        // target, so copy mbOut -> color here (the DOF composite does this
        // when DOF runs).
        imageBarrier(cmd, color, colorLayout, VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, sync::kCompute,
                     sync::kStorageWrite);
        colorLayout = VK_IMAGE_LAYOUT_GENERAL;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, postFxCopybackPipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                postFxCopybackPipelineLayout_, 0, 1, &fx.copybackSet, 0, nullptr);
        vkCmdDispatch(cmd, (fx.width + 7) / 8, (fx.height + 7) / 8, 1);
        imageBarrier(cmd, color, colorLayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     sync::kCompute, sync::kStorageWrite,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     sync::kSampled);
        colorLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
}


// --- Auto exposure -------------------------------------------------------------
// Resource model matches the pyramids: host-owned buffers + pool-owned set,
// DeferredCore creates/writes/records/destroys.  The state buffer is seeded
// with the initial EV so the frames before the first CPU readback still match
// the manual-exposure look.

bool DeferredCore::createAutoExposure(const VulkanContext& ctx, float initialEV,
                                      AutoExposure& out) const {
    if (createBuffer(ctx, kExposureHistogramBins * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out.histogram,
                     out.histogramMemory) != VK_SUCCESS)
        return false;
    if (createBuffer(ctx, sizeof(ExposureState),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out.state,
                     out.stateMemory) != VK_SUCCESS)
        return false;

    // Seed the state (exposure = exp2(-initialEV), ev = initialEV) via a
    // one-shot update; the histogram starts zeroed.
    submitOneShot(ctx, [&](VkCommandBuffer cmd) {
        const ExposureState seed = {std::exp2(-initialEV), 0.f, initialEV, initialEV};
        vkCmdUpdateBuffer(cmd, out.state, 0, sizeof(seed), &seed);
        // The update must be visible before the first solve pass reads it.
        VkMemoryBarrier2 barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask =
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        VkDependencyInfo dep = {};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    });
    return true;
}

bool DeferredCore::writeAutoExposureSet(const VulkanContext& ctx, VkDescriptorPool pool,
                                        VkImageView srcColor, AutoExposure& out) const {
    VkDescriptorSetAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorPool = pool;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &exposureSetLayout_;
    if (vkAllocateDescriptorSets(ctx.device, &alloc, &out.set) != VK_SUCCESS) return false;

    VkDescriptorImageInfo src = {};
    src.sampler = hizSampler_; // nearest + clamp; texelFetch ignores it anyway
    src.imageView = srcColor;
    src.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorBufferInfo hist = {};
    hist.buffer = out.histogram;
    hist.range = kExposureHistogramBins * 4;
    VkDescriptorBufferInfo state = {};
    state.buffer = out.state;
    state.range = sizeof(ExposureState);

    VkWriteDescriptorSet w[3] = {};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = out.set;
    w[0].dstBinding = 0;
    w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w[0].pImageInfo = &src;
    for (uint32_t i = 1; i < 3; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = out.set;
        w[i].dstBinding = i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = i == 1 ? &hist : &state;
    }
    vkUpdateDescriptorSets(ctx.device, 3, w, 0, nullptr);
    return true;
}

void DeferredCore::destroyAutoExposure(const VulkanContext& ctx, AutoExposure& ae) const {
    if (ae.histogram) {
        vmaDestroyBuffer(ctx.allocator, ae.histogram, ae.histogramMemory);
        ae.histogram = VK_NULL_HANDLE;
        ae.histogramMemory = VK_NULL_HANDLE;
    }
    if (ae.state) {
        vmaDestroyBuffer(ctx.allocator, ae.state, ae.stateMemory);
        ae.state = VK_NULL_HANDLE;
        ae.stateMemory = VK_NULL_HANDLE;
    }
    ae.set = VK_NULL_HANDLE; // pool-owned; freed with the host's descriptor pool
    ae.srcWidth = ae.srcHeight = 0;
}

void DeferredCore::recordAutoExposurePass(VkCommandBuffer cmd, const AutoExposure& ae,
                                          const ExposureSolvePush& solve,
                                          VkBuffer readbackDst) const {
    if (!ae.set) return;

    // WAR: last frame's solve read the histogram; this frame's histogram pass
    // rewrites it.  (First frame after creation: covered by the seed barrier.)
    auto bufferBarrier = [&](VkAccessFlags2 srcAccess, VkAccessFlags2 dstAccess,
                             VkPipelineStageFlags2 dstStage) {
        VkMemoryBarrier2 barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = srcAccess;
        barrier.dstStageMask = dstStage;
        barrier.dstAccessMask = dstAccess;
        VkDependencyInfo dep = {};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    };
    bufferBarrier(VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, histogramPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, histogramPipelineLayout_, 0, 1,
                            &ae.set, 0, nullptr);
    HistogramPush hp;
    hp.sampleDims[0] = static_cast<int32_t>((ae.srcWidth + 1) / 2);
    hp.sampleDims[1] = static_cast<int32_t>((ae.srcHeight + 1) / 2);
    vkCmdPushConstants(cmd, histogramPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(hp),
                       &hp);
    vkCmdDispatch(cmd, (hp.sampleDims[0] + 15) / 16, (hp.sampleDims[1] + 15) / 16, 1);

    // Histogram writes -> solve reads.
    bufferBarrier(VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, exposureSolvePipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, exposureSolvePipelineLayout_, 0,
                            1, &ae.set, 0, nullptr);
    // Push the EV clamp/offset/reset block + the fixed tuning block
    // (speeds, fixed dt, log2 key).  dt is a constant by design: smoothing
    // steps per FRAME, not per wall-clock second, for bench determinism.
    struct {
        ExposureSolvePush ev;
        float tuning[4];
    } push = {solve,
              {kExposureSpeedUp, kExposureSpeedDown, kExposureFixedDt,
               std::log2(kExposureKeyValue)}};
    static_assert(sizeof(push) == 32, "exposure solve push size mismatch");
    vkCmdPushConstants(cmd, exposureSolvePipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(push), &push);
    vkCmdDispatch(cmd, 1, 1, 1);

    if (readbackDst) {
        // Solve writes -> transfer read; the host harvests the staging copy
        // after this frame's fence (kFramesInFlight frames of latency).
        bufferBarrier(VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                      VK_PIPELINE_STAGE_2_COPY_BIT);
        VkBufferCopy region = {};
        region.size = sizeof(ExposureState);
        vkCmdCopyBuffer(cmd, ae.state, readbackDst, 1, &region);
    }
}

bool DeferredCore::createExposureChannel(const VulkanContext& ctx, VkDescriptorPool pool,
                                         VkImageView srcColor, uint32_t srcW, uint32_t srcH,
                                         float initialEV, ExposureChannel& out) const {
    if (!createAutoExposure(ctx, initialEV, out.gpu)) return false;
    out.gpu.srcWidth = srcW;
    out.gpu.srcHeight = srcH;
    if (!writeAutoExposureSet(ctx, pool, srcColor, out.gpu)) return false;
    for (uint32_t i = 0; i < kExposureSlots; ++i) {
        if (createBuffer(ctx, sizeof(ExposureState), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         out.staging[i], out.stagingMemory[i]) != VK_SUCCESS)
            return false;
        if (vmaMapMemory(ctx.allocator, out.stagingMemory[i], &out.stagingMapped[i]) != VK_SUCCESS)
            return false;
    }
    // The CPU-visible value starts at the seed exposure (pre-readback frames).
    out.value = std::exp2(-initialEV);
    return true;
}

void DeferredCore::destroyExposureChannel(const VulkanContext& ctx, ExposureChannel& channel) const {
    for (uint32_t i = 0; i < kExposureSlots; ++i) {
        if (channel.staging[i]) {
            vmaUnmapMemory(ctx.allocator, channel.stagingMemory[i]);
            vmaDestroyBuffer(ctx.allocator, channel.staging[i], channel.stagingMemory[i]);
            channel.staging[i] = VK_NULL_HANDLE;
            channel.stagingMemory[i] = VK_NULL_HANDLE;
            channel.stagingMapped[i] = nullptr;
        }
        channel.pending[i] = false;
    }
    destroyAutoExposure(ctx, channel.gpu);
}

void DeferredCore::harvestExposureChannel(ExposureChannel& channel, uint32_t slot) const {
    if (!channel.pending[slot]) return;
    const auto* s = static_cast<const ExposureState*>(channel.stagingMapped[slot]);
    channel.value = s->exposure;
    channel.pending[slot] = false;
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
        if (inst.lodCulled) continue; // LOD screen-size cull
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

        const LodDraw& draw = inst.lodDraws[inst.lodLevel];
        vkCmdDrawIndexed(cmd, draw.indexCount, 1, draw.firstIndex, draw.vertexOffset, 0);
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

// Shared draw loop of the shadow depth passes: binds the (skinned) shadow
// pipelines and draws every caster intersecting the light frustum into the
// currently open rendering block.  Used by recordShadowPass (one call per
// cascade) and recordSpotShadowPass (one call per atlas tile).
void DeferredCore::recordShadowDraws(VkCommandBuffer cmd, const Scene& scene, const Mat4& lightVp,
                                     VkDescriptorSet sceneSet, VkDescriptorSet textureSet,
                                     uint32_t materialStride) const {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline_);
    vkCmdSetDepthBias(cmd, kShadowDepthBiasConstant, 0.f, kShadowDepthBiasSlope);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipelineLayout_, 1, 1,
                            &textureSet, 0, nullptr);

    const VkDeviceSize zeroOffset = 0;
    if (scene.mergedVertexBuffer) {
        vkCmdBindVertexBuffers(cmd, 0, 1, &scene.mergedVertexBuffer, &zeroOffset);
        vkCmdBindIndexBuffer(cmd, scene.mergedIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
    }

    const Frustum frustum = extractFrustum(lightVp);

    uint32_t lastMaterial = UINT32_MAX;
    bool skinnedBound = false;
    for (const auto& inst : scene.instances) {
        if (scene.materials[inst.materialIndex].blend) continue; // glass does not occlude
        if (!aabbIntersectsFrustum(frustum, inst.aabbMin, inst.aabbMax)) continue;
        // Same LOD choice as the camera passes: a caster too small to draw
        // casts a shadow too small to miss, and the ranges are free here.
        if (inst.lodCulled) continue;

        if (inst.skinIndex >= 0) {
            // Skinned caster: current-frame palette only (no motion here).
            if (!skinnedBound) {
                skinnedBound = true;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  shadowSkinnedPipeline_);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        scenePipelineLayout_, 1, 1, &textureSet, 0, nullptr);
                lastMaterial = UINT32_MAX;
            }
            const Skin& skin = scene.skins[static_cast<size_t>(inst.skinIndex)];
            SkinnedShadowPush push;
            std::memcpy(push.lightVp, lightVp.m, sizeof(push.lightVp));
            push.paletteCur = skin.paletteCur;
            vkCmdPushConstants(cmd, scenePipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0,
                               sizeof(push), &push);

            if (inst.materialIndex != lastMaterial) {
                lastMaterial = inst.materialIndex;
                const uint32_t dynOffset = inst.materialIndex * materialStride;
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        scenePipelineLayout_, 0, 1, &sceneSet, 1, &dynOffset);
            }

            const Mesh& mesh = scene.skinnedMeshes[inst.meshIndex];
            vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertexBuffer, &zeroOffset);
            vkCmdBindIndexBuffer(cmd, mesh.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, mesh.indexCount, 1, 0, 0, 0);
            continue;
        }

        if (skinnedBound) {
            skinnedBound = false;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipelineLayout_,
                                    1, 1, &textureSet, 0, nullptr);
            vkCmdBindVertexBuffers(cmd, 0, 1, &scene.mergedVertexBuffer, &zeroOffset);
            vkCmdBindIndexBuffer(cmd, scene.mergedIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
            lastMaterial = UINT32_MAX;
        }

        ShadowPush push;
        std::memcpy(push.model, inst.model.m, sizeof(push.model));
        std::memcpy(push.lightVp, lightVp.m, sizeof(push.lightVp));
        vkCmdPushConstants(cmd, scenePipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0,
                           sizeof(push), &push);

        if (inst.materialIndex != lastMaterial) {
            lastMaterial = inst.materialIndex;
            const uint32_t dynOffset = inst.materialIndex * materialStride;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipelineLayout_,
                                    0, 1, &sceneSet, 1, &dynOffset);
        }

        const LodDraw& draw = inst.lodDraws[inst.lodLevel];
        vkCmdDrawIndexed(cmd, draw.indexCount, 1, draw.firstIndex, draw.vertexOffset, 0);
    }
}

void DeferredCore::recordShadowPass(VkCommandBuffer cmd, const ShadowTargets& targets,
                                    const Scene& scene, const Mat4 cascadeVp[kShadowCascadeCount],
                                    VkDescriptorSet sceneSet, VkDescriptorSet textureSet,
                                    uint32_t materialStride) const {
    VkViewport viewport = {0.f, 0.f, static_cast<float>(kShadowMapSize),
                           static_cast<float>(kShadowMapSize), 0.f, 1.f};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor = {{0, 0}, {kShadowMapSize, kShadowMapSize}};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

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
        recordShadowDraws(cmd, scene, cascadeVp[c], sceneSet, textureSet, materialStride);
        vkCmdEndRendering(cmd);
    }
}

bool DeferredCore::createShadowAtlas(const VulkanContext& ctx, ShadowAtlas& out) const {
    if (createImage(ctx, kShadowAtlasSize, kShadowAtlasSize, deferred::kDepthFormat,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    out.image, out.memory) != VK_SUCCESS)
        return false;
    // Whole-atlas 2D view: depth attachment (per-tile renderArea) and the
    // comparison-sampled view of lighting set binding 14.
    out.view = createImageView(ctx, out.image, deferred::kDepthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
    if (!out.view) return false;
    out.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    return true;
}

void DeferredCore::destroyShadowAtlas(const VulkanContext& ctx, ShadowAtlas& atlas) const {
    if (!ctx.device) return;
    if (atlas.view) { vkDestroyImageView(ctx.device, atlas.view, nullptr); atlas.view = VK_NULL_HANDLE; }
    if (atlas.image) { vmaDestroyImage(ctx.allocator, atlas.image, atlas.memory); atlas.image = VK_NULL_HANDLE; atlas.memory = VK_NULL_HANDLE; }
    atlas.layout = VK_IMAGE_LAYOUT_UNDEFINED;
}

uint32_t DeferredCore::selectSpotShadowLights(std::vector<Light>& lights, const Vec3& cameraPos) {
    for (Light& l : lights) l.shadowIndex = -1;
    // Candidates: shadow-casting spots only (point lights stay unshadowed this
    // phase — see the atlas comment in the header).
    std::vector<uint32_t> candidates;
    for (uint32_t i = 0; i < lights.size(); ++i) {
        if (lights[i].type == LightType::Spot && lights[i].castShadow)
            candidates.push_back(i);
    }
    // Importance = intensity / distance^2 to the camera; stable_sort keeps the
    // lower scene index on ties, so the same frame state always produces the
    // same tile assignment on every host (resolution-independent).
    const auto score = [&](uint32_t i) {
        const Vec3 d = lights[i].positionOrDirection - cameraPos;
        return lights[i].intensity / std::max(dot(d, d), 1e-3f);
    };
    std::stable_sort(candidates.begin(), candidates.end(),
                     [&](uint32_t a, uint32_t b) { return score(a) > score(b); });
    const uint32_t count =
        std::min(static_cast<uint32_t>(candidates.size()), kShadowAtlasTiles);
    for (uint32_t t = 0; t < count; ++t)
        lights[candidates[t]].shadowIndex = static_cast<int32_t>(t);
    return count;
}

Mat4 DeferredCore::computeSpotShadowVp(const Light& light) {
    const float fovY = std::min(2.f * light.outerConeAngle + kSpotShadowFovMargin, 2.9f);
    const float farZ = light.range > 0.f ? light.range : kSpotShadowInfiniteRange;
    const float nearZ = std::max(farZ * 1e-3f, 0.02f);
    Vec3 up{0.f, 1.f, 0.f};
    if (std::fabs(dot(light.spotDirection, up)) > 0.99f) up = {1.f, 0.f, 0.f};
    const Mat4 view = Mat4::lookAt(light.positionOrDirection,
                                   light.positionOrDirection + light.spotDirection, up);
    return Mat4::multiply(Mat4::perspective(fovY, 1.f, nearZ, farZ), view);
}

void DeferredCore::recordSpotShadowPass(VkCommandBuffer cmd, const ShadowAtlas& atlas,
                                        const Scene& scene, const Mat4* tileVp, uint32_t tileCount,
                                        VkDescriptorSet sceneSet, VkDescriptorSet textureSet,
                                        uint32_t materialStride) const {
    // One rendering block per tile: simple, and each block clears only its own
    // tile rect (renderArea == tile).  Unassigned tiles are never rendered and
    // never sampled (unshadowed lights keep shadowIndex == -1).
    for (uint32_t t = 0; t < tileCount; ++t) {
        const uint32_t x = (t % kShadowAtlasGrid) * kShadowAtlasTileSize;
        const uint32_t y = (t / kShadowAtlasGrid) * kShadowAtlasTileSize;
        VkRenderingAttachmentInfo depth =
            makeDepthAttachment(atlas.view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                VK_ATTACHMENT_LOAD_OP_CLEAR);
        VkRenderingInfo ri = {};
        ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea = {{static_cast<int32_t>(x), static_cast<int32_t>(y)},
                         {kShadowAtlasTileSize, kShadowAtlasTileSize}};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 0;
        ri.pDepthAttachment = &depth;
        vkCmdBeginRendering(cmd, &ri);

        VkViewport viewport = {static_cast<float>(x), static_cast<float>(y),
                               static_cast<float>(kShadowAtlasTileSize),
                               static_cast<float>(kShadowAtlasTileSize), 0.f, 1.f};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        VkRect2D scissor = {{static_cast<int32_t>(x), static_cast<int32_t>(y)},
                            {kShadowAtlasTileSize, kShadowAtlasTileSize}};
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        recordShadowDraws(cmd, scene, tileVp[t], sceneSet, textureSet, materialStride);
        vkCmdEndRendering(cmd);
    }
}

} // namespace sr
