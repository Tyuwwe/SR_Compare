#include "renderer/ibl/Ibl.h"

#include "renderer/core/PathUtil.h"
#include "renderer/core/VkUtil.h"
#include "renderer/core/Vma.h"
#include "renderer/ibl/SkyAtmosphere.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#include <stb_image.h> // implementation lives in scene/GltfLoader.cpp
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace sr {

namespace {

constexpr VkFormat kHdrFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

uint16_t floatToHalf(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int exp = static_cast<int>((x >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) return static_cast<uint16_t>(sign); // underflow -> +/-0
    // Clamp to the largest finite half (sun disc pixels exceed 65504; inf
    // would poison the whole mip chain during downsampling).
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7BFFu);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

bool loadShaderModule(const VulkanContext& ctx, const char* name, VkShaderModule& out) {
    const std::string path = sr::resolveShaderPath(SR_SHADER_DIR, name);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "ibl: failed to open shader %s\n", path.c_str());
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
    if (vkCreateShaderModule(ctx.device, &ci, nullptr, &out) != VK_SUCCESS) return false;
    return true;
}

// Upload RGBA16F texels to a 1-level 2D image, ending in SHADER_READ_ONLY.
bool uploadHdrImage(const VulkanContext& ctx, uint32_t w, uint32_t h, const uint16_t* data,
                    VkImage& image, VmaAllocation& memory, VkImageView& view) {
    if (createImage(ctx, w, h, kHdrFormat,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, image,
                    memory) != VK_SUCCESS)
        return false;
    const VkDeviceSize size = static_cast<VkDeviceSize>(w) * h * 8;
    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation stagingMemory = VK_NULL_HANDLE;
    if (createBuffer(ctx, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging, stagingMemory) != VK_SUCCESS)
        return false;
    void* mapped = nullptr;
    vmaMapMemory(ctx.allocator, stagingMemory, &mapped);
    std::memcpy(mapped, data, static_cast<size_t>(size));
    vmaUnmapMemory(ctx.allocator, stagingMemory);
    submitOneShot(ctx, [&](VkCommandBuffer cmd) {
        copyBufferToImage(cmd, staging, image, w, h, kHdrFormat);
    });
    vmaDestroyBuffer(ctx.allocator, staging, stagingMemory);
    view = createImageView(ctx, image, kHdrFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    return view != VK_NULL_HANDLE;
}

// Generic compute stage: `sampledCount` combined-image-sampler bindings
// (0..N-1) followed by one storage image (binding N), plus optional push
// constants.  Covers the equirect/irradiance/prefilter/LUT stages (0-1
// samplers) and the sky stage (2 samplers).
bool createStage(const VulkanContext& ctx, VkShaderModule module, uint32_t sampledCount,
                 uint32_t pushSize, IblMaps::ComputeStage& out) {
    VkDescriptorSetLayoutBinding bindings[3] = {};
    uint32_t count = 0;
    for (uint32_t i = 0; i < sampledCount; ++i) {
        bindings[count].binding = count;
        bindings[count].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[count].descriptorCount = 1;
        bindings[count].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        ++count;
    }
    bindings[count].binding = count;
    bindings[count].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[count].descriptorCount = 1;
    bindings[count].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    ++count;
    VkDescriptorSetLayoutCreateInfo setCi = {};
    setCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    setCi.bindingCount = count;
    setCi.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(ctx.device, &setCi, nullptr, &out.setLayout) != VK_SUCCESS)
        return false;

    VkPushConstantRange push = {};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.size = pushSize;
    VkPipelineLayoutCreateInfo layoutCi = {};
    layoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCi.setLayoutCount = 1;
    layoutCi.pSetLayouts = &out.setLayout;
    layoutCi.pushConstantRangeCount = pushSize > 0 ? 1u : 0u;
    layoutCi.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(ctx.device, &layoutCi, nullptr, &out.layout) != VK_SUCCESS)
        return false;

    VkComputePipelineCreateInfo pipeCi = {};
    pipeCi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeCi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeCi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeCi.stage.module = module;
    pipeCi.stage.pName = "main";
    pipeCi.layout = out.layout;
    return createComputePipeline(ctx, pipeCi, out.pipeline) == VK_SUCCESS;
}

// Producing scope for an image that was left in SHADER_READ_ONLY by a previous
// run (sync2).  UNDEFINED (fresh image) forces the source scope to NONE.
VkPipelineStageFlags2 sampledSrcStage(VkImageLayout oldLayout) {
    return oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
               ? VK_PIPELINE_STAGE_2_NONE
               : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
}
VkAccessFlags2 sampledSrcAccess(VkImageLayout oldLayout) {
    return oldLayout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_ACCESS_2_NONE
                                                  : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
}

} // namespace

void IblMaps::ComputeStage::destroy(const VulkanContext& ctx) {
    if (pipeline) vkDestroyPipeline(ctx.device, pipeline, nullptr);
    if (layout) vkDestroyPipelineLayout(ctx.device, layout, nullptr);
    if (setLayout) vkDestroyDescriptorSetLayout(ctx.device, setLayout, nullptr);
    pipeline = VK_NULL_HANDLE;
    layout = VK_NULL_HANDLE;
    setLayout = VK_NULL_HANDLE;
}

bool IblMaps::createStages(const VulkanContext& ctx, bool withEquirect, bool withSky) {
    VkShaderModule modEquirect = VK_NULL_HANDLE, modIrradiance = VK_NULL_HANDLE,
                   modPrefilter = VK_NULL_HANDLE, modLut = VK_NULL_HANDLE,
                   modSky = VK_NULL_HANDLE;
    const bool loaded =
        (!withEquirect || loadShaderModule(ctx, "ibl_equirect.comp.spv", modEquirect)) &&
        loadShaderModule(ctx, "ibl_irradiance.comp.spv", modIrradiance) &&
        loadShaderModule(ctx, "ibl_prefilter.comp.spv", modPrefilter) &&
        loadShaderModule(ctx, "ibl_brdf_lut.comp.spv", modLut) &&
        (!withSky || loadShaderModule(ctx, "sky_render.comp.spv", modSky));
    if (!loaded) {
        if (modEquirect) vkDestroyShaderModule(ctx.device, modEquirect, nullptr);
        if (modIrradiance) vkDestroyShaderModule(ctx.device, modIrradiance, nullptr);
        if (modPrefilter) vkDestroyShaderModule(ctx.device, modPrefilter, nullptr);
        if (modLut) vkDestroyShaderModule(ctx.device, modLut, nullptr);
        if (modSky) vkDestroyShaderModule(ctx.device, modSky, nullptr);
        return false;
    }

    const bool stagesOk =
        (!withEquirect || createStage(ctx, modEquirect, 1, 0, equirectStage_)) &&
        createStage(ctx, modIrradiance, 1, 4, irradianceStage_) &&
        createStage(ctx, modPrefilter, 1, 8, prefilterStage_) &&
        createStage(ctx, modLut, 0, 0, brdfLutStage_) &&
        (!withSky || createStage(ctx, modSky, 2, 16, skyStage_));
    if (modEquirect) vkDestroyShaderModule(ctx.device, modEquirect, nullptr);
    vkDestroyShaderModule(ctx.device, modIrradiance, nullptr);
    vkDestroyShaderModule(ctx.device, modPrefilter, nullptr);
    vkDestroyShaderModule(ctx.device, modLut, nullptr);
    if (modSky) vkDestroyShaderModule(ctx.device, modSky, nullptr);
    return stagesOk;
}

bool IblMaps::createTargets(const VulkanContext& ctx) {
    envMips_ = 1;
    for (uint32_t s = kEnvSize; s > 1; s >>= 1) ++envMips_;

    if (createImage(ctx, kEnvSize, kEnvSize, kHdrFormat,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    envImage_, envMemory_, envMips_, 6, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) !=
            VK_SUCCESS ||
        createImage(ctx, kIrradianceSize, kIrradianceSize, kHdrFormat,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, irrImage_, irrMemory_,
                    1, 6, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) != VK_SUCCESS ||
        createImage(ctx, kPrefilterSize, kPrefilterSize, kHdrFormat,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, preImage_, preMemory_,
                    kPrefilterMips, 6, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) != VK_SUCCESS ||
        createImage(ctx, kLutSize, kLutSize, kHdrFormat,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, lutImage_,
                    lutMemory_) != VK_SUCCESS)
        return false;

    envView = createImageView(ctx, envImage_, kHdrFormat, VK_IMAGE_ASPECT_COLOR_BIT, 0, envMips_,
                              VK_IMAGE_VIEW_TYPE_CUBE, 6);
    irradianceView = createImageView(ctx, irrImage_, kHdrFormat, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1,
                                     VK_IMAGE_VIEW_TYPE_CUBE, 6);
    prefilterView = createImageView(ctx, preImage_, kHdrFormat, VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                    kPrefilterMips, VK_IMAGE_VIEW_TYPE_CUBE, 6);
    brdfLutView = createImageView(ctx, lutImage_, kHdrFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    envStoreMip0_ = createImageView(ctx, envImage_, kHdrFormat, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1,
                                    VK_IMAGE_VIEW_TYPE_2D_ARRAY, 6);
    irrStore_ = createImageView(ctx, irrImage_, kHdrFormat, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1,
                                VK_IMAGE_VIEW_TYPE_2D_ARRAY, 6);
    if (!envView || !irradianceView || !prefilterView || !brdfLutView || !envStoreMip0_ ||
        !irrStore_)
        return false;
    for (uint32_t mip = 0; mip < kPrefilterMips; ++mip) {
        preStore_[mip] = createImageView(ctx, preImage_, kHdrFormat, VK_IMAGE_ASPECT_COLOR_BIT,
                                         mip, 1, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 6);
        if (!preStore_[mip]) return false;
    }

    cubeSampler = createSampler(ctx, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.f,
                                static_cast<float>(envMips_ - 1));
    lutSampler = createSampler(ctx, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    if (!cubeSampler || !lutSampler) return false;

    // Sets: equirect/sky (1) + irradiance (1) + prefilter (kPrefilterMips) +
    // BRDF LUT (1); the sky stage binds two samplers.
    VkDescriptorPoolSize poolSizes[2] = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = 16;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = 16;
    VkDescriptorPoolCreateInfo poolCi = {};
    poolCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCi.maxSets = 16;
    poolCi.poolSizeCount = 2;
    poolCi.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(ctx.device, &poolCi, nullptr, &pool_) != VK_SUCCESS) return false;

    auto allocSet = [&](VkDescriptorSetLayout layout) -> VkDescriptorSet {
        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = pool_;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &layout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        vkAllocateDescriptorSets(ctx.device, &alloc, &set);
        return set;
    };
    // srcView == VK_NULL_HANDLE: storage-only set (BRDF LUT).
    auto writeSet = [&](VkDescriptorSet set, VkSampler sampler, VkImageView srcView,
                        VkImageView dstView, uint32_t sampledCount) {
        VkDescriptorImageInfo src = {};
        src.sampler = sampler;
        src.imageView = srcView;
        src.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo dst = {};
        dst.imageView = dstView;
        dst.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet writes[2] = {};
        uint32_t count = 0;
        if (srcView) {
            writes[count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[count].dstSet = set;
            writes[count].dstBinding = sampledCount - 1;
            writes[count].descriptorCount = 1;
            writes[count].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[count].pImageInfo = &src;
            ++count;
        }
        writes[count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[count].dstSet = set;
        writes[count].dstBinding = srcView ? sampledCount : 0;
        writes[count].descriptorCount = 1;
        writes[count].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[count].pImageInfo = &dst;
        ++count;
        vkUpdateDescriptorSets(ctx.device, count, writes, 0, nullptr);
    };

    // Persistent sun-independent set writes: irradiance + prefilter read the
    // env cube and write their store views; sky re-runs reuse the same sets.
    irradianceSet_ = allocSet(irradianceStage_.setLayout);
    brdfLutSet_ = allocSet(brdfLutStage_.setLayout);
    if (!irradianceSet_ || !brdfLutSet_) return false;
    writeSet(irradianceSet_, cubeSampler, envView, irrStore_, 1);
    writeSet(brdfLutSet_, VK_NULL_HANDLE, VK_NULL_HANDLE, brdfLutView, 0);
    for (uint32_t mip = 0; mip < kPrefilterMips; ++mip) {
        prefilterSets_[mip] = allocSet(prefilterStage_.setLayout);
        if (!prefilterSets_[mip]) return false;
        writeSet(prefilterSets_[mip], cubeSampler, envView, preStore_[mip], 1);
    }
    return true;
}

void IblMaps::recordEnvMipChain(VkCommandBuffer cmd, VkImageLayout initialLayout) const {
    // mip 0 is in GENERAL (just written by the equirect/sky dispatch).
    imageBarrier(cmd, envImage_, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                 VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6);
    // Downsample the remaining env mip levels (per layer).
    for (uint32_t level = 1; level < envMips_; ++level) {
        imageBarrier(cmd, envImage_, initialLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     sampledSrcStage(initialLayout), sampledSrcAccess(initialLayout),
                     VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 0, 6);
        for (uint32_t layer = 0; layer < 6; ++layer) {
            VkImageBlit blit = {};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, layer, 1};
            blit.srcOffsets[0] = {0, 0, 0};
            blit.srcOffsets[1] = {static_cast<int32_t>(kEnvSize >> (level - 1)),
                                  static_cast<int32_t>(kEnvSize >> (level - 1)), 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, layer, 1};
            blit.dstOffsets[0] = {0, 0, 0};
            blit.dstOffsets[1] = {static_cast<int32_t>(kEnvSize >> level),
                                  static_cast<int32_t>(kEnvSize >> level), 1};
            vkCmdBlitImage(cmd, envImage_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, envImage_,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
        }
        imageBarrier(cmd, envImage_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_BLIT_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT,
                     VK_ACCESS_2_TRANSFER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 0, 6);
    }
    // Consumers: the irradiance/prefilter compute passes and the
    // lighting/skybox fragment shaders.
    imageBarrier(cmd, envImage_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_BLIT_BIT,
                 VK_ACCESS_2_TRANSFER_READ_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, envMips_, 0,
                 6);
}

void IblMaps::recordDownstream(VkCommandBuffer cmd, VkImageLayout initialLayout,
                               bool withBrdfLut) const {
    // --- irradiance convolution ----------------------------------------------
    imageBarrier(cmd, irrImage_, initialLayout, VK_IMAGE_LAYOUT_GENERAL,
                 sampledSrcStage(initialLayout), sampledSrcAccess(initialLayout),
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, irradianceStage_.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, irradianceStage_.layout, 0, 1,
                            &irradianceSet_, 0, nullptr);
    const float maxLod = 4.f < static_cast<float>(envMips_ - 1)
                             ? 4.f
                             : static_cast<float>(envMips_ - 1); // band-limit, keep directionality
    vkCmdPushConstants(cmd, irradianceStage_.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &maxLod);
    vkCmdDispatch(cmd, kIrradianceSize / 8, kIrradianceSize / 8, 6);
    // Consumer: lighting/transparent fragment shaders (diffuse IBL).
    imageBarrier(cmd, irrImage_, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                 VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6);

    // --- prefiltered specular (one dispatch per mip) --------------------------
    for (uint32_t mip = 0; mip < kPrefilterMips; ++mip) {
        const float roughness = static_cast<float>(mip) / static_cast<float>(kPrefilterMips - 1);
        const float push[2] = {roughness, static_cast<float>(kEnvSize)};
        const uint32_t size = kPrefilterSize >> mip;
        imageBarrier(cmd, preImage_, initialLayout, VK_IMAGE_LAYOUT_GENERAL,
                     sampledSrcStage(initialLayout), sampledSrcAccess(initialLayout),
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 6);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prefilterStage_.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prefilterStage_.layout, 0, 1,
                                &prefilterSets_[mip], 0, nullptr);
        vkCmdPushConstants(cmd, prefilterStage_.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, push);
        vkCmdDispatch(cmd, (size + 7) / 8, (size + 7) / 8, 6);
        // Consumer: lighting/transparent fragment shaders (specular IBL).
        imageBarrier(cmd, preImage_, VK_IMAGE_LAYOUT_GENERAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 6);
    }

    // --- BRDF LUT (sun-independent; first build only) --------------------------
    if (withBrdfLut) {
        imageBarrier(cmd, lutImage_, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, brdfLutStage_.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, brdfLutStage_.layout, 0, 1,
                                &brdfLutSet_, 0, nullptr);
        vkCmdDispatch(cmd, kLutSize / 8, kLutSize / 8, 1);
        // Consumer: lighting/transparent fragment shaders (split-sum LUT).
        imageBarrier(cmd, lutImage_, VK_IMAGE_LAYOUT_GENERAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }
}

void IblMaps::recordSkyRender(VkCommandBuffer cmd, const Vec3& sunDir,
                              VkImageLayout initialLayout) const {
    // sky_render.comp writes env cube mip 0 (ends in GENERAL for the mip chain).
    imageBarrier(cmd, envImage_, initialLayout, VK_IMAGE_LAYOUT_GENERAL,
                 sampledSrcStage(initialLayout), sampledSrcAccess(initialLayout),
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, skyStage_.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, skyStage_.layout, 0, 1, &skySet_,
                            0, nullptr);
    const float push[4] = {sunDir.x, sunDir.y, sunDir.z, SkyAtmosphere::kSunIlluminance};
    vkCmdPushConstants(cmd, skyStage_.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, push);
    vkCmdDispatch(cmd, kEnvSize / 8, kEnvSize / 8, 6);
}

bool IblMaps::build(const VulkanContext& ctx, const char* hdrPath) {
    // --- load / synthesize the equirect environment -------------------------
    int w = 0, h = 0, comp = 0;
    float* hdr = nullptr;
    std::string resolvedPath;
    if (hdrPath && hdrPath[0] != '\0') {
        // Packaged assets resolve relative to the exe, not just the CWD.
        resolvedPath = resolveAssetPath(hdrPath);
        hdr = stbi_loadf(resolvedPath.c_str(), &w, &h, &comp, 3);
    }
    std::vector<uint16_t> pixels;
    if (hdr) {
        fromFile = true;
        pixels.resize(static_cast<size_t>(w) * h * 4);
        for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
            pixels[i * 4 + 0] = floatToHalf(hdr[i * 3 + 0]);
            pixels[i * 4 + 1] = floatToHalf(hdr[i * 3 + 1]);
            pixels[i * 4 + 2] = floatToHalf(hdr[i * 3 + 2]);
            pixels[i * 4 + 3] = floatToHalf(1.f);
        }
        stbi_image_free(hdr);
    } else {
        // Fallback: dim vertical gradient (dark blue sky, grayish ground),
        // roughly matching the legacy constant ambient light.
        if (hdrPath && hdrPath[0] != '\0')
            std::fprintf(stderr, "ibl: failed to load '%s', using fallback gradient\n", hdrPath);
        w = 64;
        h = 32;
        pixels.resize(static_cast<size_t>(w) * h * 4);
        for (int y = 0; y < h; ++y) {
            const float t = static_cast<float>(y) / static_cast<float>(h - 1); // 0 = up
            const float sky = std::max(0.f, 1.f - t * 2.f);
            const float ground = std::max(0.f, (t - 0.5f) * 2.f);
            const float r = 0.03f + 0.05f * sky + 0.06f * ground;
            const float g = 0.04f + 0.05f * sky + 0.06f * ground;
            const float b = 0.07f + 0.08f * sky + 0.05f * ground;
            for (int x = 0; x < w; ++x) {
                const size_t i = static_cast<size_t>(y) * w + x;
                pixels[i * 4 + 0] = floatToHalf(r);
                pixels[i * 4 + 1] = floatToHalf(g);
                pixels[i * 4 + 2] = floatToHalf(b);
                pixels[i * 4 + 3] = floatToHalf(1.f);
            }
        }
    }

    if (!createStages(ctx, /*withEquirect=*/true, /*withSky=*/false) ||
        !createTargets(ctx))
        return false;

    // --- equirect source upload ------------------------------------------------
    VkImage eqImage = VK_NULL_HANDLE;
    VmaAllocation eqMemory = VK_NULL_HANDLE;
    VkImageView eqView = VK_NULL_HANDLE;
    if (!uploadHdrImage(ctx, static_cast<uint32_t>(w), static_cast<uint32_t>(h), pixels.data(),
                        eqImage, eqMemory, eqView))
        return false;

    VkDescriptorSetAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorPool = pool_;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &equirectStage_.setLayout;
    VkDescriptorSet eqSet = VK_NULL_HANDLE;
    vkAllocateDescriptorSets(ctx.device, &alloc, &eqSet);
    if (!eqSet) return false;
    {
        VkDescriptorImageInfo src = {};
        src.sampler = cubeSampler;
        src.imageView = eqView;
        src.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo dst = {};
        dst.imageView = envStoreMip0_;
        dst.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet writes[2] = {};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = eqSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &src;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = eqSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &dst;
        vkUpdateDescriptorSets(ctx.device, 2, writes, 0, nullptr);
    }

    // --- equirect -> cube mip 0, then the shared downstream chain -------------
    submitOneShot(ctx, [&](VkCommandBuffer cmd) {
        imageBarrier(cmd, envImage_, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, equirectStage_.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, equirectStage_.layout, 0, 1,
                                &eqSet, 0, nullptr);
        vkCmdDispatch(cmd, kEnvSize / 8, kEnvSize / 8, 6);
        recordEnvMipChain(cmd, VK_IMAGE_LAYOUT_UNDEFINED);
        recordDownstream(cmd, VK_IMAGE_LAYOUT_UNDEFINED, /*withBrdfLut=*/true);
    });

    // --- cleanup of intermediates ----------------------------------------------
    vkDestroyImageView(ctx.device, eqView, nullptr);
    vmaDestroyImage(ctx.allocator, eqImage, eqMemory);
    return true;
}

bool IblMaps::buildAtmosphere(const VulkanContext& ctx, const SkyAtmosphere& sky,
                              const Vec3& sunDir) {
    if (!createStages(ctx, /*withEquirect=*/false, /*withSky=*/true) || !createTargets(ctx))
        return false;

    // Sky set: transmittance + multi-scatter LUTs (sampled) + env cube mip 0
    // (storage).  Persistent across updateAtmosphereSky re-runs.
    VkDescriptorSetAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorPool = pool_;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &skyStage_.setLayout;
    vkAllocateDescriptorSets(ctx.device, &alloc, &skySet_);
    if (!skySet_) return false;
    {
        VkDescriptorImageInfo trans = {};
        trans.sampler = sky.lutSampler();
        trans.imageView = sky.transmittanceView();
        trans.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo ms = trans;
        ms.imageView = sky.multiScatterView();
        VkDescriptorImageInfo dst = {};
        dst.imageView = envStoreMip0_;
        dst.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet writes[3] = {};
        for (uint32_t i = 0; i < 3; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = skySet_;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
        }
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &trans;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo = &ms;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[2].pImageInfo = &dst;
        vkUpdateDescriptorSets(ctx.device, 3, writes, 0, nullptr);
    }

    submitOneShot(ctx, [&](VkCommandBuffer cmd) {
        recordSkyRender(cmd, sunDir, VK_IMAGE_LAYOUT_UNDEFINED);
        recordEnvMipChain(cmd, VK_IMAGE_LAYOUT_UNDEFINED);
        recordDownstream(cmd, VK_IMAGE_LAYOUT_UNDEFINED, /*withBrdfLut=*/true);
    });
    fromAtmosphere = true;
    fromFile = false;
    return true;
}

bool IblMaps::updateAtmosphereSky(const VulkanContext& ctx, const SkyAtmosphere& sky,
                                  const Vec3& sunDir) {
    (void)sky; // the sky LUT set was written by buildAtmosphere (same object)
    if (!fromAtmosphere || !skyStage_.pipeline || !skySet_) return false;
    submitOneShot(ctx, [&](VkCommandBuffer cmd) {
        recordSkyRender(cmd, sunDir, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        recordEnvMipChain(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        recordDownstream(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         /*withBrdfLut=*/false);
    });
    return true;
}

void IblMaps::destroy(const VulkanContext& ctx) {
    if (!ctx.device) return;
    if (envView) vkDestroyImageView(ctx.device, envView, nullptr);
    if (irradianceView) vkDestroyImageView(ctx.device, irradianceView, nullptr);
    if (prefilterView) vkDestroyImageView(ctx.device, prefilterView, nullptr);
    if (brdfLutView) vkDestroyImageView(ctx.device, brdfLutView, nullptr);
    if (envStoreMip0_) vkDestroyImageView(ctx.device, envStoreMip0_, nullptr);
    if (irrStore_) vkDestroyImageView(ctx.device, irrStore_, nullptr);
    for (uint32_t mip = 0; mip < kPrefilterMips; ++mip) {
        if (preStore_[mip]) vkDestroyImageView(ctx.device, preStore_[mip], nullptr);
    }
    if (envImage_) vmaDestroyImage(ctx.allocator, envImage_, envMemory_);
    if (irrImage_) vmaDestroyImage(ctx.allocator, irrImage_, irrMemory_);
    if (preImage_) vmaDestroyImage(ctx.allocator, preImage_, preMemory_);
    if (lutImage_) vmaDestroyImage(ctx.allocator, lutImage_, lutMemory_);
    if (cubeSampler) vkDestroySampler(ctx.device, cubeSampler, nullptr);
    if (lutSampler) vkDestroySampler(ctx.device, lutSampler, nullptr);
    if (pool_) vkDestroyDescriptorPool(ctx.device, pool_, nullptr);
    equirectStage_.destroy(ctx);
    irradianceStage_.destroy(ctx);
    prefilterStage_.destroy(ctx);
    brdfLutStage_.destroy(ctx);
    skyStage_.destroy(ctx);
    *this = IblMaps{};
}

} // namespace sr
