#include "renderer/ibl/Ibl.h"

#include "renderer/core/PathUtil.h"
#include "renderer/core/VkUtil.h"

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
                    VkImage& image, VkDeviceMemory& memory, VkImageView& view) {
    if (createImage(ctx, w, h, kHdrFormat,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, image,
                    memory) != VK_SUCCESS)
        return false;
    const VkDeviceSize size = static_cast<VkDeviceSize>(w) * h * 8;
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    if (createBuffer(ctx, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging, stagingMemory) != VK_SUCCESS)
        return false;
    void* mapped = nullptr;
    vkMapMemory(ctx.device, stagingMemory, 0, size, 0, &mapped);
    std::memcpy(mapped, data, static_cast<size_t>(size));
    vkUnmapMemory(ctx.device, stagingMemory);
    submitOneShot(ctx, [&](VkCommandBuffer cmd) {
        copyBufferToImage(cmd, staging, image, w, h, kHdrFormat);
    });
    vkDestroyBuffer(ctx.device, staging, nullptr);
    vkFreeMemory(ctx.device, stagingMemory, nullptr);
    view = createImageView(ctx, image, kHdrFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    return view != VK_NULL_HANDLE;
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

    // --- shaders + compute pipelines ----------------------------------------
    VkShaderModule modEquirect = VK_NULL_HANDLE, modIrradiance = VK_NULL_HANDLE,
                   modPrefilter = VK_NULL_HANDLE, modLut = VK_NULL_HANDLE;
    if (!loadShaderModule(ctx, "ibl_equirect.comp.spv", modEquirect) ||
        !loadShaderModule(ctx, "ibl_irradiance.comp.spv", modIrradiance) ||
        !loadShaderModule(ctx, "ibl_prefilter.comp.spv", modPrefilter) ||
        !loadShaderModule(ctx, "ibl_brdf_lut.comp.spv", modLut))
        return false;

    auto createStage = [&](VkShaderModule module, VkDescriptorType srcType, bool hasSrc,
                           uint32_t pushSize, ComputeStage& out) -> bool {
        VkDescriptorSetLayoutBinding bindings[2] = {};
        uint32_t count = 0;
        if (hasSrc) {
            bindings[0].binding = 0;
            bindings[0].descriptorType = srcType;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            ++count;
        }
        bindings[count].binding = hasSrc ? 1 : 0;
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
        return vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeCi, nullptr,
                                        &out.pipeline) == VK_SUCCESS;
    };

    const bool stagesOk =
        createStage(modEquirect, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, true, 0,
                    equirectStage_) &&
        createStage(modIrradiance, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, true, 4,
                    irradianceStage_) &&
        createStage(modPrefilter, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, true, 8,
                    prefilterStage_) &&
        createStage(modLut, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, false, 0, brdfLutStage_);
    vkDestroyShaderModule(ctx.device, modEquirect, nullptr);
    vkDestroyShaderModule(ctx.device, modIrradiance, nullptr);
    vkDestroyShaderModule(ctx.device, modPrefilter, nullptr);
    vkDestroyShaderModule(ctx.device, modLut, nullptr);
    if (!stagesOk) return false;

    // --- GPU resources ------------------------------------------------------
    VkImage eqImage = VK_NULL_HANDLE;
    VkDeviceMemory eqMemory = VK_NULL_HANDLE;
    VkImageView eqView = VK_NULL_HANDLE;
    if (!uploadHdrImage(ctx, static_cast<uint32_t>(w), static_cast<uint32_t>(h), pixels.data(),
                        eqImage, eqMemory, eqView))
        return false;

    uint32_t envMips = 1;
    for (uint32_t s = kEnvSize; s > 1; s >>= 1) ++envMips;

    if (createImage(ctx, kEnvSize, kEnvSize, kHdrFormat,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    envImage_, envMemory_, envMips, 6, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) !=
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

    envView = createImageView(ctx, envImage_, kHdrFormat, VK_IMAGE_ASPECT_COLOR_BIT, 0, envMips,
                              VK_IMAGE_VIEW_TYPE_CUBE, 6);
    irradianceView = createImageView(ctx, irrImage_, kHdrFormat, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1,
                                     VK_IMAGE_VIEW_TYPE_CUBE, 6);
    prefilterView = createImageView(ctx, preImage_, kHdrFormat, VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                    kPrefilterMips, VK_IMAGE_VIEW_TYPE_CUBE, 6);
    brdfLutView = createImageView(ctx, lutImage_, kHdrFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    VkImageView envStoreMip0 = createImageView(ctx, envImage_, kHdrFormat,
                                               VK_IMAGE_ASPECT_COLOR_BIT, 0, 1,
                                               VK_IMAGE_VIEW_TYPE_2D_ARRAY, 6);
    VkImageView irrStore = createImageView(ctx, irrImage_, kHdrFormat, VK_IMAGE_ASPECT_COLOR_BIT,
                                           0, 1, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 6);
    if (!envView || !irradianceView || !prefilterView || !brdfLutView || !envStoreMip0 ||
        !irrStore)
        return false;

    cubeSampler = createSampler(ctx, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.f,
                                static_cast<float>(envMips - 1));
    lutSampler = createSampler(ctx, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    VkDescriptorPoolSize poolSizes[2] = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = 8;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = 8;
    VkDescriptorPoolCreateInfo poolCi = {};
    poolCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCi.maxSets = 8;
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
    auto writeSet = [&](VkDescriptorSet set, VkSampler sampler, VkImageView srcView,
                        VkImageView dstView) {
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
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = set;
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[0].pImageInfo = &src;
            ++count;
        }
        writes[count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[count].dstSet = set;
        writes[count].dstBinding = srcView ? 1 : 0;
        writes[count].descriptorCount = 1;
        writes[count].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[count].pImageInfo = &dst;
        ++count;
        vkUpdateDescriptorSets(ctx.device, count, writes, 0, nullptr);
    };

    // --- equirect -> cube mip 0 ---------------------------------------------
    {
        VkDescriptorSet set = allocSet(equirectStage_.setLayout);
        writeSet(set, cubeSampler, eqView, envStoreMip0);
        submitOneShot(ctx, [&](VkCommandBuffer cmd) {
            imageBarrier(cmd, envImage_, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, equirectStage_.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, equirectStage_.layout, 0,
                                    1, &set, 0, nullptr);
            vkCmdDispatch(cmd, kEnvSize / 8, kEnvSize / 8, 6);
            imageBarrier(cmd, envImage_, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                         6);
            // Downsample the remaining env mip levels (per layer).
            for (uint32_t level = 1; level < envMips; ++level) {
                imageBarrier(cmd, envImage_, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT,
                             level, 1, 0, 6);
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
                    vkCmdBlitImage(cmd, envImage_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   envImage_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                                   VK_FILTER_LINEAR);
                }
                imageBarrier(cmd, envImage_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT,
                             level, 1, 0, 6);
            }
            imageBarrier(cmd, envImage_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, 0,
                         envMips, 0, 6);
        });
    }

    // --- irradiance convolution ----------------------------------------------
    {
        VkDescriptorSet set = allocSet(irradianceStage_.setLayout);
        writeSet(set, cubeSampler, envView, irrStore);
        submitOneShot(ctx, [&](VkCommandBuffer cmd) {
            imageBarrier(cmd, irrImage_, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, irradianceStage_.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, irradianceStage_.layout,
                                    0, 1, &set, 0, nullptr);
            const float maxLod = 4.f < static_cast<float>(envMips - 1)
                                     ? 4.f
                                     : static_cast<float>(envMips - 1); // band-limit, keep directionality
            vkCmdPushConstants(cmd, irradianceStage_.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4,
                               &maxLod);
            vkCmdDispatch(cmd, kIrradianceSize / 8, kIrradianceSize / 8, 6);
            imageBarrier(cmd, irrImage_, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1,
                         0, 6);
        });
    }

    // --- prefiltered specular (one dispatch per mip) --------------------------
    for (uint32_t mip = 0; mip < kPrefilterMips; ++mip) {
        VkImageView dst = createImageView(ctx, preImage_, kHdrFormat, VK_IMAGE_ASPECT_COLOR_BIT,
                                          mip, 1, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 6);
        if (!dst) return false;
        VkDescriptorSet set = allocSet(prefilterStage_.setLayout);
        writeSet(set, cubeSampler, envView, dst);
        const float roughness = static_cast<float>(mip) / static_cast<float>(kPrefilterMips - 1);
        const float push[2] = {roughness, static_cast<float>(kEnvSize)};
        const uint32_t size = kPrefilterSize >> mip;
        submitOneShot(ctx, [&](VkCommandBuffer cmd) {
            imageBarrier(cmd, preImage_, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 6);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prefilterStage_.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prefilterStage_.layout, 0,
                                    1, &set, 0, nullptr);
            vkCmdPushConstants(cmd, prefilterStage_.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8,
                               push);
            vkCmdDispatch(cmd, (size + 7) / 8, (size + 7) / 8, 6);
            imageBarrier(cmd, preImage_, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, mip,
                         1, 0, 6);
        });
        vkDestroyImageView(ctx.device, dst, nullptr); // set holds no ownership; safe post-submit
    }

    // --- BRDF LUT --------------------------------------------------------------
    {
        VkDescriptorSet set = allocSet(brdfLutStage_.setLayout);
        writeSet(set, VK_NULL_HANDLE, VK_NULL_HANDLE, brdfLutView);
        submitOneShot(ctx, [&](VkCommandBuffer cmd) {
            imageBarrier(cmd, lutImage_, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, brdfLutStage_.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, brdfLutStage_.layout, 0,
                                    1, &set, 0, nullptr);
            vkCmdDispatch(cmd, kLutSize / 8, kLutSize / 8, 1);
            imageBarrier(cmd, lutImage_, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });
    }

    // --- cleanup of intermediates ----------------------------------------------
    vkDestroyImageView(ctx.device, eqView, nullptr);
    vkDestroyImage(ctx.device, eqImage, nullptr);
    vkFreeMemory(ctx.device, eqMemory, nullptr);
    vkDestroyImageView(ctx.device, envStoreMip0, nullptr);
    vkDestroyImageView(ctx.device, irrStore, nullptr);
    return true;
}

void IblMaps::destroy(const VulkanContext& ctx) {
    if (!ctx.device) return;
    if (envView) vkDestroyImageView(ctx.device, envView, nullptr);
    if (irradianceView) vkDestroyImageView(ctx.device, irradianceView, nullptr);
    if (prefilterView) vkDestroyImageView(ctx.device, prefilterView, nullptr);
    if (brdfLutView) vkDestroyImageView(ctx.device, brdfLutView, nullptr);
    if (envImage_) vkDestroyImage(ctx.device, envImage_, nullptr);
    if (envMemory_) vkFreeMemory(ctx.device, envMemory_, nullptr);
    if (irrImage_) vkDestroyImage(ctx.device, irrImage_, nullptr);
    if (irrMemory_) vkFreeMemory(ctx.device, irrMemory_, nullptr);
    if (preImage_) vkDestroyImage(ctx.device, preImage_, nullptr);
    if (preMemory_) vkFreeMemory(ctx.device, preMemory_, nullptr);
    if (lutImage_) vkDestroyImage(ctx.device, lutImage_, nullptr);
    if (lutMemory_) vkFreeMemory(ctx.device, lutMemory_, nullptr);
    if (cubeSampler) vkDestroySampler(ctx.device, cubeSampler, nullptr);
    if (lutSampler) vkDestroySampler(ctx.device, lutSampler, nullptr);
    if (pool_) vkDestroyDescriptorPool(ctx.device, pool_, nullptr);
    equirectStage_.destroy(ctx);
    irradianceStage_.destroy(ctx);
    prefilterStage_.destroy(ctx);
    brdfLutStage_.destroy(ctx);
    *this = IblMaps{};
}

} // namespace sr
