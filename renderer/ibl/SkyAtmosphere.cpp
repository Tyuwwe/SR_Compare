#include "renderer/ibl/SkyAtmosphere.h"

#include "renderer/core/PathUtil.h"
#include "renderer/core/VkUtil.h"
#include "renderer/core/Vma.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

namespace sr {

namespace {

constexpr VkFormat kLutFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

bool loadShaderModule(const VulkanContext& ctx, const char* name, VkShaderModule& out) {
    const std::string path = sr::resolveShaderPath(SR_SHADER_DIR, name);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "sky: failed to open shader %s\n", path.c_str());
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

// Compute stage: <sampledCount> combined-image-sampler bindings followed by
// one storage image (binding <sampledCount>), no push constants.
bool createStage(const VulkanContext& ctx, VkShaderModule module, uint32_t sampledCount,
                 SkyAtmosphere::Stage& out) {
    VkDescriptorSetLayoutBinding bindings[3] = {};
    for (uint32_t i = 0; i < sampledCount; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    bindings[sampledCount].binding = sampledCount;
    bindings[sampledCount].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[sampledCount].descriptorCount = 1;
    bindings[sampledCount].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo setCi = {};
    setCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    setCi.bindingCount = sampledCount + 1;
    setCi.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(ctx.device, &setCi, nullptr, &out.setLayout) != VK_SUCCESS)
        return false;

    VkPipelineLayoutCreateInfo layoutCi = {};
    layoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCi.setLayoutCount = 1;
    layoutCi.pSetLayouts = &out.setLayout;
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

void destroyStage(const VulkanContext& ctx, SkyAtmosphere::Stage& stage) {
    if (stage.pipeline) vkDestroyPipeline(ctx.device, stage.pipeline, nullptr);
    if (stage.layout) vkDestroyPipelineLayout(ctx.device, stage.layout, nullptr);
    if (stage.setLayout) vkDestroyDescriptorSetLayout(ctx.device, stage.setLayout, nullptr);
    stage = SkyAtmosphere::Stage{};
}

} // namespace

bool SkyAtmosphere::init(const VulkanContext& ctx) {
    // --- LUT images -----------------------------------------------------------
    if (createImage(ctx, kTransmittanceWidth, kTransmittanceHeight, kLutFormat,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, transImage_,
                    transMemory_) != VK_SUCCESS ||
        createImage(ctx, kMultiScatterSize, kMultiScatterSize, kLutFormat,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, msImage_,
                    msMemory_) != VK_SUCCESS)
        return false;
    transView_ = createImageView(ctx, transImage_, kLutFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    msView_ = createImageView(ctx, msImage_, kLutFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    sampler_ = createSampler(ctx, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    if (!transView_ || !msView_ || !sampler_) return false;

    // --- pipelines --------------------------------------------------------------
    VkShaderModule modTrans = VK_NULL_HANDLE, modMs = VK_NULL_HANDLE;
    if (!loadShaderModule(ctx, "sky_transmittance.comp.spv", modTrans) ||
        !loadShaderModule(ctx, "sky_multiscatter.comp.spv", modMs))
        return false;
    const bool stagesOk = createStage(ctx, modTrans, 0, transStage_) &&
                          createStage(ctx, modMs, 1, msStage_);
    vkDestroyShaderModule(ctx.device, modTrans, nullptr);
    vkDestroyShaderModule(ctx.device, modMs, nullptr);
    if (!stagesOk) return false;

    // --- descriptors --------------------------------------------------------------
    VkDescriptorPoolSize poolSizes[2] = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = 2;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = 2;
    VkDescriptorPoolCreateInfo poolCi = {};
    poolCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCi.maxSets = 2;
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

    VkDescriptorSet transSet = allocSet(transStage_.setLayout);
    VkDescriptorSet msSet = allocSet(msStage_.setLayout);
    if (!transSet || !msSet) return false;
    {
        VkDescriptorImageInfo dst = {};
        dst.imageView = transView_;
        dst.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = transSet;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write.pImageInfo = &dst;
        vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
    }
    {
        VkDescriptorImageInfo src = {};
        src.sampler = sampler_;
        src.imageView = transView_;
        src.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo dst = {};
        dst.imageView = msView_;
        dst.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet writes[2] = {};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = msSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &src;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = msSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &dst;
        vkUpdateDescriptorSets(ctx.device, 2, writes, 0, nullptr);
    }

    // --- bake (two one-shot dispatches; multi-scatter reads the transmittance LUT)
    submitOneShot(ctx, [&](VkCommandBuffer cmd) {
        imageBarrier(cmd, transImage_, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, transStage_.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, transStage_.layout, 0, 1,
                                &transSet, 0, nullptr);
        vkCmdDispatch(cmd, kTransmittanceWidth / 8, kTransmittanceHeight / 8, 1);
        imageBarrier(cmd, transImage_, VK_IMAGE_LAYOUT_GENERAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

        imageBarrier(cmd, msImage_, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, msStage_.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, msStage_.layout, 0, 1, &msSet,
                                0, nullptr);
        vkCmdDispatch(cmd, kMultiScatterSize / 8, kMultiScatterSize / 8, 1);
        imageBarrier(cmd, msImage_, VK_IMAGE_LAYOUT_GENERAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    });
    return true;
}

void SkyAtmosphere::destroy(const VulkanContext& ctx) {
    if (!ctx.device) return;
    if (transView_) vkDestroyImageView(ctx.device, transView_, nullptr);
    if (msView_) vkDestroyImageView(ctx.device, msView_, nullptr);
    if (transImage_) vmaDestroyImage(ctx.allocator, transImage_, transMemory_);
    if (msImage_) vmaDestroyImage(ctx.allocator, msImage_, msMemory_);
    if (sampler_) vkDestroySampler(ctx.device, sampler_, nullptr);
    if (pool_) vkDestroyDescriptorPool(ctx.device, pool_, nullptr);
    destroyStage(ctx, transStage_);
    destroyStage(ctx, msStage_);
    *this = SkyAtmosphere{};
}

Vec3 SkyAtmosphere::sunTransmittanceFromGround(const Vec3& sunDir) {
    // Constants mirror shaders/atmosphere.glsl (keep in sync).
    constexpr float kBottomRadius = 6360.f;
    constexpr float kTopRadius = 6460.f;
    constexpr float kRayleighDensityScale = -1.f / 8.f;
    constexpr float kMieDensityScale = -1.f / 1.2f;
    constexpr float kMieExtinction = 0.004440f;
    const Vec3 kRayleigh{0.005802f, 0.013558f, 0.033100f};
    const Vec3 kOzone{0.000650f, 0.001881f, 0.000085f};
    constexpr int kSteps = 64;

    const Vec3 dir = normalize(sunDir);
    const Vec3 pos{0.f, kBottomRadius + 0.2f, 0.f}; // 200 m above ground, up = +Y
    if (dir.y <= 0.f) return {0.f, 0.f, 0.f}; // below the horizon

    // Distance to the top of the atmosphere along the sun ray.
    const float b = dot(pos, dir);
    const float c = dot(pos, pos) - kTopRadius * kTopRadius;
    const float disc = b * b - c;
    if (disc < 0.f) return {0.f, 0.f, 0.f};
    const float tMax = -b + std::sqrt(disc);

    const float dt = tMax / static_cast<float>(kSteps);
    Vec3 od{0.f, 0.f, 0.f};
    for (int i = 0; i < kSteps; ++i) {
        const float t = (static_cast<float>(i) + 0.5f) * dt;
        const Vec3 p = pos + dir * t;
        const float h = length(p) - kBottomRadius;
        const float dMie = std::exp(kMieDensityScale * h);
        const float dRay = std::exp(kRayleighDensityScale * h);
        const float tent = h < 25.f ? h / 15.f - 2.f / 3.f : 8.f / 3.f - h / 15.f;
        const float dOzo = tent < 0.f ? 0.f : (tent > 1.f ? 1.f : tent);
        od.x += (kRayleigh.x * dRay + kMieExtinction * dMie + kOzone.x * dOzo) * dt;
        od.y += (kRayleigh.y * dRay + kMieExtinction * dMie + kOzone.y * dOzo) * dt;
        od.z += (kRayleigh.z * dRay + kMieExtinction * dMie + kOzone.z * dOzo) * dt;
    }
    return {std::exp(-od.x), std::exp(-od.y), std::exp(-od.z)};
}

} // namespace sr
