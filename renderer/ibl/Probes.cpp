#include "renderer/ibl/Probes.h"

#include "renderer/core/PathUtil.h"
#include "renderer/core/VkUtil.h"
#include "renderer/core/Vma.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace sr {

namespace {

constexpr VkFormat kHdrFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr uint32_t kFileMagic = 0x31505253u; // "SRP1" little-endian
constexpr uint32_t kFileVersion = 1;

bool loadShaderModule(const VulkanContext& ctx, const char* name, VkShaderModule& out) {
    const std::string path = sr::resolveShaderPath(SR_SHADER_DIR, name);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "probes: failed to open shader %s\n", path.c_str());
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

// 2D_ARRAY storage view of (mip, layers [baseLayer, baseLayer+6)) — the write
// target of one probe's irradiance/prefilter dispatch.  VkUtil's
// createImageView has no baseLayer, hence the raw call.
VkImageView createLayerMipView(const VulkanContext& ctx, VkImage image, VkFormat format,
                               uint32_t mip, uint32_t baseLayer, uint32_t layers) {
    VkImageViewCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ci.image = image;
    ci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    ci.format = format;
    ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, baseLayer, layers};
    VkImageView view = VK_NULL_HANDLE;
    if (vkCreateImageView(ctx.device, &ci, nullptr, &view) != VK_SUCCESS) return VK_NULL_HANDLE;
    return view;
}

} // namespace

bool saveProbeFile(const char* path, const std::vector<ReflectionProbe>& probes,
                   uint32_t faceSize, const std::vector<uint16_t>& rgba) {
    const size_t expected =
        static_cast<size_t>(probes.size()) * 6 * faceSize * faceSize * 4;
    if (rgba.size() != expected) {
        std::fprintf(stderr, "probes: save buffer size mismatch (%zu != %zu)\n", rgba.size(),
                     expected);
        return false;
    }
    ensureParentDir(path);
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        std::fprintf(stderr, "probes: cannot write %s\n", path);
        return false;
    }
    const uint32_t header[4] = {kFileMagic, kFileVersion,
                                static_cast<uint32_t>(probes.size()), faceSize};
    f.write(reinterpret_cast<const char*>(header), sizeof(header));
    for (const ReflectionProbe& p : probes) {
        const float box[9] = {p.position.x, p.position.y, p.position.z, p.boxMin.x, p.boxMin.y,
                              p.boxMin.z,   p.boxMax.x,   p.boxMax.y, p.boxMax.z};
        f.write(reinterpret_cast<const char*>(box), sizeof(box));
    }
    f.write(reinterpret_cast<const char*>(rgba.data()),
            static_cast<std::streamsize>(rgba.size() * sizeof(uint16_t)));
    return f.good();
}

bool loadProbeFile(const char* path, const std::vector<ReflectionProbe>& defs,
                   uint32_t expectedFaceSize, std::vector<uint16_t>& rgbaOut) {
    std::ifstream f(resolveAssetPath(path), std::ios::binary);
    if (!f) return false; // no bake file: probes stay inactive (global env only)
    uint32_t header[4] = {};
    f.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!f || header[0] != kFileMagic || header[1] != kFileVersion ||
        header[2] != defs.size() || header[3] != expectedFaceSize) {
        std::fprintf(stderr, "probes: %s is missing/stale, probes disabled\n", path);
        return false;
    }
    for (const ReflectionProbe& d : defs) {
        float box[9] = {};
        f.read(reinterpret_cast<char*>(box), sizeof(box));
        const float want[9] = {d.position.x, d.position.y, d.position.z, d.boxMin.x, d.boxMin.y,
                               d.boxMin.z,   d.boxMax.x,   d.boxMax.y, d.boxMax.z};
        if (!f || std::memcmp(box, want, sizeof(box)) != 0) {
            std::fprintf(stderr, "probes: %s no longer matches the scene placements, rebake\n",
                         path);
            return false;
        }
    }
    rgbaOut.resize(static_cast<size_t>(defs.size()) * 6 * expectedFaceSize * expectedFaceSize * 4);
    f.read(reinterpret_cast<char*>(rgbaOut.data()),
           static_cast<std::streamsize>(rgbaOut.size() * sizeof(uint16_t)));
    if (!f) {
        std::fprintf(stderr, "probes: %s truncated, probes disabled\n", path);
        rgbaOut.clear();
        return false;
    }
    return true;
}

void ReflectionProbes::ComputeStage::destroy(const VulkanContext& ctx) {
    if (pipeline) vkDestroyPipeline(ctx.device, pipeline, nullptr);
    if (layout) vkDestroyPipelineLayout(ctx.device, layout, nullptr);
    if (setLayout) vkDestroyDescriptorSetLayout(ctx.device, setLayout, nullptr);
    pipeline = VK_NULL_HANDLE;
    layout = VK_NULL_HANDLE;
    setLayout = VK_NULL_HANDLE;
}

bool ReflectionProbes::create(const VulkanContext& ctx) {
    // --- GPU resources (empty; count stays 0 until load()) ---------------------
    const uint32_t layers = kMaxProbes * 6;
    if (createImage(ctx, kBakeSize, kBakeSize, kHdrFormat,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, prefilterImage_,
                    prefilterMemory_, kPrefilterMips, layers,
                    VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) != VK_SUCCESS ||
        createImage(ctx, kIrradianceSize, kIrradianceSize, kHdrFormat,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, irradianceImage_,
                    irradianceMemory_, 1, layers,
                    VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) != VK_SUCCESS)
        return false;
    prefilterView_ = createImageView(ctx, prefilterImage_, kHdrFormat, VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                     kPrefilterMips, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY, layers);
    irradianceView_ = createImageView(ctx, irradianceImage_, kHdrFormat, VK_IMAGE_ASPECT_COLOR_BIT,
                                      0, 1, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY, layers);
    if (!prefilterView_ || !irradianceView_) return false;
    sampler_ = createSampler(ctx, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.f,
                             static_cast<float>(kPrefilterMips - 1));
    if (!sampler_) return false;

    // ProbeUBO, persistently mapped; count 0 until load().
    if (createBuffer(ctx, sizeof(ProbeUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     ubo_, uboMemory_) != VK_SUCCESS)
        return false;
    vmaMapMemory(ctx.allocator, uboMemory_, &uboMapped_);
    ProbeUBO zero = {};
    zero.info[1] = static_cast<float>(kPrefilterMips - 1);
    zero.info[2] = kBlendDistance;
    std::memcpy(uboMapped_, &zero, sizeof(zero));

    // Both arrays start SHADER_READ_ONLY (contents undefined but never read:
    // the shaders only sample probes < count).  load() re-transitions the
    // layer groups it writes.
    submitOneShot(ctx, [&](VkCommandBuffer cmd) {
        imageBarrier(cmd, prefilterImage_, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_NONE,
                     VK_ACCESS_2_NONE, sync::kSampleStages, sync::kSampled,
                     VK_IMAGE_ASPECT_COLOR_BIT, 0, kPrefilterMips, 0, layers);
        imageBarrier(cmd, irradianceImage_, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_NONE,
                     VK_ACCESS_2_NONE, sync::kSampleStages, sync::kSampled,
                     VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers);
    });

    // --- prefilter pipelines (same compute stages as the global IBL) -----------
    VkShaderModule modIrradiance = VK_NULL_HANDLE, modPrefilter = VK_NULL_HANDLE;
    if (!loadShaderModule(ctx, "ibl_irradiance.comp.spv", modIrradiance) ||
        !loadShaderModule(ctx, "ibl_prefilter.comp.spv", modPrefilter))
        return false;

    auto createStage = [&](VkShaderModule module, uint32_t pushSize, ComputeStage& out) -> bool {
        VkDescriptorSetLayoutBinding bindings[2] = {};
        bindings[0].binding = 0; // source cubemap (combined sampler)
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].binding = 1; // destination (mip, layer group) storage image
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo setCi = {};
        setCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        setCi.bindingCount = 2;
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
    };
    const bool ok = createStage(modIrradiance, 4, irradianceStage_) &&
                    createStage(modPrefilter, 8, prefilterStage_);
    vkDestroyShaderModule(ctx.device, modIrradiance, nullptr);
    vkDestroyShaderModule(ctx.device, modPrefilter, nullptr);
    if (!ok) return false;

    VkDescriptorPoolSize poolSizes[2] = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = 2 * kMaxProbes * (1 + kPrefilterMips);
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = 2 * kMaxProbes * (1 + kPrefilterMips);
    VkDescriptorPoolCreateInfo poolCi = {};
    poolCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCi.maxSets = kMaxProbes * (1 + kPrefilterMips);
    poolCi.poolSizeCount = 2;
    poolCi.pPoolSizes = poolSizes;
    return vkCreateDescriptorPool(ctx.device, &poolCi, nullptr, &pool_) == VK_SUCCESS;
}

bool ReflectionProbes::load(const VulkanContext& ctx, const std::vector<ReflectionProbe>& defs,
                            const std::string& filePath) {
    count_ = 0;
    // The irradiance/prefilter descriptor sets allocated below are transient
    // (only used by this call's one-shot dispatches, which have completed by
    // the time we return) but are never freed individually — without a pool
    // reset, repeated load() calls (GUI scene switches, deferred rebuilds)
    // would eventually exhaust the fixed-size pool and silently fail allocSet.
    if (pool_) vkResetDescriptorPool(ctx.device, pool_, 0);
    if (defs.empty()) return true;
    std::vector<uint16_t> rgba;
    if (!loadProbeFile(filePath.c_str(), defs, kBakeSize, rgba)) return true;

    const uint32_t count = static_cast<uint32_t>(defs.size());
    const VkDeviceSize faceBytes = static_cast<VkDeviceSize>(kBakeSize) * kBakeSize * 8;
    const VkDeviceSize cubeBytes = faceBytes * 6;

    for (uint32_t i = 0; i < count; ++i) {
        // Upload the raw baked cube into a transient single-probe source image
        // (1 mip; the irradiance/prefilter shaders clamp their computed LOD to
        // the view's mip range, so no source mip chain is built — documented
        // bake-quality simplification).
        VkImage srcImage = VK_NULL_HANDLE;
        VmaAllocation srcMemory = VK_NULL_HANDLE;
        if (createImage(ctx, kBakeSize, kBakeSize, kHdrFormat,
                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, srcImage,
                        srcMemory, 1, 6, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) != VK_SUCCESS)
            return false;
        VkBuffer staging = VK_NULL_HANDLE;
        VmaAllocation stagingMemory = VK_NULL_HANDLE;
        if (createBuffer(ctx, cubeBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         staging, stagingMemory) != VK_SUCCESS)
            return false;
        void* mapped = nullptr;
        vmaMapMemory(ctx.allocator, stagingMemory, &mapped);
        std::memcpy(mapped, rgba.data() + static_cast<size_t>(i) * 6 * kBakeSize * kBakeSize * 4,
                    static_cast<size_t>(cubeBytes));
        vmaUnmapMemory(ctx.allocator, stagingMemory);
        submitOneShot(ctx, [&](VkCommandBuffer cmd) {
            imageBarrier(cmd, srcImage, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE,
                         VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_COPY_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6);
            VkBufferImageCopy region = {};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 6};
            region.imageExtent = {kBakeSize, kBakeSize, 1};
            vkCmdCopyBufferToImage(cmd, staging, srcImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                   &region);
            imageBarrier(cmd, srcImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                         6);
        });
        vmaDestroyBuffer(ctx.allocator, staging, stagingMemory);
        VkImageView srcView = createImageView(ctx, srcImage, kHdrFormat, VK_IMAGE_ASPECT_COLOR_BIT,
                                              0, 1, VK_IMAGE_VIEW_TYPE_CUBE, 6);
        if (!srcView) return false;

        const uint32_t baseLayer = i * 6;
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
        auto writeSet = [&](VkDescriptorSet set, VkImageView dstView) {
            VkDescriptorImageInfo src = {};
            src.sampler = sampler_;
            src.imageView = srcView;
            src.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkDescriptorImageInfo dst = {};
            dst.imageView = dstView;
            dst.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkWriteDescriptorSet writes[2] = {};
            for (uint32_t k = 0; k < 2; ++k) {
                writes[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[k].dstSet = set;
                writes[k].dstBinding = k;
                writes[k].descriptorCount = 1;
                writes[k].descriptorType = k == 0 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                  : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                writes[k].pImageInfo = k == 0 ? &src : &dst;
            }
            vkUpdateDescriptorSets(ctx.device, 2, writes, 0, nullptr);
        };

        // The whole layer group flips SHADER_READ_ONLY -> GENERAL once, each
        // mip is written, then the group goes back to SHADER_READ_ONLY.
        submitOneShot(ctx, [&](VkCommandBuffer cmd) {
            imageBarrier(cmd, irradianceImage_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, baseLayer, 6);
            imageBarrier(cmd, prefilterImage_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_IMAGE_ASPECT_COLOR_BIT, 0, kPrefilterMips, baseLayer, 6);
        });

        // Diffuse irradiance (cosine convolution of the baked cube).
        {
            VkImageView dst =
                createLayerMipView(ctx, irradianceImage_, kHdrFormat, 0, baseLayer, 6);
            VkDescriptorSet set = allocSet(irradianceStage_.setLayout);
            writeSet(set, dst);
            submitOneShot(ctx, [&](VkCommandBuffer cmd) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, irradianceStage_.pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        irradianceStage_.layout, 0, 1, &set, 0, nullptr);
                const float maxLod = 0.f; // 1-mip source (see the upload comment)
                vkCmdPushConstants(cmd, irradianceStage_.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4,
                                   &maxLod);
                vkCmdDispatch(cmd, kIrradianceSize / 8, kIrradianceSize / 8, 6);
            });
            vkDestroyImageView(ctx.device, dst, nullptr);
        }

        // Prefiltered specular: one dispatch per roughness mip, same GGX
        // importance sampling as the global environment prefilter.
        for (uint32_t mip = 0; mip < kPrefilterMips; ++mip) {
            VkImageView dst =
                createLayerMipView(ctx, prefilterImage_, kHdrFormat, mip, baseLayer, 6);
            VkDescriptorSet set = allocSet(prefilterStage_.setLayout);
            writeSet(set, dst);
            const float roughness =
                static_cast<float>(mip) / static_cast<float>(kPrefilterMips - 1);
            const float push[2] = {roughness, static_cast<float>(kBakeSize)};
            const uint32_t size = kBakeSize >> mip;
            submitOneShot(ctx, [&](VkCommandBuffer cmd) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prefilterStage_.pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        prefilterStage_.layout, 0, 1, &set, 0, nullptr);
                vkCmdPushConstants(cmd, prefilterStage_.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8,
                                   push);
                vkCmdDispatch(cmd, (size + 7) / 8, (size + 7) / 8, 6);
            });
            vkDestroyImageView(ctx.device, dst, nullptr);
        }

        submitOneShot(ctx, [&](VkCommandBuffer cmd) {
            imageBarrier(cmd, irradianceImage_, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         sync::kSampleStages, sync::kSampled, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1,
                         baseLayer, 6);
            imageBarrier(cmd, prefilterImage_, VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         sync::kSampleStages, sync::kSampled, VK_IMAGE_ASPECT_COLOR_BIT, 0,
                         kPrefilterMips, baseLayer, 6);
        });
        vkDestroyImageView(ctx.device, srcView, nullptr);
        vmaDestroyImage(ctx.allocator, srcImage, srcMemory);
    }

    // UBO: boxes + capture positions; count arms the shader path.
    ProbeUBO ubo = {};
    ubo.info[0] = static_cast<float>(count);
    ubo.info[1] = static_cast<float>(kPrefilterMips - 1);
    ubo.info[2] = kBlendDistance;
    for (uint32_t i = 0; i < count; ++i) {
        ubo.boxMin[i][0] = defs[i].boxMin.x;
        ubo.boxMin[i][1] = defs[i].boxMin.y;
        ubo.boxMin[i][2] = defs[i].boxMin.z;
        ubo.boxMax[i][0] = defs[i].boxMax.x;
        ubo.boxMax[i][1] = defs[i].boxMax.y;
        ubo.boxMax[i][2] = defs[i].boxMax.z;
        ubo.position[i][0] = defs[i].position.x;
        ubo.position[i][1] = defs[i].position.y;
        ubo.position[i][2] = defs[i].position.z;
    }
    std::memcpy(uboMapped_, &ubo, sizeof(ubo));
    count_ = count;
    std::printf("probes: loaded %u baked reflection probe(s) from %s\n", count,
                filePath.c_str());
    return true;
}

void ReflectionProbes::destroy(const VulkanContext& ctx) {
    if (!ctx.device) return;
    if (prefilterView_) vkDestroyImageView(ctx.device, prefilterView_, nullptr);
    if (irradianceView_) vkDestroyImageView(ctx.device, irradianceView_, nullptr);
    if (prefilterImage_) vmaDestroyImage(ctx.allocator, prefilterImage_, prefilterMemory_);
    if (irradianceImage_) vmaDestroyImage(ctx.allocator, irradianceImage_, irradianceMemory_);
    if (sampler_) vkDestroySampler(ctx.device, sampler_, nullptr);
    if (ubo_) {
        if (uboMapped_) vmaUnmapMemory(ctx.allocator, uboMemory_);
        vmaDestroyBuffer(ctx.allocator, ubo_, uboMemory_);
    }
    if (pool_) vkDestroyDescriptorPool(ctx.device, pool_, nullptr);
    irradianceStage_.destroy(ctx);
    prefilterStage_.destroy(ctx);
    *this = ReflectionProbes{};
}

} // namespace sr
