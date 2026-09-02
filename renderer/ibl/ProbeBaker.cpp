#include "renderer/ibl/ProbeBaker.h"

#include "renderer/core/VkUtil.h"
#include "renderer/core/Vma.h"
#include "renderer/ibl/Probes.h"
#include "renderer/scene/Camera.h"

#include <cstdio>
#include <cstring>

namespace sr {

namespace {

// Bake-local render target (own layout tracking; the hosts' ImageResource
// structs are private to them).  Layout starts UNDEFINED and is kept current
// by the transition lambda in bakeReflectionProbes.
struct RtImage {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    void destroy(const VulkanContext& ctx) {
        if (view) vkDestroyImageView(ctx.device, view, nullptr);
        if (image) vmaDestroyImage(ctx.allocator, image, memory);
        view = VK_NULL_HANDLE;
        image = VK_NULL_HANDLE;
    }
};

// Same tiny helpers the three front ends carry in their anonymous namespaces.
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

bool bakeReflectionProbes(const ProbeBakeParams& p) {
    const VulkanContext& ctx = *p.ctx;
    DeferredCore& deferred = *p.deferred;
    Scene& scene = *p.scene;
    const std::vector<ReflectionProbe>& defs = scene.probes;
    if (defs.empty()) {
        std::fprintf(stderr, "bake-probes: scene has no probe placements\n");
        return true;
    }
    constexpr uint32_t S = ReflectionProbes::kBakeSize;
    std::fprintf(stderr, "bake-probes: %zu probe(s), %ux%u per face -> %s\n", defs.size(), S, S,
                 p.outPath.c_str());

    // Cube face capture orientations (see the header comment; readback flips X).
    static const Vec3 kFaceDir[6] = {{1.f, 0.f, 0.f}, {-1.f, 0.f, 0.f}, {0.f, 1.f, 0.f},
                                     {0.f, -1.f, 0.f}, {0.f, 0.f, 1.f},  {0.f, 0.f, -1.f}};
    static const Vec3 kFaceUp[6] = {{0.f, 1.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, -1.f},
                                    {0.f, 0.f, 1.f}, {0.f, 1.f, 0.f}, {0.f, 1.f, 0.f}};

    // --- bake-local targets (kBakeSize^2 GBuffer + HDR + white AO stand-in) --
    // Phase 6b: the GT GBuffer pipeline now has five attachments, so the bake
    // keeps a (discarded) motion target too.
    RtImage albedo, normal, material, emissive, depth, hdr, ao, motion;
    auto createRT = [&](RtImage& rt, VkFormat format, VkImageUsageFlags usage,
                        VkImageAspectFlags aspect) {
        if (createImage(ctx, S, S, format, usage, rt.image, rt.memory) != VK_SUCCESS)
            return false;
        rt.view = createImageView(ctx, rt.image, format, aspect);
        return rt.view != VK_NULL_HANDLE;
    };
    const VkImageUsageFlags gbUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    const bool targetsOk =
        createRT(albedo, deferred::kAlbedoFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT) &&
        createRT(normal, deferred::kNormalFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT) &&
        createRT(material, deferred::kMaterialFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT) &&
        createRT(emissive, deferred::kEmissiveFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT) &&
        createRT(motion, deferred::kMotionFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT) &&
        createRT(depth, deferred::kDepthFormat,
                 VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                 VK_IMAGE_ASPECT_DEPTH_BIT) &&
        createRT(hdr, deferred::kHdrColorFormat,
                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                 VK_IMAGE_ASPECT_COLOR_BIT) &&
        createRT(ao, VK_FORMAT_R16_SFLOAT,
                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                 VK_IMAGE_ASPECT_COLOR_BIT);
    if (!targetsOk) return false;
    // White AO (the bake skips the GTAO chain; AO=1 keeps the env term intact).
    submitOneShot(ctx, [&](VkCommandBuffer cmd) {
        imageBarrier(cmd, ao.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_CLEAR_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT);
        const VkClearColorValue white = {{1.f, 1.f, 1.f, 1.f}};
        const VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, ao.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &white, 1,
                             &range);
        imageBarrier(cmd, ao.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_CLEAR_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT, sync::kFragment, sync::kSampled);
    });
    ao.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // --- bake-local descriptors (lighting set + cluster grid at kBakeSize^2) --
    VkDescriptorPoolSize sizes[3] = {};
    sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = 14; // lighting set image bindings
    sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[1].descriptorCount = 2; // lighting UBO + probe UBO
    sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[2].descriptorCount = 2 + 2 * kClusterSlots; // lighting SSBOs + cluster assign sets
    VkDescriptorPoolCreateInfo poolCi = {};
    poolCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCi.maxSets = 1 + kClusterSlots;
    poolCi.poolSizeCount = 3;
    poolCi.pPoolSizes = sizes;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(ctx.device, &poolCi, nullptr, &pool) != VK_SUCCESS) return false;

    ClusterGrid cluster;
    bool bakeOk = deferred.createClusterGrid(ctx, S, S, cluster) &&
                  deferred.writeClusterGridSets(ctx, pool, cluster);

    VkBuffer lightingUbo = VK_NULL_HANDLE;
    VmaAllocation lightingUboMemory = VK_NULL_HANDLE;
    void* lightingUboMapped = nullptr;
    if (bakeOk &&
        createBuffer(ctx, sizeof(LightingUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     lightingUbo, lightingUboMemory) == VK_SUCCESS) {
        vmaMapMemory(ctx.allocator, lightingUboMemory, &lightingUboMapped);
    } else {
        bakeOk = false;
    }

    VkDescriptorSet lightingSet = VK_NULL_HANDLE;
    if (bakeOk) {
        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = pool;
        alloc.descriptorSetCount = 1;
        const VkDescriptorSetLayout layout = deferred.lightingSetLayout();
        alloc.pSetLayouts = &layout;
        bakeOk = vkAllocateDescriptorSets(ctx.device, &alloc, &lightingSet) == VK_SUCCESS;
    }
    if (bakeOk) {
        deferred.writeLightingSet(ctx, lightingSet, lightingUbo, albedo.view, normal.view,
                                  material.view, emissive.view, depth.view, ao.view,
                                  p.shadowArrayView, p.spotAtlasView, cluster.lightsBuffer[0],
                                  cluster.gridBuffer[0]);
    }

    // Readback staging for one face (RGBA16F).
    const VkDeviceSize faceBytes = static_cast<VkDeviceSize>(S) * S * 8;
    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation stagingMemory = VK_NULL_HANDLE;
    void* stagingMapped = nullptr;
    if (bakeOk &&
        createBuffer(ctx, faceBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging, stagingMemory) == VK_SUCCESS) {
        vmaMapMemory(ctx.allocator, stagingMemory, &stagingMapped);
    } else {
        bakeOk = false;
    }

    std::vector<uint16_t> all; // probe-major, face-major RGBA16F
    if (bakeOk)
        all.resize(static_cast<size_t>(defs.size()) * 6 * S * S * 4);

    auto transition = [&](VkCommandBuffer cmd, RtImage& rt, VkImageLayout target,
                          VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                          VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
                          VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
        imageBarrier(cmd, rt.image, rt.layout, target, srcStage, srcAccess, dstStage, dstAccess,
                     aspect);
        rt.layout = target;
    };

    for (uint32_t pi = 0; bakeOk && pi < defs.size(); ++pi) {
        for (uint32_t face = 0; face < 6; ++face) {
            Camera cam;
            cam.position = defs[pi].position;
            cam.forward = kFaceDir[face];
            cam.up = kFaceUp[face];
            cam.fovY = 1.5707963f; // 90 deg, square face
            cam.nearPlane = p.nearPlane;
            cam.farPlane = p.farPlane;
            const Mat4 view = cam.view();
            const Mat4 proj = cam.proj(1.f);
            const Mat4 viewProj = Mat4::multiply(proj, view);

            scene.advanceToFrame(0);
            scene.updateLodSelection(cam.position, cam.fovY, lodEnabledByDefault());
            SceneUBO subo;
            deferred.fillSceneUBO(subo, scene, cam, view, proj, proj, Mat4::identity(), S, S,
                                  0.f, 0.f, false);
            std::memcpy(p.sceneUboMapped, &subo, sizeof(subo));
            LightingUBO lubo;
            deferred.fillLightingUBO(lubo, scene, cam, viewProj, Mat4::inverse(viewProj),
                                     nullptr, nullptr, p.iblIntensity);
            std::memcpy(lightingUboMapped, &lubo, sizeof(lubo));
            deferred.fillClusterLights(cluster.lightsMapped[0],
                                       DeferredCore::effectiveLights(scene, nullptr));

            submitOneShot(ctx, [&](VkCommandBuffer cmd) {
                transition(cmd, albedo, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           sync::kSampleStages, sync::kSampled, sync::kColorAttach,
                           sync::kColorWrite);
                transition(cmd, normal, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           sync::kSampleStages, sync::kSampled, sync::kColorAttach,
                           sync::kColorWrite);
                transition(cmd, material, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           sync::kSampleStages, sync::kSampled, sync::kColorAttach,
                           sync::kColorWrite);
                transition(cmd, emissive, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           sync::kSampleStages, sync::kSampled, sync::kColorAttach,
                           sync::kColorWrite);
                transition(cmd, depth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                           sync::kSampleStages, sync::kSampled, sync::kDepthTests,
                           sync::kDepthReadWrite, VK_IMAGE_ASPECT_DEPTH_BIT);
                transition(cmd, motion, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           sync::kSampleStages, sync::kSampled, sync::kColorAttach,
                           sync::kColorWrite);
                {
                    // The bake shares the indirect GBuffer path (no cull pass;
                    // every candidate visible).  Slot 0 staging is safe here:
                    // each face renders in its own one-shot submit.
                    const uint32_t candidates = deferred.buildInstanceList(
                        scene, viewProj, p.instances->capacity, p.cullInstCpu->data(),
                        p.cullCmdCpu->data(), *p.cullRuns);
                    if (candidates > 0) {
                        std::memcpy(p.instances->stagingMapped[0], p.cullInstCpu->data(),
                                    static_cast<size_t>(candidates) * sizeof(GpuInstance));
                        std::memcpy(p.cull->cmdStagingMapped[0], p.cullCmdCpu->data(),
                                    static_cast<size_t>(candidates) *
                                        sizeof(VkDrawIndexedIndirectCommand));
                    }
                    deferred.recordInstanceUpload(cmd, 0, *p.instances, candidates);
                    deferred.recordCommandUpload(cmd, 0, *p.cull, candidates, /*culled=*/false);
                    VkRenderingAttachmentInfo colors[5] = {
                        makeColorAttachment(albedo.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                            VK_ATTACHMENT_LOAD_OP_CLEAR),
                        makeColorAttachment(normal.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                            VK_ATTACHMENT_LOAD_OP_CLEAR, 0.f, 0.f, 1.f),
                        makeColorAttachment(material.view,
                                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                            VK_ATTACHMENT_LOAD_OP_CLEAR),
                        makeColorAttachment(emissive.view,
                                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                            VK_ATTACHMENT_LOAD_OP_CLEAR),
                        makeColorAttachment(motion.view,
                                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                            VK_ATTACHMENT_LOAD_OP_CLEAR)};
                    VkRenderingAttachmentInfo depthAtt = makeDepthAttachment(
                        depth.view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        VK_ATTACHMENT_LOAD_OP_CLEAR);
                    beginRendering(cmd, S, S, 5, colors, &depthAtt);
                    // GT pipeline variant (five attachments incl. motion; the
                    // bake discards motion — prevViewProj is identity anyway).
                    deferred.recordGBufferDraws(cmd, scene, /*gtPass=*/true, p.sceneSet,
                                                p.textureSet, p.materialStride, S, S, viewProj,
                                                *p.cull, p.cullRuns->data(),
                                                static_cast<uint32_t>(p.cullRuns->size()));
                    vkCmdEndRendering(cmd);
                }
                transition(cmd, motion, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled);
                transition(cmd, albedo, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled);
                transition(cmd, normal, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled);
                transition(cmd, material, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled);
                transition(cmd, emissive, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled);
                transition(cmd, depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           sync::kDepthTests, sync::kDepthWrite, sync::kFragment, sync::kSampled,
                           VK_IMAGE_ASPECT_DEPTH_BIT);
                transition(cmd, hdr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kCopy,
                           sync::kTransferRead, sync::kColorAttach, sync::kColorWrite);
                deferred.recordLightingPass(cmd, lightingSet, cluster, 0, view, proj, hdr.view,
                                            S, S);
                transition(cmd, hdr, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sync::kColorAttach,
                           sync::kColorWrite, sync::kCopy, sync::kTransferRead);
                VkBufferImageCopy region = {};
                region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.imageExtent = {S, S, 1};
                vkCmdCopyImageToBuffer(cmd, hdr.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       staging, 1, &region);
            });

            // Readback + horizontal flip (raster face -> Vulkan cube face).
            const uint16_t* srcPx = static_cast<const uint16_t*>(stagingMapped);
            uint16_t* dstPx =
                all.data() + (static_cast<size_t>(pi) * 6 + face) * S * S * 4;
            for (uint32_t y = 0; y < S; ++y)
                for (uint32_t x = 0; x < S; ++x)
                    for (uint32_t c = 0; c < 4; ++c)
                        dstPx[(static_cast<size_t>(y) * S + x) * 4 + c] =
                            srcPx[(static_cast<size_t>(y) * S + (S - 1 - x)) * 4 + c];
        }
        std::fprintf(stderr, "bake-probes: probe %u done\n", pi);
    }

    if (bakeOk && !saveProbeFile(p.outPath.c_str(), defs, S, all)) {
        std::fprintf(stderr, "bake-probes: failed to write %s\n", p.outPath.c_str());
        bakeOk = false;
    }

    // --- cleanup ---------------------------------------------------------------
    if (staging) {
        if (stagingMapped) vmaUnmapMemory(ctx.allocator, stagingMemory);
        vmaDestroyBuffer(ctx.allocator, staging, stagingMemory);
    }
    deferred.destroyClusterGrid(ctx, cluster);
    if (lightingUbo) {
        if (lightingUboMapped) vmaUnmapMemory(ctx.allocator, lightingUboMemory);
        vmaDestroyBuffer(ctx.allocator, lightingUbo, lightingUboMemory);
    }
    if (pool) vkDestroyDescriptorPool(ctx.device, pool, nullptr);
    albedo.destroy(ctx);
    normal.destroy(ctx);
    material.destroy(ctx);
    emissive.destroy(ctx);
    motion.destroy(ctx);
    depth.destroy(ctx);
    hdr.destroy(ctx);
    ao.destroy(ctx);
    return bakeOk;
}

} // namespace sr
