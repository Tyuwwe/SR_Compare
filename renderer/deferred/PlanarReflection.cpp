#include "renderer/deferred/PlanarReflection.h"

#include "renderer/core/VkUtil.h"
#include "renderer/core/Vma.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace sr {

bool PlanarReflection::enabledByEnv() {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996) // std::getenv is portable; _dupenv_s is MSVC-only
#endif
    const char* v = std::getenv("SR_PLANAR");
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    return !v || v[0] != '0';
}

namespace {

VkRenderingAttachmentInfo colorAttachment(VkImageView view, VkAttachmentLoadOp loadOp,
                                          float r = 0.f, float g = 0.f, float b = 0.f) {
    VkRenderingAttachmentInfo a = {};
    a.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    a.imageView = view;
    a.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    a.loadOp = loadOp;
    a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    if (loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) a.clearValue.color = {{r, g, b, 1.f}};
    return a;
}

VkRenderingAttachmentInfo depthAttachment(VkImageView view) {
    VkRenderingAttachmentInfo a = {};
    a.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    a.imageView = view;
    a.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    a.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    a.clearValue.depthStencil = {1.f, 0};
    return a;
}

// Frustum planes of a view-projection (same construction as DeferredCore's
// CPU cull) and the positive-vertex AABB test.
struct FrustumPlanes {
    Vec4 p[6];
};

FrustumPlanes frustumOf(const Mat4& vp) {
    auto row = [&](int r) { return Vec4{vp.m[r], vp.m[4 + r], vp.m[8 + r], vp.m[12 + r]}; };
    const Vec4 r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
    auto add = [](const Vec4& a, const Vec4& b) { return Vec4{a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; };
    auto sub = [](const Vec4& a, const Vec4& b) { return Vec4{a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w}; };
    FrustumPlanes f;
    f.p[0] = add(r3, r0);
    f.p[1] = sub(r3, r0);
    f.p[2] = add(r3, r1);
    f.p[3] = sub(r3, r1);
    f.p[4] = r2;
    f.p[5] = sub(r3, r2);
    return f;
}

bool aabbInFrustum(const FrustumPlanes& f, const Vec3& mn, const Vec3& mx) {
    for (const Vec4& p : f.p) {
        const float x = p.x >= 0.f ? mx.x : mn.x;
        const float y = p.y >= 0.f ? mx.y : mn.y;
        const float z = p.z >= 0.f ? mx.z : mn.z;
        if (p.x * x + p.y * y + p.z * z + p.w < 0.f) return false;
    }
    return true;
}

Vec3 reflectPoint(const Vec3& p, const Vec3& n, float d) {
    return p - n * (2.f * (dot(n, p) + d));
}

Vec3 reflectDir(const Vec3& v, const Vec3& n) { return v - n * (2.f * dot(n, v)); }

} // namespace

void PlanarReflection::RtImage::destroy(const VulkanContext& ctx) {
    if (view) vkDestroyImageView(ctx.device, view, nullptr);
    if (image) vmaDestroyImage(ctx.allocator, image, memory);
    view = VK_NULL_HANDLE;
    image = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
    layout = VK_IMAGE_LAYOUT_UNDEFINED;
}

void PlanarReflection::transition(VkCommandBuffer cmd, RtImage& rt, VkImageLayout target,
                                  VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                                  VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
                                  VkImageAspectFlags aspect) const {
    imageBarrier(cmd, rt.image, rt.layout, target, srcStage, srcAccess, dstStage, dstAccess, aspect);
    rt.layout = target;
}

bool PlanarReflection::create(const VulkanContext& ctx, const DeferredCore& deferred,
                              const Scene& scene, uint32_t w, uint32_t h, uint32_t slots,
                              VkBuffer materialUbo, VkImageView shadowArray,
                              VkImageView spotAtlas) {
    width_ = w;
    height_ = h;
    slots_ = std::min(slots, kMaxSlots);

    auto createRT = [&](RtImage& rt, VkFormat format, VkImageUsageFlags usage,
                        VkImageAspectFlags aspect) {
        if (createImage(ctx, w, h, format, usage, rt.image, rt.memory) != VK_SUCCESS) return false;
        rt.view = createImageView(ctx, rt.image, format, aspect);
        return rt.view != VK_NULL_HANDLE;
    };
    const VkImageUsageFlags gbUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (!createRT(albedo_, deferred::kAlbedoFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT) ||
        !createRT(normal_, deferred::kNormalFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT) ||
        !createRT(material_, deferred::kMaterialFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT) ||
        !createRT(emissive_, deferred::kEmissiveFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT) ||
        !createRT(motion_, deferred::kMotionFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT) ||
        !createRT(depth_, deferred::kDepthFormat,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_DEPTH_BIT) ||
        !createRT(hdr_, deferred::kHdrColorFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT) ||
        !createRT(ao_, VK_FORMAT_R16_SFLOAT,
                  VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;

    // White AO (no GTAO in the mirrored view) and an initial shader-readable
    // layout for the images the transparency pass binds (they are sampled
    // only when a plane is active, but the bound layout must be valid).
    submitOneShot(ctx, [&](VkCommandBuffer cmd) {
        imageBarrier(cmd, ao_.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_CLEAR_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT);
        const VkClearColorValue white = {{1.f, 1.f, 1.f, 1.f}};
        const VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, ao_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &white, 1, &range);
        imageBarrier(cmd, ao_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_CLEAR_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT, sync::kFragment, sync::kSampled);
        imageBarrier(cmd, hdr_.image, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_NONE,
                     VK_ACCESS_2_NONE, sync::kFragment, sync::kSampled);
        imageBarrier(cmd, depth_.image, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_NONE,
                     VK_ACCESS_2_NONE, sync::kFragment, sync::kSampled, VK_IMAGE_ASPECT_DEPTH_BIT);
    });
    ao_.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    hdr_.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    depth_.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Private descriptor pool: per-slot scene + lighting sets, cluster assign.
    VkDescriptorPoolSize sizes[5] = {};
    sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = 14 * slots_;
    sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[1].descriptorCount = 3 * slots_; // scene UBO + lighting UBO + probe UBO
    sizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    sizes[2].descriptorCount = slots_;
    sizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[3].descriptorCount = 4 * slots_ + 2 * kClusterSlots;
    sizes[4].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    sizes[4].descriptorCount = 1; // pool minimum for drivers that reject empty types
    VkDescriptorPoolCreateInfo poolCi = {};
    poolCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCi.maxSets = 2 * slots_ + kClusterSlots;
    poolCi.poolSizeCount = 5;
    poolCi.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(ctx.device, &poolCi, nullptr, &pool_) != VK_SUCCESS) return false;

    if (!deferred.createClusterGrid(ctx, w, h, cluster_) ||
        !deferred.writeClusterGridSets(ctx, pool_, cluster_))
        return false;

    const uint32_t capacity = static_cast<uint32_t>(scene.instances.size());
    if (!deferred.createInstanceBuffer(ctx, capacity, instances_) ||
        !deferred.createCullChannel(ctx, capacity, cull_))
        return false;
    instCpu_.resize(capacity);
    cmdCpu_.resize(capacity);

    for (uint32_t i = 0; i < slots_; ++i) {
        Slot& s = slot_[i];
        if (createBuffer(ctx, sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         s.sceneUbo, s.sceneUboMemory) != VK_SUCCESS)
            return false;
        vmaMapMemory(ctx.allocator, s.sceneUboMemory, &s.sceneUboMapped);
        if (createBuffer(ctx, sizeof(LightingUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         s.lightingUbo, s.lightingUboMemory) != VK_SUCCESS)
            return false;
        vmaMapMemory(ctx.allocator, s.lightingUboMemory, &s.lightingUboMapped);

        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = pool_;
        alloc.descriptorSetCount = 1;
        const VkDescriptorSetLayout sceneLayout = deferred.sceneSetLayout();
        alloc.pSetLayouts = &sceneLayout;
        if (vkAllocateDescriptorSets(ctx.device, &alloc, &s.sceneSet) != VK_SUCCESS) return false;
        const VkDescriptorSetLayout lightingLayout = deferred.lightingSetLayout();
        alloc.pSetLayouts = &lightingLayout;
        if (vkAllocateDescriptorSets(ctx.device, &alloc, &s.lightingSet) != VK_SUCCESS) return false;

        VkDescriptorBufferInfo sceneBuf = {};
        sceneBuf.buffer = s.sceneUbo;
        sceneBuf.range = sizeof(SceneUBO);
        VkDescriptorBufferInfo materialBuf = {};
        materialBuf.buffer = materialUbo;
        materialBuf.range = sizeof(MaterialUBO);
        VkWriteDescriptorSet writes[2] = {};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = s.sceneSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo = &sceneBuf;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = s.sceneSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        writes[1].pBufferInfo = &materialBuf;
        vkUpdateDescriptorSets(ctx.device, 2, writes, 0, nullptr);
        if (instances_.buffer) deferred.writeSceneInstanceBinding(ctx, s.sceneSet, instances_.buffer);
        if (scene.hasSkinnedMeshes())
            deferred.writeSceneSkinBinding(ctx, s.sceneSet, scene.skinPalette(i));

        const uint32_t cs = i % kClusterSlots;
        deferred.writeLightingSet(ctx, s.lightingSet, s.lightingUbo, albedo_.view, normal_.view,
                                  material_.view, emissive_.view, depth_.view, ao_.view,
                                  shadowArray, spotAtlas, cluster_.lightsBuffer[cs],
                                  cluster_.gridBuffer[cs]);
    }

    hdrView_ = hdr_.view;
    depthView_ = depth_.view;
    return true;
}

void PlanarReflection::destroy(const VulkanContext& ctx, const DeferredCore& deferred) {
    for (uint32_t i = 0; i < kMaxSlots; ++i) {
        Slot& s = slot_[i];
        if (s.sceneUbo) {
            if (s.sceneUboMapped) vmaUnmapMemory(ctx.allocator, s.sceneUboMemory);
            vmaDestroyBuffer(ctx.allocator, s.sceneUbo, s.sceneUboMemory);
        }
        if (s.lightingUbo) {
            if (s.lightingUboMapped) vmaUnmapMemory(ctx.allocator, s.lightingUboMemory);
            vmaDestroyBuffer(ctx.allocator, s.lightingUbo, s.lightingUboMemory);
        }
        s = Slot{};
    }
    deferred.destroyCullChannel(ctx, cull_);
    deferred.destroyInstanceBuffer(ctx, instances_);
    deferred.destroyClusterGrid(ctx, cluster_);
    if (pool_) vkDestroyDescriptorPool(ctx.device, pool_, nullptr);
    pool_ = VK_NULL_HANDLE;
    albedo_.destroy(ctx);
    normal_.destroy(ctx);
    material_.destroy(ctx);
    emissive_.destroy(ctx);
    motion_.destroy(ctx);
    depth_.destroy(ctx);
    hdr_.destroy(ctx);
    ao_.destroy(ctx);
    hdrView_ = VK_NULL_HANDLE;
    depthView_ = VK_NULL_HANDLE;
    active_ = false;
}

bool PlanarReflection::selectPlane(const Scene& scene, const Camera& camera, float aspect) {
    active_ = false;
    if (!valid() || scene.mirrorPlanes.empty()) return false;
    const Mat4 mainVp = Mat4::multiply(camera.proj(aspect), camera.view());
    const FrustumPlanes frustum = frustumOf(mainVp);
    float bestDist = 1e30f;
    Vec3 bestN{0.f, 0.f, 1.f};
    float bestD = 0.f;
    for (const Scene::MirrorPlane& mp : scene.mirrorPlanes) {
        Vec3 n = mp.normal;
        float d = mp.d;
        // Orient toward the camera; skip planes the camera sits on.
        const float side = dot(n, camera.position) + d;
        if (side < 0.f) { n = n * -1.f; d = -d; }
        if (std::fabs(side) < 0.05f) continue;
        if (!aabbInFrustum(frustum, mp.aabbMin, mp.aabbMax)) continue;
        // Distance from the camera to the mirror bounds (closest point).
        const Vec3 c{std::clamp(camera.position.x, mp.aabbMin.x, mp.aabbMax.x),
                     std::clamp(camera.position.y, mp.aabbMin.y, mp.aabbMax.y),
                     std::clamp(camera.position.z, mp.aabbMin.z, mp.aabbMax.z)};
        const float dist = length(c - camera.position);
        if (dist < bestDist) {
            bestDist = dist;
            bestN = n;
            bestD = d;
        }
    }
    if (bestDist >= 1e30f) return false;

    planeN_ = bestN;
    planeD_ = bestD;
    mirrorCam_ = camera;
    mirrorCam_.position = reflectPoint(camera.position, bestN, bestD);
    mirrorCam_.forward = normalize(reflectDir(camera.forward, bestN));
    mirrorCam_.up = normalize(reflectDir(camera.up, bestN));
    // A regular (right-handed) camera at the mirrored pose: the image is the
    // scene seen from behind the pane, and the pane samples it by projecting
    // its own surface points with the same matrix, so handedness is moot and
    // no winding/cull flip is needed.
    view_ = mirrorCam_.view();
    proj_ = mirrorCam_.proj(aspect);
    viewProj_ = Mat4::multiply(proj_, view_);
    active_ = true;
    return true;
}

void PlanarReflection::patchMainSceneUbo(SceneUBO& ubo) const {
    if (!active_) {
        ubo.reflParams[0] = ubo.reflParams[1] = 0.f;
        return;
    }
    ubo.clipPlane[0] = planeN_.x;
    ubo.clipPlane[1] = planeN_.y;
    ubo.clipPlane[2] = planeN_.z;
    ubo.clipPlane[3] = planeD_;
    std::memcpy(ubo.reflViewProj, viewProj_.m, sizeof(ubo.reflViewProj));
    ubo.reflParams[0] = 1.f;         // reflection image available
    ubo.reflParams[1] = 0.f;         // no clipping in the main view
    ubo.reflParams[2] = proj_.m[10]; // depth unpack: viewZ = m14 / (depth + m10)
    ubo.reflParams[3] = proj_.m[14];
}

void PlanarReflection::prepare(uint32_t slot, const DeferredCore& deferred, const Scene& scene,
                               const LightingUBO& mainLighting,
                               const std::vector<Light>& lights) {
    if (!active_) return;
    const uint32_t s = slot % slots_;
    SceneUBO subo;
    deferred.fillSceneUBO(subo, scene, mirrorCam_, view_, proj_, proj_, viewProj_, width_, height_,
                          0.f, 0.f, false);
    subo.clipPlane[0] = planeN_.x;
    subo.clipPlane[1] = planeN_.y;
    subo.clipPlane[2] = planeN_.z;
    subo.clipPlane[3] = planeD_;
    subo.reflParams[0] = 0.f;
    subo.reflParams[1] = 1.f; // clip geometry behind the mirror plane
    subo.reflParams[2] = proj_.m[10];
    subo.reflParams[3] = proj_.m[14];
    std::memcpy(slot_[s].sceneUboMapped, &subo, sizeof(subo));

    // The main view's lighting data with the camera-dependent members swapped
    // for the mirrored camera.  Sun/spot shadow maps and cascade splits are
    // the main camera's (see the header).
    LightingUBO lubo = mainLighting;
    const Mat4 inv = Mat4::inverse(viewProj_);
    std::memcpy(lubo.invViewProj, inv.m, sizeof(lubo.invViewProj));
    std::memcpy(lubo.viewProj, viewProj_.m, sizeof(lubo.viewProj));
    lubo.cameraPos[0] = mirrorCam_.position.x;
    lubo.cameraPos[1] = mirrorCam_.position.y;
    lubo.cameraPos[2] = mirrorCam_.position.z;
    lubo.viewForward[0] = mirrorCam_.forward.x;
    lubo.viewForward[1] = mirrorCam_.forward.y;
    lubo.viewForward[2] = mirrorCam_.forward.z;
    lubo.lightCounts[1] = 0.f; // no glass trace in the mirrored view
    std::memcpy(slot_[s].lightingUboMapped, &lubo, sizeof(lubo));

    deferred.fillClusterLights(cluster_.lightsMapped[s % kClusterSlots], lights);

    candidates_ = deferred.buildInstanceList(scene, viewProj_, instances_.capacity, instCpu_.data(),
                                             cmdCpu_.data(), runs_);
    if (candidates_ > 0) {
        const uint32_t cs = s % kCullSlots;
        std::memcpy(instances_.stagingMapped[cs], instCpu_.data(),
                    static_cast<size_t>(candidates_) * sizeof(GpuInstance));
        std::memcpy(cull_.cmdStagingMapped[cs], cmdCpu_.data(),
                    static_cast<size_t>(candidates_) * sizeof(VkDrawIndexedIndirectCommand));
    }
}

void PlanarReflection::record(VkCommandBuffer cmd, uint32_t slot, const DeferredCore& deferred,
                              const Scene& scene, VkDescriptorSet textureSet,
                              uint32_t materialStride) {
    if (!active_) return;
    const uint32_t s = slot % slots_;

    // GBuffer targets: sampled last frame by the lighting pass -> attachments.
    RtImage* colors[5] = {&albedo_, &normal_, &material_, &emissive_, &motion_};
    for (RtImage* rt : colors)
        transition(cmd, *rt, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kSampleStages,
                   sync::kSampled, sync::kColorAttach, sync::kColorWrite);
    transition(cmd, depth_, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, sync::kSampleStages,
               sync::kSampled, sync::kDepthTests, sync::kDepthReadWrite, VK_IMAGE_ASPECT_DEPTH_BIT);

    deferred.recordInstanceUpload(cmd, s, instances_, candidates_);
    deferred.recordCommandUpload(cmd, s, cull_, candidates_, /*culled=*/false);
    VkRenderingAttachmentInfo att[5] = {
        colorAttachment(albedo_.view, VK_ATTACHMENT_LOAD_OP_CLEAR),
        colorAttachment(normal_.view, VK_ATTACHMENT_LOAD_OP_CLEAR, 0.f, 0.f, 1.f),
        colorAttachment(material_.view, VK_ATTACHMENT_LOAD_OP_CLEAR),
        colorAttachment(emissive_.view, VK_ATTACHMENT_LOAD_OP_CLEAR),
        colorAttachment(motion_.view, VK_ATTACHMENT_LOAD_OP_CLEAR)};
    VkRenderingAttachmentInfo depthAtt = depthAttachment(depth_.view);
    VkRenderingInfo ri = {};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea = {{0, 0}, {width_, height_}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 5;
    ri.pColorAttachments = att;
    ri.pDepthAttachment = &depthAtt;
    vkCmdBeginRendering(cmd, &ri);
    deferred.recordGBufferDraws(cmd, scene, /*gtPass=*/true, slot_[s].sceneSet, textureSet,
                                materialStride, width_, height_, viewProj_, cull_, runs_.data(),
                                static_cast<uint32_t>(runs_.size()));
    vkCmdEndRendering(cmd);

    for (RtImage* rt : colors)
        transition(cmd, *rt, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kColorAttach,
                   sync::kColorWrite, sync::kFragment, sync::kSampled);
    transition(cmd, depth_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kDepthTests,
               sync::kDepthWrite, sync::kFragment, sync::kSampled, VK_IMAGE_ASPECT_DEPTH_BIT);
    // HDR: sampled by last frame's transparency pass -> attachment -> sampled.
    transition(cmd, hdr_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kSampleStages,
               sync::kSampled, sync::kColorAttach, sync::kColorWrite);
    deferred.recordLightingPass(cmd, slot_[s].lightingSet, cluster_, s, view_, proj_, hdr_.view,
                                width_, height_);
    transition(cmd, hdr_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kColorAttach,
               sync::kColorWrite, sync::kFragment, sync::kSampled);
}

} // namespace sr
