#pragma once
// ============================================================================
// PlanarReflection — per-frame mirrored-scene render for planar mirror panes
// (the Bistro café storefront, Material::mirror).
//
// Screen-space reflections cannot show what a mirror really shows: the back
// faces of objects hugging the glass (the hanging lanterns) and everything
// off-screen (the building across the street behind the camera).  The
// classic answer for planar mirrors is to render the scene again from the
// camera mirrored across the pane's plane and let the pane sample that image
// at the screen projection of its own surface point — exact for a planar
// reflector, no depth-buffer guesswork, off-screen content included.
//
// Per frame the host picks ONE mirror plane (Scene::mirrorPlanes, the
// nearest one in the frustum), and this pass renders the opaque scene from
// the mirrored camera into a private GBuffer + HDR target at the path's
// resolution with the shared deferred pipeline: same lights, IBL, probes and
// sun/spot shadow maps as the main view (the cascades are the main camera's,
// so far reflected content past their reach is unshadowed), white AO, no
// fog, no transparency, no occlusion cull (CPU frustum cull only).  Geometry
// behind the mirror plane is discarded in the GBuffer fragment shaders via
// SceneUBO::clipPlane (the mirrored camera sits behind the pane and would
// otherwise see the shop interior in front of the street).
//
// The transparency pass then reads the HDR image + depth (transparent set
// bindings 12/13): mirror fragments on the active plane project themselves
// with reflViewProj (SceneUBO) and take the reflected radiance, weighted by
// the pane's Fresnel; the reflected depth also gives the virtual image point
// for the planar-mirror motion vectors.  Panes on other planes keep the SSR
// fallback.
//
// Resource model: the pass owns its images (layouts tracked internally, like
// ProbeBaker), a private descriptor pool, per-slot scene/lighting UBOs and
// sets, a cluster grid, and its own instance/indirect buffers (the candidate
// list is mirrored-frustum culled, so it cannot share the main view's).
// ============================================================================
#include "renderer/core/Vk.h"
#include "renderer/deferred/DeferredCore.h"
#include "renderer/math/Math.h"
#include "renderer/scene/Camera.h"
#include "renderer/scene/Scene.h"

#include <cstdint>
#include <vector>

namespace sr {

class PlanarReflection {
public:
    // Maximum frame slots the per-slot resources cover (hosts use 2).
    static constexpr uint32_t kMaxSlots = 2;
    // SR_PLANAR=0 opts out of the pass (the panes then keep the SSR path);
    // hosts consult this before creating the resources.
    static bool enabledByEnv();

    // Creates the targets at w x h (the path's scene resolution) and every
    // slot resource.  shadowArray / spotAtlas may be VK_NULL_HANDLE (same
    // writeLightingSet convention as the hosts).  materialUbo is the host's
    // per-material dynamic UBO; skin palettes are bound when the scene has
    // skinned meshes.  Returns false on failure (nothing is left half-built:
    // destroy() is safe afterwards).
    bool create(const VulkanContext& ctx, const DeferredCore& deferred, const Scene& scene,
                uint32_t w, uint32_t h, uint32_t slots, VkBuffer materialUbo,
                VkImageView shadowArray, VkImageView spotAtlas);
    void destroy(const VulkanContext& ctx, const DeferredCore& deferred);
    bool valid() const { return hdrView_ != VK_NULL_HANDLE; }

    // 1) Plane selection (call before the host fills its own SceneUBO): picks
    //    the nearest mirror plane whose bounds intersect the main frustum and
    //    derives the mirrored camera.  Returns false (inactive frame) when no
    //    plane qualifies; the main-view SceneUBO then advertises no reflection.
    bool selectPlane(const Scene& scene, const Camera& camera, float aspect);
    bool active() const { return active_; }

    // Writes the main view's reflection fields into a SceneUBO the host is
    // about to upload (active plane, mirrored view-projection, depth unpack).
    void patchMainSceneUbo(SceneUBO& ubo) const;

    // 2) Per-frame data (after the host built its lighting UBO for the main
    //    view): the mirrored view's scene UBO + a copy of mainLighting with
    //    the camera-dependent members replaced, the cluster light SSBO, and
    //    the mirrored-frustum candidate list.  Mutates nothing in the scene.
    void prepare(uint32_t slot, const DeferredCore& deferred, const Scene& scene,
                 const LightingUBO& mainLighting, const std::vector<Light>& lights);

    // 3) Records GBuffer + lighting of the mirrored view into the command
    //    buffer (internal barriers; the HDR + depth images end up
    //    SHADER_READ_ONLY for the transparency pass).  No-op when inactive
    //    (the previous frame's image stays bound but is never sampled: the
    //    main SceneUBO says inactive).
    void record(VkCommandBuffer cmd, uint32_t slot, const DeferredCore& deferred,
                const Scene& scene, VkDescriptorSet textureSet, uint32_t materialStride);

    VkImageView hdrView() const { return hdrView_; }
    VkImageView depthView() const { return depthView_; }
    const Mat4& viewProj() const { return viewProj_; }

private:
    struct RtImage {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        void destroy(const VulkanContext& ctx);
    };
    void transition(VkCommandBuffer cmd, RtImage& rt, VkImageLayout target,
                    VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                    VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
                    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) const;

    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t slots_ = 0;
    // Mirrored-view GBuffer + HDR (GT pipeline variant: five attachments).
    RtImage albedo_, normal_, material_, emissive_, motion_, depth_, hdr_, ao_;
    VkImageView hdrView_ = VK_NULL_HANDLE;   // = hdr_.view (valid() sentinel)
    VkImageView depthView_ = VK_NULL_HANDLE; // = depth_.view
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    ClusterGrid cluster_;
    InstanceBuffer instances_;
    CullChannel cull_;
    std::vector<GpuInstance> instCpu_;
    std::vector<VkDrawIndexedIndirectCommand> cmdCpu_;
    std::vector<CullDrawRun> runs_;
    uint32_t candidates_ = 0;
    struct Slot {
        VkBuffer sceneUbo = VK_NULL_HANDLE;
        VmaAllocation sceneUboMemory = VK_NULL_HANDLE;
        void* sceneUboMapped = nullptr;
        VkBuffer lightingUbo = VK_NULL_HANDLE;
        VmaAllocation lightingUboMemory = VK_NULL_HANDLE;
        void* lightingUboMapped = nullptr;
        VkDescriptorSet sceneSet = VK_NULL_HANDLE;
        VkDescriptorSet lightingSet = VK_NULL_HANDLE;
    };
    Slot slot_[kMaxSlots];

    // Frame state (selectPlane).
    bool active_ = false;
    Vec3 planeN_{0.f, 0.f, 1.f};
    float planeD_ = 0.f;
    Camera mirrorCam_;
    Mat4 view_ = Mat4::identity();
    Mat4 proj_ = Mat4::identity();
    Mat4 viewProj_ = Mat4::identity();
};

} // namespace sr
