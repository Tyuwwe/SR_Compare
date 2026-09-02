#pragma once
// ============================================================================
// ProbeBaker — offline reflection-probe bake (Phase 4c-2), shared by the CLI
// (--bake-probes, Renderer::bakeProbes) and the GUI's viewer-tab bake button.
//
// For each registry-placed probe the scene is rendered from the probe position
// into the six cube faces at ReflectionProbes::kBakeSize^2, reusing the real
// GBuffer + deferred lighting pipeline (GT variant: no motion/reactive
// attachments) at 90 deg FOV — so the bake sees the same punctual lights,
// materials, emissive and the global-env skybox the viewer shades with.
// Documented quality simplifications: no shadow maps, no SSAO (constant 1),
// no SSR/transparency, and the source cube keeps a single mip (the prefilter
// shaders clamp their LOD).  The bake is an offline action: it never runs
// during bench, and the GUI refuses it while a scene load is in flight.
//
// Face conventions: each face renders with a lookAt along kFaceDir (up from
// kFaceUp); the raster image is then mirrored horizontally vs the Vulkan
// cube-face layout the IBL shaders sample (cubeDir() in ibl_*.comp), so the
// readback flips X before writing the face.
// ============================================================================
#include "renderer/core/Vk.h"
#include "renderer/deferred/DeferredCore.h"
#include "renderer/scene/Scene.h"

#include <cstdint>
#include <string>
#include <vector>

namespace sr {

// Everything the bake borrows from its host (Renderer or GuiApp), resolved
// before the call.  The bake reuses the host's frame-slot-0 resources (scene
// UBO, instance/command staging), so the caller must guarantee no frame is in
// flight on the GPU (CLI: called before the frame loop starts; GUI:
// vkDeviceWaitIdle first).  All pointers must stay valid for the call.
struct ProbeBakeParams {
    const VulkanContext* ctx = nullptr;
    DeferredCore* deferred = nullptr;
    Scene* scene = nullptr; // probes/lights read; LOD selection + frame 0 mutate
    void* sceneUboMapped = nullptr;             // mapped SceneUBO destination, slot 0
    VkDescriptorSet sceneSet = VK_NULL_HANDLE;  // GT-variant scene set, slot 0
    VkDescriptorSet textureSet = VK_NULL_HANDLE;
    uint32_t materialStride = 0;
    InstanceBuffer* instances = nullptr;
    CullChannel* cull = nullptr; // any channel; only slot-0 staging is reused
    std::vector<GpuInstance>* cullInstCpu = nullptr;  // build scratch (capacity-sized)
    std::vector<VkDrawIndexedIndirectCommand>* cullCmdCpu = nullptr;
    std::vector<CullDrawRun>* cullRuns = nullptr;
    // CSM array / spot atlas views for the bake-local lighting set.  The bake
    // fills the lighting UBO with a null ShadowFrame (shadowParams stay off),
    // so the views only keep the descriptor bindings valid; VK_NULL_HANDLE
    // follows the writeLightingSet convention.
    VkImageView shadowArrayView = VK_NULL_HANDLE;
    VkImageView spotAtlasView = VK_NULL_HANDLE;
    float iblIntensity = 1.f;
    float nearPlane = 0.1f;
    float farPlane = 1000.f;
    std::string outPath; // probeFilePathForScene(scene path)
};

// Bakes scene->probes to outPath (see the file header).  Returns true with no
// work done when the scene has no probe placements; false on failure (stderr
// carries the details, "bake-probes: ..." prefix).
bool bakeReflectionProbes(const ProbeBakeParams& p);

} // namespace sr
