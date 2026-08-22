#pragma once
// ============================================================================
// Renderer — owns the window, Vulkan context, swapchain, scene, camera and the
// frame loop.  Produces the upscaler inputs (HDR color + depth + motion) at
// render scale (with Halton jitter), runs the IUpscaler, and presents either
// the upscaled image or the native-resolution ground truth.
// ============================================================================
#include "renderer/ColorGrading.h"
#include "renderer/core/Swapchain.h"
#include "renderer/core/TimestampQuery.h"
#include "renderer/core/Vk.h"
#include "renderer/core/VulkanContext.h"
#include "renderer/core/Window.h"
#include "renderer/deferred/DeferredCore.h"
#include "renderer/scene/Camera.h"
#include "renderer/scene/CameraPath.h"
#include "renderer/scene/Scene.h"
#include "upscalers/IUpscaler.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sr {

struct RendererOptions {
    uint32_t displayWidth = 1920;
    uint32_t displayHeight = 1080;
    float renderScale = 0.5f;
    bool vsync = true;
    int frames = -1;           // -1 = interactive
    std::string upscalerName = "taa";  // registered plugin name; "none" = native ground truth
    std::string screenshotPath;
    std::string frameTimesPath;  // non-empty: dump per-frame GPU timings + VRAM to CSV
    std::string cameraPath;    // empty = orbit (automation) / free-fly (interactive)
    std::string scenePath;     // empty = procedural
    // Equirect HDR environment map for IBL + skybox; empty = procedural sky
    // atmosphere (default) driven by the scene preset's sun direction, unless
    // the preset names an envFile.  An unreadable file falls back to a
    // procedural gradient (constant-ambient look).
    std::string envMapPath;
    // Debug/verification overrides for the preset sun angles (sky atmosphere
    // + key light); negative = use the scene preset.  CLI: --sun-elev/--sun-az.
    float sunElevationDeg = -1.f;
    float sunAzimuthDeg = -1.f;
    bool shadows = true;      // CSM sun shadows (CLI: --no-shadows)
    bool shadowDebug = false; // cascade tint overlay (CLI: --shadow-debug)
    float exposure = 1.f;     // manual display exposure (ACES input multiplier);
                              // also the seed value for auto exposure
    // Histogram-based auto exposure (UE4 AutoExposure style; see DeferredCore's
    // AutoExposure).  On by default; CLI --exposure switches to manual mode.
    bool autoExposure = true;
    float exposureMinEV = -8.f; // auto-exposure EV clamp range
    float exposureMaxEV = 8.f;
    bool bloom = true;        // HDR bloom pyramid before upscale (CLI: --no-bloom)
    // Terminal lens-effects chain at present (chromatic aberration / lens dirt
    // in HDR, vignette + film grain in display domain; CLI: --no-lens-fx).
    // Strengths are the shared DeferredCore defaults (kLens*Strength).
    bool lensFx = true;
    bool ssr = true;          // opaque screen-space reflections (CLI: --no-ssr)
    // Screen-space contact shadows for the CSM sun (CLI: --no-contact-shadows);
    // needs shadows on (rides the CSM sun selection).
    bool contactShadows = true;
    // Froxel volumetric fog (Phase 5a, CLI: --no-volfog); the scene lighting
    // preset carries the media parameters (VolFogParams::enabled gates too).
    bool volFog = true;
    // Motion blur (Phase 6b, McGuire 2012 tile-max gather) in the HDR domain,
    // after bloom and before upscale/present, applied to the LR and GT paths
    // with identical algorithm + parameters (CLI: --no-motion-blur).  On by
    // default everywhere, including bench: the GT blurs identically, so
    // compare/bench stay fair (every column's input pixels change by design).
    bool motionBlur = true;
    // Depth of field (Phase 6b, UE4 scatter-as-gather CoC; CLI: --no-dof).
    // Default ON for bench/automation (--frames set), OFF for interactive
    // free-fly (screen-centre autofocus gets in the way while flying);
    // runViewer applies that rule.  The GUI has a checkbox for interactive use.
    bool dof = true;
    // Offline reflection-probe baking (CLI: --bake-probes): renders each
    // probe's 6 cube faces and writes the .probes file, then exits without
    // entering the frame loop.  Not part of bench.
    bool bakeProbes = false;
    // Log-domain color grading (Phase 6c, ACES input; see grading.glsl).
    // Sentinel semantics per field: <= 0 = unset, falling back to the scene
    // lighting preset's grading override, then the neutral defaults.
    ColorGrading grading{0.f, 0.f, 0.f, 0.f};
    // Optional .cube 3D LUT (17^3/33^3), applied in the log domain; empty =
    // procedural identity LUT (bit-neutral output with default grading).
    std::string lutPath;
    // HDR swapchain output (CLI: --hdr): probe HDR10 (PQ) then scRGB, SDR
    // fallback with a stderr note.  Compare/bench stay SDR (viewer-only).
    bool hdr = false;
    // HDR10 nits calibration: scene linear 1.0 x exposure maps to this many
    // nits before the 10000-nit PQ ceiling (BT.2408 graphics white default).
    float hdrPaperWhiteNits = 203.f;
};

class Renderer {
public:
    bool init(const RendererOptions& opts);
    void run();
    void shutdown();

    // Bakes scene_.probes (registry placements) to probeFilePathForScene().
    // Call after init(); the renderer must be shut down without run().
    bool bakeProbes();

private:
    static constexpr uint32_t kFramesInFlight = 2;

    struct ImageResource {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        uint32_t width = 0;
        uint32_t height = 0;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        void destroy(const VulkanContext& ctx);
    };

    struct FrameResources {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkBuffer ubo = VK_NULL_HANDLE;
        VmaAllocation uboMemory = VK_NULL_HANDLE;
        void* uboMapped = nullptr;
        VkBuffer lightingUbo = VK_NULL_HANDLE;
        VmaAllocation lightingUboMemory = VK_NULL_HANDLE;
        void* lightingUboMapped = nullptr;
        VkDescriptorSet sceneSet = VK_NULL_HANDLE;
        VkDescriptorSet lightingSetGb = VK_NULL_HANDLE;
        VkDescriptorSet lightingSetGt = VK_NULL_HANDLE;
        VkDescriptorSet transparentSetGb = VK_NULL_HANDLE; // IBL + LR AO texture
        VkDescriptorSet transparentSetGt = VK_NULL_HANDLE; // IBL + GT AO texture
        VkDescriptorSet ssrSetGb = VK_NULL_HANDLE; // opaque SSR, LR path
        VkDescriptorSet ssrSetGt = VK_NULL_HANDLE; // opaque SSR, GT path
    };

    RendererOptions opts_;
    Window window_;
    VulkanContext ctx_;
    Swapchain swapchain_;
    Scene scene_;
    Camera camera_;
    CameraPath path_;
    std::unique_ptr<IUpscaler> upscaler_;

    uint32_t renderWidth_ = 0;
    uint32_t renderHeight_ = 0;

    Mat4 prevViewProj_ = Mat4::identity();
    float jitterX_ = 0.f;
    float jitterY_ = 0.f;
    float prevJitterX_ = 0.f;
    float prevJitterY_ = 0.f;
    bool hasTransparency_ = false; // any alphaMode BLEND material in the scene
    // Diagnostic env switch: SR_NO_JITTER=1 zeroes the Halton sub-pixel jitter
    // while keeping the low-res path (isolates jitter from resolution).
    bool diagNoJitter_ = false;
    // Wall-clock delta time (seconds) fed to the upscaler.  Interactive mode
    // updates it every frame; bench mode keeps the 1/60 default for
    // reproducible, machine-independent results.
    float deltaTime_ = 1.0f / 60.0f;

    ImageResource gbColor_;
    ImageResource gbMotion_;
    ImageResource gbDepth_;
    ImageResource gbReactive_; // R16F translucent coverage mask (render res)
    ImageResource gbAlbedo_;
    ImageResource gbNormal_;
    ImageResource gbMaterial_;
    ImageResource gbEmissive_;
    ImageResource gtDepth_;
    ImageResource gtMotion_;   // GT-path motion RT (Phase 6b; RG16F, display res)
    ImageResource gtAlbedo_;
    ImageResource gtNormal_;
    ImageResource gtMaterial_;
    ImageResource gtEmissive_;
    ImageResource finalImage_;
    // GTAO working target (RG16F: AO + view Z) and filtered R16F, GBuffer res.
    ImageResource gbAoRaw_;
    ImageResource gbAo_;
    ImageResource gtAoRaw_;
    ImageResource gtAo_;
    // HDR bloom pyramids (thresholded 5-level chains; Phase 6a), one per path.
    // GENERAL-for-life, host-owned — same resource model as the color
    // pyramids below.  The accumulated mip 0 also feeds the present pass's
    // lens-dirt term.
    BloomPyramid gbBloom_;
    BloomPyramid gtBloom_;
    // Motion-blur + DOF working targets (Phase 6b), one per path; same
    // host-owned GENERAL-for-life resource model as the bloom pyramids.
    PostFxTargets gbPostFx_;
    PostFxTargets gtPostFx_;
    // Procedural lens-dirt mask (R8, radial blobs; generated at init — no
    // external asset).  Sampled by present.frag.
    ImageResource lensDirt_;
    // Log-domain grading LUT (Phase 6c): .cube file or procedural identity,
    // uploaded as an RGBA16F 3D image; the CPU copy feeds the screenshot path.
    GradingLutGpu gradingLutGpu_;
    ColorLut gradingLutCpu_;
    ColorGrading grading_; // resolved: CLI -> scene preset -> neutral defaults
    // HDR color mip chains (lit opaque color, box-filtered) for
    // roughness-aware glass SSR; same GENERAL-for-life resource model as the
    // depth pyramids.  Deliberately separate from the bloom pyramids (see
    // ColorPyramid in DeferredCore.h).
    ColorPyramid gbColorPyramid_;
    ColorPyramid gtColorPyramid_;
    // Hi-Z depth pyramids for the SSR march (LR / GT); general DeferredCore
    // resource, later reused by GTAO/contact shadows/occlusion culling.
    DepthPyramid gbPyramid_;
    DepthPyramid gtPyramid_;
    // GTAO view-Z depth chains (XeGTAO DepthMIPFilter) + temporal accumulation
    // ping-pong state, one per path.  Per-path previous-frame view-projection
    // (jittered for LR) and frame counters drive the temporal pass; a zero
    // counter resets (bypasses) the history.
    DepthPyramid gbPyramidAo_;
    DepthPyramid gtPyramidAo_;
    AoHistory gbAoHist_;
    AoHistory gtAoHist_;
    Mat4 prevAoViewProjGb_ = Mat4::identity();
    Mat4 prevAoViewProjGt_ = Mat4::identity();
    uint32_t aoFramesGb_ = 0;
    uint32_t aoFramesGt_ = 0;
    // Opaque SSR temporal state (Phase 2d), one per path: full-res trace
    // target (rgb = composite delta, a = view |z|) + RGBA16F ping-pong
    // history.  Per-path previous-frame view-projection (jittered for LR) and
    // frame counters drive the temporal pass; a zero counter resets the
    // history.  Same conventions as the GTAO temporal state above.
    ImageResource gbSsrTrace_;
    ImageResource gtSsrTrace_;
    SsrHistory gbSsrHist_;
    SsrHistory gtSsrHist_;
    Mat4 prevSsrViewProjGb_ = Mat4::identity();
    Mat4 prevSsrViewProjGt_ = Mat4::identity();
    uint32_t ssrFramesGb_ = 0;
    uint32_t ssrFramesGt_ = 0;

    // Clustered shading state, one per path (grid resolution differs):
    // per-slot full-lights SSBO + per-cluster light list SSBO (DeferredCore).
    ClusterGrid gbCluster_;
    ClusterGrid gtCluster_;

    // Froxel volumetric fog (Phase 5a), one volume set per path: the grid
    // scales with the path resolution like the cluster grid.  The temporal
    // filter uses the UN-JITTERED previous view-projection per path; a zero
    // frame counter resets (bypasses) the history — same convention as the
    // GTAO/SSR temporal state above.
    VolFogVolume gbFog_;
    VolFogVolume gtFog_;
    Mat4 prevFogViewProjGb_ = Mat4::identity();
    Mat4 prevFogViewProjGt_ = Mat4::identity();
    uint32_t fogFramesGb_ = 0;
    uint32_t fogFramesGt_ = 0;
    VolFogParams fogParams_;    // scene preset media parameters
    bool volFogActive_ = false; // opts.volFog && fogParams_.enabled && volumes created

    // Shared deferred pipeline (shaders/layouts/pipelines/samplers + IBL maps).
    DeferredCore deferred_;

    // CSM sun shadow targets (fixed 2048^2 x 4, resolution-independent).
    // Created unconditionally (the descriptor binding must stay written);
    // shadowsActive_ is false only when creation failed, and --no-shadows
    // keeps shadowParams.z = 0 so the shaders short-circuit.
    ShadowTargets shadow_;
    bool shadowsActive_ = false;
    // Spot light shadow atlas (Phase 4b, fixed 4096^2, shared by both paths).
    ShadowAtlas spotAtlas_;
    bool spotAtlasActive_ = false;
    float iblIntensity_ = 1.f; // from lightingPresetForScene; not a CLI flag
    // Effective sun angles used for the sky atmosphere (preset + CLI override).
    float sunElevationDeg_ = 65.3f;
    float sunAzimuthDeg_ = 49.4f;

    VkDescriptorSetLayout presentSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout presentPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline presentPipeline_ = VK_NULL_HANDLE;
    VkShaderModule presentFrag_ = VK_NULL_HANDLE;

    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    FrameResources frames_[kFramesInFlight] = {};
    std::vector<VkSemaphore> renderFinished_; // one per swapchain image (present sync)

    VkBuffer materialUbo_ = VK_NULL_HANDLE;
    VmaAllocation materialUboMemory_ = VK_NULL_HANDLE;
    uint32_t materialStride_ = 0;
    VkDescriptorSet textureSet_ = VK_NULL_HANDLE;
    VkDescriptorSet presentSet_ = VK_NULL_HANDLE;
    // SSAO descriptor sets (static: the referenced textures never change).
    // The blur sets live inside AoHistory (one per ping-pong buffer); the
    // bloom pyramid sets live inside BloomPyramid.
    VkDescriptorSet ssaoSetGb_ = VK_NULL_HANDLE;
    VkDescriptorSet ssaoSetGt_ = VK_NULL_HANDLE;

    // Auto exposure: one channel bound to the active path's lit HDR target
    // (gbColor_ for the LR path, finalImage_ for native GT).  The harvested
    // value lags the GPU by kFramesInFlight frames — see ExposureChannel.
    ExposureChannel exposureChannel_;

    VkBuffer screenshotStaging_ = VK_NULL_HANDLE;
    VmaAllocation screenshotStagingMemory_ = VK_NULL_HANDLE;
    void* screenshotMapped_ = nullptr;
    VkDeviceSize screenshotSize_ = 0;

    TimestampQuery timestamps_;
    std::vector<TimestampQuery::Timings> frameTimes_;  // per-frame, when frameTimesPath set

    bool createRenderTargets();
    // Procedural 512x512 R8 lens-dirt mask (deterministic radial blobs),
    // uploaded to SHADER_READ_ONLY; sampled by present.frag.
    bool createLensDirtTexture();
    // Grading LUT (Phase 6c): loads --lut (.cube) or generates the procedural
    // identity LUT, uploads the GPU copy and keeps the CPU mirror for the
    // screenshot path.
    bool createGradingLut();
    bool createShaders();
    bool createSceneDescriptors();
    bool createPipelines();
    bool createSyncResources();
    bool createScreenshotStaging();
    // Auto-exposure buffers + descriptor set (needs the descriptor pool, so
    // runs after createSceneDescriptors).
    bool createAutoExposureResources();
    // Display exposure for the current frame: harvested auto value, or the
    // manual --exposure override when auto exposure is off.
    float displayExposure() const {
        return opts_.autoExposure ? exposureChannel_.value : opts_.exposure;
    }
    bool recreateSwapchain(uint32_t width, uint32_t height, bool vsync);
    void updateSceneUBO(uint32_t frameIndex, bool jitter, uint32_t renderW, uint32_t renderH,
                        const Mat4& view, const Mat4& proj, const Mat4& projJittered,
                        const Mat4& prevViewProj);
    void updateLightingUBO(uint32_t frameIndex, const Mat4& viewProj, const Mat4& invViewProj,
                           const ShadowFrame* shadow, const std::vector<Light>* overrideLights);
    void updateCamera(uint32_t frameIndex, float dt);
    void applyCameraKeyframe(uint32_t frameIndex);
    void recordFrame(uint32_t frameIndex, uint32_t swapchainIndex);
    void captureScreenshotIntoStaging(VkCommandBuffer cmd);
    void saveScreenshot(const std::string& path);
    bool loadShader(const char* name, VkShaderModule& out);
};

} // namespace sr
