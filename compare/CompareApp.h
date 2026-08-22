#pragma once
// ============================================================================
// CompareApp — split-screen quality comparison mode.
//
// Renders one low-resolution deferred GBuffer pass (Halton jittered) shared by
// every registered upscaler, one native-resolution ground-truth pass, and
// composites the results side by side: the leftmost column is the native 100%
// render ("NATIVE (GT)"), the remaining columns are the upscaler outputs.
// Shading is the shared DeferredCore PBR + IBL pipeline (renderer/deferred),
// identical to the viewer renderer.  Each column shows a live PSNR/SSIM
// overlay computed on the GPU (two-pass compute reduction, read back every N
// frames).
// ============================================================================
#include "renderer/ColorGrading.h"
#include "renderer/core/Swapchain.h"
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

struct CompareOptions {
    uint32_t displayWidth = 1920;
    uint32_t displayHeight = 1080;
    float renderScale = 0.5f;
    bool vsync = true;
    int frames = -1;                    // -1 = interactive
    std::vector<std::string> upscalerNames; // empty = all registered & available
    std::string screenshotPath;
    std::string cameraPath;             // empty = orbit (automation) / free-fly
    std::string scenePath;              // empty = procedural
    int metricInterval = 15;            // frames between GPU metric readbacks
    bool gtSsaa = false;                // render GT at 2x and downsample to 1080p
    bool shadows = true;                // CSM sun shadows (all paths share one map)
    bool shadowDebug = false;           // tint pixels per shadow cascade
    bool bloom = true;                  // HDR bloom before upscale
    // Terminal lens-effects chain in the column compose (chromatic
    // aberration / vignette / film grain; same strengths as the viewer
    // present pass).  CLI: --no-lens-fx.  Lens dirt is viewer-only (the
    // compare paths have no HDR bloom chain for it to modulate).
    bool lensFx = true;
    bool ssr = true;                    // opaque screen-space reflections (CLI: --no-ssr)
    // Screen-space contact shadows for the CSM sun (CLI: --no-contact-shadows);
    // needs shadows on (rides the CSM sun selection).
    bool contactShadows = true;
    bool volFog = true;                 // froxel volumetric fog (CLI: --no-volfog)
    // Motion blur + depth of field (Phase 6b), HDR domain on every path (LR
    // at render res before upscale, GT at native res, GT-SSAA at 2x before
    // the downsample) with identical algorithm + parameters; CLI:
    // --no-motion-blur / --no-dof.  On by default: the GT column runs the
    // same post chain as the algorithm columns, so the metrics stay fair —
    // MB/DOF change the pixel content of EVERY column's input by design.
    bool motionBlur = true;
    bool dof = true;
    float zoom = 1.f;                   // compare-view zoom (1..16)
    float zoomCenterU = 0.5f;           // zoom window center, normalized source UV
    float zoomCenterV = 0.5f;
    std::string envMapPath; // equirect HDR for IBL/skybox; empty = sky atmosphere (default)
    float exposure = 1.f;                       // manual display exposure (ACES input);
                                                // also the seed value for auto exposure
    // Histogram-based auto exposure (UE4 AutoExposure style; see DeferredCore's
    // AutoExposure).  On by default; CLI --exposure switches to manual mode.
    // Each path (LR / GT) auto-exposes from its own HDR target, so compare
    // columns legitimately use different exposures (independent pipelines).
    bool autoExposure = true;
    float exposureMinEV = -8.f;                 // auto-exposure EV clamp range
    float exposureMaxEV = 8.f;
};

class CompareApp {
public:
    bool init(const CompareOptions& opts);
    void run();
    void shutdown();

private:
    static constexpr uint32_t kFramesInFlight = 2;
    static constexpr uint32_t kMaxAlgos = 4;            // extra columns are truncated
    static constexpr uint32_t kMaxColumns = 1 + kMaxAlgos; // GT + algorithms
    static constexpr uint32_t kMetricFloats = 8;        // per-algorithm reduce record
    static constexpr uint32_t kTextCharsPerColumn = 96; // 4 lines x 24 chars

    struct ImageResource {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        uint32_t width = 0;
        uint32_t height = 0;
        void destroy(const VulkanContext& ctx);
    };

    struct FrameResources {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkBuffer uboGb = VK_NULL_HANDLE;       // jittered scene UBO (GBuffer pass)
        VmaAllocation uboGbMemory = VK_NULL_HANDLE;
        void* uboGbMapped = nullptr;
        VkBuffer uboGbSpatial = VK_NULL_HANDLE; // unjittered LR scene UBO (spatial plugins)
        VmaAllocation uboGbSpatialMemory = VK_NULL_HANDLE;
        void* uboGbSpatialMapped = nullptr;
        VkBuffer uboGt = VK_NULL_HANDLE;       // un-jittered scene UBO (GT pass)
        VmaAllocation uboGtMemory = VK_NULL_HANDLE;
        void* uboGtMapped = nullptr;
        VkBuffer lightingUboGb = VK_NULL_HANDLE; // lighting UBO (jittered invViewProj)
        VmaAllocation lightingUboGbMemory = VK_NULL_HANDLE;
        void* lightingUboGbMapped = nullptr;
        VkBuffer lightingUboGbSpatial = VK_NULL_HANDLE; // lighting UBO (unjittered LR)
        VmaAllocation lightingUboGbSpatialMemory = VK_NULL_HANDLE;
        void* lightingUboGbSpatialMapped = nullptr;
        VkBuffer lightingUboGt = VK_NULL_HANDLE; // lighting UBO (un-jittered invViewProj)
        VmaAllocation lightingUboGtMemory = VK_NULL_HANDLE;
        void* lightingUboGtMapped = nullptr;
        VkDescriptorSet sceneSetGb = VK_NULL_HANDLE;
        VkDescriptorSet sceneSetGbSpatial = VK_NULL_HANDLE;
        VkDescriptorSet sceneSetGt = VK_NULL_HANDLE;
        VkDescriptorSet lightingSetGb = VK_NULL_HANDLE;
        VkDescriptorSet lightingSetGbSpatial = VK_NULL_HANDLE;
        VkDescriptorSet lightingSetGt = VK_NULL_HANDLE;
        VkDescriptorSet lightingSetSsaa = VK_NULL_HANDLE; // 2x GT GBuffer (gtSsaa only)
        // IBL + per-path SSAO texture for the transparency pass.
        VkDescriptorSet transparentSetGb = VK_NULL_HANDLE;
        VkDescriptorSet transparentSetGbSpatial = VK_NULL_HANDLE;
        VkDescriptorSet transparentSetGt = VK_NULL_HANDLE;
        VkDescriptorSet transparentSetSsaa = VK_NULL_HANDLE; // gtSsaa only
        // Opaque-SSR compute sets (bind the same per-path lighting UBOs).
        VkDescriptorSet ssrSetGb = VK_NULL_HANDLE;
        VkDescriptorSet ssrSetGbSpatial = VK_NULL_HANDLE;
        VkDescriptorSet ssrSetGt = VK_NULL_HANDLE;
        VkDescriptorSet ssrSetSsaa = VK_NULL_HANDLE; // gtSsaa only
    };

    struct AlgoColumn {
        std::string id;
        std::unique_ptr<IUpscaler> upscaler;
        ImageResource output; // display-resolution RGBA16F
        VkImageLayout outputLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkBuffer blocksBuffer = VK_NULL_HANDLE; // per-block metric records
        VmaAllocation blocksMemory = VK_NULL_HANDLE;
        VkDescriptorSet metricSet = VK_NULL_HANDLE;
        VkDescriptorSet composeSet = VK_NULL_HANDLE;
        float psnr = 0.f;
        float ssim = 0.f;
        bool hasMetric = false;
    };

    CompareOptions opts_;
    Window window_;
    VulkanContext ctx_;
    Swapchain swapchain_;
    Scene scene_;
    Camera camera_;
    CameraPath path_;
    std::vector<AlgoColumn> algos_;

    uint32_t renderWidth_ = 0;
    uint32_t renderHeight_ = 0;
    uint32_t blocksPerRow_ = 0;
    uint32_t blockCount_ = 0;

    Mat4 prevViewProj_ = Mat4::identity();
    float jitterX_ = 0.f;
    float jitterY_ = 0.f;
    float prevJitterX_ = 0.f;
    float prevJitterY_ = 0.f;
    bool hasTransparency_ = false; // any alphaMode BLEND material in the scene

    // Diagnostic switches (env): SR_NO_JITTER=1 zeroes the Halton sub-pixel
    // jitter while keeping the low-res path (isolates jitter from resolution);
    // SR_METRIC_STDOUT=1 prints the per-harvest PSNR/SSIM time series.
    bool diagNoJitter_ = false;
    bool diagMetricStdout_ = false;
    uint32_t metricHarvestCount_ = 0;

    ImageResource gbColor_;
    ImageResource gbColorSpatial_; // unjittered LR HDR copy for spatial upscalers
    ImageResource gbMotion_;
    ImageResource gbReactive_; // translucent coverage mask (reactive/TC input)
    ImageResource gbDepth_;
    // Deferred GBuffer attachments of the low-res input pass.
    ImageResource gbAlbedo_;
    ImageResource gbNormal_;
    ImageResource gbMaterial_;
    ImageResource gbEmissive_;
    ImageResource gtColor_;
    ImageResource gtDepth_;
    // GT-path motion RT (Phase 6b): the GT GBuffer writes per-object motion so
    // the GT column's motion blur matches the LR path.
    ImageResource gtMotion_;
    // Deferred GBuffer attachments of the native-res GT pass.
    ImageResource gtAlbedo_;
    ImageResource gtNormal_;
    ImageResource gtMaterial_;
    ImageResource gtEmissive_;
    ImageResource gtSsaaColor_; // 2x GT render target (gtSsaa only)
    ImageResource gtSsaaDepth_;
    ImageResource gtSsaaMotion_; // 2x GT motion RT (Phase 6b, gtSsaa only)
    // Deferred GBuffer attachments of the 2x GT pass (gtSsaa only).
    ImageResource gtSsaaAlbedo_;
    ImageResource gtSsaaNormal_;
    ImageResource gtSsaaMaterial_;
    ImageResource gtSsaaEmissive_;
    // GTAO working target (RG16F: AO + view Z) and filtered R16F, path res.
    ImageResource gbAoRaw_;
    ImageResource gbAo_;
    ImageResource gtAoRaw_;
    ImageResource gtAo_;
    ImageResource gtSsaaAoRaw_; // gtSsaa only
    ImageResource gtSsaaAo_;
    // HDR color mip chains (lit opaque color, box-filtered) for
    // roughness-aware SSR; same GENERAL-for-life resource model as the
    // depth pyramids.  Deliberately separate from the Phase 6a bloom pyramid
    // (box-average reflection LOD vs thresholded extract — see DeferredCore).
    ColorPyramid gbColorPyramid_;
    ColorPyramid gtColorPyramid_;
    ColorPyramid gtSsaaColorPyramid_; // gtSsaa only
    // Clustered shading grids (per-path resolution, per-slot buffers).
    ClusterGrid gbCluster_;
    ClusterGrid gtCluster_;
    ClusterGrid gtSsaaCluster_; // gtSsaa only
    // Hi-Z depth pyramids for the SSR march (LR / GT / GT-SSAA paths); general
    // DeferredCore resource, later reused by GTAO/contact shadows/culling.
    DepthPyramid gbPyramid_;
    DepthPyramid gtPyramid_;
    DepthPyramid gtSsaaPyramid_; // gtSsaa only
    // GTAO view-Z depth chains (XeGTAO DepthMIPFilter) + temporal accumulation
    // ping-pong state, one per path.  Per-path previous-frame view-projection
    // (jittered for LR) and frame counters drive the temporal pass; a zero
    // counter resets (bypasses) the history.  In mixed temporal+spatial
    // column mode the LR lighting runs twice per frame and the LR counter
    // advances twice — history stays consistent (each record reprojects from
    // the previous record's exact view-projection).
    DepthPyramid gbPyramidAo_;
    DepthPyramid gtPyramidAo_;
    DepthPyramid gtSsaaPyramidAo_; // gtSsaa only
    AoHistory gbAoHist_;
    AoHistory gtAoHist_;
    AoHistory gtSsaaAoHist_; // gtSsaa only
    Mat4 prevAoViewProjGb_ = Mat4::identity();
    Mat4 prevAoViewProjGt_ = Mat4::identity();
    Mat4 prevAoViewProjSsaa_ = Mat4::identity();
    uint32_t aoFramesGb_ = 0;
    uint32_t aoFramesGt_ = 0;
    uint32_t aoFramesSsaa_ = 0;
    // Opaque SSR temporal state (Phase 2d), one per path: full-res trace
    // target (rgb = composite delta, a = view |z|) + RGBA16F ping-pong
    // history; per-path previous-frame view-projection and frame counters
    // drive the temporal pass (zero counter = reset).  Same conventions as
    // the GTAO temporal state above, including the mixed-mode LR double run.
    ImageResource gbSsrTrace_;
    ImageResource gtSsrTrace_;
    ImageResource gtSsaaSsrTrace_; // gtSsaa only
    SsrHistory gbSsrHist_;
    SsrHistory gtSsrHist_;
    SsrHistory gtSsaaSsrHist_; // gtSsaa only
    Mat4 prevSsrViewProjGb_ = Mat4::identity();
    Mat4 prevSsrViewProjGt_ = Mat4::identity();
    Mat4 prevSsrViewProjSsaa_ = Mat4::identity();
    uint32_t ssrFramesGb_ = 0;
    uint32_t ssrFramesGt_ = 0;
    uint32_t ssrFramesSsaa_ = 0;
    // Froxel volumetric fog (Phase 5a), one volume set per path.  Temporal
    // history reprojects with the un-jittered view-projection on all paths
    // (fog must not swim with the TAA jitter).  Unlike AO/SSR the accumulate
    // step runs once per frame per path even in mixed mode (fogAccumFrame*
    // guards), while the composite runs inside every lighting record.
    VolFogVolume gbFog_;
    VolFogVolume gtFog_;
    VolFogVolume gtSsaaFog_; // gtSsaa only
    Mat4 prevFogViewProjGb_ = Mat4::identity();
    Mat4 prevFogViewProjGt_ = Mat4::identity();
    Mat4 prevFogViewProjSsaa_ = Mat4::identity();
    uint32_t fogFramesGb_ = 0;
    uint32_t fogFramesGt_ = 0;
    uint32_t fogFramesSsaa_ = 0;
    uint32_t fogAccumFrameGb_ = ~0u;
    uint32_t fogAccumFrameGt_ = ~0u;
    uint32_t fogAccumFrameSsaa_ = ~0u;
    VolFogParams fogParams_;
    bool volFogActive_ = false;
    // Motion-blur + DOF working targets (Phase 6b), one per path (SSAA only
    // when gtSsaa); same host-owned GENERAL-for-life model as the pyramids.
    PostFxTargets gbPostFx_;
    PostFxTargets gtPostFx_;
    PostFxTargets gtSsaaPostFx_; // gtSsaa only
    ImageResource composeImage_; // RGBA8, tonemapped columns + overlay
    ImageResource fontAtlas_;
    // Log-domain grading LUT (Phase 6c): procedural identity — compare columns
    // share the neutral grading set (compare output stays SDR).
    GradingLutGpu gradingLut_;
    ColorGrading grading_;

    // Shared deferred pipeline (shaders/layouts/pipelines/samplers + IBL maps).
    DeferredCore deferred_;
    // CSM shadow map (fixed 2048^2 x 4, resolution-independent): created in
    // init, shared by the GB/GT/SSAA lighting paths; shadowsActive_ = false
    // degrades to no shadows (bindings stay unwritten, sampling stays off).
    ShadowTargets shadow_;
    bool shadowsActive_ = false;
    // Spot light shadow atlas (Phase 4b, fixed 4096^2, shared by all paths).
    ShadowAtlas spotAtlas_;
    bool spotAtlasActive_ = false;
    float iblIntensity_ = 1.f; // from lightingPresetForScene

    VkDescriptorSetLayout composeSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout copySetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout metricSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout composePipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout copyPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout metricPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline composePipeline_ = VK_NULL_HANDLE;
    VkPipeline copyPipeline_ = VK_NULL_HANDLE;
    VkPipeline downsamplePipeline_ = VK_NULL_HANDLE; // 2x GT -> 1x GT (SSAA)
    VkPipeline metricBlocksPipeline_ = VK_NULL_HANDLE;
    VkPipeline metricReducePipeline_ = VK_NULL_HANDLE;
    VkShaderModule fullscreenVert_ = VK_NULL_HANDLE;
    VkShaderModule composeFrag_ = VK_NULL_HANDLE;
    VkShaderModule copyFrag_ = VK_NULL_HANDLE;
    VkShaderModule metricBlocksComp_ = VK_NULL_HANDLE;
    VkShaderModule metricReduceComp_ = VK_NULL_HANDLE;

    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    FrameResources frames_[kFramesInFlight] = {};
    std::vector<VkSemaphore> renderFinished_;
    VkDescriptorSet textureSet_ = VK_NULL_HANDLE;
    VkDescriptorSet copySet_ = VK_NULL_HANDLE;
    VkDescriptorSet gtComposeSet_ = VK_NULL_HANDLE;
    VkDescriptorSet gtDownsampleSet_ = VK_NULL_HANDLE; // 2x GT source (SSAA)
    // SSAO descriptor sets (static: the referenced textures never change).
    // The blur sets live inside AoHistory (one per ping-pong buffer).
    VkDescriptorSet ssaoSetGb_ = VK_NULL_HANDLE;
    VkDescriptorSet ssaoSetGt_ = VK_NULL_HANDLE;
    VkDescriptorSet ssaoSetSsaa_ = VK_NULL_HANDLE;     // gtSsaa only
    VkSampler linearSampler_ = VK_NULL_HANDLE;
    VkSampler fontSampler_ = VK_NULL_HANDLE;

    VkImageLayout gbColorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbColorSpatialLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbMotionLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbReactiveLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbDepthLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbAlbedoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbNormalLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbMaterialLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbEmissiveLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtColorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtDepthLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtMotionLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtAlbedoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtNormalLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtMaterialLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtEmissiveLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaColorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaDepthLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaMotionLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaAlbedoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaNormalLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaMaterialLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaEmissiveLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbAoRawLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbAoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtAoRawLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtAoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaAoRawLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaAoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbSsrTraceLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsrTraceLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaSsrTraceLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout composeLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    VkBuffer materialUbo_ = VK_NULL_HANDLE;
    VmaAllocation materialUboMemory_ = VK_NULL_HANDLE;
    uint32_t materialStride_ = 0;

    VkBuffer textUbo_ = VK_NULL_HANDLE; // packed ASCII overlay text (all columns)
    VmaAllocation textUboMemory_ = VK_NULL_HANDLE;
    void* textUboMapped_ = nullptr;

    VkBuffer metricResultBuf_ = VK_NULL_HANDLE; // kMaxAlgos * kMetricFloats floats
    VmaAllocation metricResultMemory_ = VK_NULL_HANDLE;
    VkBuffer metricStaging_[kFramesInFlight] = {};
    VmaAllocation metricStagingMemory_[kFramesInFlight] = {};
    void* metricStagingMapped_[kFramesInFlight] = {};
    bool metricPending_[kFramesInFlight] = {};

    VkBuffer screenshotStaging_ = VK_NULL_HANDLE;
    VmaAllocation screenshotStagingMemory_ = VK_NULL_HANDLE;
    void* screenshotMapped_ = nullptr;

    // Auto exposure: the LR path (gbColor_) feeds the upscaler preExposure and
    // the algorithm columns; the GT path (gtSsaaColor_ when gtSsaa, else
    // gtColor_) feeds the GT column + metric reference.  Two independent
    // solvers — each column is its own render pipeline (engine behaviour), so
    // per-column exposures may differ.  Harvested values lag the GPU by
    // kFramesInFlight frames; see ExposureChannel.
    ExposureChannel lrExposure_;
    ExposureChannel gtExposure_;

    bool initAlgorithms();
    bool createRenderTargets();
    bool createShaders();
    bool createDescriptors();
    bool createPipelines();
    bool createSyncResources();
    bool recreateRenderFinishedSemaphores();
    bool createFontAtlas();
    bool createMetricResources();
    bool createScreenshotStaging();
    // Auto-exposure channels + sets (needs the descriptor pool; called from
    // createDescriptors once the pool exists).
    bool createAutoExposureResources();
    // Per-path display exposure: harvested auto value, or the manual
    // --exposure override when auto exposure is off.
    float lrExposure() const {
        return opts_.autoExposure ? lrExposure_.value : opts_.exposure;
    }
    float gtExposure() const {
        return opts_.autoExposure ? gtExposure_.value : opts_.exposure;
    }
    void updateSceneUBO(void* mapped, bool jitter, uint32_t renderW, uint32_t renderH,
                        const Mat4& view, const Mat4& proj, const Mat4& projJittered,
                        const Mat4& prevViewProj);
    void updateLightingUBO(void* mapped, const Mat4& viewProj, const ShadowFrame* shadow,
                           const std::vector<Light>* overrideLights);
    void updateClusterLights(uint32_t frameIndex, const std::vector<Light>& lights);
    void updateCamera(uint32_t frameIndex, float dt);
    void recordFrame(uint32_t frameIndex, uint32_t swapchainIndex);
    void captureScreenshotIntoStaging(VkCommandBuffer cmd);
    void saveScreenshot(const std::string& path);
    void harvestMetrics(uint32_t slot);
    void refreshOverlayText();
    bool loadShader(const char* name, VkShaderModule& out);
    // Visible source region (px) for a column of colW x colH: aspect-preserving
    // center-crop at zoom 1; a zoomed window centered on (centerU, centerV)
    // (normalized source UV) when zoom > 1.  out = {offX, offY, sizeX, sizeY}.
    static void computeViewRegion(uint32_t srcW, uint32_t srcH, uint32_t colW, uint32_t colH,
                                  float zoom, float centerU, float centerV, float out[4]);
};

} // namespace sr
