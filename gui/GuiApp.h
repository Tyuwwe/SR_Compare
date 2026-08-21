#pragma once
// ============================================================================
// GuiApp — Dear ImGui front end for sr_compare (`sr_compare gui`).
//
// Covers the full CLI surface in three tabs:
//   Viewer  — single-upscaler full-screen preview (scene/upscaler/scale/
//             resolution/camera path/screenshot/frame-times CSV).
//   Compare — split-screen GT + up to 4 algorithm columns with the live
//             GPU-reduction PSNR/SSIM overlay (same shaders as compare mode).
//   Bench   — spawns `sr_compare bench` as a child process and shows the
//             resulting CSV in a table.
//
// The render organization mirrors compare/CompareApp: a low-res deferred
// GBuffer (Halton jittered when any temporal plugin is selected) feeds the
// upscalers.  Mixed temporal+spatial sets add a second unjittered LR color
// for FSR1/SGSR1 (raster jitter cannot be undone by resampling).  A
// native-res deferred pass provides the ground truth, and a compose pass
// assembles an offscreen RGBA8 composite that is both presented (copied to
// the swapchain) and used as the screenshot source.  Shading is the shared
// DeferredCore PBR + IBL pipeline (renderer/deferred); the IBL maps are
// built once per process and rebuilt only when the env map changes.
// The ImGui overlay is drawn into the same swapchain dynamic-rendering
// block on top of the composite copy.
//
// Settings that change resolution/scale/scene/algorithm set/env map require
// an "Apply" rebuild: the whole render stack is destroyed and recreated (the
// Win32 window, Vulkan device and ImGui backend persist).  Apply is
// asynchronous: a worker thread loads the new scene and initializes the
// upscalers while the old stack keeps rendering under a progress overlay, and
// the new stack is swapped in at a device-idle safe point (see the
// LoadPhase/LoadResult machinery below).  Only the initial build in init()
// is synchronous.
// ============================================================================
#include "gui/BenchRunner.h"
#include "renderer/core/Swapchain.h"
#include "renderer/core/TimestampQuery.h"
#include "renderer/core/VulkanContext.h"
#include "renderer/core/Window.h"
#include "renderer/deferred/DeferredCore.h"
#include "renderer/scene/Camera.h"
#include "renderer/scene/CameraPath.h"
#include "renderer/scene/Scene.h"
#include "renderer/scene/SceneRegistry.h"

#include "upscalers/IUpscaler.h"
#include "upscalers/IFrameGen.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace sr {

// Optional launch-time configuration for `sr_compare gui` (all fields have UI
// defaults when unset).  Doubles as a headless-ish automation hook for tests:
// --frames N auto-exits, --screenshot saves the composite on the last frame,
// --bench auto-starts a bench run.
struct GuiOptions {
    std::string sceneArg;      // registry alias or glTF path
    std::string upscalerName;  // viewer upscaler; "none"/"native" = GT
    std::string compareList;   // comma-separated: start in Compare tab
    std::string benchList;     // comma-separated: start in Bench tab, auto-run
    float renderScale = 0.f;   // <= 0 keeps the UI default
    uint32_t displayW = 0;     // 0 keeps the UI default (must match a preset)
    uint32_t displayH = 0;
    int frames = -1;           // >= 0: auto-exit after N rendered frames
    std::string screenshotPath;
    float compareZoom = 0.f;   // > 0: preset compare-tab zoom (automation)
    bool compareGtSsaa = false; // preset the compare-tab GT SSAA checkbox
    std::string envMapPath;    // empty = kDefaultEnvMapPath
};

class GuiApp {
public:
    bool init(const GuiOptions& opts);
    void run();
    void shutdown();

private:
    static constexpr uint32_t kFramesInFlight = 2;
    static constexpr uint32_t kMaxAlgos = 4;               // compare columns
    static constexpr uint32_t kMaxColumns = 1 + kMaxAlgos; // GT + algorithms
    static constexpr uint32_t kMetricFloats = 8;           // per-algo reduce record
    static constexpr uint32_t kTextCharsPerColumn = 96;    // 4 lines x 24 chars
    static constexpr uint32_t kMetricInterval = 15;        // frames between readbacks
    static constexpr uint32_t kHistoryLen = 240;           // PlotLines frame history
    static constexpr uint32_t kMaxRegistered = 16;         // UI checkbox capacity
    static constexpr float kPanelWidth = 360.f;

    enum class Mode { Viewer, Compare };

    struct ImageResource {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
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
        VkBuffer uboGb = VK_NULL_HANDLE; // jittered scene UBO (GBuffer pass)
        VkDeviceMemory uboGbMemory = VK_NULL_HANDLE;
        void* uboGbMapped = nullptr;
        VkBuffer uboGbSpatial = VK_NULL_HANDLE; // unjittered LR scene UBO (spatial plugins)
        VkDeviceMemory uboGbSpatialMemory = VK_NULL_HANDLE;
        void* uboGbSpatialMapped = nullptr;
        VkBuffer uboGt = VK_NULL_HANDLE; // un-jittered scene UBO (GT pass)
        VkDeviceMemory uboGtMemory = VK_NULL_HANDLE;
        void* uboGtMapped = nullptr;
        VkBuffer lightingUboGb = VK_NULL_HANDLE; // lighting UBO (jittered invViewProj)
        VkDeviceMemory lightingUboGbMemory = VK_NULL_HANDLE;
        void* lightingUboGbMapped = nullptr;
        VkBuffer lightingUboGbSpatial = VK_NULL_HANDLE; // lighting UBO (unjittered LR)
        VkDeviceMemory lightingUboGbSpatialMemory = VK_NULL_HANDLE;
        void* lightingUboGbSpatialMapped = nullptr;
        VkBuffer lightingUboGt = VK_NULL_HANDLE; // lighting UBO (un-jittered invViewProj)
        VkDeviceMemory lightingUboGtMemory = VK_NULL_HANDLE;
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
    };

    struct AlgoColumn {
        std::string id;
        std::unique_ptr<IUpscaler> upscaler;
        std::string fg; // empty / "fsr3" / "nfru"
        std::unique_ptr<IFrameGen> frameGen;
        ImageResource output; // display-resolution RGBA16F
        VkImageLayout outputLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ImageResource fgOutput; // interpolated display-res HDR
        VkImageLayout fgOutputLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkDescriptorSet composeSetFg = VK_NULL_HANDLE;
        bool fgHistory = false;
        bool fgReady = false;
        VkBuffer blocksBuffer = VK_NULL_HANDLE;
        VkDeviceMemory blocksMemory = VK_NULL_HANDLE;
        VkDescriptorSet metricSet = VK_NULL_HANDLE;
        VkDescriptorSet composeSet = VK_NULL_HANDLE;
        float psnr = 0.f;
        float ssim = 0.f;
        bool hasMetric = false;
    };

    // What the render stack is currently built with.
    struct RenderConfig {
        Mode mode = Mode::Viewer;
        std::string scenePath; // empty = procedural
        float renderScale = 0.5f;
        uint32_t displayW = 1920;
        uint32_t displayH = 1080;
        struct AlgoSpec {
            std::string sr; // registered upscaler id
            std::string fg; // empty, "fsr3", or "nfru"
        };
        std::vector<AlgoSpec> algos; // viewer: 0 or 1; compare: up to kMaxAlgos
        bool gtSsaa = false; // compare: GT rendered at 2x, downsampled
        // compare/viewer: GT rendered at the low input resolution
        // (renderWidth_ x renderHeight_), no AA, no upscale pass.  Mutually
        // exclusive with gtSsaa; configFromUi gives gtApplyScale precedence.
        bool gtApplyScale = false;
        std::string envMapPath; // HDR env map; rebuilt when changed on Apply
    };

    // UI-editable state (shared scene/scale/resolution/camera + per-tab data).
    struct UiState {
        int sceneIndex = 0;
        char customScene[260] = {};
        float renderScale = 0.5f;
        int outputResIndex = 1; // index into kOutputResolutions
        char cameraPathFile[260] = {};
        int viewerUpscaler = 1; // 0 = native, 1..N = listUpscalers() entries
        int viewerFg = 0;       // 0 = none, 1 = FSR3, 2 = NFRU
        struct CompareSlot {
            int sr = 0; // index into upscalerNames_
            int fg = 0; // 0 = none, 1 = FSR3, 2 = NFRU
        };
        std::vector<CompareSlot> compareSlots;
        bool lockFps = true;
        int lockFpsTarget = 30; // original (render) FPS, 15..120
        bool compareGtSsaa = false; // GT 200% SSAA reference (Reference section, instant)
        // GT (Apply scale) reference: GT rendered at the low input
        // resolution, no AA/upscale (Reference section, instant).
        bool compareGtApplyScale = false;
        char viewerShotPath[260] = "output/gui_viewer.png";
        char compareShotPath[260] = "output/gui_compare.png";
        char frameTimesPath[260] = "output/gui_frame_times.csv";
        char envMap[260] = {}; // HDR env map path (filled from options at init)
        // bench tab
        bool benchSelected[kMaxRegistered] = {};
        bool benchNative = true;
        int benchFrames = 120;
        int benchWarmup = 30;
        char benchOutPath[260] = "output/gui_bench.csv";
    };

    struct BenchRowUi {
        std::vector<std::string> cols;
    };

    // --- core (window/context/swapchain/imgui), created once ----------------
    bool initImGui();
    void shutdownImGui();
    bool createUiSync();
    void destroyUiSync();
    void recordUiOnlyFrame(uint32_t slot, uint32_t swapchainIndex);

    // --- render stack (destroyed/recreated on Apply) --------------------------
    bool buildRenderStack();
    // Full teardown (everything including the scene).
    void destroyRenderStack();
    // Teardown of everything EXCEPT the scene (scene_, its texture set source
    // data and the materials survive; the material UBO / textureSet_ are
    // rebuilt from the preserved scene by the next build).
    void destroyStackResources();
    // Per-algorithm teardown only (upscaler shutdown, output image, metric
    // blocks buffer, per-algo descriptor sets) — the algo-only fast path.
    void destroyAlgoResources();
    // Resolution/window/swapchain half of a stack build (main thread only).
    bool beginStackConfig();
    // The light main-thread half: render targets, descriptors, pipelines,
    // sync objects, state resets (scene/upscalers must already be in place).
    bool buildStackTail();
    // Per-algorithm GPU resources: output image, metric blocks buffer
    // (compare), compose/metric descriptor sets + writes.  Used by
    // buildStackTail (full/middle rebuilds) and by the algo-only fast path.
    bool createAlgoResources(AlgoColumn& algo, uint32_t index);
    // Frame-history/jitter/zoom reset so temporal upscalers get resetHistory.
    void resetFrameState();
    // Compose descriptor set write (source image + text UBO + font atlas).
    void writeComposeSetInto(VkDescriptorSet set, VkImageView source);
    bool initAlgorithms(std::string& err);
    // Shared algo-init loop used by both the sync path and the async worker.
    // Appends successfully initialized upscalers to `out`; per-algo failures
    // keep the last message in `err` (fatal only in compare mode with zero
    // successes).  onAlgo fires before each init (worker progress text).
    bool initAlgorithmsFor(const VulkanEnv& env, const RenderConfig& cfg, uint32_t renderW,
                           uint32_t renderH, std::vector<AlgoColumn>& out, std::string& err,
                           const std::function<void(const char* name, uint32_t index,
                                                    uint32_t count)>& onAlgo = {});
    bool createRenderTargets();
    bool createFontAtlas();
    bool createMetricResources();
    bool loadShader(const char* name, VkShaderModule& out);
    bool createShaders();
    bool createDescriptors();
    bool createPipelines();
    bool createSyncResources();
    bool createScreenshotStaging();
    void ensurePresentSemaphores();

    // --- per-frame -------------------------------------------------------------
    void updateSceneUBO(void* mapped, bool jitter, uint32_t renderW, uint32_t renderH,
                        const Mat4& view, const Mat4& proj, const Mat4& projJittered,
                        const Mat4& prevViewProj);
    void updateLightingUBO(void* mapped, const Mat4& invViewProj,
                           const std::vector<Light>& lights, const ShadowFrame* shadow);
    // The Viewer-tab lighting section's override light list (sun direction/
    // intensity from the UI; scene_.lights untouched).  Also the source of
    // the CSM shadowed-sun index, so it is built once per frame in
    // recordFrame and shared by both lighting UBOs.
    std::vector<Light> buildLightOverride() const;
    void updateCamera(float dt);
    void recordFrame(uint32_t frameIndex, uint32_t swapchainIndex);
    // Compose columns into composeImage_ and copy to the swapchain with ImGui.
    // preferGenerated: Viewer interpolator first present / Compare columns.
    void recordComposePresent(VkCommandBuffer cmd, uint32_t swapchainIndex, bool preferGenerated);
    void recordViewerTruePresent(uint32_t uiSlot, uint32_t swapchainIndex);
    bool guiWantVsync() const;
    bool guiAllowMailbox() const;
    bool recreateGuiSwapchain();
    void ensureGuiSwapchainMode();
    void waitUntil(std::chrono::steady_clock::time_point t);
    void noteDisplayPresents(int nPres, std::chrono::steady_clock::time_point frameStart);
    void captureScreenshotIntoStaging(VkCommandBuffer cmd);
    // Debug (SR_GUI_UI_SHOT=1): screenshot variant that includes the ImGui
    // overlay — re-renders the present block into a GUI-owned image whose
    // format matches the swapchain (the ImGui backend pipeline's format).
    void captureUiScreenshotIntoStaging(VkCommandBuffer cmd);
    // Once the capture frame's fence has signaled: copy the staging pixels
    // out and hand them to a worker thread for PNG encoding (non-blocking).
    void collectScreenshotPixels();
    void harvestMetrics(uint32_t slot);
    void refreshOverlayText();
    // Compare-tab mouse wheel zoom (cursor-centered) + middle-drag pan.
    void handleCompareZoomInput();
    // Debug automation (SR_GUI_INPUT_FILE=<path>): consume one command per
    // frame from the file and inject it into the ImGui IO queue directly,
    // bypassing OS focus/cursor state (deterministic on attended machines).
    // Commands: "pos x y", "wheel f", "down n" / "up n" (0=left, 2=middle),
    // "key F1" / "keyup F1", "shot path.png", "wait" (one frame gap).
    void pumpInputFile();
    // Column layout origin: while the side panel is visible the render
    // columns start right of it; collapsed, they span the full window (the
    // GT column then touches the left edge).
    uint32_t layoutOriginX() const {
        return panelCollapsed_ ? 0u : static_cast<uint32_t>(kPanelWidth);
    }
    // Visible source region (px) for a column of colW x colH: aspect-preserving
    // center-crop at zoom 1; zoomed window centered on (centerU, centerV) when
    // zoom > 1.  out = {offX, offY, sizeX, sizeY}.
    static void computeViewRegion(uint32_t srcW, uint32_t srcH, uint32_t colW, uint32_t colH,
                                  float zoom, float centerU, float centerV, float out[4]);

    // --- UI ---------------------------------------------------------------------
    void drawUi();
    void drawCameraPose(); // bottom-left overlay: camera position + view angles
    void drawReferenceSection();
    void drawSharedControls();
    void drawViewerTab();
    void drawCompareTab();
    void drawBenchTab();
    // Vendor-grouped upscaler combo. nativeIndex>=0 adds a "native" item.
    // `index` is 0 = native when nativeIndex>=0, else 0 = first plugin.
    bool drawGroupedUpscalerCombo(const char* label, int* index, bool includeNative);
    void drawGroupedUpscalerCheckboxes(bool* selected, uint32_t count);
    void drawFrameLockControls();
    static const char* fgLabel(int fg); // 0 none / 1 FSR3 / 2 NFRU
    static const char* fgId(int fg);    // empty / "fsr3" / "nfru"
    int upscalerIndexById(const std::string& id) const;
    RenderConfig configFromUi(Mode mode) const;
    void requestRebuild(const RenderConfig& cfg);
    void applyRebuild();
    // --- async rebuild ---------------------------------------------------------
    // Apply starts a worker thread that loads the new scene into a temporary
    // Scene and initializes the upscalers while the main thread keeps
    // rendering the old stack; the swap happens in finishAsyncRebuild() at a
    // device-idle safe point.  No cancellation: a second Apply is refused
    // while a load is in flight.
    //
    // Tiered rebuild (classified in applyRebuild against the active config):
    //   sceneDirty   — scenePath changed (or no valid stack): full reload.
    //   targetsDirty — resolution/scale/gtSsaa/envMap changed: scene_ is
    //                  kept, everything else is rebuilt (worker still inits
    //                  the upscalers async, since DLSS/XeSS init is slow).
    //   (neither)    — only algos/mode changed: fast path; the worker
    //                  only initializes upscalers, finish swaps algos_ and
    //                  per-algo resources, scene/targets/descriptors persist.
    enum class LoadPhase { Idle, Loading, Ready, Failed };
    struct LoadResult {
        std::unique_ptr<Scene> scene; // null on the targets/algo-only tiers
        std::vector<AlgoColumn> algos;
        std::string error; // non-empty: hard failure, old stack stays active
        std::string note;  // non-fatal fallback message (one algo failed, ...)
    };
    void loadWorkerMain(RenderConfig cfg); // worker thread entry
    void finishAsyncRebuild();             // main thread: join + swap + tail build
    void discardLoadResult();              // main thread: free worker products
    void drawLoadOverlay();                // progress bar + stage text
    void applyLaunchOptions();
    void loadCameraPathFromUi();
    void saveScreenshot(const char* path);
    void drawScreenshotBusy(); // animated in-flight indicator next to the save buttons
    void exportFrameTimesCsv(const char* path);
    void refreshUpscalerAvailability();
    void startBench();
    void loadBenchCsv(const char* path);

    // --- core objects -------------------------------------------------------------
    Window window_;
    VulkanContext ctx_;
    Swapchain swapchain_;
    bool slRefHeld_ = false; // GUI holds one Streamline ref for the session

    // Minimal sync set for ImGui-only frames (used when the render stack is
    // broken, e.g. a failed rebuild, so the UI never goes black).  Also owns
    // the persistent per-swapchain-image present semaphores (renderFinished_).
    struct UiFrameSync {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
    };
    UiFrameSync uiFrames_[kFramesInFlight] = {};
    uint32_t uiFrameIndex_ = 0;
    bool stackOk_ = false;
    uint32_t renderFrameIndex_ = 0; // frames recorded with the current stack

    // --- render stack --------------------------------------------------------------
    RenderConfig active_;
    bool pendingRebuild_ = false;
    RenderConfig pendingConfig_;
    std::string statusLine_; // last rebuild/bench message for the UI

    // --- async rebuild state ---------------------------------------------------------
    // Written by the main thread before the worker starts; read by the worker.
    RenderConfig loadConfig_;
    // Tier classification of the pending rebuild (see the LoadPhase comment):
    // loadSceneDirty_ implies loadTargetsDirty_.
    bool loadSceneDirty_ = false;
    bool loadTargetsDirty_ = false;
    std::thread loadThread_;
    std::atomic<LoadPhase> loadPhase_{LoadPhase::Idle};
    // Progress: atomics for the counters, a short mutex-guarded string for the
    // stage text (worker writes, main reads every frame while Loading).
    std::atomic<uint32_t> loadDone_{0};
    std::atomic<uint32_t> loadTotal_{0};
    std::mutex loadStageMutex_;
    std::string loadStageText_;
    // Worker-private command pool (scene uploads + upscaler init), created by
    // the worker, destroyed by the main thread after join.
    VkCommandPool loadPool_ = VK_NULL_HANDLE;
    // Worker -> main handoff.  Only touched by the main thread after the phase
    // leaves Loading (release/acquire on loadPhase_, plus the thread join).
    LoadResult loadResult_;

    Scene scene_;
    Camera camera_;
    Camera prevCamera_;
    bool havePrevCamera_ = false;
    CameraPath path_;
    bool pathPlaying_ = false;
    uint32_t pathFrame_ = 0;
    std::vector<AlgoColumn> algos_;
    bool gtActive_ = false; // GT pass is part of the current stack

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

    // Viewer-tab "lighting" section: sun direction/intensity.  Applied per
    // frame through an override light list in updateLightingUBO — the scene
    // is never touched, so no rebuild is triggered.  Elevation/azimuth are
    // chosen to reproduce the defaultLights() sun direction
    // (normalize(0.35, 1, 0.3)) exactly, so the first frame with the override
    // active looks identical to the unmodified scene.
    bool sunEnabled_ = true;
    float sunElevationDeg_ = 65.3f; // asin(1 / sqrt(1.2125)) ~= 65.26 deg
    float sunAzimuthDeg_ = 49.4f;   // atan2(0.35, 0.3) ~= 49.40 deg
    float sunIntensity_ = 3.f;      // matches defaultLights() sun
    // CSM sun shadows (Viewer-tab lighting section, per-frame, no rebuild).
    bool shadowsEnabled_ = true;
    bool shadowDebugCascades_ = false; // tint pixels per shadow cascade

    ImageResource gbColor_;
    ImageResource gbColorSpatial_; // unjittered LR HDR copy for spatial upscalers
    ImageResource gbAlbedo_;
    ImageResource gbNormal_;
    ImageResource gbMaterial_;
    ImageResource gbEmissive_;
    ImageResource gbMotion_;
    ImageResource gbReactive_; // translucent coverage mask (reactive/TC input)
    ImageResource gbDepth_;
    ImageResource gtColor_;
    ImageResource gtAlbedo_;
    ImageResource gtNormal_;
    ImageResource gtMaterial_;
    ImageResource gtEmissive_;
    ImageResource gtDepth_;
    ImageResource gtSsaaColor_; // 2x GT render target (compare GT SSAA only)
    ImageResource gtSsaaAlbedo_;
    ImageResource gtSsaaNormal_;
    ImageResource gtSsaaMaterial_;
    ImageResource gtSsaaEmissive_;
    ImageResource gtSsaaDepth_;
    // SSAO (raw + blurred) per GBuffer path, R16_SFLOAT, path resolution.
    ImageResource gbAoRaw_;
    ImageResource gbAo_;
    ImageResource gtAoRaw_;
    ImageResource gtAo_;
    ImageResource gtSsaaAoRaw_; // compare GT SSAA only
    ImageResource gtSsaaAo_;
    ImageResource composeImage_; // RGBA8 composite; presentation + screenshot source
    ImageResource uiShotImage_;  // BGRA8 debug screenshot target incl. ImGui
    VkImageLayout uiShotLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    bool uiShot_ = false;        // SR_GUI_UI_SHOT: screenshots include ImGui
    ImageResource fontAtlas_;

    DeferredCore deferred_;        // shared deferred PBR + IBL (once per env map)
    std::string envMapActive_;     // env map deferred_ was built with
    // CSM shadow map (fixed 2048^2 x 4, resolution- and env-independent):
    // created once next to deferred_, survives scene/config rebuilds, and is
    // shared by the GB/GT/SSAA lighting paths.  shadowsActive_ = false
    // (creation failed) degrades to no shadows.
    ShadowTargets shadow_;
    bool shadowsActive_ = false;

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
    // One renderFinished semaphore per swapchain image (present sync).
    // Persistent across render-stack rebuilds: owned by createUiSync /
    // destroyUiSync, resized by ensurePresentSemaphores on swapchain recreate.
    std::vector<VkSemaphore> renderFinished_;
    VkDescriptorSet textureSet_ = VK_NULL_HANDLE;
    VkDescriptorSet copySet_ = VK_NULL_HANDLE;
    VkDescriptorSet gtComposeSet_ = VK_NULL_HANDLE;
    VkDescriptorSet gtDownsampleSet_ = VK_NULL_HANDLE; // 2x GT source (SSAA)
    // SSAO descriptor sets (static: the referenced textures never change).
    VkDescriptorSet ssaoSetGb_ = VK_NULL_HANDLE;
    VkDescriptorSet ssaoBlurSetGb_ = VK_NULL_HANDLE;
    VkDescriptorSet ssaoSetGt_ = VK_NULL_HANDLE;
    VkDescriptorSet ssaoBlurSetGt_ = VK_NULL_HANDLE;
    VkDescriptorSet ssaoSetSsaa_ = VK_NULL_HANDLE;     // gtSsaa only
    VkDescriptorSet ssaoBlurSetSsaa_ = VK_NULL_HANDLE; // gtSsaa only
    VkSampler linearSampler_ = VK_NULL_HANDLE;
    VkSampler fontSampler_ = VK_NULL_HANDLE;

    VkImageLayout gbColorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbColorSpatialLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbAlbedoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbNormalLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbMaterialLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbEmissiveLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbMotionLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbReactiveLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbDepthLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtColorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtAlbedoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtNormalLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtMaterialLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtEmissiveLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtDepthLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaColorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaAlbedoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaNormalLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaMaterialLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaEmissiveLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaDepthLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbAoRawLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbAoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtAoRawLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtAoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaAoRawLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaAoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout composeLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    VkBuffer materialUbo_ = VK_NULL_HANDLE;
    VkDeviceMemory materialUboMemory_ = VK_NULL_HANDLE;
    uint32_t materialStride_ = 0;

    VkBuffer textUbo_ = VK_NULL_HANDLE; // packed ASCII overlay text (all columns)
    VkDeviceMemory textUboMemory_ = VK_NULL_HANDLE;
    void* textUboMapped_ = nullptr;

    VkBuffer metricResultBuf_ = VK_NULL_HANDLE;
    VkDeviceMemory metricResultMemory_ = VK_NULL_HANDLE;
    VkBuffer metricStaging_[kFramesInFlight] = {};
    VkDeviceMemory metricStagingMemory_[kFramesInFlight] = {};
    void* metricStagingMapped_[kFramesInFlight] = {};
    bool metricPending_[kFramesInFlight] = {};
    uint32_t metricMask_[kFramesInFlight] = {};

    VkBuffer screenshotStaging_ = VK_NULL_HANDLE;
    VkDeviceMemory screenshotStagingMemory_ = VK_NULL_HANDLE;
    void* screenshotMapped_ = nullptr;

    TimestampQuery timestamps_;

    // --- runtime/UI state ----------------------------------------------------------
    UiState ui_;
    int currentTab_ = 0; // 0 viewer, 1 compare, 2 bench
    int tabRequest_ = -1; // one-shot ImGui tab selection (launch options)
    bool panelCollapsed_ = false; // side panel hidden (F1 / panel button toggles)
    std::vector<std::string> upscalerNames_;     // registered plugin ids
    std::vector<std::string> upscalerLabels_;    // display names (from name())
    std::vector<char> upscalerAvailable_;        // isAvailable() probe results
    std::vector<SceneEntry> scenes_;             // needs SceneRegistry include

    float frameMsHistory_[kHistoryLen] = {};
    uint32_t historyHead_ = 0;
    uint32_t historyCount_ = 0;
    TimestampQuery::Timings lastTimings_;
    float fps_ = 0.f;
    std::chrono::steady_clock::time_point fpsLockDeadline_{};
    bool swapchainVsync_ = true;
    bool swapchainMailbox_ = true;
    std::vector<TimestampQuery::Timings> frameTimesLog_; // for CSV export

    bool screenshotPending_ = false;
    std::string screenshotPathPending_;
    // Async screenshot writeback: the capture is recorded into a frame; once
    // that frame's fence signals, the staging pixels are copied out and the
    // PNG is encoded on a worker thread (stb's deflate stalls the main loop
    // for a noticeable time at 1080p+).
    bool screenshotInFlight_ = false; // capture recorded, waiting for the fence
    uint32_t screenshotSlot_ = 0;
    // Detached PNG-encoder threads may outlive the GuiApp on abnormal exits;
    // keep the state they touch in a shared block so a worker holding a copy
    // never dereferences a dangling `this`.
    struct ScreenshotShared {
        std::mutex msgMutex;
        std::string msg;
        std::atomic<int> threads{0};
        std::atomic<int> finished{0};
    };
    std::shared_ptr<ScreenshotShared> screenshotShared_ = std::make_shared<ScreenshotShared>();

    // Compare-tab view state (reset on rebuild; not part of RenderConfig).
    float compareZoom_ = 1.f;      // 1..16
    float comparePanU_ = 0.5f;     // zoom window center, normalized source UV
    float comparePanV_ = 0.5f;
    bool comparePanDrag_ = false;  // middle-drag started over the render area
    bool launchZoomPending_ = false; // apply opts_.compareZoom on frame 1

    BenchRunner bench_;
    std::string benchOutUsed_; // CSV path of the running/last bench
    std::vector<BenchRowUi> benchRows_;
    std::vector<std::string> benchHeader_;

    GuiOptions opts_;
    std::string inputFile_;    // SR_GUI_INPUT_FILE automation (empty = off)
    uint64_t inputFileLine_ = 0; // lines consumed so far
    bool benchAutoRun_ = false;
    bool benchStarted_ = false;
    // Quit deadline after an auto bench finishes (time-based: ImGui-only
    // frames are unthrottled, so a frame countdown would expire instantly).
    std::chrono::steady_clock::time_point benchQuitAt_{};
    bool benchQuitArmed_ = false;
};

} // namespace sr
