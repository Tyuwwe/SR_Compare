#pragma once
// ============================================================================
// GuiApp — Dear ImGui front end for sr_compare (`sr_compare gui`).
//
// Covers the full CLI surface in four tabs:
//   Viewer  — single-upscaler full-screen preview (scene/upscaler/scale/
//             resolution/camera path/screenshot/frame-times CSV).
//   Compare — split-screen GT + up to 4 algorithm columns with the live
//             GPU-reduction PSNR/SSIM overlay (same shaders as compare mode).
//   Bench   — spawns `sr_compare bench` as a child process and shows the
//             resulting CSV in a table.
//   Debug   — fullscreen visualization of the render inputs (HDR color,
//             linearized depth, motion vectors, reactive mask), an upscaler
//             output, or a plugin SDK debug view (patch 0004 design).  The
//             render stack is not rebuilt for this tab; while a plugin view
//             that replaces the output is active, compare metrics for that
//             column are suspended.
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
// SDL window, Vulkan device and ImGui backend persist).  Apply is
// asynchronous: a worker thread loads the new scene and initializes the
// upscalers while the old stack keeps rendering under a progress overlay, and
// the new stack is swapped in at a device-idle safe point (see the
// LoadPhase/LoadResult machinery below).  Only the initial build in init()
// is synchronous.
// ============================================================================
#include "gui/BenchRunner.h"
#include "gui/GpuProfilerWindow.h"
#include "gui/RenderGraphEditor.h"
#include "renderer/ColorGrading.h"
#include "renderer/core/EngineConfig.h"
#include "renderer/core/GpuProfiler.h"
#include "renderer/core/Swapchain.h"
#include "renderer/core/TimestampQuery.h"
#include "renderer/core/VulkanContext.h"
#include "renderer/core/Window.h"
#include "renderer/deferred/DeferredCore.h"
#include "renderer/deferred/PlanarReflection.h"
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
    std::string envMapPath;    // empty = procedural sky atmosphere (default)
    float exposure = 0.f;      // > 0 overrides the scene lighting preset
    // engine.toml contents (loaded by main before init; empty optionals when
    // absent) + the explicit-CLI mask for precedence — see EngineConfig.h.
    EngineConfig engineCfg;
    uint64_t engineCfgCli = cli::kNone;
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
    // Side panel width in UI units, scaled with the display content scale so
    // high-DPI scaling does not squeeze the (scaled) widgets into a fixed
    // pixel strip.
    float panelWidth() const { return kPanelWidth * uiScale_; }

    enum class Mode { Viewer, Compare };
    // Debug-tab texture views (patch 0004 design); UiState::debugSource.
    enum class DebugSource : int {
        InputColor,
        Depth,
        Motion,
        UpscaledOutput,
        ReactiveMask,
        PluginView,
    };

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
        VkBuffer uboGb = VK_NULL_HANDLE; // jittered scene UBO (GBuffer pass)
        VmaAllocation uboGbMemory = VK_NULL_HANDLE;
        void* uboGbMapped = nullptr;
        VkBuffer uboGbSpatial = VK_NULL_HANDLE; // unjittered LR scene UBO (spatial plugins)
        VmaAllocation uboGbSpatialMemory = VK_NULL_HANDLE;
        void* uboGbSpatialMapped = nullptr;
        VkBuffer uboGt = VK_NULL_HANDLE; // un-jittered scene UBO (GT pass)
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
        VmaAllocation blocksMemory = VK_NULL_HANDLE;
        VkDescriptorSet metricSet = VK_NULL_HANDLE;
        VkDescriptorSet composeSet = VK_NULL_HANDLE;
        float psnr = 0.f;
        float ssim = 0.f;
        bool hasMetric = false;
        // A plugin debug view that replaces this column's output is active
        // (Debug tab): compare metrics are meaningless and stay suspended.
        bool debugOutputActive = false;
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
        // debug tab (patch 0004): the tab itself activates the fullscreen
        // debug view; these select what it shows.
        int debugSource = static_cast<int>(DebugSource::InputColor);
        int debugTarget = 0;     // algo column the target-dependent views use
        int debugPluginView = 0; // index into the target's debugViewCount()
        bool debugLogDepth = true;
        float debugMaxDepth = 100.f;   // meters
        float debugMaxMotion = 16.f;   // source pixels
        float debugReactiveGain = 1.f;
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
    // DPI content scale applied to the ImGui style: ScaleAllSizes by the
    // ratio to the previous scale + FontScaleMain (dynamic fonts rasterize at
    // the scaled size, so text stays crisp).  Called at init from
    // window_.contentScale() and again on SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED.
    void applyUiScale(float scale);
    // Vulkan backend half of initImGui; re-run by setHdrEnabled after the
    // swapchain format changes (Phase 6c).
    bool initImGuiVulkanBackend();
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
    bool createPlanarReflections();
    // Auto-exposure channels + sets (needs the descriptor pool; called from
    // createDescriptors once it exists).
    bool createAutoExposureResources();
    // Phase 7a cull resources (instance SSBO + per-path cull channels; needs
    // the pool, the depth pyramids and the allocated scene sets).
    bool createCullResources();
    void destroyCullResources();
    // Per-path display exposure: harvested auto value, or the manual slider
    // value when auto exposure is off.
    float lrExposureNow() const {
        return autoExposureEnabled_ ? lrExposure_.value : exposure_;
    }
    float gtExposureNow() const {
        return autoExposureEnabled_ ? gtExposure_.value : exposure_;
    }
    bool createPipelines();
    bool createSyncResources();
    bool createScreenshotStaging();
    void ensurePresentSemaphores();

    // --- per-frame -------------------------------------------------------------
    void updateSceneUBO(void* mapped, bool jitter, uint32_t renderW, uint32_t renderH,
                        const Mat4& view, const Mat4& proj, const Mat4& projJittered,
                        const Mat4& prevViewProj);
    void updateLightingUBO(void* mapped, const Mat4& viewProj,
                           const std::vector<Light>& lights, const ShadowFrame* shadow);
    void updateClusterLights(uint32_t frameIndex, const std::vector<Light>& lights);
    // The Viewer-tab lighting section's override light list (sun direction/
    // intensity from the UI; scene_.lights untouched).  Also the source of
    // the CSM shadowed-sun index, so it is built once per frame in
    // recordFrame and shared by both lighting UBOs.
    std::vector<Light> buildLightOverride() const;
    void applyLightingPreset(const LightingPreset& p);
    // Re-renders the sky-atmosphere env + IBL maps from the current sun
    // sliders (no-op in static-env mode or while a load is in flight).
    void updateSkyFromUiSun();
    // Viewer-tab "bake reflection probes" button: synchronous offline bake of
    // scene_.probes through the shared ProbeBaker (renderer/ibl/ProbeBaker.h,
    // same code as CLI --bake-probes), then a probe reload so the fresh
    // captures take effect without a stack rebuild.  Blocks the render thread
    // for the bake duration; refused while a load or bench is in flight.
    void bakeReflectionProbesFromUi();
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
    // HDR checkbox handler (Phase 6c): re-creates the swapchain, copy pipeline
    // and ImGui backend for the new surface format (all at device idle).
    void setHdrEnabled(bool enabled);
    // Swapchain-format-dependent fullscreen copy pipeline (created by
    // createPipelines, re-created by setHdrEnabled).
    bool createCopyPipeline();
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
    // "key F1" / "keyup F1", "shot path.png", "graph" (toggle the Render
    // Graph editor window), "profiler" (toggle the GPU profiler window), "pass <name> <0|1>" (toggle a runtime pass
    // switch: shadows/contact/ssr/volfog/occlusion/bloom/mb/dof/autoexp),
    // "fullscreen <0|1>" (borderless fullscreen), "hdr <0|1>" (HDR output
    // toggle), "tab <n>" (0 viewer, 1 compare, 2 bench, 3 debug),
    // "debug <source>" (Debug-tab view: input/depth/motion/output/reactive/
    // plugin), "debugtarget <n>" (Debug-tab target column), "wait" (one
    // frame gap).
    void pumpInputFile();
    // The render columns always span the full window client area; the side
    // panel is a floating overlay on top (F1 collapses it to a slim strip).
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
    void drawDebugTab();
    // Vendor-grouped upscaler combo. nativeIndex>=0 adds a "native" item.
    // `index` is 0 = native when nativeIndex>=0, else 0 = first plugin.
    bool drawGroupedUpscalerCombo(const char* label, int* index, bool includeNative);
    void drawGroupedUpscalerCheckboxes(bool* selected, uint32_t count);
    void drawFrameLockControls();
    static const char* fgLabel(int fg); // 0 none / 1 FSR3 / 2 NFRU
    static const char* fgId(int fg);    // empty / "fsr3" / "nfru"
    int upscalerIndexById(const std::string& id) const;
    // Clamped Debug-tab target column index (0 when no upscaler is active).
    uint32_t debugTargetAlgo() const;
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
    // engine.toml application (renderer/core/EngineConfig.h).  applyEngineConfigHot
    // sets only per-frame state (effects/DOF/grading/sun sliders, exposure,
    // occlusion/lod, HDR swapchain toggle, window fullscreen) — no rebuild —
    // and is used both for the initial defaults and for hot reloads.
    // pollEngineConfig() stats the file once per second in run() and
    // re-applies on modification; resolution/scale/env-map/scene/LUT changes
    // need an Apply rebuild and are ignored (scale/env-map do land in the UI
    // fields, so the auto-save never clobbers an external edit).
    void applyEngineConfigHot(const EngineConfig& cfg, EngineConfigLog& log);
    void pollEngineConfig();
    // engine.toml auto-create/auto-save (the GUI owns the file — see
    // EngineConfig.h).  currentEngineConfig snapshots every file key from the
    // UI state; pollEngineConfigAutoSave diffs the canonical serialization
    // once per frame and writes (atomic temp+replace) after ~1.5 s without
    // further changes.  A deleted file is re-created with the hot parameters
    // reset to defaults (resetEngineConfigDefaults) by pollEngineConfig.
    EngineConfig currentEngineConfig() const;
    void saveEngineConfigNow();
    void pollEngineConfigAutoSave();
    void resetEngineConfigDefaults();
    // Borderless fullscreen switch (panel checkbox, toml hot path, input-file
    // automation); forwards to Window::setFullscreen.
    void setFullscreenEnabled(bool enabled);
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
    // True until init()'s first buildRenderStack returns: beginStackConfig
    // skips the window-size snap for that build (see the comment there).
    bool initialStackBuild_ = true;
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
    float sunElevationDeg_ = 65.3f; // matches defaultLightingPreset()
    float sunAzimuthDeg_ = 49.4f;
    float sunIntensity_ = 3.f;
    Vec3 sunColor_{1.f, 0.95f, 0.85f};
    bool fillEnabled_ = true;       // defaultLights() blue point fill
    float iblIntensity_ = 1.f;
    float exposure_ = 1.f;          // manual display exposure (ACES input multiplier);
                                    // also the seed value for auto exposure
    // Histogram-based auto exposure (UE4 AutoExposure style; per-frame params,
    // no rebuild — the solver state lives in the ExposureChannels below and is
    // re-seeded on stack rebuilds from exposure_).  CLI --exposure starts the
    // app in manual mode.  LR and GT paths solve independently.
    bool autoExposureEnabled_ = true;
    float exposureMinEV_ = -8.f;    // EV clamp range (GUI sliders)
    float exposureMaxEV_ = 8.f;
    bool autoExposureJustEnabled_ = false; // snap the solver next frame
    ExposureChannel lrExposure_;    // gbColor_ -> upscaler preExposure + algo columns
    ExposureChannel gtExposure_;    // gtColor_ / gtSsaaColor_ -> GT column + metric ref
    // CSM sun shadows (Viewer-tab lighting section, per-frame, no rebuild).
    bool shadowsEnabled_ = true;
    bool shadowDebugCascades_ = false; // tint pixels per shadow cascade
    // Screen-space contact shadows for the CSM sun (per-frame UBO flag; needs
    // shadows on — rides the CSM sun selection).
    bool contactShadowsEnabled_ = true;
    // Opaque screen-space reflections (per-frame pass skip, no rebuild).
    // Off by default — the probe/env fallback carries the specular term.
    // CLI/engine.toml-only switch: the panel checkbox and the graph node are
    // locked; applyPassToggle(Ssr) is reached only from applyEngineConfigHot
    // and the SR_GUI_INPUT_FILE debug hook.
    bool ssrEnabled_ = false;
    // Global SSR weight scale (trace-stage confidence multiplier, 0..1).
    float ssrStrength_ = 0.6f;
    // Planar mirror reflections (per-frame pass skip, no rebuild; the
    // resources exist whenever the scene has mirror glass).  Off = the
    // mirror panes fall back to the inline glass SSR trace.
    bool planarEnabled_ = true;
    // Froxel volumetric fog (per-frame pass skip; params from the scene
    // lighting preset, toggling resets the temporal history).
    bool volFogEnabled_ = true;
    VolFogParams fogParams_;
    // Screen-size LOD switching + distance cull (per-frame CPU selection).
    bool lodEnabled_ = lodEnabledByDefault();
    // GPU Hi-Z occlusion culling (Phase 7a, per-frame pass skip; SR_OCCLUSION=0
    // default-off, checkbox overrides interactively).  Indirect draws stay live
    // either way — off means "all candidates visible".
    bool occlusionEnabled_ = occlusionEnabledByDefault();
    // Terminal lens-effects chain (Phase 6a, compose push constants; lens
    // dirt stays viewer-only — the compose pass has no dirt binding, though
    // the bloom pyramid it modulates is now built per path).  Strengths are
    // the shared DeferredCore defaults (kLens*Strength).
    bool lensCaEnabled_ = true;
    bool lensVignetteEnabled_ = true;
    bool lensGrainEnabled_ = true;
    // HDR bloom (Phase 6a pyramid): per-frame pass skip like SSR, no rebuild —
    // the per-path chains stay allocated either way.  Off by default in every
    // mode (CLI --bloom / engine.toml [effects] bloom opt in).
    bool bloomEnabled_ = false;
    // Motion blur + depth of field (Phase 6b, per-frame pass skip, no temporal
    // state — no history reset needed).  Off by default in every mode (same
    // default policy as the viewer/compare CLI); when enabled the same
    // algorithm + parameters run on every path so the GT column blurs
    // identically.
    bool motionBlurEnabled_ = false;
    bool dofEnabled_ = false;
    // DOF tuning (Phase 6b; sliders in the viewer tab, applied per-frame via
    // PostFxParams — no rebuild, no temporal state).  focus 0 = auto-focus on
    // the screen-centre texel; f-stop maps to the aperture scale as
    // kDofAperture * (kDofDefaultFstop / fstop); max blur is the CoC radius
    // clamp in display px at 1080p, scaled by path height.
    float dofFocus_ = 0.f;
    float dofFstop_ = kDofDefaultFstop;
    float dofMaxBlur_ = kDofMaxCoC;
    // Log-domain color grading (Phase 6c, compose push constants; identical
    // parameters for every column — sliders in the shared controls).  The LUT
    // is the procedural identity (no LUT file UI); the GUI is the interactive
    // grading front end.
    float gradeTemperatureK_ = 6500.f;
    float gradeTint_ = 0.f;
    float gradeContrast_ = 1.f;
    float gradeSaturation_ = 1.f;
    GradingLutGpu gradingLut_; // procedural identity, created once in init
    // HDR swapchain output (Phase 6c): checkbox gated on surface support
    // (probed once at init).  Toggling re-creates the swapchain, the copy
    // pipeline and the ImGui backend (all bake the swapchain format).  The
    // composite stays the SDR display-encoded image — copy.frag re-linearizes
    // and PQ/scRGB-encodes it (SDR content in an HDR container; true scene
    // HDR headroom is viewer-only, see present.frag).
    bool hdrEnabled_ = false;
    bool hdrSupportHdr10_ = false;
    bool hdrSupportScRgb_ = false;
    // Borderless (desktop) fullscreen ([window] fullscreen in engine.toml;
    // hot-reloadable, no CLI flag).  The render resolution stays the
    // configured output size; the swapchain is recreated through the normal
    // OUT_OF_DATE path on the mode switch.
    bool fullscreenEnabled_ = false;
    // "copied!" feedback countdown (seconds) for the click-to-copy camera
    // pose overlay (drawCameraPose); 0 = hidden.
    float poseCopyFlash_ = 0.f;
    // Windowed client size ([window] width/height in engine.toml): remembered
    // across runs, tracked from resize events while not fullscreen and saved
    // back by the debounced auto-save.  0 = not initialized yet (init seeds
    // it from the toml value or the output resolution).  These are LOGICAL
    // window coordinates (SDL_GetWindowSize space); on Windows that equals
    // physical pixels, so the remembered size round-trips unchanged across
    // monitors with different display scaling.
    int windowedW_ = 0;
    int windowedH_ = 0;
    // Content scale currently applied to the ImGui style (1.0 = 100% DPI).
    float uiScale_ = 1.f;

    ImageResource gbColor_;
    ImageResource gbColorSpatial_; // unjittered LR HDR copy for spatial upscalers
    ImageResource gbAlbedo_;
    ImageResource gbNormal_;
    ImageResource gbMaterial_;
    ImageResource gbEmissive_;
    ImageResource gbMotion_;
    ImageResource gbReactive_; // translucent coverage mask (reactive/TC input)
    // 3x3-max dilated + motion-gated copy of gbReactive_
    // (reactive_dilate.comp): the plateau absorbs jittered-coordinate
    // sampling; static coverage is gated to zero so it keeps its history
    // weight.  Fed to the upscalers.
    ImageResource gbReactiveDilated_;
    VkDescriptorSet reactiveDilateSet_ = VK_NULL_HANDLE;
    ImageResource gbDepth_;
    ImageResource gtColor_;
    ImageResource gtAlbedo_;
    ImageResource gtNormal_;
    ImageResource gtMaterial_;
    ImageResource gtEmissive_;
    ImageResource gtMotion_; // GT-path motion vectors (Phase 6b MB/DOF)
    ImageResource gtDepth_;
    ImageResource gtSsaaColor_; // 2x GT render target (compare GT SSAA only)
    ImageResource gtSsaaAlbedo_;
    ImageResource gtSsaaNormal_;
    ImageResource gtSsaaMaterial_;
    ImageResource gtSsaaEmissive_;
    ImageResource gtSsaaMotion_; // 2x GT motion RT (Phase 6b)
    ImageResource gtSsaaDepth_;
    // GTAO working target (RG16F: AO + view Z) and filtered R16F, path res.
    ImageResource gbAoRaw_;
    ImageResource gbAo_;
    ImageResource gtAoRaw_;
    ImageResource gtAo_;
    ImageResource gtSsaaAoRaw_; // compare GT SSAA only
    ImageResource gtSsaaAo_;
    // HDR color mip chains (lit opaque color, box-filtered) for
    // roughness-aware SSR; same GENERAL-for-life resource model as the
    // depth pyramids.  Deliberately separate from the Phase 6a bloom pyramid
    // (box-average reflection LOD vs thresholded extract — see DeferredCore).
    ColorPyramid gbColorPyramid_;
    ColorPyramid gtColorPyramid_;
    ColorPyramid gtSsaaColorPyramid_; // compare GT SSAA only
    // Clustered shading grids (per-path resolution, per-slot buffers).
    ClusterGrid gbCluster_;
    ClusterGrid gtCluster_;
    ClusterGrid gtSsaaCluster_; // compare GT SSAA only
    // Hi-Z depth pyramids for the SSR march (LR / GT / GT-SSAA paths); general
    // DeferredCore resource, later reused by GTAO/contact shadows/culling.
    DepthPyramid gbPyramid_;
    DepthPyramid gtPyramid_;
    DepthPyramid gtSsaaPyramid_; // compare GT SSAA only
    // GPU occlusion culling + indirect GBuffer draws (Phase 7a): shared
    // instance SSBO + one cull channel per path (LR / GT / GT-SSAA), each
    // bound to that path's Hi-Z chain.  occlusionEnabled_ = opt-out; when
    // false the cull pass is skipped (the indirect path stays live).
    InstanceBuffer instances_;
    CullChannel gbCull_;
    CullChannel gtCull_;
    CullChannel gtSsaaCull_; // compare GT SSAA only
    std::vector<CullDrawRun> cullRuns_;
    std::vector<GpuInstance> cullInstCpu_; // build scratch (capacity-sized)
    std::vector<VkDrawIndexedIndirectCommand> cullCmdCpu_;
    uint32_t cullCandidates_ = 0;
    // GTAO view-Z depth chains (XeGTAO DepthMIPFilter) + temporal accumulation
    // ping-pong state, one per path.  Per-path previous-frame view-projection
    // (jittered for LR) and frame counters drive the temporal pass; a zero
    // counter resets (bypasses) the history.
    DepthPyramid gbPyramidAo_;
    DepthPyramid gtPyramidAo_;
    DepthPyramid gtSsaaPyramidAo_; // compare GT SSAA only
    AoHistory gbAoHist_;
    AoHistory gtAoHist_;
    AoHistory gtSsaaAoHist_; // compare GT SSAA only
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
    // the GTAO temporal state above.
    ImageResource gbSsrTrace_;
    ImageResource gtSsrTrace_;
    ImageResource gtSsaaSsrTrace_; // compare GT SSAA only
    SsrHistory gbSsrHist_;
    SsrHistory gtSsrHist_;
    SsrHistory gtSsaaSsrHist_; // compare GT SSAA only
    Mat4 prevSsrViewProjGb_ = Mat4::identity();
    Mat4 prevSsrViewProjGt_ = Mat4::identity();
    Mat4 prevSsrViewProjSsaa_ = Mat4::identity();
    uint32_t ssrFramesGb_ = 0;
    uint32_t ssrFramesGt_ = 0;
    uint32_t ssrFramesSsaa_ = 0;
    // Froxel volumetric fog volumes + temporal state (Phase 5a), one per
    // path.  Accumulate runs once per frame per path (fogAccumFrame* guard,
    // mixed mode records LR lighting twice); composite runs in every record.
    VolFogVolume gbFog_;
    VolFogVolume gtFog_;
    VolFogVolume gtSsaaFog_; // compare GT SSAA only
    Mat4 prevFogViewProjGb_ = Mat4::identity();
    Mat4 prevFogViewProjGt_ = Mat4::identity();
    Mat4 prevFogViewProjSsaa_ = Mat4::identity();
    uint32_t fogFramesGb_ = 0;
    uint32_t fogFramesGt_ = 0;
    uint32_t fogFramesSsaa_ = 0;
    uint32_t fogAccumFrameGb_ = ~0u;
    uint32_t fogAccumFrameGt_ = ~0u;
    uint32_t fogAccumFrameSsaa_ = ~0u;
    // Motion blur + depth of field working sets (Phase 6b), one per path.
    PostFxTargets gbPostFx_;
    PostFxTargets gtPostFx_;
    PostFxTargets gtSsaaPostFx_; // compare GT SSAA only
    // HDR bloom pyramids (Phase 6a), one per path; same host-owned
    // GENERAL-for-life resource model as the post-fx targets.  The runtime
    // checkbox skips the pass; the chains stay allocated either way.
    BloomPyramid gbBloom_;
    BloomPyramid gtBloom_;
    BloomPyramid gtSsaaBloom_; // compare GT SSAA only
    ImageResource composeImage_; // RGBA8 composite; presentation + screenshot source
    ImageResource uiShotImage_;  // BGRA8 debug screenshot target incl. ImGui
    VkImageLayout uiShotLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    bool uiShot_ = false;        // SR_GUI_UI_SHOT: screenshots include ImGui
    ImageResource fontAtlas_;

    DeferredCore deferred_;        // shared deferred PBR + IBL (once per env map)
    // Planar mirror reflection passes (LR / GT paths; the spatial-LR and
    // SSAA variants keep the SSR fallback).  Rebuilt with the stack.
    PlanarReflection gbPlanar_;
    PlanarReflection gtPlanar_;
    const PlanarReflection* activePlanar_ = nullptr; // patches the SceneUBO being filled
    std::string envMapActive_;     // env map deferred_ was built with
    // CSM shadow map (fixed 2048^2 x 4, resolution- and env-independent):
    // created once next to deferred_, survives scene/config rebuilds, and is
    // shared by the GB/GT/SSAA lighting paths.  shadowsActive_ = false
    // (creation failed) degrades to no shadows.
    ShadowTargets shadow_;
    bool shadowsActive_ = false;
    // Spot light shadow atlas (Phase 4b, fixed 4096^2): same lifetime and
    // sharing as the CSM targets.
    ShadowAtlas spotAtlas_;
    bool spotAtlasActive_ = false;

    VkDescriptorSetLayout composeSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout copySetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout metricSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout composePipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout copyPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout metricPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline composePipeline_ = VK_NULL_HANDLE;
    VkPipeline debugComposePipeline_ = VK_NULL_HANDLE; // Debug tab (debug_compose.frag)
    VkPipeline copyPipeline_ = VK_NULL_HANDLE;
    VkPipeline downsamplePipeline_ = VK_NULL_HANDLE; // 2x GT -> 1x GT (SSAA)
    VkPipeline metricBlocksPipeline_ = VK_NULL_HANDLE;
    VkPipeline metricReducePipeline_ = VK_NULL_HANDLE;
    VkShaderModule fullscreenVert_ = VK_NULL_HANDLE;
    VkShaderModule composeFrag_ = VK_NULL_HANDLE;
    VkShaderModule debugComposeFrag_ = VK_NULL_HANDLE;
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
    // Debug-tab views (patch 0004): persistent compose-layout sets over the
    // render-res input textures (written once at stack build; the images are
    // recreated with the stack, so the sets are rebuilt with it).
    VkDescriptorSet debugInputTemporalSet_ = VK_NULL_HANDLE;
    VkDescriptorSet debugInputSpatialSet_ = VK_NULL_HANDLE;
    VkDescriptorSet debugDepthSet_ = VK_NULL_HANDLE;
    VkDescriptorSet debugMotionSet_ = VK_NULL_HANDLE;
    VkDescriptorSet debugReactiveSet_ = VK_NULL_HANDLE;
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
    VkImageLayout gbAlbedoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbNormalLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbMaterialLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbEmissiveLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbMotionLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbReactiveLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbReactiveDilatedLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbDepthLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtColorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtAlbedoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtNormalLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtMaterialLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtEmissiveLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtMotionLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtDepthLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaColorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaAlbedoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaNormalLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaMaterialLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaEmissiveLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaMotionLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaDepthLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
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

    VkBuffer metricResultBuf_ = VK_NULL_HANDLE;
    VmaAllocation metricResultMemory_ = VK_NULL_HANDLE;
    VkBuffer metricStaging_[kFramesInFlight] = {};
    VmaAllocation metricStagingMemory_[kFramesInFlight] = {};
    void* metricStagingMapped_[kFramesInFlight] = {};
    bool metricPending_[kFramesInFlight] = {};
    uint32_t metricMask_[kFramesInFlight] = {};

    VkBuffer screenshotStaging_ = VK_NULL_HANDLE;
    VmaAllocation screenshotStagingMemory_ = VK_NULL_HANDLE;
    void* screenshotMapped_ = nullptr;

    TimestampQuery timestamps_;
    // Per-pass GPU timestamp zones + the ImGui profiler panel.  The window's
    // `open` flag drives profiler_.setEnabled in the run loop, so a closed
    // panel records no timestamps at all.
    GpuProfiler profiler_;
    GpuProfilerWindow profilerWindow_;
    // "Render Graph" ImNodes editor window: visualizes the PassRegistry
    // mirror of recordFrame; node toggles go through applyPassToggle.
    RenderGraphEditor graphWindow_;
    // Resolve/apply a PassToggle (shared by the panel checkboxes and the
    // graph editor nodes; applyPassToggle carries the checkbox side effects
    // like the fog history reset).
    bool passToggleValue(rg::PassToggle t) const;
    void applyPassToggle(rg::PassToggle t, bool value);
    float cpuRecordMs_ = 0.f; // CPU duration of the last recordFrame

    // --- runtime/UI state ----------------------------------------------------------
    UiState ui_;
    int currentTab_ = 0; // 0 viewer, 1 compare, 2 bench, 3 debug
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
    // SR_GUI_INPUT_FILE automation (empty = off).  Mouse coordinates in the
    // file are LOGICAL ImGui coordinates (io.AddMousePosEvent space, same as
    // SDL mouse event coordinates) — they are independent of the display DPI
    // scale, so a script recorded at 100% still hits the same widgets at 150%.
    std::string inputFile_;
    uint64_t inputFileLine_ = 0; // lines consumed so far
    // engine.toml hot-reload watch (path captured at init; ~1 s stat interval)
    // + auto-save state (debounced write of the canonical serialization).
    std::string engineCfgPath_;
    int64_t engineCfgMtime_ = 0;
    std::chrono::steady_clock::time_point engineCfgNextPoll_{};
    std::string engineCfgBaseline_;     // canonical toml of the saved/loaded state
    std::string engineCfgPendingText_;  // dirty snapshot awaiting the debounce
    std::chrono::steady_clock::time_point engineCfgDirtyAt_{};
    bool engineCfgDirty_ = false;
    std::string lutCarry_; // [grading] lut passthrough (viewer-only key the GUI
                           // does not edit but must not clobber on auto-save)
    bool benchAutoRun_ = false;
    bool benchStarted_ = false;
    // Quit deadline after an auto bench finishes (time-based: ImGui-only
    // frames are unthrottled, so a frame countdown would expire instantly).
    std::chrono::steady_clock::time_point benchQuitAt_{};
    bool benchQuitArmed_ = false;
};

} // namespace sr
