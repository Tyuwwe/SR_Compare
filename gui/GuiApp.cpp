// ============================================================================
// GuiApp — see GuiApp.h for the design overview.
// ============================================================================
#include "gui/GuiApp.h"

#include "app/CliUtils.h"
#include "compare/Font5x7.h"
#include "renderer/Screenshot.h"
#include "renderer/core/MemoryBudget.h"
#include "renderer/core/PathUtil.h"
#include "renderer/core/VkUtil.h"
#include "renderer/core/Vma.h"
#include "renderer/scene/SceneRegistry.h"
#include "upscalers/UpscalerFactory.h"
#include "upscalers/dlss/SlContext.h"

#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>
#include <imgui_internal.h> // GImGui->InputEventsQueue (input-path debug)

#include <SDL3/SDL.h> // SDL_Event/SDL_EVENT_* for the automation debug hook

#include <algorithm>
#include <cctype>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

namespace {
// Debug hook (SR_GUI_DEBUG_INPUT=1): verify the backend actually queues a
// wheel event when the event arrives.
bool dbgInputEnabled() {
    static const bool enabled = sr::envFlag("SR_GUI_DEBUG_INPUT");
    return enabled;
}

bool guiSdlEventHook(const SDL_Event& event) {
    const int queueBefore = GImGui ? GImGui->InputEventsQueue.Size : -1;
    const bool r = ImGui_ImplSDL3_ProcessEvent(&event);
    if (dbgInputEnabled() &&
        (event.type == SDL_EVENT_MOUSE_WHEEL ||
         event.type == SDL_EVENT_MOUSE_BUTTON_DOWN)) {
        const ImGuiIO& io = ImGui::GetIO();
        std::fprintf(stderr, "[hook] type=0x%04x consumed=%d queue=%d->%d wheelNow=%.3f capture=%d\n",
                     event.type, r ? 1 : 0, queueBefore,
                     GImGui->InputEventsQueue.Size, static_cast<double>(io.MouseWheel),
                     io.WantCaptureMouse ? 1 : 0);
    }
    return r;
}
} // namespace

namespace sr {

namespace {

constexpr VkFormat kComposeFormat = VK_FORMAT_R8G8B8A8_UNORM;

const uint32_t kOutputResolutions[4][2] = {
    {1280, 720}, {1920, 1080}, {2560, 1440}, {3840, 2160}};
const char* kOutputResNames[4] = {"1280x720", "1920x1080", "2560x1440", "3840x2160"};

struct AlgoGroup {
    const char* title;
    const char* ids[4];
    int count;
};
constexpr AlgoGroup kAlgoGroups[] = {
    {"Basic", {"taa"}, 1},
    {"NVIDIA", {"dlss-k", "dlss-m", "dlss-l"}, 3},
    {"AMD", {"fsr3", "fsr2", "fsr1"}, 3},
    {"Intel", {"xess"}, 1},
    {"ARM", {"nss"}, 1},
    {"Qualcomm", {"sgsr2", "sgsr1"}, 2},
};
constexpr int kFgCount = 3;
const char* kFgLabels[kFgCount] = {"none", "FSR3", "NFRU"};
const char* kFgIds[kFgCount] = {"", "fsr3", "nfru"};

Camera lerpCamera(const Camera& a, const Camera& b, float t) {
    Camera c = b;
    c.position = a.position + (b.position - a.position) * t;
    c.forward = normalize(a.forward + (b.forward - a.forward) * t);
    const Vec3 up = a.up + (b.up - a.up) * t;
    if (length(up) > 1e-8f) c.up = normalize(up);
    return c;
}

// SceneUBO / MaterialUBO / LightingUBO / ScenePush and the GBuffer formats are
// shared with the viewer renderer via renderer/deferred/DeferredCore.h.

// Compose pass push constants: column pixel size + text scale + text slot,
// the source-region window (normalized offset/size) and source dimensions,
// plus the terminal lens-effects chain (Phase 6a; same algorithm/defaults as
// the viewer present.frag — lens dirt excluded, the compose has no dirt
// binding) and the log-domain grading set (Phase 6c; identical for every
// column, sliders in the shared controls).
struct ComposePush {
    float colSize[2];
    float textScale;
    float textSlot;
    float uvRect[4];  // normalized source region: offset xy, size zw
    float srcSize[2]; // source image pixels
    float nearest;    // != 0: sample nearest (magnification >= 1:1)
    float exposure;   // display-domain ACES input multiplier
    float lensA[4];   // x = chromatic aberration, y = vignette, z = film grain,
                      // w = frame index (grain hash seed)
    float gradeA[4];  // x = contrast, y = saturation (zw unused; SDR composite)
    float gradeB[4];  // xyz = white balance, w = LUT size
};
static_assert(sizeof(ComposePush) == 96, "ComposePush size mismatch");

// Metric compute push constants (two uvec4s in the shaders).
struct MetricPush {
    uint32_t x = 0, y = 0, z = 0, w = 0;      // region offset xy, extent zw
    uint32_t x2 = 0, y2 = 0, z2 = 0, w2 = 0;  // x2 = blocks per row (region);
                                              // y2 = 1 => ref is low-res (normalized sampling);
                                              // z2/w2 = test image full size (px)
    float exposureTest = 1.f; // test image (upscaler column / LR path)
    float exposureRef = 1.f;  // reference image (GT column / GT path)
    float pad[2] = {};
};
static_assert(sizeof(MetricPush) == 48, "MetricPush std140/push size mismatch");

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

// Compute-shader write -> read barrier for the metric SSBO chain.
void computeBarrier(VkCommandBuffer cmd, VkAccessFlags dstAccess, VkPipelineStageFlags dstStage) {
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, dstStage, 0, 1, &barrier, 0,
                         nullptr, 0, nullptr);
}

char asciiUpper(char c) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
}

std::string selfExePath() {
    char buf[MAX_PATH] = {0};
    const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    return (n > 0 && n < MAX_PATH) ? std::string(buf, n) : std::string();
}

std::vector<std::string> splitCsvLines(const std::string& text) {
    std::vector<std::string> lines;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t nl = text.find('\n', start);
        std::string line = text.substr(start, nl == std::string::npos ? std::string::npos
                                                                      : nl - start);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) lines.push_back(line);
        if (nl == std::string::npos) break;
        start = nl + 1;
    }
    return lines;
}

std::vector<std::string> splitComma(const std::string& line) {
    std::vector<std::string> cols;
    size_t start = 0;
    while (start <= line.size()) {
        const size_t comma = line.find(',', start);
        cols.push_back(line.substr(start, comma == std::string::npos ? std::string::npos
                                                                     : comma - start));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return cols;
}

// Fallback material/texture for scenes that provide none (shared by the sync
// initial build and the async load worker; pool selects the upload pool).
bool ensureSceneFallbacks(Scene& scene, const VulkanContext& ctx, VkCommandPool pool) {
    if (scene.materials.empty()) {
        Material fallback;
        fallback.baseColor = {0.8f, 0.8f, 0.8f, 1.f};
        fallback.roughness = 0.6f;
        scene.materials.push_back(fallback);
    }
    if (scene.textures.empty()) {
        const uint8_t white[4] = {255, 255, 255, 255};
        Texture dummy;
        if (!scene.uploadTexture(ctx, 1, 1, white, dummy, true, pool)) return false;
        scene.textures.push_back(dummy);
    }
    return true;
}

} // namespace

void GuiApp::ImageResource::destroy(const VulkanContext& ctx) {
    if (view) { vkDestroyImageView(ctx.device, view, nullptr); view = VK_NULL_HANDLE; }
    if (image) { vmaDestroyImage(ctx.allocator, image, memory); image = VK_NULL_HANDLE; memory = VK_NULL_HANDLE; }
}

void GuiApp::computeViewRegion(uint32_t srcW, uint32_t srcH, uint32_t colW, uint32_t colH,
                               float zoom, float centerU, float centerV, float out[4]) {
    // Fill the column while keeping the source aspect: scale to cover, crop
    // the overflow; the zoom window shrinks the visible region around the
    // requested center (clamped to the image).
    const float s = std::max(static_cast<float>(colW) / static_cast<float>(srcW),
                             static_cast<float>(colH) / static_cast<float>(srcH));
    float sizeX = std::min(static_cast<float>(colW) / s, static_cast<float>(srcW));
    float sizeY = std::min(static_cast<float>(colH) / s, static_cast<float>(srcH));
    zoom = std::max(1.f, zoom);
    sizeX /= zoom;
    sizeY /= zoom;
    const float maxOffX = std::max(static_cast<float>(srcW) - sizeX, 0.f);
    const float maxOffY = std::max(static_cast<float>(srcH) - sizeY, 0.f);
    out[0] = std::clamp(centerU * static_cast<float>(srcW) - sizeX * 0.5f, 0.f, maxOffX);
    out[1] = std::clamp(centerV * static_cast<float>(srcH) - sizeY * 0.5f, 0.f, maxOffY);
    out[2] = sizeX;
    out[3] = sizeY;
}

// ---------------------------------------------------------------------------
// Core init: window -> Vulkan device (plugin gates already forced by main) ->
// swapchain -> ImGui -> initial render stack.
// ---------------------------------------------------------------------------
bool GuiApp::init(const GuiOptions& opts) {
    opts_ = opts;
    // engine.toml watch state (the file was already parsed by main into
    // opts_.engineCfg; the GUI re-stats it for hot reload in run()).
    engineCfgPath_ = engineConfigPath();
    engineCfgMtime_ = engineConfigWriteTime(engineCfgPath_);
    // CLI --exposure is a manual override: start in manual exposure mode.
    if (opts_.exposure > 0.f) autoExposureEnabled_ = false;
    // _dupenv_s / envFlag: plain getenv trips C4996 (this project builds /W4).
#ifdef _MSC_VER
    {
        char* f = nullptr;
        size_t fLen = 0;
        if (_dupenv_s(&f, &fLen, "SR_GUI_INPUT_FILE") == 0 && f != nullptr) {
            inputFile_ = f;
            std::free(f);
        }
    }
#else
    if (const char* f = std::getenv("SR_GUI_INPUT_FILE")) inputFile_ = f;
#endif
    uiShot_ = envFlag("SR_GUI_UI_SHOT");
    scenes_ = listScenes();
    upscalerNames_ = listUpscalers();
    applyLaunchOptions();

    active_.displayW = kOutputResolutions[ui_.outputResIndex][0];
    active_.displayH = kOutputResolutions[ui_.outputResIndex][1];

    // Windowed client size: the remembered [window] width/height wins over
    // the output resolution (the swapchain follows the actual window size).
    windowedW_ = static_cast<int>(active_.displayW);
    windowedH_ = static_cast<int>(active_.displayH);
    if (opts_.engineCfg.width && opts_.engineCfg.height) {
        windowedW_ = *opts_.engineCfg.width;
        windowedH_ = *opts_.engineCfg.height;
    }
    if (!window_.create("sr_compare — gui", windowedW_, windowedH_))
        return false;
    window_.setClickToCaptureEnabled(false); // LMB belongs to the UI now
    // engine.toml [window] fullscreen (no CLI flag): borderless desktop mode.
    if (opts_.engineCfg.fullscreen) setFullscreenEnabled(*opts_.engineCfg.fullscreen);
    if (!ctx_.create(window_)) return false;

    // Hold one Streamline reference for the whole GUI session: slInit already
    // ran inside ctx creation (allPluginsEnabled gate), and per-instance
    // addRef/release pairs would otherwise slShutdown/slInit on every Apply.
    if (sl_dlss::initialized()) {
        sl_dlss::addRef();
        slRefHeld_ = true;
    }

    refreshUpscalerAvailability();

    // HDR surface probe (Phase 6c): gates the UI checkbox; the default stays
    // SDR.  The checkbox handler re-creates the swapchain on toggle.
    Swapchain::queryHdrSupport(ctx_, hdrSupportHdr10_, hdrSupportScRgb_);
    // engine.toml hdr default (no gui CLI flag for it): gated on the support
    // probe, same rule as the UI checkbox.
    if (opts_.engineCfg.hdr && (opts_.engineCfgCli & cli::kHdr) == 0)
        hdrEnabled_ = *opts_.engineCfg.hdr && (hdrSupportHdr10_ || hdrSupportScRgb_);
    swapchainVsync_ = guiWantVsync();
    swapchainMailbox_ = guiAllowMailbox();
    // The swapchain extent follows the actual window client size in PHYSICAL
    // PIXELS (logical size would under-size the surface on scaled displays,
    // producing a blurry upscale); the composite scales to it in the present
    // copy pass.
    if (!swapchain_.create(ctx_, static_cast<uint32_t>(window_.pixelWidth()),
                           static_cast<uint32_t>(window_.pixelHeight()), swapchainVsync_,
                           swapchainMailbox_, hdrEnabled_))
        return false;
    // Grading LUT (Phase 6c): procedural identity, created once — it survives
    // render-stack rebuilds (independent of scene/env/resolution).
    if (!gradingLut_.create(ctx_, makeIdentityLut())) return false;
    if (!createUiSync()) return false;
    if (!initImGui()) return false;
    window_.setEventHook(&guiSdlEventHook);

    // Default selection: taa for the viewer, taa+fsr2 for compare (launch
    // options may have preset these already).
    for (uint32_t i = 0; i < upscalerNames_.size() && i < kMaxRegistered; ++i) {
        if (upscalerNames_[i] == "taa") {
            if (opts_.upscalerName.empty()) ui_.viewerUpscaler = static_cast<int>(i) + 1;
            if (opts_.benchList.empty()) ui_.benchSelected[i] = true;
        }
    }
    if (opts_.compareList.empty() && ui_.compareSlots.empty()) {
        const int taa = upscalerIndexById("taa");
        const int fsr2 = upscalerIndexById("fsr2");
        if (taa >= 0) ui_.compareSlots.push_back({taa, 0});
        if (fsr2 >= 0) ui_.compareSlots.push_back({fsr2, 0});
    }

    active_ = configFromUi(currentTab_ == 1 ? Mode::Compare : Mode::Viewer);
    // IBL maps + deferred pipelines are built once per env map, not per Apply.
    // Empty env map = sky atmosphere, rendered for the current UI sun.
    envMapActive_ = active_.envMapPath;
    stackOk_ = deferred_.init(ctx_, envMapActive_.c_str(),
                              sunDirectionFromElevAzimuth(sunElevationDeg_, sunAzimuthDeg_));
    if (stackOk_) {
        // CSM shadow targets: resolution-independent, so they live next to
        // deferred_ and survive scene/config rebuilds.  A failure degrades to
        // no shadows (bindings stay unwritten, sampling stays off).
        shadowsActive_ = deferred_.createShadowTargets(ctx_, shadow_);
        if (!shadowsActive_)
            std::fprintf(stderr, "gui: shadow target creation failed, shadows disabled\n");
        // Spot shadow atlas (Phase 4b); a failure degrades to sun-only shadows.
        spotAtlasActive_ = deferred_.createShadowAtlas(ctx_, spotAtlas_);
        if (!spotAtlasActive_)
            std::fprintf(stderr, "gui: spot shadow atlas creation failed, spot shadows disabled\n");
        // Start both shadow maps in SHADER_READ_ONLY so the descriptor
        // bindings are layout-valid even on frames that render no shadows
        // (shadows checkbox off, or a scene without casting lights).
        submitOneShot(ctx_, [&](VkCommandBuffer cmd) {
            if (shadowsActive_) {
                imageBarrier(cmd, shadow_.image, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_NONE,
                             VK_ACCESS_2_NONE, sync::kFragment, sync::kSampled,
                             VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, kShadowCascadeCount);
                shadow_.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
            if (spotAtlasActive_) {
                imageBarrier(cmd, spotAtlas_.image, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_NONE,
                             VK_ACCESS_2_NONE, sync::kFragment, sync::kSampled,
                             VK_IMAGE_ASPECT_DEPTH_BIT);
                spotAtlas_.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
        });
        stackOk_ = buildRenderStack();
    } else {
        statusLine_ = "deferred core init failed";
    }
    if (!stackOk_) {
        std::fprintf(stderr, "gui: initial render stack failed: %s\n", statusLine_.c_str());
        // Continue anyway: the UI stays up (ImGui-only frames) and shows the
        // error; Apply retries.
    }
    // engine.toml per-frame defaults: applied AFTER the initial stack build so
    // they win over the scene lighting preset (buildRenderStack re-asserts the
    // preset on every scene rebuild, same as the manual "reset" UI button).
    // The manual-exposure value additionally rides opts_.exposure (re-applied
    // on rebuild), matching CLI --exposure.
    if (!engineCfgPath_.empty()) {
        EngineConfigLog log;
        applyEngineConfigHot(opts_.engineCfg, log);
        log.flush(" gui:");
    }
    // [grading] lut is a viewer-only key the GUI cannot edit; carry the loaded
    // value through so the auto-save does not clobber it.
    lutCarry_ = opts_.engineCfg.lut.value_or("");
    // engine.toml auto-create: the GUI owns the file — a missing one is
    // written from the effective values (code defaults + CLI overrides, scene
    // preset already applied by the initial stack build above) so there is
    // always a file to edit; the debounced auto-save keeps it in sync.
    engineCfgBaseline_ = engineConfigToToml(currentEngineConfig());
    if (engineCfgPath_.empty()) {
        const std::string path = engineConfigFilePath();
        if (writeFileAtomic(path, engineCfgBaseline_)) {
            engineCfgPath_ = path;
            engineCfgMtime_ = engineConfigWriteTime(path);
            statusLine_ = "engine.toml created (defaults)";
            std::fprintf(stderr, "[engine.toml] created %s\n", path.c_str());
        } // else: no write target — hot reload keeps watching for the file
    }
    if (opts_.compareZoom > 0.f) {
        // Applied on the first run() frame (after any spurious first-frame
        // rebuild, which would reset compareZoom_ back to 1).
        launchZoomPending_ = true;
    }
    return true;
}

void GuiApp::refreshUpscalerAvailability() {
    upscalerLabels_.clear();
    upscalerAvailable_.clear();
    const VulkanEnv env = ctx_.toEnv();
    for (const std::string& n : upscalerNames_) {
        std::unique_ptr<IUpscaler> probe = createUpscaler(n.c_str());
        bool available = false;
        std::string label = n;
        if (probe) {
            available = probe->isAvailable(env);
            label = probe->name();
        }
        upscalerLabels_.push_back(label + (available ? "" : " [unavailable]"));
        upscalerAvailable_.push_back(available ? 1 : 0);
    }
}

bool GuiApp::initImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.IniFilename = nullptr; // no .ini state file next to the exe

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 4.f;
    style.FrameRounding = 3.f;
    // High-DPI: scale the whole UI by the monitor content scale (Windows
    // display scaling 125%/150%/...).  Style sizes first, then FontScaleMain —
    // dynamic fonts (1.92+) rasterize at the scaled size, staying crisp.
    applyUiScale(window_.contentScale());

    if (!ImGui_ImplSDL3_InitForVulkan(window_.sdlWindow())) return false;
    return initImGuiVulkanBackend();
}

void GuiApp::applyUiScale(float scale) {
    if (scale <= 0.f) scale = 1.f;
    if (scale == uiScale_) return;
    ImGuiStyle& style = ImGui::GetStyle();
    // ScaleAllSizes multiplies the CURRENT values, so pass the ratio against
    // the previously applied scale (from the initial 1.0 style at init).
    style.ScaleAllSizes(scale / uiScale_);
    style.FontScaleMain = scale;
    uiScale_ = scale;
    std::fprintf(stderr, "gui: display content scale %.2f -> UI scaled\n",
                 static_cast<double>(scale));
}

// Vulkan backend (device objects + pipeline) — the pipeline bakes the
// swapchain format, so setHdrEnabled shuts it down and re-runs this after a
// swapchain re-creation (the ImGui context and the SDL3 backend persist).
bool GuiApp::initImGuiVulkanBackend() {
    ImGui_ImplVulkan_InitInfo ii = {};
    ii.ApiVersion = VK_API_VERSION_1_3;
    ii.Instance = ctx_.instance;
    ii.PhysicalDevice = ctx_.physicalDevice;
    ii.Device = ctx_.device;
    ii.QueueFamily = ctx_.graphicsQueueFamily;
    ii.Queue = ctx_.graphicsQueue;
    ii.DescriptorPoolSize = 16; // let the backend create its own pool
    ii.MinImageCount = 2;
    ii.ImageCount = swapchain_.imageCount();
    ii.UseDynamicRendering = true;
    ii.PipelineInfoMain.PipelineRenderingCreateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    ii.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    const VkFormat presentFormat = swapchain_.format();
    ii.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &presentFormat;
    return ImGui_ImplVulkan_Init(&ii);
}

void GuiApp::shutdownImGui() {
    graphWindow_.destroy(); // ImNodes context (independent of the ImGui one)
    window_.setEventHook(nullptr);
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
}

void GuiApp::shutdown() {
    // An async load may still be running: wait for it before touching the
    // device (the worker submits uploads through ctx_.graphicsQueue).
    if (loadThread_.joinable()) loadThread_.join();
    loadPhase_.store(LoadPhase::Idle, std::memory_order_release);
    if (ctx_.device) vkDeviceWaitIdle(ctx_.device);
    if (loadPool_) {
        vkDestroyCommandPool(ctx_.device, loadPool_, nullptr);
        loadPool_ = VK_NULL_HANDLE;
    }
    discardLoadResult();
    bench_.stop();
    destroyRenderStack();
    if (shadowsActive_) { deferred_.destroyShadowTargets(ctx_, shadow_); shadowsActive_ = false; }
    if (spotAtlasActive_) { deferred_.destroyShadowAtlas(ctx_, spotAtlas_); spotAtlasActive_ = false; }
    deferred_.destroy(ctx_);
    destroyUiSync();
    shutdownImGui();
    // Balance the session SL reference while the device is still alive.
    if (slRefHeld_) {
        sl_dlss::release();
        slRefHeld_ = false;
    }
    swapchain_.destroy(ctx_);
    gradingLut_.destroy(ctx_);
    ctx_.destroy();
    window_.destroy();
}

GuiApp::RenderConfig GuiApp::configFromUi(Mode mode) const {
    RenderConfig cfg;
    cfg.mode = mode;
    cfg.renderScale = ui_.renderScale;
    cfg.displayW = kOutputResolutions[ui_.outputResIndex][0];
    cfg.displayH = kOutputResolutions[ui_.outputResIndex][1];

    if (ui_.customScene[0] != '\0') {
        cfg.scenePath = ui_.customScene;
    } else if (ui_.sceneIndex >= 0 &&
               ui_.sceneIndex < static_cast<int>(scenes_.size())) {
        cfg.scenePath = scenes_[static_cast<size_t>(ui_.sceneIndex)].path;
    }

    if (mode == Mode::Viewer) {
        if (ui_.viewerUpscaler > 0 &&
            ui_.viewerUpscaler <= static_cast<int>(upscalerNames_.size())) {
            RenderConfig::AlgoSpec spec;
            spec.sr = upscalerNames_[static_cast<size_t>(ui_.viewerUpscaler - 1)];
            spec.fg = ui_.viewerFg > 0 ? fgId(ui_.viewerFg) : "";
            cfg.algos.push_back(std::move(spec));
        }
    } else {
        for (const UiState::CompareSlot& slot : ui_.compareSlots) {
            if (cfg.algos.size() >= kMaxAlgos) break;
            if (slot.sr < 0 || slot.sr >= static_cast<int>(upscalerNames_.size())) continue;
            RenderConfig::AlgoSpec spec;
            spec.sr = upscalerNames_[static_cast<size_t>(slot.sr)];
            spec.fg = slot.fg > 0 ? fgId(slot.fg) : "";
            cfg.algos.push_back(std::move(spec));
        }
    }
    // The GT reference mode is global: it applies to the compare GT column
    // and to the viewer's native (GT) mode alike.  The two modes are mutually
    // exclusive; if both are somehow set, Apply scale takes precedence.
    cfg.gtApplyScale = ui_.compareGtApplyScale;
    cfg.gtSsaa = ui_.compareGtSsaa && !ui_.compareGtApplyScale;
    cfg.envMapPath = ui_.envMap;
    return cfg;
}

void GuiApp::applyLaunchOptions() {
    // Empty env map field = procedural sky atmosphere (the default look).
    std::snprintf(ui_.envMap, sizeof(ui_.envMap), "%s", opts_.envMapPath.c_str());
    if (opts_.renderScale > 0.f) ui_.renderScale = opts_.renderScale;
    if (opts_.displayW > 0 && opts_.displayH > 0) {
        for (int i = 0; i < 4; ++i) {
            if (kOutputResolutions[i][0] == opts_.displayW &&
                kOutputResolutions[i][1] == opts_.displayH)
                ui_.outputResIndex = i;
        }
    }
    if (!opts_.sceneArg.empty()) {
        bool matched = false;
        for (size_t i = 0; i < scenes_.size(); ++i) {
            if (scenes_[i].alias == opts_.sceneArg) {
                ui_.sceneIndex = static_cast<int>(i);
                matched = true;
                break;
            }
        }
        if (!matched)
            std::snprintf(ui_.customScene, sizeof(ui_.customScene), "%s", opts_.sceneArg.c_str());
    }
    if (!opts_.upscalerName.empty()) {
        if (opts_.upscalerName == "none" || opts_.upscalerName == "native") {
            ui_.viewerUpscaler = 0;
        } else {
            for (size_t i = 0; i < upscalerNames_.size(); ++i) {
                if (upscalerNames_[i] == opts_.upscalerName)
                    ui_.viewerUpscaler = static_cast<int>(i) + 1;
            }
        }
    }
    if (opts_.compareGtSsaa) ui_.compareGtSsaa = true;
    if (!opts_.compareList.empty()) {
        ui_.compareSlots.clear();
        for (const std::string& name : splitComma(opts_.compareList)) {
            const int idx = upscalerIndexById(name);
            if (idx >= 0 && ui_.compareSlots.size() < kMaxAlgos)
                ui_.compareSlots.push_back({idx, 0});
        }
        currentTab_ = 1;
        tabRequest_ = 1;
    }
    if (!opts_.benchList.empty()) {
        ui_.benchNative = false;
        for (const std::string& name : splitComma(opts_.benchList)) {
            if (name == "native") {
                ui_.benchNative = true;
                continue;
            }
            for (size_t i = 0; i < upscalerNames_.size() && i < kMaxRegistered; ++i) {
                if (upscalerNames_[i] == name) ui_.benchSelected[i] = true;
            }
        }
        currentTab_ = 2;
        tabRequest_ = 2;
        benchAutoRun_ = true;
    }
}

// engine.toml per-frame application (hot reload + initial defaults): only
// options with no rebuild cost are touched (pass toggles go through
// applyPassToggle for the checkbox side effects like the fog history reset).
// Resolution/scale/scene/env-map/LUT changes still need Apply or a restart.
void GuiApp::applyEngineConfigHot(const EngineConfig& cfg, EngineConfigLog& log) {
    const uint64_t m = opts_.engineCfgCli;
    auto takeToggle = [&](rg::PassToggle t, const std::optional<bool>& v, uint64_t bit,
                          const char* key) {
        if (v && (m & bit) == 0 && *v != passToggleValue(t)) {
            applyPassToggle(t, *v);
            log.add(key, *v);
        }
    };
    takeToggle(rg::PassToggle::Shadows, cfg.shadows, cli::kShadows, "shadows");
    takeToggle(rg::PassToggle::ContactShadows, cfg.contactShadows, cli::kContactShadows,
               "contact_shadows");
    takeToggle(rg::PassToggle::Ssr, cfg.ssr, cli::kSsr, "ssr");
    takeToggle(rg::PassToggle::VolFog, cfg.volFog, cli::kVolFog, "volfog");
    takeToggle(rg::PassToggle::Occlusion, cfg.occlusion, cli::kNone, "occlusion");
    takeToggle(rg::PassToggle::Bloom, cfg.bloom, cli::kBloom, "bloom");
    takeToggle(rg::PassToggle::MotionBlur, cfg.motionBlur, cli::kMotionBlur, "motion_blur");
    takeToggle(rg::PassToggle::Dof, cfg.dof, cli::kDof, "dof");
    takeToggle(rg::PassToggle::LensCa, cfg.lensCa, cli::kLensFx, "lens_ca");
    takeToggle(rg::PassToggle::LensVignette, cfg.lensVignette, cli::kLensFx, "lens_vignette");
    takeToggle(rg::PassToggle::LensGrain, cfg.lensGrain, cli::kLensFx, "lens_grain");
    cfgTake(ssrStrength_, cfg.ssrStrength, m, cli::kSsrStrength, "ssr_strength", log);
    cfgTake(dofFocus_, cfg.dofFocus, m, cli::kDofFocus, "dof_focus", log);
    cfgTake(dofFstop_, cfg.dofFstop, m, cli::kDofFstop, "dof_fstop", log);
    cfgTake(dofMaxBlur_, cfg.dofMaxBlur, m, cli::kDofMaxBlur, "dof_max_blur", log);
    cfgTake(lodEnabled_, cfg.lod, m, cli::kNone, "lod", log);
    cfgTake(gradeTemperatureK_, cfg.gradingTempK, m, cli::kGradingTemp, "temperature", log);
    cfgTake(gradeTint_, cfg.gradingTint, m, cli::kGradingTint, "tint", log);
    cfgTake(gradeContrast_, cfg.gradingContrast, m, cli::kGradingContrast, "contrast", log);
    cfgTake(gradeSaturation_, cfg.gradingSat, m, cli::kGradingSat, "saturation", log);
    cfgTake(exposureMinEV_, cfg.exposureMinEV, m, cli::kExposure, "exposure_min_ev", log);
    cfgTake(exposureMaxEV_, cfg.exposureMaxEV, m, cli::kExposure, "exposure_max_ev", log);
    if (cfg.exposure && (m & cli::kExposure) == 0) {
        // Same semantics as CLI --exposure: manual value, auto exposure off.
        autoExposureEnabled_ = false;
        exposure_ = *cfg.exposure;
        log.add("exposure", exposure_);
    }
    bool sunMoved = false;
    sunMoved |= cfgTake(sunElevationDeg_, cfg.sunElevationDeg, m, cli::kSunElev, "sun_elev", log);
    sunMoved |= cfgTake(sunAzimuthDeg_, cfg.sunAzimuthDeg, m, cli::kSunAz, "sun_az", log);
    if (sunMoved) updateSkyFromUiSun();
    if (cfg.hdr && (m & cli::kHdr) == 0 && *cfg.hdr != hdrEnabled_) {
        if (*cfg.hdr && !hdrSupportHdr10_ && !hdrSupportScRgb_) {
            std::fprintf(stderr, "[engine.toml] hdr=true ignored: no HDR surface support\n");
        } else {
            setHdrEnabled(*cfg.hdr);
            log.add("hdr", hdrEnabled_);
        }
    }
    // Borderless fullscreen (no CLI flag): window-state toggle, applied hot.
    if (cfg.fullscreen && *cfg.fullscreen != fullscreenEnabled_) {
        setFullscreenEnabled(*cfg.fullscreen);
        log.add("fullscreen", fullscreenEnabled_);
    }
    // Windowed size memory (no CLI flag): applied hot while windowed.  While
    // fullscreen the values are kept for the round-trip but not applied to
    // the (desktop-sized) window.
    if (cfg.width && cfg.height &&
        (*cfg.width != windowedW_ || *cfg.height != windowedH_)) {
        windowedW_ = *cfg.width;
        windowedH_ = *cfg.height;
        if (!fullscreenEnabled_) window_.setClientSize(windowedW_, windowedH_);
        log.add("width", windowedW_);
        log.add("height", windowedH_);
    }
}

// Hot-reload watch: stats engine.toml about once a second; on a modification
// the per-frame options are re-applied (CLI-masked keys stay untouched).  A
// deleted file resets the hot parameters to the code/scene-preset defaults
// and is re-created (the GUI owns the file); the re-create write updates the
// baseline, so the follow-up self-read is a no-op (idempotent).
void GuiApp::pollEngineConfig() {
    const auto now = std::chrono::steady_clock::now();
    if (now < engineCfgNextPoll_) return;
    engineCfgNextPoll_ = now + std::chrono::seconds(1);
    if (engineCfgPath_.empty()) {
        // No config at startup: keep watching so a newly created file applies.
        engineCfgPath_ = engineConfigPath();
        if (engineCfgPath_.empty()) return;
        engineCfgMtime_ = 0;
    }
    const int64_t mt = engineConfigWriteTime(engineCfgPath_);
    if (mt == engineCfgMtime_) return;
    engineCfgMtime_ = mt;
    if (mt == 0) {
        // Deleted at runtime: defaults back in, then rebuild the file from
        // them (immediate write — no debounce on the recovery path).
        resetEngineConfigDefaults();
        engineCfgBaseline_ = engineConfigToToml(currentEngineConfig());
        engineCfgPendingText_ = engineCfgBaseline_;
        engineCfgDirty_ = false;
        if (writeFileAtomic(engineCfgPath_, engineCfgPendingText_)) {
            engineCfgMtime_ = engineConfigWriteTime(engineCfgPath_);
            statusLine_ = "engine.toml deleted — defaults restored, file re-created";
            std::fprintf(stderr, "[engine.toml] deleted — defaults restored, re-created %s\n",
                         engineCfgPath_.c_str());
        }
        return;
    }
    EngineConfig cfg;
    if (!loadEngineConfig(engineCfgPath_, cfg)) return; // malformed: keep state
    EngineConfigLog log;
    applyEngineConfigHot(cfg, log);
    // Needs-Apply keys are not applied hot but DO update the UI fields (same
    // as the user moving the slider), so the auto-save never clobbers an
    // external edit with a stale value.
    if (cfg.renderScale) ui_.renderScale = *cfg.renderScale;
    if (cfg.envMap)
        std::snprintf(ui_.envMap, sizeof(ui_.envMap), "%s", cfg.envMap->c_str());
    if (cfg.lut) lutCarry_ = *cfg.lut; // preserve the viewer-only key across saves
    // The external edit becomes the save baseline: the auto-save reacts to
    // UI-side changes only, it does not rewrite the file the user just edited.
    engineCfgBaseline_ = engineConfigToToml(currentEngineConfig());
    engineCfgDirty_ = false;
    if (!log.empty()) statusLine_ = "engine.toml reloaded";
    log.flush(" gui (reload):");
}

// Full file snapshot from the current UI state (every key the GUI writes).
// [exposure] exposure is omitted while auto exposure is on — writing it would
// re-disable auto on the next load (same semantics as CLI --exposure).
EngineConfig GuiApp::currentEngineConfig() const {
    EngineConfig c;
    c.fullscreen = fullscreenEnabled_;
    if (windowedW_ > 0 && windowedH_ > 0) {
        c.width = windowedW_;
        c.height = windowedH_;
    }
    c.renderScale = ui_.renderScale;
    c.hdr = hdrEnabled_;
    c.envMap = std::string(ui_.envMap);
    if (!autoExposureEnabled_) c.exposure = exposure_;
    c.exposureMinEV = exposureMinEV_;
    c.exposureMaxEV = exposureMaxEV_;
    c.ssr = ssrEnabled_;
    c.ssrStrength = ssrStrength_;
    c.shadows = shadowsEnabled_;
    c.contactShadows = contactShadowsEnabled_;
    c.volFog = volFogEnabled_;
    c.bloom = bloomEnabled_;
    c.motionBlur = motionBlurEnabled_;
    // The GUI has no master lens-fx switch (only the per-effect items); write
    // the master as "any on" so the viewer keeps its chain when at least one
    // effect is enabled.
    c.lensFx = lensCaEnabled_ || lensVignetteEnabled_ || lensGrainEnabled_;
    c.lensCa = lensCaEnabled_;
    c.lensVignette = lensVignetteEnabled_;
    c.lensGrain = lensGrainEnabled_;
    c.occlusion = occlusionEnabled_;
    c.lod = lodEnabled_;
    c.dof = dofEnabled_;
    c.dofFocus = dofFocus_;
    c.dofFstop = dofFstop_;
    c.dofMaxBlur = dofMaxBlur_;
    c.gradingTempK = gradeTemperatureK_;
    c.gradingTint = gradeTint_;
    c.gradingContrast = gradeContrast_;
    c.gradingSat = gradeSaturation_;
    c.lut = lutCarry_;
    c.sunElevationDeg = sunElevationDeg_;
    c.sunAzimuthDeg = sunAzimuthDeg_;
    return c;
}

void GuiApp::saveEngineConfigNow() {
    engineCfgDirty_ = false;
    if (engineCfgPath_.empty()) return; // no write target (creation failed)
    if (writeFileAtomic(engineCfgPath_, engineCfgPendingText_)) {
        engineCfgMtime_ = engineConfigWriteTime(engineCfgPath_);
        statusLine_ = "engine.toml saved";
    }
}

// Debounced auto-save: once per frame, diff the canonical serialization of
// the UI state against the saved/loaded baseline; write 1.5 s after the last
// change (atomic temp+replace, so the hot-reload poll never sees a stub).
void GuiApp::pollEngineConfigAutoSave() {
    if (engineCfgPath_.empty()) return;
    const std::string cur = engineConfigToToml(currentEngineConfig());
    const auto now = std::chrono::steady_clock::now();
    if (cur != engineCfgBaseline_) {
        engineCfgBaseline_ = cur;
        engineCfgPendingText_ = cur;
        engineCfgDirty_ = true;
        engineCfgDirtyAt_ = now; // sliding window: edits keep pushing the write out
    }
    if (engineCfgDirty_ && now - engineCfgDirtyAt_ >= std::chrono::milliseconds(1500))
        saveEngineConfigNow();
}

// Inverse of applyEngineConfigHot with no file: every hot-applied parameter
// back to its code default (GUI member initializer values / scene preset).
// Toggles go through applyPassToggle for the same side effects as the panel
// checkboxes (fog history reset, auto-exposure snap).
void GuiApp::resetEngineConfigDefaults() {
    applyLightingPreset(lightingPresetForScene(active_.scenePath)); // sun/exposure/fog preset
    applyPassToggle(rg::PassToggle::AutoExposure, true);
    exposureMinEV_ = -8.f;
    exposureMaxEV_ = 8.f;
    applyPassToggle(rg::PassToggle::Shadows, true);
    applyPassToggle(rg::PassToggle::ContactShadows, true);
    applyPassToggle(rg::PassToggle::Ssr, false);
    applyPassToggle(rg::PassToggle::VolFog, volFogEnabled_); // preset value + history reset
    applyPassToggle(rg::PassToggle::Occlusion, occlusionEnabledByDefault());
    applyPassToggle(rg::PassToggle::Bloom, false);
    applyPassToggle(rg::PassToggle::MotionBlur, false);
    applyPassToggle(rg::PassToggle::Dof, false);
    applyPassToggle(rg::PassToggle::LensCa, true);
    applyPassToggle(rg::PassToggle::LensVignette, true);
    applyPassToggle(rg::PassToggle::LensGrain, true);
    ssrStrength_ = 0.6f;
    dofFocus_ = 0.f;
    dofFstop_ = kDofDefaultFstop;
    dofMaxBlur_ = kDofMaxCoC;
    lodEnabled_ = lodEnabledByDefault();
    gradeTemperatureK_ = 6500.f;
    gradeTint_ = 0.f;
    gradeContrast_ = 1.f;
    gradeSaturation_ = 1.f;
    lutCarry_.clear();
    if (hdrEnabled_) setHdrEnabled(false);
    setFullscreenEnabled(false);
}

void GuiApp::setFullscreenEnabled(bool enabled) {
    if (fullscreenEnabled_ == enabled) return;
    fullscreenEnabled_ = enabled;
    window_.setFullscreen(enabled);
}

void GuiApp::requestRebuild(const RenderConfig& cfg) {
    if (loadPhase_.load(std::memory_order_acquire) == LoadPhase::Loading) {
        // No queueing and no cancellation: one rebuild in flight at a time.
        statusLine_ = "rebuild in progress — wait for the load to finish";
        return;
    }
    pendingConfig_ = cfg;
    pendingRebuild_ = true;
}

// Async rebuild: the heavy half (glTF decode/upload when the scene changed,
// upscaler DLL/context init) runs on loadWorkerMain while the old stack keeps
// rendering; the light half runs in finishAsyncRebuild once the worker
// reports Ready.  The rebuild is tiered (see LoadPhase in GuiApp.h): scene
// reload only happens when the scene path actually changed.
void GuiApp::applyRebuild() {
    // Check the phase before clearing pendingRebuild_: run() calls this ahead of
    // finishAsyncRebuild, so at the Ready-but-not-yet-swapped point the request
    // must stay queued (it becomes effective next frame once phase is Idle)
    // instead of being silently dropped.
    if (loadPhase_.load(std::memory_order_acquire) != LoadPhase::Idle) return;
    pendingRebuild_ = false;
    loadConfig_ = pendingConfig_;

    // Dirty classification against the active stack.
    loadSceneDirty_ = !stackOk_ || loadConfig_.scenePath != active_.scenePath;
    loadTargetsDirty_ = loadSceneDirty_ || loadConfig_.displayW != active_.displayW ||
                        loadConfig_.displayH != active_.displayH ||
                        loadConfig_.renderScale != active_.renderScale ||
                        loadConfig_.gtSsaa != active_.gtSsaa ||
                        loadConfig_.gtApplyScale != active_.gtApplyScale ||
                        loadConfig_.envMapPath != active_.envMapPath;

    loadResult_ = LoadResult{};
    loadDone_.store(0, std::memory_order_relaxed);
    loadTotal_.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(loadStageMutex_);
        loadStageText_ = "starting load...";
    }
    statusLine_ = loadSceneDirty_     ? "loading scene in background..."
                  : loadTargetsDirty_ ? "rebuilding targets in background..."
                                      : "switching algorithms...";
    loadPhase_.store(LoadPhase::Loading, std::memory_order_release);
    loadThread_ = std::thread(&GuiApp::loadWorkerMain, this, loadConfig_);
}

void GuiApp::loadWorkerMain(RenderConfig cfg) {
    auto setStage = [this](const std::string& s) {
        std::lock_guard<std::mutex> lk(loadStageMutex_);
        loadStageText_ = s;
    };
    auto progress = [this, &setStage](Scene::LoadStage stage, size_t done, size_t total) {
        loadDone_.store(static_cast<uint32_t>(done), std::memory_order_relaxed);
        loadTotal_.store(static_cast<uint32_t>(total), std::memory_order_relaxed);
        const char* name = stage == Scene::LoadStage::Parse     ? "parsing glTF"
                           : stage == Scene::LoadStage::Textures ? "loading textures"
                           : stage == Scene::LoadStage::Meshes   ? "loading meshes"
                                                                 : "finalizing scene";
        char buf[96];
        if (total > 0)
            std::snprintf(buf, sizeof(buf), "%s %zu/%zu", name, done, total);
        else
            std::snprintf(buf, sizeof(buf), "%s...", name);
        setStage(buf);
    };
    // Partial results are handed to the main thread (via loadResult_) for
    // cleanup after the join — nothing GPU-side leaks on failure paths.
    auto scene = std::make_unique<Scene>();
    auto fail = [this, &scene](const std::string& err) {
        if (scene) loadResult_.scene = std::move(scene);
        loadResult_.error = err;
        loadPhase_.store(LoadPhase::Failed, std::memory_order_release);
    };

    // Worker-private command pool: command pools need external sync, so the
    // worker never touches ctx_.oneShotPool/framePool while the main thread
    // renders.  Queue submits are serialized via ctx_.queueMutex instead.
    VkCommandPoolCreateInfo poolCi = {};
    poolCi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolCi.queueFamilyIndex = ctx_.graphicsQueueFamily;
    if (vkCreateCommandPool(ctx_.device, &poolCi, nullptr, &loadPool_) != VK_SUCCESS ||
        !loadPool_) {
        loadPool_ = VK_NULL_HANDLE;
        fail("worker command pool creation failed");
        return;
    }

    try {
        // Scene tier: only reload when the scene path changed (targets/algo
        // tiers keep the active scene untouched; loadResult_.scene stays
        // null and finishAsyncRebuild preserves scene_).
        if (loadSceneDirty_) {
            bool sceneOk = false;
            // GUI is interactive: enable background fine-mip streaming
            // (Phase 7b).  Bench subprocesses run the viewer with --frames,
            // which disables it there.
            scene->streamingEnabled = texStreamingOverride() >= 0;
            if (!cfg.scenePath.empty())
                sceneOk = scene->loadGltf(ctx_, cfg.scenePath.c_str(), loadPool_, progress);
            if (!sceneOk) {
                setStage("building procedural scene");
                sceneOk = scene->loadProcedural(ctx_, loadPool_);
            }
            if (!sceneOk || !ensureSceneFallbacks(*scene, ctx_, loadPool_)) {
                fail("scene load failed (kept previous scene)");
                return;
            }
        }

        const uint32_t rw =
            std::max(1u, static_cast<uint32_t>(static_cast<float>(cfg.displayW) * cfg.renderScale));
        const uint32_t rh =
            std::max(1u, static_cast<uint32_t>(static_cast<float>(cfg.displayH) * cfg.renderScale));
        const VulkanEnv env = ctx_.toEnv(loadPool_);
        std::string err;
        const bool algosOk = initAlgorithmsFor(
            env, cfg, rw, rh, loadResult_.algos, err,
            [this, &setStage](const char* name, uint32_t index, uint32_t count) {
                loadDone_.store(index, std::memory_order_relaxed);
                loadTotal_.store(count, std::memory_order_relaxed);
                setStage(std::string("init ") + name + "...");
            });
        if (!algosOk) {
            fail(err + " (kept previous scene)");
            return;
        }
        loadResult_.note = err; // non-fatal fallback note (may be empty)
        loadDone_.store(static_cast<uint32_t>(cfg.algos.size()), std::memory_order_relaxed);
        if (loadSceneDirty_) loadResult_.scene = std::move(scene);
        loadPhase_.store(LoadPhase::Ready, std::memory_order_release);
    } catch (const std::exception& e) {
        fail(std::string("load exception: ") + e.what());
    } catch (...) {
        fail("load exception (unknown)");
    }
}

void GuiApp::discardLoadResult() {
    if (loadResult_.scene) {
        loadResult_.scene->destroy(ctx_);
        loadResult_.scene.reset();
    }
    for (AlgoColumn& algo : loadResult_.algos) {
        if (algo.upscaler) {
            algo.upscaler->shutdown();
            algo.upscaler.reset();
        }
        if (algo.frameGen) {
            algo.frameGen->shutdown();
            algo.frameGen.reset();
        }
    }
    loadResult_.algos.clear();
    loadResult_.error.clear();
    loadResult_.note.clear();
}

void GuiApp::finishAsyncRebuild() {
    if (loadThread_.joinable()) loadThread_.join();
    // The worker may have left GPU work in flight (scene uploads and
    // upscaler-SDK internal submits that we cannot verify); drain it before
    // the worker-private command pool is destroyed or failed worker products
    // are discarded, on both the success and failure paths.
    vkDeviceWaitIdle(ctx_.device);
    if (loadPool_) {
        vkDestroyCommandPool(ctx_.device, loadPool_, nullptr);
        loadPool_ = VK_NULL_HANDLE;
    }
    const bool ok = loadPhase_.load(std::memory_order_acquire) == LoadPhase::Ready;
    loadPhase_.store(LoadPhase::Idle, std::memory_order_release);
    if (!ok) {
        const std::string err =
            loadResult_.error.empty() ? "rebuild failed (kept previous scene)" : loadResult_.error;
        discardLoadResult();
        statusLine_ = err;
        return;
    }

    // Exclusive safe point: the worker has finished (joined) and the GPU is
    // idle, so teardown/swap can happen without racing in-flight frames.
    active_ = loadConfig_;
    const std::string note = loadResult_.note;
    loadResult_.note.clear();
    loadResult_.error.clear();

    if (!loadTargetsDirty_) {
        // ---- algo-only fast path: scene, render targets, descriptor pool and
        // all shared descriptor sets persist; only the upscaler set is swapped.
        destroyAlgoResources();
        algos_ = std::move(loadResult_.algos);
        loadResult_.algos.clear();
        // Switching into compare mode may need the global metric buffers
        // (viewer-mode builds skip them; self-gated on mode + existing buffer).
        if (!createMetricResources()) {
            statusLine_ = "metric resource creation failed";
            stackOk_ = false;
            return;
        }
        for (uint32_t i = 0; i < static_cast<uint32_t>(algos_.size()); ++i) {
            if (!createAlgoResources(algos_[i], i)) {
                statusLine_ = "per-algo resource creation failed";
                stackOk_ = false;
                return;
            }
        }
        gtActive_ = active_.mode == Mode::Compare || algos_.empty();
        resetFrameState();
        refreshOverlayText();
        stackOk_ = true;
    } else {
        if (loadSceneDirty_) {
            // ---- full path: everything is torn down, new scene swapped in.
            destroyRenderStack();
            scene_ = std::move(*loadResult_.scene);
            loadResult_.scene.reset();
        } else {
            // ---- middle path: scene_ survives; all resolution/mode/env
            // dependent resources are rebuilt around it.
            destroyStackResources();
        }
        algos_ = std::move(loadResult_.algos);
        loadResult_.algos.clear();

        // The env map is part of the deferred core (IBL maps): rebuild it only
        // when the path actually changed.  Empty = sky atmosphere.
        if (active_.envMapPath != envMapActive_) {
            deferred_.destroy(ctx_);
            envMapActive_ = active_.envMapPath;
            if (!deferred_.init(ctx_, envMapActive_.c_str(),
                                sunDirectionFromElevAzimuth(sunElevationDeg_, sunAzimuthDeg_))) {
                statusLine_ = "deferred core init failed (env map: " + envMapActive_ + ")";
                stackOk_ = false;
                return;
            }
        }

        if (loadSceneDirty_) {
            hasTransparency_ = deferred_.sceneHasTransparency(scene_);
            applyLightingPreset(lightingPresetForScene(active_.scenePath));
        }
        gtActive_ = active_.mode == Mode::Compare || algos_.empty();
        if (!beginStackConfig()) {
            stackOk_ = false;
            return;
        }
        stackOk_ = buildStackTail();
    }
    if (stackOk_) {
        if (!note.empty()) {
            statusLine_ = note + " (fell back to native)";
        } else {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "stack ready: %s, %ux%u @ %.2f",
                          active_.mode == Mode::Viewer ? "viewer" : "compare", active_.displayW,
                          active_.displayH, static_cast<double>(active_.renderScale));
            statusLine_ = buf;
        }
    }
}

void GuiApp::drawLoadOverlay() {
    if (loadPhase_.load(std::memory_order_acquire) != LoadPhase::Loading) return;
    const ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f),
                            ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(380.f * uiScale_, 0.f), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.94f);
    if (ImGui::Begin("Loading##overlay", nullptr,
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                         ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings)) {
        std::string stage;
        {
            std::lock_guard<std::mutex> lk(loadStageMutex_);
            stage = loadStageText_;
        }
        const uint32_t done = loadDone_.load(std::memory_order_relaxed);
        const uint32_t total = loadTotal_.load(std::memory_order_relaxed);
        ImGui::TextUnformatted(stage.c_str());
        if (total > 0) {
            char overlay[48];
            std::snprintf(overlay, sizeof(overlay), "%u/%u", done, total);
            ImGui::ProgressBar(static_cast<float>(done) / static_cast<float>(total),
                               ImVec2(-1.f, 0.f), overlay);
        } else {
            // Indeterminate stage (parse / finalize): pulse the bar.
            const float t = static_cast<float>(ImGui::GetTime());
            ImGui::ProgressBar(0.5f - 0.5f * std::cos(t * 2.5f), ImVec2(-1.f, 0.f), "working...");
        }
        ImGui::TextDisabled("the previous scene keeps rendering meanwhile");
    }
    ImGui::End();
}

void GuiApp::loadCameraPathFromUi() {
    if (ui_.cameraPathFile[0] == '\0') {
        statusLine_ = "camera path: empty filename";
        return;
    }
    if (loadCameraPath(ui_.cameraPathFile, path_)) {
        pathPlaying_ = true;
        pathFrame_ = 0;
        char buf[320];
        std::snprintf(buf, sizeof(buf), "camera path loaded: %zu keyframes", path_.size());
        statusLine_ = buf;
    } else {
        statusLine_ = std::string("failed to load camera path: ") + ui_.cameraPathFile;
        path_.clear();
        pathPlaying_ = false;
    }
}

// ---------------------------------------------------------------------------
// Render stack: everything that depends on scene/resolution/scale/algorithm
// set.  Rebuilt from scratch on Apply (the device and ImGui persist).
// The initial build in init() is synchronous (buildRenderStack); Apply goes
// through the async path (applyRebuild -> loadWorkerMain -> finishAsyncRebuild)
// so the old scene keeps rendering while the new one loads.
// ---------------------------------------------------------------------------
bool GuiApp::beginStackConfig() {
    renderWidth_ =
        std::max(1u, static_cast<uint32_t>(static_cast<float>(active_.displayW) * active_.renderScale));
    renderHeight_ =
        std::max(1u, static_cast<uint32_t>(static_cast<float>(active_.displayH) * active_.renderScale));

    // Resize the OS window + swapchain when the output resolution changed.
    if (window_.width() != static_cast<int>(active_.displayW) ||
        window_.height() != static_cast<int>(active_.displayH)) {
        window_.setClientSize(static_cast<int>(active_.displayW),
                              static_cast<int>(active_.displayH));
    }
    if (!recreateGuiSwapchain()) {
        statusLine_ = "swapchain creation failed";
        return false;
    }
    return true;
}

bool GuiApp::buildRenderStack() {
    if (!beginStackConfig()) return false;

    bool sceneOk = false;
    scene_.streamingEnabled = texStreamingOverride() >= 0; // interactive GUI (Phase 7b)
    if (!active_.scenePath.empty()) sceneOk = scene_.loadGltf(ctx_, active_.scenePath.c_str());
    if (!sceneOk) sceneOk = scene_.loadProcedural(ctx_);
    if (!sceneOk || !ensureSceneFallbacks(scene_, ctx_, VK_NULL_HANDLE)) {
        statusLine_ = "scene load failed";
        return false;
    }
    hasTransparency_ = deferred_.sceneHasTransparency(scene_);
    applyLightingPreset(lightingPresetForScene(active_.scenePath));
    if (opts_.exposure > 0.f) exposure_ = opts_.exposure;

    std::string err;
    if (!initAlgorithms(err)) {
        statusLine_ = err;
        return false;
    }
    gtActive_ = active_.mode == Mode::Compare || algos_.empty();

    return buildStackTail();
}

bool GuiApp::buildStackTail() {
    if (!createRenderTargets()) { statusLine_ = "render target creation failed"; return false; }
    if (!createFontAtlas()) { statusLine_ = "font atlas creation failed"; return false; }
    if (!createMetricResources()) { statusLine_ = "metric resource creation failed"; return false; }
    if (!createShaders()) { statusLine_ = "shader load failed"; return false; }
    if (!createDescriptors()) { statusLine_ = "descriptor setup failed"; return false; }
    // Per-algo outputs/buffers/descriptor sets (after the pool exists).
    for (uint32_t i = 0; i < static_cast<uint32_t>(algos_.size()); ++i) {
        if (!createAlgoResources(algos_[i], i)) {
            statusLine_ = "per-algo resource creation failed";
            return false;
        }
    }
    if (!createPipelines()) { statusLine_ = "pipeline creation failed"; return false; }
    if (!createSyncResources()) { statusLine_ = "sync resource creation failed"; return false; }
    if (!createScreenshotStaging()) { statusLine_ = "screenshot staging failed"; return false; }
    if (!timestamps_.create(ctx_, kFramesInFlight)) { statusLine_ = "timestamp query failed"; return false; }
    if (!profiler_.create(ctx_, kFramesInFlight)) { statusLine_ = "gpu profiler query failed"; return false; }

    resetFrameState();
    // Launch-option zoom (--compare-zoom): the first frames can trigger
    // spurious tab-switch rebuilds, so apply it inside the compare stack
    // build itself (consumed once).
    if (launchZoomPending_ && active_.mode == Mode::Compare) {
        compareZoom_ = std::clamp(opts_.compareZoom, 1.f, 16.f);
        launchZoomPending_ = false;
    }

    // glTF scenes (sponza & co.) are centered on the origin; the raw default
    // free-fly pose sits inside their outer wall and shows only black.  Prefer
    // a registered per-scene start pose, else the CLI automation orbit's
    // first keyframe.
    if (!active_.scenePath.empty()) {
        Vec3 pos, fwd;
        if (initialCameraPose(active_.scenePath, pos, fwd)) {
            camera_.setPose(pos, fwd, {0.f, 1.f, 0.f});
        } else {
            camera_.position = {6.5f, 2.f, 0.f};
            camera_.up = {0.f, 1.f, 0.f};
            camera_.lookAt({0.f, 2.f, 0.f});
        }
    } else {
        // Built-in box scene: reset to the Camera constructor default pose
        // (same as a fresh app start) so switching back from a glTF scene
        // does not keep the previous camera position.
        camera_.setPose({0.f, 3.f, 12.f}, {0.f, -0.2f, -1.f}, {0.f, 1.f, 0.f});
    }

    refreshOverlayText();
    return true;
}

void GuiApp::resetFrameState() {
    // Fresh stack: reset frame history so temporal upscalers get resetHistory.
    prevViewProj_ = Mat4::identity();
    havePrevCamera_ = false;
    jitterX_ = jitterY_ = prevJitterX_ = prevJitterY_ = 0.f;
    metricPending_[0] = metricPending_[1] = false;
    metricMask_[0] = metricMask_[1] = 0;
    autoExposureJustEnabled_ = false; // fresh channels re-seed from exposure_
    frameTimesLog_.clear();
    historyHead_ = 0;
    historyCount_ = 0;
    lastTimings_ = {};
    renderFrameIndex_ = 0;
    compareZoom_ = 1.f;
    comparePanU_ = 0.5f;
    comparePanV_ = 0.5f;
    comparePanDrag_ = false;
}

bool GuiApp::initAlgorithms(std::string& err) {
    const bool ok = initAlgorithmsFor(ctx_.toEnv(), active_, renderWidth_, renderHeight_, algos_,
                                      err);
    // Viewer tolerates a failed algorithm: fall back to the native GT view.
    if (ok && !err.empty()) statusLine_ = err + " (fell back to native)";
    return ok;
}

bool GuiApp::initAlgorithmsFor(const VulkanEnv& env, const RenderConfig& cfg, uint32_t renderW,
                               uint32_t renderH, std::vector<AlgoColumn>& out, std::string& err,
                               const std::function<void(const char* name, uint32_t index,
                                                        uint32_t count)>& onAlgo) {
    const uint32_t count = static_cast<uint32_t>(cfg.algos.size());
    for (uint32_t i = 0; i < count; ++i) {
        const RenderConfig::AlgoSpec& spec = cfg.algos[i];
        if (onAlgo) onAlgo(spec.sr.c_str(), i, count);
        std::unique_ptr<IUpscaler> up = createUpscaler(spec.sr.c_str());
        if (!up) {
            err = "unknown upscaler: " + spec.sr;
            continue;
        }
        if (!up->isAvailable(env)) {
            err = spec.sr + " is not available on this device";
            continue;
        }
        UpscalerDesc desc;
        desc.renderWidth = renderW;
        desc.renderHeight = renderH;
        desc.displayWidth = cfg.displayW;
        desc.displayHeight = cfg.displayH;
        desc.hdr = true;
        desc.invertedDepth = false;
        desc.infiniteFarPlane = true;
        if (!up->init(env, desc)) {
            err = spec.sr + " init failed";
            continue;
        }
        AlgoColumn col;
        col.id = spec.sr;
        col.fg = spec.fg;
        col.upscaler = std::move(up);
        if (!spec.fg.empty()) {
            col.frameGen = createFrameGen(spec.fg.c_str());
            if (!col.frameGen) {
                err = spec.fg + " frame interpolator is not available";
                col.fg.clear();
            } else if (!col.frameGen->isAvailable(env)) {
                err = spec.fg + " frame interpolator unavailable on this device";
                col.frameGen.reset();
                col.fg.clear();
            } else {
                FrameGenDesc fgDesc;
                fgDesc.renderWidth = renderW;
                fgDesc.renderHeight = renderH;
                fgDesc.displayWidth = cfg.displayW;
                fgDesc.displayHeight = cfg.displayH;
                fgDesc.hdr = true;
                fgDesc.invertedDepth = false;
                fgDesc.infiniteFarPlane = true;
                if (!col.frameGen->init(env, fgDesc)) {
                    err = spec.fg + " frame interpolator init failed";
                    col.frameGen.reset();
                    col.fg.clear();
                }
            }
        }
        out.push_back(std::move(col));
    }

    if (cfg.mode == Mode::Compare && out.empty()) {
        if (err.empty()) err = "compare: no usable upscaler selected";
        return false;
    }
    return true;
}

bool GuiApp::createRenderTargets() {
    const uint32_t dw = active_.displayW;
    const uint32_t dh = active_.displayH;

    auto createRT = [&](ImageResource& rt, uint32_t w, uint32_t h, VkFormat format,
                        VkImageUsageFlags usage, VkImageAspectFlags aspect) {
        rt.width = w;
        rt.height = h;
        rt.format = format;
        if (createImage(ctx_, w, h, format, usage, rt.image, rt.memory) != VK_SUCCESS) return false;
        rt.view = createImageView(ctx_, rt.image, format, aspect);
        return rt.view != VK_NULL_HANDLE;
    };

    // Shared low-resolution GBuffer inputs for every upscaler.
    // TRANSFER_SRC/DST on color: some upscalers (DLSS via Streamline) copy the
    // inputs into internal buffers instead of just sampling them.
    if (!createRT(gbColor_, renderWidth_, renderHeight_, deferred::kHdrColorFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                      VK_IMAGE_USAGE_STORAGE_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    // Unjittered LR color for spatial plugins when mixed with temporal ones
    // (raster jitter cannot be undone by resampling).
    if (!createRT(gbColorSpatial_, renderWidth_, renderHeight_, deferred::kHdrColorFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gbAlbedo_, renderWidth_, renderHeight_, deferred::kAlbedoFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gbNormal_, renderWidth_, renderHeight_, deferred::kNormalFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gbMaterial_, renderWidth_, renderHeight_, deferred::kMaterialFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gbEmissive_, renderWidth_, renderHeight_, deferred::kEmissiveFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gbMotion_, renderWidth_, renderHeight_, deferred::kMotionFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    // Translucent coverage mask (reactive/TC mask for upscalers).
    // TRANSFER_SRC/DST: some upscalers (DLSS via Streamline) copy the inputs
    // into internal buffers instead of just sampling them.
    if (!createRT(gbReactive_, renderWidth_, renderHeight_, deferred::kReactiveFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    // Dilated coverage mask (reactive_dilate.comp output; see GuiApp.h).
    if (!createRT(gbReactiveDilated_, renderWidth_, renderHeight_, deferred::kReactiveFormat,
                  VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gbDepth_, renderWidth_, renderHeight_, deferred::kDepthFormat,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_DEPTH_BIT))
        return false;

    // GTAO targets (working RG16F + filtered R16F) for the low-res path.
    const VkImageUsageFlags aoUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (!createRT(gbAoRaw_, renderWidth_, renderHeight_, kAoRawFormat, aoUsage,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gbAo_, renderWidth_, renderHeight_, VK_FORMAT_R16_SFLOAT, aoUsage,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;

    // Native-resolution ground truth (GBuffer sampled by the lighting pass).
    // "GT (Apply scale)" renders the GT at the low input resolution instead
    // (no AA, no upscale pass); compose/metrics sample it via normalized UVs.
    const uint32_t gtW = active_.gtApplyScale ? renderWidth_ : dw;
    const uint32_t gtH = active_.gtApplyScale ? renderHeight_ : dh;
    if (!createRT(gtColor_, gtW, gtH, deferred::kHdrColorFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtAlbedo_, gtW, gtH, deferred::kAlbedoFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtNormal_, gtW, gtH, deferred::kNormalFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtMaterial_, gtW, gtH, deferred::kMaterialFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtEmissive_, gtW, gtH, deferred::kEmissiveFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtMotion_, gtW, gtH, deferred::kMotionFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtDepth_, gtW, gtH, deferred::kDepthFormat,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_DEPTH_BIT))
        return false;
    if (!createRT(gtAoRaw_, gtW, gtH, kAoRawFormat, aoUsage, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtAo_, gtW, gtH, VK_FORMAT_R16_SFLOAT, aoUsage, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    // Motion blur + DOF working sets (Phase 6b), LR + GT paths.
    if (!deferred_.createPostFxTargets(ctx_, renderWidth_, renderHeight_, gbPostFx_))
        return false;
    if (!deferred_.createPostFxTargets(ctx_, gtW, gtH, gtPostFx_))
        return false;
    // HDR bloom pyramids (Phase 6a), one per path; the runtime checkbox skips
    // the pass, so creation is unconditional.
    if (!deferred_.createBloomPyramid(ctx_, renderWidth_, renderHeight_, gbBloom_))
        return false;
    if (!deferred_.createBloomPyramid(ctx_, gtW, gtH, gtBloom_))
        return false;
    // Color mip chains for roughness-aware SSR (mip 0 = the lit-color copy).
    if (!deferred_.createColorPyramid(ctx_, renderWidth_, renderHeight_, gbColorPyramid_))
        return false;
    if (!deferred_.createColorPyramid(ctx_, gtW, gtH, gtColorPyramid_))
        return false;
    // Clustered shading grids (per-path resolution, per-slot buffers).
    if (!deferred_.createClusterGrid(ctx_, renderWidth_, renderHeight_, gbCluster_))
        return false;
    if (!deferred_.createClusterGrid(ctx_, gtW, gtH, gtCluster_))
        return false;
    // Froxel volumetric fog volumes (Phase 5a), one per path; the runtime
    // checkbox skips the passes, so creation follows the preset only.
    if (fogParams_.enabled) {
        if (!deferred_.createVolFogVolume(ctx_, renderWidth_, renderHeight_, gbFog_) ||
            !deferred_.createVolFogVolume(ctx_, gtW, gtH, gtFog_)) {
            std::fprintf(stderr, "warning: volumetric fog volume creation failed; fog disabled\n");
            deferred_.destroyVolFogVolume(ctx_, gbFog_);
            deferred_.destroyVolFogVolume(ctx_, gtFog_);
            fogParams_.enabled = false;
        }
    }

    // 200% SSAA ground truth: deferred render at 2x, downsample into gtColor_.
    if (active_.gtSsaa) {
        if (!createRT(gtSsaaColor_, dw * 2, dh * 2, deferred::kHdrColorFormat,
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                      VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
        if (!createRT(gtSsaaAlbedo_, dw * 2, dh * 2, deferred::kAlbedoFormat,
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                      VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
        if (!createRT(gtSsaaNormal_, dw * 2, dh * 2, deferred::kNormalFormat,
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                      VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
        if (!createRT(gtSsaaMaterial_, dw * 2, dh * 2, deferred::kMaterialFormat,
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                      VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
        if (!createRT(gtSsaaEmissive_, dw * 2, dh * 2, deferred::kEmissiveFormat,
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                      VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
        if (!createRT(gtSsaaMotion_, dw * 2, dh * 2, deferred::kMotionFormat,
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                      VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
        if (!createRT(gtSsaaDepth_, dw * 2, dh * 2, deferred::kDepthFormat,
                      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                      VK_IMAGE_ASPECT_DEPTH_BIT))
            return false;
        if (!createRT(gtSsaaAoRaw_, dw * 2, dh * 2, kAoRawFormat, aoUsage,
                      VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
        if (!createRT(gtSsaaAo_, dw * 2, dh * 2, VK_FORMAT_R16_SFLOAT, aoUsage,
                      VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
        if (!deferred_.createDepthPyramid(ctx_, dw * 2, dh * 2, gtSsaaPyramid_))
            return false;
        if (!deferred_.createDepthPyramid(ctx_, dw * 2, dh * 2, gtSsaaPyramidAo_,
                                          /*aoFilter=*/true, camera_.nearPlane, camera_.farPlane))
            return false;
        if (!deferred_.createAoHistory(ctx_, dw * 2, dh * 2, gtSsaaAoHist_))
            return false;
        if (!createRT(gtSsaaSsrTrace_, dw * 2, dh * 2, kSsrTraceFormat, aoUsage,
                      VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
        if (!deferred_.createSsrHistory(ctx_, dw * 2, dh * 2, gtSsaaSsrHist_))
            return false;
        if (!deferred_.createColorPyramid(ctx_, dw * 2, dh * 2, gtSsaaColorPyramid_))
            return false;
        if (!deferred_.createClusterGrid(ctx_, dw * 2, dh * 2, gtSsaaCluster_))
            return false;
        if (!deferred_.createPostFxTargets(ctx_, dw * 2, dh * 2, gtSsaaPostFx_))
            return false;
        if (!deferred_.createBloomPyramid(ctx_, dw * 2, dh * 2, gtSsaaBloom_))
            return false;
        if (fogParams_.enabled &&
            !deferred_.createVolFogVolume(ctx_, dw * 2, dh * 2, gtSsaaFog_)) {
            std::fprintf(stderr,
                         "warning: SSAA volumetric fog volume creation failed; SSAA fog off\n");
            deferred_.destroyVolFogVolume(ctx_, gtSsaaFog_);
        }
    }

    // Hi-Z depth pyramids for the SSR march (LR / GT paths).
    if (!deferred_.createDepthPyramid(ctx_, renderWidth_, renderHeight_, gbPyramid_))
        return false;
    if (!deferred_.createDepthPyramid(ctx_, gtW, gtH, gtPyramid_))
        return false;
    // GTAO view-Z depth chains (XeGTAO DepthMIPFilter) + temporal history.
    if (!deferred_.createDepthPyramid(ctx_, renderWidth_, renderHeight_, gbPyramidAo_,
                                      /*aoFilter=*/true, camera_.nearPlane, camera_.farPlane))
        return false;
    if (!deferred_.createDepthPyramid(ctx_, gtW, gtH, gtPyramidAo_, /*aoFilter=*/true,
                                      camera_.nearPlane, camera_.farPlane))
        return false;
    if (!deferred_.createAoHistory(ctx_, renderWidth_, renderHeight_, gbAoHist_))
        return false;
    if (!deferred_.createAoHistory(ctx_, gtW, gtH, gtAoHist_))
        return false;
    // Opaque SSR trace targets + temporal history (Phase 2d), one per path.
    if (!createRT(gbSsrTrace_, renderWidth_, renderHeight_, kSsrTraceFormat, aoUsage,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtSsrTrace_, gtW, gtH, kSsrTraceFormat, aoUsage, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!deferred_.createSsrHistory(ctx_, renderWidth_, renderHeight_, gbSsrHist_))
        return false;
    if (!deferred_.createSsrHistory(ctx_, gtW, gtH, gtSsrHist_))
        return false;

    // (Per-algorithm output images are created by createAlgoResources, after
    // the descriptor pool exists, so algo-only rebuilds can redo just those.)

    // Tonemapped composite (columns + overlay text); presentation + screenshot.
    if (!createRT(composeImage_, dw, dh, kComposeFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;

    // Debug UI screenshot target (swapchain format, so the ImGui backend
    // pipeline can render into it).
    if (uiShot_) {
        if (!createRT(uiShotImage_, dw, dh, swapchain_.format(),
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                      VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
    }
    return true;
}

bool GuiApp::createFontAtlas() {
    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation stagingMemory = VK_NULL_HANDLE;
    const VkDeviceSize size = kFontAtlasW * kFontAtlasH;
    if (createBuffer(ctx_, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging, stagingMemory) != VK_SUCCESS)
        return false;
    void* mapped = nullptr;
    vmaMapMemory(ctx_.allocator, stagingMemory, &mapped);
    buildFontAtlas(static_cast<uint8_t*>(mapped));
    vmaUnmapMemory(ctx_.allocator, stagingMemory);

    if (createImage(ctx_, kFontAtlasW, kFontAtlasH, VK_FORMAT_R8_UNORM,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    fontAtlas_.image, fontAtlas_.memory) != VK_SUCCESS) {
        vmaDestroyBuffer(ctx_.allocator, staging, stagingMemory);
        return false;
    }
    fontAtlas_.width = kFontAtlasW;
    fontAtlas_.height = kFontAtlasH;
    fontAtlas_.format = VK_FORMAT_R8_UNORM;

    submitUploadOneShot(
        ctx_,
        [&](VkCommandBuffer cmd) {
            copyBufferToImageTransferStage(cmd, staging, fontAtlas_.image, kFontAtlasW,
                                           kFontAtlasH);
        },
        [&](VkCommandBuffer cmd) { transitionImageToShaderRead(cmd, fontAtlas_.image); });
    vmaDestroyBuffer(ctx_.allocator, staging, stagingMemory);

    fontAtlas_.view = createImageView(ctx_, fontAtlas_.image, VK_FORMAT_R8_UNORM,
                                      VK_IMAGE_ASPECT_COLOR_BIT);
    return fontAtlas_.view != VK_NULL_HANDLE;
}

bool GuiApp::createMetricResources() {
    // Only compare mode runs the PSNR/SSIM reduction.
    if (active_.mode != Mode::Compare) return true;

    const uint32_t dw = active_.displayW;
    const uint32_t dh = active_.displayH;
    blocksPerRow_ = (dw + 7) / 8;
    blockCount_ = blocksPerRow_ * ((dh + 7) / 8);

    // (Per-algo blocks buffers live in createAlgoResources.)  The reduction
    // result buffer + readback staging are sized for kMaxAlgos and survive
    // algo-only rebuilds — allocate them once.
    if (metricResultBuf_) return true;

    const VkDeviceSize resultSize = kMaxAlgos * kMetricFloats * 4;
    if (createBuffer(ctx_, resultSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, metricResultBuf_,
                     metricResultMemory_) != VK_SUCCESS)
        return false;

    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        if (createBuffer(ctx_, resultSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         metricStaging_[i], metricStagingMemory_[i]) != VK_SUCCESS)
            return false;
        vmaMapMemory(ctx_.allocator, metricStagingMemory_[i],
                    &metricStagingMapped_[i]);
    }
    return true;
}

bool GuiApp::loadShader(const char* name, VkShaderModule& out) {
    const std::string path = sr::resolveShaderPath(SR_SHADER_DIR, name);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "failed to open shader: %s\n", path.c_str());
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
    return vkCreateShaderModule(ctx_.device, &ci, nullptr, &out) == VK_SUCCESS;
}

bool GuiApp::createShaders() {
    // Deferred scene/lighting shaders live in DeferredCore; compose/copy/
    // metric shaders are shared with compare mode (compiled by the compare
    // target).
    return loadShader("fullscreen.vert.spv", fullscreenVert_) &&
           loadShader("compare_compose.frag.spv", composeFrag_) &&
           loadShader("compare_copy.frag.spv", copyFrag_) &&
           loadShader("compare_metrics_blocks.comp.spv", metricBlocksComp_) &&
           loadShader("compare_metrics_reduce.comp.spv", metricReduceComp_);
}

bool GuiApp::createDescriptors() {
    // Scene/texture/lighting set layouts are owned by DeferredCore.
    VkDescriptorSetLayoutBinding composeBindings[4] = {};
    for (uint32_t i = 0; i < 4; ++i) {
        composeBindings[i].binding = i;
        composeBindings[i].descriptorCount = 1;
        composeBindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        composeBindings[i].descriptorType =
            i == 1 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    }
    VkDescriptorSetLayoutCreateInfo composeLayoutCi = {};
    composeLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    composeLayoutCi.bindingCount = 4;
    composeLayoutCi.pBindings = composeBindings;
    if (vkCreateDescriptorSetLayout(ctx_.device, &composeLayoutCi, nullptr, &composeSetLayout_) != VK_SUCCESS)
        return false;

    VkDescriptorSetLayoutBinding copyBinding = {};
    copyBinding.binding = 0;
    copyBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    copyBinding.descriptorCount = 1;
    copyBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo copyLayoutCi = {};
    copyLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    copyLayoutCi.bindingCount = 1;
    copyLayoutCi.pBindings = &copyBinding;
    if (vkCreateDescriptorSetLayout(ctx_.device, &copyLayoutCi, nullptr, &copySetLayout_) != VK_SUCCESS)
        return false;

    VkDescriptorSetLayoutBinding metricBindings[4] = {};
    for (uint32_t i = 0; i < 4; ++i) {
        metricBindings[i].binding = i;
        metricBindings[i].descriptorCount = 1;
        metricBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        metricBindings[i].descriptorType =
            i < 2 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    VkDescriptorSetLayoutCreateInfo metricLayoutCi = {};
    metricLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    metricLayoutCi.bindingCount = 4;
    metricLayoutCi.pBindings = metricBindings;
    if (vkCreateDescriptorSetLayout(ctx_.device, &metricLayoutCi, nullptr, &metricSetLayout_) != VK_SUCCESS)
        return false;

    // --- pool ------------------------------------------------------------------
    // Sized for kMaxAlgos columns (not the current algo count) and created
    // with FREE_DESCRIPTOR_SET, so the algo-only rebuild path can free and
    // reallocate per-algorithm sets without rebuilding anything else.
    const uint32_t numAlgos = kMaxAlgos;
    const uint32_t numColumns = 1 + kMaxAlgos;
    // Lighting sets: 3 per frame (GBuffer / GT / GT-SSAA), 11 samplers + 1 UBO
    // each; transparent sets: 3 per frame (GB/GT/SSAA), 5 samplers + 1 UBO each;
    // SSAO sets: static, 3 samplers + 2 storage images per path.
    // (+1 sampler per lighting/transparent set is the CSM shadow map binding.)
    // Opaque-SSR sets: 4 per frame (GB/GT/SSAA/spatial), 9 samplers + 1 UBO +
    // 1 storage image each.
    // Hi-Z / color downsample sets: one per mip per pyramid (sampler + storage image).
    const uint32_t hizSets = gbPyramid_.mipCount + gtPyramid_.mipCount + gtSsaaPyramid_.mipCount +
                             gbPyramidAo_.mipCount + gtPyramidAo_.mipCount +
                             gtSsaaPyramidAo_.mipCount;
    const uint32_t colorSets =
        gbColorPyramid_.mipCount + gtColorPyramid_.mipCount + gtSsaaColorPyramid_.mipCount;
    VkDescriptorPoolSize sizes[5] = {};
    // Froxel fog sets (Phase 5a): per path 14 samplers + 8 storage images +
    // per-slot light sets (1 UBO + 2 SSBO each).
    const uint32_t fogPaths = gbFog_.injectImage != VK_NULL_HANDLE
                                  ? (gtSsaaFog_.injectImage != VK_NULL_HANDLE ? 3 : 2)
                                  : 0;
    // MB/DOF post-fx sets (Phase 6b): per path 17 samplers + 13 storage images
    // + 9 sets (tile/neighbor/gather + coc/gather/composite x MB/lit variants
    // + MB copy-back).
    const uint32_t postFxPaths = active_.gtSsaa ? 3 : 2;
    // Bloom pyramid sets (Phase 6a): per path 2*kBloomMipCount sets (extract +
    // down/up pairs + composite), 1 sampler + 1 storage image each.
    const uint32_t bloomSets = 2 * kBloomMipCount * postFxPaths;
    sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount =
        deferred::kMaxTextures + numColumns * 3 + 2 + numAlgos * 8 + 14 * kFramesInFlight * 4 +
        8 * kFramesInFlight * 4 + 11 * kFramesInFlight * 4 + 10 * 3 + 3 * 6 + hizSets + colorSets +
        2 + 14 * fogPaths + 17 * postFxPaths + bloomSets + 3 + 2;
                           // + ssr temporal samplers (GB/GT/SSAA x2 sets); auto-exposure HDR
                           // sources (LR + GT); volfog light/temporal/march/composite samplers;
                           // lighting sets: 14 samplers each (GB/GT/SSAA/spatial), incl. shadow +
                           // spot atlas + 2 probe arrays; transparent sets: 8 each (incl. the
                           // froxel fog volume); SSR trace sets: 11 each; post-fx sets; bloom
                           // pyramid src taps (Phase 6a);
                           // occlusion cull sets: Hi-Z chains (GB/GT/SSAA, Phase 7a);
                           // reactive mask dilate: src mask + motion
    sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[1].descriptorCount = kFramesInFlight * 3 + numColumns + numAlgos + kFramesInFlight * 4 +
                               kFramesInFlight * 4 + kFramesInFlight * 4 + // + opaque-SSR UBOs
                               kFramesInFlight * 8 + // + probe UBOs (lighting + SSR sets)
                               kClusterSlots * fogPaths; // volfog light sets
    sizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    sizes[2].descriptorCount = kFramesInFlight * 3;
    sizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[3].descriptorCount = numAlgos * 2 + 4 + // metric blocks/result + auto-exposure (LR + GT)
                               2 * kFramesInFlight * 4 + // lighting sets: cluster lights + grid SSBOs
                               2 * kClusterSlots * 3 +    // cluster assign sets (GB/GT/SSAA paths)
                               kFramesInFlight * 3 +      // scene sets: instance SSBO (Phase 7a)
                               2 * 3 +                    // occlusion cull sets (GB/GT/SSAA, Phase 7a)
                               2 * kClusterSlots * fogPaths; // volfog light sets
    sizes[4].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    // ssao raw + temporal history + blur outputs (GB/GT/SSAA) + pyramid mips +
    // SSR trace targets + SSR temporal history write / scene-color RMW (x2 sets per path)
    // + volfog inject/light/temporal/march/composite storage (per fog path)
    // + bloom pyramid dst mips (Phase 6a, per path)
    sizes[4].descriptorCount = 15 + hizSets + colorSets + kFramesInFlight * 4 + 2 * 6 +
                               8 * fogPaths + 11 * postFxPaths + 2 * postFxPaths + bloomSets +
                               1; // reactive mask dilate dst
    VkDescriptorPoolCreateInfo poolCi = {};
    poolCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolCi.maxSets = kFramesInFlight * 3 + 2 + numColumns + 1 + numAlgos * 3 + kFramesInFlight * 4 +
                     kFramesInFlight * 4 + // transparent sets (GB/GT/SSAA/spatial)
                     kFramesInFlight * 4 + // opaque-SSR sets (GB/GT/SSAA/spatial)
                     15 +                  // ssao + temporal + blur sets (GB/GT/SSAA, static)
                     6 +                   // ssr temporal sets (GB/GT/SSAA x2, static)
                     2 +                   // auto-exposure sets (LR + GT)
                     kClusterSlots * 3 +   // cluster assign sets (GB/GT/SSAA)
                     8 * fogPaths +        // volfog sets (per fog path)
                     8 * postFxPaths +     // MB/DOF post-fx sets (Phase 6b)
                     1 * postFxPaths +     // MB copy-back set (Phase 6b)
                     bloomSets +           // bloom pyramid sets (Phase 6a)
                     1 +                   // reactive mask dilate set
                     3 +                   // occlusion cull sets (GB/GT/SSAA, Phase 7a)
                     hizSets + colorSets;
    poolCi.poolSizeCount = 5;
    poolCi.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(ctx_.device, &poolCi, nullptr, &descriptorPool_) != VK_SUCCESS)
        return false;

    // Cluster assignment sets (per slot, per path); buffers from createRenderTargets.
    if (!deferred_.writeClusterGridSets(ctx_, descriptorPool_, gbCluster_)) return false;
    if (!deferred_.writeClusterGridSets(ctx_, descriptorPool_, gtCluster_)) return false;
    if (active_.gtSsaa && !deferred_.writeClusterGridSets(ctx_, descriptorPool_, gtSsaaCluster_))
        return false;

    if (!deferred_.writeDepthPyramidSets(ctx_, descriptorPool_, gbDepth_.view, gbPyramid_))
        return false;
    if (!deferred_.writeDepthPyramidSets(ctx_, descriptorPool_, gtDepth_.view, gtPyramid_))
        return false;
    if (active_.gtSsaa &&
        !deferred_.writeDepthPyramidSets(ctx_, descriptorPool_, gtSsaaDepth_.view,
                                         gtSsaaPyramid_))
        return false;
    if (!deferred_.writeDepthPyramidSets(ctx_, descriptorPool_, gbDepth_.view, gbPyramidAo_))
        return false;
    if (!deferred_.writeDepthPyramidSets(ctx_, descriptorPool_, gtDepth_.view, gtPyramidAo_))
        return false;
    if (active_.gtSsaa &&
        !deferred_.writeDepthPyramidSets(ctx_, descriptorPool_, gtSsaaDepth_.view,
                                         gtSsaaPyramidAo_))
        return false;
    // Color chain set 0 samples the lit HDR target of each path.
    if (!deferred_.writeColorPyramidSets(ctx_, descriptorPool_, gbColor_.view, gbColorPyramid_))
        return false;
    if (!deferred_.writeColorPyramidSets(ctx_, descriptorPool_, gtColor_.view, gtColorPyramid_))
        return false;
    if (active_.gtSsaa &&
        !deferred_.writeColorPyramidSets(ctx_, descriptorPool_, gtSsaaColor_.view,
                                         gtSsaaColorPyramid_))
        return false;

    // MB/DOF post-fx sets (Phase 6b): srcColor = the path's lit HDR target.
    if (!deferred_.writePostFxSets(ctx_, descriptorPool_, gbColor_.view, gbMotion_.view,
                                   gbDepth_.view, gbPostFx_))
        return false;
    if (!deferred_.writePostFxSets(ctx_, descriptorPool_, gtColor_.view, gtMotion_.view,
                                   gtDepth_.view, gtPostFx_))
        return false;
    if (active_.gtSsaa &&
        !deferred_.writePostFxSets(ctx_, descriptorPool_, gtSsaaColor_.view, gtSsaaMotion_.view,
                                   gtSsaaDepth_.view, gtSsaaPostFx_))
        return false;

    // Bloom pyramid sets (Phase 6a): srcColor = the path's lit HDR target.
    if (!deferred_.writeBloomPyramidSets(ctx_, descriptorPool_, gbColor_.view, gbBloom_))
        return false;
    if (!deferred_.writeBloomPyramidSets(ctx_, descriptorPool_, gtColor_.view, gtBloom_))
        return false;
    if (active_.gtSsaa &&
        !deferred_.writeBloomPyramidSets(ctx_, descriptorPool_, gtSsaaColor_.view, gtSsaaBloom_))
        return false;
    // Coverage mask dilate set: raw mask + motion -> dilated/gated target
    // (pool-owned set).
    if (!deferred_.writeReactiveDilateSet(ctx_, descriptorPool_, gbReactive_.view,
                                          gbReactiveDilated_.view, gbMotion_.view,
                                          reactiveDilateSet_))
        return false;

    if (!createAutoExposureResources()) return false;

    linearSampler_ = createSampler(ctx_, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    fontSampler_ = createSampler(ctx_, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    if (!linearSampler_ || !fontSampler_) return false;

    // --- material UBO (shared layout/fill via DeferredCore) ----------------------
    if (!deferred_.createMaterialUbo(ctx_, scene_, materialUbo_, materialUboMemory_,
                                     materialStride_))
        return false;

    // --- packed overlay text UBO ---------------------------------------------------
    const VkDeviceSize textSize = kMaxColumns * kTextCharsPerColumn * 4;
    if (createBuffer(ctx_, textSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     textUbo_, textUboMemory_) != VK_SUCCESS)
        return false;
    vmaMapMemory(ctx_.allocator, textUboMemory_, &textUboMapped_);

    // --- allocate sets -------------------------------------------------------------
    auto allocSet = [&](VkDescriptorSetLayout layout, VkDescriptorSet& set) {
        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = descriptorPool_;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &layout;
        return vkAllocateDescriptorSets(ctx_.device, &alloc, &set) == VK_SUCCESS;
    };

    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        if (!allocSet(deferred_.sceneSetLayout(), frames_[i].sceneSetGb)) return false;
        if (!allocSet(deferred_.sceneSetLayout(), frames_[i].sceneSetGbSpatial)) return false;
        if (!allocSet(deferred_.sceneSetLayout(), frames_[i].sceneSetGt)) return false;
        if (!allocSet(deferred_.lightingSetLayout(), frames_[i].lightingSetGb)) return false;
        if (!allocSet(deferred_.lightingSetLayout(), frames_[i].lightingSetGbSpatial)) return false;
        if (!allocSet(deferred_.lightingSetLayout(), frames_[i].lightingSetGt)) return false;
        if (!allocSet(deferred_.lightingSetLayout(), frames_[i].lightingSetSsaa)) return false;
        if (!allocSet(deferred_.transparentSetLayout(), frames_[i].transparentSetGb)) return false;
        if (!allocSet(deferred_.transparentSetLayout(), frames_[i].transparentSetGbSpatial))
            return false;
        if (!allocSet(deferred_.transparentSetLayout(), frames_[i].transparentSetGt)) return false;
        if (!allocSet(deferred_.transparentSetLayout(), frames_[i].transparentSetSsaa))
            return false;
        if (!allocSet(deferred_.ssrSetLayout(), frames_[i].ssrSetGb)) return false;
        if (!allocSet(deferred_.ssrSetLayout(), frames_[i].ssrSetGbSpatial)) return false;
        if (!allocSet(deferred_.ssrSetLayout(), frames_[i].ssrSetGt)) return false;
        if (!allocSet(deferred_.ssrSetLayout(), frames_[i].ssrSetSsaa)) return false;
    }
    if (!allocSet(deferred_.textureSetLayout(), textureSet_)) return false;
    if (!allocSet(deferred_.ssaoSetLayout(), ssaoSetGb_)) return false;
    if (!allocSet(deferred_.ssaoSetLayout(), ssaoSetGt_)) return false;
    if (active_.gtSsaa) {
        if (!allocSet(deferred_.ssaoSetLayout(), ssaoSetSsaa_)) return false;
    }
    if (!allocSet(composeSetLayout_, gtComposeSet_)) return false;
    if (!allocSet(copySetLayout_, copySet_)) return false;
    if (active_.gtSsaa && !allocSet(copySetLayout_, gtDownsampleSet_)) return false;
    // (Per-algo compose/metric sets are allocated by createAlgoResources.)

    // --- writes ---------------------------------------------------------------------
    deferred_.writeTextureSet(ctx_, textureSet_, scene_);

    writeComposeSetInto(gtComposeSet_, gtColor_.view);

    {
        VkDescriptorImageInfo info = {};
        info.sampler = linearSampler_;
        info.imageView = composeImage_.view;
        info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = copySet_;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &info;
        vkUpdateDescriptorSets(ctx_.device, 1, &write, 0, nullptr);
    }

    // SSAA downsample source (2x GT), sampled with the linear sampler: at an
    // exact 2:1 ratio bilinear weights land on a 2x2 box filter.
    if (active_.gtSsaa) {
        VkDescriptorImageInfo info = {};
        info.sampler = linearSampler_;
        info.imageView = gtSsaaColor_.view;
        info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = gtDownsampleSet_;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &info;
        vkUpdateDescriptorSets(ctx_.device, 1, &write, 0, nullptr);
    }

    // Phase 7a cull resources: instance SSBO + per-path cull channels bound to
    // the depth pyramids (created by createRenderTargets above); scene sets
    // are already allocated, so binding 3 can be written here.
    if (!createCullResources()) return false;

    return true;
}

void GuiApp::writeComposeSetInto(VkDescriptorSet set, VkImageView source) {
    VkDescriptorImageInfo srcInfo = {};
    srcInfo.sampler = linearSampler_;
    srcInfo.imageView = source;
    srcInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorBufferInfo textInfo = {};
    textInfo.buffer = textUbo_;
    textInfo.offset = 0;
    textInfo.range = kMaxColumns * kTextCharsPerColumn * 4;
    VkDescriptorImageInfo fontInfo = {};
    fontInfo.sampler = fontSampler_;
    fontInfo.imageView = fontAtlas_.view;
    fontInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo lutInfo = {};
    lutInfo.sampler = gradingLut_.sampler();
    lutInfo.imageView = gradingLut_.view();
    lutInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet writes[4] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = set;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &srcInfo;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = set;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[1].pBufferInfo = &textInfo;
    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = set;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].pImageInfo = &fontInfo;
    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = set;
    writes[3].dstBinding = 3;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[3].pImageInfo = &lutInfo;
    vkUpdateDescriptorSets(ctx_.device, 4, writes, 0, nullptr);
}

bool GuiApp::createAlgoResources(AlgoColumn& algo, uint32_t index) {
    // Display-resolution output (storage during dispatch).
    algo.output.width = active_.displayW;
    algo.output.height = active_.displayH;
    algo.output.format = deferred::kHdrColorFormat;
    if (createImage(ctx_, active_.displayW, active_.displayH, deferred::kHdrColorFormat,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, algo.output.image,
                    algo.output.memory) != VK_SUCCESS)
        return false;
    algo.output.view = createImageView(ctx_, algo.output.image, deferred::kHdrColorFormat,
                                       VK_IMAGE_ASPECT_COLOR_BIT);
    if (!algo.output.view) return false;
    algo.outputLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    // PSNR/SSIM reduction scratch (compare mode only; blockCount_ is already
    // computed by createMetricResources).
    if (active_.mode == Mode::Compare) {
        const VkDeviceSize blocksSize =
            static_cast<VkDeviceSize>(blockCount_) * kMetricFloats * 4;
        if (createBuffer(ctx_, blocksSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, algo.blocksBuffer,
                         algo.blocksMemory) != VK_SUCCESS)
            return false;
    }

    // Descriptor sets (the pool has FREE_DESCRIPTOR_SET: destroyAlgoResources
    // returns these individually on the algo-only rebuild path).
    auto allocSet = [&](VkDescriptorSetLayout layout, VkDescriptorSet& set) {
        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = descriptorPool_;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &layout;
        return vkAllocateDescriptorSets(ctx_.device, &alloc, &set) == VK_SUCCESS;
    };
    if (!allocSet(composeSetLayout_, algo.composeSet)) return false;
    if (active_.mode == Mode::Compare && !allocSet(metricSetLayout_, algo.metricSet))
        return false;

    writeComposeSetInto(algo.composeSet, algo.output.view);

    if (algo.frameGen) {
        algo.fgOutput.width = active_.displayW;
        algo.fgOutput.height = active_.displayH;
        algo.fgOutput.format = deferred::kHdrColorFormat;
        if (createImage(ctx_, active_.displayW, active_.displayH, deferred::kHdrColorFormat,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, algo.fgOutput.image,
                        algo.fgOutput.memory) != VK_SUCCESS)
            return false;
        algo.fgOutput.view = createImageView(ctx_, algo.fgOutput.image, deferred::kHdrColorFormat,
                                             VK_IMAGE_ASPECT_COLOR_BIT);
        if (!algo.fgOutput.view) return false;
        algo.fgOutputLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (!allocSet(composeSetLayout_, algo.composeSetFg)) return false;
        writeComposeSetInto(algo.composeSetFg, algo.fgOutput.view);
    }

    if (active_.mode == Mode::Compare) {
        VkDescriptorImageInfo testInfo = {};
        testInfo.sampler = linearSampler_;
        testInfo.imageView = algo.frameGen ? algo.fgOutput.view : algo.output.view;
        testInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo refInfo = {};
        refInfo.sampler = linearSampler_;
        refInfo.imageView = gtColor_.view;
        refInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorBufferInfo blocksInfo = {};
        blocksInfo.buffer = algo.blocksBuffer;
        blocksInfo.offset = 0;
        blocksInfo.range = VK_WHOLE_SIZE;
        VkDescriptorBufferInfo resultInfo = {};
        resultInfo.buffer = metricResultBuf_;
        resultInfo.offset = index * kMetricFloats * 4;
        resultInfo.range = kMetricFloats * 4;

        VkWriteDescriptorSet writes[4] = {};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = algo.metricSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &testInfo;
        writes[1] = writes[0];
        writes[1].dstBinding = 1;
        writes[1].pImageInfo = &refInfo;
        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = algo.metricSet;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &blocksInfo;
        writes[3] = writes[2];
        writes[3].dstBinding = 3;
        writes[3].pBufferInfo = &resultInfo;
        vkUpdateDescriptorSets(ctx_.device, 4, writes, 0, nullptr);
    }
    return true;
}

bool GuiApp::createAutoExposureResources() {
    // Seed both solvers with the current (preset/slider) exposure so the
    // frames before the first readback match the fixed-exposure look.
    const float initialEV = -std::log2(std::max(exposure_, 1e-4f));
    if (!deferred_.createExposureChannel(ctx_, descriptorPool_, gbColor_.view, renderWidth_,
                                         renderHeight_, initialEV, lrExposure_))
        return false;
    const ImageResource& gtSrc = active_.gtSsaa ? gtSsaaColor_ : gtColor_;
    return deferred_.createExposureChannel(ctx_, descriptorPool_, gtSrc.view, gtSrc.width,
                                           gtSrc.height, initialEV, gtExposure_);
}

bool GuiApp::createCullResources() {
    // Phase 7a: shared instance SSBO + one cull channel per path (LR / GT /
    // GT-SSAA), each bound to that path's Hi-Z chain.  Capacity covers the
    // full instance list (the candidate build skips blend/skinned/culled).
    const uint32_t capacity = static_cast<uint32_t>(scene_.instances.size());
    if (!deferred_.createInstanceBuffer(ctx_, capacity, instances_)) return false;
    if (!deferred_.createCullChannel(ctx_, capacity, gbCull_)) return false;
    if (!deferred_.createCullChannel(ctx_, capacity, gtCull_)) return false;
    if (!deferred_.writeCullSet(ctx_, descriptorPool_, instances_, gbPyramid_.chainView, gbCull_))
        return false;
    if (!deferred_.writeCullSet(ctx_, descriptorPool_, instances_, gtPyramid_.chainView, gtCull_))
        return false;
    if (active_.gtSsaa) {
        if (!deferred_.createCullChannel(ctx_, capacity, gtSsaaCull_)) return false;
        if (!deferred_.writeCullSet(ctx_, descriptorPool_, instances_, gtSsaaPyramid_.chainView,
                                    gtSsaaCull_))
            return false;
    }
    // Scene set binding 3 (instance SSBO) for every slot's GB/spatial/GT set.
    if (instances_.buffer) {
        for (uint32_t i = 0; i < kFramesInFlight; ++i) {
            deferred_.writeSceneInstanceBinding(ctx_, frames_[i].sceneSetGb, instances_.buffer);
            deferred_.writeSceneInstanceBinding(ctx_, frames_[i].sceneSetGbSpatial,
                                                instances_.buffer);
            deferred_.writeSceneInstanceBinding(ctx_, frames_[i].sceneSetGt, instances_.buffer);
        }
    }
    cullInstCpu_.resize(capacity);
    cullCmdCpu_.resize(capacity);
    return true;
}

void GuiApp::destroyCullResources() {
    deferred_.destroyCullChannel(ctx_, gbCull_);
    deferred_.destroyCullChannel(ctx_, gtCull_);
    deferred_.destroyCullChannel(ctx_, gtSsaaCull_);
    deferred_.destroyInstanceBuffer(ctx_, instances_);
}

bool GuiApp::createPipelines() {
    VkPushConstantRange composePushRange = {};
    composePushRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    composePushRange.offset = 0;
    composePushRange.size = sizeof(ComposePush);
    VkPipelineLayoutCreateInfo composeLayoutCi = {};
    composeLayoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    composeLayoutCi.setLayoutCount = 1;
    composeLayoutCi.pSetLayouts = &composeSetLayout_;
    composeLayoutCi.pushConstantRangeCount = 1;
    composeLayoutCi.pPushConstantRanges = &composePushRange;
    if (vkCreatePipelineLayout(ctx_.device, &composeLayoutCi, nullptr, &composePipelineLayout_) != VK_SUCCESS)
        return false;

    VkPushConstantRange copyPushRange = {};
    copyPushRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    copyPushRange.offset = 0;
    copyPushRange.size = 16; // vec4: hdr mode + paper white (copy.frag)
    VkPipelineLayoutCreateInfo copyLayoutCi = {};
    copyLayoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    copyLayoutCi.setLayoutCount = 1;
    copyLayoutCi.pSetLayouts = &copySetLayout_;
    copyLayoutCi.pushConstantRangeCount = 1;
    copyLayoutCi.pPushConstantRanges = &copyPushRange;
    if (vkCreatePipelineLayout(ctx_.device, &copyLayoutCi, nullptr, &copyPipelineLayout_) != VK_SUCCESS)
        return false;

    VkPushConstantRange metricPushRange = {};
    metricPushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    metricPushRange.offset = 0;
    metricPushRange.size = sizeof(MetricPush);
    VkPipelineLayoutCreateInfo metricLayoutCi = {};
    metricLayoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    metricLayoutCi.setLayoutCount = 1;
    metricLayoutCi.pSetLayouts = &metricSetLayout_;
    metricLayoutCi.pushConstantRangeCount = 1;
    metricLayoutCi.pPushConstantRanges = &metricPushRange;
    if (vkCreatePipelineLayout(ctx_.device, &metricLayoutCi, nullptr, &metricPipelineLayout_) != VK_SUCCESS)
        return false;

    // --- shared fullscreen state (compose/copy/downsample) ----------------------
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.f;

    VkPipelineMultisampleStateCreateInfo multisample = {};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAttachments[1] = {};
    blendAttachments[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo colorBlend = {};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = blendAttachments;

    VkDynamicState dynamicStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState = {};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    // --- fullscreen pipelines (compose + swapchain copy) ------------------------
    VkPipelineVertexInputStateCreateInfo emptyVertexInput = {};
    emptyVertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineDepthStencilStateCreateInfo noDepth = {};
    noDepth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

    VkPipelineShaderStageCreateInfo fsStages[2] = {};
    fsStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fsStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    fsStages[0].module = fullscreenVert_;
    fsStages[0].pName = "main";
    fsStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fsStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fsStages[1].module = composeFrag_;
    fsStages[1].pName = "main";

    VkGraphicsPipelineCreateInfo fsCi = {};
    fsCi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    fsCi.stageCount = 2;
    fsCi.pStages = fsStages;
    fsCi.pVertexInputState = &emptyVertexInput;
    fsCi.pInputAssemblyState = &inputAssembly;
    fsCi.pViewportState = &viewportState;
    fsCi.pRasterizationState = &rasterizer;
    fsCi.pMultisampleState = &multisample;
    fsCi.pDepthStencilState = &noDepth;
    fsCi.pColorBlendState = &colorBlend;
    fsCi.pDynamicState = &dynamicState;

    VkPipelineRenderingCreateInfo composeRendering = {};
    composeRendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    composeRendering.colorAttachmentCount = 1;
    composeRendering.pColorAttachmentFormats = &kComposeFormat;
    fsCi.pNext = &composeRendering;
    fsCi.layout = composePipelineLayout_;
    if (createGraphicsPipeline(ctx_, fsCi, composePipeline_) != VK_SUCCESS)
        return false;

    if (!createCopyPipeline()) return false;

    // GT SSAA downsample: same passthrough fragment shader, HDR target.
    fsStages[1].module = copyFrag_;
    fsCi.layout = copyPipelineLayout_;
    VkPipelineRenderingCreateInfo downsampleRendering = {};
    downsampleRendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    downsampleRendering.colorAttachmentCount = 1;
    downsampleRendering.pColorAttachmentFormats = &deferred::kHdrColorFormat;
    fsCi.pNext = &downsampleRendering;
    if (createGraphicsPipeline(ctx_, fsCi, downsamplePipeline_) != VK_SUCCESS)
        return false;

    // --- metric compute pipelines --------------------------------------------------
    VkComputePipelineCreateInfo compCi = {};
    compCi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compCi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compCi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compCi.stage.module = metricBlocksComp_;
    compCi.stage.pName = "main";
    compCi.layout = metricPipelineLayout_;
    if (createComputePipeline(ctx_, compCi, metricBlocksPipeline_) != VK_SUCCESS)
        return false;
    compCi.stage.module = metricReduceComp_;
    if (createComputePipeline(ctx_, compCi, metricReducePipeline_) != VK_SUCCESS)
        return false;

    return true;
}

bool GuiApp::createCopyPipeline() {
    // Swapchain copy (compare_copy.frag into the swapchain image).  Factored
    // out of createPipelines: setHdrEnabled re-creates just this pipeline
    // when the swapchain format changes (Phase 6c).
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.f;
    VkPipelineMultisampleStateCreateInfo multisample = {};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState blend = {};
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo colorBlend = {};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &blend;
    VkDynamicState dynamicStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState = {};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;
    VkPipelineVertexInputStateCreateInfo emptyVertexInput = {};
    emptyVertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineDepthStencilStateCreateInfo noDepth = {};
    noDepth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = fullscreenVert_;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = copyFrag_;
    stages[1].pName = "main";

    VkPipelineRenderingCreateInfo rendering = {};
    rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering.colorAttachmentCount = 1;
    const VkFormat presentFormat = swapchain_.format();
    rendering.pColorAttachmentFormats = &presentFormat;

    VkGraphicsPipelineCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    ci.stageCount = 2;
    ci.pStages = stages;
    ci.pVertexInputState = &emptyVertexInput;
    ci.pInputAssemblyState = &inputAssembly;
    ci.pViewportState = &viewportState;
    ci.pRasterizationState = &rasterizer;
    ci.pMultisampleState = &multisample;
    ci.pDepthStencilState = &noDepth;
    ci.pColorBlendState = &colorBlend;
    ci.pDynamicState = &dynamicState;
    ci.pNext = &rendering;
    ci.layout = copyPipelineLayout_;
    return createGraphicsPipeline(ctx_, ci, copyPipeline_) == VK_SUCCESS;
}

void GuiApp::setHdrEnabled(bool enabled) {
    if (hdrEnabled_ == enabled) return;
    hdrEnabled_ = enabled;
    vkDeviceWaitIdle(ctx_.device);
    // Everything that bakes the swapchain format is re-created: the swapchain
    // itself, the copy pipeline and the ImGui backend pipeline.
    if (!recreateGuiSwapchain()) {
        std::fprintf(stderr, "gui: hdr swapchain re-creation failed, reverting to SDR\n");
        hdrEnabled_ = false;
        recreateGuiSwapchain();
    }
    if (copyPipeline_) {
        vkDestroyPipeline(ctx_.device, copyPipeline_, nullptr);
        copyPipeline_ = VK_NULL_HANDLE;
    }
    if (stackOk_ && !createCopyPipeline())
        std::fprintf(stderr, "gui: copy pipeline re-creation failed\n");
    ImGui_ImplVulkan_Shutdown();
    initImGuiVulkanBackend();
    // The debug UI-shot target bakes the swapchain format too.
    if (uiShot_ && uiShotImage_.image) {
        uiShotImage_.destroy(ctx_);
        uiShotLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        uiShotImage_.width = active_.displayW;
        uiShotImage_.height = active_.displayH;
        uiShotImage_.format = swapchain_.format();
        if (createImage(ctx_, uiShotImage_.width, uiShotImage_.height, uiShotImage_.format,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                        uiShotImage_.image, uiShotImage_.memory) == VK_SUCCESS) {
            uiShotImage_.view = createImageView(ctx_, uiShotImage_.image, uiShotImage_.format,
                                                VK_IMAGE_ASPECT_COLOR_BIT);
        }
    }
}

bool GuiApp::createSyncResources() {
    VkCommandBufferAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = ctx_.framePool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = kFramesInFlight;
    VkCommandBuffer cmds[kFramesInFlight] = {};
    if (vkAllocateCommandBuffers(ctx_.device, &alloc, cmds) != VK_SUCCESS) return false;
    for (uint32_t i = 0; i < kFramesInFlight; ++i) frames_[i].cmd = cmds[i];

    VkSemaphoreCreateInfo semCi = {};
    semCi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fenceCi = {};
    fenceCi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCi.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    const VkDeviceSize uboSize = sizeof(SceneUBO);
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        FrameResources& fr = frames_[i];
        if (vkCreateSemaphore(ctx_.device, &semCi, nullptr, &fr.imageAvailable) != VK_SUCCESS) return false;
        if (vkCreateFence(ctx_.device, &fenceCi, nullptr, &fr.fence) != VK_SUCCESS) return false;

        VkBuffer ubos[3] = {};
        VmaAllocation uboMems[3] = {};
        void* uboMaps[3] = {};
        VkDescriptorSet sets[3] = {fr.sceneSetGb, fr.sceneSetGbSpatial, fr.sceneSetGt};
        for (int k = 0; k < 3; ++k) {
            if (createBuffer(ctx_, uboSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             ubos[k], uboMems[k]) != VK_SUCCESS)
                return false;
            vmaMapMemory(ctx_.allocator, uboMems[k], &uboMaps[k]);

            VkDescriptorBufferInfo sceneBuf = {};
            sceneBuf.buffer = ubos[k];
            sceneBuf.offset = 0;
            sceneBuf.range = uboSize;
            VkDescriptorBufferInfo materialBuf = {};
            materialBuf.buffer = materialUbo_;
            materialBuf.offset = 0;
            materialBuf.range = sizeof(MaterialUBO);

            VkWriteDescriptorSet writes[2] = {};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = sets[k];
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].pBufferInfo = &sceneBuf;
            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = sets[k];
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            writes[1].pBufferInfo = &materialBuf;
            vkUpdateDescriptorSets(ctx_.device, 2, writes, 0, nullptr);

            // Joint palette for skinned draws (per-slot buffer, matching the
            // slot advanceToFrame(frameIndex) writes).
            if (scene_.hasSkinnedMeshes())
                deferred_.writeSceneSkinBinding(ctx_, sets[k], scene_.skinPalette(i));
        }
        fr.uboGb = ubos[0]; fr.uboGbMemory = uboMems[0]; fr.uboGbMapped = uboMaps[0];
        fr.uboGbSpatial = ubos[1]; fr.uboGbSpatialMemory = uboMems[1];
        fr.uboGbSpatialMapped = uboMaps[1];
        fr.uboGt = ubos[2]; fr.uboGtMemory = uboMems[2]; fr.uboGtMapped = uboMaps[2];

        // Lighting UBOs (GB jittered / GT+SSAA un-jittered) + lighting sets.
        const VkDeviceSize lightingSize = sizeof(LightingUBO);
        if (createBuffer(ctx_, lightingSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         fr.lightingUboGb, fr.lightingUboGbMemory) != VK_SUCCESS)
            return false;
        vmaMapMemory(ctx_.allocator, fr.lightingUboGbMemory,
                    &fr.lightingUboGbMapped);
        if (createBuffer(ctx_, lightingSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         fr.lightingUboGbSpatial, fr.lightingUboGbSpatialMemory) != VK_SUCCESS)
            return false;
        vmaMapMemory(ctx_.allocator, fr.lightingUboGbSpatialMemory,
                    &fr.lightingUboGbSpatialMapped);
        if (createBuffer(ctx_, lightingSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         fr.lightingUboGt, fr.lightingUboGtMemory) != VK_SUCCESS)
            return false;
        vmaMapMemory(ctx_.allocator, fr.lightingUboGtMemory,
                    &fr.lightingUboGtMapped);

        // Shadow map binding: the array view is always bound when the targets
        // exist; the "shadows" checkbox only zeroes shadowParams.z (sampling
        // off) via fillLightingUBO.  VK_NULL_HANDLE (creation failed) leaves
        // binding 11 unwritten, which is safe only while shadows stay off.
        const VkImageView shadowView = shadowsActive_ ? shadow_.arrayView : VK_NULL_HANDLE;
        const VkImageView spotAtlasView = spotAtlasActive_ ? spotAtlas_.view : VK_NULL_HANDLE;
        deferred_.writeLightingSet(ctx_, fr.lightingSetGb, fr.lightingUboGb, gbAlbedo_.view,
                                   gbNormal_.view, gbMaterial_.view, gbEmissive_.view,
                                   gbDepth_.view, gbAo_.view, shadowView, spotAtlasView,
                                   gbCluster_.lightsBuffer[i], gbCluster_.gridBuffer[i]);
        deferred_.writeLightingSet(ctx_, fr.lightingSetGbSpatial, fr.lightingUboGbSpatial,
                                   gbAlbedo_.view, gbNormal_.view, gbMaterial_.view,
                                   gbEmissive_.view, gbDepth_.view, gbAo_.view, shadowView,
                                   spotAtlasView,
                                   gbCluster_.lightsBuffer[i], gbCluster_.gridBuffer[i]);
        deferred_.writeLightingSet(ctx_, fr.lightingSetGt, fr.lightingUboGt, gtAlbedo_.view,
                                   gtNormal_.view, gtMaterial_.view, gtEmissive_.view,
                                   gtDepth_.view, gtAo_.view, shadowView, spotAtlasView,
                                   gtCluster_.lightsBuffer[i], gtCluster_.gridBuffer[i]);
        if (active_.gtSsaa) {
            // GT and GT-SSAA share the same (resolution-independent) UBO.
            deferred_.writeLightingSet(ctx_, fr.lightingSetSsaa, fr.lightingUboGt,
                                       gtSsaaAlbedo_.view, gtSsaaNormal_.view,
                                       gtSsaaMaterial_.view, gtSsaaEmissive_.view,
                                       gtSsaaDepth_.view, gtSsaaAo_.view, shadowView, spotAtlasView,
                                       gtSsaaCluster_.lightsBuffer[i], gtSsaaCluster_.gridBuffer[i]);
        }

        // The transparency shader reads iblParams (identical in both lighting
        // UBOs) plus the path's own SSAO texture: one set per path.  Binding 8
        // is the path's ray-integrated froxel volume (volumetric fog on
        // translucency); the spatial LR variant shares gbFog_ (same rule as
        // the fog light sets).  Written whenever the volume exists — the
        // per-frame checkbox gate lives in the record-side push constant.
        deferred_.writeTransparentSet(ctx_, fr.transparentSetGb, fr.lightingUboGb, gbAo_.view,
                                      shadowView, gbColorPyramid_.chainView,
                                      gbPyramid_.chainView, gbFog_.intView);
        deferred_.writeTransparentSet(ctx_, fr.transparentSetGbSpatial, fr.lightingUboGbSpatial,
                                      gbAo_.view, shadowView, gbColorPyramid_.chainView,
                                      gbPyramid_.chainView, gbFog_.intView);
        deferred_.writeTransparentSet(ctx_, fr.transparentSetGt, fr.lightingUboGt, gtAo_.view,
                                      shadowView, gtColorPyramid_.chainView,
                                      gtPyramid_.chainView, gtFog_.intView);
        if (active_.gtSsaa) {
            deferred_.writeTransparentSet(ctx_, fr.transparentSetSsaa, fr.lightingUboGt,
                                          gtSsaaAo_.view, shadowView,
                                          gtSsaaColorPyramid_.chainView,
                                          gtSsaaPyramid_.chainView, gtSsaaFog_.intView);
        }

        // Opaque-SSR trace sets: binding 0 reuses the path's lighting UBO; the
        // rest is the GBuffer + SSAO + pyramids + the path's trace target
        // (write-only storage; the temporal pass owns the lit-target RMW).
        deferred_.writeSsrSet(ctx_, fr.ssrSetGb, fr.lightingUboGb, gbAlbedo_.view,
                              gbNormal_.view, gbMaterial_.view, gbDepth_.view, gbAo_.view,
                              gbColorPyramid_.chainView, gbPyramid_.chainView, gbSsrTrace_.view);
        deferred_.writeSsrSet(ctx_, fr.ssrSetGbSpatial, fr.lightingUboGbSpatial, gbAlbedo_.view,
                              gbNormal_.view, gbMaterial_.view, gbDepth_.view, gbAo_.view,
                              gbColorPyramid_.chainView, gbPyramid_.chainView, gbSsrTrace_.view);
        deferred_.writeSsrSet(ctx_, fr.ssrSetGt, fr.lightingUboGt, gtAlbedo_.view,
                              gtNormal_.view, gtMaterial_.view, gtDepth_.view, gtAo_.view,
                              gtColorPyramid_.chainView, gtPyramid_.chainView, gtSsrTrace_.view);
        if (active_.gtSsaa) {
            deferred_.writeSsrSet(ctx_, fr.ssrSetSsaa, fr.lightingUboGt, gtSsaaAlbedo_.view,
                                  gtSsaaNormal_.view, gtSsaaMaterial_.view, gtSsaaDepth_.view,
                                  gtSsaaAo_.view, gtSsaaColorPyramid_.chainView,
                                  gtSsaaPyramid_.chainView, gtSsaaSsrTrace_.view);
        }
    }

    // Froxel volumetric fog sets (Phase 5a), one writeVolFogSets per path.
    // GB light sets bind the non-spatial lightingUboGb for both slots (the
    // fog light pass only reads camera-independent fields; see CompareApp).
    if (gbFog_.injectImage != VK_NULL_HANDLE) {
        const VkImageView shadowView = shadowsActive_ ? shadow_.arrayView : VK_NULL_HANDLE;
        const VkImageView spotAtlasView = spotAtlasActive_ ? spotAtlas_.view : VK_NULL_HANDLE;
        const VkBuffer gbFogUbos[kClusterSlots] = {frames_[0].lightingUboGb,
                                                   frames_[1].lightingUboGb};
        const VkBuffer gtFogUbos[kClusterSlots] = {frames_[0].lightingUboGt,
                                                   frames_[1].lightingUboGt};
        if (!deferred_.writeVolFogSets(ctx_, descriptorPool_, gbFog_, gbCluster_, gbFogUbos,
                                       shadowView, spotAtlasView, gbDepth_.view, gbColor_.view))
            return false;
        if (!deferred_.writeVolFogSets(ctx_, descriptorPool_, gtFog_, gtCluster_, gtFogUbos,
                                       shadowView, spotAtlasView, gtDepth_.view, gtColor_.view))
            return false;
        if (active_.gtSsaa && gtSsaaFog_.injectImage != VK_NULL_HANDLE &&
            !deferred_.writeVolFogSets(ctx_, descriptorPool_, gtSsaaFog_, gtSsaaCluster_,
                                       gtFogUbos, shadowView, spotAtlasView, gtSsaaDepth_.view,
                                       gtSsaaColor_.view))
            return false;
    }

    // SSAO sets (static bindings; per-frame data goes through push constants).
    deferred_.writeSsaoSet(ctx_, ssaoSetGb_, gbPyramidAo_.chainView, gbNormal_.view,
                           gbAoRaw_.view);
    deferred_.writeSsaoSet(ctx_, ssaoSetGt_, gtPyramidAo_.chainView, gtNormal_.view,
                           gtAoRaw_.view);
    if (!deferred_.writeAoHistorySets(ctx_, descriptorPool_, gbAoRaw_.view, gbDepth_.view,
                                      gbAo_.view, gbAoHist_))
        return false;
    if (!deferred_.writeAoHistorySets(ctx_, descriptorPool_, gtAoRaw_.view, gtDepth_.view,
                                      gtAo_.view, gtAoHist_))
        return false;
    if (active_.gtSsaa) {
        deferred_.writeSsaoSet(ctx_, ssaoSetSsaa_, gtSsaaPyramidAo_.chainView,
                               gtSsaaNormal_.view, gtSsaaAoRaw_.view);
        if (!deferred_.writeAoHistorySets(ctx_, descriptorPool_, gtSsaaAoRaw_.view,
                                          gtSsaaDepth_.view, gtSsaaAo_.view, gtSsaaAoHist_))
            return false;
    }
    if (!deferred_.writeSsrHistorySets(ctx_, descriptorPool_, gbSsrTrace_.view, gbDepth_.view,
                                       gbColor_.view, gbSsrHist_))
        return false;
    if (!deferred_.writeSsrHistorySets(ctx_, descriptorPool_, gtSsrTrace_.view, gtDepth_.view,
                                       gtColor_.view, gtSsrHist_))
        return false;
    if (active_.gtSsaa) {
        if (!deferred_.writeSsrHistorySets(ctx_, descriptorPool_, gtSsaaSsrTrace_.view,
                                           gtSsaaDepth_.view, gtSsaaColor_.view, gtSsaaSsrHist_))
            return false;
    }
    return true;
}

void GuiApp::ensurePresentSemaphores() {
    // Recreate the per-swapchain-image present semaphores if the swapchain
    // image count changed across a recreation.
    if (renderFinished_.size() == swapchain_.imageCount()) return;
    for (VkSemaphore sem : renderFinished_) {
        if (sem) vkDestroySemaphore(ctx_.device, sem, nullptr);
    }
    renderFinished_.assign(swapchain_.imageCount(), VK_NULL_HANDLE);
    VkSemaphoreCreateInfo semCi = {};
    semCi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (size_t i = 0; i < renderFinished_.size(); ++i)
        vkCreateSemaphore(ctx_.device, &semCi, nullptr, &renderFinished_[i]);
}

bool GuiApp::createScreenshotStaging() {
    const VkDeviceSize size = static_cast<VkDeviceSize>(active_.displayW) * active_.displayH * 4;
    if (createBuffer(ctx_, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     screenshotStaging_, screenshotStagingMemory_) != VK_SUCCESS)
        return false;
    vmaMapMemory(ctx_.allocator, screenshotStagingMemory_, &screenshotMapped_);
    return true;
}

void GuiApp::destroyAlgoResources() {
    for (AlgoColumn& algo : algos_) {
        if (algo.upscaler) { algo.upscaler->shutdown(); algo.upscaler.reset(); }
        if (algo.frameGen) { algo.frameGen->shutdown(); algo.frameGen.reset(); }
        algo.output.destroy(ctx_);
        algo.fgOutput.destroy(ctx_);
        algo.fgOutputLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        algo.fgHistory = false;
        algo.fgReady = false;
        if (algo.blocksBuffer) {
            vmaDestroyBuffer(ctx_.allocator, algo.blocksBuffer, algo.blocksMemory);
            algo.blocksBuffer = VK_NULL_HANDLE;
            algo.blocksMemory = VK_NULL_HANDLE;
        }
        // The pool is created with FREE_DESCRIPTOR_SET: return per-algo sets
        // individually so the algo-only rebuild path can keep the pool (and
        // every other set in it) alive.
        if (descriptorPool_) {
            if (algo.composeSet) {
                vkFreeDescriptorSets(ctx_.device, descriptorPool_, 1, &algo.composeSet);
                algo.composeSet = VK_NULL_HANDLE;
            }
            if (algo.composeSetFg) {
                vkFreeDescriptorSets(ctx_.device, descriptorPool_, 1, &algo.composeSetFg);
                algo.composeSetFg = VK_NULL_HANDLE;
            }
            if (algo.metricSet) {
                vkFreeDescriptorSets(ctx_.device, descriptorPool_, 1, &algo.metricSet);
                algo.metricSet = VK_NULL_HANDLE;
            }
        }
    }
    algos_.clear();
}

void GuiApp::destroyStackResources() {
    if (!ctx_.device) return;
    vkDeviceWaitIdle(ctx_.device);

    timestamps_.destroy(ctx_);
    profiler_.destroy(ctx_);

    destroyAlgoResources();

    if (screenshotStaging_) {
        vmaDestroyBuffer(ctx_.allocator, screenshotStaging_, screenshotStagingMemory_);
        screenshotStaging_ = VK_NULL_HANDLE;
        screenshotStagingMemory_ = VK_NULL_HANDLE;
        screenshotMapped_ = nullptr;
    }
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        if (metricStaging_[i]) {
            vmaDestroyBuffer(ctx_.allocator, metricStaging_[i], metricStagingMemory_[i]);
            metricStaging_[i] = VK_NULL_HANDLE;
            metricStagingMemory_[i] = VK_NULL_HANDLE;
            metricStagingMapped_[i] = nullptr;
        }
        metricPending_[i] = false;
    }
    if (metricResultBuf_) {
        vmaDestroyBuffer(ctx_.allocator, metricResultBuf_, metricResultMemory_);
        metricResultBuf_ = VK_NULL_HANDLE;
        metricResultMemory_ = VK_NULL_HANDLE;
    }

    deferred_.destroyExposureChannel(ctx_, lrExposure_);
    deferred_.destroyExposureChannel(ctx_, gtExposure_);
    if (textUbo_) {
        vmaDestroyBuffer(ctx_.allocator, textUbo_, textUboMemory_);
        textUbo_ = VK_NULL_HANDLE;
        textUboMemory_ = VK_NULL_HANDLE;
        textUboMapped_ = nullptr;
    }

    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        FrameResources& fr = frames_[i];
        if (fr.cmd) {
            vkFreeCommandBuffers(ctx_.device, ctx_.framePool, 1, &fr.cmd);
            fr.cmd = VK_NULL_HANDLE;
        }
        if (fr.uboGb) {
            vmaDestroyBuffer(ctx_.allocator, fr.uboGb, fr.uboGbMemory);
            fr.uboGb = VK_NULL_HANDLE;
            fr.uboGbMemory = VK_NULL_HANDLE;
            fr.uboGbMapped = nullptr;
        }
        if (fr.uboGbSpatial) {
            vmaDestroyBuffer(ctx_.allocator, fr.uboGbSpatial, fr.uboGbSpatialMemory);
            fr.uboGbSpatial = VK_NULL_HANDLE;
            fr.uboGbSpatialMemory = VK_NULL_HANDLE;
            fr.uboGbSpatialMapped = nullptr;
        }
        if (fr.uboGt) {
            vmaDestroyBuffer(ctx_.allocator, fr.uboGt, fr.uboGtMemory);
            fr.uboGt = VK_NULL_HANDLE;
            fr.uboGtMemory = VK_NULL_HANDLE;
            fr.uboGtMapped = nullptr;
        }
        if (fr.lightingUboGb) {
            vmaDestroyBuffer(ctx_.allocator, fr.lightingUboGb, fr.lightingUboGbMemory);
            fr.lightingUboGb = VK_NULL_HANDLE;
            fr.lightingUboGbMemory = VK_NULL_HANDLE;
            fr.lightingUboGbMapped = nullptr;
        }
        if (fr.lightingUboGbSpatial) {
            vmaDestroyBuffer(ctx_.allocator, fr.lightingUboGbSpatial, fr.lightingUboGbSpatialMemory);
            fr.lightingUboGbSpatial = VK_NULL_HANDLE;
            fr.lightingUboGbSpatialMemory = VK_NULL_HANDLE;
            fr.lightingUboGbSpatialMapped = nullptr;
        }
        if (fr.lightingUboGt) {
            vmaDestroyBuffer(ctx_.allocator, fr.lightingUboGt, fr.lightingUboGtMemory);
            fr.lightingUboGt = VK_NULL_HANDLE;
            fr.lightingUboGtMemory = VK_NULL_HANDLE;
            fr.lightingUboGtMapped = nullptr;
        }
        fr.sceneSetGb = VK_NULL_HANDLE;
        fr.sceneSetGbSpatial = VK_NULL_HANDLE;
        fr.sceneSetGt = VK_NULL_HANDLE;
        fr.lightingSetGb = VK_NULL_HANDLE;
        fr.lightingSetGbSpatial = VK_NULL_HANDLE;
        fr.lightingSetGt = VK_NULL_HANDLE;
        fr.lightingSetSsaa = VK_NULL_HANDLE;
        fr.transparentSetGb = VK_NULL_HANDLE;
        fr.transparentSetGbSpatial = VK_NULL_HANDLE;
        fr.transparentSetGt = VK_NULL_HANDLE;
        fr.transparentSetSsaa = VK_NULL_HANDLE;
        if (fr.imageAvailable) { vkDestroySemaphore(ctx_.device, fr.imageAvailable, nullptr); fr.imageAvailable = VK_NULL_HANDLE; }
        if (fr.fence) { vkDestroyFence(ctx_.device, fr.fence, nullptr); fr.fence = VK_NULL_HANDLE; }
    }
    // renderFinished_ persists across rebuilds (owned by createUiSync).

    if (composePipeline_) { vkDestroyPipeline(ctx_.device, composePipeline_, nullptr); composePipeline_ = VK_NULL_HANDLE; }
    if (copyPipeline_) { vkDestroyPipeline(ctx_.device, copyPipeline_, nullptr); copyPipeline_ = VK_NULL_HANDLE; }
    if (downsamplePipeline_) { vkDestroyPipeline(ctx_.device, downsamplePipeline_, nullptr); downsamplePipeline_ = VK_NULL_HANDLE; }
    if (metricBlocksPipeline_) { vkDestroyPipeline(ctx_.device, metricBlocksPipeline_, nullptr); metricBlocksPipeline_ = VK_NULL_HANDLE; }
    if (metricReducePipeline_) { vkDestroyPipeline(ctx_.device, metricReducePipeline_, nullptr); metricReducePipeline_ = VK_NULL_HANDLE; }
    if (composePipelineLayout_) { vkDestroyPipelineLayout(ctx_.device, composePipelineLayout_, nullptr); composePipelineLayout_ = VK_NULL_HANDLE; }
    if (copyPipelineLayout_) { vkDestroyPipelineLayout(ctx_.device, copyPipelineLayout_, nullptr); copyPipelineLayout_ = VK_NULL_HANDLE; }
    if (metricPipelineLayout_) { vkDestroyPipelineLayout(ctx_.device, metricPipelineLayout_, nullptr); metricPipelineLayout_ = VK_NULL_HANDLE; }
    if (fullscreenVert_) { vkDestroyShaderModule(ctx_.device, fullscreenVert_, nullptr); fullscreenVert_ = VK_NULL_HANDLE; }
    if (composeFrag_) { vkDestroyShaderModule(ctx_.device, composeFrag_, nullptr); composeFrag_ = VK_NULL_HANDLE; }
    if (copyFrag_) { vkDestroyShaderModule(ctx_.device, copyFrag_, nullptr); copyFrag_ = VK_NULL_HANDLE; }
    if (metricBlocksComp_) { vkDestroyShaderModule(ctx_.device, metricBlocksComp_, nullptr); metricBlocksComp_ = VK_NULL_HANDLE; }
    if (metricReduceComp_) { vkDestroyShaderModule(ctx_.device, metricReduceComp_, nullptr); metricReduceComp_ = VK_NULL_HANDLE; }

    if (descriptorPool_) { vkDestroyDescriptorPool(ctx_.device, descriptorPool_, nullptr); descriptorPool_ = VK_NULL_HANDLE; }
    if (composeSetLayout_) { vkDestroyDescriptorSetLayout(ctx_.device, composeSetLayout_, nullptr); composeSetLayout_ = VK_NULL_HANDLE; }
    if (copySetLayout_) { vkDestroyDescriptorSetLayout(ctx_.device, copySetLayout_, nullptr); copySetLayout_ = VK_NULL_HANDLE; }
    if (metricSetLayout_) { vkDestroyDescriptorSetLayout(ctx_.device, metricSetLayout_, nullptr); metricSetLayout_ = VK_NULL_HANDLE; }
    if (linearSampler_) { vkDestroySampler(ctx_.device, linearSampler_, nullptr); linearSampler_ = VK_NULL_HANDLE; }
    if (fontSampler_) { vkDestroySampler(ctx_.device, fontSampler_, nullptr); fontSampler_ = VK_NULL_HANDLE; }

    textureSet_ = VK_NULL_HANDLE;
    copySet_ = VK_NULL_HANDLE;
    gtComposeSet_ = VK_NULL_HANDLE;
    gtDownsampleSet_ = VK_NULL_HANDLE;
    ssaoSetGb_ = VK_NULL_HANDLE;
    ssaoSetGt_ = VK_NULL_HANDLE;
    ssaoSetSsaa_ = VK_NULL_HANDLE;

    if (materialUbo_) {
        vmaDestroyBuffer(ctx_.allocator, materialUbo_, materialUboMemory_);
        materialUbo_ = VK_NULL_HANDLE;
        materialUboMemory_ = VK_NULL_HANDLE;
    }

    gbColor_.destroy(ctx_);
    gbColorSpatial_.destroy(ctx_);
    gbAlbedo_.destroy(ctx_);
    gbNormal_.destroy(ctx_);
    gbMaterial_.destroy(ctx_);
    gbEmissive_.destroy(ctx_);
    gbMotion_.destroy(ctx_);
    gbReactive_.destroy(ctx_);
    gbReactiveDilated_.destroy(ctx_);
    gbDepth_.destroy(ctx_);
    gtColor_.destroy(ctx_);
    gtAlbedo_.destroy(ctx_);
    gtNormal_.destroy(ctx_);
    gtMaterial_.destroy(ctx_);
    gtEmissive_.destroy(ctx_);
    gtMotion_.destroy(ctx_);
    gtDepth_.destroy(ctx_);
    gtSsaaColor_.destroy(ctx_);
    gtSsaaAlbedo_.destroy(ctx_);
    gtSsaaNormal_.destroy(ctx_);
    gtSsaaMaterial_.destroy(ctx_);
    gtSsaaEmissive_.destroy(ctx_);
    gtSsaaMotion_.destroy(ctx_);
    gtSsaaDepth_.destroy(ctx_);
    deferred_.destroyPostFxTargets(ctx_, gbPostFx_);
    deferred_.destroyPostFxTargets(ctx_, gtPostFx_);
    deferred_.destroyPostFxTargets(ctx_, gtSsaaPostFx_);
    deferred_.destroyBloomPyramid(ctx_, gbBloom_);
    deferred_.destroyBloomPyramid(ctx_, gtBloom_);
    deferred_.destroyBloomPyramid(ctx_, gtSsaaBloom_);
    gbAoRaw_.destroy(ctx_);
    gbAo_.destroy(ctx_);
    gtAoRaw_.destroy(ctx_);
    gtAo_.destroy(ctx_);
    gtSsaaAoRaw_.destroy(ctx_);
    gtSsaaAo_.destroy(ctx_);
    deferred_.destroyColorPyramid(ctx_, gbColorPyramid_);
    deferred_.destroyColorPyramid(ctx_, gtColorPyramid_);
    deferred_.destroyColorPyramid(ctx_, gtSsaaColorPyramid_);
    deferred_.destroyClusterGrid(ctx_, gbCluster_);
    deferred_.destroyClusterGrid(ctx_, gtCluster_);
    deferred_.destroyClusterGrid(ctx_, gtSsaaCluster_);
    deferred_.destroyDepthPyramid(ctx_, gbPyramid_);
    deferred_.destroyDepthPyramid(ctx_, gtPyramid_);
    deferred_.destroyDepthPyramid(ctx_, gtSsaaPyramid_);
    // Phase 7a cull channels + shared instance SSBO (sets die with the pool).
    destroyCullResources();
    deferred_.destroyDepthPyramid(ctx_, gbPyramidAo_);
    deferred_.destroyDepthPyramid(ctx_, gtPyramidAo_);
    deferred_.destroyDepthPyramid(ctx_, gtSsaaPyramidAo_);
    deferred_.destroyAoHistory(ctx_, gbAoHist_);
    deferred_.destroyAoHistory(ctx_, gtAoHist_);
    deferred_.destroyAoHistory(ctx_, gtSsaaAoHist_);
    deferred_.destroySsrHistory(ctx_, gbSsrHist_);
    deferred_.destroySsrHistory(ctx_, gtSsrHist_);
    deferred_.destroySsrHistory(ctx_, gtSsaaSsrHist_);
    gbSsrTrace_.destroy(ctx_);
    gtSsrTrace_.destroy(ctx_);
    gtSsaaSsrTrace_.destroy(ctx_);
    deferred_.destroyVolFogVolume(ctx_, gbFog_);
    deferred_.destroyVolFogVolume(ctx_, gtFog_);
    deferred_.destroyVolFogVolume(ctx_, gtSsaaFog_);
    // AO/SSR temporal state restarts after a stack rebuild (history buffers are fresh).
    aoFramesGb_ = aoFramesGt_ = aoFramesSsaa_ = 0;
    prevAoViewProjGb_ = prevAoViewProjGt_ = prevAoViewProjSsaa_ = Mat4::identity();
    ssrFramesGb_ = ssrFramesGt_ = ssrFramesSsaa_ = 0;
    prevSsrViewProjGb_ = prevSsrViewProjGt_ = prevSsrViewProjSsaa_ = Mat4::identity();
    // Same for the froxel fog history volumes.
    fogFramesGb_ = fogFramesGt_ = fogFramesSsaa_ = 0;
    prevFogViewProjGb_ = prevFogViewProjGt_ = prevFogViewProjSsaa_ = Mat4::identity();
    fogAccumFrameGb_ = fogAccumFrameGt_ = fogAccumFrameSsaa_ = ~0u;
    composeImage_.destroy(ctx_);
    uiShotImage_.destroy(ctx_);
    uiShotLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    fontAtlas_.destroy(ctx_);

    gbColorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gbColorSpatialLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gbAlbedoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gbNormalLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gbMaterialLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gbEmissiveLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gbMotionLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gbReactiveLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gbReactiveDilatedLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gbDepthLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gtColorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gtAlbedoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gtNormalLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gtMaterialLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gtEmissiveLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gtDepthLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gtSsaaColorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gtSsaaAlbedoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gtSsaaNormalLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gtSsaaMaterialLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gtSsaaEmissiveLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gtSsaaDepthLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gbAoRawLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gbAoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gtAoRawLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gtAoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gtSsaaAoRawLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gtSsaaAoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gbSsrTraceLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gtSsrTraceLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gtSsaaSsrTraceLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    composeLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
}

void GuiApp::destroyRenderStack() {
    destroyStackResources();
    if (ctx_.device) scene_.destroy(ctx_);
}

// ---------------------------------------------------------------------------
// ImGui-only fallback frames (render stack broken after a failed rebuild).
// ---------------------------------------------------------------------------
bool GuiApp::createUiSync() {
    VkCommandBufferAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = ctx_.framePool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = kFramesInFlight;
    VkCommandBuffer cmds[kFramesInFlight] = {};
    if (vkAllocateCommandBuffers(ctx_.device, &alloc, cmds) != VK_SUCCESS) return false;

    VkSemaphoreCreateInfo semCi = {};
    semCi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fenceCi = {};
    fenceCi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        uiFrames_[i].cmd = cmds[i];
        if (vkCreateSemaphore(ctx_.device, &semCi, nullptr, &uiFrames_[i].imageAvailable) != VK_SUCCESS)
            return false;
        if (vkCreateFence(ctx_.device, &fenceCi, nullptr, &uiFrames_[i].fence) != VK_SUCCESS)
            return false;
    }
    renderFinished_.assign(swapchain_.imageCount(), VK_NULL_HANDLE);
    for (size_t i = 0; i < renderFinished_.size(); ++i) {
        if (vkCreateSemaphore(ctx_.device, &semCi, nullptr, &renderFinished_[i]) != VK_SUCCESS)
            return false;
    }
    return true;
}

void GuiApp::destroyUiSync() {
    if (!ctx_.device) return;
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        UiFrameSync& f = uiFrames_[i];
        if (f.cmd) { vkFreeCommandBuffers(ctx_.device, ctx_.framePool, 1, &f.cmd); f.cmd = VK_NULL_HANDLE; }
        if (f.imageAvailable) { vkDestroySemaphore(ctx_.device, f.imageAvailable, nullptr); f.imageAvailable = VK_NULL_HANDLE; }
        if (f.fence) { vkDestroyFence(ctx_.device, f.fence, nullptr); f.fence = VK_NULL_HANDLE; }
    }
    for (VkSemaphore sem : renderFinished_) {
        if (sem) vkDestroySemaphore(ctx_.device, sem, nullptr);
    }
    renderFinished_.clear();
}

void GuiApp::recordUiOnlyFrame(uint32_t slot, uint32_t swapchainIndex) {
    VkCommandBuffer cmd = uiFrames_[slot].cmd;
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    const VkImage swapImage = swapchain_.image(swapchainIndex);
    const VkImageView swapView = swapchain_.view(swapchainIndex);
    imageBarrier(cmd, swapImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 sync::kColorAttach, sync::kColorWrite, sync::kColorAttach, sync::kColorWrite);
    {
        VkRenderingAttachmentInfo color =
            makeColorAttachment(swapView, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_ATTACHMENT_LOAD_OP_CLEAR, 0.01f, 0.01f, 0.015f);
        beginRendering(cmd, swapchain_.extent().width, swapchain_.extent().height, 1, &color,
                       nullptr);
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
        vkCmdEndRendering(cmd);
    }
    imageBarrier(cmd, swapImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 sync::kColorAttach, sync::kColorWrite, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE);
    vkEndCommandBuffer(cmd);
}

// ---------------------------------------------------------------------------
// Per-frame rendering.
// ---------------------------------------------------------------------------
void GuiApp::updateSceneUBO(void* mapped, bool jitter, uint32_t renderW, uint32_t renderH,
                            const Mat4& view, const Mat4& proj, const Mat4& projJittered,
                            const Mat4& prevViewProj) {
    SceneUBO ubo;
    deferred_.fillSceneUBO(ubo, scene_, camera_, view, proj, projJittered, prevViewProj,
                           renderW, renderH, jitterX_, jitterY_, jitter);
    std::memcpy(mapped, &ubo, sizeof(ubo));
}

std::vector<Light> GuiApp::buildLightOverride() const {
    // The Viewer-tab lighting section drives the sun through an override
    // light list; scene_.lights is never modified (no rebuild needed).
    // Loaders always fill scene_.lights (defaultLights() when unauthored), so
    // copying it keeps every non-directional light untouched.
    std::vector<Light> lights = scene_.lights;
    if (sunEnabled_) {
        const Vec3 dir = sunDirectionFromElevAzimuth(sunElevationDeg_, sunAzimuthDeg_);
        Light* sun = nullptr;
        for (Light& l : lights) {
            if (l.type == LightType::Directional) {
                sun = &l;
                break;
            }
        }
        if (!sun) {
            // No authored sun: prepend one (kMaxLights truncation in
            // fillLightingUBO drops the tail if the scene is full).
            lights.insert(lights.begin(), Light{});
            sun = &lights.front();
            sun->type = LightType::Directional;
        }
        sun->positionOrDirection = dir;
        sun->intensity = sunIntensity_;
        sun->color = sunColor_;
        // The UI-driven sun is the CSM caster: Light{} defaults castShadow to
        // false and glTF suns may leave it unauthored.  (LightGPU.params.y is
        // reserved in the shaders; only the ShadowFrame light index matters.)
        sun->castShadow = true;
    } else {
        lights.erase(std::remove_if(lights.begin(), lights.end(),
                                    [](const Light& l) {
                                        return l.type == LightType::Directional;
                                    }),
                     lights.end());
    }
    if (!fillEnabled_) {
        // Drop the canned default fill only; authored point lights (PR3) stay.
        const Light canned = defaultFillLight();
        lights.erase(std::remove_if(lights.begin(), lights.end(),
                                    [&](const Light& l) {
                                        return l.type == LightType::Point &&
                                               l.positionOrDirection.x == canned.positionOrDirection.x &&
                                               l.positionOrDirection.y == canned.positionOrDirection.y &&
                                               l.positionOrDirection.z == canned.positionOrDirection.z;
                                    }),
                     lights.end());
    }
    return lights;
}

void GuiApp::applyLightingPreset(const LightingPreset& p) {
    sunEnabled_ = p.sunEnabled;
    sunElevationDeg_ = p.sunElevationDeg;
    sunAzimuthDeg_ = p.sunAzimuthDeg;
    sunIntensity_ = p.sunIntensity;
    sunColor_ = p.sunColor;
    fillEnabled_ = p.fillEnabled;
    iblIntensity_ = p.iblIntensity;
    exposure_ = p.exposure;
    fogParams_ = p.fog;
    volFogEnabled_ = p.fog.enabled;
    // Atmosphere mode: the sky + IBL follow the preset sun.
    updateSkyFromUiSun();
}

void GuiApp::updateSkyFromUiSun() {
    if (!stackOk_ || !deferred_.atmosphereSky()) return;
    // The one-shot command pool is shared with the async loader; skip while a
    // load is in flight (the post-load applyLightingPreset re-runs this).
    if (loadPhase_.load(std::memory_order_acquire) == LoadPhase::Loading) return;
    deferred_.updateAtmosphereSky(
        ctx_, sunDirectionFromElevAzimuth(sunElevationDeg_, sunAzimuthDeg_));
}

void GuiApp::updateLightingUBO(void* mapped, const Mat4& viewProj,
                               const std::vector<Light>& lights, const ShadowFrame* shadow) {
    LightingUBO ubo;
    deferred_.fillLightingUBO(ubo, scene_, camera_, viewProj, Mat4::inverse(viewProj), &lights,
                              shadow, iblIntensity_);
    ubo.shadowAtlasParams[3] = contactShadowsEnabled_ ? 1.f : 0.f;
    std::memcpy(mapped, &ubo, sizeof(ubo));
}

void GuiApp::updateClusterLights(uint32_t frameIndex, const std::vector<Light>& lights) {
    // Full point/spot set for the clustered pass (the GUI override list, so
    // sun/fill toggles apply); same per-slot rule as the UBOs.
    const uint32_t slot = frameIndex % kFramesInFlight;
    deferred_.fillClusterLights(gbCluster_.lightsMapped[slot], lights);
    deferred_.fillClusterLights(gtCluster_.lightsMapped[slot], lights);
    if (gtSsaaCluster_.lightsMapped[slot])
        deferred_.fillClusterLights(gtSsaaCluster_.lightsMapped[slot], lights);
}

// ---------------------------------------------------------------------------
// Compare-tab zoom: mouse wheel zooms around the cursor (1x..16x), middle
// drag pans.  Ignored while the pointer is over any UI element.
// ---------------------------------------------------------------------------
void GuiApp::handleCompareZoomInput() {
    if (active_.mode != Mode::Compare || !stackOk_) return;
    const ImGuiIO& io = ImGui::GetIO();
    // Debug hook (SR_GUI_DEBUG_INPUT=1): trace the input path for the
    // interactive zoom/pan verification.
    static const bool dbgInput = envFlag("SR_GUI_DEBUG_INPUT");
    if (dbgInput && (io.MouseWheel != 0.f || ImGui::IsMouseDown(ImGuiMouseButton_Middle))) {
        std::fprintf(stderr,
                     "[zoomdbg] wheel=%.2f mbtn=%d pos=%.0f,%.0f wantCapture=%d drag=%d "
                     "zoom=%.2f pan=%.3f,%.3f\n",
                     static_cast<double>(io.MouseWheel),
                     ImGui::IsMouseDown(ImGuiMouseButton_Middle) ? 1 : 0,
                     static_cast<double>(io.MousePos.x), static_cast<double>(io.MousePos.y),
                     io.WantCaptureMouse ? 1 : 0, comparePanDrag_ ? 1 : 0,
                     static_cast<double>(compareZoom_), static_cast<double>(comparePanU_),
                     static_cast<double>(comparePanV_));
    }

    const uint32_t dw = active_.displayW;
    const uint32_t dh = active_.displayH;
    const uint32_t numColumns = 1 + static_cast<uint32_t>(algos_.size());
    const float x0 = static_cast<float>(layoutOriginX());
    const float colW = (static_cast<float>(dw) - x0) / static_cast<float>(numColumns);

    bool changed = false;

    // Hover gate.  The side panel is pinned to the left edge, so a hard
    // region test is the deterministic part; io.WantCaptureMouse additionally
    // covers popups/menus that extend past the panel edge.  This is reliable
    // for the wheel because no button is held while wheeling (once a button
    // is down, ImGui may keep capture set for the whole drag — hence the
    // separate pan latch below).
    const bool overPanel =
        panelCollapsed_
            ? (io.MousePos.x >= 0.f && io.MousePos.x < 56.f * uiScale_ && io.MousePos.y >= 0.f &&
               io.MousePos.y < 34.f * uiScale_)
            : (io.MousePos.x >= 0.f && io.MousePos.x < panelWidth() && io.MousePos.y >= 0.f);
    const bool uiHover = overPanel || io.WantCaptureMouse;

    // Wheel zoom at the cursor.
    if (io.MouseWheel != 0.f && !uiHover && io.MousePos.y >= 0.f) {
        // Cursor position in column-local UV (every column shows the same
        // region, so the column index only matters for the local X).
        const float relX = io.MousePos.x - x0;
        const float colIdx = std::floor(relX / colW);
        const float localU = std::clamp((relX - colIdx * colW) / colW, 0.f, 1.f);
        const float localV = std::clamp(io.MousePos.y / static_cast<float>(dh), 0.f, 1.f);

        float rect[4];
        computeViewRegion(dw, dh, static_cast<uint32_t>(colW), dh, compareZoom_, comparePanU_,
                          comparePanV_, rect);
        // Source point currently under the cursor (normalized).
        const float srcU = (rect[0] + localU * rect[2]) / static_cast<float>(dw);
        const float srcV = (rect[1] + localV * rect[3]) / static_cast<float>(dh);

        const float newZoom =
            std::clamp(compareZoom_ * std::pow(1.25f, io.MouseWheel), 1.f, 16.f);
        if (newZoom != compareZoom_) {
            compareZoom_ = newZoom;
            // Keep the cursor's source point fixed: solve the new window
            // offset, then store it back as the window center.
            float rect2[4];
            computeViewRegion(dw, dh, static_cast<uint32_t>(colW), dh, compareZoom_, srcU, srcV,
                              rect2);
            // computeViewRegion centers on (srcU, srcV); shift so the point
            // lands under the cursor's local UV again.
            float offX = srcU * static_cast<float>(dw) - localU * rect2[2];
            float offY = srcV * static_cast<float>(dh) - localV * rect2[3];
            offX = std::clamp(offX, 0.f, std::max(static_cast<float>(dw) - rect2[2], 0.f));
            offY = std::clamp(offY, 0.f, std::max(static_cast<float>(dh) - rect2[3], 0.f));
            comparePanU_ = (offX + rect2[2] * 0.5f) / static_cast<float>(dw);
            comparePanV_ = (offY + rect2[3] * 0.5f) / static_cast<float>(dh);
            changed = true;
        }
    }

    // Middle-drag pan.  Own latch: the drag must START over the render area
    // (on the click frame no button was held yet, so the hover gate is
    // reliable there; ImGui latches capture for the held button afterwards).
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle))
        comparePanDrag_ = !uiHover;
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Middle))
        comparePanDrag_ = false;

    if (comparePanDrag_ && compareZoom_ > 1.f) {
        float rect[4];
        computeViewRegion(dw, dh, static_cast<uint32_t>(colW), dh, compareZoom_, comparePanU_,
                          comparePanV_, rect);
        // Dragging the image moves the window the other way.
        const float du = -io.MouseDelta.x / colW * (rect[2] / static_cast<float>(dw));
        const float dv = -io.MouseDelta.y / static_cast<float>(dh) *
                         (rect[3] / static_cast<float>(dh));
        if (du != 0.f || dv != 0.f) {
            const float halfW = rect[2] / static_cast<float>(dw) * 0.5f;
            const float halfH = rect[3] / static_cast<float>(dh) * 0.5f;
            comparePanU_ = std::clamp(comparePanU_ + du, halfW, 1.f - halfW);
            comparePanV_ = std::clamp(comparePanV_ + dv, halfH, 1.f - halfH);
            changed = true;
        }
    }

    if (changed) refreshOverlayText();
}

void GuiApp::updateCamera(float dt) {
    if (pathPlaying_ && !path_.empty()) {
        const CameraKeyframe& kf = path_[pathFrame_ % path_.size()];
        camera_.setPose(kf.position, kf.forward, kf.up);
        ++pathFrame_;
        return;
    }

    // Free-fly driven by the ImGui IO so the UI gets first claim on input.
    Window::Input in = {};
    const ImGuiIO& io = ImGui::GetIO();
    if (!io.WantTextInput) {
        in.keys['W'] = ImGui::IsKeyDown(ImGuiKey_W);
        in.keys['A'] = ImGui::IsKeyDown(ImGuiKey_A);
        in.keys['S'] = ImGui::IsKeyDown(ImGuiKey_S);
        in.keys['D'] = ImGui::IsKeyDown(ImGuiKey_D);
        in.keys['Q'] = ImGui::IsKeyDown(ImGuiKey_Q);
        in.keys['E'] = ImGui::IsKeyDown(ImGuiKey_E);
        in.keys[kKeyShift] = ImGui::IsKeyDown(ImGuiKey_LeftShift) || ImGui::IsKeyDown(ImGuiKey_RightShift);
        in.keys[kKeySpace] = ImGui::IsKeyDown(ImGuiKey_Space);
        in.keys[kKeyControl] = ImGui::IsKeyDown(ImGuiKey_LeftCtrl) || ImGui::IsKeyDown(ImGuiKey_RightCtrl);
    }
    // Rotate only when dragging on the render area (not over any UI element).
    const bool rotate = !io.WantCaptureMouse &&
                        (ImGui::IsMouseDown(ImGuiMouseButton_Left) ||
                         ImGui::IsMouseDown(ImGuiMouseButton_Right));
    if (rotate) {
        in.mouseDX = io.MouseDelta.x;
        in.mouseDY = io.MouseDelta.y;
    }
    camera_.updateFreeFly(in, dt);
}

void GuiApp::captureScreenshotIntoStaging(VkCommandBuffer cmd) {
    imageBarrier(cmd, composeImage_.image, composeLayout_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 sync::kFragment, sync::kSampled, sync::kCopy, sync::kTransferRead);
    VkBufferImageCopy region = {};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {active_.displayW, active_.displayH, 1};
    vkCmdCopyImageToBuffer(cmd, composeImage_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           screenshotStaging_, 1, &region);
    imageBarrier(cmd, composeImage_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 sync::kCopy, sync::kTransferRead, sync::kFragment, sync::kSampled);
    composeLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void GuiApp::harvestMetrics(uint32_t slot) {
    const auto* f = static_cast<const float*>(metricStagingMapped_[slot]);
    const uint32_t numAlgos = static_cast<uint32_t>(algos_.size());
    const uint32_t mask = metricMask_[slot];
    for (uint32_t i = 0; i < numAlgos; ++i) {
        if ((mask & (1u << i)) == 0) {
            algos_[i].hasMetric = false;
            continue;
        }
        const float* r = f + i * kMetricFloats;
        // PSNR = 10*log10(3*N / sumSqDiff), data range 1 (tonemapped domain).
        const double d2 = static_cast<double>(r[0]) + r[1] + r[2];
        const double n = static_cast<double>(r[3]);
        float psnr = 99.f;
        if (n > 0.0 && d2 > 0.0) {
            psnr = static_cast<float>(10.0 * std::log10(3.0 * n / d2));
        }
        const double blocks = std::max(static_cast<double>(r[7]), 1.0);
        const float ssim =
            static_cast<float>((static_cast<double>(r[4]) + r[5] + r[6]) / (3.0 * blocks));
        algos_[i].psnr = psnr;
        algos_[i].ssim = ssim;
        algos_[i].hasMetric = true;
    }
    refreshOverlayText();
}

void GuiApp::refreshOverlayText() {
    if (!textUboMapped_) return;
    auto* dst = static_cast<uint32_t*>(textUboMapped_);
    std::memset(dst, 0, kMaxColumns * kTextCharsPerColumn * 4);

    auto writeLine = [&](uint32_t slot, uint32_t line, const char* text) {
        uint32_t* row = dst + slot * kTextCharsPerColumn + line * 24;
        for (uint32_t c = 0; c < 24 && text[c] != '\0'; ++c) row[c] = asciiUpper(text[c]);
    };

    auto displayFps = [&](bool fg) {
        // Viewer: measured present rate (true+interpolated).  Compare still
        // presents once per true frame; FG columns show the ×2 equivalent.
        if (active_.mode == Mode::Viewer) return fps_;
        float n = ui_.lockFps ? static_cast<float>(std::max(15, std::min(120, ui_.lockFpsTarget)))
                              : fps_;
        if (fg) n *= 2.f;
        return n;
    };
    auto columnTitle = [](const AlgoColumn& algo, char* out, size_t outSize) {
        const char* sr = algo.upscaler ? algo.upscaler->name() : algo.id.c_str();
        if (algo.fg.empty())
            std::snprintf(out, outSize, "%s", sr);
        else if (algo.fg == "nfru")
            std::snprintf(out, outSize, "%s + NFRU", sr);
        else if (algo.fg == "fsr3")
            std::snprintf(out, outSize, "%s + FSR3", sr);
        else
            std::snprintf(out, outSize, "%s + %s", sr, algo.fg.c_str());
    };

    char line[25];
    if (active_.mode == Mode::Viewer) {
        if (algos_.empty()) {
            writeLine(0, 0, "Native (GT)");
        } else {
            columnTitle(algos_[0], line, sizeof(line));
            writeLine(0, 0, line);
            std::snprintf(line, sizeof(line), "FPS %.1f",
                          static_cast<double>(displayFps(!algos_[0].fg.empty())));
            writeLine(0, 1, line);
            std::snprintf(line, sizeof(line), "SCENE %.2f MS", lastTimings_.sceneMs);
            writeLine(0, 2, line);
            std::snprintf(line, sizeof(line), "UPSCALE %.2f MS", lastTimings_.upscaleMs);
            writeLine(0, 3, line);
        }
        return;
    }

    writeLine(0, 0, "Native (GT)");
    {
        std::snprintf(line, sizeof(line), "ZOOM %.2fX", static_cast<double>(compareZoom_));
        writeLine(0, 1, line);
        if (active_.gtApplyScale)
            writeLine(0, 2, "GT APPLY SCALE");
        else if (active_.gtSsaa)
            writeLine(0, 2, "GT SSAA 2X");
        bool anyFg = false;
        for (const AlgoColumn& a : algos_) {
            if (!a.fg.empty()) anyFg = true;
        }
        if (anyFg) writeLine(0, 3, "GT MIDPOINT");
    }
    for (uint32_t i = 0; i < algos_.size(); ++i) {
        const AlgoColumn& algo = algos_[i];
        const uint32_t slot = i + 1;
        columnTitle(algo, line, sizeof(line));
        writeLine(slot, 0, line);
        std::snprintf(line, sizeof(line), "FPS %.1f",
                      static_cast<double>(displayFps(!algo.fg.empty())));
        writeLine(slot, 1, line);
        if (algo.hasMetric) {
            std::snprintf(line, sizeof(line), "PSNR %.2f dB", static_cast<double>(algo.psnr));
            writeLine(slot, 2, line);
            std::snprintf(line, sizeof(line), "SSIM %.4f", static_cast<double>(algo.ssim));
            writeLine(slot, 3, line);
        } else {
            writeLine(slot, 2, "PSNR --");
            writeLine(slot, 3, "SSIM --");
        }
    }
}

void GuiApp::recordFrame(uint32_t frameIndex, uint32_t swapchainIndex) {
    const uint32_t slot = frameIndex % kFramesInFlight;
    FrameResources& fr = frames_[slot];
    VkCommandBuffer cmd = fr.cmd;

    // Frame-index driven scene animation (dynamic boxes / glTF clips), before
    // any recording; no-op for static scenes.
    scene_.advanceToFrame(frameIndex);
    // CPU LOD selection for this frame's camera (GUI checkbox toggles it).
    scene_.updateLodSelection(camera_.position, camera_.fovY, lodEnabled_);
    // Background fine-mip streaming tick (no-op until a streamed scene loads).
    scene_.updateTextureStreaming(ctx_, camera_.position);

    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    timestamps_.resetForFrame(cmd, slot);
    timestamps_.frameBegin(cmd, slot);
    profiler_.beginFrame(cmd, slot, frameIndex);

    const uint32_t dw = active_.displayW;
    const uint32_t dh = active_.displayH;
    const bool compareMode = active_.mode == Mode::Compare;
    const bool gbuffer = !algos_.empty();
    bool hasTemporal = false;
    bool hasSpatial = false;
    bool anyFg = false;
    for (const AlgoColumn& a : algos_) {
        if (a.frameGen) anyFg = true;
        if (!a.upscaler) continue;
        if (upscalerNeedsJitter(a.upscaler.get())) hasTemporal = true;
        else hasSpatial = true;
    }
    // Raster jitter cannot be undone by resampling.  Temporal plugins still
    // get a jittered GBuffer; mixed sets render a second unjittered LR pass
    // for spatial plugins (FSR1 / SGSR1).
    const bool temporal = hasTemporal;
    const bool mixedSpatial = hasTemporal && hasSpatial;
    const float aspect = static_cast<float>(dw) / static_cast<float>(dh);
    const Mat4 view = camera_.view();
    const Mat4 proj = camera_.proj(aspect);
    // Any FG column: true frames stay at the current pose; GT is the midpoint
    // between this pose and the previous true frame so PSNR matches the
    // interpolated output.  No previous pose yet → same-time GT (no metric).
    const Camera gtCam =
        (anyFg && havePrevCamera_) ? lerpCamera(prevCamera_, camera_, 0.5f) : camera_;
    const Mat4 viewGt = gtCam.view();
    const Mat4 projGt = gtCam.proj(aspect);
    Mat4 projJittered = proj;
    prevJitterX_ = jitterX_;
    prevJitterY_ = jitterY_;
    const Vec2 h = halton23(frameIndex + 1);
    jitterX_ = (gbuffer && temporal) ? (h.x - 0.5f) : 0.f;
    jitterY_ = (gbuffer && temporal) ? (h.y - 0.5f) : 0.f;
    // Uniform NDC shift: clip.xy += offset * clip.w (clip.w = -z_view via
    // m[11] = -1), so the offset belongs in column 2 with a negative sign —
    // see Renderer.cpp for the full rationale.
    projJittered.m[8] -= jitterX_ * 2.f / static_cast<float>(renderWidth_);
    projJittered.m[9] -= jitterY_ * 2.f / static_cast<float>(renderHeight_);

    updateSceneUBO(fr.uboGbMapped, temporal, renderWidth_, renderHeight_, view, proj, projJittered,
                   prevViewProj_);
    if (mixedSpatial) {
        updateSceneUBO(fr.uboGbSpatialMapped, false, renderWidth_, renderHeight_, view, proj, proj,
                       prevViewProj_);
    }
    const uint32_t gtW = active_.gtSsaa ? dw * 2 : (active_.gtApplyScale ? renderWidth_ : dw);
    const uint32_t gtH = active_.gtSsaa ? dh * 2 : (active_.gtApplyScale ? renderHeight_ : dh);
    updateSceneUBO(fr.uboGtMapped, false, gtW, gtH, viewGt, projGt, projGt, prevViewProj_);

    // CSM sun shadows: the shadowed sun is picked from the same override
    // light list that fills the LightingUBO, so lightIndex refers to the
    // packed override order (the GUI sun may be prepended at index 0).  The
    // GT and SSAA paths sample the same map.  The "shadows" checkbox off
    // leaves shadow null (shadowParams.z = 0).
    // Phase 4b: shadow-casting spots are additionally scored by
    // intensity/distance^2 and the top kShadowAtlasTiles get an atlas tile
    // (selectSpotShadowLights writes shadowIndex into the override list).
    std::vector<Light> lights = buildLightOverride();
    ShadowFrame shadowFrame;
    const ShadowFrame* shadow = nullptr;
    if (shadowsActive_ && shadowsEnabled_) {
        for (uint32_t i = 0; i < lights.size(); ++i) {
            if (lights[i].type == LightType::Directional && lights[i].castShadow) {
                DeferredCore::computeCascadeVPs(camera_, aspect, lights[i].positionOrDirection,
                                                shadowFrame.cascadeVp, shadowFrame.splitDepth);
                shadowFrame.lightIndex = static_cast<int32_t>(i);
                break;
            }
        }
        if (spotAtlasActive_) {
            shadowFrame.atlasTileCount =
                DeferredCore::selectSpotShadowLights(lights, camera_.position);
            for (const Light& l : lights) {
                if (l.shadowIndex >= 0)
                    shadowFrame.atlasVp[l.shadowIndex] = DeferredCore::computeSpotShadowVp(l);
            }
        }
        shadowFrame.debugCascades = shadowDebugCascades_;
        shadowFrame.frameIndex = frameIndex;
        if (shadowFrame.lightIndex >= 0 || shadowFrame.atlasTileCount > 0)
            shadow = &shadowFrame;
    }
    updateLightingUBO(fr.lightingUboGbMapped, Mat4::multiply(projJittered, view), lights, shadow);
    if (mixedSpatial) {
        updateLightingUBO(fr.lightingUboGbSpatialMapped, Mat4::multiply(proj, view),
                          lights, shadow);
    }
    updateLightingUBO(fr.lightingUboGtMapped, Mat4::multiply(projGt, viewGt),
                      lights, shadow);
    updateClusterLights(frameIndex, lights);

    auto transition = [&](VkImage image, VkImageLayout& current, VkImageLayout target,
                          VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                          VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
                          VkImageAspectFlags aspect_) {
        imageBarrier(cmd, image, current, target, srcStage, srcAccess, dstStage, dstAccess,
                     aspect_);
        current = target;
    };
    const Mat4 cullViewProj = Mat4::multiply(proj, view); // un-jittered (sub-pixel)
    const Mat4 cullViewProjGt = Mat4::multiply(projGt, viewGt);

    // --- Phase 7a: candidate build + staged instance/command data -------------
    // One CPU build (frustum + LOD, path-independent); each path's channel
    // gets a copy of the same command list and zeroes its own occluded
    // instanceCounts against its own Hi-Z chain.
    cullCandidates_ =
        deferred_.buildInstanceList(scene_, cullViewProj, instances_.capacity,
                                    cullInstCpu_.data(), cullCmdCpu_.data(), cullRuns_);
    if (cullCandidates_ > 0) {
        std::memcpy(instances_.stagingMapped[slot], cullInstCpu_.data(),
                    static_cast<size_t>(cullCandidates_) * sizeof(GpuInstance));
        const size_t cmdBytes =
            static_cast<size_t>(cullCandidates_) * sizeof(VkDrawIndexedIndirectCommand);
        std::memcpy(gbCull_.cmdStagingMapped[slot], cullCmdCpu_.data(), cmdBytes);
        std::memcpy(gtCull_.cmdStagingMapped[slot], cullCmdCpu_.data(), cmdBytes);
        if (active_.gtSsaa)
            std::memcpy(gtSsaaCull_.cmdStagingMapped[slot], cullCmdCpu_.data(), cmdBytes);
    }
    deferred_.recordInstanceUpload(cmd, slot, instances_, cullCandidates_);

    // Phase 6b: same MB/DOF algorithm + parameters on every path; the blur
    // clamps scale with path height so the display-space blur matches the GT.
    auto recordPostFx = [&](PostFxTargets& fxTargets, VkImage color, VkImageLayout& colorLayout,
                            const Mat4& pathProj, uint32_t pathH) {
        SR_GPU_ZONE(profiler_, cmd, "postfx");
        PostFxParams fx;
        fx.depthM10 = pathProj.m[10];
        fx.depthM14 = pathProj.m[14];
        fx.farPlane = camera_.farPlane;
        const float resScale = static_cast<float>(pathH) / 1080.f;
        fx.maxBlurPx = std::max(8.f, kMotionBlurMaxPixels * resScale);
        fx.maxCocPx = std::max(2.f, dofMaxBlur_ * resScale);
        fx.aperture = kDofAperture * (kDofDefaultFstop / dofFstop_);
        fx.focusDistance = dofFocus_;
        fx.motionBlur = motionBlurEnabled_;
        fx.dof = dofEnabled_;
        deferred_.recordPostFxPass(cmd, fxTargets, color, colorLayout, fx, frameIndex);
    };

    // Phase 6a bloom on the lit HDR target, before the Phase 6b post-fx chain
    // (same order as the viewer).  Per-frame pass skip; the color image is
    // SHADER_READ_ONLY on entry and the helper hands it back the same way.
    auto recordBloom = [&](BloomPyramid& pyramid, VkImage color, VkImageLayout& colorLayout,
                           uint32_t pathW, uint32_t pathH) {
        if (!bloomEnabled_) return;
        SR_GPU_ZONE(profiler_, cmd, "bloom");
        deferred_.recordBloomPyramidPass(cmd, pyramid, color, colorLayout, pathW, pathH);
    };

    // --- Shadow pass (sun CSM, one 2048^2 layer per cascade) -------------------
    // Must run BEFORE any lighting pass: the LR lighting below samples the map
    // while the LightingUBO already carries this frame's cascade VPs, so a map
    // rendered later in the frame would be sampled with a one-frame-stale
    // content/VP pairing (false occlusion on lit surfaces whenever the camera
    // moves).  CompareApp/Renderer place this pass ahead of lighting for the
    // same reason.  It only needs scene geometry, so it runs before the GBuffer
    // pass and also covers the gbuffer-less (GT-only) configuration.
    {
        SR_GPU_ZONE(profiler_, cmd, "shadows");
        if (shadow && shadow->lightIndex >= 0) {
            SR_GPU_ZONE(profiler_, cmd, "shadow_csm");
            imageBarrier(cmd, shadow_.image, shadow_.layout,
                     VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, sync::kFragment,
                     sync::kSampled, sync::kDepthTests, sync::kDepthReadWrite,
                     VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, kShadowCascadeCount);
        shadow_.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        deferred_.recordShadowPass(cmd, shadow_, scene_, shadow->cascadeVp, fr.sceneSetGb,
                                   textureSet_, materialStride_);
        imageBarrier(cmd, shadow_.image, shadow_.layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     sync::kDepthTests, sync::kDepthWrite, sync::kFragment, sync::kSampled,
                     VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, kShadowCascadeCount);
        shadow_.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    // Spot shadow atlas (Phase 4b): one 1024^2 tile per selected spot, shared
    // by all lighting paths like the CSM map.
        if (shadow && shadow->atlasTileCount > 0) {
            SR_GPU_ZONE(profiler_, cmd, "shadow_spot");
            imageBarrier(cmd, spotAtlas_.image, spotAtlas_.layout,
                     VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, sync::kFragment,
                     sync::kSampled, sync::kDepthTests, sync::kDepthReadWrite,
                     VK_IMAGE_ASPECT_DEPTH_BIT);
        spotAtlas_.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        deferred_.recordSpotShadowPass(cmd, spotAtlas_, scene_, shadow->atlasVp,
                                       shadow->atlasTileCount, fr.sceneSetGb, textureSet_,
                                       materialStride_);
        imageBarrier(cmd, spotAtlas_.image, spotAtlas_.layout,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kDepthTests,
                     sync::kDepthWrite, sync::kFragment, sync::kSampled,
                     VK_IMAGE_ASPECT_DEPTH_BIT);
        spotAtlas_.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
    }

    // --- 1) low-resolution GBuffer (jittered for temporal; extra unjittered
    //     pass when mixed with spatial plugins) --------------------------------
    timestamps_.sceneBegin(cmd, slot);
    auto recordLrDeferred = [&](VkDescriptorSet sceneSet, VkDescriptorSet lightingSet,
                                VkDescriptorSet transparentSet, VkDescriptorSet ssrSet,
                                const Mat4& ssaoViewProj, const Mat4& projUsed,
                                const char* tag) {
        SR_GPU_ZONE(profiler_, cmd, tag);
        // Phase 7a: upload this frame's commands and run the occlusion cull
        // against the LR Hi-Z chain as it currently stands (previous frame's
        // pyramid — or, in mixed mode's second record, this frame's spatial
        // rebuild; gbCull_.prevViewProj always tracks the producing VP).
        const bool cullActive = occlusionEnabled_ && gbCull_.prevValid && cullCandidates_ > 0;
        {
            SR_GPU_ZONE(profiler_, cmd, "occlusion_cull");
            deferred_.recordCommandUpload(cmd, slot, gbCull_, cullCandidates_, cullActive);
            if (cullActive)
                deferred_.recordOcclusionCull(cmd, gbCull_, cullCandidates_, gbCull_.prevViewProj,
                                              gbPyramid_.mipCount, renderWidth_, renderHeight_);
        }
        { SR_GPU_ZONE(profiler_, cmd, "gbuffer");
        transition(gbAlbedo_.image, gbAlbedoLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbNormal_.image, gbNormalLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbMaterial_.image, gbMaterialLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbEmissive_.image, gbEmissiveLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbMotion_.image, gbMotionLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbDepth_.image, gbDepthLayout_, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kDepthTests, sync::kDepthReadWrite,
                   VK_IMAGE_ASPECT_DEPTH_BIT);
        {
            VkRenderingAttachmentInfo colors[5] = {
                makeColorAttachment(gbAlbedo_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gbNormal_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR, 0.f, 0.f, 1.f),
                makeColorAttachment(gbMaterial_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gbEmissive_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gbMotion_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR)};
            VkRenderingAttachmentInfo depth =
                makeDepthAttachment(gbDepth_.view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR);
            beginRendering(cmd, renderWidth_, renderHeight_, 5, colors, &depth);
            deferred_.recordGBufferDraws(cmd, scene_, false, sceneSet, textureSet_,
                                         materialStride_, renderWidth_, renderHeight_,
                                         cullViewProj, gbCull_, cullRuns_.data(),
                                         static_cast<uint32_t>(cullRuns_.size()));
            vkCmdEndRendering(cmd);
        }
        } // zone "gbuffer"
        // dst scope includes compute: the opaque-SSR pass (ssr_opaque.comp)
        // samples the GBuffer after the lighting fragment shader.
        {
            SR_GPU_ZONE(profiler_, cmd, "hiz");
        transition(gbAlbedo_.image, gbAlbedoLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbNormal_.image, gbNormalLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbMaterial_.image, gbMaterialLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbEmissive_.image, gbEmissiveLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbMotion_.image, gbMotionLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbDepth_.image, gbDepthLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kDepthTests, sync::kDepthWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_DEPTH_BIT);
        // Hi-Z pyramid for the SSR marchers (glass transparency + opaque-SSR
        // compute; both LR lighting variants share the same GBuffer depth) and
        // the occlusion cull (Phase 7a).  ssaoViewProj is the exact
        // view-projection of the GBuffer record that fed this depth — the cull
        // channel reprojects with it from the next record on.
        if (hasTransparency_ || ssrEnabled_ || occlusionEnabled_) {
            deferred_.recordDepthPyramidPass(cmd, gbPyramid_);
            gbCull_.prevViewProj = ssaoViewProj;
            gbCull_.prevValid = true;
        }
        } // zone "hiz"

        // GTAO: view-Z depth chain (sampled at per-step LODs) -> main pass ->
        // temporal EMA -> denoise.  The chain is rebuilt every record.
        const Mat4 invAoVp = Mat4::inverse(ssaoViewProj);
        { SR_GPU_ZONE(profiler_, cmd, "gtao");
        deferred_.recordDepthPyramidPass(cmd, gbPyramidAo_);
        const float aoMaxLodGb = static_cast<float>(std::min(gbPyramidAo_.mipCount - 1, 4u));
        { SR_GPU_ZONE(profiler_, cmd, "ao_main");
        transition(gbAoRaw_.image, gbAoRawLayout_, VK_IMAGE_LAYOUT_GENERAL, sync::kCompute,
                   sync::kSampled, sync::kCompute, sync::kStorageWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordSsaoPass(cmd, ssaoSetGb_, invAoVp, frameIndex, camera_.nearPlane,
                                 camera_.farPlane, aoMaxLodGb, renderWidth_, renderHeight_);
        transition(gbAoRaw_.image, gbAoRawLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kCompute, sync::kStorageWrite, sync::kCompute, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        }
        const uint32_t aoWriteGb = aoFramesGb_ & 1u;
        { SR_GPU_ZONE(profiler_, cmd, "ao_temporal");
        deferred_.recordSsaoTemporalPass(cmd, gbAoHist_, aoWriteGb, invAoVp, prevAoViewProjGb_,
                                         renderWidth_, renderHeight_, /*reset=*/aoFramesGb_ == 0);
        }
        prevAoViewProjGb_ = ssaoViewProj;
        ++aoFramesGb_;
        { SR_GPU_ZONE(profiler_, cmd, "ao_blur");
        transition(gbAo_.image, gbAoLayout_, VK_IMAGE_LAYOUT_GENERAL, sync::kSampleStages,
                   sync::kSampled, sync::kCompute, sync::kStorageWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordSsaoBlurPass(cmd, gbAoHist_.blurSet[aoWriteGb], renderWidth_,
                                     renderHeight_);
        transition(gbAo_.image, gbAoLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kCompute, sync::kStorageWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        }
        } // zone "gtao"

        { SR_GPU_ZONE(profiler_, cmd, "lighting");
        transition(gbColor_.image, gbColorLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordLightingPass(cmd, lightingSet, gbCluster_, slot, view, projUsed,
                                     gbColor_.view, renderWidth_, renderHeight_);
        }

        // Froxel volumetric fog (Phase 5a): accumulate once per frame (mixed
        // mode records LR lighting twice; the first record wins the guard),
        // composite into the lit HDR target on every record.  Un-jittered
        // matrices so the volume does not swim under TAA jitter.
        if (volFogEnabled_ && fogParams_.enabled && gbFog_.injectImage != VK_NULL_HANDLE) {
            SR_GPU_ZONE(profiler_, cmd, "volfog");
            if (fogAccumFrameGb_ != frameIndex) {
                SR_GPU_ZONE(profiler_, cmd, "fog_accumulate");
                deferred_.recordVolFogAccumulate(cmd, gbFog_, gbCluster_, slot, view, proj,
                                                 prevFogViewProjGb_, fogParams_, frameIndex,
                                                 fogFramesGb_ & 1u, /*reset=*/fogFramesGb_ == 0);
                prevFogViewProjGb_ = cullViewProj;
                ++fogFramesGb_;
                fogAccumFrameGb_ = frameIndex;
            }
            { SR_GPU_ZONE(profiler_, cmd, "fog_composite");
            transition(gbColor_.image, gbColorLayout_, VK_IMAGE_LAYOUT_GENERAL,
                       sync::kColorAttach, sync::kColorWrite, sync::kCompute,
                       sync::kStorageReadWrite, VK_IMAGE_ASPECT_COLOR_BIT);
            deferred_.recordVolFogComposite(cmd, gbFog_, proj, fogParams_.maxDistance,
                                            renderWidth_, renderHeight_);
            transition(gbColor_.image, gbColorLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                       sync::kCompute, sync::kStorageWrite, sync::kColorAttach,
                       sync::kColorReadWrite, VK_IMAGE_ASPECT_COLOR_BIT);
            }
        }

        if (hasTransparency_ || ssrEnabled_) {
            SR_GPU_ZONE(profiler_, cmd, "reflections");
            // Color mip chain: mip 0 is the opaque HDR copy glass SSR used to
            // make with a transfer; the chain stays GENERAL for life.  The
            // opaque-SSR pass consumes the same chain, so it is built whenever
            // either consumer runs.
            { SR_GPU_ZONE(profiler_, cmd, "color_pyramid");
            transition(gbColor_.image, gbColorLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       sync::kColorAttach, sync::kColorWrite, sync::kCompute, sync::kSampled,
                       VK_IMAGE_ASPECT_COLOR_BIT);
            deferred_.recordColorPyramidPass(cmd, gbColorPyramid_);
            }
            if (ssrEnabled_) {
                // Opaque SSR, Phase 2d: trace into the LR RT, then temporal
                // EMA + fused composite (in-place RMW on gbColor_).
                { SR_GPU_ZONE(profiler_, cmd, "ssr_trace");
                transition(gbSsrTrace_.image, gbSsrTraceLayout_, VK_IMAGE_LAYOUT_GENERAL,
                           sync::kCompute, sync::kSampled, sync::kCompute,
                           sync::kStorageWrite, VK_IMAGE_ASPECT_COLOR_BIT);
                deferred_.recordSsrPass(cmd, ssrSet, ssaoViewProj, renderWidth_, renderHeight_,
                                      ssrStrength_);
                transition(gbSsrTrace_.image, gbSsrTraceLayout_,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kCompute,
                           sync::kStorageWrite, sync::kCompute, sync::kSampled,
                           VK_IMAGE_ASPECT_COLOR_BIT);
                }
                { SR_GPU_ZONE(profiler_, cmd, "ssr_temporal");
                transition(gbColor_.image, gbColorLayout_, VK_IMAGE_LAYOUT_GENERAL,
                           sync::kCompute, sync::kSampled, sync::kCompute,
                           sync::kStorageReadWrite, VK_IMAGE_ASPECT_COLOR_BIT);
                deferred_.recordSsrTemporalPass(cmd, gbSsrHist_, ssrFramesGb_ & 1u, invAoVp,
                                                prevSsrViewProjGb_, renderWidth_, renderHeight_,
                                                /*reset=*/ssrFramesGb_ == 0);
                prevSsrViewProjGb_ = ssaoViewProj;
                ++ssrFramesGb_;
                transition(gbColor_.image, gbColorLayout_,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kCompute,
                           sync::kStorageWrite, sync::kColorAttach, sync::kColorReadWrite,
                           VK_IMAGE_ASPECT_COLOR_BIT);
                }
            } else {
                transition(gbColor_.image, gbColorLayout_,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kCompute,
                           sync::kSampled, sync::kColorAttach, sync::kColorReadWrite,
                           VK_IMAGE_ASPECT_COLOR_BIT);
            }
        }

        if (hasTransparency_) {
            SR_GPU_ZONE(profiler_, cmd, "transparent");
            transition(gbMotion_.image, gbMotionLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                       sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorReadWrite,
                       VK_IMAGE_ASPECT_COLOR_BIT);
            transition(gbReactive_.image, gbReactiveLayout_,
                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kSampleStages,
                       sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                       VK_IMAGE_ASPECT_COLOR_BIT);
            transition(gbDepth_.image, gbDepthLayout_,
                       VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, sync::kFragment,
                       sync::kSampled, sync::kDepthTests, sync::kDepthRead,
                       VK_IMAGE_ASPECT_DEPTH_BIT);
            {
                VkRenderingAttachmentInfo tColors[3] = {
                    makeColorAttachment(gbColor_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                        VK_ATTACHMENT_LOAD_OP_LOAD),
                    makeColorAttachment(gbMotion_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                        VK_ATTACHMENT_LOAD_OP_LOAD),
                    makeColorAttachment(gbReactive_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                        VK_ATTACHMENT_LOAD_OP_CLEAR)};
                VkRenderingAttachmentInfo tDepth =
                    makeDepthAttachment(gbDepth_.view,
                                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                                        VK_ATTACHMENT_LOAD_OP_LOAD);
                beginRendering(cmd, renderWidth_, renderHeight_, 3, tColors, &tDepth);
                deferred_.recordTransparentDraws(cmd, scene_, false, sceneSet, textureSet_,
                                                 transparentSet, materialStride_, renderWidth_,
                                                 renderHeight_, cullViewProj, camera_.position,
                                                 proj.m[14] / proj.m[10], fogParams_.maxDistance,
                                                 volFogEnabled_ && fogParams_.enabled &&
                                                     gbFog_.injectImage != VK_NULL_HANDLE);
                vkCmdEndRendering(cmd);
            }
            transition(gbMotion_.image, gbMotionLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                       VK_IMAGE_ASPECT_COLOR_BIT);
            transition(gbReactive_.image, gbReactiveLayout_,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kColorAttach,
                       sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                       VK_IMAGE_ASPECT_COLOR_BIT);
            transition(gbDepth_.image, gbDepthLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       sync::kDepthTests, sync::kDepthRead, sync::kSampleStages, sync::kSampled,
                       VK_IMAGE_ASPECT_DEPTH_BIT);
        }

        transition(gbColor_.image, gbColorLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        recordBloom(gbBloom_, gbColor_.image, gbColorLayout_, renderWidth_, renderHeight_);
        recordPostFx(gbPostFx_, gbColor_.image, gbColorLayout_, proj, renderHeight_);
    };

    if (gbuffer) {
        if (mixedSpatial) {
            recordLrDeferred(fr.sceneSetGbSpatial, fr.lightingSetGbSpatial,
                             fr.transparentSetGbSpatial, fr.ssrSetGbSpatial,
                             Mat4::multiply(proj, view), proj, "lr_pass_spatial");
            transition(gbColor_.image, gbColorLayout_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       sync::kSampleStages, sync::kSampled, sync::kCopy, sync::kTransferRead,
                       VK_IMAGE_ASPECT_COLOR_BIT);
            transition(gbColorSpatial_.image, gbColorSpatialLayout_,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, sync::kSampleStages, sync::kSampled,
                       sync::kCopy, sync::kTransferWrite, VK_IMAGE_ASPECT_COLOR_BIT);
            VkImageCopy copy = {};
            copy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy.srcSubresource.layerCount = 1;
            copy.dstSubresource = copy.srcSubresource;
            copy.extent = {renderWidth_, renderHeight_, 1};
            vkCmdCopyImage(cmd, gbColor_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           gbColorSpatial_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            transition(gbColorSpatial_.image, gbColorSpatialLayout_,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kCopy, sync::kTransferWrite,
                       sync::kSampleStages, sync::kSampled, VK_IMAGE_ASPECT_COLOR_BIT);
        }
        recordLrDeferred(fr.sceneSetGb, fr.lightingSetGb, fr.transparentSetGb, fr.ssrSetGb,
                         Mat4::multiply(projJittered, view), projJittered, "lr_pass");
        // Coverage mask conditioning: upscalers consume the 3x3-max dilated,
        // motion-gated copy — the plateau covers samplers at the jittered
        // coordinate, and static pixels keep their history weight
        // (see reactive_dilate.comp).  gbMotion_ is already SHADER_READ_ONLY.
        if (hasTransparency_) {
            SR_GPU_ZONE(profiler_, cmd, "reactive_dilate");
            transition(gbReactiveDilated_.image, gbReactiveDilatedLayout_, VK_IMAGE_LAYOUT_GENERAL,
                       sync::kSampleStages, sync::kSampled, sync::kCompute, sync::kStorageWrite,
                       VK_IMAGE_ASPECT_COLOR_BIT);
            deferred_.recordReactiveDilatePass(cmd, reactiveDilateSet_, renderWidth_, renderHeight_);
            transition(gbReactiveDilated_.image, gbReactiveDilatedLayout_,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kCompute,
                       sync::kStorageWrite, sync::kSampleStages, sync::kSampled,
                       VK_IMAGE_ASPECT_COLOR_BIT);
        }
    }

    // --- Auto exposure: LR histogram of this frame's lit HDR (gbColor_) --------
    // Feeds the upscaler preExposure + the algorithm columns.  The applied
    // value is the solve harvested kFramesInFlight frames ago (engine-style
    // latency).  gbColor_ is SHADER_READ_ONLY here with compute in the
    // sampled-dst scope.
    if (gbuffer && autoExposureEnabled_) {
        SR_GPU_ZONE(profiler_, cmd, "auto_exposure");
        ExposureSolvePush solve;
        solve.minEV = exposureMinEV_;
        solve.maxEV = exposureMaxEV_;
        solve.resetState = (frameIndex == 0 || autoExposureJustEnabled_) ? 1.f : 0.f;
        deferred_.recordAutoExposurePass(cmd, lrExposure_.gpu, solve,
                                         lrExposure_.staging[slot]);
        lrExposure_.pending[slot] = true;
    }

    // --- 2) per-algorithm dispatch ---------------------------------------------
    if (gbuffer) {
        timestamps_.upscaleBegin(cmd, slot);
        { SR_GPU_ZONE(profiler_, cmd, "upscale");

        CameraParams cam;
        std::memcpy(cam.view, view.m, sizeof(cam.view));
        std::memcpy(cam.proj, proj.m, sizeof(cam.proj));
        std::memcpy(cam.prevViewProj, prevViewProj_.m, sizeof(cam.prevViewProj));
        cam.jitterX = jitterX_;
        cam.jitterY = jitterY_;
        cam.prevJitterX = prevJitterX_;
        cam.prevJitterY = prevJitterY_;
        cam.cameraNear = camera_.nearPlane;
        cam.cameraFar = camera_.farPlane;
        cam.fovY = camera_.fovY;
        CameraParams camSpatial = cam;
        camSpatial.jitterX = camSpatial.jitterY = 0.f;
        camSpatial.prevJitterX = camSpatial.prevJitterY = 0.f;

        FrameParams frame;
        frame.frameIndex = static_cast<int>(frameIndex);
        frame.deltaTime = 1.f / 60.f;
        // Real exposure input (was hardcoded 1): the LR path's display
        // exposure — see InputAdapter.h for the preExposure convention.
        frame.preExposure = lrExposureNow();
        frame.resetHistory = (frameIndex == 0);

        for (AlgoColumn& algo : algos_) {
            SR_GPU_ZONE(profiler_, cmd, algo.id.c_str());
            transition(algo.output.image, algo.outputLayout, VK_IMAGE_LAYOUT_GENERAL,
                       sync::kSampleStages, sync::kSampled, sync::kSampleStages,
                       sync::kStorageWrite, VK_IMAGE_ASPECT_COLOR_BIT);

            const bool spatialAlgo = !upscalerNeedsJitter(algo.upscaler.get());
            const bool useSpatialColor = mixedSpatial && spatialAlgo;
            UpscalerResources res;
            res.color = useSpatialColor ? gbColorSpatial_.image : gbColor_.image;
            res.colorView = useSpatialColor ? gbColorSpatial_.view : gbColor_.view;
            res.depth = gbDepth_.image;
            res.depthView = gbDepth_.view;
            res.motion = gbMotion_.image;
            res.motionView = gbMotion_.view;
            if (hasTransparency_) {
                res.reactive = gbReactiveDilated_.image;
                res.reactiveView = gbReactiveDilated_.view;
            }
            res.output = algo.output.image;
            res.outputView = algo.output.view;
            algo.upscaler->dispatch(cmd, res, spatialAlgo ? camSpatial : cam, frame);

            transition(algo.output.image, algo.outputLayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       sync::kSampleStages, sync::kStorageWrite, sync::kSampleStages,
                       sync::kSampled, VK_IMAGE_ASPECT_COLOR_BIT);

            algo.fgReady = false;
            if (algo.frameGen) {
                SR_GPU_ZONE(profiler_, cmd, "frame_gen");
                if (frame.resetHistory) algo.fgHistory = false;
                if (algo.fgHistory) {
                    transition(algo.fgOutput.image, algo.fgOutputLayout, VK_IMAGE_LAYOUT_GENERAL,
                               sync::kSampleStages, sync::kSampled, sync::kSampleStages,
                               sync::kStorageWrite, VK_IMAGE_ASPECT_COLOR_BIT);
                    FrameGenResources fgRes;
                    fgRes.color = algo.output.image;
                    fgRes.colorView = algo.output.view;
                    fgRes.depth = res.depth;
                    fgRes.depthView = res.depthView;
                    fgRes.motion = res.motion;
                    fgRes.motionView = res.motionView;
                    fgRes.output = algo.fgOutput.image;
                    fgRes.outputView = algo.fgOutput.view;
                    FrameParams fgFrame = frame;
                    fgFrame.deltaTime = ui_.lockFps
                                            ? 1.f / static_cast<float>(std::max(15, ui_.lockFpsTarget))
                                            : (fps_ > 1.f ? 1.f / fps_ : 1.f / 30.f);
                    algo.frameGen->dispatch(cmd, fgRes, spatialAlgo ? camSpatial : cam, fgFrame);
                    transition(algo.fgOutput.image, algo.fgOutputLayout,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kSampleStages,
                               sync::kStorageWrite, sync::kFragment, sync::kSampled,
                               VK_IMAGE_ASPECT_COLOR_BIT);
                    algo.fgReady = true;
                }
                algo.fgHistory = true;
            }
        }
        } // zone "upscale"
        timestamps_.upscaleEnd(cmd, slot);
    } else {
        // No upscaler pass: emit a zero-width range so every query slot is
        // written (readback with WAIT_BIT otherwise blocks forever).
        timestamps_.upscaleBegin(cmd, slot);
        timestamps_.upscaleEnd(cmd, slot);
    }

    // --- 3) native-resolution ground truth (no jitter) ---------------------------
    if (gtActive_ && active_.gtSsaa) {
        SR_GPU_ZONE(profiler_, cmd, "gt_ssaa");
        // 200% SSAA: deferred render at 2x into gtSsaaColor_, then
        // box-downsample to display resolution (gtColor_), which stays the
        // metric/display ref.
        const uint32_t sw = dw * 2;
        const uint32_t sh = dh * 2;
        // Phase 7a: SSAA-path occlusion cull against the 2x Hi-Z chain.
        { SR_GPU_ZONE(profiler_, cmd, "occlusion_cull");
        const bool cullActiveSsaa = occlusionEnabled_ && gtSsaaCull_.prevValid && cullCandidates_ > 0;
        deferred_.recordCommandUpload(cmd, slot, gtSsaaCull_, cullCandidates_, cullActiveSsaa);
        if (cullActiveSsaa)
            deferred_.recordOcclusionCull(cmd, gtSsaaCull_, cullCandidates_,
                                          gtSsaaCull_.prevViewProj, gtSsaaPyramid_.mipCount, sw, sh);
        }
        { SR_GPU_ZONE(profiler_, cmd, "gbuffer");
        transition(gtSsaaAlbedo_.image, gtSsaaAlbedoLayout_,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kSampleStages, sync::kSampled,
                   sync::kColorAttach, sync::kColorWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaNormal_.image, gtSsaaNormalLayout_,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kSampleStages, sync::kSampled,
                   sync::kColorAttach, sync::kColorWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaMaterial_.image, gtSsaaMaterialLayout_,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kSampleStages, sync::kSampled,
                   sync::kColorAttach, sync::kColorWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaEmissive_.image, gtSsaaEmissiveLayout_,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kSampleStages, sync::kSampled,
                   sync::kColorAttach, sync::kColorWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaMotion_.image, gtSsaaMotionLayout_,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kSampleStages, sync::kSampled,
                   sync::kColorAttach, sync::kColorWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaDepth_.image, gtSsaaDepthLayout_,
                   VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, sync::kSampleStages,
                   sync::kSampled, sync::kDepthTests, sync::kDepthReadWrite,
                   VK_IMAGE_ASPECT_DEPTH_BIT);
        {
            VkRenderingAttachmentInfo colors[5] = {
                makeColorAttachment(gtSsaaAlbedo_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gtSsaaNormal_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR, 0.f, 0.f, 1.f),
                makeColorAttachment(gtSsaaMaterial_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gtSsaaEmissive_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gtSsaaMotion_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR)};
            VkRenderingAttachmentInfo depth =
                makeDepthAttachment(gtSsaaDepth_.view,
                                    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR);
            beginRendering(cmd, sw, sh, 5, colors, &depth);
            deferred_.recordGBufferDraws(cmd, scene_, true, fr.sceneSetGt, textureSet_,
                                         materialStride_, sw, sh, cullViewProjGt, gtSsaaCull_,
                                         cullRuns_.data(), static_cast<uint32_t>(cullRuns_.size()));
            vkCmdEndRendering(cmd);
        }
        } // zone "gbuffer"
        // dst scope includes compute: the opaque-SSR pass samples the GBuffer.
        { SR_GPU_ZONE(profiler_, cmd, "hiz");
        transition(gtSsaaAlbedo_.image, gtSsaaAlbedoLayout_,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kColorAttach, sync::kColorWrite,
                   sync::kSampleStages, sync::kSampled, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaNormal_.image, gtSsaaNormalLayout_,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kColorAttach, sync::kColorWrite,
                   sync::kSampleStages, sync::kSampled, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaMaterial_.image, gtSsaaMaterialLayout_,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kColorAttach, sync::kColorWrite,
                   sync::kSampleStages, sync::kSampled, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaEmissive_.image, gtSsaaEmissiveLayout_,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kColorAttach, sync::kColorWrite,
                   sync::kSampleStages, sync::kSampled, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaMotion_.image, gtSsaaMotionLayout_,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kColorAttach, sync::kColorWrite,
                   sync::kSampleStages, sync::kSampled, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaDepth_.image, gtSsaaDepthLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kDepthTests, sync::kDepthWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_DEPTH_BIT);
        if (hasTransparency_ || ssrEnabled_ || occlusionEnabled_) {
            deferred_.recordDepthPyramidPass(cmd, gtSsaaPyramid_);
            gtSsaaCull_.prevViewProj = cullViewProjGt; // GT paths never jitter
            gtSsaaCull_.prevValid = true;
        }
        } // zone "hiz"

        // GTAO for the 2x GT path (un-jittered view-projection): view-Z depth
        // chain -> main pass -> temporal EMA -> denoise.  The AO output is
        // also sampled by the opaque-SSR compute pass.
        const Mat4 invAoVpSsaa = Mat4::inverse(cullViewProjGt);
        { SR_GPU_ZONE(profiler_, cmd, "gtao");
        deferred_.recordDepthPyramidPass(cmd, gtSsaaPyramidAo_);
        const float aoMaxLodSsaa =
            static_cast<float>(std::min(gtSsaaPyramidAo_.mipCount - 1, 4u));
        { SR_GPU_ZONE(profiler_, cmd, "ao_main");
        transition(gtSsaaAoRaw_.image, gtSsaaAoRawLayout_, VK_IMAGE_LAYOUT_GENERAL, sync::kCompute,
                   sync::kSampled, sync::kCompute, sync::kStorageWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordSsaoPass(cmd, ssaoSetSsaa_, invAoVpSsaa, frameIndex, camera_.nearPlane,
                                 camera_.farPlane, aoMaxLodSsaa, sw, sh);
        transition(gtSsaaAoRaw_.image, gtSsaaAoRawLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kCompute, sync::kStorageWrite, sync::kCompute, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        }
        const uint32_t aoWriteSsaa = aoFramesSsaa_ & 1u;
        { SR_GPU_ZONE(profiler_, cmd, "ao_temporal");
        deferred_.recordSsaoTemporalPass(cmd, gtSsaaAoHist_, aoWriteSsaa, invAoVpSsaa,
                                         prevAoViewProjSsaa_, sw, sh,
                                         /*reset=*/aoFramesSsaa_ == 0);
        }
        prevAoViewProjSsaa_ = cullViewProjGt;
        ++aoFramesSsaa_;
        { SR_GPU_ZONE(profiler_, cmd, "ao_blur");
        transition(gtSsaaAo_.image, gtSsaaAoLayout_, VK_IMAGE_LAYOUT_GENERAL, sync::kSampleStages,
                   sync::kSampled, sync::kCompute, sync::kStorageWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordSsaoBlurPass(cmd, gtSsaaAoHist_.blurSet[aoWriteSsaa], sw, sh);
        transition(gtSsaaAo_.image, gtSsaaAoLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kCompute, sync::kStorageWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        }
        } // zone "gtao"

        { SR_GPU_ZONE(profiler_, cmd, "lighting");
        transition(gtSsaaColor_.image, gtSsaaColorLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordLightingPass(cmd, fr.lightingSetSsaa, gtSsaaCluster_, slot, viewGt, projGt,
                                     gtSsaaColor_.view, sw, sh);
        }

        // Froxel volumetric fog (Phase 5a) on the 2x GT path.
        if (volFogEnabled_ && fogParams_.enabled && gtSsaaFog_.injectImage != VK_NULL_HANDLE) {
            SR_GPU_ZONE(profiler_, cmd, "volfog");
            if (fogAccumFrameSsaa_ != frameIndex) {
                deferred_.recordVolFogAccumulate(cmd, gtSsaaFog_, gtSsaaCluster_, slot, viewGt,
                                                 projGt, prevFogViewProjSsaa_, fogParams_,
                                                 frameIndex, fogFramesSsaa_ & 1u,
                                                 /*reset=*/fogFramesSsaa_ == 0);
                prevFogViewProjSsaa_ = cullViewProjGt;
                ++fogFramesSsaa_;
                fogAccumFrameSsaa_ = frameIndex;
            }
            transition(gtSsaaColor_.image, gtSsaaColorLayout_, VK_IMAGE_LAYOUT_GENERAL,
                       sync::kColorAttach, sync::kColorWrite, sync::kCompute,
                       sync::kStorageReadWrite, VK_IMAGE_ASPECT_COLOR_BIT);
            deferred_.recordVolFogComposite(cmd, gtSsaaFog_, projGt, fogParams_.maxDistance,
                                            sw, sh);
            transition(gtSsaaColor_.image, gtSsaaColorLayout_,
                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kCompute,
                       sync::kStorageWrite, sync::kColorAttach, sync::kColorReadWrite,
                       VK_IMAGE_ASPECT_COLOR_BIT);
        }

        if (hasTransparency_ || ssrEnabled_) {
            SR_GPU_ZONE(profiler_, cmd, "reflections");
            // Same color mip chain build as the GB path (GENERAL for life);
            // the opaque-SSR pass consumes the same chain.
            transition(gtSsaaColor_.image, gtSsaaColorLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       sync::kColorAttach, sync::kColorWrite, sync::kCompute, sync::kSampled,
                       VK_IMAGE_ASPECT_COLOR_BIT);
            deferred_.recordColorPyramidPass(cmd, gtSsaaColorPyramid_);
            if (ssrEnabled_) {
                // Opaque SSR at 2x before the box downsample: trace into the
                // SSAA RT, then temporal EMA + fused composite on gtSsaaColor_.
                transition(gtSsaaSsrTrace_.image, gtSsaaSsrTraceLayout_, VK_IMAGE_LAYOUT_GENERAL,
                           sync::kCompute, sync::kSampled, sync::kCompute,
                           sync::kStorageWrite, VK_IMAGE_ASPECT_COLOR_BIT);
                deferred_.recordSsrPass(cmd, fr.ssrSetSsaa, cullViewProjGt, sw, sh, ssrStrength_);
                transition(gtSsaaSsrTrace_.image, gtSsaaSsrTraceLayout_,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kCompute,
                           sync::kStorageWrite, sync::kCompute, sync::kSampled,
                           VK_IMAGE_ASPECT_COLOR_BIT);
                transition(gtSsaaColor_.image, gtSsaaColorLayout_, VK_IMAGE_LAYOUT_GENERAL,
                           sync::kCompute, sync::kSampled, sync::kCompute,
                           sync::kStorageReadWrite, VK_IMAGE_ASPECT_COLOR_BIT);
                deferred_.recordSsrTemporalPass(cmd, gtSsaaSsrHist_, ssrFramesSsaa_ & 1u,
                                                invAoVpSsaa, prevSsrViewProjSsaa_, sw, sh,
                                                /*reset=*/ssrFramesSsaa_ == 0);
                prevSsrViewProjSsaa_ = cullViewProjGt;
                ++ssrFramesSsaa_;
                transition(gtSsaaColor_.image, gtSsaaColorLayout_,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kCompute,
                           sync::kStorageWrite, sync::kColorAttach, sync::kColorReadWrite,
                           VK_IMAGE_ASPECT_COLOR_BIT);
            } else {
                transition(gtSsaaColor_.image, gtSsaaColorLayout_,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kCompute,
                           sync::kSampled, sync::kColorAttach, sync::kColorReadWrite,
                           VK_IMAGE_ASPECT_COLOR_BIT);
            }
        }

        // Transparency pass: alpha-blended surfaces over the lit scene (GT
        // path: color only, no motion/mask outputs).
        if (hasTransparency_) {
            SR_GPU_ZONE(profiler_, cmd, "transparent");
            transition(gtSsaaDepth_.image, gtSsaaDepthLayout_,
                       VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, sync::kFragment,
                       sync::kSampled, sync::kDepthTests, sync::kDepthRead,
                       VK_IMAGE_ASPECT_DEPTH_BIT);
            {
                VkRenderingAttachmentInfo tColor =
                    makeColorAttachment(gtSsaaColor_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                        VK_ATTACHMENT_LOAD_OP_LOAD);
                VkRenderingAttachmentInfo tDepth =
                    makeDepthAttachment(gtSsaaDepth_.view,
                                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                                        VK_ATTACHMENT_LOAD_OP_LOAD);
                beginRendering(cmd, sw, sh, 1, &tColor, &tDepth);
                deferred_.recordTransparentDraws(cmd, scene_, true, fr.sceneSetGt, textureSet_,
                                                 fr.transparentSetSsaa, materialStride_, sw, sh,
                                                 cullViewProjGt, gtCam.position,
                                                 projGt.m[14] / projGt.m[10],
                                                 fogParams_.maxDistance,
                                                 volFogEnabled_ && fogParams_.enabled &&
                                                     gtSsaaFog_.injectImage != VK_NULL_HANDLE);
                vkCmdEndRendering(cmd);
            }
            transition(gtSsaaDepth_.image, gtSsaaDepthLayout_,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kDepthTests,
                       sync::kDepthRead, sync::kFragment, sync::kSampled, VK_IMAGE_ASPECT_DEPTH_BIT);
        }

        transition(gtSsaaColor_.image, gtSsaaColorLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);

        // Phase 6a/6b: bloom, then MB/DOF in the 2x domain, before the box
        // downsample.
        recordBloom(gtSsaaBloom_, gtSsaaColor_.image, gtSsaaColorLayout_, sw, sh);
        recordPostFx(gtSsaaPostFx_, gtSsaaColor_.image, gtSsaaColorLayout_, projGt, sh);

        transition(gtColor_.image, gtColorLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        { SR_GPU_ZONE(profiler_, cmd, "ssaa_downsample");
        {
            VkRenderingAttachmentInfo color =
                makeColorAttachment(gtColor_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_DONT_CARE);
            beginRendering(cmd, dw, dh, 1, &color, nullptr);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, downsamplePipeline_);
            VkViewport viewport = {0.f, 0.f, static_cast<float>(dw), static_cast<float>(dh), 0.f,
                                   1.f};
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            VkRect2D scissor = {{0, 0}, {dw, dh}};
            vkCmdSetScissor(cmd, 0, 1, &scissor);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, copyPipelineLayout_, 0, 1,
                                    &gtDownsampleSet_, 0, nullptr);
            // Mode 0: passthrough — the downsample target is HDR linear, not
            // the swapchain (copy.frag HDR branch is for presentation only).
            const float copyPush[4] = {0.f, 0.f, 0.f, 0.f};
            vkCmdPushConstants(cmd, copyPipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof(copyPush), copyPush);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            vkCmdEndRendering(cmd);
        }
        } // zone "ssaa_downsample"
        transition(gtColor_.image, gtColorLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
    } else if (gtActive_) {
        SR_GPU_ZONE(profiler_, cmd, "gt");
        // Phase 7a: GT-path occlusion cull against the 1x Hi-Z chain.
        { SR_GPU_ZONE(profiler_, cmd, "occlusion_cull");
        const bool cullActiveGt = occlusionEnabled_ && gtCull_.prevValid && cullCandidates_ > 0;
        deferred_.recordCommandUpload(cmd, slot, gtCull_, cullCandidates_, cullActiveGt);
        if (cullActiveGt)
            deferred_.recordOcclusionCull(cmd, gtCull_, cullCandidates_, gtCull_.prevViewProj,
                                          gtPyramid_.mipCount, gtW, gtH);
        }
        { SR_GPU_ZONE(profiler_, cmd, "gbuffer");
        transition(gtAlbedo_.image, gtAlbedoLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtNormal_.image, gtNormalLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtMaterial_.image, gtMaterialLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtEmissive_.image, gtEmissiveLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtMotion_.image, gtMotionLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtDepth_.image, gtDepthLayout_, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kDepthTests, sync::kDepthReadWrite,
                   VK_IMAGE_ASPECT_DEPTH_BIT);
        {
            VkRenderingAttachmentInfo colors[5] = {
                makeColorAttachment(gtAlbedo_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gtNormal_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR, 0.f, 0.f, 1.f),
                makeColorAttachment(gtMaterial_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gtEmissive_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gtMotion_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR)};
            VkRenderingAttachmentInfo depth =
                makeDepthAttachment(gtDepth_.view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR);
            beginRendering(cmd, gtW, gtH, 5, colors, &depth);
            deferred_.recordGBufferDraws(cmd, scene_, true, fr.sceneSetGt, textureSet_,
                                         materialStride_, gtW, gtH, cullViewProjGt, gtCull_,
                                         cullRuns_.data(), static_cast<uint32_t>(cullRuns_.size()));
            vkCmdEndRendering(cmd);
        }
        } // zone "gbuffer"
        // dst scope includes compute: the opaque-SSR pass samples the GBuffer.
        { SR_GPU_ZONE(profiler_, cmd, "hiz");
        transition(gtAlbedo_.image, gtAlbedoLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtNormal_.image, gtNormalLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtMaterial_.image, gtMaterialLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtEmissive_.image, gtEmissiveLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtMotion_.image, gtMotionLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtDepth_.image, gtDepthLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kDepthTests, sync::kDepthWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_DEPTH_BIT);
        if (hasTransparency_ || ssrEnabled_ || occlusionEnabled_) {
            deferred_.recordDepthPyramidPass(cmd, gtPyramid_);
            gtCull_.prevViewProj = cullViewProjGt; // GT paths never jitter
            gtCull_.prevValid = true;
        }
        } // zone "hiz"

        // GTAO for the 1x GT path (un-jittered view-projection): view-Z depth
        // chain -> main pass -> temporal EMA -> denoise.  In "GT (Apply
        // scale)" mode gtW/gtH are the low input resolution.  The AO output
        // is also sampled by the opaque-SSR compute pass.
        const Mat4 invAoVpGt = Mat4::inverse(cullViewProjGt);
        { SR_GPU_ZONE(profiler_, cmd, "gtao");
        deferred_.recordDepthPyramidPass(cmd, gtPyramidAo_);
        const float aoMaxLodGt = static_cast<float>(std::min(gtPyramidAo_.mipCount - 1, 4u));
        { SR_GPU_ZONE(profiler_, cmd, "ao_main");
        transition(gtAoRaw_.image, gtAoRawLayout_, VK_IMAGE_LAYOUT_GENERAL, sync::kCompute,
                   sync::kSampled, sync::kCompute, sync::kStorageWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordSsaoPass(cmd, ssaoSetGt_, invAoVpGt, frameIndex, camera_.nearPlane,
                                 camera_.farPlane, aoMaxLodGt, gtW, gtH);
        transition(gtAoRaw_.image, gtAoRawLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kCompute, sync::kStorageWrite, sync::kCompute, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        }
        const uint32_t aoWriteGt = aoFramesGt_ & 1u;
        { SR_GPU_ZONE(profiler_, cmd, "ao_temporal");
        deferred_.recordSsaoTemporalPass(cmd, gtAoHist_, aoWriteGt, invAoVpGt, prevAoViewProjGt_,
                                         gtW, gtH, /*reset=*/aoFramesGt_ == 0);
        }
        prevAoViewProjGt_ = cullViewProjGt;
        ++aoFramesGt_;
        { SR_GPU_ZONE(profiler_, cmd, "ao_blur");
        transition(gtAo_.image, gtAoLayout_, VK_IMAGE_LAYOUT_GENERAL, sync::kSampleStages,
                   sync::kSampled, sync::kCompute, sync::kStorageWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordSsaoBlurPass(cmd, gtAoHist_.blurSet[aoWriteGt], gtW, gtH);
        transition(gtAo_.image, gtAoLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kCompute, sync::kStorageWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        }
        } // zone "gtao"

        { SR_GPU_ZONE(profiler_, cmd, "lighting");
        transition(gtColor_.image, gtColorLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordLightingPass(cmd, fr.lightingSetGt, gtCluster_, slot, viewGt, projGt,
                                     gtColor_.view, gtW, gtH);
        }

        // Froxel volumetric fog (Phase 5a) on the 1x GT path.
        if (volFogEnabled_ && fogParams_.enabled && gtFog_.injectImage != VK_NULL_HANDLE) {
            SR_GPU_ZONE(profiler_, cmd, "volfog");
            if (fogAccumFrameGt_ != frameIndex) {
                deferred_.recordVolFogAccumulate(cmd, gtFog_, gtCluster_, slot, viewGt, projGt,
                                                 prevFogViewProjGt_, fogParams_, frameIndex,
                                                 fogFramesGt_ & 1u, /*reset=*/fogFramesGt_ == 0);
                prevFogViewProjGt_ = cullViewProjGt;
                ++fogFramesGt_;
                fogAccumFrameGt_ = frameIndex;
            }
            transition(gtColor_.image, gtColorLayout_, VK_IMAGE_LAYOUT_GENERAL,
                       sync::kColorAttach, sync::kColorWrite, sync::kCompute,
                       sync::kStorageReadWrite, VK_IMAGE_ASPECT_COLOR_BIT);
            deferred_.recordVolFogComposite(cmd, gtFog_, projGt, fogParams_.maxDistance, gtW, gtH);
            transition(gtColor_.image, gtColorLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                       sync::kCompute, sync::kStorageWrite, sync::kColorAttach,
                       sync::kColorReadWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        }

        if (hasTransparency_ || ssrEnabled_) {
            SR_GPU_ZONE(profiler_, cmd, "reflections");
            // Same color mip chain build as the GB path (GENERAL for life);
            // the opaque-SSR pass consumes the same chain.
            transition(gtColor_.image, gtColorLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       sync::kColorAttach, sync::kColorWrite, sync::kCompute, sync::kSampled,
                       VK_IMAGE_ASPECT_COLOR_BIT);
            deferred_.recordColorPyramidPass(cmd, gtColorPyramid_);
            if (ssrEnabled_) {
                // Opaque SSR, Phase 2d: trace into the GT RT, then temporal
                // EMA + fused composite (in-place RMW on gtColor_).
                transition(gtSsrTrace_.image, gtSsrTraceLayout_, VK_IMAGE_LAYOUT_GENERAL,
                           sync::kCompute, sync::kSampled, sync::kCompute,
                           sync::kStorageWrite, VK_IMAGE_ASPECT_COLOR_BIT);
                deferred_.recordSsrPass(cmd, fr.ssrSetGt, cullViewProjGt, gtW, gtH, ssrStrength_);
                transition(gtSsrTrace_.image, gtSsrTraceLayout_,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kCompute,
                           sync::kStorageWrite, sync::kCompute, sync::kSampled,
                           VK_IMAGE_ASPECT_COLOR_BIT);
                transition(gtColor_.image, gtColorLayout_, VK_IMAGE_LAYOUT_GENERAL,
                           sync::kCompute, sync::kSampled, sync::kCompute,
                           sync::kStorageReadWrite, VK_IMAGE_ASPECT_COLOR_BIT);
                deferred_.recordSsrTemporalPass(cmd, gtSsrHist_, ssrFramesGt_ & 1u, invAoVpGt,
                                                prevSsrViewProjGt_, gtW, gtH,
                                                /*reset=*/ssrFramesGt_ == 0);
                prevSsrViewProjGt_ = cullViewProjGt;
                ++ssrFramesGt_;
                transition(gtColor_.image, gtColorLayout_,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kCompute,
                           sync::kStorageWrite, sync::kColorAttach, sync::kColorReadWrite,
                           VK_IMAGE_ASPECT_COLOR_BIT);
            } else {
                transition(gtColor_.image, gtColorLayout_,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kCompute,
                           sync::kSampled, sync::kColorAttach, sync::kColorReadWrite,
                           VK_IMAGE_ASPECT_COLOR_BIT);
            }
        }

        // Transparency pass: alpha-blended surfaces over the lit scene (GT
        // path: color only, no motion/mask outputs).
        if (hasTransparency_) {
            SR_GPU_ZONE(profiler_, cmd, "transparent");
            transition(gtDepth_.image, gtDepthLayout_,
                       VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, sync::kFragment,
                       sync::kSampled, sync::kDepthTests, sync::kDepthRead,
                       VK_IMAGE_ASPECT_DEPTH_BIT);
            {
                VkRenderingAttachmentInfo tColor =
                    makeColorAttachment(gtColor_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                        VK_ATTACHMENT_LOAD_OP_LOAD);
                VkRenderingAttachmentInfo tDepth =
                    makeDepthAttachment(gtDepth_.view,
                                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                                        VK_ATTACHMENT_LOAD_OP_LOAD);
                beginRendering(cmd, gtW, gtH, 1, &tColor, &tDepth);
                deferred_.recordTransparentDraws(cmd, scene_, true, fr.sceneSetGt, textureSet_,
                                                 fr.transparentSetGt, materialStride_, gtW, gtH,
                                                 cullViewProjGt, gtCam.position,
                                                 projGt.m[14] / projGt.m[10],
                                                 fogParams_.maxDistance,
                                                 volFogEnabled_ && fogParams_.enabled &&
                                                     gtFog_.injectImage != VK_NULL_HANDLE);
                vkCmdEndRendering(cmd);
            }
            transition(gtDepth_.image, gtDepthLayout_,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kDepthTests,
                       sync::kDepthRead, sync::kFragment, sync::kSampled, VK_IMAGE_ASPECT_DEPTH_BIT);
        }

        transition(gtColor_.image, gtColorLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        // Phase 6a/6b: bloom, then MB/DOF on the 1x GT HDR.
        recordBloom(gtBloom_, gtColor_.image, gtColorLayout_, gtW, gtH);
        recordPostFx(gtPostFx_, gtColor_.image, gtColorLayout_, projGt, gtH);
    }
    // --- Auto exposure: GT histogram (own HDR source; gtSsaa uses its 2x HDR) --
    // The GT column solves independently of the LR path — separate pipelines,
    // so differing column exposures are the correct engine behaviour.
    if (gtActive_ && autoExposureEnabled_) {
        SR_GPU_ZONE(profiler_, cmd, "auto_exposure_gt");
        ExposureSolvePush solve;
        solve.minEV = exposureMinEV_;
        solve.maxEV = exposureMaxEV_;
        solve.resetState = (frameIndex == 0 || autoExposureJustEnabled_) ? 1.f : 0.f;
        deferred_.recordAutoExposurePass(cmd, gtExposure_.gpu, solve,
                                         gtExposure_.staging[slot]);
        gtExposure_.pending[slot] = true;
    }
    autoExposureJustEnabled_ = false;
    timestamps_.sceneEnd(cmd, slot);

    // --- 4) GPU metric reduction (compare only, every kMetricInterval frames) ----
    // Metric region: full image at zoom 1; the zoomed view window otherwise.
    const uint32_t numColumns = compareMode ? (1 + static_cast<uint32_t>(algos_.size())) : 1;
    const uint32_t layoutX0 = layoutOriginX();
    const uint32_t colW = (dw - layoutX0) / numColumns;
    uint32_t regX = 0, regY = 0, regW = dw, regH = dh;
    if (compareMode && compareZoom_ > 1.f) {
        float rect[4];
        computeViewRegion(dw, dh, colW, dh, compareZoom_, comparePanU_, comparePanV_, rect);
        regX = std::min(static_cast<uint32_t>(rect[0]), dw - 1);
        regY = std::min(static_cast<uint32_t>(rect[1]), dh - 1);
        regW = std::max(1u, std::min(static_cast<uint32_t>(rect[2]), dw - regX));
        regH = std::max(1u, std::min(static_cast<uint32_t>(rect[3]), dh - regY));
    }
    const uint32_t regBlocksPerRow = (regW + 7) / 8;
    const uint32_t regBlockCount = regBlocksPerRow * ((regH + 7) / 8);
    if (compareMode && !algos_.empty() && frameIndex % kMetricInterval == 0) {
        SR_GPU_ZONE(profiler_, cmd, "metrics");
        uint32_t mask = 0;
        uint32_t algoIndex = 0;
        for (AlgoColumn& algo : algos_) {
            // Half-rate FG: only interpolator columns vs midpoint GT.  Other
            // columns are a visual reference and show "--".
            const bool eligible =
                !anyFg ? true : (algo.frameGen != nullptr && algo.fgReady);
            if (!eligible) {
                ++algoIndex;
                continue;
            }
            mask |= (1u << algoIndex);
            ++algoIndex;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, metricBlocksPipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, metricPipelineLayout_, 0,
                                    1, &algo.metricSet, 0, nullptr);
            MetricPush push;
            push.x = regX;
            push.y = regY;
            push.z = regW;
            push.w = regH;
            push.x2 = regBlocksPerRow;
            // Low-res reference ("GT (Apply scale)"): the ref texture is
            // renderWidth_ x renderHeight_ while the region is in test-image
            // (display) pixels — the shader samples it via normalized UVs.
            push.y2 = active_.gtApplyScale ? 1u : 0u;
            push.z2 = dw;
            push.w2 = dh;
            // Each image is tonemapped with its own path's exposure (test =
            // LR, ref = GT): with auto exposure the PSNR/SSIM numbers include
            // the exposure difference the user actually sees between columns.
            push.exposureTest = lrExposureNow();
            push.exposureRef = gtExposureNow();
            vkCmdPushConstants(cmd, metricPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(push), &push);
            vkCmdDispatch(cmd, regBlocksPerRow, (regH + 7) / 8, 1);

            computeBarrier(cmd, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, metricReducePipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, metricPipelineLayout_, 0,
                                    1, &algo.metricSet, 0, nullptr);
            MetricPush reducePush;
            reducePush.x = regBlockCount;
            vkCmdPushConstants(cmd, metricPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(reducePush), &reducePush);
            vkCmdDispatch(cmd, 1, 1, 1);

            computeBarrier(cmd, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
        if (mask != 0) {
            computeBarrier(cmd, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            VkBufferCopy copyRegion = {};
            copyRegion.srcOffset = 0;
            copyRegion.dstOffset = 0;
            copyRegion.size = static_cast<VkDeviceSize>(algos_.size()) * kMetricFloats * 4;
            vkCmdCopyBuffer(cmd, metricResultBuf_, metricStaging_[slot], 1, &copyRegion);
            metricPending_[slot] = true;
            metricMask_[slot] = mask;
        }
    }

    // --- 5) compose + present.  Viewer interpolator: this pass is the
    //     generated frame (t-0.5); the run loop presents the true frame after.
    // Screenshot capture (composite; with SR_GUI_UI_SHOT also the ImGui UI).
    // The readback+encode is deferred: the run loop polls this frame's fence
    // and hands the pixels to a worker thread (see screenshotInFlight_).
    const bool composeGenerated = compareMode || anyFg;
    recordComposePresent(cmd, swapchainIndex, composeGenerated);
    if (screenshotPending_) {
        if (uiShot_)
            captureUiScreenshotIntoStaging(cmd);
        else
            captureScreenshotIntoStaging(cmd);
        screenshotSlot_ = slot;
        screenshotInFlight_ = true;
        screenshotPending_ = false;
    }

    timestamps_.frameEnd(cmd, slot);
    profiler_.endFrame();
    vkEndCommandBuffer(cmd);

    prevViewProj_ = Mat4::multiply(proj, view);
    prevCamera_ = camera_;
    havePrevCamera_ = true;
}

void GuiApp::recordComposePresent(VkCommandBuffer cmd, uint32_t swapchainIndex,
                                  bool preferGenerated) {
    const uint32_t dw = active_.displayW;
    const uint32_t dh = active_.displayH;
    const bool compareMode = active_.mode == Mode::Compare;
    const uint32_t numColumns = compareMode ? (1 + static_cast<uint32_t>(algos_.size())) : 1;
    const uint32_t layoutX0 = layoutOriginX();
    const uint32_t colW = (dw - layoutX0) / numColumns;

    imageBarrier(cmd, composeImage_.image, composeLayout_,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 sync::kFragment, sync::kSampled, sync::kColorAttach, sync::kColorWrite);
    composeLayout_ = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    {
        SR_GPU_ZONE(profiler_, cmd, "compose");
        VkRenderingAttachmentInfo color =
            makeColorAttachment(composeImage_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_ATTACHMENT_LOAD_OP_CLEAR);
        beginRendering(cmd, dw, dh, 1, &color, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, composePipeline_);

        const float textScale = colW >= 300 ? 2.f : 1.f;
        const float viewZoom = compareMode ? compareZoom_ : 1.f;
        const float viewPanU = compareMode ? comparePanU_ : 0.5f;
        const float viewPanV = compareMode ? comparePanV_ : 0.5f;
        for (uint32_t i = 0; i < numColumns; ++i) {
            const uint32_t x = layoutX0 + i * colW;
            const uint32_t w = (i == numColumns - 1) ? (dw - x) : colW;
            VkViewport viewport = {static_cast<float>(x), 0.f, static_cast<float>(w),
                                   static_cast<float>(dh), 0.f, 1.f};
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            VkRect2D scissor = {{static_cast<int32_t>(x), 0}, {w, dh}};
            vkCmdSetScissor(cmd, 0, 1, &scissor);

            VkDescriptorSet set = gtComposeSet_;
            if (compareMode) {
                if (i == 0) set = gtComposeSet_;
                else {
                    const AlgoColumn& a = algos_[i - 1];
                    set = (preferGenerated && a.fgReady && a.composeSetFg) ? a.composeSetFg
                                                                          : a.composeSet;
                }
            } else if (!algos_.empty()) {
                const AlgoColumn& a = algos_[0];
                set = (preferGenerated && a.fgReady && a.composeSetFg) ? a.composeSetFg
                                                                      : a.composeSet;
            }
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, composePipelineLayout_,
                                    0, 1, &set, 0, nullptr);
            ComposePush push;
            push.colSize[0] = static_cast<float>(w);
            push.colSize[1] = static_cast<float>(dh);
            push.textScale = textScale;
            push.textSlot = static_cast<float>(i);
            const bool isGtColumn = (set == gtComposeSet_);
            const float srcW =
                (isGtColumn && active_.gtApplyScale) ? static_cast<float>(renderWidth_)
                                                     : static_cast<float>(dw);
            const float srcH =
                (isGtColumn && active_.gtApplyScale) ? static_cast<float>(renderHeight_)
                                                     : static_cast<float>(dh);
            float rect[4];
            computeViewRegion(dw, dh, w, dh, viewZoom, viewPanU, viewPanV, rect);
            push.uvRect[0] = rect[0] / static_cast<float>(dw);
            push.uvRect[1] = rect[1] / static_cast<float>(dh);
            push.uvRect[2] = rect[2] / static_cast<float>(dw);
            push.uvRect[3] = rect[3] / static_cast<float>(dh);
            push.srcSize[0] = srcW;
            push.srcSize[1] = srcH;
            const float srcRegionW = rect[2] * (srcW / static_cast<float>(dw));
            // Nearest only above 1:1 magnification (see CompareApp); at
            // exactly 1:1 linear sampling hits texel centers anyway and the
            // lens chain stays active.
            push.nearest = (static_cast<float>(w) > srcRegionW) ? 1.f : 0.f;
            // Per-column exposure: GT column uses the GT path's solver,
            // algorithm columns the LR path's (manual mode shares the slider).
            push.exposure = isGtColumn ? gtExposureNow() : lrExposureNow();
            // Terminal lens chain (Phase 6a): per-frame push constants, same
            // strengths for every column (each column presents independently).
            push.lensA[0] = lensCaEnabled_ ? kLensCaStrength : 0.f;
            push.lensA[1] = lensVignetteEnabled_ ? kLensVignetteStrength : 0.f;
            push.lensA[2] = lensGrainEnabled_ ? kLensGrainStrength : 0.f;
            push.lensA[3] = static_cast<float>(renderFrameIndex_);
            // Grading (Phase 6c): shared sliders, identical for every column.
            const Vec3 wb = whiteBalanceForTemperatureTint(gradeTemperatureK_, gradeTint_);
            push.gradeA[0] = gradeContrast_;
            push.gradeA[1] = gradeSaturation_;
            push.gradeA[2] = 0.f;
            push.gradeA[3] = 0.f;
            push.gradeB[0] = wb.x;
            push.gradeB[1] = wb.y;
            push.gradeB[2] = wb.z;
            push.gradeB[3] = 17.f; // makeIdentityLut() default edge length
            vkCmdPushConstants(cmd, composePipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof(push), &push);
            vkCmdDraw(cmd, 3, 1, 0, 0);
        }
        vkCmdEndRendering(cmd);
    }
    imageBarrier(cmd, composeImage_.image, composeLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled);
    composeLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    const VkImage swapImage = swapchain_.image(swapchainIndex);
    const VkImageView swapView = swapchain_.view(swapchainIndex);
    imageBarrier(cmd, swapImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 sync::kColorAttach, sync::kColorWrite, sync::kColorAttach, sync::kColorWrite);
    {
        SR_GPU_ZONE(profiler_, cmd, "present");
        VkRenderingAttachmentInfo color =
            makeColorAttachment(swapView, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_ATTACHMENT_LOAD_OP_CLEAR);
        beginRendering(cmd, swapchain_.extent().width, swapchain_.extent().height, 1, &color,
                       nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, copyPipeline_);
        VkViewport viewport = {0.f, 0.f, static_cast<float>(swapchain_.extent().width),
                               static_cast<float>(swapchain_.extent().height), 0.f, 1.f};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        VkRect2D scissor = {{0, 0}, {swapchain_.extent().width, swapchain_.extent().height}};
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, copyPipelineLayout_, 0, 1,
                                &copySet_, 0, nullptr);
        // HDR swapchain (Phase 6c): re-linearize the SDR composite and encode
        // (SDR content in an HDR container; paper white 203 nits, BT.2408).
        const float copyPush[4] = {static_cast<float>(swapchain_.hdrMode()), 203.f, 0.f, 0.f};
        vkCmdPushConstants(cmd, copyPipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(copyPush), copyPush);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
        vkCmdEndRendering(cmd);
    }
    imageBarrier(cmd, swapImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 sync::kColorAttach, sync::kColorWrite, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE);
}

void GuiApp::recordViewerTruePresent(uint32_t uiSlot, uint32_t swapchainIndex) {
    VkCommandBuffer cmd = uiFrames_[uiSlot].cmd;
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);
    recordComposePresent(cmd, swapchainIndex, false);
    vkEndCommandBuffer(cmd);
}

bool GuiApp::guiWantVsync() const {
    // Lock owns present cadence (15 → 30 with FG).  FIFO vsync on a 60 Hz
    // panel would either cap us at refresh or stack extra waits on the sleep.
    return !ui_.lockFps;
}

bool GuiApp::guiAllowMailbox() const {
    if (!guiWantVsync()) return false;
    if (active_.mode != Mode::Viewer) return true;
    for (const RenderConfig::AlgoSpec& a : active_.algos) {
        if (!a.fg.empty()) return false;
    }
    return true;
}

bool GuiApp::recreateGuiSwapchain() {
    swapchainVsync_ = guiWantVsync();
    swapchainMailbox_ = guiAllowMailbox();
    // The swapchain extent follows the actual window client size in PHYSICAL
    // PIXELS (border drag-resize, fullscreen switches, output-resolution
    // Apply); the fixed output-resolution composite is scaled to it by the
    // present copy pass.  A minimized window reports 0 — keep the configured
    // size then (the acquire/present OUT_OF_DATE loop retries on restore).
    uint32_t w = active_.displayW;
    uint32_t h = active_.displayH;
    if (window_.pixelWidth() > 0 && window_.pixelHeight() > 0) {
        w = static_cast<uint32_t>(window_.pixelWidth());
        h = static_cast<uint32_t>(window_.pixelHeight());
    }
    if (!swapchain_.create(ctx_, w, h, swapchainVsync_, swapchainMailbox_, hdrEnabled_))
        return false;
    ensurePresentSemaphores();
    return true;
}

void GuiApp::ensureGuiSwapchainMode() {
    if (swapchainVsync_ == guiWantVsync() && swapchainMailbox_ == guiAllowMailbox()) return;
    vkDeviceWaitIdle(ctx_.device);
    recreateGuiSwapchain();
}

void GuiApp::waitUntil(std::chrono::steady_clock::time_point t) {
    if (std::chrono::steady_clock::now() < t) std::this_thread::sleep_until(t);
}

void GuiApp::noteDisplayPresents(int nPres, std::chrono::steady_clock::time_point frameStart) {
    const float dt = std::chrono::duration<float>(std::chrono::steady_clock::now() - frameStart).count();
    if (dt <= 0.f || nPres <= 0) return;
    fps_ = fps_ * 0.95f + (static_cast<float>(nPres) / dt) * 0.05f;
    const float dispMs = dt * 1000.f / static_cast<float>(nPres);
    for (int i = 0; i < nPres; ++i) {
        frameMsHistory_[historyHead_] = dispMs;
        historyHead_ = (historyHead_ + 1) % kHistoryLen;
        if (historyCount_ < kHistoryLen) ++historyCount_;
    }
}

// Debug UI screenshot: replicate the present block (composite copy + ImGui
// draw data) into uiShotImage_ (swapchain format, so the copy pipeline and
// the ImGui backend pipeline both apply), then read it back.  Runs in
// addition to the real present; used only when SR_GUI_UI_SHOT is set.
void GuiApp::captureUiScreenshotIntoStaging(VkCommandBuffer cmd) {
    const uint32_t dw = active_.displayW;
    const uint32_t dh = active_.displayH;

    imageBarrier(cmd, uiShotImage_.image, uiShotLayout_,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 sync::kCopy, sync::kTransferRead, sync::kColorAttach, sync::kColorWrite);
    uiShotLayout_ = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    {
        VkRenderingAttachmentInfo color =
            makeColorAttachment(uiShotImage_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_ATTACHMENT_LOAD_OP_CLEAR);
        beginRendering(cmd, dw, dh, 1, &color, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, copyPipeline_);
        VkViewport viewport = {0.f, 0.f, static_cast<float>(dw), static_cast<float>(dh), 0.f, 1.f};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        VkRect2D scissor = {{0, 0}, {dw, dh}};
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, copyPipelineLayout_, 0, 1,
                                &copySet_, 0, nullptr);
        // Same encode as the real present (the UI-shot target uses the
        // swapchain format).
        const float copyPush[4] = {static_cast<float>(swapchain_.hdrMode()), 203.f, 0.f, 0.f};
        vkCmdPushConstants(cmd, copyPipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(copyPush), copyPush);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
        vkCmdEndRendering(cmd);
    }
    imageBarrier(cmd, uiShotImage_.image, uiShotLayout_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 sync::kColorAttach, sync::kColorWrite, sync::kCopy, sync::kTransferRead);
    uiShotLayout_ = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkBufferImageCopy region = {};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {dw, dh, 1};
    vkCmdCopyImageToBuffer(cmd, uiShotImage_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           screenshotStaging_, 1, &region);
}

void GuiApp::collectScreenshotPixels() {
    // The capture copy is complete; hand the pixels to a worker thread for
    // PNG encoding (stb deflate is slow) so the main loop never stalls.
    screenshotInFlight_ = false;
    const uint32_t dw = active_.displayW;
    const uint32_t dh = active_.displayH;
    const size_t bytes = static_cast<size_t>(dw) * dh * 4;
    auto pixels = std::make_shared<std::vector<uint8_t>>(bytes);
    std::memcpy(pixels->data(), screenshotMapped_, bytes);
    const std::string path = screenshotPathPending_;
    screenshotPathPending_.clear();
    const bool swizzle = uiShot_; // UI-shot target is BGRA
    ++screenshotShared_->threads;
    // Capture a shared_ptr copy, not `this`: the worker is detached and may
    // outlive the GuiApp on an abnormal exit.
    auto shot = screenshotShared_;
    std::thread([pixels, path, dw, dh, swizzle, shot]() {
        if (swizzle) {
            uint8_t* p = pixels->data();
            const size_t n = static_cast<size_t>(dw) * dh;
            for (size_t i = 0; i < n; ++i) std::swap(p[i * 4 + 0], p[i * 4 + 2]);
        }
        const bool ok = savePngFromRgba8(path.c_str(), pixels->data(), dw, dh);
        {
            std::lock_guard<std::mutex> lk(shot->msgMutex);
            shot->msg = (ok ? "screenshot saved -> " : "screenshot FAILED -> ") + path;
        }
        ++shot->finished;
        --shot->threads;
    }).detach();
}

void GuiApp::saveScreenshot(const char* path) {
    // Reject while a capture is already queued or in flight: a second request
    // would overwrite screenshotPathPending_/screenshotSlot_, so the first
    // capture's fence would never be polled again and the shot silently lost.
    if (screenshotPending_ || screenshotInFlight_) {
        statusLine_ = "screenshot: already in progress";
        return;
    }
    if (!stackOk_) {
        statusLine_ = "screenshot: render stack is not ready";
        return;
    }
    if (!path || path[0] == '\0' || !screenshotMapped_) {
        statusLine_ = "screenshot: invalid path or staging buffer";
        return;
    }
    // Anchor relative paths at the exe dir (not the CWD, which depends on
    // how the app was launched) and report the full path so the file is
    // always findable.
    screenshotPending_ = true;
    screenshotPathPending_ = resolveOutputPath(path);
    statusLine_ = "capturing screenshot... -> " + screenshotPathPending_;
}

// Small animated indicator shown next to the save-screenshot buttons while a
// capture/encode is in flight (request queued, GPU readback pending, or a
// worker thread still PNG-encoding).
void GuiApp::drawScreenshotBusy() {
    if (!screenshotPending_ && !screenshotInFlight_ &&
        screenshotShared_->threads.load(std::memory_order_relaxed) == 0)
        return;
    ImGui::SameLine();
    static const char* kDots[] = {"", ".", "..", "..."};
    const int phase = static_cast<int>(ImGui::GetTime() * 4.0) % 4;
    ImGui::TextDisabled("working%s", kDots[phase]);
}

void GuiApp::exportFrameTimesCsv(const char* path) {
    if (!path || path[0] == '\0') {
        statusLine_ = "frame-times CSV: invalid path";
        return;
    }
    const MemoryBudgetInfo budget = queryMemoryBudget(ctx_);
    VkDeviceSize heapTotal = 0;
    VkDeviceSize vmaTotal = 0;
    for (uint32_t i = 0; i < budget.heapCount; ++i) {
        heapTotal += budget.heapUsage[i];
        vmaTotal += budget.vmaAllocationBytes[i];
    }
    uint64_t algoBytes = 0;
    for (const AlgoColumn& algo : algos_) algoBytes += algo.upscaler->gpuMemoryBytes();

    std::ofstream csv(path);
    if (!csv) {
        statusLine_ = std::string("frame-times CSV: cannot open ") + path;
        return;
    }
    csv << "frame,frameMs,sceneMs,upscaleMs,vramAlgoBytes,vramTotalBytes,vramVmaBytes\n";
    for (size_t i = 0; i < frameTimesLog_.size(); ++i) {
        csv << i << ',' << frameTimesLog_[i].frameMs << ',' << frameTimesLog_[i].sceneMs << ','
            << frameTimesLog_[i].upscaleMs << ',' << algoBytes << ',' << heapTotal << ','
            << vmaTotal << '\n';
    }
    char buf[320];
    std::snprintf(buf, sizeof(buf), "frame-times: %zu frames -> %s", frameTimesLog_.size(), path);
    statusLine_ = buf;
}

// ---------------------------------------------------------------------------
// Debug input automation — see GuiApp.h for the command list.  One command is
// consumed per frame; events are queued after the platform backend's own
// per-frame events so they win the same-frame ordering.
// ---------------------------------------------------------------------------
void GuiApp::pumpInputFile() {
    if (inputFile_.empty()) return;
    std::ifstream file(inputFile_);
    if (!file) return;
    ImGuiIO& io = ImGui::GetIO();
    std::string line;
    uint64_t n = 0;
    while (std::getline(file, line)) {
        if (++n <= inputFileLine_) continue;
        inputFileLine_ = n;
        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;
        if (cmd == "pos") {
            float x = 0.f, y = 0.f;
            ss >> x >> y;
            io.AddMousePosEvent(x, y);
        } else if (cmd == "wheel") {
            float w = 0.f;
            ss >> w;
            io.AddMouseWheelEvent(0.f, w);
        } else if (cmd == "down" || cmd == "up") {
            int b = 0;
            ss >> b;
            io.AddMouseButtonEvent(b, cmd == "down");
        } else if (cmd == "key" || cmd == "keyup") {
            std::string k;
            ss >> k;
            if (k == "F1") io.AddKeyEvent(ImGuiKey_F1, cmd == "key");
        } else if (cmd == "shot") {
            std::string path;
            ss >> path;
            if (!path.empty()) saveScreenshot(path.c_str());
        } else if (cmd == "graph") {
            graphWindow_.open = !graphWindow_.open; // Render Graph editor window
        } else if (cmd == "profiler") {
            profilerWindow_.open = !profilerWindow_.open; // GPU profiler window
        } else if (cmd == "pass") {
            // pass <name> <0|1>: toggle a runtime-switchable pass by its
            // PassToggle shorthand (same path as the graph node checkbox).
            std::string name;
            int on = 0;
            ss >> name >> on;
            rg::PassToggle t = rg::PassToggle::None;
            if (name == "shadows") t = rg::PassToggle::Shadows;
            else if (name == "contact") t = rg::PassToggle::ContactShadows;
            else if (name == "ssr") t = rg::PassToggle::Ssr;
            else if (name == "volfog") t = rg::PassToggle::VolFog;
            else if (name == "occlusion") t = rg::PassToggle::Occlusion;
            else if (name == "bloom") t = rg::PassToggle::Bloom;
            else if (name == "mb") t = rg::PassToggle::MotionBlur;
            else if (name == "dof") t = rg::PassToggle::Dof;
            else if (name == "autoexp") t = rg::PassToggle::AutoExposure;
            if (t != rg::PassToggle::None) applyPassToggle(t, on != 0);
        } else if (cmd == "fullscreen") {
            // fullscreen <0|1>: borderless fullscreen (same as the checkbox).
            int on = 0;
            ss >> on;
            setFullscreenEnabled(on != 0);
        } // "wait" and unknown commands just consume the frame
        if (dbgInputEnabled())
            std::fprintf(stderr, "[inputfile] line %llu: %s\n",
                         static_cast<unsigned long long>(n), line.c_str());
        break;
    }
}

// ---------------------------------------------------------------------------
// Main loop.
// ---------------------------------------------------------------------------
void GuiApp::run() {
    auto lastTime = std::chrono::steady_clock::now();

    while (window_.poll()) {
        // Track the windowed client size (border drags, output-resolution
        // Applies, fullscreen-exit snap) for the engine.toml width/height
        // memory; while fullscreen the window holds the desktop size, which
        // must not overwrite the remembered windowed size.
        if (!fullscreenEnabled_ && window_.width() > 0 && window_.height() > 0) {
            windowedW_ = window_.width();
            windowedH_ = window_.height();
        }
        // Monitor change with a different Windows display scaling: re-apply
        // the ImGui style scale before the next ImGui frame starts.
        if (window_.input().displayScaleChanged) {
            window_.clearDisplayScaleChanged();
            applyUiScale(window_.contentScale());
        }
        ensureGuiSwapchainMode();
        const int lockN = std::max(15, std::min(120, ui_.lockFpsTarget));
        if (currentTab_ != 2 && ui_.lockFps) {
            const auto truePeriod = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(1.0 / static_cast<double>(lockN)));
            const auto nowLock = std::chrono::steady_clock::now();
            if (fpsLockDeadline_.time_since_epoch().count() == 0 ||
                nowLock > fpsLockDeadline_ + truePeriod)
                fpsLockDeadline_ = nowLock;
        }
        const auto now = std::chrono::steady_clock::now();
        const float dtWall = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;
        const float dt = (currentTab_ != 2 && ui_.lockFps)
                             ? (1.f / static_cast<float>(lockN))
                             : dtWall;

        if (pendingRebuild_) applyRebuild();
        // Async rebuild handoff: the worker is done (joined inside), swap the
        // new scene/upscalers in at a device-idle safe point.
        {
            const LoadPhase ph = loadPhase_.load(std::memory_order_acquire);
            if (ph == LoadPhase::Ready || ph == LoadPhase::Failed) finishAsyncRebuild();
        }

        // Automation: auto-exit after N rendered frames (screenshot of the
        // last frame is already flushed by then).
        if (opts_.frames >= 0 && static_cast<int>(renderFrameIndex_) >= opts_.frames &&
            !screenshotPending_)
            break;
        if (benchQuitArmed_ && now >= benchQuitAt_) break;

        // Bench child process bookkeeping.
        bench_.poll();
        if (bench_.finished()) {
            loadBenchCsv(benchOutUsed_.c_str());
            char buf[64];
            std::snprintf(buf, sizeof(buf), "bench finished (exit %d)", bench_.exitCode());
            statusLine_ = buf;
            bench_.stop(); // acknowledge: finished_ -> false, handles released
            if (benchAutoRun_ && !benchQuitArmed_) {
                benchQuitArmed_ = true;
                benchQuitAt_ = now + std::chrono::seconds(10);
            }
        }
        if (benchAutoRun_ && !benchStarted_ && !bench_.running()) {
            startBench();
            benchStarted_ = true;
        }

        updateCamera(dt);

        // engine.toml hot reload: ~1 s mtime poll, per-frame options only.
        pollEngineConfig();
        // engine.toml auto-save: debounced write of UI-side changes.
        pollEngineConfigAutoSave();

        // Debug hook (SR_GUI_DEBUG_INPUT=1): heartbeat to correlate input
        // event dispatch with the frame loop during automation tests.
        if (dbgInputEnabled() && renderFrameIndex_ % 120 == 0)
            std::fprintf(stderr, "[run] frame=%u ms=%llu\n", renderFrameIndex_,
                         static_cast<unsigned long long>(
                             std::chrono::duration_cast<std::chrono::milliseconds>(
                                 now.time_since_epoch())
                                 .count()));

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        pumpInputFile(); // debug automation: queued after backend events
        ImGui::NewFrame();
        drawUi();
        // Must run BEFORE ImGui::Render(): EndFrame() zeroes io.MouseWheel,
        // so reading the wheel after Render() always sees 0 (this was the
        // root cause of the dead wheel zoom).
        handleCompareZoomInput();
        drawLoadOverlay(); // centered progress window while an async rebuild loads
        ImGui::Render();

        // --- degraded path: ImGui only (render stack broken) ------------------
        if (!stackOk_) {
            const uint32_t slot = uiFrameIndex_ % kFramesInFlight;
            vkWaitForFences(ctx_.device, 1, &uiFrames_[slot].fence, VK_TRUE, UINT64_MAX);

            uint32_t swapIndex = 0;
            VkResult acq =
                swapchain_.acquireNext(ctx_, uiFrames_[slot].imageAvailable, swapIndex);
            if (acq == VK_ERROR_OUT_OF_DATE_KHR || acq == VK_SUBOPTIMAL_KHR) {
                vkDeviceWaitIdle(ctx_.device);
                recreateGuiSwapchain();
                continue;
            }
            if (acq != VK_SUCCESS) break;

            vkResetFences(ctx_.device, 1, &uiFrames_[slot].fence);
            recordUiOnlyFrame(slot, swapIndex);

            VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            VkSubmitInfo submit = {};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.waitSemaphoreCount = 1;
            submit.pWaitSemaphores = &uiFrames_[slot].imageAvailable;
            submit.pWaitDstStageMask = &waitStage;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &uiFrames_[slot].cmd;
            submit.signalSemaphoreCount = 1;
            submit.pSignalSemaphores = &renderFinished_[swapIndex];
            VkResult submitRes;
            {
                std::lock_guard<std::mutex> lk(ctx_.queueMutex);
                submitRes = vkQueueSubmit(ctx_.graphicsQueue, 1, &submit, uiFrames_[slot].fence);
            }
            if (submitRes != VK_SUCCESS) break;
            VkResult pres = swapchain_.present(ctx_, swapIndex, renderFinished_[swapIndex]);
            if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) {
                vkDeviceWaitIdle(ctx_.device);
                recreateGuiSwapchain();
            } else if (pres != VK_SUCCESS) {
                break;
            }
            ++uiFrameIndex_;
            if (currentTab_ != 2 && ui_.lockFps) {
                const auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(1.0 / static_cast<double>(lockN)));
                auto nowLock = std::chrono::steady_clock::now();
                if (fpsLockDeadline_.time_since_epoch().count() == 0 || nowLock > fpsLockDeadline_ + period)
                    fpsLockDeadline_ = nowLock;
                waitUntil(fpsLockDeadline_ + period);
                fpsLockDeadline_ += period;
            }
            noteDisplayPresents(1, lastTime);
            continue;
        }

        // --- normal path: full 3D frame + ImGui overlay ------------------------
        const uint32_t frameIndex = renderFrameIndex_;
        const uint32_t slot = frameIndex % kFramesInFlight;

        vkWaitForFences(ctx_.device, 1, &frames_[slot].fence, VK_TRUE, UINT64_MAX);

        // The frame that used this slot (frameIndex - kFramesInFlight) is now
        // complete; harvest GPU timings + metric readback before reuse.
        if (frameIndex >= kFramesInFlight) {
            lastTimings_ = timestamps_.read(ctx_, slot);
            profiler_.harvest(ctx_, slot); // per-pass GPU zones for the profiler panel
            frameTimesLog_.push_back(lastTimings_);
            // Cap the log but keep the most recent half, so a CSV export never
            // loses the entire history when the runaway guard fires.
            if (frameTimesLog_.size() > 100000) {
                const auto drop =
                    static_cast<std::vector<TimestampQuery::Timings>::difference_type>(
                        frameTimesLog_.size() / 2);
                frameTimesLog_.erase(frameTimesLog_.begin(), frameTimesLog_.begin() + drop);
            }
            if (active_.mode == Mode::Viewer && frameIndex % 15 == 0) refreshOverlayText();
        }
        if (metricPending_[slot]) {
            harvestMetrics(slot);
            metricPending_[slot] = false;
        }
        // Same completion event: harvest the auto-exposure solves recorded in
        // that frame (2 frames of latency, engine-style).
        deferred_.harvestExposureChannel(lrExposure_, slot);
        deferred_.harvestExposureChannel(gtExposure_, slot);

        uint32_t swapIndex = 0;
        VkResult acq = swapchain_.acquireNext(ctx_, frames_[slot].imageAvailable, swapIndex);
        if (acq == VK_ERROR_OUT_OF_DATE_KHR || acq == VK_SUBOPTIMAL_KHR) {
            vkDeviceWaitIdle(ctx_.device);
            recreateGuiSwapchain();
            continue;
        }
        if (acq != VK_SUCCESS) break;

        // Reset the fence only once a submission is guaranteed (an early
        // continue on swapchain recreation would otherwise leave it
        // unsignaled forever).
        vkResetFences(ctx_.device, 1, &frames_[slot].fence);

        // Automation: capture the composite on the last frame.
        if (!opts_.screenshotPath.empty() && opts_.frames > 0 &&
            static_cast<int>(frameIndex) == opts_.frames - 1 && !screenshotPending_)
            saveScreenshot(opts_.screenshotPath.c_str());

        // The profiler panel's open flag is the profiler's enable switch: a
        // closed panel records no timestamps (near-zero overhead).  The
        // Render Graph window shows per-pass timings too, so it enables the
        // profiler the same way while open.
        profiler_.setEnabled(profilerWindow_.open || graphWindow_.open);
        const auto recordStart = std::chrono::steady_clock::now();
        recordFrame(frameIndex, swapIndex);
        cpuRecordMs_ = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() -
                                                                recordStart)
                           .count();

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit = {};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &frames_[slot].imageAvailable;
        submit.pWaitDstStageMask = &waitStage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &frames_[slot].cmd;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &renderFinished_[swapIndex];
        VkResult submitRes;
        {
            std::lock_guard<std::mutex> lk(ctx_.queueMutex);
            submitRes = vkQueueSubmit(ctx_.graphicsQueue, 1, &submit, frames_[slot].fence);
        }
        if (submitRes != VK_SUCCESS) break;

        VkResult pres = swapchain_.present(ctx_, swapIndex, renderFinished_[swapIndex]);
        if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) {
            vkDeviceWaitIdle(ctx_.device);
            recreateGuiSwapchain();
        } else if (pres != VK_SUCCESS) {
            break;
        }

        if (screenshotInFlight_ &&
            vkGetFenceStatus(ctx_.device, frames_[screenshotSlot_].fence) == VK_SUCCESS) {
            collectScreenshotPixels();
        }
        if (screenshotShared_->finished.load(std::memory_order_relaxed) > 0) {
            std::lock_guard<std::mutex> lk(screenshotShared_->msgMutex);
            if (!screenshotShared_->msg.empty()) {
                statusLine_ = screenshotShared_->msg;
                screenshotShared_->msg.clear();
            }
            screenshotShared_->finished = 0;
        }

        const bool viewerFg =
            currentTab_ != 2 && active_.mode == Mode::Viewer && !algos_.empty() &&
            algos_[0].fgReady;
        const bool lock = currentTab_ != 2 && ui_.lockFps;
        using clock = std::chrono::steady_clock;
        const auto truePeriod = std::chrono::duration_cast<clock::duration>(
            std::chrono::duration<double>(1.0 / static_cast<double>(lockN)));
        const auto displayPeriod = std::chrono::duration_cast<clock::duration>(
            std::chrono::duration<double>(1.0 / static_cast<double>(lockN * 2)));
        if (!lock) fpsLockDeadline_ = {};

        // Viewer interpolator: first present was the generated frame (t-0.5);
        // second present is the true upscaled frame (t).  The gap between the
        // two presents is 1/(2N) measured from the first present, so they
        // cannot collapse into one monitor refresh (which looked like N fps).
        bool didSecondPresent = false;
        if (viewerFg && pres == VK_SUCCESS) {
            vkWaitForFences(ctx_.device, 1, &frames_[slot].fence, VK_TRUE, UINT64_MAX);
            const auto tAfterFirst = clock::now();
            const uint32_t uiSlot = uiFrameIndex_ % kFramesInFlight;
            vkWaitForFences(ctx_.device, 1, &uiFrames_[uiSlot].fence, VK_TRUE, UINT64_MAX);
            uint32_t swapIndex2 = 0;
            VkResult acq2 =
                swapchain_.acquireNext(ctx_, uiFrames_[uiSlot].imageAvailable, swapIndex2);
            if (acq2 == VK_ERROR_OUT_OF_DATE_KHR || acq2 == VK_SUBOPTIMAL_KHR) {
                vkDeviceWaitIdle(ctx_.device);
                recreateGuiSwapchain();
            } else if (acq2 == VK_SUCCESS) {
                vkResetFences(ctx_.device, 1, &uiFrames_[uiSlot].fence);
                recordViewerTruePresent(uiSlot, swapIndex2);
                if (lock) waitUntil(tAfterFirst + displayPeriod);
                VkPipelineStageFlags waitStage2 = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                VkSubmitInfo submit2 = {};
                submit2.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submit2.waitSemaphoreCount = 1;
                submit2.pWaitSemaphores = &uiFrames_[uiSlot].imageAvailable;
                submit2.pWaitDstStageMask = &waitStage2;
                submit2.commandBufferCount = 1;
                submit2.pCommandBuffers = &uiFrames_[uiSlot].cmd;
                submit2.signalSemaphoreCount = 1;
                submit2.pSignalSemaphores = &renderFinished_[swapIndex2];
                VkResult submitRes2;
                {
                    std::lock_guard<std::mutex> lk(ctx_.queueMutex);
                    submitRes2 =
                        vkQueueSubmit(ctx_.graphicsQueue, 1, &submit2, uiFrames_[uiSlot].fence);
                }
                if (submitRes2 == VK_SUCCESS) {
                    VkResult pres2 =
                        swapchain_.present(ctx_, swapIndex2, renderFinished_[swapIndex2]);
                    if (pres2 == VK_ERROR_OUT_OF_DATE_KHR || pres2 == VK_SUBOPTIMAL_KHR) {
                        vkDeviceWaitIdle(ctx_.device);
                        recreateGuiSwapchain();
                    } else if (pres2 == VK_SUCCESS) {
                        didSecondPresent = true;
                    }
                }
                ++uiFrameIndex_;
            }
        }

        if (lock) {
            waitUntil(fpsLockDeadline_ + truePeriod);
            fpsLockDeadline_ += truePeriod;
        }

        noteDisplayPresents(didSecondPresent ? 2 : 1, lastTime);
        ++renderFrameIndex_;
    }

    vkDeviceWaitIdle(ctx_.device);
    // Drain a pending screenshot (automation --screenshot exits right after
    // the capture frame; the device is idle so the copy has completed).
    if (screenshotInFlight_) collectScreenshotPixels();
    while (screenshotShared_->threads.load(std::memory_order_relaxed) > 0)
        std::this_thread::yield();
    if (dbgInputEnabled())
        std::fprintf(stderr, "[run] loop exited (shouldClose=%d)\n",
                     window_.shouldClose() ? 1 : 0);
}

// ---------------------------------------------------------------------------
// UI.
// ---------------------------------------------------------------------------
const char* GuiApp::fgLabel(int fg) {
    if (fg < 0 || fg >= kFgCount) return kFgLabels[0];
    return kFgLabels[fg];
}

const char* GuiApp::fgId(int fg) {
    if (fg < 0 || fg >= kFgCount) return kFgIds[0];
    return kFgIds[fg];
}

int GuiApp::upscalerIndexById(const std::string& id) const {
    for (size_t i = 0; i < upscalerNames_.size(); ++i)
        if (upscalerNames_[i] == id) return static_cast<int>(i);
    return -1;
}

bool GuiApp::drawGroupedUpscalerCombo(const char* label, int* index, bool includeNative) {
    const int n = static_cast<int>(upscalerNames_.size());
    std::string preview;
    if (includeNative && *index <= 0)
        preview = "native (ground truth)";
    else {
        const int plugin = includeNative ? *index - 1 : *index;
        if (plugin >= 0 && plugin < n) preview = upscalerLabels_[static_cast<size_t>(plugin)];
        else preview = "(none)";
    }
    bool changed = false;
    if (!ImGui::BeginCombo(label, preview.c_str())) return false;
    if (includeNative) {
        if (ImGui::Selectable("native (ground truth)", *index == 0)) {
            *index = 0;
            changed = true;
        }
    }
    for (const AlgoGroup& g : kAlgoGroups) {
        ImGui::Separator();
        ImGui::TextDisabled("%s", g.title);
        for (int k = 0; k < g.count; ++k) {
            const int plugin = upscalerIndexById(g.ids[k]);
            if (plugin < 0) continue;
            const int stored = includeNative ? plugin + 1 : plugin;
            ImGuiSelectableFlags flags = 0;
            if (static_cast<size_t>(plugin) < upscalerAvailable_.size() &&
                !upscalerAvailable_[static_cast<size_t>(plugin)])
                flags |= ImGuiSelectableFlags_Disabled;
            if (ImGui::Selectable(upscalerLabels_[static_cast<size_t>(plugin)].c_str(),
                                  *index == stored, flags)) {
                *index = stored;
                changed = true;
            }
        }
    }
    ImGui::EndCombo();
    return changed;
}

void GuiApp::drawGroupedUpscalerCheckboxes(bool* selected, uint32_t count) {
    for (const AlgoGroup& g : kAlgoGroups) {
        ImGui::TextDisabled("%s", g.title);
        bool first = true;
        for (int k = 0; k < g.count; ++k) {
            const int plugin = upscalerIndexById(g.ids[k]);
            if (plugin < 0 || static_cast<uint32_t>(plugin) >= count) continue;
            if (!first) ImGui::SameLine();
            first = false;
            ImGui::PushID(plugin);
            const bool unavailable = static_cast<size_t>(plugin) < upscalerAvailable_.size() &&
                                     !upscalerAvailable_[static_cast<size_t>(plugin)] &&
                                     !selected[plugin];
            if (unavailable) ImGui::BeginDisabled();
            ImGui::Checkbox(upscalerLabels_[static_cast<size_t>(plugin)].c_str(),
                            &selected[plugin]);
            if (unavailable) ImGui::EndDisabled();
            ImGui::PopID();
        }
    }
}

void GuiApp::drawFrameLockControls() {
    ImGui::Checkbox("lock frame rate", &ui_.lockFps);
    if (!ui_.lockFps) ImGui::BeginDisabled();
    ImGui::SliderInt("render FPS", &ui_.lockFpsTarget, 15, 120, "%d");
    if (!ui_.lockFps) ImGui::EndDisabled();
    const int n = std::max(15, std::min(120, ui_.lockFpsTarget));
    ImGui::TextDisabled("lock %d → interpolator presents %d times/sec (true + generated)", n,
                        n * 2);
}

void GuiApp::drawSharedControls() {
    // Scene: named registry entries + optional custom glTF path override.
    std::vector<const char*> sceneNames;
    for (const SceneEntry& s : scenes_) sceneNames.push_back(s.alias.c_str());
    if (!sceneNames.empty()) {
        ImGui::Combo("scene", &ui_.sceneIndex, sceneNames.data(),
                     static_cast<int>(sceneNames.size()));
        if (ui_.sceneIndex >= 0 && ui_.sceneIndex < static_cast<int>(scenes_.size())) {
            const SceneEntry& s = scenes_[static_cast<size_t>(ui_.sceneIndex)];
            ImGui::TextDisabled("%s%s", s.description.c_str(),
                                s.available ? "" : "  [ASSET MISSING]");
        }
    }
    ImGui::InputTextWithHint("custom glTF", "path overrides the dropdown", ui_.customScene,
                             sizeof(ui_.customScene));
    ImGui::InputText("env map (hdr)", ui_.envMap, sizeof(ui_.envMap));
    ImGui::TextDisabled("env map takes effect on apply (rebuild)");

    ImGui::SliderFloat("render scale", &ui_.renderScale, 0.25f, 1.0f, "%.2f");
    ImGui::Combo("output res", &ui_.outputResIndex, kOutputResNames, 4);

    ImGui::InputText("camera path", ui_.cameraPathFile, sizeof(ui_.cameraPathFile));
    if (ImGui::Button("load path")) loadCameraPathFromUi();
    if (!path_.empty()) {
        ImGui::SameLine();
        if (ImGui::Button(pathPlaying_ ? "pause" : "play")) pathPlaying_ = !pathPlaying_;
        ImGui::SameLine();
        if (ImGui::Button("clear path")) {
            path_.clear();
            pathPlaying_ = false;
        }
        ImGui::TextDisabled("%zu keyframes%s", path_.size(),
                            pathPlaying_ ? " (playing)" : " (paused)");
    }

    // Color grading (Phase 6c): log domain, pre-ACES; identical parameters on
    // every column (GT included) so compare stays fair.  Per-frame push
    // constants — no rebuild.
    ImGui::Separator();
    ImGui::Text("color grading");
    ImGui::SliderFloat("temperature", &gradeTemperatureK_, 3000.f, 10000.f, "%.0f K");
    ImGui::SliderFloat("tint", &gradeTint_, -1.f, 1.f, "%.2f");
    ImGui::SliderFloat("contrast", &gradeContrast_, 0.5f, 2.f, "%.2f");
    ImGui::SliderFloat("saturation", &gradeSaturation_, 0.f, 2.f, "%.2f");
    if (ImGui::Button("reset grading")) {
        gradeTemperatureK_ = 6500.f;
        gradeTint_ = 0.f;
        gradeContrast_ = 1.f;
        gradeSaturation_ = 1.f;
    }

    // HDR output (Phase 6c): gated on surface support; toggling re-creates
    // the swapchain + copy pipeline + ImGui backend.  The composite is SDR
    // display-encoded, so this is an HDR-container compatibility mode here —
    // true scene-HDR headroom is viewer-only (--hdr).
    const bool hdrAny = hdrSupportHdr10_ || hdrSupportScRgb_;
    if (!hdrAny) ImGui::BeginDisabled();
    if (ImGui::Checkbox("hdr output", &hdrEnabled_)) setHdrEnabled(hdrEnabled_);
    if (!hdrAny) ImGui::EndDisabled();
    if (hdrAny) {
        ImGui::TextDisabled("%s%s (sdr content in hdr container)",
                            hdrSupportHdr10_ ? "hdr10" : "",
                            hdrSupportScRgb_ ? (hdrSupportHdr10_ ? " + scrgb" : "scrgb") : "");
    }
    // Borderless fullscreen (desktop mode; [window] fullscreen in
    // engine.toml, hot-reloadable).  The render resolution stays the
    // configured output size; the swapchain is recreated through the normal
    // OUT_OF_DATE path on the mode switch.
    {
        bool fs = fullscreenEnabled_;
        if (ImGui::Checkbox("fullscreen (borderless)", &fs)) setFullscreenEnabled(fs);
    }
}

void GuiApp::drawViewerTab() {
    drawSharedControls();
    ImGui::Separator();

    drawGroupedUpscalerCombo("upscaler", &ui_.viewerUpscaler, true);
    if (ui_.viewerUpscaler == 0) ImGui::BeginDisabled();
    if (ImGui::BeginCombo("frame interpolator", fgLabel(ui_.viewerFg))) {
        for (int i = 0; i < kFgCount; ++i) {
            if (ImGui::Selectable(kFgLabels[i], ui_.viewerFg == i)) ui_.viewerFg = i;
        }
        ImGui::EndCombo();
    }
    if (ui_.viewerUpscaler == 0) ImGui::EndDisabled();
    drawFrameLockControls();
    ImGui::TextDisabled("Viewer presents interpolated, then the true frame (2x)");

    const bool loadInFlight = loadPhase_.load(std::memory_order_acquire) == LoadPhase::Loading;
    if (loadInFlight) ImGui::BeginDisabled();
    if (ImGui::Button("apply (rebuild)", ImVec2(-1.f, 0.f)))
        requestRebuild(configFromUi(Mode::Viewer));
    if (loadInFlight) ImGui::EndDisabled();
    if (loadInFlight) ImGui::TextDisabled("loading... (apply disabled)");
    ImGui::Separator();

    // Lighting: sun / IBL / exposure.  Per-frame UBO, no rebuild.  Scene
    // load applies lightingPresetForScene(); "golden hour" writes the Bistro
    // exterior look onto the sliders.
    ImGui::Text("lighting");
    ImGui::Checkbox("sun", &sunEnabled_);
    if (!sunEnabled_) ImGui::BeginDisabled();
    ImGui::SliderFloat("elevation", &sunElevationDeg_, 5.f, 90.f, "%.0f deg");
    bool sunMoved = ImGui::IsItemDeactivatedAfterEdit();
    ImGui::SliderFloat("azimuth", &sunAzimuthDeg_, 0.f, 360.f, "%.0f deg");
    sunMoved = sunMoved || ImGui::IsItemDeactivatedAfterEdit();
    // Sky atmosphere follows the sun (Phase 5b); the IBL re-render is a
    // blocking one-shot, so it runs on slider release only, atmosphere mode.
    if (sunMoved) updateSkyFromUiSun();
    ImGui::SliderFloat("intensity", &sunIntensity_, 0.f, 10.f, "%.2f");
    if (!sunEnabled_) ImGui::EndDisabled();
    ImGui::Checkbox("fill light", &fillEnabled_);
    ImGui::SliderFloat("IBL intensity", &iblIntensity_, 0.f, 3.f, "%.2f");
    // Auto exposure: histogram-based EV solver (UE4 AutoExposure style),
    // per-frame parameters — no rebuild.  The manual slider applies when the
    // checkbox is off; switching back to manual keeps the current auto value.
    {
        // Routed through applyPassToggle so the Render Graph editor's node
        // checkbox applies the identical side effects.
        bool autoExposure = autoExposureEnabled_;
        if (ImGui::Checkbox("auto exposure", &autoExposure))
            applyPassToggle(rg::PassToggle::AutoExposure, autoExposure);
    }
    if (autoExposureEnabled_) {
        ImGui::SliderFloat("min EV", &exposureMinEV_, -16.f, 16.f, "%.1f");
        ImGui::SliderFloat("max EV", &exposureMaxEV_, -16.f, 16.f, "%.1f");
        exposureMinEV_ = std::min(exposureMinEV_, exposureMaxEV_);
        exposureMaxEV_ = std::max(exposureMinEV_, exposureMaxEV_);
        ImGui::Text("exposure  lr %.2f  gt %.2f (auto)", static_cast<double>(lrExposure_.value),
                    static_cast<double>(gtExposure_.value));
        ImGui::BeginDisabled();
        ImGui::SliderFloat("exposure", &exposure_, 0.1f, 4.f, "%.2f");
        ImGui::EndDisabled();
    } else {
        ImGui::SliderFloat("exposure", &exposure_, 0.1f, 4.f, "%.2f");
    }
    if (ImGui::Button("golden hour")) applyLightingPreset(goldenHourPreset());
    ImGui::SameLine();
    if (ImGui::Button("scene default"))
        applyLightingPreset(lightingPresetForScene(active_.scenePath));
    // CSM sun shadows: per-frame UBO flag + one extra depth pass (no
    // rebuild).  Unavailable when the shadow targets failed to create.
    if (!shadowsActive_) ImGui::BeginDisabled();
    ImGui::Checkbox("shadows", &shadowsEnabled_);
    if (!shadowsEnabled_) ImGui::BeginDisabled();
    ImGui::Checkbox("cascade debug", &shadowDebugCascades_);
    // Screen-space contact shadows (sun only): per-frame UBO flag, no rebuild.
    ImGui::Checkbox("contact shadows", &contactShadowsEnabled_);
    if (!shadowsEnabled_) ImGui::EndDisabled();
    if (!shadowsActive_) ImGui::EndDisabled();
    // Opaque SSR: per-frame pass skip (no rebuild), all deferred paths.
    // CLI/engine.toml-only switch (--ssr / [effects] ssr) — the GUI checkbox
    // is locked and only displays the current state (kept visible so a
    // toml/CLI-enabled SSR is not hidden).
    ImGui::BeginDisabled();
    ImGui::Checkbox("ssr (opaque)", &ssrEnabled_);
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("CLI only: --ssr (or engine.toml [effects] ssr)");
    if (!ssrEnabled_) ImGui::BeginDisabled();
    // Global SSR weight: scales the trace-stage hit confidence; below 1 the
    // composite leans on the IBL fallback (rough ground reads less greasy).
    ImGui::SliderFloat("ssr strength", &ssrStrength_, 0.f, 1.f, "%.2f");
    if (!ssrEnabled_) ImGui::EndDisabled();
    // Froxel volumetric fog: per-frame pass skip; re-enabling restarts the
    // temporal history so stale frames do not bleed in.
    if (!fogParams_.enabled || gbFog_.injectImage == VK_NULL_HANDLE) ImGui::BeginDisabled();
    {
        // Same applyPassToggle routing as "auto exposure" above.
        bool volFog = volFogEnabled_;
        if (ImGui::Checkbox("volumetric fog", &volFog))
            applyPassToggle(rg::PassToggle::VolFog, volFog);
    }
    if (!fogParams_.enabled || gbFog_.injectImage == VK_NULL_HANDLE) ImGui::EndDisabled();
    // Screen-size LOD + small-object cull: per-frame CPU selection, no rebuild.
    ImGui::Checkbox("lod", &lodEnabled_);
    // GPU Hi-Z occlusion culling (Phase 7a): per-frame pass skip, no rebuild.
    ImGui::Checkbox("occlusion cull", &occlusionEnabled_);
    // HDR bloom (Phase 6a pyramid): per-frame pass skip like SSR, no rebuild —
    // the per-path chains stay allocated either way.
    ImGui::Checkbox("bloom", &bloomEnabled_);
    // Motion blur + DOF (Phase 6b): per-frame pass skip, no temporal state.
    ImGui::Checkbox("motion blur", &motionBlurEnabled_);
    ImGui::Checkbox("depth of field", &dofEnabled_);
    // DOF tuning: per-frame push constants, no rebuild.  Focus 0 = auto-focus
    // on the screen-centre depth texel.
    if (!dofEnabled_) ImGui::BeginDisabled();
    ImGui::SliderFloat("dof focus (m)", &dofFocus_, 0.f, 60.f, "%.1f (0 = auto)");
    ImGui::SliderFloat("dof f-stop", &dofFstop_, 0.7f, 22.f, "f/%.1f");
    ImGui::SliderFloat("dof max blur (px)", &dofMaxBlur_, 1.f, 32.f, "%.0f");
    if (!dofEnabled_) ImGui::EndDisabled();
    // Terminal lens-effects chain (Phase 6a, compare_compose.frag; per-frame
    // push constants, no rebuild).  Lens dirt stays viewer-only: the compose
    // pass has no dirt binding to modulate with the bloom pyramid.
    ImGui::Text("lens fx");
    ImGui::Checkbox("chromatic aberration", &lensCaEnabled_);
    ImGui::Checkbox("vignette", &lensVignetteEnabled_);
    ImGui::Checkbox("film grain", &lensGrainEnabled_);
    ImGui::Separator();

    // Live performance readout.
    {
        const float dispMs =
            fps_ > 1.f ? 1000.f / fps_ : static_cast<float>(lastTimings_.frameMs);
        ImGui::Text("display %6.2f ms  (%5.1f FPS)", static_cast<double>(dispMs),
                    static_cast<double>(fps_));
    }
    ImGui::Text("gpu     %6.2f ms", lastTimings_.frameMs);
    ImGui::Text("scene   %6.2f ms", lastTimings_.sceneMs);
    ImGui::Text("upscale %6.2f ms", lastTimings_.upscaleMs);
    if (historyCount_ > 1) {
        // Linearize the ring buffer for PlotLines.
        float ordered[kHistoryLen];
        for (uint32_t i = 0; i < historyCount_; ++i) {
            ordered[i] = frameMsHistory_[(historyHead_ + kHistoryLen - historyCount_ + i) %
                                         kHistoryLen];
        }
        ImGui::PlotLines("display ms", ordered, static_cast<int>(historyCount_), 0, nullptr, 0.f,
                         33.f, ImVec2(-1.f, 60.f));
    }
    ImGui::Separator();

    ImGui::InputText("screenshot png", ui_.viewerShotPath, sizeof(ui_.viewerShotPath));
    if (ImGui::Button("save screenshot")) saveScreenshot(ui_.viewerShotPath);
    drawScreenshotBusy();
    ImGui::InputText("frame-times csv", ui_.frameTimesPath, sizeof(ui_.frameTimesPath));
    if (ImGui::Button("export frame-times CSV")) exportFrameTimesCsv(ui_.frameTimesPath);
    ImGui::SameLine();
    if (ImGui::Button("clear log")) frameTimesLog_.clear();
    ImGui::TextDisabled("%zu frames logged", frameTimesLog_.size());
}

void GuiApp::drawCompareTab() {
    drawSharedControls();
    ImGui::Separator();

    const uint32_t selected = static_cast<uint32_t>(ui_.compareSlots.size());
    ImGui::Text("algorithms (%u/%u)", selected, kMaxAlgos);
    const bool canAdd = selected < kMaxAlgos && !upscalerNames_.empty();
    if (!canAdd) ImGui::BeginDisabled();
    if (ImGui::Button("+")) {
        UiState::CompareSlot slot;
        slot.sr = 0;
        for (size_t i = 0; i < upscalerNames_.size(); ++i) {
            if (i < upscalerAvailable_.size() && upscalerAvailable_[i]) {
                slot.sr = static_cast<int>(i);
                break;
            }
        }
        ui_.compareSlots.push_back(slot);
    }
    if (!canAdd) ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextDisabled("sr / interpolator");

    int removeAt = -1;
    for (int row = 0; row < static_cast<int>(ui_.compareSlots.size()); ++row) {
        UiState::CompareSlot& slot = ui_.compareSlots[static_cast<size_t>(row)];
        ImGui::PushID(row);
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 10.f);
        drawGroupedUpscalerCombo("##sr", &slot.sr, false);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 6.f);
        if (ImGui::BeginCombo("##fg", fgLabel(slot.fg))) {
            for (int i = 0; i < kFgCount; ++i) {
                if (ImGui::Selectable(kFgLabels[i], slot.fg == i)) slot.fg = i;
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        if (ImGui::Button("x")) removeAt = row;
        ImGui::PopID();
    }
    if (removeAt >= 0)
        ui_.compareSlots.erase(ui_.compareSlots.begin() + removeAt);

    drawFrameLockControls();
    ImGui::TextDisabled("FG columns: PSNR vs midpoint GT; columns without FG show --");

    ImGui::TextDisabled("GT reference: %s (Reference checkbox, above the tabs)",
                        ui_.compareGtSsaa ? "200% SSAA" : "native res");

    // Zoom state (mouse wheel over the render area, middle drag to pan).
    ImGui::Text("zoom %.2fx", static_cast<double>(compareZoom_));
    ImGui::SameLine();
    if (ImGui::Button("reset zoom")) {
        compareZoom_ = 1.f;
        comparePanU_ = 0.5f;
        comparePanV_ = 0.5f;
        refreshOverlayText();
    }
    ImGui::TextDisabled("wheel: zoom at cursor, middle drag: pan");

    const bool any = !ui_.compareSlots.empty();
    const bool loadInFlight = loadPhase_.load(std::memory_order_acquire) == LoadPhase::Loading;
    if (!any || loadInFlight) ImGui::BeginDisabled();
    if (ImGui::Button("apply (rebuild)", ImVec2(-1.f, 0.f)))
        requestRebuild(configFromUi(Mode::Compare));
    if (!any || loadInFlight) ImGui::EndDisabled();
    if (!any) ImGui::TextDisabled("select at least one algorithm");
    if (loadInFlight) ImGui::TextDisabled("loading... (apply disabled)");
    ImGui::Separator();

    // Live metrics (also drawn on the columns themselves).
    if (ImGui::BeginTable("metrics", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("algo");
        ImGui::TableSetupColumn("FPS");
        ImGui::TableSetupColumn("PSNR dB");
        ImGui::TableSetupColumn("SSIM");
        ImGui::TableHeadersRow();
        for (const AlgoColumn& algo : algos_) {
            const float n = ui_.lockFps ? static_cast<float>(std::max(15, std::min(120, ui_.lockFpsTarget)))
                                        : fps_;
            const float colFps = algo.fg.empty() ? n : n * 2.f;
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            if (!algo.fg.empty())
                ImGui::Text("%s + %s", algo.upscaler->name(),
                            algo.fg == "nfru" ? "NFRU" : "FSR3");
            else
                ImGui::TextUnformatted(algo.upscaler->name());
            ImGui::TableNextColumn();
            ImGui::Text("%.0f", static_cast<double>(colFps));
            ImGui::TableNextColumn();
            if (algo.hasMetric) ImGui::Text("%.2f", static_cast<double>(algo.psnr));
            else ImGui::TextDisabled("--");
            ImGui::TableNextColumn();
            if (algo.hasMetric) ImGui::Text("%.4f", static_cast<double>(algo.ssim));
            else ImGui::TextDisabled("--");
        }
        ImGui::EndTable();
    }
    ImGui::Separator();

    ImGui::InputText("screenshot png", ui_.compareShotPath, sizeof(ui_.compareShotPath));
    if (ImGui::Button("save screenshot")) saveScreenshot(ui_.compareShotPath);
    drawScreenshotBusy();
}

void GuiApp::startBench() {
    if (bench_.running()) return;

    std::string list;
    for (uint32_t i = 0; i < upscalerNames_.size() && i < kMaxRegistered; ++i) {
        if (!ui_.benchSelected[i]) continue;
        if (!list.empty()) list += ',';
        list += upscalerNames_[i];
    }
    if (ui_.benchNative) {
        if (!list.empty()) list += ',';
        list += "native";
    }
    if (list.empty()) {
        statusLine_ = "bench: no algorithms selected";
        return;
    }

    // Scene argument: custom path or the registry alias (the child resolves it).
    std::string sceneArg = ui_.customScene;
    if (sceneArg.empty() && ui_.sceneIndex >= 0 &&
        ui_.sceneIndex < static_cast<int>(scenes_.size()))
        sceneArg = scenes_[static_cast<size_t>(ui_.sceneIndex)].alias;

    char scaleBuf[32];
    std::snprintf(scaleBuf, sizeof(scaleBuf), "%.3f", static_cast<double>(ui_.renderScale));
    char resBuf[32];
    std::snprintf(resBuf, sizeof(resBuf), "%ux%u", kOutputResolutions[ui_.outputResIndex][0],
                  kOutputResolutions[ui_.outputResIndex][1]);

    const std::vector<std::string> args = {
        "bench", "--upscalers", list,
        "--frames", std::to_string(ui_.benchFrames),
        "--warmup", std::to_string(ui_.benchWarmup),
        "--output", resBuf,
        "--render-scale", scaleBuf,
        "--scene", sceneArg,
        "--out", ui_.benchOutPath,
    };
    benchOutUsed_ = ui_.benchOutPath;
    const std::string exe = selfExePath();
    if (exe.empty() || !bench_.start(exe, args)) {
        statusLine_ = "bench: failed to spawn child process";
        return;
    }
    statusLine_ = "bench running: " + list;
}

void GuiApp::loadBenchCsv(const char* path) {
    benchRows_.clear();
    benchHeader_.clear();
    std::ifstream file(path);
    if (!file) return;
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const std::vector<std::string> lines = splitCsvLines(text);
    if (lines.empty()) return;
    benchHeader_ = splitComma(lines[0]);
    for (size_t i = 1; i < lines.size(); ++i) {
        BenchRowUi row;
        row.cols = splitComma(lines[i]);
        while (row.cols.size() < benchHeader_.size()) row.cols.push_back("");
        benchRows_.push_back(std::move(row));
    }
}

void GuiApp::drawBenchTab() {
    ImGui::TextWrapped("The bench spawns `sr_compare bench` as a child process and shares the "
                       "GPU with this window — the live view will stutter while it runs.");
    ImGui::Separator();

    ImGui::Text("algorithms:");
    drawGroupedUpscalerCheckboxes(ui_.benchSelected, kMaxRegistered);
    ImGui::Checkbox("native (GT baseline)", &ui_.benchNative);
    ImGui::Separator();

    ImGui::InputInt("frames", &ui_.benchFrames);
    if (ui_.benchFrames < 1) ui_.benchFrames = 1;
    ImGui::InputInt("warmup", &ui_.benchWarmup);
    if (ui_.benchWarmup < 0) ui_.benchWarmup = 0;
    ImGui::TextDisabled("scene / scale / resolution come from the shared controls above:");
    drawSharedControls();
    ImGui::InputText("output csv", ui_.benchOutPath, sizeof(ui_.benchOutPath));

    if (bench_.running()) {
        if (ImGui::Button("stop", ImVec2(-1.f, 0.f))) bench_.stop();
        const int total = bench_.progressTotal();
        const int done = bench_.progressDone();
        const float frac = total > 0 ? static_cast<float>(done) / static_cast<float>(total)
                                     : 0.f;
        ImGui::ProgressBar(frac, ImVec2(-1.f, 0.f),
                           total > 0 ? (std::to_string(done) + "/" + std::to_string(total)).c_str()
                                     : "running...");
    } else {
        if (ImGui::Button("run bench", ImVec2(-1.f, 0.f))) startBench();
        ImGui::SameLine();
        if (ImGui::Button("reload csv")) loadBenchCsv(ui_.benchOutPath);
    }

    // Child log tail.
    if (!bench_.log().empty()) {
        if (ImGui::BeginChild("benchlog", ImVec2(0.f, 120.f), true)) {
            ImGui::TextUnformatted(bench_.log().c_str());
            if (bench_.running()) ImGui::SetScrollHereY(1.f);
        }
        ImGui::EndChild();
    }
    ImGui::Separator();

    // Results table.
    if (!benchHeader_.empty() && ImGui::BeginTable("benchresults",
                                                   static_cast<int>(benchHeader_.size()),
                                                   ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                                       ImGuiTableFlags_ScrollX |
                                                       ImGuiTableFlags_SizingFixedFit,
                                                   ImVec2(0.f, 0.f))) {
        for (const std::string& h : benchHeader_) ImGui::TableSetupColumn(h.c_str());
        ImGui::TableHeadersRow();
        for (const BenchRowUi& row : benchRows_) {
            ImGui::TableNextRow();
            for (size_t c = 0; c < benchHeader_.size(); ++c) {
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(row.cols[c].c_str());
            }
        }
        ImGui::EndTable();
    }
}

// ---------------------------------------------------------------------------
// Reference section: sits inside the side panel, above the tab bar (it
// switches the global comparison standard for all three tabs).  Toggling
// applies immediately (rebuild), no Apply round-trip.
// ---------------------------------------------------------------------------
void GuiApp::drawReferenceSection() {
    ImGui::SeparatorText("Reference");
    // Toggling applies immediately (rebuild); refuse while a load is in flight.
    const bool loadInFlight = loadPhase_.load(std::memory_order_acquire) == LoadPhase::Loading;
    if (loadInFlight) ImGui::BeginDisabled();
    bool rebuild = false;
    bool ssaa = ui_.compareGtSsaa;
    if (ImGui::Checkbox("GT 200% SSAA", &ssaa) && !loadInFlight) {
        ui_.compareGtSsaa = ssaa;
        if (ssaa) ui_.compareGtApplyScale = false; // radio: one GT mode at a time
        rebuild = true;
    }
    bool applyScale = ui_.compareGtApplyScale;
    if (ImGui::Checkbox("GT (Apply scale)", &applyScale) && !loadInFlight) {
        ui_.compareGtApplyScale = applyScale;
        if (applyScale) ui_.compareGtSsaa = false; // radio: one GT mode at a time
        rebuild = true;
    }
    // Instant apply: rebuild the current render stack with the new GT.  The
    // bench tab spawns a child process instead — the flag is picked up by
    // the next viewer/compare rebuild.
    if (rebuild && currentTab_ != 2)
        requestRebuild(configFromUi(currentTab_ == 0 ? Mode::Viewer : Mode::Compare));
    if (loadInFlight) ImGui::EndDisabled();
    ImGui::TextDisabled("SSAA: GT at 2x, downsampled.\nApply scale: GT at the low input res,\nno AA/upscale. Metrics compare against it.");
}

void GuiApp::drawCameraPose() {
    // Bottom-left readout of the camera pose, drawn on top of the render area
    // (after the side panel so the panel does not cover it).  The user reports
    // repro locations with these numbers.  NoInputs: never eats mouse look.
    const ImGuiIO& io = ImGui::GetIO();
    const Vec3& p = camera_.position;
    const Vec3& f = camera_.forward;
    const float deg = 57.2957795f;
    // Camera looks down -Z: yaw 0 faces -Z, positive yaw turns right (+X).
    const float yawDeg = std::atan2(f.x, -f.z) * deg;
    const float pitchDeg = std::asin(std::clamp(f.y, -1.f, 1.f)) * deg;
    const float x = panelCollapsed_ ? 8.f : panelWidth() + 8.f;
    ImGui::SetNextWindowPos(ImVec2(x, io.DisplaySize.y - 8.f), ImGuiCond_Always,
                            ImVec2(0.f, 1.f));
    ImGui::SetNextWindowBgAlpha(0.55f);
    if (ImGui::Begin("##campose", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                         ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav |
                         ImGuiWindowFlags_AlwaysAutoResize |
                         ImGuiWindowFlags_NoFocusOnAppearing)) {
        ImGui::Text("pos (%.1f, %.1f, %.1f)   yaw %.1f° pitch %.1f°", p.x, p.y, p.z,
                    yawDeg, pitchDeg);
        ImGui::Text("fwd (%.3f, %.3f, %.3f)", f.x, f.y, f.z);
    }
    ImGui::End();
}

bool GuiApp::passToggleValue(rg::PassToggle t) const {
    switch (t) {
    case rg::PassToggle::Shadows: return shadowsEnabled_;
    case rg::PassToggle::ContactShadows: return contactShadowsEnabled_;
    case rg::PassToggle::Ssr: return ssrEnabled_;
    case rg::PassToggle::VolFog: return volFogEnabled_;
    case rg::PassToggle::Occlusion: return occlusionEnabled_;
    case rg::PassToggle::Bloom: return bloomEnabled_;
    case rg::PassToggle::MotionBlur: return motionBlurEnabled_;
    case rg::PassToggle::Dof: return dofEnabled_;
    case rg::PassToggle::AutoExposure: return autoExposureEnabled_;
    case rg::PassToggle::LensCa: return lensCaEnabled_;
    case rg::PassToggle::LensVignette: return lensVignetteEnabled_;
    case rg::PassToggle::LensGrain: return lensGrainEnabled_;
    case rg::PassToggle::None: break;
    }
    return true;
}

void GuiApp::applyPassToggle(rg::PassToggle t, bool value) {
    switch (t) {
    case rg::PassToggle::Shadows: shadowsEnabled_ = value; break;
    case rg::PassToggle::ContactShadows: contactShadowsEnabled_ = value; break;
    case rg::PassToggle::Ssr: ssrEnabled_ = value; break;
    case rg::PassToggle::VolFog:
        volFogEnabled_ = value;
        // Same side effect as the panel checkbox: restart the temporal
        // history so stale frames do not bleed in.
        fogFramesGb_ = fogFramesGt_ = fogFramesSsaa_ = 0;
        fogAccumFrameGb_ = fogAccumFrameGt_ = fogAccumFrameSsaa_ = ~0u;
        break;
    case rg::PassToggle::Occlusion: occlusionEnabled_ = value; break;
    case rg::PassToggle::Bloom: bloomEnabled_ = value; break;
    case rg::PassToggle::MotionBlur: motionBlurEnabled_ = value; break;
    case rg::PassToggle::Dof: dofEnabled_ = value; break;
    case rg::PassToggle::AutoExposure:
        autoExposureEnabled_ = value;
        // Same side effect as the panel checkbox: snap the solver when
        // re-enabling, keep the current look when switching to manual.
        if (value)
            autoExposureJustEnabled_ = true;
        else
            exposure_ = lrExposure_.value;
        break;
    case rg::PassToggle::LensCa: lensCaEnabled_ = value; break;
    case rg::PassToggle::LensVignette: lensVignetteEnabled_ = value; break;
    case rg::PassToggle::LensGrain: lensGrainEnabled_ = value; break;
    case rg::PassToggle::None: break;
    }
}

void GuiApp::drawUi() {
    const ImGuiIO& io = ImGui::GetIO();

    // F1 toggles the side panel (same as View > Hide side panel).
    if (ImGui::IsKeyPressed(ImGuiKey_F1, false) && !io.WantTextInput)
        panelCollapsed_ = !panelCollapsed_;

    // Independent floating profiler panel (default closed, anchored top-right
    // on first open); drawn in both panel states.
    profilerWindow_.draw(profiler_, cpuRecordMs_);
    // Render Graph editor window (default closed); same both-states rule.
    graphWindow_.draw(profiler_, [this](rg::PassToggle t) { return passToggleValue(t); },
                      [this](rg::PassToggle t, bool v) { applyPassToggle(t, v); });

    if (panelCollapsed_) {
        // Slim strip with just an expand button; the render columns span the
        // full window width underneath.  Parked below the on-composite
        // overlay text band (3 lines x 8px x textScale(2) + 6px pad = 54px
        // in the top-left corner, see compare_compose.frag) so it never
        // covers that text.  NoScrollbar: the full-width button would
        // otherwise push the content region past the client width.
        ImGui::SetNextWindowPos(ImVec2(0.f, 60.f * uiScale_), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(56.f * uiScale_, 34.f * uiScale_), ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.88f);
        if (ImGui::Begin("##panelstrip", nullptr,
                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar |
                             ImGuiWindowFlags_NoScrollbar |
                             ImGuiWindowFlags_NoScrollWithMouse)) {
            if (ImGui::Button(">>", ImVec2(-1.f, 0.f))) panelCollapsed_ = false;
        }
        ImGui::End();
        drawCameraPose();
        return;
    }

    ImGui::SetNextWindowPos(ImVec2(0.f, 0.f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(panelWidth(), io.DisplaySize.y), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.88f);
    ImGui::Begin("sr_compare", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoCollapse);

    // Collapse button (mirrors the F1 hotkey).
    if (ImGui::SmallButton("<<")) panelCollapsed_ = true;
    ImGui::SameLine();
    ImGui::TextDisabled("hide panel (F1)");
    ImGui::SameLine();
    ImGui::Checkbox("gpu profiler", &profilerWindow_.open);
    ImGui::SameLine();
    ImGui::Checkbox("render graph", &graphWindow_.open);

    // Global reference selector, shared by all three tabs.
    drawReferenceSection();

    // The tab area lives in a child window so a long (scrollable) tab can
    // never push the footer — status line + Exit — past the bottom edge.
    // The footer height is measured from the wrapped status text so long
    // paths cannot push the Exit button out of the clickable area.
    const float wrapW =
        ImGui::GetWindowWidth() - 2.f * ImGui::GetStyle().WindowPadding.x;
    const ImVec2 statusSize =
        ImGui::CalcTextSize(statusLine_.c_str(), nullptr, false, wrapW);
    const float footerH = ImGui::GetFrameHeightWithSpacing() + // Exit button
                          ImGui::GetStyle().ItemSpacing.y * 2.f +
                          std::max(statusSize.y, ImGui::GetTextLineHeight());
    ImGui::BeginChild("##tabarea", ImVec2(0.f, -footerH));

    int newTab = currentTab_;
    if (ImGui::BeginTabBar("modes")) {
        // Launch options may preselect a tab; ImGui otherwise activates the
        // first one, which would look like a user tab switch and trigger a
        // spurious rebuild on the first frame.
        const int req = tabRequest_;
        tabRequest_ = -1;
        if (ImGui::BeginTabItem("Viewer", nullptr,
                                req == 0 ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None)) {
            newTab = 0;
            drawViewerTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Compare", nullptr,
                                req == 1 ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None)) {
            newTab = 1;
            drawCompareTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Bench", nullptr,
                                req == 2 ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None)) {
            newTab = 2;
            drawBenchTab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::EndChild();

    // Tab switches between Viewer/Compare rebuild the render stack with the
    // target tab's stored settings.
    if (newTab != currentTab_) {
        const int oldTab = currentTab_;
        currentTab_ = newTab;
        if (newTab != 2 && (oldTab == 2 || (newTab == 0) != (active_.mode == Mode::Viewer)))
            requestRebuild(configFromUi(newTab == 0 ? Mode::Viewer : Mode::Compare));
    }

    ImGui::Separator();
    ImGui::TextWrapped("%s", statusLine_.c_str());
    // Exit through the normal window-close path: the next Window::poll()
    // returns false and run()/shutdown() destroy the Vulkan stack as usual.
    if (ImGui::Button("Exit", ImVec2(-1.f, 0.f))) window_.requestClose();
    ImGui::End();

    drawCameraPose();
}

} // namespace sr
