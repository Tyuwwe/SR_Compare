// ============================================================================
// GuiApp — see GuiApp.h for the design overview.
// ============================================================================
#include "gui/GuiApp.h"

#include "compare/Font5x7.h"
#include "renderer/Screenshot.h"
#include "renderer/core/MemoryBudget.h"
#include "renderer/core/PathUtil.h"
#include "renderer/core/VkUtil.h"
#include "upscalers/UpscalerFactory.h"
#include "upscalers/dlss/SlContext.h"

#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <imgui_impl_win32.h>
#include <imgui_internal.h> // GImGui->InputEventsQueue (input-path debug)

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

// The Win32 backend header intentionally leaves this declaration inside an
// #if 0 block (to avoid dragging <windows.h> into the header); declare it here.
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg,
                                                             WPARAM wParam, LPARAM lParam);

namespace {
// Debug hook (SR_GUI_DEBUG_INPUT=1): verify the backend actually queues a
// wheel event when the message arrives.
bool dbgInputEnabled() {
    static const bool enabled = std::getenv("SR_GUI_DEBUG_INPUT") != nullptr;
    return enabled;
}

LRESULT guiWndProcHook(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    const int queueBefore = GImGui ? GImGui->InputEventsQueue.Size : -1;
    const LRESULT r = ImGui_ImplWin32_WndProcHandler(hwnd, msg, wp, lp);
    if (dbgInputEnabled() && (msg == WM_MOUSEWHEEL || msg == WM_MBUTTONDOWN)) {
        const ImGuiIO& io = ImGui::GetIO();
        std::fprintf(stderr, "[hook] msg=0x%04x ret=%lld queue=%d->%d wheelNow=%.3f capture=%d\n",
                     msg, static_cast<long long>(r), queueBefore,
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

// SceneUBO / MaterialUBO / LightingUBO / ScenePush and the GBuffer formats are
// shared with the viewer renderer via renderer/deferred/DeferredCore.h.

// Compose pass push constants: column pixel size + text scale + text slot,
// the source-region window (normalized offset/size) and source dimensions.
struct ComposePush {
    float colSize[2];
    float textScale;
    float textSlot;
    float uvRect[4];  // normalized source region: offset xy, size zw
    float srcSize[2]; // source image pixels
    float nearest;    // != 0: sample nearest (magnification >= 1:1)
    float pad;
};
static_assert(sizeof(ComposePush) == 48, "ComposePush size mismatch");

// Metric compute push constants (two uvec4s in the shaders).
struct MetricPush {
    uint32_t x = 0, y = 0, z = 0, w = 0;      // region offset xy, extent zw
    uint32_t x2 = 0, y2 = 0, z2 = 0, w2 = 0;  // x2 = blocks per row (region);
                                              // y2 = 1 => ref is low-res (normalized sampling);
                                              // z2/w2 = test image full size (px)
};

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
    if (image) { vkDestroyImage(ctx.device, image, nullptr); image = VK_NULL_HANDLE; }
    if (memory) { vkFreeMemory(ctx.device, memory, nullptr); memory = VK_NULL_HANDLE; }
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
    if (const char* f = std::getenv("SR_GUI_INPUT_FILE")) inputFile_ = f;
    uiShot_ = std::getenv("SR_GUI_UI_SHOT") != nullptr;
    scenes_ = listScenes();
    upscalerNames_ = listUpscalers();
    applyLaunchOptions();

    active_.displayW = kOutputResolutions[ui_.outputResIndex][0];
    active_.displayH = kOutputResolutions[ui_.outputResIndex][1];

    if (!window_.create("sr_compare — gui", static_cast<int>(active_.displayW),
                        static_cast<int>(active_.displayH)))
        return false;
    window_.setClickToCaptureEnabled(false); // LMB belongs to the UI now
    if (!ctx_.create(window_)) return false;

    // Hold one Streamline reference for the whole GUI session: slInit already
    // ran inside ctx creation (allPluginsEnabled gate), and per-instance
    // addRef/release pairs would otherwise slShutdown/slInit on every Apply.
    if (sl_dlss::initialized()) {
        sl_dlss::addRef();
        slRefHeld_ = true;
    }

    refreshUpscalerAvailability();

    if (!swapchain_.create(ctx_, active_.displayW, active_.displayH, true)) return false;
    if (!createUiSync()) return false;
    if (!initImGui()) return false;
    window_.setWndProcHook(&guiWndProcHook);

    // Default selection: taa for the viewer, taa+fsr2 for compare (launch
    // options may have preset these already).
    for (uint32_t i = 0; i < upscalerNames_.size() && i < kMaxRegistered; ++i) {
        if (upscalerNames_[i] == "taa") {
            if (opts_.upscalerName.empty()) ui_.viewerUpscaler = static_cast<int>(i) + 1;
            if (opts_.compareList.empty()) ui_.compareSelected[i] = true;
            if (opts_.benchList.empty()) ui_.benchSelected[i] = true;
        }
        if (upscalerNames_[i] == "fsr2" && opts_.compareList.empty())
            ui_.compareSelected[i] = true;
    }

    active_ = configFromUi(currentTab_ == 1 ? Mode::Compare : Mode::Viewer);
    // IBL maps + deferred pipelines are built once per env map, not per Apply.
    envMapActive_ = active_.envMapPath;
    stackOk_ = deferred_.init(ctx_, envMapActive_.c_str());
    if (stackOk_) {
        stackOk_ = buildRenderStack();
    } else {
        statusLine_ = "deferred core init failed";
    }
    if (!stackOk_) {
        std::fprintf(stderr, "gui: initial render stack failed: %s\n", statusLine_.c_str());
        // Continue anyway: the UI stays up (ImGui-only frames) and shows the
        // error; Apply retries.
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

    if (!ImGui_ImplWin32_Init(window_.hwnd())) return false;

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
    window_.setWndProcHook(nullptr);
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplWin32_Shutdown();
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
    deferred_.destroy(ctx_);
    destroyUiSync();
    shutdownImGui();
    // Balance the session SL reference while the device is still alive.
    if (slRefHeld_) {
        sl_dlss::release();
        slRefHeld_ = false;
    }
    swapchain_.destroy(ctx_);
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
            cfg.algoNames.push_back(
                upscalerNames_[static_cast<size_t>(ui_.viewerUpscaler - 1)]);
        }
    } else {
        for (uint32_t i = 0; i < upscalerNames_.size() && i < kMaxRegistered; ++i) {
            if (ui_.compareSelected[i] && cfg.algoNames.size() < kMaxAlgos)
                cfg.algoNames.push_back(upscalerNames_[i]);
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
    std::snprintf(ui_.envMap, sizeof(ui_.envMap), "%s",
                  opts_.envMapPath.empty() ? kDefaultEnvMapPath : opts_.envMapPath.c_str());
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
        for (const std::string& name : splitComma(opts_.compareList)) {
            for (size_t i = 0; i < upscalerNames_.size() && i < kMaxRegistered; ++i) {
                if (upscalerNames_[i] == name) ui_.compareSelected[i] = true;
            }
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
    pendingRebuild_ = false;
    if (loadPhase_.load(std::memory_order_acquire) != LoadPhase::Idle) return;
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
        loadDone_.store(static_cast<uint32_t>(cfg.algoNames.size()), std::memory_order_relaxed);
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
    }
    loadResult_.algos.clear();
    loadResult_.error.clear();
    loadResult_.note.clear();
}

void GuiApp::finishAsyncRebuild() {
    if (loadThread_.joinable()) loadThread_.join();
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
    vkDeviceWaitIdle(ctx_.device);
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
        // when the path actually changed.
        if (active_.envMapPath != envMapActive_) {
            deferred_.destroy(ctx_);
            envMapActive_ = active_.envMapPath;
            if (!deferred_.init(ctx_, envMapActive_.c_str())) {
                statusLine_ = "deferred core init failed (env map: " + envMapActive_ + ")";
                stackOk_ = false;
                return;
            }
        }

        if (loadSceneDirty_) hasTransparency_ = deferred_.sceneHasTransparency(scene_);
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
    ImGui::SetNextWindowSize(ImVec2(380.f, 0.f), ImGuiCond_Always);
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
    if (!swapchain_.create(ctx_, active_.displayW, active_.displayH, true)) {
        statusLine_ = "swapchain creation failed";
        return false;
    }
    ensurePresentSemaphores();
    return true;
}

bool GuiApp::buildRenderStack() {
    if (!beginStackConfig()) return false;

    bool sceneOk = false;
    if (!active_.scenePath.empty()) sceneOk = scene_.loadGltf(ctx_, active_.scenePath.c_str());
    if (!sceneOk) sceneOk = scene_.loadProcedural(ctx_);
    if (!sceneOk || !ensureSceneFallbacks(scene_, ctx_, VK_NULL_HANDLE)) {
        statusLine_ = "scene load failed";
        return false;
    }
    hasTransparency_ = deferred_.sceneHasTransparency(scene_);

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

    resetFrameState();
    // Launch-option zoom (--compare-zoom): the first frames can trigger
    // spurious tab-switch rebuilds, so apply it inside the compare stack
    // build itself (consumed once).
    if (launchZoomPending_ && active_.mode == Mode::Compare) {
        compareZoom_ = std::clamp(opts_.compareZoom, 1.f, 16.f);
        launchZoomPending_ = false;
    }

    // glTF scenes (sponza & co.) are centered on the origin; the raw default
    // free-fly pose sits inside their outer wall and shows only black.  Start
    // from the CLI automation orbit's first keyframe instead.
    if (!active_.scenePath.empty()) {
        camera_.position = {6.5f, 2.f, 0.f};
        camera_.up = {0.f, 1.f, 0.f};
        camera_.lookAt({0.f, 2.f, 0.f});
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
    jitterX_ = jitterY_ = prevJitterX_ = prevJitterY_ = 0.f;
    metricPending_[0] = metricPending_[1] = false;
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
    const uint32_t count = static_cast<uint32_t>(cfg.algoNames.size());
    for (uint32_t i = 0; i < count; ++i) {
        const std::string& name = cfg.algoNames[i];
        if (onAlgo) onAlgo(name.c_str(), i, count);
        std::unique_ptr<IUpscaler> up = createUpscaler(name.c_str());
        if (!up) {
            err = "unknown upscaler: " + name;
            continue;
        }
        if (!up->isAvailable(env)) {
            err = name + " is not available on this device";
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
            err = name + " init failed";
            continue;
        }
        AlgoColumn col;
        col.id = name;
        col.upscaler = std::move(up);
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
    if (!createRT(gbDepth_, renderWidth_, renderHeight_, deferred::kDepthFormat,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_DEPTH_BIT))
        return false;

    // SSAO targets (raw + blurred) for the low-res path.
    const VkImageUsageFlags aoUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (!createRT(gbAoRaw_, renderWidth_, renderHeight_, VK_FORMAT_R16_SFLOAT, aoUsage,
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
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
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
    if (!createRT(gtDepth_, gtW, gtH, deferred::kDepthFormat,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_DEPTH_BIT))
        return false;
    if (!createRT(gtAoRaw_, gtW, gtH, VK_FORMAT_R16_SFLOAT, aoUsage, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtAo_, gtW, gtH, VK_FORMAT_R16_SFLOAT, aoUsage, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;

    // 200% SSAA ground truth: deferred render at 2x, downsample into gtColor_.
    if (active_.gtSsaa) {
        if (!createRT(gtSsaaColor_, dw * 2, dh * 2, deferred::kHdrColorFormat,
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
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
        if (!createRT(gtSsaaDepth_, dw * 2, dh * 2, deferred::kDepthFormat,
                      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                      VK_IMAGE_ASPECT_DEPTH_BIT))
            return false;
        if (!createRT(gtSsaaAoRaw_, dw * 2, dh * 2, VK_FORMAT_R16_SFLOAT, aoUsage,
                      VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
        if (!createRT(gtSsaaAo_, dw * 2, dh * 2, VK_FORMAT_R16_SFLOAT, aoUsage,
                      VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
    }

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
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    const VkDeviceSize size = kFontAtlasW * kFontAtlasH;
    if (createBuffer(ctx_, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging, stagingMemory) != VK_SUCCESS)
        return false;
    void* mapped = nullptr;
    vkMapMemory(ctx_.device, stagingMemory, 0, size, 0, &mapped);
    buildFontAtlas(static_cast<uint8_t*>(mapped));
    vkUnmapMemory(ctx_.device, stagingMemory);

    if (createImage(ctx_, kFontAtlasW, kFontAtlasH, VK_FORMAT_R8_UNORM,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    fontAtlas_.image, fontAtlas_.memory) != VK_SUCCESS) {
        vkDestroyBuffer(ctx_.device, staging, nullptr);
        vkFreeMemory(ctx_.device, stagingMemory, nullptr);
        return false;
    }
    fontAtlas_.width = kFontAtlasW;
    fontAtlas_.height = kFontAtlasH;
    fontAtlas_.format = VK_FORMAT_R8_UNORM;

    submitOneShot(ctx_, [&](VkCommandBuffer cmd) {
        copyBufferToImage(cmd, staging, fontAtlas_.image, kFontAtlasW, kFontAtlasH,
                          VK_FORMAT_R8_UNORM);
    });
    vkDestroyBuffer(ctx_.device, staging, nullptr);
    vkFreeMemory(ctx_.device, stagingMemory, nullptr);

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
        vkMapMemory(ctx_.device, metricStagingMemory_[i], 0, resultSize, 0,
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
    VkDescriptorSetLayoutBinding composeBindings[3] = {};
    for (uint32_t i = 0; i < 3; ++i) {
        composeBindings[i].binding = i;
        composeBindings[i].descriptorCount = 1;
        composeBindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        composeBindings[i].descriptorType =
            i == 1 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    }
    VkDescriptorSetLayoutCreateInfo composeLayoutCi = {};
    composeLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    composeLayoutCi.bindingCount = 3;
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
    // Lighting sets: 3 per frame (GBuffer / GT / GT-SSAA), 10 samplers + 1 UBO
    // each; transparent sets: 3 per frame (GB/GT/SSAA), 4 samplers + 1 UBO each;
    // SSAO sets: static, 3 samplers + 2 storage images per path.
    VkDescriptorPoolSize sizes[5] = {};
    sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount =
        deferred::kMaxTextures + numColumns * 2 + 2 + numAlgos * 2 + 10 * kFramesInFlight * 3 +
        4 * kFramesInFlight * 3 + 3 * 3;
    sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[1].descriptorCount = kFramesInFlight * 2 + numColumns + kFramesInFlight * 3 +
                               kFramesInFlight * 3;
    sizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    sizes[2].descriptorCount = kFramesInFlight * 2;
    sizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[3].descriptorCount = numAlgos * 2;
    sizes[4].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    sizes[4].descriptorCount = 6; // ssao raw + blurred outputs (GB/GT/SSAA)
    VkDescriptorPoolCreateInfo poolCi = {};
    poolCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolCi.maxSets = kFramesInFlight * 2 + 2 + numColumns + 1 + numAlgos + kFramesInFlight * 3 +
                     kFramesInFlight * 3 + // transparent sets (GB/GT/SSAA)
                     6;                     // ssao sets (GB/GT/SSAA, static)
    poolCi.poolSizeCount = 5;
    poolCi.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(ctx_.device, &poolCi, nullptr, &descriptorPool_) != VK_SUCCESS)
        return false;

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
    vkMapMemory(ctx_.device, textUboMemory_, 0, textSize, 0, &textUboMapped_);

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
        if (!allocSet(deferred_.sceneSetLayout(), frames_[i].sceneSetGt)) return false;
        if (!allocSet(deferred_.lightingSetLayout(), frames_[i].lightingSetGb)) return false;
        if (!allocSet(deferred_.lightingSetLayout(), frames_[i].lightingSetGt)) return false;
        if (!allocSet(deferred_.lightingSetLayout(), frames_[i].lightingSetSsaa)) return false;
        if (!allocSet(deferred_.transparentSetLayout(), frames_[i].transparentSetGb)) return false;
        if (!allocSet(deferred_.transparentSetLayout(), frames_[i].transparentSetGt)) return false;
        if (!allocSet(deferred_.transparentSetLayout(), frames_[i].transparentSetSsaa))
            return false;
    }
    if (!allocSet(deferred_.textureSetLayout(), textureSet_)) return false;
    if (!allocSet(deferred_.ssaoSetLayout(), ssaoSetGb_)) return false;
    if (!allocSet(deferred_.ssaoBlurSetLayout(), ssaoBlurSetGb_)) return false;
    if (!allocSet(deferred_.ssaoSetLayout(), ssaoSetGt_)) return false;
    if (!allocSet(deferred_.ssaoBlurSetLayout(), ssaoBlurSetGt_)) return false;
    if (active_.gtSsaa) {
        if (!allocSet(deferred_.ssaoSetLayout(), ssaoSetSsaa_)) return false;
        if (!allocSet(deferred_.ssaoBlurSetLayout(), ssaoBlurSetSsaa_)) return false;
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

    VkWriteDescriptorSet writes[3] = {};
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
    vkUpdateDescriptorSets(ctx_.device, 3, writes, 0, nullptr);
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

    if (active_.mode == Mode::Compare) {
        VkDescriptorImageInfo testInfo = {};
        testInfo.sampler = linearSampler_;
        testInfo.imageView = algo.output.view;
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

    VkPipelineLayoutCreateInfo copyLayoutCi = {};
    copyLayoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    copyLayoutCi.setLayoutCount = 1;
    copyLayoutCi.pSetLayouts = &copySetLayout_;
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
    if (vkCreateGraphicsPipelines(ctx_.device, VK_NULL_HANDLE, 1, &fsCi, nullptr, &composePipeline_) != VK_SUCCESS)
        return false;

    fsStages[1].module = copyFrag_;
    VkPipelineRenderingCreateInfo copyRendering = {};
    copyRendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    copyRendering.colorAttachmentCount = 1;
    const VkFormat presentFormat = swapchain_.format();
    copyRendering.pColorAttachmentFormats = &presentFormat;
    fsCi.pNext = &copyRendering;
    fsCi.layout = copyPipelineLayout_;
    if (vkCreateGraphicsPipelines(ctx_.device, VK_NULL_HANDLE, 1, &fsCi, nullptr, &copyPipeline_) != VK_SUCCESS)
        return false;

    // GT SSAA downsample: same passthrough fragment shader, HDR target.
    VkPipelineRenderingCreateInfo downsampleRendering = {};
    downsampleRendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    downsampleRendering.colorAttachmentCount = 1;
    downsampleRendering.pColorAttachmentFormats = &deferred::kHdrColorFormat;
    fsCi.pNext = &downsampleRendering;
    if (vkCreateGraphicsPipelines(ctx_.device, VK_NULL_HANDLE, 1, &fsCi, nullptr,
                                  &downsamplePipeline_) != VK_SUCCESS)
        return false;

    // --- metric compute pipelines --------------------------------------------------
    VkComputePipelineCreateInfo compCi = {};
    compCi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compCi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compCi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compCi.stage.module = metricBlocksComp_;
    compCi.stage.pName = "main";
    compCi.layout = metricPipelineLayout_;
    if (vkCreateComputePipelines(ctx_.device, VK_NULL_HANDLE, 1, &compCi, nullptr,
                                 &metricBlocksPipeline_) != VK_SUCCESS)
        return false;
    compCi.stage.module = metricReduceComp_;
    if (vkCreateComputePipelines(ctx_.device, VK_NULL_HANDLE, 1, &compCi, nullptr,
                                 &metricReducePipeline_) != VK_SUCCESS)
        return false;

    return true;
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

        VkBuffer ubos[2] = {};
        VkDeviceMemory uboMems[2] = {};
        void* uboMaps[2] = {};
        VkDescriptorSet sets[2] = {fr.sceneSetGb, fr.sceneSetGt};
        for (int k = 0; k < 2; ++k) {
            if (createBuffer(ctx_, uboSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             ubos[k], uboMems[k]) != VK_SUCCESS)
                return false;
            vkMapMemory(ctx_.device, uboMems[k], 0, uboSize, 0, &uboMaps[k]);

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
        }
        fr.uboGb = ubos[0]; fr.uboGbMemory = uboMems[0]; fr.uboGbMapped = uboMaps[0];
        fr.uboGt = ubos[1]; fr.uboGtMemory = uboMems[1]; fr.uboGtMapped = uboMaps[1];

        // Lighting UBOs (GB jittered / GT+SSAA un-jittered) + lighting sets.
        const VkDeviceSize lightingSize = sizeof(LightingUBO);
        if (createBuffer(ctx_, lightingSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         fr.lightingUboGb, fr.lightingUboGbMemory) != VK_SUCCESS)
            return false;
        vkMapMemory(ctx_.device, fr.lightingUboGbMemory, 0, lightingSize, 0,
                    &fr.lightingUboGbMapped);
        if (createBuffer(ctx_, lightingSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         fr.lightingUboGt, fr.lightingUboGtMemory) != VK_SUCCESS)
            return false;
        vkMapMemory(ctx_.device, fr.lightingUboGtMemory, 0, lightingSize, 0,
                    &fr.lightingUboGtMapped);

        deferred_.writeLightingSet(ctx_, fr.lightingSetGb, fr.lightingUboGb, gbAlbedo_.view,
                                   gbNormal_.view, gbMaterial_.view, gbEmissive_.view,
                                   gbDepth_.view, gbAo_.view);
        deferred_.writeLightingSet(ctx_, fr.lightingSetGt, fr.lightingUboGt, gtAlbedo_.view,
                                   gtNormal_.view, gtMaterial_.view, gtEmissive_.view,
                                   gtDepth_.view, gtAo_.view);
        if (active_.gtSsaa) {
            // GT and GT-SSAA share the same (resolution-independent) UBO.
            deferred_.writeLightingSet(ctx_, fr.lightingSetSsaa, fr.lightingUboGt,
                                       gtSsaaAlbedo_.view, gtSsaaNormal_.view,
                                       gtSsaaMaterial_.view, gtSsaaEmissive_.view,
                                       gtSsaaDepth_.view, gtSsaaAo_.view);
        }

        // The transparency shader reads iblParams (identical in both lighting
        // UBOs) plus the path's own SSAO texture: one set per path.
        deferred_.writeTransparentSet(ctx_, fr.transparentSetGb, fr.lightingUboGb, gbAo_.view);
        deferred_.writeTransparentSet(ctx_, fr.transparentSetGt, fr.lightingUboGb, gtAo_.view);
        if (active_.gtSsaa) {
            deferred_.writeTransparentSet(ctx_, fr.transparentSetSsaa, fr.lightingUboGb,
                                          gtSsaaAo_.view);
        }
    }

    // SSAO sets (static bindings; per-frame data goes through push constants).
    deferred_.writeSsaoSet(ctx_, ssaoSetGb_, gbDepth_.view, gbNormal_.view, gbAoRaw_.view);
    deferred_.writeSsaoBlurSet(ctx_, ssaoBlurSetGb_, gbAoRaw_.view, gbAo_.view);
    deferred_.writeSsaoSet(ctx_, ssaoSetGt_, gtDepth_.view, gtNormal_.view, gtAoRaw_.view);
    deferred_.writeSsaoBlurSet(ctx_, ssaoBlurSetGt_, gtAoRaw_.view, gtAo_.view);
    if (active_.gtSsaa) {
        deferred_.writeSsaoSet(ctx_, ssaoSetSsaa_, gtSsaaDepth_.view, gtSsaaNormal_.view,
                               gtSsaaAoRaw_.view);
        deferred_.writeSsaoBlurSet(ctx_, ssaoBlurSetSsaa_, gtSsaaAoRaw_.view, gtSsaaAo_.view);
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
    vkMapMemory(ctx_.device, screenshotStagingMemory_, 0, size, 0, &screenshotMapped_);
    return true;
}

void GuiApp::destroyAlgoResources() {
    for (AlgoColumn& algo : algos_) {
        if (algo.upscaler) { algo.upscaler->shutdown(); algo.upscaler.reset(); }
        algo.output.destroy(ctx_);
        if (algo.blocksBuffer) {
            vkDestroyBuffer(ctx_.device, algo.blocksBuffer, nullptr);
            vkFreeMemory(ctx_.device, algo.blocksMemory, nullptr);
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

    destroyAlgoResources();

    if (screenshotStaging_) {
        vkDestroyBuffer(ctx_.device, screenshotStaging_, nullptr);
        vkFreeMemory(ctx_.device, screenshotStagingMemory_, nullptr);
        screenshotStaging_ = VK_NULL_HANDLE;
        screenshotStagingMemory_ = VK_NULL_HANDLE;
        screenshotMapped_ = nullptr;
    }
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        if (metricStaging_[i]) {
            vkDestroyBuffer(ctx_.device, metricStaging_[i], nullptr);
            vkFreeMemory(ctx_.device, metricStagingMemory_[i], nullptr);
            metricStaging_[i] = VK_NULL_HANDLE;
            metricStagingMemory_[i] = VK_NULL_HANDLE;
            metricStagingMapped_[i] = nullptr;
        }
        metricPending_[i] = false;
    }
    if (metricResultBuf_) {
        vkDestroyBuffer(ctx_.device, metricResultBuf_, nullptr);
        vkFreeMemory(ctx_.device, metricResultMemory_, nullptr);
        metricResultBuf_ = VK_NULL_HANDLE;
        metricResultMemory_ = VK_NULL_HANDLE;
    }
    if (textUbo_) {
        vkDestroyBuffer(ctx_.device, textUbo_, nullptr);
        vkFreeMemory(ctx_.device, textUboMemory_, nullptr);
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
            vkDestroyBuffer(ctx_.device, fr.uboGb, nullptr);
            vkFreeMemory(ctx_.device, fr.uboGbMemory, nullptr);
            fr.uboGb = VK_NULL_HANDLE;
            fr.uboGbMemory = VK_NULL_HANDLE;
            fr.uboGbMapped = nullptr;
        }
        if (fr.uboGt) {
            vkDestroyBuffer(ctx_.device, fr.uboGt, nullptr);
            vkFreeMemory(ctx_.device, fr.uboGtMemory, nullptr);
            fr.uboGt = VK_NULL_HANDLE;
            fr.uboGtMemory = VK_NULL_HANDLE;
            fr.uboGtMapped = nullptr;
        }
        if (fr.lightingUboGb) {
            vkDestroyBuffer(ctx_.device, fr.lightingUboGb, nullptr);
            vkFreeMemory(ctx_.device, fr.lightingUboGbMemory, nullptr);
            fr.lightingUboGb = VK_NULL_HANDLE;
            fr.lightingUboGbMemory = VK_NULL_HANDLE;
            fr.lightingUboGbMapped = nullptr;
        }
        if (fr.lightingUboGt) {
            vkDestroyBuffer(ctx_.device, fr.lightingUboGt, nullptr);
            vkFreeMemory(ctx_.device, fr.lightingUboGtMemory, nullptr);
            fr.lightingUboGt = VK_NULL_HANDLE;
            fr.lightingUboGtMemory = VK_NULL_HANDLE;
            fr.lightingUboGtMapped = nullptr;
        }
        fr.sceneSetGb = VK_NULL_HANDLE;
        fr.sceneSetGt = VK_NULL_HANDLE;
        fr.lightingSetGb = VK_NULL_HANDLE;
        fr.lightingSetGt = VK_NULL_HANDLE;
        fr.lightingSetSsaa = VK_NULL_HANDLE;
        fr.transparentSetGb = VK_NULL_HANDLE;
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
    ssaoBlurSetGb_ = VK_NULL_HANDLE;
    ssaoSetGt_ = VK_NULL_HANDLE;
    ssaoBlurSetGt_ = VK_NULL_HANDLE;
    ssaoSetSsaa_ = VK_NULL_HANDLE;
    ssaoBlurSetSsaa_ = VK_NULL_HANDLE;

    if (materialUbo_) {
        vkDestroyBuffer(ctx_.device, materialUbo_, nullptr);
        vkFreeMemory(ctx_.device, materialUboMemory_, nullptr);
        materialUbo_ = VK_NULL_HANDLE;
        materialUboMemory_ = VK_NULL_HANDLE;
    }

    gbColor_.destroy(ctx_);
    gbAlbedo_.destroy(ctx_);
    gbNormal_.destroy(ctx_);
    gbMaterial_.destroy(ctx_);
    gbEmissive_.destroy(ctx_);
    gbMotion_.destroy(ctx_);
    gbReactive_.destroy(ctx_);
    gbDepth_.destroy(ctx_);
    gtColor_.destroy(ctx_);
    gtAlbedo_.destroy(ctx_);
    gtNormal_.destroy(ctx_);
    gtMaterial_.destroy(ctx_);
    gtEmissive_.destroy(ctx_);
    gtDepth_.destroy(ctx_);
    gtSsaaColor_.destroy(ctx_);
    gtSsaaAlbedo_.destroy(ctx_);
    gtSsaaNormal_.destroy(ctx_);
    gtSsaaMaterial_.destroy(ctx_);
    gtSsaaEmissive_.destroy(ctx_);
    gtSsaaDepth_.destroy(ctx_);
    gbAoRaw_.destroy(ctx_);
    gbAo_.destroy(ctx_);
    gtAoRaw_.destroy(ctx_);
    gtAo_.destroy(ctx_);
    gtSsaaAoRaw_.destroy(ctx_);
    gtSsaaAo_.destroy(ctx_);
    composeImage_.destroy(ctx_);
    uiShotImage_.destroy(ctx_);
    uiShotLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    fontAtlas_.destroy(ctx_);

    gbColorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gbAlbedoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gbNormalLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gbMaterialLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gbEmissiveLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gbMotionLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    gbReactiveLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
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
    imageBarrier(cmd, swapImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
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
                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
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

void GuiApp::updateLightingUBO(void* mapped, const Mat4& invViewProj) {
    LightingUBO ubo;
    deferred_.fillLightingUBO(ubo, scene_, camera_, invViewProj);
    std::memcpy(mapped, &ubo, sizeof(ubo));
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
    static const bool dbgInput = std::getenv("SR_GUI_DEBUG_INPUT") != nullptr;
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
            ? (io.MousePos.x >= 0.f && io.MousePos.x < 56.f && io.MousePos.y >= 0.f &&
               io.MousePos.y < 34.f)
            : (io.MousePos.x >= 0.f && io.MousePos.x < kPanelWidth && io.MousePos.y >= 0.f);
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

void GuiApp::updateCamera(float dt) {    if (pathPlaying_ && !path_.empty()) {
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
        in.keys[VK_SHIFT] = ImGui::IsKeyDown(ImGuiKey_LeftShift) || ImGui::IsKeyDown(ImGuiKey_RightShift);
        in.keys[VK_SPACE] = ImGui::IsKeyDown(ImGuiKey_Space);
        in.keys[VK_CONTROL] = ImGui::IsKeyDown(ImGuiKey_LeftCtrl) || ImGui::IsKeyDown(ImGuiKey_RightCtrl);
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
    imageBarrier(cmd, composeImage_.image, composeLayout_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy region = {};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {active_.displayW, active_.displayH, 1};
    vkCmdCopyImageToBuffer(cmd, composeImage_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           screenshotStaging_, 1, &region);
    imageBarrier(cmd, composeImage_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    composeLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void GuiApp::harvestMetrics(uint32_t slot) {
    const auto* f = static_cast<const float*>(metricStagingMapped_[slot]);
    const uint32_t numAlgos = static_cast<uint32_t>(algos_.size());
    for (uint32_t i = 0; i < numAlgos; ++i) {
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

    char line[25];
    if (active_.mode == Mode::Viewer) {
        if (algos_.empty()) {
            writeLine(0, 0, "Native (GT)");
        } else {
            writeLine(0, 0, algos_[0].upscaler->name());
            std::snprintf(line, sizeof(line), "SCENE %.2f MS", lastTimings_.sceneMs);
            writeLine(0, 1, line);
            std::snprintf(line, sizeof(line), "UPSCALE %.2f MS", lastTimings_.upscaleMs);
            writeLine(0, 2, line);
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
    }
    for (uint32_t i = 0; i < algos_.size(); ++i) {
        const AlgoColumn& algo = algos_[i];
        const uint32_t slot = i + 1;
        writeLine(slot, 0, algo.upscaler->name());
        if (algo.hasMetric) {
            std::snprintf(line, sizeof(line), "PSNR %.2f dB", static_cast<double>(algo.psnr));
            writeLine(slot, 1, line);
            std::snprintf(line, sizeof(line), "SSIM %.4f", static_cast<double>(algo.ssim));
            writeLine(slot, 2, line);
        } else {
            writeLine(slot, 1, "PSNR --");
            writeLine(slot, 2, "SSIM --");
        }
    }
}

void GuiApp::recordFrame(uint32_t frameIndex, uint32_t swapchainIndex) {
    const uint32_t slot = frameIndex % kFramesInFlight;
    FrameResources& fr = frames_[slot];
    VkCommandBuffer cmd = fr.cmd;

    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    timestamps_.resetForFrame(cmd, slot);
    timestamps_.frameBegin(cmd, slot);

    const uint32_t dw = active_.displayW;
    const uint32_t dh = active_.displayH;
    const bool compareMode = active_.mode == Mode::Compare;
    const bool gbuffer = !algos_.empty();
    const float aspect = static_cast<float>(dw) / static_cast<float>(dh);
    const Mat4 view = camera_.view();
    const Mat4 proj = camera_.proj(aspect);
    Mat4 projJittered = proj;
    prevJitterX_ = jitterX_;
    prevJitterY_ = jitterY_;
    const Vec2 h = halton23(frameIndex + 1);
    jitterX_ = gbuffer ? (h.x - 0.5f) : 0.f;
    jitterY_ = gbuffer ? (h.y - 0.5f) : 0.f;
    projJittered.m[12] += jitterX_ * 2.f / static_cast<float>(renderWidth_);
    projJittered.m[13] += jitterY_ * 2.f / static_cast<float>(renderHeight_);

    updateSceneUBO(fr.uboGbMapped, true, renderWidth_, renderHeight_, view, proj, projJittered,
                   prevViewProj_);
    const uint32_t gtW = active_.gtSsaa ? dw * 2 : (active_.gtApplyScale ? renderWidth_ : dw);
    const uint32_t gtH = active_.gtSsaa ? dh * 2 : (active_.gtApplyScale ? renderHeight_ : dh);
    updateSceneUBO(fr.uboGtMapped, false, gtW, gtH, view, proj, proj, prevViewProj_);
    updateLightingUBO(fr.lightingUboGbMapped,
                      Mat4::inverse(Mat4::multiply(projJittered, view)));
    updateLightingUBO(fr.lightingUboGtMapped, Mat4::inverse(Mat4::multiply(proj, view)));

    auto transition = [&](VkImage image, VkImageLayout& current, VkImageLayout target,
                          VkImageAspectFlags aspect_) {
        imageBarrier(cmd, image, current, target, aspect_);
        current = target;
    };
    const Mat4 cullViewProj = Mat4::multiply(proj, view); // un-jittered (sub-pixel)

    // --- 1) shared low-resolution GBuffer pass (jittered) ----------------------
    timestamps_.sceneBegin(cmd, slot);
    if (gbuffer) {
        transition(gbAlbedo_.image, gbAlbedoLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbNormal_.image, gbNormalLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbMaterial_.image, gbMaterialLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbEmissive_.image, gbEmissiveLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbMotion_.image, gbMotionLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbDepth_.image, gbDepthLayout_, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
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
            deferred_.recordGBufferDraws(cmd, scene_, false, fr.sceneSetGb, textureSet_,
                                         materialStride_, renderWidth_, renderHeight_,
                                         cullViewProj);
            vkCmdEndRendering(cmd);
        }
        transition(gbAlbedo_.image, gbAlbedoLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbNormal_.image, gbNormalLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbMaterial_.image, gbMaterialLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbEmissive_.image, gbEmissiveLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbMotion_.image, gbMotionLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbDepth_.image, gbDepthLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_IMAGE_ASPECT_DEPTH_BIT);

        // SSAO (depth + normal -> raw AO -> cross-box blur), jittered LR view-proj.
        transition(gbAoRaw_.image, gbAoRawLayout_, VK_IMAGE_LAYOUT_GENERAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordSsaoPass(cmd, ssaoSetGb_, Mat4::multiply(projJittered, view), frameIndex,
                                 renderWidth_, renderHeight_);
        transition(gbAoRaw_.image, gbAoRawLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbAo_.image, gbAoLayout_, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordSsaoBlurPass(cmd, ssaoBlurSetGb_, renderWidth_, renderHeight_);
        transition(gbAo_.image, gbAoLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);

        // Deferred lighting (PBR direct + IBL, skybox on far-plane pixels) -> gbColor_.
        transition(gbColor_.image, gbColorLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordLightingPass(cmd, fr.lightingSetGb, gbColor_.view, renderWidth_,
                                     renderHeight_);

        // --- Transparency pass (alpha-blended surfaces over the lit scene) -----
        // Overwrites motion (static glass = camera motion, the "Output Velocity"
        // equivalent) and accumulates the translucent coverage mask consumed by
        // the upscalers as the reactive / TC / bias mask.
        if (hasTransparency_) {
            transition(gbMotion_.image, gbMotionLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                       VK_IMAGE_ASPECT_COLOR_BIT);
            transition(gbReactive_.image, gbReactiveLayout_,
                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
            transition(gbDepth_.image, gbDepthLayout_,
                       VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
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
                deferred_.recordTransparentDraws(cmd, scene_, false, fr.sceneSetGb, textureSet_,
                                                 fr.transparentSetGb, materialStride_,
                                                 renderWidth_, renderHeight_, cullViewProj,
                                                 camera_.position);
                vkCmdEndRendering(cmd);
            }
            transition(gbMotion_.image, gbMotionLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_IMAGE_ASPECT_COLOR_BIT);
            transition(gbReactive_.image, gbReactiveLayout_,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
            transition(gbDepth_.image, gbDepthLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_IMAGE_ASPECT_DEPTH_BIT);
        }

        transition(gbColor_.image, gbColorLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
    }

    // --- 2) per-algorithm dispatch ---------------------------------------------
    if (gbuffer) {
        timestamps_.upscaleBegin(cmd, slot);

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

        FrameParams frame;
        frame.frameIndex = static_cast<int>(frameIndex);
        frame.deltaTime = 1.f / 60.f;
        frame.preExposure = 1.f;
        frame.resetHistory = (frameIndex == 0);

        for (AlgoColumn& algo : algos_) {
            transition(algo.output.image, algo.outputLayout, VK_IMAGE_LAYOUT_GENERAL,
                       VK_IMAGE_ASPECT_COLOR_BIT);

            UpscalerResources res;
            res.color = gbColor_.image;
            res.colorView = gbColor_.view;
            res.depth = gbDepth_.image;
            res.depthView = gbDepth_.view;
            res.motion = gbMotion_.image;
            res.motionView = gbMotion_.view;
            if (hasTransparency_) {
                res.reactive = gbReactive_.image;
                res.reactiveView = gbReactive_.view;
            }
            res.output = algo.output.image;
            res.outputView = algo.output.view;
            algo.upscaler->dispatch(cmd, res, cam, frame);

            transition(algo.output.image, algo.outputLayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_IMAGE_ASPECT_COLOR_BIT);
        }
        timestamps_.upscaleEnd(cmd, slot);
    } else {
        // No upscaler pass: emit a zero-width range so every query slot is
        // written (readback with WAIT_BIT otherwise blocks forever).
        timestamps_.upscaleBegin(cmd, slot);
        timestamps_.upscaleEnd(cmd, slot);
    }

    // --- 3) native-resolution ground truth (no jitter) ---------------------------
    if (gtActive_ && active_.gtSsaa) {
        // 200% SSAA: deferred render at 2x into gtSsaaColor_, then
        // box-downsample to display resolution (gtColor_), which stays the
        // metric/display ref.
        const uint32_t sw = dw * 2;
        const uint32_t sh = dh * 2;
        transition(gtSsaaAlbedo_.image, gtSsaaAlbedoLayout_,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaNormal_.image, gtSsaaNormalLayout_,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaMaterial_.image, gtSsaaMaterialLayout_,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaEmissive_.image, gtSsaaEmissiveLayout_,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaDepth_.image, gtSsaaDepthLayout_,
                   VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
        {
            VkRenderingAttachmentInfo colors[4] = {
                makeColorAttachment(gtSsaaAlbedo_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gtSsaaNormal_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR, 0.f, 0.f, 1.f),
                makeColorAttachment(gtSsaaMaterial_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gtSsaaEmissive_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR)};
            VkRenderingAttachmentInfo depth =
                makeDepthAttachment(gtSsaaDepth_.view,
                                    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR);
            beginRendering(cmd, sw, sh, 4, colors, &depth);
            deferred_.recordGBufferDraws(cmd, scene_, true, fr.sceneSetGt, textureSet_,
                                         materialStride_, sw, sh, cullViewProj);
            vkCmdEndRendering(cmd);
        }
        transition(gtSsaaAlbedo_.image, gtSsaaAlbedoLayout_,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaNormal_.image, gtSsaaNormalLayout_,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaMaterial_.image, gtSsaaMaterialLayout_,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaEmissive_.image, gtSsaaEmissiveLayout_,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaDepth_.image, gtSsaaDepthLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_IMAGE_ASPECT_DEPTH_BIT);

        // SSAO for the 2x GT path (un-jittered view-projection).
        transition(gtSsaaAoRaw_.image, gtSsaaAoRawLayout_, VK_IMAGE_LAYOUT_GENERAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordSsaoPass(cmd, ssaoSetSsaa_, cullViewProj, frameIndex, sw, sh);
        transition(gtSsaaAoRaw_.image, gtSsaaAoRawLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaAo_.image, gtSsaaAoLayout_, VK_IMAGE_LAYOUT_GENERAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordSsaoBlurPass(cmd, ssaoBlurSetSsaa_, sw, sh);
        transition(gtSsaaAo_.image, gtSsaaAoLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);

        transition(gtSsaaColor_.image, gtSsaaColorLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordLightingPass(cmd, fr.lightingSetSsaa, gtSsaaColor_.view, sw, sh);

        // Transparency pass: alpha-blended surfaces over the lit scene (GT
        // path: color only, no motion/mask outputs).
        if (hasTransparency_) {
            transition(gtSsaaDepth_.image, gtSsaaDepthLayout_,
                       VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
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
                                                 cullViewProj, camera_.position);
                vkCmdEndRendering(cmd);
            }
            transition(gtSsaaDepth_.image, gtSsaaDepthLayout_,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
        }

        transition(gtSsaaColor_.image, gtSsaaColorLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);

        transition(gtColor_.image, gtColorLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
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
            vkCmdDraw(cmd, 3, 1, 0, 0);
            vkCmdEndRendering(cmd);
        }
        transition(gtColor_.image, gtColorLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
    } else if (gtActive_) {
        transition(gtAlbedo_.image, gtAlbedoLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtNormal_.image, gtNormalLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtMaterial_.image, gtMaterialLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtEmissive_.image, gtEmissiveLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtDepth_.image, gtDepthLayout_, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_ASPECT_DEPTH_BIT);
        {
            VkRenderingAttachmentInfo colors[4] = {
                makeColorAttachment(gtAlbedo_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gtNormal_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR, 0.f, 0.f, 1.f),
                makeColorAttachment(gtMaterial_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gtEmissive_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR)};
            VkRenderingAttachmentInfo depth =
                makeDepthAttachment(gtDepth_.view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR);
            beginRendering(cmd, gtW, gtH, 4, colors, &depth);
            deferred_.recordGBufferDraws(cmd, scene_, true, fr.sceneSetGt, textureSet_,
                                         materialStride_, gtW, gtH, cullViewProj);
            vkCmdEndRendering(cmd);
        }
        transition(gtAlbedo_.image, gtAlbedoLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtNormal_.image, gtNormalLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtMaterial_.image, gtMaterialLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtEmissive_.image, gtEmissiveLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtDepth_.image, gtDepthLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_IMAGE_ASPECT_DEPTH_BIT);

        // SSAO for the 1x GT path (un-jittered view-projection).  In
        // "GT (Apply scale)" mode gtW/gtH are the low input resolution.
        transition(gtAoRaw_.image, gtAoRawLayout_, VK_IMAGE_LAYOUT_GENERAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordSsaoPass(cmd, ssaoSetGt_, cullViewProj, frameIndex, gtW, gtH);
        transition(gtAoRaw_.image, gtAoRawLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtAo_.image, gtAoLayout_, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordSsaoBlurPass(cmd, ssaoBlurSetGt_, gtW, gtH);
        transition(gtAo_.image, gtAoLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);

        transition(gtColor_.image, gtColorLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordLightingPass(cmd, fr.lightingSetGt, gtColor_.view, gtW, gtH);

        // Transparency pass: alpha-blended surfaces over the lit scene (GT
        // path: color only, no motion/mask outputs).
        if (hasTransparency_) {
            transition(gtDepth_.image, gtDepthLayout_,
                       VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
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
                                                 cullViewProj, camera_.position);
                vkCmdEndRendering(cmd);
            }
            transition(gtDepth_.image, gtDepthLayout_,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
        }

        transition(gtColor_.image, gtColorLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_IMAGE_ASPECT_COLOR_BIT);
    }
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
        for (AlgoColumn& algo : algos_) {
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
        computeBarrier(cmd, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferCopy copyRegion = {};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = static_cast<VkDeviceSize>(algos_.size()) * kMetricFloats * 4;
        vkCmdCopyBuffer(cmd, metricResultBuf_, metricStaging_[slot], 1, &copyRegion);
        metricPending_[slot] = true;
    }

    // --- 5) compose columns + overlay into the offscreen composite -----------------
    transition(composeImage_.image, composeLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
               VK_IMAGE_ASPECT_COLOR_BIT);
    {
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
                set = (i == 0) ? gtComposeSet_ : algos_[i - 1].composeSet;
            } else if (!algos_.empty()) {
                set = algos_[0].composeSet;
            }
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, composePipelineLayout_,
                                    0, 1, &set, 0, nullptr);
            ComposePush push;
            push.colSize[0] = static_cast<float>(w);
            push.colSize[1] = static_cast<float>(dh);
            push.textScale = textScale;
            push.textSlot = static_cast<float>(i);
            // Aspect-preserving crop + zoom window over the source.  The GT
            // column's source (gtColor_) is low-res in "GT (Apply scale)"
            // mode; the region window is normalized so it stretches to fill.
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
            // Nearest sampling once the on-screen magnification passes 1:1
            // (measured in source texels, so a low-res GT goes nearest only
            // when its own pixels are magnified past 1:1).
            const float srcRegionW = rect[2] * (srcW / static_cast<float>(dw));
            push.nearest = (static_cast<float>(w) >= srcRegionW) ? 1.f : 0.f;
            push.pad = 0.f;
            vkCmdPushConstants(cmd, composePipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof(push), &push);
            vkCmdDraw(cmd, 3, 1, 0, 0);
        }
        vkCmdEndRendering(cmd);
    }
    transition(composeImage_.image, composeLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_IMAGE_ASPECT_COLOR_BIT);

    // Screenshot capture (composite; with SR_GUI_UI_SHOT also the ImGui UI).
    // The readback+encode is deferred: the run loop polls this frame's fence
    // and hands the pixels to a worker thread (see screenshotInFlight_).
    if (screenshotPending_) {
        if (uiShot_)
            captureUiScreenshotIntoStaging(cmd);
        else
            captureScreenshotIntoStaging(cmd);
        screenshotSlot_ = slot;
        screenshotInFlight_ = true;
        screenshotPending_ = false;
    }

    // --- 6) copy composite + ImGui overlay into the swapchain -----------------------
    const VkImage swapImage = swapchain_.image(swapchainIndex);
    const VkImageView swapView = swapchain_.view(swapchainIndex);
    imageBarrier(cmd, swapImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    {
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
        vkCmdDraw(cmd, 3, 1, 0, 0);

        // The backend records its draws into the active dynamic-rendering block.
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

        vkCmdEndRendering(cmd);
    }
    imageBarrier(cmd, swapImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    timestamps_.frameEnd(cmd, slot);
    vkEndCommandBuffer(cmd);

    prevViewProj_ = Mat4::multiply(proj, view);
}

// Debug UI screenshot: replicate the present block (composite copy + ImGui
// draw data) into uiShotImage_ (swapchain format, so the copy pipeline and
// the ImGui backend pipeline both apply), then read it back.  Runs in
// addition to the real present; used only when SR_GUI_UI_SHOT is set.
void GuiApp::captureUiScreenshotIntoStaging(VkCommandBuffer cmd) {
    const uint32_t dw = active_.displayW;
    const uint32_t dh = active_.displayH;

    imageBarrier(cmd, uiShotImage_.image, uiShotLayout_,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
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
        vkCmdDraw(cmd, 3, 1, 0, 0);
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
        vkCmdEndRendering(cmd);
    }
    imageBarrier(cmd, uiShotImage_.image, uiShotLayout_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
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
    ++screenshotThreads_;
    std::thread([this, pixels, path, dw, dh, swizzle]() {
        if (swizzle) {
            uint8_t* p = pixels->data();
            const size_t n = static_cast<size_t>(dw) * dh;
            for (size_t i = 0; i < n; ++i) std::swap(p[i * 4 + 0], p[i * 4 + 2]);
        }
        const bool ok = savePngFromRgba8(path.c_str(), pixels->data(), dw, dh);
        {
            std::lock_guard<std::mutex> lk(screenshotMsgMutex_);
            screenshotMsg_ = (ok ? "screenshot saved -> " : "screenshot FAILED -> ") + path;
        }
        ++screenshotFinished_;
        --screenshotThreads_;
    }).detach();
}

void GuiApp::saveScreenshot(const char* path) {
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
        screenshotThreads_.load(std::memory_order_relaxed) == 0)
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
    for (uint32_t i = 0; i < budget.heapCount; ++i) heapTotal += budget.heapUsage[i];
    uint64_t algoBytes = 0;
    for (const AlgoColumn& algo : algos_) algoBytes += algo.upscaler->gpuMemoryBytes();

    std::ofstream csv(path);
    if (!csv) {
        statusLine_ = std::string("frame-times CSV: cannot open ") + path;
        return;
    }
    csv << "frame,frameMs,sceneMs,upscaleMs,vramAlgoBytes,vramTotalBytes\n";
    for (size_t i = 0; i < frameTimesLog_.size(); ++i) {
        csv << i << ',' << frameTimesLog_[i].frameMs << ',' << frameTimesLog_[i].sceneMs << ','
            << frameTimesLog_[i].upscaleMs << ',' << algoBytes << ',' << heapTotal << '\n';
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
        const auto now = std::chrono::steady_clock::now();
        const float dt = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;
        if (dt > 0.f) fps_ = fps_ * 0.95f + (1.f / dt) * 0.05f;

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

        // Debug hook (SR_GUI_DEBUG_INPUT=1): heartbeat to correlate input
        // message dispatch with the frame loop during automation tests.
        if (dbgInputEnabled() && renderFrameIndex_ % 120 == 0)
            std::fprintf(stderr, "[run] frame=%u tick=%llu\n", renderFrameIndex_,
                         static_cast<unsigned long long>(GetTickCount64()));

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplWin32_NewFrame();
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
                swapchain_.create(ctx_, active_.displayW, active_.displayH, true);
                ensurePresentSemaphores();
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
                swapchain_.create(ctx_, active_.displayW, active_.displayH, true);
                ensurePresentSemaphores();
            } else if (pres != VK_SUCCESS) {
                break;
            }
            ++uiFrameIndex_;
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
            frameMsHistory_[historyHead_] = static_cast<float>(lastTimings_.frameMs);
            historyHead_ = (historyHead_ + 1) % kHistoryLen;
            if (historyCount_ < kHistoryLen) ++historyCount_;
            frameTimesLog_.push_back(lastTimings_);
            if (frameTimesLog_.size() > 100000) frameTimesLog_.clear(); // runaway guard
            if (active_.mode == Mode::Viewer && frameIndex % 15 == 0) refreshOverlayText();
        }
        if (metricPending_[slot]) {
            harvestMetrics(slot);
            metricPending_[slot] = false;
        }

        uint32_t swapIndex = 0;
        VkResult acq = swapchain_.acquireNext(ctx_, frames_[slot].imageAvailable, swapIndex);
        if (acq == VK_ERROR_OUT_OF_DATE_KHR || acq == VK_SUBOPTIMAL_KHR) {
            vkDeviceWaitIdle(ctx_.device);
            swapchain_.create(ctx_, active_.displayW, active_.displayH, true);
            ensurePresentSemaphores();
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

        recordFrame(frameIndex, swapIndex);

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
            swapchain_.create(ctx_, active_.displayW, active_.displayH, true);
            ensurePresentSemaphores();
        } else if (pres != VK_SUCCESS) {
            break;
        }

        if (screenshotInFlight_ &&
            vkGetFenceStatus(ctx_.device, frames_[screenshotSlot_].fence) == VK_SUCCESS) {
            collectScreenshotPixels();
        }
        if (screenshotFinished_.load(std::memory_order_relaxed) > 0) {
            std::lock_guard<std::mutex> lk(screenshotMsgMutex_);
            if (!screenshotMsg_.empty()) {
                statusLine_ = screenshotMsg_;
                screenshotMsg_.clear();
            }
            screenshotFinished_ = 0;
        }

        ++renderFrameIndex_;
    }

    vkDeviceWaitIdle(ctx_.device);
    // Drain a pending screenshot (automation --screenshot exits right after
    // the capture frame; the device is idle so the copy has completed).
    if (screenshotInFlight_) collectScreenshotPixels();
    while (screenshotThreads_.load(std::memory_order_relaxed) > 0)
        std::this_thread::yield();
    if (dbgInputEnabled())
        std::fprintf(stderr, "[run] loop exited (shouldClose=%d)\n",
                     window_.shouldClose() ? 1 : 0);
}

// ---------------------------------------------------------------------------
// UI.
// ---------------------------------------------------------------------------
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
}

void GuiApp::drawViewerTab() {
    drawSharedControls();
    ImGui::Separator();

    // Upscaler: native + every registered plugin (availability probed).
    {
        const std::string current =
            ui_.viewerUpscaler == 0
                ? "native (ground truth)"
                : upscalerLabels_[static_cast<size_t>(ui_.viewerUpscaler - 1)];
        if (ImGui::BeginCombo("upscaler", current.c_str())) {
            if (ImGui::Selectable("native (ground truth)", ui_.viewerUpscaler == 0))
                ui_.viewerUpscaler = 0;
            for (uint32_t i = 0; i < upscalerNames_.size(); ++i) {
                ImGuiSelectableFlags flags = 0;
                if (i < upscalerAvailable_.size() && !upscalerAvailable_[i])
                    flags |= ImGuiSelectableFlags_Disabled;
                if (ImGui::Selectable(upscalerLabels_[i].c_str(),
                                      ui_.viewerUpscaler == static_cast<int>(i) + 1, flags))
                    ui_.viewerUpscaler = static_cast<int>(i) + 1;
            }
            ImGui::EndCombo();
        }
    }

    const bool loadInFlight = loadPhase_.load(std::memory_order_acquire) == LoadPhase::Loading;
    if (loadInFlight) ImGui::BeginDisabled();
    if (ImGui::Button("apply (rebuild)", ImVec2(-1.f, 0.f)))
        requestRebuild(configFromUi(Mode::Viewer));
    if (loadInFlight) ImGui::EndDisabled();
    if (loadInFlight) ImGui::TextDisabled("loading... (apply disabled)");
    ImGui::Separator();

    // Live performance readout.
    ImGui::Text("frame   %6.2f ms  (%5.1f FPS)", lastTimings_.frameMs, static_cast<double>(fps_));
    ImGui::Text("scene   %6.2f ms", lastTimings_.sceneMs);
    ImGui::Text("upscale %6.2f ms", lastTimings_.upscaleMs);
    if (historyCount_ > 1) {
        // Linearize the ring buffer for PlotLines.
        float ordered[kHistoryLen];
        for (uint32_t i = 0; i < historyCount_; ++i) {
            ordered[i] = frameMsHistory_[(historyHead_ + kHistoryLen - historyCount_ + i) %
                                         kHistoryLen];
        }
        ImGui::PlotLines("frame ms", ordered, static_cast<int>(historyCount_), 0, nullptr, 0.f,
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

    // Multi-select up to kMaxAlgos columns; extra checkboxes are disabled.
    uint32_t selected = 0;
    for (uint32_t i = 0; i < upscalerNames_.size() && i < kMaxRegistered; ++i)
        if (ui_.compareSelected[i]) ++selected;
    ImGui::Text("algorithms (%u/%u):", selected, kMaxAlgos);
    for (uint32_t i = 0; i < upscalerNames_.size() && i < kMaxRegistered; ++i) {
        const bool unavailable =
            i < upscalerAvailable_.size() && !upscalerAvailable_[i] && !ui_.compareSelected[i];
        const bool capped = !ui_.compareSelected[i] && selected >= kMaxAlgos;
        if (unavailable || capped) ImGui::BeginDisabled();
        ImGui::Checkbox(upscalerLabels_[i].c_str(), &ui_.compareSelected[i]);
        if (unavailable || capped) ImGui::EndDisabled();
    }
    if (selected >= kMaxAlgos) ImGui::TextDisabled("max %u columns reached", kMaxAlgos);

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

    const bool any = selected > 0;
    const bool loadInFlight = loadPhase_.load(std::memory_order_acquire) == LoadPhase::Loading;
    if (!any || loadInFlight) ImGui::BeginDisabled();
    if (ImGui::Button("apply (rebuild)", ImVec2(-1.f, 0.f)))
        requestRebuild(configFromUi(Mode::Compare));
    if (!any || loadInFlight) ImGui::EndDisabled();
    if (!any) ImGui::TextDisabled("select at least one algorithm");
    if (loadInFlight) ImGui::TextDisabled("loading... (apply disabled)");
    ImGui::Separator();

    // Live metrics (also drawn on the columns themselves).
    if (ImGui::BeginTable("metrics", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("algo");
        ImGui::TableSetupColumn("PSNR dB");
        ImGui::TableSetupColumn("SSIM");
        ImGui::TableHeadersRow();
        for (const AlgoColumn& algo : algos_) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(algo.upscaler->name());
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
    for (uint32_t i = 0; i < upscalerNames_.size() && i < kMaxRegistered; ++i) {
        ImGui::Checkbox(upscalerLabels_[i].c_str(), &ui_.benchSelected[i]);
    }
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

void GuiApp::drawUi() {
    const ImGuiIO& io = ImGui::GetIO();

    // F1 toggles the side panel (same as View > Hide side panel).
    if (ImGui::IsKeyPressed(ImGuiKey_F1, false) && !io.WantTextInput)
        panelCollapsed_ = !panelCollapsed_;

    if (panelCollapsed_) {
        // Slim strip with just an expand button; the render columns span the
        // full window width underneath.  Parked below the on-composite
        // overlay text band (3 lines x 8px x textScale(2) + 6px pad = 54px
        // in the top-left corner, see compare_compose.frag) so it never
        // covers that text.  NoScrollbar: the full-width button would
        // otherwise push the content region past the client width.
        ImGui::SetNextWindowPos(ImVec2(0.f, 60.f), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(56.f, 34.f), ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.88f);
        if (ImGui::Begin("##panelstrip", nullptr,
                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar |
                             ImGuiWindowFlags_NoScrollbar |
                             ImGuiWindowFlags_NoScrollWithMouse)) {
            if (ImGui::Button(">>", ImVec2(-1.f, 0.f))) panelCollapsed_ = false;
        }
        ImGui::End();
        return;
    }

    ImGui::SetNextWindowPos(ImVec2(0.f, 0.f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(kPanelWidth, io.DisplaySize.y), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.88f);
    ImGui::Begin("sr_compare", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoCollapse);

    // Collapse button (mirrors the F1 hotkey).
    if (ImGui::SmallButton("<<")) panelCollapsed_ = true;
    ImGui::SameLine();
    ImGui::TextDisabled("hide panel (F1)");

    // Global reference selector, shared by all three tabs.
    drawReferenceSection();

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
    ImGui::End();
}

} // namespace sr
