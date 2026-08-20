// ============================================================================
// SlContext — see SlContext.h for the overall design.
// ============================================================================
#include "upscalers/dlss/SlContext.h"

#include "renderer/core/Vk.h" // Vulkan headers must precede the SL headers
#include "upscalers/UpscalerFactory.h"

#include <sl.h>
#include <sl_consts.h>
#include <sl_core_types.h>
#include <sl_helpers.h>
#include <sl_helpers_vk.h>

#include <windows.h> // must precede shellapi.h
#include <shellapi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace sl_dlss {
namespace {

struct State {
    bool initAttempted = false;
    bool initialized = false;
    VkDevice boundDevice = VK_NULL_HANDLE;
    int refCount = 0;

    // Owned copies of the strings SL hands out in FeatureRequirements so the
    // data stays valid regardless of SL-internal storage lifetimes.
    std::vector<std::string> instExtStorage;
    std::vector<std::string> devExtStorage;
    std::vector<const char*> instExts;
    std::vector<const char*> devExts;

    // Feature structs fed to vkCreateDevice through the device-needs hook.
    VkPhysicalDeviceVulkan12Features features12 = {};
    VkPhysicalDeviceVulkan13Features features13 = {};
    bool hasFeatures12 = false;
    bool hasFeatures13 = false;

    sl::FeatureVersion dlssVersion = {};
};

State& state() {
    static State* s = new State();
    return *s;
}

void logCallback(sl::LogType type, const char* msg) {
    const char* tag = type == sl::LogType::eError ? "error" : (type == sl::LogType::eWarn ? "warn" : "info");
    std::fprintf(stderr, "[sl:%s] %s\n", tag, msg);
}

uint32_t applicationIdFromEnv() {
    // A real applicationId is issued through an NVIDIA Developer account;
    // 0 is accepted for development when engine type/version are provided.
    const char* env = std::getenv("SR_DLSS_APP_ID");
    if (!env || !*env) return 0;
    return static_cast<uint32_t>(std::strtoul(env, nullptr, 10));
}

void cacheRequirements() {
    State& s = state();
    sl::FeatureRequirements reqs = {};
    const sl::Result res = slGetFeatureRequirements(sl::kFeatureDLSS, reqs);
    if (res != sl::Result::eOk) {
        std::fprintf(stderr, "[dlss] slGetFeatureRequirements failed: %s (%d)\n",
                     sl::getResultAsStr(res), static_cast<int>(res));
        return;
    }

    std::fprintf(stderr,
                 "[dlss] kFeatureDLSS requirements: flags=0x%x vulkan=%d "
                 "graphicsQueues=%u computeQueues=%u viewports=%u\n",
                 static_cast<uint32_t>(reqs.flags),
                 (reqs.flags & sl::FeatureRequirementFlags::eVulkanSupported) ? 1 : 0,
                 reqs.vkNumGraphicsQueuesRequired, reqs.vkNumComputeQueuesRequired,
                 reqs.maxNumViewports);

    for (uint32_t i = 0; i < reqs.vkNumInstanceExtensions; ++i) {
        s.instExtStorage.emplace_back(reqs.vkInstanceExtensions[i]);
        std::fprintf(stderr, "[dlss]   instance ext: %s\n", reqs.vkInstanceExtensions[i]);
    }
    for (uint32_t i = 0; i < reqs.vkNumDeviceExtensions; ++i) {
        s.devExtStorage.emplace_back(reqs.vkDeviceExtensions[i]);
        std::fprintf(stderr, "[dlss]   device ext:   %s\n", reqs.vkDeviceExtensions[i]);
    }
    s.instExts.clear();
    for (const auto& e : s.instExtStorage) s.instExts.push_back(e.c_str());
    s.devExts.clear();
    for (const auto& e : s.devExtStorage) s.devExts.push_back(e.c_str());

    if (reqs.vkNumFeatures12 > 0) {
        s.features12 = sl::getVkPhysicalDeviceVulkan12Features(reqs.vkNumFeatures12, reqs.vkFeatures12);
        s.hasFeatures12 = true;
        for (uint32_t i = 0; i < reqs.vkNumFeatures12; ++i)
            std::fprintf(stderr, "[dlss]   vk 1.2 feature: %s\n", reqs.vkFeatures12[i]);
    }
    if (reqs.vkNumFeatures13 > 0) {
        s.features13 = sl::getVkPhysicalDeviceVulkan13Features(reqs.vkNumFeatures13, reqs.vkFeatures13);
        s.hasFeatures13 = true;
        for (uint32_t i = 0; i < reqs.vkNumFeatures13; ++i)
            std::fprintf(stderr, "[dlss]   vk 1.3 feature: %s\n", reqs.vkFeatures13[i]);
        // NOTE: the renderer already chains its own VkPhysicalDeviceVulkan13Features
        // (dynamicRendering).  A second struct of the same sType violates the
        // pNext uniqueness VUID; logged loudly so it can be resolved if hit.
        std::fprintf(stderr, "[dlss] WARNING: DLSS requests Vulkan 1.3 features; chaining a "
                             "second VkPhysicalDeviceVulkan13Features struct\n");
    }

    if (slGetFeatureVersion(sl::kFeatureDLSS, s.dlssVersion) == sl::Result::eOk) {
        std::fprintf(stderr, "[dlss] SL %u.%u.%u, NGX %u.%u.%u\n",
                     s.dlssVersion.versionSL.major, s.dlssVersion.versionSL.minor,
                     s.dlssVersion.versionSL.build, s.dlssVersion.versionNGX.major,
                     s.dlssVersion.versionNGX.minor, s.dlssVersion.versionNGX.build);
    }
}

} // namespace

bool ensureInitialized() {
    State& s = state();
    if (s.initAttempted) return s.initialized;
    s.initAttempted = true;

    sl::Preferences pref = {};
#if defined(_DEBUG)
    pref.showConsole = true;
#endif
    pref.logLevel = sl::LogLevel::eDefault;
    pref.logMessageCallback = logCallback;
    pref.pathsToPlugins = nullptr;     // plugins are deployed next to the exe
    pref.numPathsToPlugins = 0;
    pref.pathToLogsAndData = nullptr;  // keep stdout/stderr logging only
    // eDisableDebugText hides the on-screen watermark of development-signed
    // binaries; eUseFrameBasedResourceTagging is required by slSetTagForFrame.
    pref.flags = sl::PreferenceFlags::eDisableCLStateTracking |
                 sl::PreferenceFlags::eAllowOTA |
                 sl::PreferenceFlags::eLoadDownloadedPlugins |
                 sl::PreferenceFlags::eUseFrameBasedResourceTagging |
                 sl::PreferenceFlags::eDisableDebugText;
    const sl::Feature features[] = {sl::kFeatureDLSS};
    pref.featuresToLoad = features;
    pref.numFeaturesToLoad = 1;
    pref.applicationId = applicationIdFromEnv();
    pref.engine = sl::EngineType::eCustom;
    pref.engineVersion = "sr_compare 1.0";
    pref.projectId = "a0f57b54-1daf-4934-90ae-c4035c19df04";
    pref.renderAPI = sl::RenderAPI::eVulkan;

    const sl::Result res = slInit(pref, sl::kSDKVersion);
    if (res != sl::Result::eOk) {
        std::fprintf(stderr, "[dlss] slInit failed: %s (%d)\n", sl::getResultAsStr(res),
                     static_cast<int>(res));
        return false;
    }
    s.initialized = true;
    std::fprintf(stderr, "[dlss] slInit ok (applicationId=%u)\n", pref.applicationId);

    cacheRequirements();
    return true;
}

bool initialized() { return state().initialized; }

bool dlssSupported(VkPhysicalDevice physicalDevice) {
    if (!ensureInitialized()) return false;
    sl::AdapterInfo adapter = {};
    adapter.vkPhysicalDevice = physicalDevice;
    const sl::Result res = slIsFeatureSupported(sl::kFeatureDLSS, adapter);
    if (res != sl::Result::eOk) {
        std::fprintf(stderr, "[dlss] kFeatureDLSS not supported: %s (%d)\n",
                     sl::getResultAsStr(res), static_cast<int>(res));
        return false;
    }
    return true;
}

bool bindDevice(const sr::VulkanEnv& env) {
    State& s = state();
    if (!ensureInitialized()) return false;
    if (s.boundDevice == env.device) return true; // already bound to this device

    // The final executable resolves vkCreateInstance/vkCreateDevice (and the
    // other hooked entry points) to sl.interposer.dll — verified with
    // dumpbin /imports — so the SL proxies associate the device with SL at
    // vkCreateDevice time.  Calling slSetVulkanInfo on top of that is an
    // integration error (eErrorInvalidIntegration); it is only required for
    // a pure-native link where vulkan-1.lib wins symbol resolution.
    s.boundDevice = env.device;
    std::fprintf(stderr, "[dlss] device bound via SL proxies\n");
    return true;
}

void addRef() { ++state().refCount; }

void release() {
    State& s = state();
    if (s.refCount > 0) --s.refCount;
    shutdownIfIdle();
}

// slShutdown when no DLSS upscaler is alive; used both by release() and by
// instances whose init() failed halfway, so SL never leaks past teardown.
void shutdownIfIdle() {
    State& s = state();
    if (s.refCount != 0 || !s.initialized) return;
    // The renderer waits idle and shuts upscalers down before destroying the
    // Vulkan device, so this is a safe point.
    const sl::Result res = slShutdown();
    std::fprintf(stderr, "[dlss] slShutdown: %s (%d)\n", sl::getResultAsStr(res),
                 static_cast<int>(res));
    s.initialized = false;
    s.initAttempted = false; // allow a later re-init within the same process
    s.boundDevice = VK_NULL_HANDLE;
}

// ---------------------------------------------------------------------------
// VulkanDeviceNeeds hooks.  Their essential job is to run slInit before
// vkCreateInstance (the renderer invokes appendInstanceExtensionsHook first),
// as Streamline requires.
//
// The final executable resolves the hooked Vulkan entry points
// (vkCreateInstance/vkCreateDevice/...) to sl.interposer.dll — verified with
// `dumpbin /imports sr_compare.exe` — so the SL proxies inject the required
// instance/device extensions and feature structs themselves.  The hooks
// therefore deliberately contribute NOTHING (feeding the same extensions
// through the host as well risks duplicate pNext structs, and *detecting*
// proxy mode at runtime would require referencing vk* symbols from this
// module, which perturbs import-library symbol resolution at link time and
// can flip vkCreateInstance to vulkan-1.dll — observed as a crash inside the
// device-creation proxy).
//
// slInit is skipped entirely unless a DLSS plugin was requested on the
// command line: once SL initializes its plugins on the device, slShutdown
// becomes mandatory BEFORE device destruction (late/at-exit shutdown crashes
// inside SL), and for non-DLSS runs there is no upscaler instance around to
// perform it.  Skipping slInit leaves the proxies in pass-through mode and
// keeps `--upscaler taa|none` runs clean.
// ---------------------------------------------------------------------------

bool dlssRequestedOnCommandLine() {
    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!argv) return true; // cannot parse -> assume DLSS may be used
    bool requested = false;
    for (int i = 1; i < argc && !requested; ++i) {
        const std::wstring a = argv[i];
        if ((a == L"--upscaler" || a == L"--upscalers") && i + 1 < argc) {
            requested = std::wstring(argv[i + 1]).find(L"dlss") != std::wstring::npos;
        }
    }
    LocalFree(argv);
    return requested;
}

void appendInstanceExtensionsHook(std::vector<const char*>& instanceExts) {
    (void)instanceExts;
    if (sr::allPluginsEnabled() || dlssRequestedOnCommandLine())
        ensureInitialized(); // slInit precedes vkCreateInstance
}

void appendDeviceExtensionsHook(std::vector<const char*>& deviceExts) {
    (void)deviceExts;
    if (sr::allPluginsEnabled() || dlssRequestedOnCommandLine()) ensureInitialized();
}

const void* featureChainHook() {
    if (sr::allPluginsEnabled() || dlssRequestedOnCommandLine()) ensureInitialized();
    return nullptr;
}

} // namespace sl_dlss

namespace {
const sr::VulkanDeviceNeeds kDlssVulkanDeviceNeeds = {
    &sl_dlss::appendDeviceExtensionsHook,
    &sl_dlss::featureChainHook,
    &sl_dlss::appendInstanceExtensionsHook,
    nullptr,
};
} // namespace

SR_REGISTER_VULKAN_DEVICE_NEEDS(kDlssVulkanDeviceNeeds);
