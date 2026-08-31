// DLSS 5 Neural Rendering (DLSSNR) technology-validation spike.
//
// Goal: prove (or disprove) that the leaked DLSS 5 "neural rendering" NGX
// feature (feature id 18 / NVSDK_NGX_Feature_Reserved18, runtime
// nvngx_dlssnr.dll 310.8.0.0) can be created and executed through the NGX
// API on this machine (RTX 4070 SUPER, driver 610.62, Windows, Vulkan).
//
// VERDICT (2026-08-31, see spike/dlss5/README.md for the full write-up):
// blocked on this machine/driver by the shipping driver's NGX runtime,
// which does not know feature 18; NVIDIA's OTA servers (which would deliver
// new feature definitions) are unreachable from this network (curl -1).
// All of the following routes were implemented and tested, with identical
// outcomes for the stock signed DLL and the community "310.8.SF" patch:
//
//   1. NGX core route (nvsdk_ngx_d glue -> nvngx_dlss.dll -> plugin):
//      Init succeeds, but CreateFeature1(18) -> 0xBAD0000C (OutOfDate);
//      nvngx.log: "required feature is not supported by NGX runtime,
//      please update display driver". The driver core never even probes
//      nvngx_dlssnr.dll; dlss/dlssg validate fine, so the runtime works -
//      it just has no feature-18 entry.
//   2. Direct plugin route (the renodx-dlss5 recipe: LoadLibrary +
//      IAT-patch GetModuleFileNameW + direct Init): Vulkan and D3D12
//      Init/Init_Ext/Init_Ext2/PopulateParameters_Impl all -> 0xBAD00002
//      (PlatformError). An IAT spy shows Init makes no registry/file/
//      module calls before failing - the plugin expects in-process runtime
//      state that only its real host loader (NGX core / Streamline) sets up.
//   3. Streamline route (sl.interposer + sl.dlss_nr.dll): slInit succeeds,
//      DLSS NR's sl::Feature id was identified as 1004 (kFeatureDLSS_NR in
//      sl.log), but sl.dlss_nr's NGX requirements query -> 0xBAD0000C:
//      "DLSS-NR feature is not supported. Please check if you have a valid
//      nvngx_dlssnr.dll or your driver supports DLSS-NR."
//
// Modes:
//   --api vulkan   direct-plugin Vulkan spike (default): full pipeline from
//                  extension negotiation to EvaluateFeature + PNG readback
//   --api d3d12    direct-plugin D3D12 spike (renodx-dlss5's API surface)
//   --api slprobe  Streamline feature-id probe via sl.interposer
// Flags: --dll <nvngx_dlssnr.dll> --out <dir> --no-iat-patch --spy
//
// Every NGX call is wrapped in SEH (__try/__except) because the plugin is a
// beta-quality leaked DLL and may crash instead of returning an error.

#include <vulkan/vulkan.h>

#include "nvsdk_ngx_vk.h"
#include "nvsdk_ngx_helpers_vk.h"

#include "sl_core_api.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace
{

constexpr uint32_t kWidth = 1280;
constexpr uint32_t kHeight = 720;
constexpr NVSDK_NGX_Feature kFeatureId = static_cast<NVSDK_NGX_Feature>(18); // NVSDK_NGX_Feature_Reserved18
constexpr unsigned long long kAppId = 0x1000000ULL;
constexpr NVSDK_NGX_Result kSehSentinel = static_cast<NVSDK_NGX_Result>(0xDEAD0000);

void Log(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stdout, fmt, args);
    va_end(args);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

const char* ResultName(NVSDK_NGX_Result r)
{
    if (r == kSehSentinel)
        return "SEH-EXCEPTION";
    switch (r)
    {
    case NVSDK_NGX_Result_Success: return "Success";
    case NVSDK_NGX_Result_Fail: return "Fail";
    case NVSDK_NGX_Result_FAIL_FeatureNotSupported: return "FeatureNotSupported";
    case NVSDK_NGX_Result_FAIL_PlatformError: return "PlatformError";
    case NVSDK_NGX_Result_FAIL_FeatureAlreadyExists: return "FeatureAlreadyExists";
    case NVSDK_NGX_Result_FAIL_FeatureNotFound: return "FeatureNotFound";
    case NVSDK_NGX_Result_FAIL_InvalidParameter: return "InvalidParameter";
    case NVSDK_NGX_Result_FAIL_NotInitialized: return "NotInitialized";
    case NVSDK_NGX_Result_FAIL_UnsupportedInputFormat: return "UnsupportedInputFormat";
    case NVSDK_NGX_Result_FAIL_RWFlagMissing: return "RWFlagMissing";
    case NVSDK_NGX_Result_FAIL_MissingInput: return "MissingInput";
    case NVSDK_NGX_Result_FAIL_UnableToInitializeFeature: return "UnableToInitializeFeature";
    case NVSDK_NGX_Result_FAIL_OutOfDate: return "OutOfDate";
    case NVSDK_NGX_Result_FAIL_OutOfGPUMemory: return "OutOfGPUMemory";
    case NVSDK_NGX_Result_FAIL_UnsupportedFormat: return "UnsupportedFormat";
    case NVSDK_NGX_Result_FAIL_UnableToWriteToAppDataPath: return "UnableToWriteToAppDataPath";
    case NVSDK_NGX_Result_FAIL_UnsupportedParameter: return "UnsupportedParameter";
    case NVSDK_NGX_Result_FAIL_Denied: return "Denied";
    case NVSDK_NGX_Result_FAIL_NotImplemented: return "NotImplemented";
    default: return "?";
    }
}

// ---------------------------------------------------------------------------
// SEH-wrapped NGX call. The lambda must return NVSDK_NGX_Result. No locals
// with destructors are allowed in this function (C2712); keep it minimal.
// ---------------------------------------------------------------------------
template <typename F>
NVSDK_NGX_Result NgxCall(const char* what, F fn)
{
    NVSDK_NGX_Result r = NVSDK_NGX_Result_Fail;
    unsigned long code = 0;
    __try
    {
        r = fn();
    }
    __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER)
    {
        Log("[NGX] %-45s -> SEH exception 0x%08lX", what, code);
        return kSehSentinel;
    }
    Log("[NGX] %-45s -> 0x%08X (%s)", what, static_cast<unsigned int>(r), ResultName(r));
    return r;
}

// ---------------------------------------------------------------------------
// Application-side NVSDK_NGX_Parameter implementation.
//
// No 310.x NGX DLL exports AllocateParameters / NVSDK_NGX_Parameter_Set*:
// the NGX runtime compiled into the plugin expects the caller to hand it an
// object implementing this vtable (same approach as DLSS5-Feeder). Values
// are stored loosely typed and converted between numeric types on Get.
// ---------------------------------------------------------------------------
class SpikeParameterMap final : public NVSDK_NGX_Parameter
{
public:
    void Set(const char* name, unsigned long long value) override { m_Num[name] = static_cast<double>(value); }
    void Set(const char* name, float value) override { m_Num[name] = static_cast<double>(value); }
    void Set(const char* name, double value) override { m_Num[name] = value; }
    void Set(const char* name, unsigned int value) override { m_Num[name] = static_cast<double>(value); }
    void Set(const char* name, int value) override { m_Num[name] = static_cast<double>(value); }
    void Set(const char* name, ID3D11Resource* value) override { m_Ptr[name] = value; }
    void Set(const char* name, ID3D12Resource* value) override { m_Ptr[name] = value; }
    void Set(const char* name, void* value) override { m_Ptr[name] = value; }

    NVSDK_NGX_Result Get(const char* name, unsigned long long* out) const override { return GetNum(name, out); }
    NVSDK_NGX_Result Get(const char* name, float* out) const override { return GetNum(name, out); }
    NVSDK_NGX_Result Get(const char* name, double* out) const override { return GetNum(name, out); }
    NVSDK_NGX_Result Get(const char* name, unsigned int* out) const override { return GetNum(name, out); }
    NVSDK_NGX_Result Get(const char* name, int* out) const override { return GetNum(name, out); }
    NVSDK_NGX_Result Get(const char* name, ID3D11Resource** out) const override { return GetPtr(name, reinterpret_cast<void**>(out)); }
    NVSDK_NGX_Result Get(const char* name, ID3D12Resource** out) const override { return GetPtr(name, reinterpret_cast<void**>(out)); }
    NVSDK_NGX_Result Get(const char* name, void** out) const override { return GetPtr(name, out); }

    void Reset() override { m_Num.clear(); m_Ptr.clear(); }

private:
    template <typename T>
    NVSDK_NGX_Result GetNum(const char* name, T* out) const
    {
        if (!out)
            return NVSDK_NGX_Result_FAIL_InvalidParameter;
        auto it = m_Num.find(name);
        if (it == m_Num.end())
            return NVSDK_NGX_Result_FAIL_InvalidParameter;
        *out = static_cast<T>(it->second);
        return NVSDK_NGX_Result_Success;
    }
    NVSDK_NGX_Result GetPtr(const char* name, void** out) const
    {
        if (!out)
            return NVSDK_NGX_Result_FAIL_InvalidParameter;
        auto it = m_Ptr.find(name);
        if (it == m_Ptr.end())
            return NVSDK_NGX_Result_FAIL_InvalidParameter;
        *out = it->second;
        return NVSDK_NGX_Result_Success;
    }

    std::unordered_map<std::string, double> m_Num;
    std::unordered_map<std::string, void*> m_Ptr;
};

// ---------------------------------------------------------------------------
// IAT patching (the renodx-dlss5 recipe): neutralize the plugin's
// GetModuleFileNameW import. Queries for a real module handle are passed
// through; only the host-executable query (hModule == NULL) is hidden.
// ---------------------------------------------------------------------------
UINT WINAPI GetModuleFileNameW_Stub(HMODULE hModule, LPWSTR lpFilename, DWORD nSize)
{
    if (hModule != nullptr)
        return ::GetModuleFileNameW(hModule, lpFilename, nSize);
    Log("[IAT] GetModuleFileNameW(NULL) intercepted -> reporting failure");
    if (nSize > 0 && lpFilename)
        lpFilename[0] = L'\0';
    return 0;
}

bool PatchIatByName(HMODULE mod, const char* funcName, void* newFunc)
{
    auto* base = reinterpret_cast<uint8_t*>(mod);
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE)
        return false;
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    const auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dir.VirtualAddress)
        return false;
    auto* desc = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(base + dir.VirtualAddress);
    bool patched = false;
    for (; desc->Name; ++desc)
    {
        const char* dllName = reinterpret_cast<const char*>(base + desc->Name);
        auto* thunk = reinterpret_cast<IMAGE_THUNK_DATA64*>(base + (desc->OriginalFirstThunk ? desc->OriginalFirstThunk : desc->FirstThunk));
        auto* iat = reinterpret_cast<IMAGE_THUNK_DATA64*>(base + desc->FirstThunk);
        for (; thunk->u1.AddressOfData; ++thunk, ++iat)
        {
            if (thunk->u1.Ordinal & IMAGE_ORDINAL_FLAG64)
                continue;
            auto* imp = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(base + thunk->u1.AddressOfData);
            if (std::strcmp(imp->Name, funcName) != 0)
                continue;
            DWORD oldProt = 0;
            if (!::VirtualProtect(&iat->u1.Function, sizeof(void*), PAGE_READWRITE, &oldProt))
            {
                Log("[IAT] VirtualProtect failed for %s!%s (err %lu)", dllName, funcName, ::GetLastError());
                return false;
            }
            Log("[IAT] patched %s!%s: 0x%p -> 0x%p", dllName, funcName,
                reinterpret_cast<void*>(iat->u1.Function), newFunc);
            iat->u1.Function = reinterpret_cast<ULONGLONG>(newFunc);
            ::VirtualProtect(&iat->u1.Function, sizeof(void*), oldProt, &oldProt);
            patched = true;
        }
    }
    if (!patched)
        Log("[IAT] %s not found in plugin imports", funcName);
    return patched;
}

// ---------------------------------------------------------------------------
// "--spy" IAT wrappers: log the plugin's own Win32 calls during Init so we
// can see where PlatformError comes from (registry keys, module loads...).
// ---------------------------------------------------------------------------
typedef HMODULE(WINAPI* PFN_LoadLibraryW_)(LPCWSTR);
typedef HMODULE(WINAPI* PFN_LoadLibraryExW_)(LPCWSTR, HANDLE, DWORD);
typedef HMODULE(WINAPI* PFN_GetModuleHandleW_)(LPCWSTR);
typedef BOOL(WINAPI* PFN_GetModuleHandleExW_)(DWORD, LPCWSTR, HMODULE*);
typedef FARPROC(WINAPI* PFN_GetProcAddress_)(HMODULE, LPCSTR);
typedef LSTATUS(WINAPI* PFN_RegOpenKeyExW_)(HKEY, LPCWSTR, DWORD, REGSAM, PHKEY);
typedef LSTATUS(WINAPI* PFN_RegQueryValueExW_)(HKEY, LPCWSTR, LPDWORD, LPDWORD, LPBYTE, LPDWORD);

PFN_LoadLibraryW_ g_realLoadLibraryW = nullptr;
PFN_LoadLibraryExW_ g_realLoadLibraryExW = nullptr;
PFN_GetModuleHandleW_ g_realGetModuleHandleW = nullptr;
PFN_GetModuleHandleExW_ g_realGetModuleHandleExW = nullptr;
PFN_GetProcAddress_ g_realGetProcAddress = nullptr;
PFN_RegOpenKeyExW_ g_realRegOpenKeyExW = nullptr;
PFN_RegQueryValueExW_ g_realRegQueryValueExW = nullptr;

HMODULE WINAPI LoadLibraryW_Spy(LPCWSTR name)
{
    HMODULE h = g_realLoadLibraryW(name);
    Log("[SPY] LoadLibraryW(%ls) -> 0x%p", name ? name : L"(null)", static_cast<void*>(h));
    return h;
}
HMODULE WINAPI LoadLibraryExW_Spy(LPCWSTR name, HANDLE file, DWORD flags)
{
    HMODULE h = g_realLoadLibraryExW(name, file, flags);
    Log("[SPY] LoadLibraryExW(%ls, 0x%lX) -> 0x%p", name ? name : L"(null)", flags, static_cast<void*>(h));
    return h;
}
HMODULE WINAPI GetModuleHandleW_Spy(LPCWSTR name)
{
    HMODULE h = g_realGetModuleHandleW(name);
    Log("[SPY] GetModuleHandleW(%ls) -> 0x%p", name ? name : L"(null)", static_cast<void*>(h));
    return h;
}
BOOL WINAPI GetModuleHandleExW_Spy(DWORD flags, LPCWSTR name, HMODULE* out)
{
    BOOL r = g_realGetModuleHandleExW(flags, name, out);
    Log("[SPY] GetModuleHandleExW(%ls) -> %d (0x%p)", name ? name : L"(null)", r, out ? static_cast<void*>(*out) : nullptr);
    return r;
}
FARPROC WINAPI GetProcAddress_Spy(HMODULE mod, LPCSTR name)
{
    FARPROC p = g_realGetProcAddress(mod, name);
    if (reinterpret_cast<ULONG_PTR>(name) <= 0xFFFF)
        Log("[SPY] GetProcAddress(0x%p, #%llu) -> 0x%p", static_cast<void*>(mod),
            static_cast<unsigned long long>(reinterpret_cast<ULONG_PTR>(name)), reinterpret_cast<void*>(p));
    else
        Log("[SPY] GetProcAddress(0x%p, %s) -> 0x%p", static_cast<void*>(mod), name, reinterpret_cast<void*>(p));
    return p;
}
LSTATUS WINAPI RegOpenKeyExW_Spy(HKEY key, LPCWSTR sub, DWORD opt, REGSAM sam, PHKEY out)
{
    LSTATUS r = g_realRegOpenKeyExW(key, sub, opt, sam, out);
    Log("[SPY] RegOpenKeyExW(0x%p, %ls) -> %ld", static_cast<void*>(key), sub ? sub : L"(null)", r);
    return r;
}
LSTATUS WINAPI RegQueryValueExW_Spy(HKEY key, LPCWSTR value, LPDWORD res, LPDWORD type, LPBYTE data, LPDWORD size)
{
    LSTATUS r = g_realRegQueryValueExW(key, value, res, type, data, size);
    if (r == ERROR_SUCCESS && type && data && size && (*type == REG_SZ || *type == REG_EXPAND_SZ))
        Log("[SPY] RegQueryValueExW(%ls) -> %ld = '%ls'", value ? value : L"(null)", r, reinterpret_cast<LPCWSTR>(data));
    else if (r == ERROR_SUCCESS && type && data && size && *type == REG_DWORD && *size >= 4)
        Log("[SPY] RegQueryValueExW(%ls) -> %ld = %lu", value ? value : L"(null)", r, *reinterpret_cast<DWORD*>(data));
    else
        Log("[SPY] RegQueryValueExW(%ls) -> %ld", value ? value : L"(null)", r);
    return r;
}

void InstallSpy(HMODULE plugin)
{
    HMODULE k32 = ::GetModuleHandleW(L"KERNEL32.dll");
    HMODULE adv = ::GetModuleHandleW(L"ADVAPI32.dll");
    g_realLoadLibraryW = reinterpret_cast<PFN_LoadLibraryW_>(::GetProcAddress(k32, "LoadLibraryW"));
    g_realLoadLibraryExW = reinterpret_cast<PFN_LoadLibraryExW_>(::GetProcAddress(k32, "LoadLibraryExW"));
    g_realGetModuleHandleW = reinterpret_cast<PFN_GetModuleHandleW_>(::GetProcAddress(k32, "GetModuleHandleW"));
    g_realGetModuleHandleExW = reinterpret_cast<PFN_GetModuleHandleExW_>(::GetProcAddress(k32, "GetModuleHandleExW"));
    g_realGetProcAddress = reinterpret_cast<PFN_GetProcAddress_>(::GetProcAddress(k32, "GetProcAddress"));
    g_realRegOpenKeyExW = reinterpret_cast<PFN_RegOpenKeyExW_>(::GetProcAddress(adv, "RegOpenKeyExW"));
    g_realRegQueryValueExW = reinterpret_cast<PFN_RegQueryValueExW_>(::GetProcAddress(adv, "RegQueryValueExW"));
    PatchIatByName(plugin, "LoadLibraryW", reinterpret_cast<void*>(&LoadLibraryW_Spy));
    PatchIatByName(plugin, "LoadLibraryExW", reinterpret_cast<void*>(&LoadLibraryExW_Spy));
    PatchIatByName(plugin, "GetModuleHandleW", reinterpret_cast<void*>(&GetModuleHandleW_Spy));
    PatchIatByName(plugin, "GetModuleHandleExW", reinterpret_cast<void*>(&GetModuleHandleExW_Spy));
    PatchIatByName(plugin, "GetProcAddress", reinterpret_cast<void*>(&GetProcAddress_Spy));
    PatchIatByName(plugin, "RegOpenKeyExW", reinterpret_cast<void*>(&RegOpenKeyExW_Spy));
    PatchIatByName(plugin, "RegQueryValueExW", reinterpret_cast<void*>(&RegQueryValueExW_Spy));
}

// ---------------------------------------------------------------------------
// NGX function table (resolved from nvngx_dlssnr.dll via GetProcAddress)
// ---------------------------------------------------------------------------
typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_GetFeatureInstanceExtensionRequirements)(const NVSDK_NGX_FeatureDiscoveryInfo*, uint32_t*, VkExtensionProperties**);
typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_GetFeatureDeviceExtensionRequirements)(VkInstance, VkPhysicalDevice, const NVSDK_NGX_FeatureDiscoveryInfo*, uint32_t*, VkExtensionProperties**);
typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_GetFeatureRequirements)(VkInstance, VkPhysicalDevice, const NVSDK_NGX_FeatureDiscoveryInfo*, NVSDK_NGX_FeatureRequirement*);
typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_Init)(unsigned long long, const wchar_t*, VkInstance, VkPhysicalDevice, VkDevice, NVSDK_NGX_Version);
typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_Init_Ext)(unsigned long long, const wchar_t*, VkInstance, VkPhysicalDevice, VkDevice, NVSDK_NGX_Version, const NVSDK_NGX_Parameter*);
typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_Init_Ext2)(unsigned long long, const wchar_t*, VkInstance, VkPhysicalDevice, VkDevice, PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr, NVSDK_NGX_Version, const NVSDK_NGX_Parameter*);
typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_CreateFeature1)(VkDevice, VkCommandBuffer, NVSDK_NGX_Feature, const NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**);
typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_EvaluateFeature)(VkCommandBuffer, const NVSDK_NGX_Handle*, const NVSDK_NGX_Parameter*, PFN_NVSDK_NGX_ProgressCallback);
typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_ReleaseFeature)(NVSDK_NGX_Handle*);
typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_Shutdown1)(VkDevice);

struct NgxApi
{
    HMODULE dll = nullptr;
    PFN_GetFeatureInstanceExtensionRequirements GetInstanceExts = nullptr;
    PFN_GetFeatureDeviceExtensionRequirements GetDeviceExts = nullptr;
    PFN_GetFeatureRequirements GetFeatureRequirements = nullptr;
    PFN_Init Init = nullptr;
    PFN_Init_Ext InitExt = nullptr;
    PFN_Init_Ext2 InitExt2 = nullptr;
    PFN_CreateFeature1 CreateFeature1 = nullptr;
    PFN_EvaluateFeature EvaluateFeature = nullptr;
    PFN_ReleaseFeature ReleaseFeature = nullptr;
    PFN_Shutdown1 Shutdown1 = nullptr;
};

template <typename T>
bool LoadProc(HMODULE dll, const char* name, T& out, bool required = true)
{
    out = reinterpret_cast<T>(::GetProcAddress(dll, name));
    if (!out)
    {
        Log("[NGX] GetProcAddress(%s) FAILED (err %lu)%s", name, ::GetLastError(), required ? "" : " [optional]");
        return !required;
    }
    return true;
}

bool LoadNgx(const std::string& dllPath, NgxApi& api)
{
    api.dll = ::LoadLibraryA(dllPath.c_str());
    if (!api.dll)
    {
        Log("[NGX] LoadLibrary(%s) failed, err %lu", dllPath.c_str(), ::GetLastError());
        return false;
    }
    Log("[NGX] LoadLibrary(%s) OK", dllPath.c_str());

    bool ok = true;
    ok &= LoadProc(api.dll, "NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements", api.GetInstanceExts);
    ok &= LoadProc(api.dll, "NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements", api.GetDeviceExts);
    ok &= LoadProc(api.dll, "NVSDK_NGX_VULKAN_GetFeatureRequirements", api.GetFeatureRequirements);
    ok &= LoadProc(api.dll, "NVSDK_NGX_VULKAN_Init_Ext2", api.InitExt2, false);
    ok &= LoadProc(api.dll, "NVSDK_NGX_VULKAN_Init_Ext", api.InitExt, false);
    ok &= LoadProc(api.dll, "NVSDK_NGX_VULKAN_Init", api.Init, false);
    ok &= LoadProc(api.dll, "NVSDK_NGX_VULKAN_CreateFeature1", api.CreateFeature1);
    ok &= LoadProc(api.dll, "NVSDK_NGX_VULKAN_EvaluateFeature", api.EvaluateFeature);
    ok &= LoadProc(api.dll, "NVSDK_NGX_VULKAN_ReleaseFeature", api.ReleaseFeature);
    ok &= LoadProc(api.dll, "NVSDK_NGX_VULKAN_Shutdown1", api.Shutdown1);
    if (!api.InitExt2 && !api.InitExt && !api.Init)
    {
        Log("[NGX] no usable Init entry point");
        ok = false;
    }
    return ok;
}

// ---------------------------------------------------------------------------
// Minimal Vulkan scaffolding
// ---------------------------------------------------------------------------
struct VkCheck
{
    VkCheck(VkResult r, const char* what)
    {
        if (r != VK_SUCCESS)
        {
            Log("[VK] %s failed: %d", what, static_cast<int>(r));
            std::exit(1);
        }
    }
};

struct GpuImage
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspect = 0;
};

struct VulkanCtx
{
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memProps{};
};

uint32_t FindMemoryType(const VulkanCtx& ctx, uint32_t typeBits, VkMemoryPropertyFlags props)
{
    for (uint32_t i = 0; i < ctx.memProps.memoryTypeCount; ++i)
    {
        if ((typeBits & (1u << i)) && (ctx.memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    Log("[VK] no suitable memory type (bits 0x%08X props 0x%08X)", typeBits, static_cast<unsigned>(props));
    std::exit(1);
}

void BeginCmd(VulkanCtx& ctx)
{
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkCheck(vkBeginCommandBuffer(ctx.cmd, &bi), "vkBeginCommandBuffer");
}

void SubmitWait(VulkanCtx& ctx)
{
    VkCheck(vkEndCommandBuffer(ctx.cmd), "vkEndCommandBuffer");
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &ctx.cmd;
    VkCheck(vkQueueSubmit(ctx.queue, 1, &si, ctx.fence), "vkQueueSubmit");
    VkCheck(vkWaitForFences(ctx.device, 1, &ctx.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
    VkCheck(vkResetFences(ctx.device, 1, &ctx.fence), "vkResetFences");
    VkCheck(vkResetCommandBuffer(ctx.cmd, 0), "vkResetCommandBuffer");
}

GpuImage CreateGpuImage(VulkanCtx& ctx, VkFormat format, VkImageAspectFlags aspect, VkImageUsageFlags usage)
{
    GpuImage gi{};
    gi.format = format;
    gi.aspect = aspect;

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {kWidth, kHeight, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkCheck(vkCreateImage(ctx.device, &ici, nullptr, &gi.image), "vkCreateImage");

    VkMemoryRequirements mr{};
    vkGetImageMemoryRequirements(ctx.device, gi.image, &mr);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = FindMemoryType(ctx, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkCheck(vkAllocateMemory(ctx.device, &mai, nullptr, &gi.memory), "vkAllocateMemory(image)");
    VkCheck(vkBindImageMemory(ctx.device, gi.image, gi.memory, 0), "vkBindImageMemory");

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = gi.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.subresourceRange = {aspect, 0, 1, 0, 1};
    VkCheck(vkCreateImageView(ctx.device, &vci, nullptr, &gi.view), "vkCreateImageView");
    return gi;
}

struct StagingBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
};

StagingBuffer CreateStaging(VulkanCtx& ctx, VkDeviceSize size)
{
    StagingBuffer sb{};
    sb.size = size;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkCheck(vkCreateBuffer(ctx.device, &bci, nullptr, &sb.buffer), "vkCreateBuffer(staging)");
    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(ctx.device, sb.buffer, &mr);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = FindMemoryType(ctx, mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkCheck(vkAllocateMemory(ctx.device, &mai, nullptr, &sb.memory), "vkAllocateMemory(staging)");
    VkCheck(vkBindBufferMemory(ctx.device, sb.buffer, sb.memory, 0), "vkBindBufferMemory");
    VkCheck(vkMapMemory(ctx.device, sb.memory, 0, size, 0, &sb.mapped), "vkMapMemory");
    return sb;
}

void TransitionToGeneral(VulkanCtx& ctx, VkImage image, VkImageAspectFlags aspect)
{
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = 0;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {aspect, 0, 1, 0, 1};
    vkCmdPipelineBarrier(ctx.cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
        0, nullptr, 0, nullptr, 1, &b);
}

void CopyBufferToImage(VulkanCtx& ctx, VkBuffer src, VkImage dst, VkImageAspectFlags aspect)
{
    VkBufferImageCopy region{};
    region.imageSubresource = {aspect, 0, 0, 1};
    region.imageExtent = {kWidth, kHeight, 1};
    vkCmdCopyBufferToImage(ctx.cmd, src, dst, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
}

void CopyImageToBuffer(VulkanCtx& ctx, VkImage src, VkImageAspectFlags aspect, VkBuffer dst)
{
    VkBufferImageCopy region{};
    region.imageSubresource = {aspect, 0, 0, 1};
    region.imageExtent = {kWidth, kHeight, 1};
    vkCmdCopyImageToBuffer(ctx.cmd, src, VK_IMAGE_LAYOUT_GENERAL, dst, 1, &region);
}

// ---------------------------------------------------------------------------
// float16 helpers + test pattern
// ---------------------------------------------------------------------------
uint16_t F32ToF16(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int exp = static_cast<int>((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0)
        return static_cast<uint16_t>(sign); // flush to zero (our inputs are well inside range)
    if (exp >= 31)
        return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

float F16ToF32(uint16_t h)
{
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t x;
    if (exp == 0)
        x = sign; // treat denormals as zero, fine for stats
    else if (exp == 31)
        x = sign | 0x7F800000u | (mant << 13);
    else
        x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// Top half: 8 vertical color bars. Bottom half: r/g gradient + b checker.
void FillColorPattern(std::vector<uint16_t>& pix)
{
    static const float kBars[8][3] = {
        {1, 1, 1}, {1, 1, 0}, {0, 1, 1}, {0, 1, 0},
        {1, 0, 1}, {1, 0, 0}, {0, 0, 1}, {0, 0, 0},
    };
    for (uint32_t y = 0; y < kHeight; ++y)
    {
        for (uint32_t x = 0; x < kWidth; ++x)
        {
            float r, g, b;
            if (y < kHeight / 2)
            {
                const float* c = kBars[(x * 8) / kWidth];
                r = c[0]; g = c[1]; b = c[2];
            }
            else
            {
                r = static_cast<float>(x) / static_cast<float>(kWidth - 1);
                g = static_cast<float>(y) / static_cast<float>(kHeight - 1);
                b = ((x / 40 + y / 40) % 2) ? 0.75f : 0.25f;
            }
            size_t i = (static_cast<size_t>(y) * kWidth + x) * 4;
            pix[i + 0] = F32ToF16(r);
            pix[i + 1] = F32ToF16(g);
            pix[i + 2] = F32ToF16(b);
            pix[i + 3] = F32ToF16(1.0f);
        }
    }
}

bool WritePng(const std::string& path, const std::vector<uint16_t>& rgba16)
{
    std::vector<uint8_t> rgb(static_cast<size_t>(kWidth) * kHeight * 3);
    for (size_t i = 0; i < static_cast<size_t>(kWidth) * kHeight; ++i)
    {
        for (int c = 0; c < 3; ++c)
        {
            float v = std::max(0.0f, std::min(1.0f, F16ToF32(rgba16[i * 4 + c])));
            v = std::pow(v, 1.0f / 2.2f); // simple gamma so the PNG is viewable
            rgb[i * 3 + c] = static_cast<uint8_t>(v * 255.0f + 0.5f);
        }
    }
    return stbi_write_png(path.c_str(), static_cast<int>(kWidth), static_cast<int>(kHeight), 3,
        rgb.data(), static_cast<int>(kWidth) * 3) != 0;
}

void PrintStats(const char* label, const std::vector<uint16_t>& rgba16, const std::vector<uint16_t>* ref)
{
    float mn[3] = {1e30f, 1e30f, 1e30f};
    float mx[3] = {-1e30f, -1e30f, -1e30f};
    double sum[3] = {0, 0, 0};
    double diffSum = 0;
    size_t n = static_cast<size_t>(kWidth) * kHeight;
    for (size_t i = 0; i < n; ++i)
    {
        for (int c = 0; c < 3; ++c)
        {
            float v = F16ToF32(rgba16[i * 4 + c]);
            mn[c] = std::min(mn[c], v);
            mx[c] = std::max(mx[c], v);
            sum[c] += v;
            if (ref)
                diffSum += std::fabs(v - F16ToF32((*ref)[i * 4 + c]));
        }
    }
    Log("[STATS] %s: min=(%.4f %.4f %.4f) max=(%.4f %.4f %.4f) mean=(%.4f %.4f %.4f)",
        label, mn[0], mn[1], mn[2], mx[0], mx[1], mx[2],
        sum[0] / n, sum[1] / n, sum[2] / n);
    if (ref)
        Log("[STATS] %s: mean abs diff vs input color = %.6f", label, diffSum / (3.0 * n));
}

std::wstring ToWide(const std::string& s)
{
    return std::wstring(s.begin(), s.end());
}

} // namespace

// ===========================================================================
// D3D12 mode (--api d3d12)
//
// The renodx-dlss5 add-on drives the plugin exclusively through its
// NVSDK_NGX_D3D12_* exports ("signed DLSSNR 310.8.0 D3D12 runtime
// initialized"), while the plugin's Vulkan Init returns 0xBAD00002 in our
// direct tests. This mode replicates the add-on's recipe standalone:
// private D3D12 device on the NVIDIA adapter -> D3D12_Init_Ext ->
// CreateFeature(18) -> EvaluateFeature on R16G16B16A16_FLOAT textures.
// ===========================================================================
namespace d3d12mode
{

using Microsoft::WRL::ComPtr;

typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_D3D12_Init)(unsigned long long, const wchar_t*, ID3D12Device*, NVSDK_NGX_Version);
typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_D3D12_Init_Ext)(unsigned long long, const wchar_t*, ID3D12Device*, NVSDK_NGX_Version, const NVSDK_NGX_Parameter*);
typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_D3D12_PopulateParameters)(NVSDK_NGX_Parameter*);
typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_D3D12_GetFeatureRequirements)(IDXGIAdapter*, const NVSDK_NGX_FeatureDiscoveryInfo*, NVSDK_NGX_FeatureRequirement*);
typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_D3D12_CreateFeature)(ID3D12GraphicsCommandList*, NVSDK_NGX_Feature, const NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**);
typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_D3D12_EvaluateFeature)(ID3D12GraphicsCommandList*, const NVSDK_NGX_Handle*, const NVSDK_NGX_Parameter*, PFN_NVSDK_NGX_ProgressCallback);
typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_D3D12_ReleaseFeature)(NVSDK_NGX_Handle*);
typedef NVSDK_NGX_Result(NVSDK_CONV* PFN_D3D12_Shutdown1)(ID3D12Device*);

struct Ngx12Api
{
    PFN_D3D12_Init Init = nullptr;
    PFN_D3D12_Init_Ext InitExt = nullptr;
    PFN_D3D12_PopulateParameters PopulateParameters = nullptr;
    PFN_D3D12_GetFeatureRequirements GetFeatureRequirements = nullptr;
    PFN_D3D12_CreateFeature CreateFeature = nullptr;
    PFN_D3D12_EvaluateFeature EvaluateFeature = nullptr;
    PFN_D3D12_ReleaseFeature ReleaseFeature = nullptr;
    PFN_D3D12_Shutdown1 Shutdown1 = nullptr;
};

void HrCheck(HRESULT hr, const char* what)
{
    if (FAILED(hr))
    {
        Log("[D3D12] %s failed: 0x%08lX", what, static_cast<unsigned long>(hr));
        std::exit(1);
    }
}

struct CmdCtx
{
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE fenceEvent = nullptr;
    UINT64 fenceValue = 0;
};

void CmdBegin(ID3D12Device* dev, CmdCtx& c)
{
    HrCheck(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&c.alloc)), "CreateCommandAllocator");
    HrCheck(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, c.alloc.Get(), nullptr, IID_PPV_ARGS(&c.list)), "CreateCommandList");
}

void CmdSubmitWait(ID3D12CommandQueue* queue, CmdCtx& c)
{
    HrCheck(c.list->Close(), "Close");
    ID3D12CommandList* lists[] = {c.list.Get()};
    queue->ExecuteCommandLists(1, lists);
    HrCheck(queue->Signal(c.fence.Get(), ++c.fenceValue), "Signal");
    if (c.fence->GetCompletedValue() < c.fenceValue)
    {
        HrCheck(c.fence->SetEventOnCompletion(c.fenceValue, c.fenceEvent), "SetEventOnCompletion");
        WaitForSingleObject(c.fenceEvent, 30000);
    }
    c.list.Reset();
    c.alloc.Reset();
}

ComPtr<ID3D12Resource> CreateTex2D(ID3D12Device* dev, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = kWidth;
    desc.Height = kHeight;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = flags;
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    ComPtr<ID3D12Resource> tex;
    HrCheck(dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&tex)),
        "CreateCommittedResource");
    return tex;
}

void UploadTex2D(ID3D12Device* dev, ID3D12CommandQueue* queue, CmdCtx& c,
    ID3D12Resource* dst, const void* data, uint32_t rowBytes)
{
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT numRows = 0;
    UINT64 rowSize = 0, totalSize = 0;
    D3D12_RESOURCE_DESC desc = dst->GetDesc();
    dev->GetCopyableFootprints(&desc, 0, 1, 0, &fp, &numRows, &rowSize, &totalSize);

    D3D12_HEAP_PROPERTIES upHeap{D3D12_HEAP_TYPE_UPLOAD};
    D3D12_RESOURCE_DESC bufDesc{};
    bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufDesc.Width = totalSize;
    bufDesc.Height = 1;
    bufDesc.DepthOrArraySize = 1;
    bufDesc.MipLevels = 1;
    bufDesc.SampleDesc.Count = 1;
    bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> upload;
    HrCheck(dev->CreateCommittedResource(&upHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload)), "upload buffer");

    void* mapped = nullptr;
    HrCheck(upload->Map(0, nullptr, &mapped), "Map(upload)");
    for (UINT r = 0; r < numRows; ++r)
        std::memcpy(static_cast<uint8_t*>(mapped) + static_cast<size_t>(r) * fp.Footprint.RowPitch,
            static_cast<const uint8_t*>(data) + static_cast<size_t>(r) * rowBytes, rowBytes);
    upload->Unmap(0, nullptr);

    CmdBegin(dev, c);
    D3D12_RESOURCE_BARRIER toCopy{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION};
    toCopy.Transition = {dst, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST};
    c.list->ResourceBarrier(1, &toCopy);
    D3D12_TEXTURE_COPY_LOCATION dstLoc{dst, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    dstLoc.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION srcLoc{upload.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    srcLoc.PlacedFootprint = fp;
    c.list->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
    D3D12_RESOURCE_BARRIER toCommon = toCopy;
    toCommon.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    toCommon.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    c.list->ResourceBarrier(1, &toCommon);
    CmdSubmitWait(queue, c);
}

int Run(const std::string& dllPath, const std::string& outDir, bool iatPatch, bool spy)
{
    // -- Load plugin + IAT patch + D3D12 exports ------------------------------
    HMODULE dll = ::LoadLibraryA(dllPath.c_str());
    if (!dll)
    {
        Log("[NGX] LoadLibrary(%s) failed, err %lu", dllPath.c_str(), ::GetLastError());
        return 1;
    }
    Log("[NGX] LoadLibrary(%s) OK", dllPath.c_str());
    if (iatPatch)
        PatchIatByName(dll, "GetModuleFileNameW", reinterpret_cast<void*>(&GetModuleFileNameW_Stub));
    if (spy)
        InstallSpy(dll);

    Ngx12Api ngx;
    bool ok = true;
    ok &= LoadProc(dll, "NVSDK_NGX_D3D12_Init_Ext", ngx.InitExt, false);
    ok &= LoadProc(dll, "NVSDK_NGX_D3D12_Init", ngx.Init, false);
    ok &= LoadProc(dll, "NVSDK_NGX_D3D12_PopulateParameters_Impl", ngx.PopulateParameters, false);
    ok &= LoadProc(dll, "NVSDK_NGX_D3D12_GetFeatureRequirements", ngx.GetFeatureRequirements, false);
    ok &= LoadProc(dll, "NVSDK_NGX_D3D12_CreateFeature", ngx.CreateFeature);
    ok &= LoadProc(dll, "NVSDK_NGX_D3D12_EvaluateFeature", ngx.EvaluateFeature);
    ok &= LoadProc(dll, "NVSDK_NGX_D3D12_ReleaseFeature", ngx.ReleaseFeature);
    ok &= LoadProc(dll, "NVSDK_NGX_D3D12_Shutdown1", ngx.Shutdown1);
    if (!ok || (!ngx.InitExt && !ngx.Init))
    {
        Log("FATAL: missing D3D12 NGX exports");
        return 1;
    }

    const std::wstring outDirW = ToWide(std::filesystem::absolute(outDir).string());

    // -- Device on the NVIDIA adapter -----------------------------------------
    ComPtr<IDXGIFactory6> factory;
    HrCheck(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0;; ++i)
    {
        ComPtr<IDXGIAdapter1> a;
        if (factory->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND)
            break;
        DXGI_ADAPTER_DESC1 d{};
        a->GetDesc1(&d);
        Log("[D3D12] adapter %u: %ls (vendor 0x%04X)", i, d.Description, d.VendorId);
        if (d.VendorId == 0x10DE && !adapter)
            adapter = a;
    }
    if (!adapter)
    {
        Log("FATAL: no NVIDIA adapter");
        return 1;
    }
    ComPtr<ID3D12Device> dev;
    HrCheck(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)), "D3D12CreateDevice");
    D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
    ComPtr<ID3D12CommandQueue> queue;
    HrCheck(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)), "CreateCommandQueue");

    CmdCtx cmd;
    HrCheck(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&cmd.fence)), "CreateFence");
    cmd.fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    // -- Requirements + init ----------------------------------------------------
    NVSDK_NGX_FeatureDiscoveryInfo discovery{};
    discovery.SDKVersion = NVSDK_NGX_Version_API;
    discovery.FeatureID = kFeatureId;
    discovery.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Application_Id;
    discovery.Identifier.v.ApplicationId = kAppId;
    discovery.ApplicationDataPath = outDirW.c_str();

    if (ngx.GetFeatureRequirements)
    {
        NVSDK_NGX_FeatureRequirement req{};
        NVSDK_NGX_Result r = NgxCall("D3D12_GetFeatureRequirements(18)", [&] {
            return ngx.GetFeatureRequirements(adapter.Get(), &discovery, &req);
        });
        if (r == NVSDK_NGX_Result_Success)
            Log("[NGX] feature 18 FeatureSupported = %u, MinHWArch = 0x%08X, MinOS = %s",
                static_cast<unsigned>(req.FeatureSupported), req.MinHWArchitecture, req.MinOSVersion);
    }

    bool inited = false;
    SpikeParameterMap initParams;
    NVSDK_NGX_Parameter* initParamsPtr = nullptr;
    if (ngx.PopulateParameters)
    {
        NVSDK_NGX_Result r = NgxCall("D3D12_PopulateParameters_Impl", [&] {
            return ngx.PopulateParameters(&initParams);
        });
        if (r == NVSDK_NGX_Result_Success)
            initParamsPtr = &initParams;
    }
    if (ngx.InitExt)
    {
        NVSDK_NGX_Result r = NgxCall("D3D12_Init_Ext(appId=0x1000000)", [&] {
            return ngx.InitExt(kAppId, outDirW.c_str(), dev.Get(), NVSDK_NGX_Version_API, initParamsPtr);
        });
        inited = (r == NVSDK_NGX_Result_Success);
    }
    if (!inited && ngx.Init)
    {
        NVSDK_NGX_Result r = NgxCall("D3D12_Init(appId=0x1000000)", [&] {
            return ngx.Init(kAppId, outDirW.c_str(), dev.Get(), NVSDK_NGX_Version_API);
        });
        inited = (r == NVSDK_NGX_Result_Success);
    }
    if (!inited)
    {
        Log("FATAL: D3D12 NGX init failed");
        return 1;
    }

    // -- CreateFeature(18) with parameter-set candidates ------------------------
    NVSDK_NGX_Handle* feature = nullptr;
    const char* groupNames[] = {
        "base Width/Height/OutWidth/OutHeight",
        "base + DLSSNR.{Width,Height,InputWidth,InputHeight,OutputWidth,OutputHeight,Enabled,ScalingRatio}",
        "DLSSNR.* only",
        "empty parameters",
    };
    for (int group = 0; group < 4 && !feature; ++group)
    {
        Log("== CreateFeature attempt %d: %s", group, groupNames[group]);
        SpikeParameterMap params;
        if (group == 0 || group == 1)
        {
            params.Set("Width", static_cast<unsigned int>(kWidth));
            params.Set("Height", static_cast<unsigned int>(kHeight));
            params.Set("OutWidth", static_cast<unsigned int>(kWidth));
            params.Set("OutHeight", static_cast<unsigned int>(kHeight));
        }
        if (group == 1 || group == 2)
        {
            params.Set("DLSSNR.Width", static_cast<unsigned int>(kWidth));
            params.Set("DLSSNR.Height", static_cast<unsigned int>(kHeight));
            params.Set("DLSSNR.InputWidth", static_cast<unsigned int>(kWidth));
            params.Set("DLSSNR.InputHeight", static_cast<unsigned int>(kHeight));
            params.Set("DLSSNR.OutputWidth", static_cast<unsigned int>(kWidth));
            params.Set("DLSSNR.OutputHeight", static_cast<unsigned int>(kHeight));
            params.Set("DLSSNR.Enabled", 1);
            params.Set("DLSSNR.ScalingRatio", 1.0f);
        }

        CmdBegin(dev.Get(), cmd);
        NVSDK_NGX_Result r = NgxCall("D3D12_CreateFeature(18)", [&] {
            return ngx.CreateFeature(cmd.list.Get(), kFeatureId, &params, &feature);
        });
        if (cmd.list)
            CmdSubmitWait(queue.Get(), cmd);
        if (r == NVSDK_NGX_Result_Success && feature)
            Log("[NGX] feature 18 created with group %d (%s), handle %p",
                group, groupNames[group], static_cast<void*>(feature));
        else
            feature = nullptr;
    }
    if (!feature)
    {
        Log("FATAL: CreateFeature failed for all parameter sets");
        return 1;
    }

    // -- Textures + upload -------------------------------------------------------
    const D3D12_RESOURCE_FLAGS rwFlags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    ComPtr<ID3D12Resource> texColor = CreateTex2D(dev.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT, rwFlags);
    ComPtr<ID3D12Resource> texBackbuffer = CreateTex2D(dev.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT, rwFlags);
    ComPtr<ID3D12Resource> texOutput = CreateTex2D(dev.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT, rwFlags);
    ComPtr<ID3D12Resource> texMotion = CreateTex2D(dev.Get(), DXGI_FORMAT_R16G16_FLOAT, rwFlags);
    ComPtr<ID3D12Resource> texDepth = CreateTex2D(dev.Get(), DXGI_FORMAT_R32_FLOAT, rwFlags);
    Log("[D3D12] textures created (%ux%u)", kWidth, kHeight);

    std::vector<uint16_t> colorPix(static_cast<size_t>(kWidth) * kHeight * 4);
    FillColorPattern(colorPix);
    std::vector<float> depthPix(static_cast<size_t>(kWidth) * kHeight, 0.5f);
    std::vector<uint16_t> motionPix(static_cast<size_t>(kWidth) * kHeight * 2, 0);

    UploadTex2D(dev.Get(), queue.Get(), cmd, texColor.Get(), colorPix.data(), kWidth * 8);
    UploadTex2D(dev.Get(), queue.Get(), cmd, texBackbuffer.Get(), colorPix.data(), kWidth * 8);
    UploadTex2D(dev.Get(), queue.Get(), cmd, texMotion.Get(), motionPix.data(), kWidth * 4);
    UploadTex2D(dev.Get(), queue.Get(), cmd, texDepth.Get(), depthPix.data(), kWidth * 4);
    Log("[D3D12] inputs uploaded");

    WritePng(outDir + "/input_color.png", colorPix);
    PrintStats("input color", colorPix, nullptr);

    // -- Evaluate ------------------------------------------------------------
    {
        SpikeParameterMap params;
        params.Set("DLSSNR.Color", texColor.Get());
        params.Set("DLSSNR.Backbuffer", texBackbuffer.Get());
        params.Set("DLSSNR.MVec", texMotion.Get());
        params.Set("DLSSNR.Depth", texDepth.Get());
        params.Set("DLSSNR.Output", texOutput.Get());
        params.Set("DLSSNR.Intensity", 1.0f);
        params.Set("DLSSNR.Reset", 1);
        params.Set("DLSSNR.Enabled", 1);
        params.Set("DLSSNR.MVecScaleX", static_cast<float>(kWidth));
        params.Set("DLSSNR.MVecScaleY", static_cast<float>(kHeight));
        params.Set("DLSSNR.DepthInverted", 0);
        params.Set("Width", static_cast<unsigned int>(kWidth));
        params.Set("Height", static_cast<unsigned int>(kHeight));

        CmdBegin(dev.Get(), cmd);
        NVSDK_NGX_Result r = NgxCall("D3D12_EvaluateFeature(18)", [&] {
            return ngx.EvaluateFeature(cmd.list.Get(), feature, &params, nullptr);
        });
        if (cmd.list)
            CmdSubmitWait(queue.Get(), cmd);
        if (r != NVSDK_NGX_Result_Success)
            Log("[NGX] EvaluateFeature did not succeed -- output may be untouched");
    }

    // -- Readback + stats + PNG -------------------------------------------------
    std::vector<uint16_t> outPix(static_cast<size_t>(kWidth) * kHeight * 4, 0);
    {
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        UINT numRows = 0;
        UINT64 rowSize = 0, totalSize = 0;
        D3D12_RESOURCE_DESC desc = texOutput->GetDesc();
        dev->GetCopyableFootprints(&desc, 0, 1, 0, &fp, &numRows, &rowSize, &totalSize);
        D3D12_HEAP_PROPERTIES rbHeap{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC bufDesc{};
        bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bufDesc.Width = totalSize;
        bufDesc.Height = 1;
        bufDesc.DepthOrArraySize = 1;
        bufDesc.MipLevels = 1;
        bufDesc.SampleDesc.Count = 1;
        bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> readback;
        HrCheck(dev->CreateCommittedResource(&rbHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)), "readback buffer");

        CmdBegin(dev.Get(), cmd);
        D3D12_RESOURCE_BARRIER toSrc{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION};
        toSrc.Transition = {texOutput.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE};
        cmd.list->ResourceBarrier(1, &toSrc);
        D3D12_TEXTURE_COPY_LOCATION srcLoc{texOutput.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        srcLoc.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION dstLoc{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        dstLoc.PlacedFootprint = fp;
        cmd.list->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
        CmdSubmitWait(queue.Get(), cmd);

        void* mapped = nullptr;
        HrCheck(readback->Map(0, nullptr, &mapped), "Map(readback)");
        for (UINT r = 0; r < numRows; ++r)
            std::memcpy(outPix.data() + static_cast<size_t>(r) * kWidth * 4,
                static_cast<const uint8_t*>(mapped) + static_cast<size_t>(r) * fp.Footprint.RowPitch,
                kWidth * 8);
        readback->Unmap(0, nullptr);
    }
    PrintStats("DLSSNR output", outPix, &colorPix);
    const std::string pngPath = outDir + "/dlssnr_output.png";
    if (WritePng(pngPath, outPix))
        Log("[OUT] wrote %s", pngPath.c_str());
    else
        Log("[OUT] FAILED to write %s", pngPath.c_str());

    // -- Teardown ---------------------------------------------------------------
    NgxCall("D3D12_ReleaseFeature", [&] { return ngx.ReleaseFeature(feature); });
    NgxCall("D3D12_Shutdown1", [&] { return ngx.Shutdown1(dev.Get()); });
    CloseHandle(cmd.fenceEvent);
    ::FreeLibrary(dll);
    Log("== dlss5_spike (d3d12) done ==");
    return 0;
}

} // namespace d3d12mode

// ===========================================================================
// Streamline probe mode (--api slprobe)
//
// Direct plugin Init fails with 0xBAD00002 in both Vulkan and D3D12 modes:
// the spy run shows the plugin decides this internally (it expects runtime
// state that only its host loader - NGX core or Streamline - sets up). The
// shipping driver NGX core does not know feature 18, which leaves the
// Streamline path (sl.interposer + sl.dlss_nr.dll, what the renodx-dlss5
// add-on stack uses). SL 2.13 is not public, so DLSS NR's sl::Feature id is
// unknown; this mode brute-forces candidate ids through the public
// sl.interposer API and reports which ones load.
// ===========================================================================
namespace slprobe
{

int Run(const std::string& outDir, const std::string& dllPath)
{
    // Stage the requested nvngx_dlssnr.dll build next to the exe - sl.dlss_nr
    // loads it from the application directory.
    char exeDirBuf0[MAX_PATH]{};
    ::GetModuleFileNameA(nullptr, exeDirBuf0, MAX_PATH);
    const std::filesystem::path pluginSrc = std::filesystem::absolute(dllPath);
    const std::filesystem::path pluginDst = std::filesystem::path(exeDirBuf0).parent_path() / "nvngx_dlssnr.dll";
    std::error_code ec;
    if (!std::filesystem::equivalent(pluginSrc, pluginDst, ec))
    {
        std::filesystem::copy_file(pluginSrc, pluginDst, std::filesystem::copy_options::overwrite_existing, ec);
        if (ec)
        {
            Log("FATAL: cannot stage plugin (%s)", ec.message().c_str());
            return 1;
        }
    }
    Log("[SL] plugin staged: %s", pluginDst.string().c_str());

    HMODULE sl = ::LoadLibraryA("sl.interposer.dll");
    if (!sl)
    {
        Log("[SL] LoadLibrary(sl.interposer.dll) failed, err %lu - put SL 2.13 DLLs next to the exe", ::GetLastError());
        return 1;
    }
    Log("[SL] sl.interposer.dll loaded");

    auto init = reinterpret_cast<PFun_slInit*>(::GetProcAddress(sl, "slInit"));
    auto shutdown = reinterpret_cast<PFun_slShutdown*>(::GetProcAddress(sl, "slShutdown"));
    auto isLoaded = reinterpret_cast<PFun_slIsFeatureLoaded*>(::GetProcAddress(sl, "slIsFeatureLoaded"));
    auto setLoaded = reinterpret_cast<PFun_slSetFeatureLoaded*>(::GetProcAddress(sl, "slSetFeatureLoaded"));
    auto getVersion = reinterpret_cast<PFun_slGetFeatureVersion*>(::GetProcAddress(sl, "slGetFeatureVersion"));
    auto setD3D = reinterpret_cast<PFun_slSetD3DDevice*>(::GetProcAddress(sl, "slSetD3DDevice"));
    if (!init || !shutdown || !isLoaded || !setLoaded || !getVersion)
    {
        Log("FATAL: missing sl exports");
        return 1;
    }

    char exeDirBuf[MAX_PATH]{};
    ::GetModuleFileNameA(nullptr, exeDirBuf, MAX_PATH);
    const std::wstring pluginDir = ToWide(std::filesystem::path(exeDirBuf).parent_path().string());
    const std::wstring outDirW = ToWide(std::filesystem::absolute(outDir).string());
    const wchar_t* pluginPaths[] = {pluginDir.c_str()};

    sl::Preferences pref{};
    pref.showConsole = false;
    pref.logLevel = sl::LogLevel::eVerbose;
    pref.pathsToPlugins = pluginPaths;
    pref.numPathsToPlugins = 1;
    pref.pathToLogsAndData = outDirW.c_str();
    pref.flags = sl::PreferenceFlags::eDisableCLStateTracking |
        sl::PreferenceFlags::eAllowOTA |
        sl::PreferenceFlags::eLoadDownloadedPlugins |
        sl::PreferenceFlags::eDisableDebugText;
    pref.featuresToLoad = nullptr;
    pref.numFeaturesToLoad = 0;
    pref.applicationId = 0;
    pref.engine = sl::EngineType::eCustom;
    pref.engineVersion = "dlss5_spike 1.0";
    pref.projectId = "a0f57b54-1daf-4934-90ae-c4035c19df04";
    pref.renderAPI = sl::RenderAPI::eD3D12;

    sl::Result r = init(pref, sl::kSDKVersion);
    Log("[SL] slInit -> %d", static_cast<int>(r));
    if (r != sl::Result::eOk)
        return 1;

    // A D3D12 device is needed before features can be loaded.
    if (setD3D)
    {
        Microsoft::WRL::ComPtr<IDXGIFactory6> factory;
        d3d12mode::HrCheck(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        for (UINT i = 0; factory->EnumAdapters1(i, adapter.ReleaseAndGetAddressOf()) != DXGI_ERROR_NOT_FOUND; ++i)
        {
            DXGI_ADAPTER_DESC1 d{};
            adapter->GetDesc1(&d);
            if (d.VendorId == 0x10DE)
                break;
            adapter.Reset();
        }
        Microsoft::WRL::ComPtr<ID3D12Device> dev;
        d3d12mode::HrCheck(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)), "D3D12CreateDevice");
        sl::Result rs = setD3D(dev.Get());
        Log("[SL] slSetD3DDevice -> %d", static_cast<int>(rs));
    }

    // Sanity: plain DLSS (feature 0) through the same probe - if THIS fails
    // too the probe environment is at fault, not the feature.
    {
        bool dlssLoaded = false;
        sl::Result rs = setLoaded(0, true);
        sl::Result rl = isLoaded(0, dlssLoaded);
        Log("[SL] sanity kFeatureDLSS(0): setLoaded -> %d, isLoaded -> %d loaded=%d",
            static_cast<int>(rs), static_cast<int>(rl), dlssLoaded ? 1 : 0);
        if (dlssLoaded)
        {
            sl::FeatureVersion ver{};
            if (getVersion(0, ver) == sl::Result::eOk)
                Log("[SL] kFeatureDLSS version SL %u.%u.%u NGX %u.%u.%u",
                    ver.versionSL.major, ver.versionSL.minor, ver.versionSL.build,
                    ver.versionNGX.major, ver.versionNGX.minor, ver.versionNGX.build);
            setLoaded(0, false);
        }
    }

    const sl::Feature candidates[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 9999,
    };
    for (sl::Feature f : candidates)
    {
        sl::Result rs = setLoaded(f, true);
        if (rs != sl::Result::eOk)
        {
            Log("[SL] feature %5u: slSetFeatureLoaded -> %d", f, static_cast<int>(rs));
            continue;
        }
        bool loaded = false;
        sl::Result rl = isLoaded(f, loaded);
        Log("[SL] feature %5u: setLoaded ok, isLoaded -> %d loaded=%d", f, static_cast<int>(rl), loaded ? 1 : 0);
        if (loaded)
        {
            sl::FeatureVersion ver{};
            sl::Result rv = getVersion(f, ver);
            if (rv == sl::Result::eOk)
                Log("[SL] feature %5u: version SL %u.%u.%u NGX %u.%u.%u", f,
                    ver.versionSL.major, ver.versionSL.minor, ver.versionSL.build,
                    ver.versionNGX.major, ver.versionNGX.minor, ver.versionNGX.build);
        }
    }

    sl::Result rs = shutdown();
    Log("[SL] slShutdown -> %d", static_cast<int>(rs));
    ::FreeLibrary(sl);
    Log("== dlss5_spike (slprobe) done ==");
    return 0;
}

} // namespace slprobe

int main(int argc, char** argv)
{
    std::string dllPath = R"(D:\Code\SL 2.13\nvngx_dlssnr.dll)";
    std::string outDir = "spike_dlss5_out";
    std::string api = "vulkan";
    bool iatPatch = true;
    bool spy = false;
    for (int i = 1; i < argc; ++i)
    {
        if (!std::strcmp(argv[i], "--dll") && i + 1 < argc)
            dllPath = argv[++i];
        else if (!std::strcmp(argv[i], "--out") && i + 1 < argc)
            outDir = argv[++i];
        else if (!std::strcmp(argv[i], "--api") && i + 1 < argc)
            api = argv[++i];
        else if (!std::strcmp(argv[i], "--no-iat-patch"))
            iatPatch = false;
        else if (!std::strcmp(argv[i], "--spy"))
            spy = true;
        else
            Log("unknown argument: %s", argv[i]);
    }
    std::filesystem::create_directories(outDir);
    Log("== dlss5_spike == api=%s dll=%s out=%s iat-patch=%d spy=%d", api.c_str(), dllPath.c_str(), outDir.c_str(), iatPatch ? 1 : 0, spy ? 1 : 0);

    if (api == "d3d12")
        return d3d12mode::Run(dllPath, outDir, iatPatch, spy);
    if (api == "slprobe")
        return slprobe::Run(outDir, dllPath);
    if (api != "vulkan")
    {
        Log("FATAL: unknown --api %s (expected vulkan|d3d12)", api.c_str());
        return 1;
    }

    // -- 1. Load nvngx_dlssnr.dll, patch its IAT, resolve NGX entry points ---
    NgxApi ngx;
    if (!LoadNgx(dllPath, ngx))
    {
        Log("FATAL: failed to load NGX API from %s", dllPath.c_str());
        return 1;
    }
    if (iatPatch)
        PatchIatByName(ngx.dll, "GetModuleFileNameW", reinterpret_cast<void*>(&GetModuleFileNameW_Stub));
    if (spy)
        InstallSpy(ngx.dll);

    const std::wstring outDirW = ToWide(std::filesystem::absolute(outDir).string());

    NVSDK_NGX_FeatureDiscoveryInfo discovery{};
    discovery.SDKVersion = NVSDK_NGX_Version_API;
    discovery.FeatureID = kFeatureId;
    discovery.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Application_Id;
    discovery.Identifier.v.ApplicationId = kAppId;
    discovery.ApplicationDataPath = outDirW.c_str();
    discovery.FeatureInfo = nullptr;

    // -- 2. Required Vulkan extensions (before instance/device creation) ----
    std::vector<std::string> instExtNames;
    {
        uint32_t count = 0;
        VkExtensionProperties* props = nullptr;
        NVSDK_NGX_Result r = NgxCall("GetFeatureInstanceExtensionRequirements", [&] {
            return ngx.GetInstanceExts(&discovery, &count, &props);
        });
        if (r == NVSDK_NGX_Result_Success)
        {
            Log("[NGX] feature 18 requires %u instance extension(s):", count);
            for (uint32_t i = 0; i < count; ++i)
            {
                Log("      %s (spec %u)", props[i].extensionName, props[i].specVersion);
                instExtNames.push_back(props[i].extensionName);
            }
        }
    }

    // Create instance (enable NGX-required extensions that actually exist).
    VulkanCtx ctx;
    {
        uint32_t availCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &availCount, nullptr);
        std::vector<VkExtensionProperties> avail(availCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &availCount, avail.data());

        std::vector<const char*> enabled;
        for (const std::string& want : instExtNames)
        {
            bool found = false;
            for (const auto& a : avail)
                if (want == a.extensionName) { found = true; break; }
            if (found)
                enabled.push_back(want.c_str());
            else
                Log("[VK] NGX-required instance extension %s NOT available, skipping", want.c_str());
        }

        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "dlss5_spike";
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        ici.enabledExtensionCount = static_cast<uint32_t>(enabled.size());
        ici.ppEnabledExtensionNames = enabled.data();
        VkCheck(vkCreateInstance(&ici, nullptr, &ctx.instance), "vkCreateInstance");
        Log("[VK] instance created with %u extension(s)", static_cast<unsigned>(enabled.size()));
    }

    // Pick an NVIDIA physical device.
    {
        uint32_t n = 0;
        vkEnumeratePhysicalDevices(ctx.instance, &n, nullptr);
        std::vector<VkPhysicalDevice> devs(n);
        vkEnumeratePhysicalDevices(ctx.instance, &n, devs.data());
        for (VkPhysicalDevice d : devs)
        {
            VkPhysicalDeviceProperties p{};
            vkGetPhysicalDeviceProperties(d, &p);
            Log("[VK] physical device: %s (vendor 0x%04X)", p.deviceName, p.vendorID);
            if (p.vendorID == 0x10DE && !ctx.phys)
                ctx.phys = d;
        }
        if (!ctx.phys)
        {
            Log("FATAL: no NVIDIA GPU found");
            return 1;
        }
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(ctx.phys, &p);
        Log("[VK] using: %s (driver %u.%u.%u)", p.deviceName,
            (p.driverVersion >> 22) & 0x3FF, (p.driverVersion >> 14) & 0xFF, p.driverVersion & 0x3FFF);
        vkGetPhysicalDeviceMemoryProperties(ctx.phys, &ctx.memProps);
    }

    // Required device extensions (with a known-good fallback list).
    std::vector<std::string> devExtNames;
    {
        uint32_t count = 0;
        VkExtensionProperties* props = nullptr;
        NVSDK_NGX_Result r = NgxCall("GetFeatureDeviceExtensionRequirements", [&] {
            return ngx.GetDeviceExts(ctx.instance, ctx.phys, &discovery, &count, &props);
        });
        if (r == NVSDK_NGX_Result_Success)
        {
            Log("[NGX] feature 18 requires %u device extension(s):", count);
            for (uint32_t i = 0; i < count; ++i)
            {
                Log("      %s (spec %u)", props[i].extensionName, props[i].specVersion);
                devExtNames.push_back(props[i].extensionName);
            }
        }
        else
        {
            // The query itself can fail, but NGX still needs these to run -
            // the list below is what the query returns when it succeeds.
            Log("[NGX] device extension query failed; falling back to known NGX-required set");
            devExtNames = {
                VK_NVX_BINARY_IMPORT_EXTENSION_NAME,
                VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME,
                VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
                VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
            };
        }
    }

    // Create device + queue.
    {
        uint32_t famCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(ctx.phys, &famCount, nullptr);
        std::vector<VkQueueFamilyProperties> fams(famCount);
        vkGetPhysicalDeviceQueueFamilyProperties(ctx.phys, &famCount, fams.data());
        bool found = false;
        for (uint32_t i = 0; i < famCount; ++i)
        {
            if ((fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT))
            {
                ctx.queueFamily = i;
                found = true;
                break;
            }
        }
        if (!found)
        {
            Log("FATAL: no graphics+compute queue family");
            return 1;
        }

        uint32_t availCount = 0;
        vkEnumerateDeviceExtensionProperties(ctx.phys, nullptr, &availCount, nullptr);
        std::vector<VkExtensionProperties> avail(availCount);
        vkEnumerateDeviceExtensionProperties(ctx.phys, nullptr, &availCount, avail.data());
        std::vector<const char*> enabled;
        for (const std::string& want : devExtNames)
        {
            bool extFound = false;
            for (const auto& a : avail)
                if (want == a.extensionName) { extFound = true; break; }
            if (extFound)
                enabled.push_back(want.c_str());
            else
                Log("[VK] NGX-required device extension %s NOT available, skipping", want.c_str());
        }

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = ctx.queueFamily;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = static_cast<uint32_t>(enabled.size());
        dci.ppEnabledExtensionNames = enabled.data();
        VkCheck(vkCreateDevice(ctx.phys, &dci, nullptr, &ctx.device), "vkCreateDevice");
        vkGetDeviceQueue(ctx.device, ctx.queueFamily, 0, &ctx.queue);
        Log("[VK] device created with %u extension(s), queue family %u",
            static_cast<unsigned>(enabled.size()), ctx.queueFamily);
    }

    // Command pool / buffer / fence.
    {
        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = ctx.queueFamily;
        VkCheck(vkCreateCommandPool(ctx.device, &cpci, nullptr, &ctx.cmdPool), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = ctx.cmdPool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCheck(vkAllocateCommandBuffers(ctx.device, &cbai, &ctx.cmd), "vkAllocateCommandBuffers");
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkCheck(vkCreateFence(ctx.device, &fci, nullptr, &ctx.fence), "vkCreateFence");
    }

    // -- 3. Feature requirements + NGX init ----------------------------------
    {
        NVSDK_NGX_FeatureRequirement req{};
        NVSDK_NGX_Result r = NgxCall("GetFeatureRequirements(feature 18)", [&] {
            return ngx.GetFeatureRequirements(ctx.instance, ctx.phys, &discovery, &req);
        });
        if (r == NVSDK_NGX_Result_Success)
        {
            Log("[NGX] feature 18 FeatureSupported = %u (0 = supported)", static_cast<unsigned>(req.FeatureSupported));
            Log("[NGX] feature 18 MinHWArchitecture = 0x%08X, MinOSVersion = %s",
                req.MinHWArchitecture, req.MinOSVersion);
        }
        else
        {
            Log("[NGX] GetFeatureRequirements failed -- continuing anyway (direct-plugin path)");
        }
    }

    // Direct-plugin init (snippet ABI; app-allocated parameter maps).
    bool ngxInitialized = false;
    if (ngx.InitExt2)
    {
        NVSDK_NGX_Result r = NgxCall("Init_Ext2(appId=0x1000000)", [&] {
            return ngx.InitExt2(kAppId, outDirW.c_str(), ctx.instance, ctx.phys, ctx.device,
                vkGetInstanceProcAddr, vkGetDeviceProcAddr, NVSDK_NGX_Version_API, nullptr);
        });
        ngxInitialized = (r == NVSDK_NGX_Result_Success);
    }
    if (!ngxInitialized && ngx.InitExt)
    {
        NVSDK_NGX_Result r = NgxCall("Init_Ext(appId=0x1000000)", [&] {
            return ngx.InitExt(kAppId, outDirW.c_str(), ctx.instance, ctx.phys, ctx.device,
                NVSDK_NGX_Version_API, nullptr);
        });
        ngxInitialized = (r == NVSDK_NGX_Result_Success);
    }
    if (!ngxInitialized && ngx.Init)
    {
        NVSDK_NGX_Result r = NgxCall("Init(appId=0x1000000)", [&] {
            return ngx.Init(kAppId, outDirW.c_str(), ctx.instance, ctx.phys, ctx.device,
                NVSDK_NGX_Version_API);
        });
        ngxInitialized = (r == NVSDK_NGX_Result_Success);
    }
    if (!ngxInitialized)
    {
        Log("FATAL: NGX init failed (Init_Ext2 / Init_Ext / Init)");
        return 1;
    }

    // -- 4. CreateFeature1 with several parameter-set candidates -------------
    NVSDK_NGX_Handle* feature = nullptr;
    const char* groupNames[] = {
        "base Width/Height/OutWidth/OutHeight",
        "base + DLSSNR.{Width,Height,Enabled,ScalingRatio}",
        "DLSSNR.* only",
        "empty parameters",
    };
    for (int group = 0; group < 4 && !feature; ++group)
    {
        Log("== CreateFeature1 attempt %d: %s", group, groupNames[group]);
        SpikeParameterMap params;
        if (group == 0 || group == 1)
        {
            params.Set("Width", static_cast<unsigned int>(kWidth));
            params.Set("Height", static_cast<unsigned int>(kHeight));
            params.Set("OutWidth", static_cast<unsigned int>(kWidth));
            params.Set("OutHeight", static_cast<unsigned int>(kHeight));
        }
        if (group == 1 || group == 2)
        {
            params.Set("DLSSNR.Width", static_cast<unsigned int>(kWidth));
            params.Set("DLSSNR.Height", static_cast<unsigned int>(kHeight));
            params.Set("DLSSNR.Enabled", 1);
            params.Set("DLSSNR.ScalingRatio", 1.0f);
        }

        BeginCmd(ctx);
        NVSDK_NGX_Result r = NgxCall("CreateFeature1(feature 18)", [&] {
            return ngx.CreateFeature1(ctx.device, ctx.cmd, kFeatureId, &params, &feature);
        });
        SubmitWait(ctx);
        if (r == NVSDK_NGX_Result_Success && feature)
            Log("[NGX] feature 18 created with group %d (%s), handle %p",
                group, groupNames[group], static_cast<void*>(feature));
        else
            feature = nullptr;
    }
    if (!feature)
    {
        Log("FATAL: CreateFeature1 failed for all parameter sets");
        return 1;
    }

    // -- 5. Test images -------------------------------------------------------
    const VkImageUsageFlags colorUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    GpuImage imgColor = CreateGpuImage(ctx, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, colorUsage);
    GpuImage imgBackbuffer = CreateGpuImage(ctx, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, colorUsage);
    GpuImage imgOutput = CreateGpuImage(ctx, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, colorUsage);
    GpuImage imgMotion = CreateGpuImage(ctx, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, colorUsage);
    GpuImage imgDepth = CreateGpuImage(ctx, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    Log("[VK] test images created (%ux%u)", kWidth, kHeight);

    // Host-side patterns.
    std::vector<uint16_t> colorPix(static_cast<size_t>(kWidth) * kHeight * 4);
    FillColorPattern(colorPix);
    std::vector<float> depthPix(static_cast<size_t>(kWidth) * kHeight, 0.5f);
    std::vector<uint16_t> motionPix(static_cast<size_t>(kWidth) * kHeight * 2, 0);

    StagingBuffer stColor = CreateStaging(ctx, colorPix.size() * sizeof(uint16_t));
    StagingBuffer stDepth = CreateStaging(ctx, depthPix.size() * sizeof(float));
    StagingBuffer stMotion = CreateStaging(ctx, motionPix.size() * sizeof(uint16_t));
    std::memcpy(stColor.mapped, colorPix.data(), colorPix.size() * sizeof(uint16_t));
    std::memcpy(stDepth.mapped, depthPix.data(), depthPix.size() * sizeof(float));
    std::memcpy(stMotion.mapped, motionPix.data(), motionPix.size() * sizeof(uint16_t));

    // Transition everything to GENERAL, upload inputs.
    BeginCmd(ctx);
    TransitionToGeneral(ctx, imgColor.image, VK_IMAGE_ASPECT_COLOR_BIT);
    TransitionToGeneral(ctx, imgBackbuffer.image, VK_IMAGE_ASPECT_COLOR_BIT);
    TransitionToGeneral(ctx, imgOutput.image, VK_IMAGE_ASPECT_COLOR_BIT);
    TransitionToGeneral(ctx, imgMotion.image, VK_IMAGE_ASPECT_COLOR_BIT);
    TransitionToGeneral(ctx, imgDepth.image, VK_IMAGE_ASPECT_DEPTH_BIT);
    CopyBufferToImage(ctx, stColor.buffer, imgColor.image, VK_IMAGE_ASPECT_COLOR_BIT);
    CopyBufferToImage(ctx, stColor.buffer, imgBackbuffer.image, VK_IMAGE_ASPECT_COLOR_BIT);
    CopyBufferToImage(ctx, stDepth.buffer, imgDepth.image, VK_IMAGE_ASPECT_DEPTH_BIT);
    CopyBufferToImage(ctx, stMotion.buffer, imgMotion.image, VK_IMAGE_ASPECT_COLOR_BIT);
    SubmitWait(ctx);
    Log("[VK] inputs uploaded, all images in GENERAL");

    WritePng(outDir + "/input_color.png", colorPix);
    PrintStats("input color", colorPix, nullptr);

    // -- 6. EvaluateFeature ---------------------------------------------------
    {
        SpikeParameterMap params;

        VkImageSubresourceRange colorRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageSubresourceRange depthRange{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        NVSDK_NGX_Resource_VK resColor = NVSDK_NGX_Create_ImageView_Resource_VK(
            imgColor.view, imgColor.image, colorRange, imgColor.format, kWidth, kHeight, false);
        NVSDK_NGX_Resource_VK resBackbuffer = NVSDK_NGX_Create_ImageView_Resource_VK(
            imgBackbuffer.view, imgBackbuffer.image, colorRange, imgBackbuffer.format, kWidth, kHeight, false);
        NVSDK_NGX_Resource_VK resMotion = NVSDK_NGX_Create_ImageView_Resource_VK(
            imgMotion.view, imgMotion.image, colorRange, imgMotion.format, kWidth, kHeight, false);
        NVSDK_NGX_Resource_VK resDepth = NVSDK_NGX_Create_ImageView_Resource_VK(
            imgDepth.view, imgDepth.image, depthRange, imgDepth.format, kWidth, kHeight, false);
        NVSDK_NGX_Resource_VK resOutput = NVSDK_NGX_Create_ImageView_Resource_VK(
            imgOutput.view, imgOutput.image, colorRange, imgOutput.format, kWidth, kHeight, true);

        params.Set("DLSSNR.Color", static_cast<void*>(&resColor));
        params.Set("DLSSNR.Backbuffer", static_cast<void*>(&resBackbuffer));
        params.Set("DLSSNR.MVec", static_cast<void*>(&resMotion));
        params.Set("DLSSNR.Depth", static_cast<void*>(&resDepth));
        params.Set("DLSSNR.Output", static_cast<void*>(&resOutput));
        params.Set("DLSSNR.Intensity", 1.0f);
        params.Set("DLSSNR.Reset", 1);
        params.Set("DLSSNR.Enabled", 1);
        params.Set("DLSSNR.MVecScaleX", static_cast<float>(kWidth));
        params.Set("DLSSNR.MVecScaleY", static_cast<float>(kHeight));
        params.Set("DLSSNR.DepthInverted", 0);
        params.Set("Width", static_cast<unsigned int>(kWidth));
        params.Set("Height", static_cast<unsigned int>(kHeight));

        BeginCmd(ctx);
        NVSDK_NGX_Result r = NgxCall("EvaluateFeature(feature 18)", [&] {
            return ngx.EvaluateFeature(ctx.cmd, feature, &params, nullptr);
        });
        SubmitWait(ctx);
        if (r != NVSDK_NGX_Result_Success)
            Log("[NGX] EvaluateFeature did not succeed -- output may be untouched");
    }

    // -- 7. Readback + stats + PNG -------------------------------------------
    std::vector<uint16_t> outPix(static_cast<size_t>(kWidth) * kHeight * 4, 0);
    {
        StagingBuffer stRead = CreateStaging(ctx, outPix.size() * sizeof(uint16_t));
        BeginCmd(ctx);
        CopyImageToBuffer(ctx, imgOutput.image, VK_IMAGE_ASPECT_COLOR_BIT, stRead.buffer);
        SubmitWait(ctx);
        std::memcpy(outPix.data(), stRead.mapped, outPix.size() * sizeof(uint16_t));
        vkDestroyBuffer(ctx.device, stRead.buffer, nullptr);
        vkFreeMemory(ctx.device, stRead.memory, nullptr);
    }
    PrintStats("DLSSNR output", outPix, &colorPix);
    const std::string pngPath = outDir + "/dlssnr_output.png";
    if (WritePng(pngPath, outPix))
        Log("[OUT] wrote %s", pngPath.c_str());
    else
        Log("[OUT] FAILED to write %s", pngPath.c_str());

    // -- 8. Teardown -----------------------------------------------------------
    NgxCall("ReleaseFeature", [&] { return ngx.ReleaseFeature(feature); });
    NgxCall("Shutdown1", [&] { return ngx.Shutdown1(ctx.device); });

    vkDestroyFence(ctx.device, ctx.fence, nullptr);
    vkDestroyCommandPool(ctx.device, ctx.cmdPool, nullptr);
    for (GpuImage* gi : {&imgColor, &imgBackbuffer, &imgOutput, &imgMotion, &imgDepth})
    {
        vkDestroyImageView(ctx.device, gi->view, nullptr);
        vkDestroyImage(ctx.device, gi->image, nullptr);
        vkFreeMemory(ctx.device, gi->memory, nullptr);
    }
    for (StagingBuffer* sb : {&stColor, &stDepth, &stMotion})
    {
        vkUnmapMemory(ctx.device, sb->memory);
        vkDestroyBuffer(ctx.device, sb->buffer, nullptr);
        vkFreeMemory(ctx.device, sb->memory, nullptr);
    }
    vkDestroyDevice(ctx.device, nullptr);
    vkDestroyInstance(ctx.instance, nullptr);
    ::FreeLibrary(ngx.dll);

    Log("== dlss5_spike done ==");
    return 0;
}
