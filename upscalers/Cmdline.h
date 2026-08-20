#pragma once
// ============================================================================
// Shared command-line gate for upscaler device-requirement hooks.
//
// The dlss/nss/xess modules must only inject their device extensions, feature
// chains and instance layers (and run SDK init) when the run actually selects
// them.  Substring-matching the whole command line is wrong — e.g.
// `--output output/nss` would look like an nss request — so parse argv for
// --upscaler/--upscalers and match the requested value(s) instead.
// ============================================================================
#include <windows.h> // must precede shellapi.h
#include <shellapi.h>

#include <cstring>
#include <string>

namespace sr {
namespace cmdline {

// True when `key` is among the --upscaler/--upscalers values.  Values are
// comma-separated and the token "all" matches every plugin.  Each token is
// matched by substring so a single key ("dlss") covers "dlss-k", "dlss-l"
// and "dlss-m".  When the command line cannot be parsed we return true (fail
// open, matching the original DLSS gate).
inline bool pluginRequested(const char* key) {
    if (!key || !*key) return true;

    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!argv) return true; // cannot parse -> assume the plugin may be used

    const std::wstring needle(key, key + std::strlen(key));
    bool requested = false;
    for (int i = 1; i < argc && !requested; ++i) {
        const std::wstring arg = argv[i];
        if ((arg != L"--upscaler" && arg != L"--upscalers") || i + 1 >= argc) continue;
        const std::wstring value = argv[i + 1];
        size_t begin = 0;
        while (begin <= value.size()) {
            const size_t comma = value.find(L',', begin);
            const std::wstring item = value.substr(
                begin, comma == std::wstring::npos ? std::wstring::npos : comma - begin);
            if (item == L"all" || item.find(needle) != std::wstring::npos) {
                requested = true;
                break;
            }
            if (comma == std::wstring::npos) break;
            begin = comma + 1;
        }
    }
    LocalFree(argv);
    return requested;
}

} // namespace cmdline
} // namespace sr
