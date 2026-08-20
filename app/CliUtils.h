#pragma once
// ============================================================================
// Shared command-line parsing helpers for the sr_compare CLI (viewer /
// compare / bench / gui).  Consolidates the duplicated argument-value,
// resolution and render-scale parsers so the entry points stay in sync.
//
// Keep this header dependency-light: it is included by every CLI mode and
// must not pull in Vulkan or renderer headers.
// ============================================================================
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace sr {

// Parse "WxH" into w/h.  Returns false on malformed or non-positive values.
// Uses sscanf_s (MSVC), matching the pre-existing CLI behaviour.
inline bool parseResolution(const char* s, uint32_t& w, uint32_t& h) {
    int iw = 0, ih = 0;
    if (::sscanf_s(s, "%dx%d", &iw, &ih) != 2 || iw <= 0 || ih <= 0) return false;
    w = static_cast<uint32_t>(iw);
    h = static_cast<uint32_t>(ih);
    return true;
}

// Fetch the value following option `name`, advancing `i` past it.  A missing
// value is fatal (the original per-mode lambdas all called exit(1)).
inline const char* nextArg(int& i, int argc, char** argv, const char* name) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", name);
        std::exit(1);
    }
    return argv[++i];
}

// Split a comma-separated list, dropping empty items.
inline std::vector<std::string> splitCsv(const char* s) {
    std::vector<std::string> out;
    const std::string str = s ? s : "";
    size_t start = 0;
    while (start <= str.size()) {
        const size_t comma = str.find(',', start);
        const std::string item =
            str.substr(start, comma == std::string::npos ? comma : comma - start);
        if (!item.empty()) out.push_back(item);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

// Parse a render-resolution scale and validate it is in (0, 1].  NaN is
// rejected via !(v > 0) (NaN compares false against any value).
inline bool parseRenderScale(const char* s, float& out) {
    const float v = static_cast<float>(std::atof(s));
    if (!(v > 0.0f) || v > 1.0f) return false;
    out = v;
    return true;
}

// True when environment variable `name` is set (any value).  Uses _dupenv_s
// on MSVC: plain getenv trips C4996 and this project builds /W4 zero-warning.
inline bool envFlag(const char* name) {
#ifdef _MSC_VER
    char* value = nullptr;
    size_t len = 0;
    const bool set = _dupenv_s(&value, &len, name) == 0 && value != nullptr;
    std::free(value);
    return set;
#else
    return std::getenv(name) != nullptr;
#endif
}

} // namespace sr
