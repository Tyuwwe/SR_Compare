#pragma once
// ============================================================================
// Path utilities — resolve project-asset relative paths (e.g.
// "assets/env/x.hdr") regardless of the process working directory, so the
// packaged exe can be launched from anywhere (build/app/Release, another
// drive, a shortcut).
// ============================================================================
#include <string>
#include <sys/stat.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace sr {

inline bool assetFileExists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && (st.st_mode & _S_IFREG) != 0;
}

// Returns `path` unchanged when it exists as given (CWD-relative) or is
// absolute; otherwise probes <exe dir> and up to 4 parents for it.
inline std::string resolveAssetPath(const std::string& path) {
    if (path.empty() || assetFileExists(path)) return path;
    if (path.size() > 1 && (path[1] == ':' || path[0] == '/' || path[0] == '\\')) return path;

    char buf[MAX_PATH] = {};
    const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return path;
    std::string dir(buf, n);
    const size_t pos = dir.find_last_of("\\/");
    dir = (pos == std::string::npos) ? std::string(".") : dir.substr(0, pos);

    for (int up = 0; up < 5; ++up) {
        const std::string candidate = dir + "/" + path;
        if (assetFileExists(candidate)) return candidate;
        dir += "/..";
    }
    return path;
}

// Directory of the running executable (no trailing slash).
inline std::string exeDir() {
    char buf[MAX_PATH] = {};
    const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return ".";
    std::string dir(buf, n);
    const size_t pos = dir.find_last_of("\\/");
    return (pos == std::string::npos) ? std::string(".") : dir.substr(0, pos);
}

// Resolve an output path for writing.  Absolute paths are returned
// unchanged; relative paths are anchored at the exe directory (NOT the CWD,
// which depends on how the app was launched) and normalized to a full path,
// so outputs always land in a deterministic, user-findable place.
inline std::string resolveOutputPath(const std::string& path) {
    if (path.empty()) return path;
    if (path.size() > 1 && (path[1] == ':' || path[0] == '/' || path[0] == '\\')) return path;
    const std::string joined = exeDir() + "/" + path;
    char full[MAX_PATH] = {};
    const DWORD n = GetFullPathNameA(joined.c_str(), MAX_PATH, full, nullptr);
    return (n > 0 && n < MAX_PATH) ? std::string(full, n) : joined;
}

// Resolve a compiled-in SPIR-V path for packaged runs.  Order:
//   1. bakedDir + name          (dev machine: CMAKE_BINARY_DIR/shaders)
//   2. <exe dir>/shaders/name   (packaged layout)
//   3. <exe dir>/../../shaders/name etc. (exe inside build/app/<cfg>)
inline std::string resolveShaderPath(const std::string& bakedDir, const std::string& name) {
    const std::string baked = bakedDir + name;
    if (assetFileExists(baked)) return baked;
    std::string dir = exeDir();
    for (int up = 0; up < 5; ++up) {
        const std::string candidate = dir + "/shaders/" + name;
        if (assetFileExists(candidate)) return candidate;
        dir += "/..";
    }
    return baked;  // caller prints this on failure
}

// Ensure the parent directory of `path` exists (single level is enough for
// our flat output/ layout; recursive for safety).
inline void ensureParentDir(const std::string& path) {
    const size_t pos = path.find_last_of("\\/");
    if (pos == std::string::npos) return;
    std::string dir = path.substr(0, pos);
    // Iterative mkdir -p (Windows CreateDirectoryA fails silently if exists).
    for (size_t i = 0; i < dir.size(); ++i) {
        if (dir[i] == '/' || dir[i] == '\\') {
            if (i > 0 && dir[i - 1] != ':' && dir[i - 1] != '/' && dir[i - 1] != '\\') {
                CreateDirectoryA(dir.substr(0, i).c_str(), nullptr);
            }
        }
    }
    CreateDirectoryA(dir.c_str(), nullptr);
}

} // namespace sr
