#pragma once
// ============================================================================
// EngineConfig — optional runtime config file (engine.toml next to the exe,
// resolved like other exe-relative assets, see PathUtil.h).
//
// Precedence: explicit CLI flag > engine.toml > code default.  Each CLI
// parser records a bitmask of the options the user passed explicitly
// (namespace cli below); the apply helpers only fill keys whose bit is
// clear, so toml never overrides the command line.
//
// A missing engine.toml is a silent no-op (behaviour identical to before);
// a malformed one prints the parse error to stderr and every key falls back
// to defaults.  Out-of-range / mistyped values are rejected per key with a
// stderr note (same ranges as the CLI validators in app/CliUtils.h).
//
// Modes: viewer/compare/bench/gui all load it once at startup.  The GUI
// additionally hot-reloads on file modification (~1 s stat interval):
// per-frame parameters (effects toggles, DOF/grading/sun sliders, exposure,
// occlusion/lod, HDR) apply immediately; resolution/scale/env-map/scene/LUT
// changes need an Apply rebuild or a restart and are ignored by the reload.
// ============================================================================
#include <cstdint>
#include <optional>
#include <string>

namespace sr {

// Bits marking options explicitly given on the command line.  Shared by all
// modes (a mode simply never sets bits for flags it does not have).
namespace cli {
inline constexpr uint64_t kNone = 0;
inline constexpr uint64_t kRenderScale = 1ull << 0;
inline constexpr uint64_t kExposure = 1ull << 1;
inline constexpr uint64_t kSsr = 1ull << 2;
inline constexpr uint64_t kSsrStrength = 1ull << 3;
inline constexpr uint64_t kShadows = 1ull << 4;
inline constexpr uint64_t kContactShadows = 1ull << 5;
inline constexpr uint64_t kVolFog = 1ull << 6;
inline constexpr uint64_t kBloom = 1ull << 7;
inline constexpr uint64_t kMotionBlur = 1ull << 8;
inline constexpr uint64_t kDof = 1ull << 9;
inline constexpr uint64_t kDofFocus = 1ull << 10;
inline constexpr uint64_t kDofFstop = 1ull << 11;
inline constexpr uint64_t kDofMaxBlur = 1ull << 12;
inline constexpr uint64_t kLensFx = 1ull << 13;
inline constexpr uint64_t kSunElev = 1ull << 14;
inline constexpr uint64_t kSunAz = 1ull << 15;
inline constexpr uint64_t kEnvMap = 1ull << 16;
inline constexpr uint64_t kHdr = 1ull << 17;
inline constexpr uint64_t kLut = 1ull << 18;
inline constexpr uint64_t kGradingTemp = 1ull << 19;
inline constexpr uint64_t kGradingTint = 1ull << 20;
inline constexpr uint64_t kGradingContrast = 1ull << 21;
inline constexpr uint64_t kGradingSat = 1ull << 22;
inline constexpr uint64_t kOutput = 1ull << 23;
} // namespace cli

// Parsed engine.toml content.  Absent keys stay empty optionals; the apply
// step leaves the corresponding option at its code default.
struct EngineConfig {
    // [renderer]
    std::optional<float> renderScale;      // (0, 1]
    std::optional<bool> hdr;               // HDR swapchain (viewer/gui)
    std::optional<std::string> envMap;     // equirect HDR; "" = sky atmosphere
    // [exposure]
    std::optional<float> exposure;         // manual display exposure (0, 16]; disables auto
    std::optional<float> exposureMinEV;    // auto-exposure EV clamp range
    std::optional<float> exposureMaxEV;
    // [effects]
    std::optional<bool> ssr;
    std::optional<float> ssrStrength;      // [0, 1]
    std::optional<bool> shadows;
    std::optional<bool> contactShadows;
    std::optional<bool> volFog;
    std::optional<bool> bloom;
    std::optional<bool> motionBlur;
    std::optional<bool> lensFx;            // master switch (viewer/compare)
    // [lens_fx] — per-effect sub-items (GUI compose chain)
    std::optional<bool> lensCa;
    std::optional<bool> lensVignette;
    std::optional<bool> lensGrain;
    // [culling]
    std::optional<bool> occlusion;         // GPU Hi-Z occlusion culling (GUI)
    std::optional<bool> lod;               // screen-size LOD selection (GUI)
    // [dof]
    std::optional<bool> dof;
    std::optional<float> dofFocus;         // [0, 10000] m, 0 = auto-focus
    std::optional<float> dofFstop;         // [0.5, 64]
    std::optional<float> dofMaxBlur;       // [1, 64] px at 1080p
    // [grading]
    std::optional<float> gradingTempK;     // > 0
    std::optional<float> gradingTint;      // [-1, 1]
    std::optional<float> gradingContrast;  // > 0
    std::optional<float> gradingSat;       // > 0
    std::optional<std::string> lut;        // .cube path (viewer only)
    // [sun]
    std::optional<float> sunElevationDeg;  // negative = scene preset
    std::optional<float> sunAzimuthDeg;
};

// Path of the config file (<exe dir>/engine.toml), or "" when absent.
std::string engineConfigPath();

// Parse `path` into `out`.  Returns false (leaving `out` at defaults) when
// the file is missing or malformed; errors go to stderr, never fatal.
bool loadEngineConfig(const std::string& path, EngineConfig& out);

// Convenience: engineConfigPath() + loadEngineConfig().  `out.loaded()`
// below reports whether a file was found.
inline bool loadEngineConfig(EngineConfig& out) {
    const std::string path = engineConfigPath();
    return !path.empty() && loadEngineConfig(path, out);
}

// Modification time (seconds since epoch) for the hot-reload watch; 0 when
// the file is missing/unstattable.
int64_t engineConfigWriteTime(const std::string& path);

// Accumulates the "key=value" list of options a toml apply actually changed
// (i.e. present in the file AND not masked by an explicit CLI flag), printed
// to stderr by flush() so scripted runs can verify the config took effect.
struct EngineConfigLog {
    std::string text;
    void add(const char* key, bool v);
    void add(const char* key, float v);
    void add(const char* key, const std::string& v);
    bool empty() const { return text.empty(); }
    // "[engine.toml] <mode>: k=v k=v ..." to stderr (no-op when empty).
    void flush(const char* mode) const;
};

// Shared take-helpers: apply a toml key only when the CLI did not set it.
inline bool cfgTake(bool& dst, const std::optional<bool>& v, uint64_t cliMask, uint64_t bit,
                    const char* key, EngineConfigLog& log) {
    if (v && (cliMask & bit) == 0) {
        dst = *v;
        log.add(key, dst);
        return true;
    }
    return false;
}
inline bool cfgTake(float& dst, const std::optional<float>& v, uint64_t cliMask, uint64_t bit,
                    const char* key, EngineConfigLog& log) {
    if (v && (cliMask & bit) == 0) {
        dst = *v;
        log.add(key, dst);
        return true;
    }
    return false;
}
inline bool cfgTake(std::string& dst, const std::optional<std::string>& v, uint64_t cliMask,
                    uint64_t bit, const char* key, EngineConfigLog& log) {
    if (v && (cliMask & bit) == 0) {
        dst = *v;
        log.add(key, dst);
        return true;
    }
    return false;
}

struct RendererOptions;
// Fills RendererOptions from `cfg` (viewer mode).  Manual exposure from toml
// switches off auto exposure exactly like CLI --exposure.
void applyEngineConfig(RendererOptions& opts, const EngineConfig& cfg, uint64_t cliMask,
                       EngineConfigLog& log);

} // namespace sr
