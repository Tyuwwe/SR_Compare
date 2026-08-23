// ============================================================================
// EngineConfig — engine.toml parsing (toml++, header-only) + the shared
// apply helpers.  See EngineConfig.h for the precedence model.
// ============================================================================
#include "renderer/core/EngineConfig.h"
#include "renderer/Renderer.h"
#include "renderer/core/PathUtil.h"

#include <toml.hpp>

#include <cstdio>
#include <fstream>
#include <sys/stat.h>

namespace sr {

std::string engineConfigPath() {
    const std::string p = engineConfigFilePath();
    return assetFileExists(p) ? p : std::string();
}

std::string engineConfigFilePath() { return exeDir() + "/engine.toml"; }

int64_t engineConfigWriteTime(const std::string& path) {
    struct stat st;
    return (!path.empty() && ::stat(path.c_str(), &st) == 0) ? static_cast<int64_t>(st.st_mtime)
                                                             : 0;
}

void EngineConfigLog::add(const char* key, bool v) {
    text += ' ';
    text += key;
    text += v ? "=true" : "=false";
}

void EngineConfigLog::add(const char* key, int32_t v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), " %s=%d", key, static_cast<int>(v));
    text += buf;
}

void EngineConfigLog::add(const char* key, float v) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), " %s=%.4g", key, static_cast<double>(v));
    text += buf;
}

void EngineConfigLog::add(const char* key, const std::string& v) {
    text += ' ';
    text += key;
    text += "=\"";
    text += v;
    text += '"';
}

void EngineConfigLog::flush(const char* mode) const {
    if (text.empty()) return;
    std::fprintf(stderr, "[engine.toml]%s%s\n", mode ? mode : "", text.c_str());
}

namespace {

// Typed key fetch: type mismatches warn and leave the key unset (defaults).
std::optional<bool> getBool(const toml::table& t, const char* key) {
    if (const toml::node* n = t.get(key)) {
        if (const auto v = n->value<bool>()) return v;
        std::fprintf(stderr, "[engine.toml] ignoring %s: expected boolean\n", key);
    }
    return std::nullopt;
}

std::optional<float> getFloat(const toml::table& t, const char* key) {
    if (const toml::node* n = t.get(key)) {
        // toml++ separates int64/double; accept both for float keys.
        if (const auto v = n->value<double>()) return static_cast<float>(*v);
        if (const auto v = n->value<int64_t>()) return static_cast<float>(*v);
        std::fprintf(stderr, "[engine.toml] ignoring %s: expected number\n", key);
    }
    return std::nullopt;
}

// Integer keys (window size): int64 only, range-gated like clampCheck.
std::optional<int32_t> getInt(const toml::table& t, const char* key, int32_t lo, int32_t hi) {
    if (const toml::node* n = t.get(key)) {
        if (const auto v = n->value<int64_t>()) {
            if (*v >= lo && *v <= hi) return static_cast<int32_t>(*v);
            std::fprintf(stderr, "[engine.toml] ignoring %s: value %lld out of range\n", key,
                         static_cast<long long>(*v));
            return std::nullopt;
        }
        std::fprintf(stderr, "[engine.toml] ignoring %s: expected integer\n", key);
    }
    return std::nullopt;
}

std::optional<std::string> getString(const toml::table& t, const char* key) {
    if (const toml::node* n = t.get(key)) {
        if (const auto v = n->value<std::string>()) return v;
        std::fprintf(stderr, "[engine.toml] ignoring %s: expected string\n", key);
    }
    return std::nullopt;
}

// Range gate matching the CLI validators in app/CliUtils.h.  Rejected values
// warn and leave the key unset (default kept), they are never clamped.
std::optional<float> clampCheck(std::optional<float> v, float lo, float hi, bool loExclusive,
                                const char* key) {
    if (!v) return v;
    const bool ok = (loExclusive ? (*v > lo) : (*v >= lo)) && *v <= hi;
    if (!ok) {
        std::fprintf(stderr, "[engine.toml] ignoring %s: value %.4g out of range\n", key,
                     static_cast<double>(*v));
        return std::nullopt;
    }
    return v;
}

const toml::table* section(const toml::table& root, const char* name) {
    return root.get(name) ? root.get(name)->as_table() : nullptr;
}

} // namespace

bool loadEngineConfig(const std::string& path, EngineConfig& out) {
    out = EngineConfig{};
    toml::table root;
    try {
        root = toml::parse_file(path);
    } catch (const toml::parse_error& err) {
        const std::string desc(err.description()); // string_view in toml++ v3
        std::fprintf(stderr, "[engine.toml] parse error in %s: %s — using defaults\n",
                     path.c_str(), desc.c_str());
        return false;
    }
    std::fprintf(stderr, "[engine.toml] loaded %s\n", path.c_str());

    if (const toml::table* t = section(root, "window")) {
        out.fullscreen = getBool(*t, "fullscreen");
        out.width = getInt(*t, "width", 64, 16384);
        out.height = getInt(*t, "height", 64, 16384);
    }
    if (const toml::table* t = section(root, "renderer")) {
        out.renderScale = clampCheck(getFloat(*t, "render_scale"), 0.f, 1.f, true, "render_scale");
        out.hdr = getBool(*t, "hdr");
        out.envMap = getString(*t, "env_map");
    }
    if (const toml::table* t = section(root, "exposure")) {
        out.exposure = clampCheck(getFloat(*t, "exposure"), 0.f, 16.f, true, "exposure");
        out.exposureMinEV = getFloat(*t, "min_ev");
        out.exposureMaxEV = getFloat(*t, "max_ev");
        if (out.exposureMinEV && out.exposureMaxEV && *out.exposureMinEV >= *out.exposureMaxEV) {
            std::fprintf(stderr, "[engine.toml] ignoring min_ev/max_ev: min >= max\n");
            out.exposureMinEV.reset();
            out.exposureMaxEV.reset();
        }
    }
    if (const toml::table* t = section(root, "effects")) {
        out.ssr = getBool(*t, "ssr");
        out.ssrStrength = clampCheck(getFloat(*t, "ssr_strength"), 0.f, 1.f, false, "ssr_strength");
        out.shadows = getBool(*t, "shadows");
        out.contactShadows = getBool(*t, "contact_shadows");
        out.volFog = getBool(*t, "volfog");
        out.bloom = getBool(*t, "bloom");
        out.motionBlur = getBool(*t, "motion_blur");
        out.lensFx = getBool(*t, "lens_fx");
    }
    if (const toml::table* t = section(root, "lens_fx")) {
        out.lensCa = getBool(*t, "chromatic_aberration");
        out.lensVignette = getBool(*t, "vignette");
        out.lensGrain = getBool(*t, "film_grain");
    }
    if (const toml::table* t = section(root, "culling")) {
        out.occlusion = getBool(*t, "occlusion");
        out.lod = getBool(*t, "lod");
    }
    if (const toml::table* t = section(root, "dof")) {
        out.dof = getBool(*t, "enabled");
        out.dofFocus = clampCheck(getFloat(*t, "focus"), 0.f, 10000.f, false, "dof.focus");
        out.dofFstop = clampCheck(getFloat(*t, "fstop"), 0.5f, 64.f, false, "dof.fstop");
        out.dofMaxBlur = clampCheck(getFloat(*t, "max_blur"), 1.f, 64.f, false, "dof.max_blur");
    }
    if (const toml::table* t = section(root, "grading")) {
        out.gradingTempK = clampCheck(getFloat(*t, "temperature"), 0.f, 40000.f, true, "temperature");
        out.gradingTint = clampCheck(getFloat(*t, "tint"), -1.f, 1.f, false, "tint");
        out.gradingContrast = clampCheck(getFloat(*t, "contrast"), 0.f, 8.f, true, "contrast");
        out.gradingSat = clampCheck(getFloat(*t, "saturation"), 0.f, 8.f, true, "saturation");
        out.lut = getString(*t, "lut");
    }
    if (const toml::table* t = section(root, "sun")) {
        // Negative elevation/azimuth = "use the scene preset" (same sentinel
        // as the RendererOptions defaults).
        out.sunElevationDeg = getFloat(*t, "elevation");
        out.sunAzimuthDeg = getFloat(*t, "azimuth");
    }
    return true;
}

void applyEngineConfig(RendererOptions& opts, const EngineConfig& cfg, uint64_t cliMask,
                       EngineConfigLog& log) {
    cfgTake(opts.renderScale, cfg.renderScale, cliMask, cli::kRenderScale, "render_scale", log);
    cfgTake(opts.hdr, cfg.hdr, cliMask, cli::kHdr, "hdr", log);
    cfgTake(opts.envMapPath, cfg.envMap, cliMask, cli::kEnvMap, "env_map", log);
    if (cfg.exposure && (cliMask & cli::kExposure) == 0) {
        // Same semantics as CLI --exposure: a manual value disables auto exposure.
        opts.exposure = *cfg.exposure;
        opts.autoExposure = false;
        log.add("exposure", opts.exposure);
    }
    cfgTake(opts.exposureMinEV, cfg.exposureMinEV, cliMask, cli::kExposure, "exposure_min_ev", log);
    cfgTake(opts.exposureMaxEV, cfg.exposureMaxEV, cliMask, cli::kExposure, "exposure_max_ev", log);
    cfgTake(opts.ssr, cfg.ssr, cliMask, cli::kSsr, "ssr", log);
    cfgTake(opts.ssrStrength, cfg.ssrStrength, cliMask, cli::kSsrStrength, "ssr_strength", log);
    cfgTake(opts.shadows, cfg.shadows, cliMask, cli::kShadows, "shadows", log);
    cfgTake(opts.contactShadows, cfg.contactShadows, cliMask, cli::kContactShadows,
            "contact_shadows", log);
    cfgTake(opts.volFog, cfg.volFog, cliMask, cli::kVolFog, "volfog", log);
    cfgTake(opts.bloom, cfg.bloom, cliMask, cli::kBloom, "bloom", log);
    cfgTake(opts.motionBlur, cfg.motionBlur, cliMask, cli::kMotionBlur, "motion_blur", log);
    cfgTake(opts.lensFx, cfg.lensFx, cliMask, cli::kLensFx, "lens_fx", log);
    cfgTake(opts.dof, cfg.dof, cliMask, cli::kDof, "dof", log);
    cfgTake(opts.dofFocus, cfg.dofFocus, cliMask, cli::kDofFocus, "dof_focus", log);
    cfgTake(opts.dofFstop, cfg.dofFstop, cliMask, cli::kDofFstop, "dof_fstop", log);
    cfgTake(opts.dofMaxBlurPx, cfg.dofMaxBlur, cliMask, cli::kDofMaxBlur, "dof_max_blur", log);
    cfgTake(opts.grading.temperatureK, cfg.gradingTempK, cliMask, cli::kGradingTemp, "temperature",
            log);
    cfgTake(opts.grading.tint, cfg.gradingTint, cliMask, cli::kGradingTint, "tint", log);
    cfgTake(opts.grading.contrast, cfg.gradingContrast, cliMask, cli::kGradingContrast, "contrast",
            log);
    cfgTake(opts.grading.saturation, cfg.gradingSat, cliMask, cli::kGradingSat, "saturation", log);
    cfgTake(opts.lutPath, cfg.lut, cliMask, cli::kLut, "lut", log);
    cfgTake(opts.sunElevationDeg, cfg.sunElevationDeg, cliMask, cli::kSunElev, "sun_elev", log);
    cfgTake(opts.sunAzimuthDeg, cfg.sunAzimuthDeg, cliMask, cli::kSunAz, "sun_az", log);
}

namespace {

// %.6g matches the EngineConfigLog style and round-trips the GUI slider
// values (the baseline-sync after each write keeps the save/reload loop
// stable even in the last ulp).
void putFloat(std::string& out, const char* key, float v) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%s = %.6g\n", key, static_cast<double>(v));
    out += buf;
}

void putBool(std::string& out, const char* key, bool v) {
    out += key;
    out += v ? " = true\n" : " = false\n";
}

void putInt(std::string& out, const char* key, int32_t v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s = %d\n", key, static_cast<int>(v));
    out += buf;
}

// TOML literal string (single quotes): Windows paths keep their backslashes.
void putString(std::string& out, const char* key, const std::string& v) {
    out += key;
    out += " = '";
    for (const char c : v) out += (c == '\'') ? '"' : c; // a stray ' would break the literal
    out += "'\n";
}

} // namespace

std::string engineConfigToToml(const EngineConfig& cfg) {
    std::string out =
        "# ============================================================================\n"
        "# engine.toml — GENERATED by `sr_compare gui` (auto-create / auto-save).\n"
        "# Reference template with full per-key comments: engine.toml.example.\n"
        "# Precedence: explicit CLI flag > this file > code defaults.  The GUI\n"
        "# hot-reloads edits (~1 s) and rewrites this file after UI changes, so\n"
        "# comments and formatting added by hand are not preserved.\n"
        "# ============================================================================\n"
        "\n";
    if (cfg.fullscreen || cfg.width || cfg.height) {
        out += "[window]\n";
        if (cfg.fullscreen) putBool(out, "fullscreen", *cfg.fullscreen);
        if (cfg.width) putInt(out, "width", *cfg.width);
        if (cfg.height) putInt(out, "height", *cfg.height);
        out += "\n";
    }
    out += "[renderer]\n";
    if (cfg.renderScale) putFloat(out, "render_scale", *cfg.renderScale);
    if (cfg.hdr) putBool(out, "hdr", *cfg.hdr);
    if (cfg.envMap) putString(out, "env_map", *cfg.envMap);
    out += "\n[exposure]\n";
    if (cfg.exposure) putFloat(out, "exposure", *cfg.exposure);
    if (cfg.exposureMinEV) putFloat(out, "min_ev", *cfg.exposureMinEV);
    if (cfg.exposureMaxEV) putFloat(out, "max_ev", *cfg.exposureMaxEV);
    out += "\n[effects]\n";
    if (cfg.ssr) putBool(out, "ssr", *cfg.ssr);
    if (cfg.ssrStrength) putFloat(out, "ssr_strength", *cfg.ssrStrength);
    if (cfg.shadows) putBool(out, "shadows", *cfg.shadows);
    if (cfg.contactShadows) putBool(out, "contact_shadows", *cfg.contactShadows);
    if (cfg.volFog) putBool(out, "volfog", *cfg.volFog);
    if (cfg.bloom) putBool(out, "bloom", *cfg.bloom);
    if (cfg.motionBlur) putBool(out, "motion_blur", *cfg.motionBlur);
    if (cfg.lensFx) putBool(out, "lens_fx", *cfg.lensFx);
    out += "\n[lens_fx]\n";
    if (cfg.lensCa) putBool(out, "chromatic_aberration", *cfg.lensCa);
    if (cfg.lensVignette) putBool(out, "vignette", *cfg.lensVignette);
    if (cfg.lensGrain) putBool(out, "film_grain", *cfg.lensGrain);
    out += "\n[culling]\n";
    if (cfg.occlusion) putBool(out, "occlusion", *cfg.occlusion);
    if (cfg.lod) putBool(out, "lod", *cfg.lod);
    out += "\n[dof]\n";
    if (cfg.dof) putBool(out, "enabled", *cfg.dof);
    if (cfg.dofFocus) putFloat(out, "focus", *cfg.dofFocus);
    if (cfg.dofFstop) putFloat(out, "fstop", *cfg.dofFstop);
    if (cfg.dofMaxBlur) putFloat(out, "max_blur", *cfg.dofMaxBlur);
    out += "\n[grading]\n";
    if (cfg.gradingTempK) putFloat(out, "temperature", *cfg.gradingTempK);
    if (cfg.gradingTint) putFloat(out, "tint", *cfg.gradingTint);
    if (cfg.gradingContrast) putFloat(out, "contrast", *cfg.gradingContrast);
    if (cfg.gradingSat) putFloat(out, "saturation", *cfg.gradingSat);
    if (cfg.lut) putString(out, "lut", *cfg.lut);
    out += "\n[sun]\n";
    if (cfg.sunElevationDeg) putFloat(out, "elevation", *cfg.sunElevationDeg);
    if (cfg.sunAzimuthDeg) putFloat(out, "azimuth", *cfg.sunAzimuthDeg);
    return out;
}

bool writeFileAtomic(const std::string& path, const std::string& text) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            std::fprintf(stderr, "[engine.toml] cannot open %s for writing\n", tmp.c_str());
            return false;
        }
        f.write(text.data(), static_cast<std::streamsize>(text.size()));
        f.close();
        if (!f) {
            std::fprintf(stderr, "[engine.toml] write failed: %s\n", tmp.c_str());
            std::remove(tmp.c_str());
            return false;
        }
    }
    // Replace-existing keeps a complete file visible at `path` at all times
    // (plain rename would fail with the destination present on Windows; a
    // remove+rename gap could trip the deleted-file reset in the hot poll).
#ifdef _WIN32
    if (!MoveFileExA(tmp.c_str(), path.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
#else
    std::remove(path.c_str());
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
#endif
        std::fprintf(stderr, "[engine.toml] rename %s -> %s failed\n", tmp.c_str(), path.c_str());
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

} // namespace sr
