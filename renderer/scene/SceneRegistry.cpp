#include "renderer/scene/SceneRegistry.h"

#include "renderer/core/PathUtil.h"

#include <cstdio>

namespace sr {

std::vector<SceneEntry> listScenes() {
    std::vector<SceneEntry> scenes;
    scenes.push_back({"boxes", "procedural box room (built-in, no assets)", "", true});
    auto addGltf = [&scenes](const char* alias, const char* desc, const char* relPath) {
        // Assets resolve against CWD first, then the exe location (packaged).
        const std::string resolved = resolveAssetPath(relPath);
        scenes.push_back({alias, desc, resolved, assetFileExists(resolved)});
    };
    addGltf("sponza", "Sponza atrium (Khronos glTF sample)", "assets/sponza/Sponza.gltf");
    addGltf("bistro_exterior", "Bistro street (Amazon Lumberyard Bistro)",
            "assets/bistro/BistroExterior.gltf");
    addGltf("bistro_interior", "Bistro cafe interior (Amazon Lumberyard Bistro)",
            "assets/bistro/BistroInterior.gltf");
    return scenes;
}

std::string resolveSceneArg(const std::string& arg) {
    for (const SceneEntry& s : listScenes()) {
        if (arg == s.alias) {
            if (!s.available) {
                std::fprintf(stderr, "warning: scene '%s' asset missing (%s)\n", s.alias.c_str(),
                             s.path.c_str());
            }
            return s.path;
        }
    }
    if (arg == "procedural") return "";  // legacy alias of "boxes" (built-in, no assets)
    return resolveAssetPath(arg);  // treat as a glTF path (may still be CWD-relative)
}

LightingPreset defaultLightingPreset() { return LightingPreset{}; }

LightingPreset goldenHourPreset() {
    LightingPreset p;
    // Low warm sun in front-left of the BistroExterior start camera (looking
    // down -Z): long shadows toward the camera-right, matching the ORCA shot.
    p.sunElevationDeg = 22.f;
    p.sunAzimuthDeg = 215.f;
    p.sunIntensity = 4.5f;
    p.sunColor = {1.f, 0.72f, 0.45f};
    p.sunEnabled = true;
    p.fillEnabled = false;
    p.iblIntensity = 0.45f;
    p.exposure = 1.f;
    p.preferPresetSun = true;
    return p;
}

LightingPreset lightingPresetForScene(const std::string& scenePath) {
    if (scenePath.find("BistroExterior") != std::string::npos) return goldenHourPreset();
    return defaultLightingPreset();
}

std::vector<Light> lightsFromPreset(const LightingPreset& p) {
    std::vector<Light> v;
    if (p.sunEnabled)
        v.push_back(makeSunLight(p.sunElevationDeg, p.sunAzimuthDeg, p.sunIntensity, p.sunColor));
    if (p.fillEnabled) v.push_back(defaultFillLight());
    return v;
}

bool initialCameraPose(const std::string& scenePath, Vec3& pos, Vec3& fwd) {
    // BistroExterior: on the street looking into the patisserie block.
    // (Reported repro location; the generic glTF start pose sits in a wall.)
    if (scenePath.find("BistroExterior") != std::string::npos) {
        pos = {-11.2f, 4.6f, 12.0f};
        fwd = {0.198f, -0.015f, -0.980f};
        return true;
    }
    return false;
}

} // namespace sr
