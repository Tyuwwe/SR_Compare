#include "renderer/scene/SceneRegistry.h"

#include "renderer/core/PathUtil.h"
#include "renderer/ibl/SkyAtmosphere.h"

#include <algorithm>
#include <cstdio>

namespace sr {

// Warm low-sun key colour straight from the atmosphere model (Hillaire 2020):
// transmittance from the ground towards the sun, normalized so the strongest
// channel stays 1 and sunIntensity keeps its meaning.
Vec3 atmosphereSunColor(float elevationDeg, float azimuthDeg) {
    const Vec3 t = SkyAtmosphere::sunTransmittanceFromGround(
        sunDirectionFromElevAzimuth(elevationDeg, azimuthDeg));
    const float m = std::max({t.x, t.y, t.z, 1e-4f});
    return t / m;
}

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

LightingPreset defaultLightingPreset() {
    LightingPreset p;
    // Neutral high sun: atmosphere-derived near-white key (matches the old
    // hand-tuned (1, 0.95, 0.85) closely).  envFile stays empty: the sky and
    // IBL come from the procedural atmosphere.
    p.sunColor = atmosphereSunColor(p.sunElevationDeg, p.sunAzimuthDeg);
    // Mild neutral height fog (boxes room / Sponza atrium): visible shafts
    // through the Sponza arches without washing the scene out.
    p.fog.enabled = true;
    p.fog.density = 0.012f;
    p.fog.heightFalloff = 0.2f;
    p.fog.baseHeight = 0.f;
    p.fog.anisotropy = 0.55f;
    p.fog.albedo = {0.92f, 0.92f, 0.94f};
    p.fog.noiseStrength = 0.4f;
    p.fog.noiseScale = 0.08f;
    p.fog.maxDistance = 200.f;
    p.fog.ambient = 0.4f;
    return p;
}

LightingPreset goldenHourPreset() {
    LightingPreset p;
    // Low sun in front-left of the BistroExterior start camera (looking down
    // -Z): long shadows toward the camera-right, matching the ORCA shot.  The
    // warm key colour now comes from the atmosphere model itself (low sun ->
    // long optical path -> red-shifted transmittance), replacing the old
    // hand-tuned (1, 0.72, 0.45).
    p.sunElevationDeg = 22.f;
    p.sunAzimuthDeg = 215.f;
    p.sunIntensity = 4.5f;
    p.sunColor = atmosphereSunColor(p.sunElevationDeg, p.sunAzimuthDeg);
    p.sunEnabled = true;
    p.fillEnabled = false;
    // The procedural sky at 22 deg elevation is already dimmer and warmer than
    // the noon default, so the old hand-lowered iblIntensity (0.45, tuned for
    // the static san_giuseppe HDR) is gone: keep 1.0.
    p.iblIntensity = 1.f;
    p.exposure = 1.f;
    p.preferPresetSun = true;
    // Showcase volumetric fog: warm dense street-level haze with strong
    // forward scattering so the low sun throws god rays between the
    // buildings; the height falloff keeps the rooftops clear.
    p.fog.enabled = true;
    p.fog.density = 0.03f;
    p.fog.heightFalloff = 0.09f;
    p.fog.baseHeight = 0.f;
    p.fog.anisotropy = 0.7f;
    p.fog.albedo = {1.0f, 0.85f, 0.66f};
    p.fog.noiseStrength = 0.55f;
    p.fog.noiseScale = 0.05f;
    p.fog.maxDistance = 200.f;
    p.fog.ambient = 0.5f;
    // Gentle golden-hour grade (Phase 6c preset override): warm balance and a
    // touch of contrast/saturation.  CLI grading flags win over this.
    p.gradeTemperatureK = 5600.f;
    p.gradeContrast = 1.06f;
    p.gradeSaturation = 1.08f;
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

std::vector<ReflectionProbe> reflectionProbesForScene(const std::string& scenePath) {
    std::vector<ReflectionProbe> probes;
    if (scenePath.empty()) {
        // boxes: one probe covering the whole 20x10x20 room (walls at
        // |x| = 10, z = +-10, floor y = 0; see ProceduralScene.cpp).
        probes.push_back({{0.f, 2.5f, 0.f}, {-9.5f, 0.05f, -9.5f}, {9.5f, 9.5f, 9.5f}});
    } else if (scenePath.find("Sponza") != std::string::npos) {
        // Sponza atrium: one demo probe over the central aisle.
        probes.push_back({{0.f, 4.f, 0.f}, {-6.f, 0.5f, -2.5f}, {6.f, 10.f, 2.5f}});
    } else if (scenePath.find("BistroInterior") != std::string::npos) {
        // Bistro cafe interior: three overlapping boxes (cafe floor + bar) so
        // the two-probe blend path is exercised.
        probes.push_back({{-2.f, 2.f, 0.f}, {-6.f, 0.2f, -3.f}, {2.f, 5.5f, 3.f}});
        probes.push_back({{4.f, 2.f, 0.5f}, {2.f, 0.2f, -2.5f}, {7.5f, 5.5f, 3.5f}});
        probes.push_back({{-2.f, 2.f, -1.5f}, {-5.f, 0.2f, -4.5f}, {1.5f, 5.5f, 0.5f}});
    }
    if (probes.size() > kMaxReflectionProbes) probes.resize(kMaxReflectionProbes);
    return probes;
}

std::string probeFilePathForScene(const std::string& scenePath) {
    if (scenePath.empty()) return resolveAssetPath("assets/probes/boxes.probes");
    return scenePath + ".probes";
}

} // namespace sr
