#include "renderer/scene/SceneRegistry.h"

#include "renderer/core/PathUtil.h"
#include "renderer/ibl/SkyAtmosphere.h"

#include <algorithm>
#include <cstdio>

namespace sr {

namespace {
// Selected procedural generator variant, set by resolveSceneArg ("" / any
// non-alias = "boxes").  Read by Scene::loadProcedural so a second built-in
// scene can be registered without touching the host's load fallback chain.
std::string g_proceduralVariant = "boxes";
} // namespace

const std::string& proceduralSceneVariant() { return g_proceduralVariant; }

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
    scenes.push_back({"ssrlab", "SSR mirror test lab (built-in, no assets)", "", true});
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
    g_proceduralVariant = (arg == "ssrlab") ? "ssrlab" : "boxes";
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
        if (proceduralSceneVariant() == "ssrlab") {
            // ssrlab mirror room: one probe over the room in front of the
            // mirror wall (z = -2, x [-8,8], y [0,3]; room shell x +-13,
            // z [-2.6,12.6]) so the pane's SSR-miss fallback reflects this
            // room's lamps/benches/colour boxes instead of the boxes bake.
            // z-min reaches the back wall so the pane pixels sit ~0.6 m
            // inside the box (near-full influence through the 0.75 m ramp).
            probes.push_back({{0.f, 2.5f, 3.f}, {-12.5f, 0.05f, -2.55f}, {12.5f, 5.9f, 12.2f}});
        } else {
            // boxes: one probe covering the whole 20x10x20 room (walls at
            // |x| = 10, z = +-10, floor y = 0; see ProceduralScene.cpp).
            probes.push_back({{0.f, 2.5f, 0.f}, {-9.5f, 0.05f, -9.5f}, {9.5f, 9.5f, 9.5f}});
        }
    } else if (scenePath.find("Sponza") != std::string::npos) {
        // Sponza atrium: one demo probe over the central aisle.
        probes.push_back({{0.f, 4.f, 0.f}, {-6.f, 0.5f, -2.5f}, {6.f, 10.f, 2.5f}});
    } else if (scenePath.find("BistroInterior") != std::string::npos) {
        // Bistro cafe interior: three overlapping boxes (cafe floor + bar) so
        // the two-probe blend path is exercised.
        probes.push_back({{-2.f, 2.f, 0.f}, {-6.f, 0.2f, -3.f}, {2.f, 5.5f, 3.f}});
        probes.push_back({{4.f, 2.f, 0.5f}, {2.f, 0.2f, -2.5f}, {7.5f, 5.5f, 3.5f}});
        probes.push_back({{-2.f, 2.f, -1.5f}, {-5.f, 0.2f, -4.5f}, {1.5f, 5.5f, 0.5f}});
    } else if (scenePath.find("BistroExterior") != std::string::npos) {
        // Bistro street: four boxes along the two streets.  The corner boxes
        // cover the cafe storefront mirror glass (glTF material bounds:
        // MASTER_Glass_Exterior x [-1.8,14.1] z [-16.4,7.6]); the capture
        // points sit ~1.5 m off the glass line on the street side so the
        // baked cubes see the reflected street, not the pane plane.  Neighbour
        // boxes overlap ~2 m so the influence ramps cross-fade instead of
        // dipping to the global env at a shared face.
        // Box 0 reaches across the street to the opposite facade (the
        // paris_building_11 wall at z ~19): the storefront panes' SSR misses
        // are exactly the rays heading that way (the facade is behind the
        // camera), and the parallax box projection re-aims them onto the
        // face they hit — with the old z-max of 8.5 they landed on the +X
        // side face and every miss read as a flat grey wall instead of the
        // building across the street.
        probes.push_back({{5.f, 3.2f, 4.5f}, {-4.5f, 0.1f, -4.f}, {14.5f, 7.5f, 19.f}});
        probes.push_back({{5.f, 3.2f, -8.5f}, {-4.5f, 0.1f, -17.f}, {14.5f, 7.5f, -2.f}});
        probes.push_back({{-12.f, 3.5f, 4.f}, {-30.f, 0.1f, -6.f}, {-2.5f, 10.f, 12.f}});
        probes.push_back({{25.f, 3.5f, -1.f}, {13.5f, 0.1f, -12.f}, {45.f, 9.f, 10.f}});
    }
    if (probes.size() > kMaxReflectionProbes) probes.resize(kMaxReflectionProbes);
    return probes;
}

std::string probeFilePathForScene(const std::string& scenePath) {
    // The two procedural scenes share the empty scenePath but need distinct
    // bakes: a mirror-room capture is wrong content for the box room.
    if (scenePath.empty())
        return resolveAssetPath(proceduralSceneVariant() == "ssrlab"
                                    ? "assets/probes/ssrlab.probes"
                                    : "assets/probes/boxes.probes");
    return scenePath + ".probes";
}

} // namespace sr
