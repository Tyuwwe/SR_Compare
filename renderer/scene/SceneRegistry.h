#pragma once
// ============================================================================
// Scene registry — named built-in scenes selectable via --scene.
//   --scene boxes       procedural box room (no assets needed)
//   --scene sponza      bundled Sponza glTF (assets/sponza/Sponza.gltf)
//   --scene <path>      any glTF file path (passed through unchanged)
// ============================================================================
#include "renderer/math/Math.h"
#include "renderer/scene/Scene.h"

#include <string>
#include <vector>

namespace sr {

struct SceneEntry {
    std::string alias;        // name accepted by --scene
    std::string description;  // one-line human description
    std::string path;         // glTF path; empty = procedural generator
    bool available = true;    // false when the asset file is missing on disk
};

// Per-scene default look (sun + optional fill + IBL + display exposure).
// Applied by the glTF/procedural loaders (full preset when the file has no
// lights; Bistro also replaces authored directionals with the golden-hour sun
// while keeping point lights) and by the GUI lighting sliders on scene load.
struct LightingPreset {
    float sunElevationDeg = 65.3f;
    float sunAzimuthDeg = 49.4f;
    float sunIntensity = 3.f;
    Vec3 sunColor{1.f, 0.95f, 0.85f};
    bool sunEnabled = true;
    bool fillEnabled = true;   // defaultLights() blue point fill
    float iblIntensity = 1.f;
    float exposure = 1.f;      // display-domain multiplier (ACES input)
    // When true and the glTF has KHR_lights, authored directionals are
    // replaced by this preset's sun so Bistro keeps the golden-hour key.
    bool preferPresetSun = false;
};

LightingPreset defaultLightingPreset();
LightingPreset goldenHourPreset(); // Bistro-exterior look; GUI "golden hour" button
LightingPreset lightingPresetForScene(const std::string& scenePath);
std::vector<Light> lightsFromPreset(const LightingPreset& p);

// All named scenes, procedural first.
std::vector<SceneEntry> listScenes();

// Resolve a --scene argument: known alias -> its path ("" for procedural);
// anything else is treated as a glTF path and returned unchanged.
// Prints a warning to stderr when a known alias's asset is missing.
std::string resolveSceneArg(const std::string& arg);

// Per-scene free-fly starting pose.  Returns false when the scene has no
// registered pose (callers then use their generic default).  The generic
// default sits outside the Bistro street geometry (inside a wall), which is
// why BistroExterior has an explicit one.
bool initialCameraPose(const std::string& scenePath, Vec3& pos, Vec3& fwd);

// Hand-placed reflection probe placements per scene (Phase 4c-2; scenePath is
// the resolved glTF path, "" = procedural boxes).  Empty = no probes.
std::vector<ReflectionProbe> reflectionProbesForScene(const std::string& scenePath);

// Bake file location for a scene: alongside the glTF (<path>.probes), or
// assets/probes/boxes.probes for the built-in procedural scene.
std::string probeFilePathForScene(const std::string& scenePath);

} // namespace sr
