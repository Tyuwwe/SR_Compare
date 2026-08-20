#pragma once
// ============================================================================
// Scene registry — named built-in scenes selectable via --scene.
//   --scene boxes       procedural box room (no assets needed)
//   --scene sponza      bundled Sponza glTF (assets/sponza/Sponza.gltf)
//   --scene <path>      any glTF file path (passed through unchanged)
// ============================================================================
#include <string>
#include <vector>

namespace sr {

struct SceneEntry {
    std::string alias;        // name accepted by --scene
    std::string description;  // one-line human description
    std::string path;         // glTF path; empty = procedural generator
    bool available = true;    // false when the asset file is missing on disk
};

// All named scenes, procedural first.
std::vector<SceneEntry> listScenes();

// Resolve a --scene argument: known alias -> its path ("" for procedural);
// anything else is treated as a glTF path and returned unchanged.
// Prints a warning to stderr when a known alias's asset is missing.
std::string resolveSceneArg(const std::string& arg);

} // namespace sr
