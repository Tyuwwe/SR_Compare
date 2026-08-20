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

} // namespace sr
