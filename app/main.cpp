// ============================================================================
// sr_compare entry point.  Dispatches to viewer / compare / bench / gui.
//
//   sr_compare viewer [--scene procedural|<gltf>] [--upscaler taa|none]
//                     [--render-scale 0.5] [--output 1920x1080]
//                     [--camera-path path.json] [--frames N] [--screenshot out.png]
//   sr_compare gui    interactive Dear ImGui front end (all modes)
// ============================================================================
#include "bench/BenchMode.h"
#include "compare/CompareMode.h"
#include "gui/GuiApp.h"
#include "renderer/Renderer.h"
#include "renderer/scene/SceneRegistry.h"
#include "upscalers/UpscalerFactory.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

void printUsage() {
    std::fprintf(stderr,
                 "usage: sr_compare <viewer|compare|bench|gui> [options]\n"
                 "gui: interactive Dear ImGui front end covering all modes\n"
                 "gui options (all optional; the UI defaults apply otherwise):\n"
                 "  --scene <name|gltf path>  --upscaler <name|none>  --render-scale <f>\n"
                 "  --output <WxH>            --frames <N> (auto-exit)  --screenshot <png>\n"
                 "  --compare <a,b,...>       start in the Compare tab with these columns\n"
                 "  --compare-zoom <f>        preset compare-tab zoom 1..16 (automation)\n"
                 "  --compare-gt-ssaa         preset the compare-tab GT 200%% SSAA checkbox\n"
                 "  --env-map <hdr>           IBL environment map (default san_giuseppe_bridge)\n"
                 "  --bench <a,b,...>         start in the Bench tab and auto-run\n"
                 "viewer options:\n"
                 "  --scene <name|gltf path>     scene: boxes, sponza, or a glTF path (--list-scenes)\n"
                 "  --upscaler <name|none>         upscaler plugin (default taa; --list-upscalers shows all)\n"
                 "  --render-scale <f>               render resolution scale (default 0.5)\n"
                 "  --output <WxH>                   output resolution (default 1920x1080)\n"
                 "  --camera-path <json>             fixed camera path\n"
                 "  --env-map <hdr>                    equirect HDR for IBL/skybox (default: Bistro san_giuseppe)\n"
                 "  --frames <N>                     render N frames then exit\n"
                 "  --screenshot <out.png>           save the final frame as PNG\n");
}

bool parseResolution(const char* s, uint32_t& w, uint32_t& h) {
    int iw = 0, ih = 0;
    if (::sscanf_s(s, "%dx%d", &iw, &ih) != 2 || iw <= 0 || ih <= 0) return false;
    w = static_cast<uint32_t>(iw);
    h = static_cast<uint32_t>(ih);
    return true;
}

int runViewer(int argc, char** argv) {
    sr::RendererOptions opts;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--scene") {
            opts.scenePath = sr::resolveSceneArg(next("--scene"));
        } else if (a == "--list-scenes") {
            std::printf("available scenes:\n");
            for (const sr::SceneEntry& s : sr::listScenes()) {
                std::printf("  %-12s %s%s\n", s.alias.c_str(), s.description.c_str(),
                            s.available ? "" : "  [ASSET MISSING]");
            }
            return 0;
        } else if (a == "--upscaler") {
            opts.upscalerName = next("--upscaler");
        } else if (a == "--list-upscalers") {
            std::printf("registered upscalers:\n");
            for (const std::string& n : sr::listUpscalers()) std::printf("  %s\n", n.c_str());
            return 0;
        } else if (a == "--render-scale") {
            opts.renderScale = static_cast<float>(std::atof(next("--render-scale")));
        } else if (a == "--output") {
            if (!parseResolution(next("--output"), opts.displayWidth, opts.displayHeight)) {
                std::fprintf(stderr, "invalid --output resolution\n");
                return 1;
            }
        } else if (a == "--camera-path") {
            opts.cameraPath = next("--camera-path");
        } else if (a == "--env-map") {
            opts.envMapPath = next("--env-map");
        } else if (a == "--frames") {
            opts.frames = std::atoi(next("--frames"));
        } else if (a == "--screenshot") {
            opts.screenshotPath = next("--screenshot");
        } else if (a == "--frame-times") {
            opts.frameTimesPath = next("--frame-times");
        } else if (a == "--vsync") {
            opts.vsync = true;
        } else {
            std::fprintf(stderr, "unknown viewer option: %s\n", a.c_str());
            return 1;
        }
    }

    // Interactive keeps vsync; automated runs render as fast as possible.
    opts.vsync = opts.frames < 0;

    sr::Renderer renderer;
    if (!renderer.init(opts)) {
        std::fprintf(stderr, "renderer init failed\n");
        return 1;
    }
    renderer.run();
    renderer.shutdown();
    return 0;
}

int runGui(int argc, char** argv) {
    sr::GuiOptions opts;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--scene") {
            opts.sceneArg = next("--scene");
        } else if (a == "--upscaler") {
            opts.upscalerName = next("--upscaler");
        } else if (a == "--compare") {
            opts.compareList = next("--compare");
        } else if (a == "--compare-zoom") {
            opts.compareZoom = static_cast<float>(std::atof(next("--compare-zoom")));
        } else if (a == "--compare-gt-ssaa") {
            opts.compareGtSsaa = true;
        } else if (a == "--env-map") {
            opts.envMapPath = next("--env-map");
        } else if (a == "--bench") {
            opts.benchList = next("--bench");
        } else if (a == "--render-scale") {
            opts.renderScale = static_cast<float>(std::atof(next("--render-scale")));
        } else if (a == "--output") {
            if (!parseResolution(next("--output"), opts.displayW, opts.displayH)) {
                std::fprintf(stderr, "invalid --output resolution\n");
                return 1;
            }
        } else if (a == "--frames") {
            opts.frames = std::atoi(next("--frames"));
        } else if (a == "--screenshot") {
            opts.screenshotPath = next("--screenshot");
        } else {
            std::fprintf(stderr, "unknown gui option: %s\n", a.c_str());
            return 1;
        }
    }

    // The dlss/xess/nss device-requirement hooks and slInit are gated on
    // command-line substrings; the GUI selects plugins at runtime, so force
    // every gate open BEFORE any Vulkan object is created.
    sr::setAllPluginsEnabled(true);

    sr::GuiApp app;
    if (!app.init(opts)) {
        std::fprintf(stderr, "gui init failed\n");
        return 1;
    }
    app.run();
    app.shutdown();
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage();
        return 1;
    }

    const std::string mode = argv[1];
    if (mode == "viewer") return runViewer(argc - 2, argv + 2);
    if (mode == "compare") return sr::runCompareMode(argc - 2, argv + 2);
    if (mode == "bench") return sr::runBenchMode(argc - 2, argv + 2);
    if (mode == "gui") return runGui(argc - 2, argv + 2);

    printUsage();
    return 1;
}
