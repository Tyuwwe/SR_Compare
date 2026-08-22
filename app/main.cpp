// ============================================================================
// sr_compare entry point.  Dispatches to viewer / compare / bench / gui.
//
//   sr_compare viewer [--scene procedural|<gltf>] [--upscaler taa|none]
//                     [--render-scale 0.5] [--output 1920x1080]
//                     [--camera-path path.json] [--frames N] [--screenshot out.png]
//   sr_compare gui    interactive Dear ImGui front end (all modes)
// ============================================================================
#include "app/CliUtils.h"
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
                 "  --env-map <hdr>           static IBL environment map (default: sky atmosphere)\n"
                 "  --exposure <f>            manual display exposure (disables auto exposure)\n"
                 "  --bench <a,b,...>         start in the Bench tab and auto-run\n"
                 "viewer options:\n"
                 "  --scene <name|gltf path>     scene: boxes, sponza, or a glTF path (--list-scenes)\n"
                 "  --upscaler <name|none>         upscaler plugin (default taa; --list-upscalers shows all)\n"
                 "  --render-scale <f>               render resolution scale (default 0.5)\n"
                 "  --output <WxH>                   output resolution (default 1920x1080)\n"
                 "  --camera-path <json>             fixed camera path\n"
                 "  --env-map <hdr>                    static equirect HDR for IBL/skybox (default: procedural sky atmosphere)\n"
                 "  --sun-elev <deg>                   override the preset sun elevation (sky + key light)\n"
                 "  --sun-az <deg>                     override the preset sun azimuth (sky + key light)\n"
                 "  --frames <N>                     render N frames then exit\n"
                 "  --screenshot <out.png>           save the final frame as PNG\n"
                 "  --no-shadows                     disable CSM sun shadows\n"
                 "  --shadow-debug                   tint pixels per shadow cascade\n"
                 "  --exposure <f>                   manual display exposure (disables auto exposure)\n"
                 "  --no-bloom                       disable HDR bloom\n"
                 "  --no-lens-fx                     disable the present lens chain (CA/dirt/vignette/grain)\n"
                 "  --no-ssr                         disable opaque screen-space reflections\n"
                 "  --no-contact-shadows             disable screen-space contact shadows (sun)\n"
                 "  --no-volfog                      disable froxel volumetric fog\n"
                 "  --no-motion-blur                 disable HDR motion blur (McGuire 2012 tile-max gather)\n"
                 "  --no-dof                         disable depth of field (default on for --frames runs,\n"
                 "                                 off for interactive free-fly)\n"
                 "  --bake-probes                    bake reflection probes to the scene's .probes file, then exit\n"
                 "  --hdr                            HDR swapchain output (HDR10 PQ or scRGB probe, SDR fallback)\n"
                 "  --lut <file.cube>                3D LUT (17^3/33^3), log domain pre-ACES (default: identity)\n"
                 "  --temperature <K>                grading white balance (default 6500, scene preset may override)\n"
                 "  --tint <-1..1>                   grading green/magenta shift (default 0)\n"
                 "  --contrast <f>                   grading log-domain contrast (default 1)\n"
                 "  --saturation <f>                 grading saturation (default 1)\n");
}

int runViewer(int argc, char** argv) {
    sr::RendererOptions opts;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--scene") {
            opts.scenePath = sr::resolveSceneArg(sr::nextArg(i, argc, argv, "--scene"));
        } else if (a == "--list-scenes") {
            std::printf("available scenes:\n");
            for (const sr::SceneEntry& s : sr::listScenes()) {
                std::printf("  %-12s %s%s\n", s.alias.c_str(), s.description.c_str(),
                            s.available ? "" : "  [ASSET MISSING]");
            }
            return 0;
        } else if (a == "--upscaler") {
            opts.upscalerName = sr::nextArg(i, argc, argv, "--upscaler");
        } else if (a == "--list-upscalers") {
            std::printf("registered upscalers:\n");
            for (const std::string& n : sr::listUpscalers()) std::printf("  %s\n", n.c_str());
            return 0;
        } else if (a == "--render-scale") {
            if (!sr::parseRenderScale(sr::nextArg(i, argc, argv, "--render-scale"),
                                      opts.renderScale)) {
                std::fprintf(stderr, "invalid --render-scale value\n");
                return 1;
            }
        } else if (a == "--output") {
            if (!sr::parseResolution(sr::nextArg(i, argc, argv, "--output"),
                                     opts.displayWidth, opts.displayHeight)) {
                std::fprintf(stderr, "invalid --output resolution\n");
                return 1;
            }
        } else if (a == "--camera-path") {
            opts.cameraPath = sr::nextArg(i, argc, argv, "--camera-path");
        } else if (a == "--env-map") {
            opts.envMapPath = sr::nextArg(i, argc, argv, "--env-map");
        } else if (a == "--sun-elev") {
            opts.sunElevationDeg =
                static_cast<float>(std::atof(sr::nextArg(i, argc, argv, "--sun-elev")));
        } else if (a == "--sun-az") {
            opts.sunAzimuthDeg =
                static_cast<float>(std::atof(sr::nextArg(i, argc, argv, "--sun-az")));
        } else if (a == "--frames") {
            opts.frames = std::atoi(sr::nextArg(i, argc, argv, "--frames"));
        } else if (a == "--screenshot") {
            opts.screenshotPath = sr::nextArg(i, argc, argv, "--screenshot");
        } else if (a == "--frame-times") {
            opts.frameTimesPath = sr::nextArg(i, argc, argv, "--frame-times");
        } else if (a == "--vsync") {
            opts.vsync = true;
        } else if (a == "--no-shadows") {
            opts.shadows = false;
        } else if (a == "--shadow-debug") {
            opts.shadowDebug = true;
        } else if (a == "--exposure") {
            // Manual override: a given value switches off auto exposure.
            if (!sr::parseExposure(sr::nextArg(i, argc, argv, "--exposure"), opts.exposure)) {
                std::fprintf(stderr, "invalid --exposure value\n");
                return 1;
            }
            opts.autoExposure = false;
        } else if (a == "--no-bloom") {
            opts.bloom = false;
        } else if (a == "--no-lens-fx") {
            opts.lensFx = false;
        } else if (a == "--no-ssr") {
            opts.ssr = false;
        } else if (a == "--no-contact-shadows") {
            opts.contactShadows = false;
        } else if (a == "--no-volfog") {
            opts.volFog = false;
        } else if (a == "--no-motion-blur") {
            opts.motionBlur = false;
        } else if (a == "--no-dof") {
            opts.dof = false;
        } else if (a == "--bake-probes") {
            opts.bakeProbes = true;
        } else if (a == "--hdr") {
            opts.hdr = true;
        } else if (a == "--lut") {
            opts.lutPath = sr::nextArg(i, argc, argv, "--lut");
        } else if (a == "--temperature") {
            opts.grading.temperatureK =
                static_cast<float>(std::atof(sr::nextArg(i, argc, argv, "--temperature")));
        } else if (a == "--tint") {
            opts.grading.tint =
                static_cast<float>(std::atof(sr::nextArg(i, argc, argv, "--tint")));
        } else if (a == "--contrast") {
            opts.grading.contrast =
                static_cast<float>(std::atof(sr::nextArg(i, argc, argv, "--contrast")));
        } else if (a == "--saturation") {
            opts.grading.saturation =
                static_cast<float>(std::atof(sr::nextArg(i, argc, argv, "--saturation")));
        } else {
            std::fprintf(stderr, "unknown viewer option: %s\n", a.c_str());
            return 1;
        }
    }

    // Explicit --vsync wins; automated runs default to vsync off.
    opts.vsync = opts.vsync || (opts.frames < 0);
    // DOF default (see RendererOptions): on for bench/automation, off for
    // interactive free-fly.  Motion blur stays on in both.
    if (opts.frames < 0) opts.dof = false;

    sr::Renderer renderer;
    if (!renderer.init(opts)) {
        std::fprintf(stderr, "renderer init failed\n");
        return 1;
    }
    if (opts.bakeProbes) {
        // Offline reflection-probe bake: no frame loop, no bench involvement.
        const bool ok = renderer.bakeProbes();
        renderer.shutdown();
        return ok ? 0 : 1;
    }
    renderer.run();
    renderer.shutdown();
    return 0;
}

int runGui(int argc, char** argv) {
    sr::GuiOptions opts;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--scene") {
            opts.sceneArg = sr::nextArg(i, argc, argv, "--scene");
        } else if (a == "--upscaler") {
            opts.upscalerName = sr::nextArg(i, argc, argv, "--upscaler");
        } else if (a == "--compare") {
            opts.compareList = sr::nextArg(i, argc, argv, "--compare");
        } else if (a == "--compare-zoom") {
            opts.compareZoom = static_cast<float>(std::atof(sr::nextArg(i, argc, argv, "--compare-zoom")));
        } else if (a == "--compare-gt-ssaa") {
            opts.compareGtSsaa = true;
        } else if (a == "--env-map") {
            opts.envMapPath = sr::nextArg(i, argc, argv, "--env-map");
        } else if (a == "--bench") {
            opts.benchList = sr::nextArg(i, argc, argv, "--bench");
        } else if (a == "--render-scale") {
            if (!sr::parseRenderScale(sr::nextArg(i, argc, argv, "--render-scale"),
                                      opts.renderScale)) {
                std::fprintf(stderr, "invalid --render-scale value\n");
                return 1;
            }
        } else if (a == "--output") {
            if (!sr::parseResolution(sr::nextArg(i, argc, argv, "--output"),
                                     opts.displayW, opts.displayH)) {
                std::fprintf(stderr, "invalid --output resolution\n");
                return 1;
            }
        } else if (a == "--frames") {
            opts.frames = std::atoi(sr::nextArg(i, argc, argv, "--frames"));
        } else if (a == "--screenshot") {
            opts.screenshotPath = sr::nextArg(i, argc, argv, "--screenshot");
        } else if (a == "--exposure") {
            if (!sr::parseExposure(sr::nextArg(i, argc, argv, "--exposure"), opts.exposure)) {
                std::fprintf(stderr, "invalid --exposure value\n");
                return 1;
            }
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
