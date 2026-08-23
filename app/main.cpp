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
#include "renderer/core/EngineConfig.h"
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
                 "  --bloom                          enable HDR bloom (default off; --no-bloom accepted\n"
                 "                                 for compatibility)\n"
                 "  --no-lens-fx                     disable the present lens chain (CA/dirt/vignette/grain)\n"
                 "  --ssr                            enable opaque screen-space reflections (default off;\n"
                 "                                 --no-ssr accepted for compatibility)\n"
                 "  --ssr-strength <0..1>            global SSR weight scale (default 0.6)\n"
                 "  --no-contact-shadows             disable screen-space contact shadows (sun)\n"
                 "  --no-volfog                      disable froxel volumetric fog\n"
                 "  --motion-blur                    enable HDR motion blur (McGuire 2012 tile-max gather;\n"
                 "                                 default off; --no-motion-blur accepted for compatibility)\n"
                 "  --dof                            enable depth of field (default off; --no-dof accepted\n"
                 "                                 for compatibility)\n"
                 "  --dof-focus <m>                  DOF focus distance in metres (0 = auto-focus on\n"
                 "                                 the screen centre; default 0)\n"
                 "  --dof-fstop <f>                  DOF aperture f-stop 0.5..64 (default 4; smaller =\n"
                 "                                 wider bokeh)\n"
                 "  --dof-max-blur <px>              DOF max bokeh radius at 1080p, 1..64 (default 12)\n"
                 "  --bake-probes                    bake reflection probes to the scene's .probes file, then exit\n"
                 "  --hdr                            HDR swapchain output (HDR10 PQ or scRGB probe, SDR fallback)\n"
                 "  --lut <file.cube>                3D LUT (17^3/33^3), log domain pre-ACES (default: identity)\n"
                 "  --temperature <K>                grading white balance (default 6500, scene preset may override)\n"
                 "  --tint <-1..1>                   grading green/magenta shift (default 0)\n"
                 "  --contrast <f>                   grading log-domain contrast (default 1)\n"
                 "  --saturation <f>                 grading saturation (default 1)\n"
                 "\n"
                 "engine.toml (next to the exe, see engine.toml.example) supplies defaults\n"
                 "for the window/renderer/effect/grading/sun options in every mode; explicit\n"
                 "CLI flags always win.  The GUI hot-reloads it (~1 s) for per-frame options,\n"
                 "auto-creates it when missing and auto-saves UI changes back to it.\n");
}

int runViewer(int argc, char** argv) {
    sr::RendererOptions opts;
    uint64_t cliMask = sr::cli::kNone; // explicit-CLI mask: engine.toml never overrides these
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
            cliMask |= sr::cli::kRenderScale;
        } else if (a == "--output") {
            if (!sr::parseResolution(sr::nextArg(i, argc, argv, "--output"),
                                     opts.displayWidth, opts.displayHeight)) {
                std::fprintf(stderr, "invalid --output resolution\n");
                return 1;
            }
            cliMask |= sr::cli::kOutput;
        } else if (a == "--camera-path") {
            opts.cameraPath = sr::nextArg(i, argc, argv, "--camera-path");
        } else if (a == "--env-map") {
            opts.envMapPath = sr::nextArg(i, argc, argv, "--env-map");
            cliMask |= sr::cli::kEnvMap;
        } else if (a == "--sun-elev") {
            opts.sunElevationDeg =
                static_cast<float>(std::atof(sr::nextArg(i, argc, argv, "--sun-elev")));
            cliMask |= sr::cli::kSunElev;
        } else if (a == "--sun-az") {
            opts.sunAzimuthDeg =
                static_cast<float>(std::atof(sr::nextArg(i, argc, argv, "--sun-az")));
            cliMask |= sr::cli::kSunAz;
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
            cliMask |= sr::cli::kShadows;
        } else if (a == "--shadow-debug") {
            opts.shadowDebug = true;
        } else if (a == "--exposure") {
            // Manual override: a given value switches off auto exposure.
            if (!sr::parseExposure(sr::nextArg(i, argc, argv, "--exposure"), opts.exposure)) {
                std::fprintf(stderr, "invalid --exposure value\n");
                return 1;
            }
            opts.autoExposure = false;
            cliMask |= sr::cli::kExposure;
        } else if (a == "--bloom") {
            opts.bloom = true;
            cliMask |= sr::cli::kBloom;
        } else if (a == "--no-bloom") {
            // Accepted for compatibility (bloom is off by default now).
            opts.bloom = false;
            cliMask |= sr::cli::kBloom;
        } else if (a == "--no-lens-fx") {
            opts.lensFx = false;
            cliMask |= sr::cli::kLensFx;
        } else if (a == "--ssr") {
            opts.ssr = true;
            cliMask |= sr::cli::kSsr;
        } else if (a == "--no-ssr") {
            opts.ssr = false;
            cliMask |= sr::cli::kSsr;
        } else if (a == "--ssr-strength") {
            if (!sr::parseUnitInterval(sr::nextArg(i, argc, argv, "--ssr-strength"),
                                       opts.ssrStrength)) {
                std::fprintf(stderr, "invalid --ssr-strength value\n");
                return 1;
            }
            cliMask |= sr::cli::kSsrStrength;
        } else if (a == "--no-contact-shadows") {
            opts.contactShadows = false;
            cliMask |= sr::cli::kContactShadows;
        } else if (a == "--no-volfog") {
            opts.volFog = false;
            cliMask |= sr::cli::kVolFog;
        } else if (a == "--motion-blur") {
            opts.motionBlur = true;
            cliMask |= sr::cli::kMotionBlur;
        } else if (a == "--no-motion-blur") {
            opts.motionBlur = false;
            cliMask |= sr::cli::kMotionBlur;
        } else if (a == "--dof") {
            opts.dof = true;
            cliMask |= sr::cli::kDof;
        } else if (a == "--no-dof") {
            opts.dof = false;
            cliMask |= sr::cli::kDof;
        } else if (a == "--dof-focus") {
            if (!sr::parseDofFocus(sr::nextArg(i, argc, argv, "--dof-focus"), opts.dofFocus)) {
                std::fprintf(stderr, "invalid --dof-focus value\n");
                return 1;
            }
            cliMask |= sr::cli::kDofFocus;
        } else if (a == "--dof-fstop") {
            if (!sr::parseDofFstop(sr::nextArg(i, argc, argv, "--dof-fstop"), opts.dofFstop)) {
                std::fprintf(stderr, "invalid --dof-fstop value\n");
                return 1;
            }
            cliMask |= sr::cli::kDofFstop;
        } else if (a == "--dof-max-blur") {
            if (!sr::parseDofMaxBlur(sr::nextArg(i, argc, argv, "--dof-max-blur"),
                                     opts.dofMaxBlurPx)) {
                std::fprintf(stderr, "invalid --dof-max-blur value\n");
                return 1;
            }
            cliMask |= sr::cli::kDofMaxBlur;
        } else if (a == "--bake-probes") {
            opts.bakeProbes = true;
        } else if (a == "--hdr") {
            opts.hdr = true;
            cliMask |= sr::cli::kHdr;
        } else if (a == "--lut") {
            opts.lutPath = sr::nextArg(i, argc, argv, "--lut");
            cliMask |= sr::cli::kLut;
        } else if (a == "--temperature") {
            opts.grading.temperatureK =
                static_cast<float>(std::atof(sr::nextArg(i, argc, argv, "--temperature")));
            cliMask |= sr::cli::kGradingTemp;
        } else if (a == "--tint") {
            opts.grading.tint =
                static_cast<float>(std::atof(sr::nextArg(i, argc, argv, "--tint")));
            cliMask |= sr::cli::kGradingTint;
        } else if (a == "--contrast") {
            opts.grading.contrast =
                static_cast<float>(std::atof(sr::nextArg(i, argc, argv, "--contrast")));
            cliMask |= sr::cli::kGradingContrast;
        } else if (a == "--saturation") {
            opts.grading.saturation =
                static_cast<float>(std::atof(sr::nextArg(i, argc, argv, "--saturation")));
            cliMask |= sr::cli::kGradingSat;
        } else {
            std::fprintf(stderr, "unknown viewer option: %s\n", a.c_str());
            return 1;
        }
    }

    // engine.toml (exe-relative) fills every option the CLI did not set
    // explicitly; a missing file leaves the code defaults untouched.
    {
        sr::EngineConfig cfg;
        if (sr::loadEngineConfig(cfg)) {
            sr::EngineConfigLog log;
            applyEngineConfig(opts, cfg, cliMask, log);
            log.flush(" viewer:");
        }
    }

    // Explicit --vsync wins; automated runs default to vsync off.
    opts.vsync = opts.vsync || (opts.frames < 0);
    // Motion blur and DOF default off in every mode (see RendererOptions);
    // --motion-blur / --dof opt in, --no-motion-blur / --no-dof are accepted
    // for compatibility with older scripts.

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
    uint64_t cliMask = sr::cli::kNone; // explicit-CLI mask: engine.toml never overrides these
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
            cliMask |= sr::cli::kEnvMap;
        } else if (a == "--bench") {
            opts.benchList = sr::nextArg(i, argc, argv, "--bench");
        } else if (a == "--render-scale") {
            if (!sr::parseRenderScale(sr::nextArg(i, argc, argv, "--render-scale"),
                                      opts.renderScale)) {
                std::fprintf(stderr, "invalid --render-scale value\n");
                return 1;
            }
            cliMask |= sr::cli::kRenderScale;
        } else if (a == "--output") {
            if (!sr::parseResolution(sr::nextArg(i, argc, argv, "--output"),
                                     opts.displayW, opts.displayH)) {
                std::fprintf(stderr, "invalid --output resolution\n");
                return 1;
            }
            cliMask |= sr::cli::kOutput;
        } else if (a == "--frames") {
            opts.frames = std::atoi(sr::nextArg(i, argc, argv, "--frames"));
        } else if (a == "--screenshot") {
            opts.screenshotPath = sr::nextArg(i, argc, argv, "--screenshot");
        } else if (a == "--exposure") {
            if (!sr::parseExposure(sr::nextArg(i, argc, argv, "--exposure"), opts.exposure)) {
                std::fprintf(stderr, "invalid --exposure value\n");
                return 1;
            }
            cliMask |= sr::cli::kExposure;
        } else {
            std::fprintf(stderr, "unknown gui option: %s\n", a.c_str());
            return 1;
        }
    }

    // engine.toml (exe-relative): fills launch options the CLI did not set;
    // the remaining keys become the GUI's initial effect/grading/sun state
    // and are hot-reloaded by GuiApp (see GuiApp::pollEngineConfig).
    opts.engineCfgCli = cliMask;
    if (sr::loadEngineConfig(opts.engineCfg)) {
        sr::EngineConfigLog log;
        sr::cfgTake(opts.renderScale, opts.engineCfg.renderScale, cliMask, sr::cli::kRenderScale,
                    "render_scale", log);
        sr::cfgTake(opts.envMapPath, opts.engineCfg.envMap, cliMask, sr::cli::kEnvMap, "env_map", log);
        sr::cfgTake(opts.exposure, opts.engineCfg.exposure, cliMask, sr::cli::kExposure, "exposure",
                    log);
        log.flush(" gui:");
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
