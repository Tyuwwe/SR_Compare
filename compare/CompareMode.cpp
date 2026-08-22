#include "app/CliUtils.h"
#include "compare/CompareMode.h"

#include "compare/CompareApp.h"
#include "renderer/core/EngineConfig.h"
#include "renderer/scene/SceneRegistry.h"
#include "upscalers/UpscalerFactory.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace sr {

namespace {

void printCompareUsage() {
    std::fprintf(stderr,
                 "usage: sr_compare compare [options]\n"
                 "  --scene <name|gltf path>     scene: boxes, sponza, or a glTF path\n"
                 "  --upscalers <a,b,...>          column subset & order (default: all registered)\n"
                 "  --render-scale <f>             render resolution scale (default 0.5)\n"
                 "  --output <WxH>                 output resolution (default 1920x1080)\n"
                 "  --camera-path <json>           fixed camera path\n"
                 "  --frames <N>                   render N frames then exit\n"
                 "  --screenshot <out.png>         save the final frame as PNG\n"
                 "  --metric-interval <N>          frames between metric readbacks (default 15)\n"
                 "  --gt-ssaa                      GT rendered at 2x, downsampled to output res\n"
                 "  --no-shadows                   disable CSM sun shadows\n"
                 "  --ssr                          enable opaque screen-space reflections (default off;\n"
                 "                               --no-ssr accepted for compatibility)\n"
                 "  --ssr-strength <0..1>          global SSR weight scale (default 0.6)\n"
                 "  --no-contact-shadows           disable screen-space contact shadows (sun)\n"
                 "  --no-volfog                    disable froxel volumetric fog\n"
                 "  --motion-blur                  enable motion blur (default off, all paths;\n"
                 "                               --no-motion-blur accepted for compatibility)\n"
                 "  --dof                          enable depth of field (default off, all paths;\n"
                 "                               --no-dof accepted for compatibility)\n"
                 "  --dof-focus <m>                DOF focus distance in metres (0 = auto-focus on the\n"
                 "                               screen centre; default 0; all paths)\n"
                 "  --dof-fstop <f>                DOF aperture f-stop 0.5..64 (default 4; smaller =\n"
                 "                               wider bokeh; all paths)\n"
                 "  --dof-max-blur <px>            DOF max bokeh radius at 1080p, 1..64 (default 12;\n"
                 "                               all paths)\n"
                 "  --no-lens-fx                   disable the compose lens chain (CA/vignette/grain)\n"
                 "                                 (grading is the neutral default set; HDR output is\n"
                 "                                 viewer/GUI-only — compare/bench stay SDR)\n"
                 "  --shadow-debug                 tint pixels per shadow cascade\n"
                 "  --env-map <hdr>                static IBL environment map (default: sky atmosphere)\n"
                 "  --exposure <f>                 manual display exposure (disables auto exposure)\n"
                 "  --zoom <f>                     compare-view zoom 1..16 (default 1)\n"
                 "  --zoom-center <u,v>            zoom window center, normalized (default 0.5,0.5)\n"
                 "  --list-upscalers               print registered upscalers and exit\n");
}

} // namespace

int runCompareMode(int argc, char** argv) {
    CompareOptions opts;
    uint64_t cliMask = cli::kNone; // explicit-CLI mask: engine.toml never overrides these
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--scene") {
            opts.scenePath = sr::resolveSceneArg(nextArg(i, argc, argv, "--scene"));
        } else if (a == "--upscalers") {
            opts.upscalerNames = splitCsv(nextArg(i, argc, argv, "--upscalers"));
        } else if (a == "--list-upscalers") {
            std::printf("registered upscalers:\n");
            for (const std::string& n : listUpscalers()) std::printf("  %s\n", n.c_str());
            return 0;
        } else if (a == "--render-scale") {
            if (!parseRenderScale(nextArg(i, argc, argv, "--render-scale"), opts.renderScale)) {
                std::fprintf(stderr, "invalid --render-scale value\n");
                return 1;
            }
            cliMask |= cli::kRenderScale;
        } else if (a == "--output") {
            if (!parseResolution(nextArg(i, argc, argv, "--output"),
                                 opts.displayWidth, opts.displayHeight)) {
                std::fprintf(stderr, "invalid --output resolution\n");
                return 1;
            }
            cliMask |= cli::kOutput;
        } else if (a == "--camera-path") {
            opts.cameraPath = nextArg(i, argc, argv, "--camera-path");
        } else if (a == "--frames") {
            opts.frames = std::atoi(nextArg(i, argc, argv, "--frames"));
        } else if (a == "--screenshot") {
            opts.screenshotPath = nextArg(i, argc, argv, "--screenshot");
        } else if (a == "--metric-interval") {
            opts.metricInterval = std::atoi(nextArg(i, argc, argv, "--metric-interval"));
        } else if (a == "--gt-ssaa") {
            opts.gtSsaa = true;
        } else if (a == "--no-shadows") {
            opts.shadows = false;
            cliMask |= cli::kShadows;
        } else if (a == "--ssr") {
            opts.ssr = true;
            cliMask |= cli::kSsr;
        } else if (a == "--no-ssr") {
            opts.ssr = false;
            cliMask |= cli::kSsr;
        } else if (a == "--ssr-strength") {
            if (!parseUnitInterval(nextArg(i, argc, argv, "--ssr-strength"), opts.ssrStrength)) {
                std::fprintf(stderr, "invalid --ssr-strength value\n");
                return 1;
            }
            cliMask |= cli::kSsrStrength;
        } else if (a == "--no-contact-shadows") {
            opts.contactShadows = false;
            cliMask |= cli::kContactShadows;
        } else if (a == "--no-volfog") {
            opts.volFog = false;
            cliMask |= cli::kVolFog;
        } else if (a == "--motion-blur") {
            opts.motionBlur = true;
            cliMask |= cli::kMotionBlur;
        } else if (a == "--no-motion-blur") {
            opts.motionBlur = false;
            cliMask |= cli::kMotionBlur;
        } else if (a == "--dof") {
            opts.dof = true;
            cliMask |= cli::kDof;
        } else if (a == "--no-dof") {
            opts.dof = false;
            cliMask |= cli::kDof;
        } else if (a == "--dof-focus") {
            if (!parseDofFocus(nextArg(i, argc, argv, "--dof-focus"), opts.dofFocus)) {
                std::fprintf(stderr, "invalid --dof-focus value\n");
                return 1;
            }
            cliMask |= cli::kDofFocus;
        } else if (a == "--dof-fstop") {
            if (!parseDofFstop(nextArg(i, argc, argv, "--dof-fstop"), opts.dofFstop)) {
                std::fprintf(stderr, "invalid --dof-fstop value\n");
                return 1;
            }
            cliMask |= cli::kDofFstop;
        } else if (a == "--dof-max-blur") {
            if (!parseDofMaxBlur(nextArg(i, argc, argv, "--dof-max-blur"), opts.dofMaxBlurPx)) {
                std::fprintf(stderr, "invalid --dof-max-blur value\n");
                return 1;
            }
            cliMask |= cli::kDofMaxBlur;
        } else if (a == "--no-lens-fx") {
            opts.lensFx = false;
            cliMask |= cli::kLensFx;
        } else if (a == "--shadow-debug") {
            opts.shadowDebug = true;
        } else if (a == "--env-map") {
            opts.envMapPath = nextArg(i, argc, argv, "--env-map");
            cliMask |= cli::kEnvMap;
        } else if (a == "--exposure") {
            // Manual override: a given value switches off auto exposure.
            if (!parseExposure(nextArg(i, argc, argv, "--exposure"), opts.exposure)) {
                std::fprintf(stderr, "invalid --exposure value\n");
                return 1;
            }
            opts.autoExposure = false;
            cliMask |= cli::kExposure;
        } else if (a == "--zoom") {
            opts.zoom = static_cast<float>(std::atof(nextArg(i, argc, argv, "--zoom")));
        } else if (a == "--zoom-center") {
            const char* v = nextArg(i, argc, argv, "--zoom-center");
            float u = 0.f, vv = 0.f;
            if (::sscanf_s(v, "%f,%f", &u, &vv) != 2) {
                std::fprintf(stderr, "invalid --zoom-center (want u,v)\n");
                return 1;
            }
            opts.zoomCenterU = u;
            opts.zoomCenterV = vv;
        } else {
            std::fprintf(stderr, "unknown compare option: %s\n", a.c_str());
            printCompareUsage();
            return 1;
        }
    }

    // engine.toml (exe-relative) fills every option the CLI did not set
    // explicitly; a missing file leaves the code defaults untouched.
    {
        EngineConfig cfg;
        if (loadEngineConfig(cfg)) {
            EngineConfigLog log;
            cfgTake(opts.renderScale, cfg.renderScale, cliMask, cli::kRenderScale, "render_scale",
                    log);
            cfgTake(opts.shadows, cfg.shadows, cliMask, cli::kShadows, "shadows", log);
            cfgTake(opts.ssr, cfg.ssr, cliMask, cli::kSsr, "ssr", log);
            cfgTake(opts.ssrStrength, cfg.ssrStrength, cliMask, cli::kSsrStrength, "ssr_strength",
                    log);
            cfgTake(opts.contactShadows, cfg.contactShadows, cliMask, cli::kContactShadows,
                    "contact_shadows", log);
            cfgTake(opts.volFog, cfg.volFog, cliMask, cli::kVolFog, "volfog", log);
            cfgTake(opts.bloom, cfg.bloom, cliMask, cli::kBloom, "bloom", log);
            cfgTake(opts.motionBlur, cfg.motionBlur, cliMask, cli::kMotionBlur, "motion_blur",
                    log);
            cfgTake(opts.dof, cfg.dof, cliMask, cli::kDof, "dof", log);
            cfgTake(opts.dofFocus, cfg.dofFocus, cliMask, cli::kDofFocus, "dof_focus", log);
            cfgTake(opts.dofFstop, cfg.dofFstop, cliMask, cli::kDofFstop, "dof_fstop", log);
            cfgTake(opts.dofMaxBlurPx, cfg.dofMaxBlur, cliMask, cli::kDofMaxBlur, "dof_max_blur",
                    log);
            cfgTake(opts.lensFx, cfg.lensFx, cliMask, cli::kLensFx, "lens_fx", log);
            cfgTake(opts.envMapPath, cfg.envMap, cliMask, cli::kEnvMap, "env_map", log);
            if (cfg.exposure && (cliMask & cli::kExposure) == 0) {
                // Same semantics as CLI --exposure: manual value, auto exposure off.
                opts.exposure = *cfg.exposure;
                opts.autoExposure = false;
                log.add("exposure", opts.exposure);
            }
            cfgTake(opts.exposureMinEV, cfg.exposureMinEV, cliMask, cli::kExposure,
                    "exposure_min_ev", log);
            cfgTake(opts.exposureMaxEV, cfg.exposureMaxEV, cliMask, cli::kExposure,
                    "exposure_max_ev", log);
            log.flush(" compare:");
        }
    }

    // Interactive keeps vsync; automated runs render as fast as possible.
    opts.vsync = opts.frames < 0;

    CompareApp app;
    if (!app.init(opts)) {
        std::fprintf(stderr, "compare init failed\n");
        return 1;
    }
    app.run();
    app.shutdown();
    return 0;
}

} // namespace sr
