#include "app/CliUtils.h"
#include "compare/CompareMode.h"

#include "compare/CompareApp.h"
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
                 "  --no-ssr                       disable opaque screen-space reflections\n"
                 "  --no-contact-shadows           disable screen-space contact shadows (sun)\n"
                 "  --no-volfog                    disable froxel volumetric fog\n"
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
        } else if (a == "--output") {
            if (!parseResolution(nextArg(i, argc, argv, "--output"),
                                 opts.displayWidth, opts.displayHeight)) {
                std::fprintf(stderr, "invalid --output resolution\n");
                return 1;
            }
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
        } else if (a == "--no-ssr") {
            opts.ssr = false;
        } else if (a == "--no-contact-shadows") {
            opts.contactShadows = false;
        } else if (a == "--no-volfog") {
            opts.volFog = false;
        } else if (a == "--shadow-debug") {
            opts.shadowDebug = true;
        } else if (a == "--env-map") {
            opts.envMapPath = nextArg(i, argc, argv, "--env-map");
        } else if (a == "--exposure") {
            // Manual override: a given value switches off auto exposure.
            if (!parseExposure(nextArg(i, argc, argv, "--exposure"), opts.exposure)) {
                std::fprintf(stderr, "invalid --exposure value\n");
                return 1;
            }
            opts.autoExposure = false;
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
