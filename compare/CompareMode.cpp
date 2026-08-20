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

bool parseResolution(const char* s, uint32_t& w, uint32_t& h) {
    int iw = 0, ih = 0;
    if (::sscanf_s(s, "%dx%d", &iw, &ih) != 2 || iw <= 0 || ih <= 0) return false;
    w = static_cast<uint32_t>(iw);
    h = static_cast<uint32_t>(ih);
    return true;
}

std::vector<std::string> splitCsv(const std::string& s) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        const size_t comma = s.find(',', start);
        std::string item = s.substr(start, comma == std::string::npos ? comma : comma - start);
        if (!item.empty()) out.push_back(item);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

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
                 "  --env-map <hdr>                IBL environment map (default san_giuseppe_bridge)\n"
                 "  --zoom <f>                     compare-view zoom 1..16 (default 1)\n"
                 "  --zoom-center <u,v>            zoom window center, normalized (default 0.5,0.5)\n"
                 "  --list-upscalers               print registered upscalers and exit\n");
}

} // namespace

int runCompareMode(int argc, char** argv) {
    CompareOptions opts;
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
        } else if (a == "--upscalers") {
            opts.upscalerNames = splitCsv(next("--upscalers"));
        } else if (a == "--list-upscalers") {
            std::printf("registered upscalers:\n");
            for (const std::string& n : listUpscalers()) std::printf("  %s\n", n.c_str());
            return 0;
        } else if (a == "--render-scale") {
            opts.renderScale = static_cast<float>(std::atof(next("--render-scale")));
            if (opts.renderScale <= 0.0f || opts.renderScale > 1.0f) {
                std::fprintf(stderr, "invalid --render-scale value\n");
                return 1;
            }
        } else if (a == "--output") {
            if (!parseResolution(next("--output"), opts.displayWidth, opts.displayHeight)) {
                std::fprintf(stderr, "invalid --output resolution\n");
                return 1;
            }
        } else if (a == "--camera-path") {
            opts.cameraPath = next("--camera-path");
        } else if (a == "--frames") {
            opts.frames = std::atoi(next("--frames"));
        } else if (a == "--screenshot") {
            opts.screenshotPath = next("--screenshot");
        } else if (a == "--metric-interval") {
            opts.metricInterval = std::atoi(next("--metric-interval"));
        } else if (a == "--gt-ssaa") {
            opts.gtSsaa = true;
        } else if (a == "--env-map") {
            opts.envMapPath = next("--env-map");
        } else if (a == "--zoom") {
            opts.zoom = static_cast<float>(std::atof(next("--zoom")));
        } else if (a == "--zoom-center") {
            const char* v = next("--zoom-center");
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
