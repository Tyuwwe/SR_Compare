// ============================================================================
// Benchmark mode (sr_compare bench).
//
// For each upscaler x resolution config, spawns a fresh
//   sr_compare viewer --frames <warmup+frames> --frame-times <tmp.csv> ...
// child process (same executable), captures its stdout/stderr to a log file,
// then aggregates the per-frame CSV written by the renderer.  A failing child
// marks its row as failed and the run continues with the next combination.
//
// Output CSV columns are kept in exact sync with metrics/report.py:
//   algo,resolution,upscale_factor,upscale_pass_ms_avg,upscale_pass_ms_p50,
//   frame_ms_avg,fps_avg,fps_p50,fps_1pct_low,vram_algo_bytes,
//   vram_total_bytes,gpu_name,driver_version,notes
// ============================================================================
#include "app/CliUtils.h"
#include "bench/BenchMode.h"
#include "renderer/core/EngineConfig.h"
#include "upscalers/UpscalerFactory.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <numeric>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace sr {
namespace {

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------
struct BenchOptions {
    std::string upscalersArg = "all";   // "all" or comma-separated names
    uint32_t displayW = 1920;
    uint32_t displayH = 1080;
    float renderScale = 0.5f;
    int frames = 300;                   // measured frames (after warmup)
    int warmup = 60;                    // discarded leading frames
    std::string scene = "procedural";   // viewer --scene argument
    std::string outPath;                // empty -> output/bench_<timestamp>.csv
    std::string exePath;                // empty -> derive from own executable
};

// One aggregated result row (numeric fields valid only when !failed).
struct BenchRow {
    std::string algo;
    std::string resolution;             // e.g. "1920x1080"
    std::string upscaleFactor;          // display/render, e.g. "2.0"
    double upscaleMsAvg = 0.0;
    double upscaleMsP50 = 0.0;
    double frameMsAvg = 0.0;
    double fpsAvg = 0.0;
    double fpsP50 = 0.0;
    double fps1PctLow = 0.0;
    uint64_t vramAlgoBytes = 0;
    uint64_t vramTotalBytes = 0;
    std::string gpuName;                // scraped from child stdout (may be "")
    std::string driverVersion;          // scraped from child stdout (may be "")
    std::string notes;
    bool failed = false;
};

// One parsed frame line of the renderer's --frame-times CSV.
struct FrameSample {
    double frameMs = 0.0;
    double upscaleMs = 0.0;
    uint64_t vramAlgoBytes = 0;
    uint64_t vramTotalBytes = 0;
};

std::string dirnameOf(const std::string& path) {
    const size_t pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? std::string() : path.substr(0, pos);
}

std::string joinPath(const std::string& dir, const std::string& file) {
    if (dir.empty()) return file;
    const char last = dir.back();
    return (last == '/' || last == '\\') ? dir + file : dir + "/" + file;
}

std::string defaultOutPath() {
    char stamp[32];
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &now);
#else
    localtime_r(&now, &tm);
#endif
    std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &tm);
    return joinPath("output", std::string("bench_") + stamp + ".csv");
}

std::string selfExePath() {
#ifdef _WIN32
    char buf[MAX_PATH] = {0};
    const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    return (n > 0 && n < MAX_PATH) ? std::string(buf, n) : std::string();
#else
    char buf[PATH_MAX] = {0};
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    return n > 0 ? std::string(buf, static_cast<size_t>(n)) : std::string();
#endif
}

std::string formatFactor(double f) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.4g", f);
    std::string s = buf;
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos) s += ".0";
    return s;
}

// ---------------------------------------------------------------------------
// Child process: spawn viewer, capture stdout+stderr into `log`, wait, and
// return its exit code (-1 on spawn failure).
// ---------------------------------------------------------------------------
int runChild(const std::string& exe, const std::vector<std::string>& args, std::string& log) {
    log.clear();
#ifdef _WIN32
    // Build the command line with simple quoting (paths may contain spaces).
    std::string cmdline = "\"" + exe + "\"";
    for (const std::string& a : args) {
        cmdline += " \"";
        cmdline += a;
        cmdline += "\"";
    }
    std::vector<char> mutableCmd(cmdline.begin(), cmdline.end());
    mutableCmd.push_back('\0');

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE pipeRead = nullptr, pipeWrite = nullptr;
    if (!CreatePipe(&pipeRead, &pipeWrite, &sa, 0)) return -1;
    SetHandleInformation(pipeRead, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = pipeWrite;
    si.hStdError = pipeWrite;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

    PROCESS_INFORMATION pi{};
    const BOOL ok = CreateProcessA(nullptr, mutableCmd.data(), nullptr, nullptr,
                                   TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    CloseHandle(pipeWrite);  // child holds its own copy; EOF when it exits
    if (!ok) {
        CloseHandle(pipeRead);
        return -1;
    }
    // Drain until EOF (child exit closes the write end), avoiding pipe
    // buffer deadlocks, then collect the exit code.
    char chunk[4096];
    DWORD got = 0;
    while (ReadFile(pipeRead, chunk, sizeof(chunk), &got, nullptr) && got > 0)
        log.append(chunk, got);
    CloseHandle(pipeRead);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return static_cast<int>(code);
#endif
}

// ---------------------------------------------------------------------------
// Frame-times CSV parsing + aggregation
// ---------------------------------------------------------------------------
bool parseFrameTimesCsv(const std::string& path, std::vector<FrameSample>& out) {
    out.clear();
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    char line[512];
    bool first = true;
    while (std::fgets(line, sizeof(line), f)) {
        if (first) {  // header: frame,frameMs,sceneMs,upscaleMs,vramAlgoBytes,vramTotalBytes[,vramVmaBytes]
            first = false;
            continue;
        }
        unsigned long long frame = 0, vramAlgo = 0, vramTotal = 0;
        double frameMs = 0.0, sceneMs = 0.0, upscaleMs = 0.0;
        if (::sscanf_s(line, "%llu,%lf,%lf,%lf,%llu,%llu", &frame, &frameMs, &sceneMs,
                       &upscaleMs, &vramAlgo, &vramTotal) != 6)
            continue;
        FrameSample s;
        s.frameMs = frameMs;
        s.upscaleMs = upscaleMs;
        s.vramAlgoBytes = vramAlgo;
        s.vramTotalBytes = vramTotal;
        out.push_back(s);
    }
    std::fclose(f);
    return !out.empty();
}

double meanOf(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

double medianOf(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    if (v.size() % 2 == 0) {
        // Even count: average the two middle values. nth_element places the
        // upper middle at v[mid]; the lower middle is the max of the lower half.
        const double lower = *std::max_element(v.begin(), v.begin() + mid);
        return 0.5 * (lower + v[mid]);
    }
    return v[mid];
}

// Fill aggregate stats from the samples after warmup.
void aggregate(const std::vector<FrameSample>& samples, int warmup, BenchRow& row) {
    const size_t begin = (warmup > 0) ? static_cast<size_t>(warmup) : 0;
    std::vector<double> frameMs, upscaleMs, fps;
    for (size_t i = begin; i < samples.size(); ++i) {
        const FrameSample& s = samples[i];
        frameMs.push_back(s.frameMs);
        upscaleMs.push_back(s.upscaleMs);
        if (s.frameMs > 0.0) fps.push_back(1000.0 / s.frameMs);
    }
    if (frameMs.empty()) {
        row.failed = true;
        row.notes = "failed: no frames left after warmup";
        return;
    }
    // Require a minimum number of post-warmup frames so the aggregate stats
    // (mean/median/1% low) are statistically meaningful.
    constexpr size_t kMinSamples = 30;
    if (frameMs.size() < kMinSamples) {
        row.failed = true;
        row.notes = "failed: insufficient samples after warmup (" +
                    std::to_string(frameMs.size()) + " < " +
                    std::to_string(kMinSamples) + ")";
        return;
    }
    row.upscaleMsAvg = meanOf(upscaleMs);
    row.upscaleMsP50 = medianOf(upscaleMs);
    row.frameMsAvg = meanOf(frameMs);
    // fpsAvg is the reciprocal of frameMsAvg so the two stay consistent;
    // mean(1000/frameMs) is not equal to 1000/mean(frameMs).
    row.fpsAvg = row.frameMsAvg > 0.0 ? 1000.0 / row.frameMsAvg : 0.0;
    row.fpsP50 = medianOf(fps);

    // 1% low: mean of the slowest 1% frame times, converted to fps.
    std::vector<double> sorted = frameMs;
    std::sort(sorted.begin(), sorted.end(), std::greater<double>());
    const size_t count = std::max<size_t>(1, sorted.size() / 100);
    double worst = 0.0;
    for (size_t i = 0; i < count; ++i) worst += sorted[i];
    worst /= static_cast<double>(count);
    row.fps1PctLow = (worst > 0.0) ? 1000.0 / worst : 0.0;

    // VRAM figures are constant per run; take the last row.
    row.vramAlgoBytes = samples.back().vramAlgoBytes;
    row.vramTotalBytes = samples.back().vramTotalBytes;
}

// Best-effort scrape of GPU identity from child stdout.  The current viewer
// does not print these, so both stay empty unless a future viewer emits
// "gpu_name=..." / "driver_version=..." lines.
void scrapeGpuInfo(const std::string& log, BenchRow& row) {
    const auto scrape = [&](const char* key) -> std::string {
        const size_t pos = log.find(key);
        if (pos == std::string::npos) return std::string();
        size_t begin = pos + std::strlen(key);
        size_t end = log.find_first_of("\r\n", begin);
        if (end == std::string::npos) end = log.size();
        while (begin < end && log[begin] == ' ') ++begin;
        return log.substr(begin, end - begin);
    };
    row.gpuName = scrape("gpu_name=");
    row.driverVersion = scrape("driver_version=");
}

std::string autoNotes(const std::string& algo) {
    // Qualcomm-oriented plugins: mobile-targeted with emulated inference on
    // desktop, so their timings are indicative only.
    if (algo.rfind("sgsr", 0) == 0 || algo == "nss")
        return "移动导向/模拟推理，仅供参考";
    return std::string();
}

void writeLogFile(const std::string& path, const std::string& log) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    if (!log.empty()) std::fwrite(log.data(), 1, log.size(), f);
    std::fclose(f);
}

bool writeBenchCsv(const std::string& path, const std::vector<BenchRow>& rows) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return false;
    std::fprintf(f, "algo,resolution,upscale_factor,upscale_pass_ms_avg,upscale_pass_ms_p50,"
                    "frame_ms_avg,fps_avg,fps_p50,fps_1pct_low,vram_algo_bytes,"
                    "vram_total_bytes,gpu_name,driver_version,notes\n");
    for (const BenchRow& r : rows) {
        if (r.failed) {
            std::fprintf(f, "%s,%s,%s,,,,,,,,,%s,%s,%s\n", r.algo.c_str(),
                         r.resolution.c_str(), r.upscaleFactor.c_str(), r.gpuName.c_str(),
                         r.driverVersion.c_str(), r.notes.c_str());
        } else {
            std::fprintf(f,
                         "%s,%s,%s,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%" PRIu64 ",%" PRIu64
                         ",%s,%s,%s\n",
                         r.algo.c_str(), r.resolution.c_str(), r.upscaleFactor.c_str(),
                         r.upscaleMsAvg, r.upscaleMsP50, r.frameMsAvg, r.fpsAvg, r.fpsP50,
                         r.fps1PctLow, r.vramAlgoBytes, r.vramTotalBytes, r.gpuName.c_str(),
                         r.driverVersion.c_str(), r.notes.c_str());
        }
    }
    std::fclose(f);
    return true;
}

void printUsage() {
    std::fprintf(stderr,
                 "usage: sr_compare bench [options]\n"
                 "  --upscalers all|a,b,c    algorithm set (default all; 'native' GT baseline\n"
                 "                           is always appended to 'all')\n"
                 "  --output WxH             output resolution (default 1920x1080)\n"
                 "  --render-scale <f>       render resolution scale (default 0.5)\n"
                 "  --frames <N>             measured frames per run (default 300)\n"
                 "  --warmup <N>             discarded leading frames (default 60)\n"
                 "  --scene <name|gltf path>  scene: boxes, sponza, or a glTF path (default boxes)\n"
                 "  --out <csv>              output CSV (default output/bench_<timestamp>.csv)\n"
                 "  --exe <path>             sr_compare executable (default: this executable)\n");
}

} // namespace

int runBenchMode(int argc, char** argv) {
    BenchOptions opts;
    uint64_t cliMask = cli::kNone; // explicit-CLI mask: engine.toml never overrides these
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--upscalers") {
            opts.upscalersArg = nextArg(i, argc, argv, "--upscalers");
        } else if (a == "--output") {
            if (!parseResolution(nextArg(i, argc, argv, "--output"),
                                 opts.displayW, opts.displayH)) {
                std::fprintf(stderr, "invalid --output resolution\n");
                return 1;
            }
        } else if (a == "--render-scale") {
            opts.renderScale = static_cast<float>(std::atof(nextArg(i, argc, argv, "--render-scale")));
            cliMask |= cli::kRenderScale;
        } else if (a == "--frames") {
            opts.frames = std::atoi(nextArg(i, argc, argv, "--frames"));
        } else if (a == "--warmup") {
            opts.warmup = std::atoi(nextArg(i, argc, argv, "--warmup"));
        } else if (a == "--scene") {
            opts.scene = nextArg(i, argc, argv, "--scene");
        } else if (a == "--out") {
            opts.outPath = nextArg(i, argc, argv, "--out");
        } else if (a == "--exe") {
            opts.exePath = nextArg(i, argc, argv, "--exe");
        } else if (a == "--help" || a == "-h") {
            printUsage();
            return 0;
        } else {
            std::fprintf(stderr, "unknown bench option: %s\n", a.c_str());
            printUsage();
            return 1;
        }
    }
    // engine.toml (exe-relative): render_scale fills in when the CLI did not
    // set it.  The per-run viewer children load the same file themselves
    // (same exe directory), so effect/grading keys apply there directly.
    {
        EngineConfig cfg;
        if (loadEngineConfig(cfg)) {
            EngineConfigLog log;
            cfgTake(opts.renderScale, cfg.renderScale, cliMask, cli::kRenderScale, "render_scale",
                    log);
            log.flush(" bench:");
        }
    }
    if (opts.frames <= 0 || opts.warmup < 0 || opts.renderScale <= 0.0f ||
        opts.renderScale > 1.0f) {
        std::fprintf(stderr, "invalid --frames/--warmup/--render-scale value\n");
        return 1;
    }
    if (opts.exePath.empty()) opts.exePath = selfExePath();
    if (opts.exePath.empty()) {
        std::fprintf(stderr, "cannot determine own executable path; pass --exe\n");
        return 1;
    }
    if (opts.outPath.empty()) opts.outPath = defaultOutPath();

    // Resolve the algorithm list.  "all" matches --list-upscalers plus the
    // native GT baseline row (viewer --upscaler none).
    std::vector<std::string> algos;
    if (opts.upscalersArg == "all") {
        algos = listUpscalers();
        algos.push_back("native");
    } else {
        algos = splitCsv(opts.upscalersArg.c_str());
    }
    if (algos.empty()) {
        std::fprintf(stderr, "no upscalers selected\n");
        return 1;
    }

    const std::string outDir = dirnameOf(opts.outPath);
    const std::string logDir = joinPath(outDir.empty() ? "." : outDir, "bench_logs");
#ifdef _WIN32
    if (!outDir.empty()) CreateDirectoryA(outDir.c_str(), nullptr);
    CreateDirectoryA(logDir.c_str(), nullptr);
#else
    if (!outDir.empty()) mkdir(outDir.c_str(), 0755);
    mkdir(logDir.c_str(), 0755);
#endif

    const uint32_t renderW = static_cast<uint32_t>(opts.displayW * opts.renderScale + 0.5f);
    const uint32_t renderH = static_cast<uint32_t>(opts.displayH * opts.renderScale + 0.5f);
    const std::string resolution = std::to_string(opts.displayW) + "x" +
                                   std::to_string(opts.displayH);
    const double factor = (renderW > 0 && renderH > 0)
                              ? 0.5 * (static_cast<double>(opts.displayW) / renderW +
                                       static_cast<double>(opts.displayH) / renderH)
                              : 0.0;
    const int totalFrames = opts.warmup + opts.frames;

    std::printf("[bench] exe=%s\n[bench] matrix: %zu algo(s) x %s @ scale %.2f "
                "(render %ux%u), frames=%d warmup=%d\n",
                opts.exePath.c_str(), algos.size(), resolution.c_str(), opts.renderScale,
                renderW, renderH, opts.frames, opts.warmup);

    std::vector<BenchRow> rows;
    size_t succeeded = 0;
    for (size_t idx = 0; idx < algos.size(); ++idx) {
        BenchRow row;
        row.algo = algos[idx];
        row.resolution = resolution;
        row.upscaleFactor = formatFactor(factor);
        row.notes = autoNotes(row.algo);

        const std::string viewerUpscaler = (row.algo == "native") ? "none" : row.algo;
        const std::string tmpCsv = joinPath(logDir, "bench_tmp_" + row.algo + ".csv");
        const std::string logPath = joinPath(logDir, row.algo + ".log");
        std::remove(tmpCsv.c_str());

        std::vector<std::string> args = {"viewer",
                                         "--upscaler", viewerUpscaler,
                                         "--frames", std::to_string(totalFrames),
                                         "--render-scale", std::to_string(opts.renderScale),
                                         "--output", resolution,
                                         "--scene", opts.scene,
                                         "--frame-times", tmpCsv};

        std::string childLog;
        const int exitCode = runChild(opts.exePath, args, childLog);
        writeLogFile(logPath, childLog);
        scrapeGpuInfo(childLog, row);

        std::vector<FrameSample> samples;
        if (exitCode != 0) {
            row.failed = true;
            row.notes = "failed: child exit code " + std::to_string(exitCode) +
                        (row.notes.empty() ? "" : "; " + row.notes);
        } else if (!parseFrameTimesCsv(tmpCsv, samples)) {
            row.failed = true;
            row.notes = "failed: no/empty frame-times CSV" +
                        (row.notes.empty() ? "" : "; " + row.notes);
        } else {
            aggregate(samples, opts.warmup, row);  // may set failed on empty window
        }
        std::remove(tmpCsv.c_str());

        rows.push_back(row);
        if (row.failed) {
            std::printf("[bench] %zu/%zu algo=%s res=%s FAILED (%s) log=%s\n", idx + 1,
                        algos.size(), row.algo.c_str(), resolution.c_str(),
                        row.notes.c_str(), logPath.c_str());
        } else {
            ++succeeded;
            std::printf("[bench] %zu/%zu algo=%s res=%s fps=%.1f (1%%low %.1f) "
                        "upscaleMs=%.3f frameMs=%.2f vramAlgo=%" PRIu64 "\n",
                        idx + 1, algos.size(), row.algo.c_str(), resolution.c_str(),
                        row.fpsAvg, row.fps1PctLow, row.upscaleMsAvg, row.frameMsAvg,
                        row.vramAlgoBytes);
        }
        std::fflush(stdout);
    }

    if (!writeBenchCsv(opts.outPath, rows)) {
        std::fprintf(stderr, "failed to write %s\n", opts.outPath.c_str());
        return 1;
    }
    std::printf("[bench] wrote %zu rows (%zu ok, %zu failed) to %s\n", rows.size(),
                succeeded, rows.size() - succeeded, opts.outPath.c_str());
    return succeeded > 0 ? 0 : 1;
}

} // namespace sr
