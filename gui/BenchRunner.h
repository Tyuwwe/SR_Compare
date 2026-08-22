#pragma once
// ============================================================================
// BenchRunner — non-blocking child-process driver for `sr_compare bench`.
//
// The GUI spawns its own executable in bench mode (reusing the production
// CLI), drains the child's stdout/stderr incrementally so the UI stays
// responsive, parses "[bench] i/n" progress lines, and reports the exit code.
//
// Process management (CreateProcess + pipe) is deliberately kept on the Win32
// API: bench is a Windows-only tool.  The window/input layer, by contrast, is
// pure SDL3 (renderer/core/Window).
// ============================================================================
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <string>
#include <vector>

namespace sr {

class BenchRunner {
public:
    ~BenchRunner() { stop(); }

    // Spawn `<exe> bench ...`; returns false on CreateProcess failure.
    bool start(const std::string& exe, const std::vector<std::string>& args);

    // Drain any available child output and detect process exit.  Call once
    // per UI frame while running().
    void poll();

    // Terminate the child if still running and release all handles.
    void stop();

    bool running() const { return running_; }
    bool finished() const { return finished_; }
    int exitCode() const { return exitCode_; }
    const std::string& log() const { return log_; }

    // Progress parsed from "[bench] <done>/<total> ..." lines (0/0 = unknown).
    int progressDone() const { return progressDone_; }
    int progressTotal() const { return progressTotal_; }

private:
    void parseProgress();

    HANDLE pipeRead_ = nullptr;
    HANDLE process_ = nullptr;
    HANDLE thread_ = nullptr;
    std::string log_;
    size_t parsedOffset_ = 0;
    bool running_ = false;
    bool finished_ = false;
    int exitCode_ = -1;
    int progressDone_ = 0;
    int progressTotal_ = 0;
};

} // namespace sr
