#include "gui/BenchRunner.h"

#include <cstdio>

namespace sr {

bool BenchRunner::start(const std::string& exe, const std::vector<std::string>& args) {
    stop();
    log_.clear();
    parsedOffset_ = 0;
    exitCode_ = -1;
    progressDone_ = 0;
    progressTotal_ = 0;
    finished_ = false;

    // Quoted command line (paths may contain spaces).
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
    HANDLE pipeWrite = nullptr;
    if (!CreatePipe(&pipeRead_, &pipeWrite, &sa, 0)) return false;
    SetHandleInformation(pipeRead_, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = pipeWrite;
    si.hStdError = pipeWrite;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

    PROCESS_INFORMATION pi{};
    const BOOL ok = CreateProcessA(nullptr, mutableCmd.data(), nullptr, nullptr, TRUE,
                                   CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    CloseHandle(pipeWrite); // the child holds its own copy; EOF when it exits
    if (!ok) {
        CloseHandle(pipeRead_);
        pipeRead_ = nullptr;
        return false;
    }
    process_ = pi.hProcess;
    thread_ = pi.hThread;
    running_ = true;
    return true;
}

void BenchRunner::poll() {
    if (!running_) return;

    // Drain whatever is available without blocking.
    for (;;) {
        DWORD avail = 0;
        if (!PeekNamedPipe(pipeRead_, nullptr, 0, nullptr, &avail, nullptr) || avail == 0) break;
        char chunk[4096];
        DWORD want = avail > sizeof(chunk) ? sizeof(chunk) : avail;
        DWORD got = 0;
        if (!ReadFile(pipeRead_, chunk, want, &got, nullptr) || got == 0) break;
        log_.append(chunk, got);
    }
    parseProgress();

    if (WaitForSingleObject(process_, 0) == WAIT_OBJECT_0) {
        // Final drain after exit (data may still sit in the pipe).
        for (;;) {
            DWORD avail = 0;
            if (!PeekNamedPipe(pipeRead_, nullptr, 0, nullptr, &avail, nullptr) || avail == 0) break;
            char chunk[4096];
            DWORD want = avail > sizeof(chunk) ? sizeof(chunk) : avail;
            DWORD got = 0;
            if (!ReadFile(pipeRead_, chunk, want, &got, nullptr) || got == 0) break;
            log_.append(chunk, got);
        }
        DWORD code = 1;
        GetExitCodeProcess(process_, &code);
        exitCode_ = static_cast<int>(code);
        running_ = false;
        finished_ = true;
        parseProgress();
        CloseHandle(pipeRead_); pipeRead_ = nullptr;
        CloseHandle(process_); process_ = nullptr;
        CloseHandle(thread_); thread_ = nullptr;
    }
}

void BenchRunner::stop() {
    if (running_ && process_) {
        TerminateProcess(process_, 1);
        WaitForSingleObject(process_, 5000);
    }
    running_ = false;
    finished_ = false;
    if (pipeRead_) { CloseHandle(pipeRead_); pipeRead_ = nullptr; }
    if (process_) { CloseHandle(process_); process_ = nullptr; }
    if (thread_) { CloseHandle(thread_); thread_ = nullptr; }
}

void BenchRunner::parseProgress() {
    // Scan newly appended text for "[bench] <done>/<total>" line prefixes.
    size_t pos = parsedOffset_;
    for (;;) {
        pos = log_.find("[bench] ", pos);
        if (pos == std::string::npos) break;
        // Only trust it at a line start.
        if (pos > 0 && log_[pos - 1] != '\n') { pos += 8; continue; }
        int done = 0, total = 0;
        char tail = 0;
        if (sscanf_s(log_.c_str() + pos + 8, "%d/%d%c", &done, &total, &tail, 1u) >= 2) {
            progressDone_ = done;
            progressTotal_ = total;
        }
        pos += 8;
    }
    // Keep the scan offset just before the last partial line.
    const size_t lastNl = log_.rfind('\n');
    parsedOffset_ = (lastNl == std::string::npos) ? 0 : lastNl + 1;
}

} // namespace sr
