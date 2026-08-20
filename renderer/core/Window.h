#pragma once
// ============================================================================
// Minimal Win32 window + input.  No third-party windowing library: we create a
// native HWND, expose a surface-compatible handle, and accumulate a tiny input
// state (keyboard + relative mouse) for the free-fly camera.
// ============================================================================
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define UNICODE
#define _UNICODE
#include <windows.h>

#include <cstdint>

namespace sr {

class Window {
public:
    struct Input {
        bool keys[256] = {};
        float mouseDX = 0.f;
        float mouseDY = 0.f;
        bool mouseCaptured = false;
        bool resized = false;
    };

    bool create(const char* title, int width, int height);
    void destroy();

    // Pump pending messages and refresh input deltas.  Returns false on quit.
    bool poll();

    HWND hwnd() const { return hwnd_; }
    HINSTANCE hinstance() const { return hinstance_; }
    int width() const { return width_; }
    int height() const { return height_; }
    bool shouldClose() const { return shouldClose_; }

    const Input& input() const { return input_; }
    void clearMouseDelta() { input_.mouseDX = 0.f; input_.mouseDY = 0.f; input_.resized = false; }

    // GUI integration: optional hook (e.g. ImGui_ImplWin32_WndProcHandler)
    // invoked at the top of wndProc; a non-zero return short-circuits the
    // default handling.  clickToCapture=false disables the viewer's
    // left-click mouse capture so UI clicks keep the cursor free.
    using WndProcHook = LRESULT (*)(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp);
    void setWndProcHook(WndProcHook hook) { wndProcHook_ = hook; }
    void setClickToCaptureEnabled(bool enabled) { clickToCapture_ = enabled; }

    // Resize the client area (keeps the window position/top-left corner).
    void setClientSize(int width, int height);

private:
    static LRESULT CALLBACK wndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp);

    void setCaptured(bool captured);

    HWND hwnd_ = nullptr;
    HINSTANCE hinstance_ = nullptr;
    int width_ = 0, height_ = 0;
    bool shouldClose_ = false;
    Input input_;
    WndProcHook wndProcHook_ = nullptr;
    bool clickToCapture_ = true;
};

} // namespace sr
