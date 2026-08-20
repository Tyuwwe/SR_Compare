#include "renderer/core/Window.h"

#include <cstdio>
#include <cstdlib>

namespace sr {

namespace {
// Per-window back pointer, set in WM_NCCREATE and cleared on destroy.
constexpr wchar_t kWndProp[] = L"sr_compare_window";

// Debug hook (SR_GUI_DEBUG_INPUT=1): confirm wheel/middle-button messages
// actually reach the window procedure and the installed hook.
bool debugInputEnabled() {
    static const bool enabled = std::getenv("SR_GUI_DEBUG_INPUT") != nullptr;
    return enabled;
}
} // namespace

bool Window::create(const char* title, int width, int height) {
    (void)title; // fixed window title; keep the parameter for the API
    hinstance_ = GetModuleHandleW(nullptr);
    width_ = width;
    height_ = height;

    WNDCLASSEXW wc = {};
    wc.cbSize = sizeof(wc);
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = &Window::wndProc;
    wc.hInstance = hinstance_;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.lpszClassName = L"sr_compare_window";
    if (!RegisterClassExW(&wc)) {
        // Already registered by a previous window in this process is fine.
        if (GetLastError() != ERROR_CLASS_ALREADY_EXISTS) return false;
    }

    RECT rect = {0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);

    hwnd_ = CreateWindowExW(0, wc.lpszClassName, L"sr_compare",
                            WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
                            rect.right - rect.left, rect.bottom - rect.top,
                            nullptr, nullptr, hinstance_, this);
    if (!hwnd_) return false;

    SetPropW(hwnd_, kWndProp, this);
    ShowWindow(hwnd_, SW_SHOW);
    UpdateWindow(hwnd_);
    return true;
}

void Window::destroy() {
    if (hwnd_) {
        RemovePropW(hwnd_, kWndProp);
        DestroyWindow(hwnd_);
        hwnd_ = nullptr;
    }
    if (hinstance_) {
        UnregisterClassW(L"sr_compare_window", hinstance_);
        hinstance_ = nullptr;
    }
}

bool Window::poll() {
    MSG msg;
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            shouldClose_ = true;
            return false;
        }
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    if (input_.mouseCaptured) {
        POINT p;
        GetCursorPos(&p);
        RECT rc;
        GetWindowRect(hwnd_, &rc);
        const int cx = (rc.left + rc.right) / 2;
        const int cy = (rc.top + rc.bottom) / 2;
        input_.mouseDX += static_cast<float>(p.x - cx);
        input_.mouseDY += static_cast<float>(p.y - cy);
        SetCursorPos(cx, cy);
    }
    return !shouldClose_;
}

void Window::setCaptured(bool captured) {
    if (captured == input_.mouseCaptured) return;
    input_.mouseCaptured = captured;
    if (captured) {
        ShowCursor(FALSE);
        SetCapture(hwnd_);
        RECT rc;
        GetWindowRect(hwnd_, &rc);
        SetCursorPos((rc.left + rc.right) / 2, (rc.top + rc.bottom) / 2);
        input_.mouseDX = 0.f;
        input_.mouseDY = 0.f;
    } else {
        ShowCursor(TRUE);
        ReleaseCapture();
    }
}

void Window::setClientSize(int width, int height) {
    if (!hwnd_) return;
    RECT rect = {0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
    SetWindowPos(hwnd_, nullptr, 0, 0, rect.right - rect.left, rect.bottom - rect.top,
                 SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
    width_ = width;
    height_ = height;
}

LRESULT CALLBACK Window::wndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    Window* self = reinterpret_cast<Window*>(GetPropW(hwnd, kWndProp));
    if (debugInputEnabled() &&
        (msg == WM_MOUSEWHEEL || msg == WM_MBUTTONDOWN || msg == WM_MBUTTONUP ||
         msg == WM_MOUSEMOVE || msg == WM_MOUSELEAVE)) {
        std::fprintf(stderr, "[wndproc] tick=%llu msg=0x%04x wp=0x%zx hook=%d\n",
                     static_cast<unsigned long long>(GetTickCount64()), msg,
                     static_cast<size_t>(wp), self && self->wndProcHook_ ? 1 : 0);
    }
    if (self && self->wndProcHook_) {
        const LRESULT handled = self->wndProcHook_(hwnd, msg, wp, lp);
        if (handled != 0) return handled;
    }
    switch (msg) {
    case WM_CLOSE:
        if (debugInputEnabled()) std::fprintf(stderr, "[wndproc] WM_CLOSE\n");
        if (self) self->shouldClose_ = true;
        return 0;
    case WM_SIZE:
        if (self && wp != SIZE_MINIMIZED) {
            self->width_ = LOWORD(lp);
            self->height_ = HIWORD(lp);
            self->input_.resized = true;
        }
        return 0;
    case WM_KEYDOWN:
    case WM_SYSKEYDOWN:
        if (self && wp < 256) self->input_.keys[wp] = true;
        return 0;
    case WM_KEYUP:
    case WM_SYSKEYUP:
        if (self && wp < 256) self->input_.keys[wp] = false;
        return 0;
    case WM_LBUTTONDOWN:
        if (self && self->clickToCapture_) self->setCaptured(true);
        return 0;
    case WM_RBUTTONDOWN:
        if (self) self->setCaptured(false);
        return 0;
    case WM_KILLFOCUS:
        if (self) self->setCaptured(false);
        return 0;
    default:
        break;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

} // namespace sr
