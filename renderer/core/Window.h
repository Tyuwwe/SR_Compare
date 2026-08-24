#pragma once
// ============================================================================
// Minimal SDL3 window + input.  SDL owns the native window, the event pump
// and the Vulkan surface (SDL_Vulkan_CreateSurface); we accumulate a tiny
// input state (keyboard + relative mouse) for the free-fly camera.
// ============================================================================
#include <cstdint>

union SDL_Event;
struct SDL_Window;

namespace sr {

// Virtual-key-style indices into Window::Input::keys (same values as the
// Win32 VK_* codes they replace: ASCII for letters/digits).
constexpr int kKeyShift = 0x10;
constexpr int kKeyControl = 0x11;
constexpr int kKeySpace = 0x20;

class Window {
public:
    struct Input {
        bool keys[256] = {};
        float mouseDX = 0.f;
        float mouseDY = 0.f;
        bool mouseCaptured = false;
        bool resized = false;
        // Set when SDL reports SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED (e.g.
        // the window was dragged to a monitor with a different DPI scaling).
        bool displayScaleChanged = false;
    };

    bool create(const char* title, int width, int height);
    void destroy();

    // Pump pending events and refresh input deltas.  Returns false on quit.
    bool poll();

    SDL_Window* sdlWindow() const { return window_; }
    // Logical window (client) size in SDL window coordinates — what
    // SDL_GetWindowSize reports and what engine.toml [window] width/height
    // stores.  On Windows logical == physical pixels; on platforms where the
    // OS scales the window (macOS points) they differ.
    int width() const { return width_; }
    int height() const { return height_; }
    // Framebuffer size in physical pixels (SDL_GetWindowSizeInPixels) — the
    // size Vulkan surfaces/swapchains must use.  Falls back to the logical
    // size if the query failed at creation time.
    int pixelWidth() const { return pixelWidth_ > 0 ? pixelWidth_ : width_; }
    int pixelHeight() const { return pixelHeight_ > 0 ? pixelHeight_ : height_; }
    // Display content scale for the monitor the window is on (1.0 = 100%,
    // 1.5 = 150% Windows display scaling).  Used to scale the ImGui UI.
    float contentScale() const;
    bool shouldClose() const { return shouldClose_; }
    // Request a clean quit (e.g. a UI Exit button): the next poll() returns
    // false, so the caller's normal shutdown path runs.
    void requestClose() { shouldClose_ = true; }

    const Input& input() const { return input_; }
    void clearMouseDelta() { input_.mouseDX = 0.f; input_.mouseDY = 0.f; input_.resized = false; }
    void clearDisplayScaleChanged() { input_.displayScaleChanged = false; }

    // GUI integration: optional hook (e.g. ImGui_ImplSDL3_ProcessEvent)
    // invoked for every event before the default handling; returning true
    // suppresses only the app-level input handling (keys/mouse below) —
    // window lifecycle events (resize, close) are always processed.
    // clickToCapture=false disables the viewer's left-click mouse capture so
    // UI clicks keep the cursor free.
    using EventHook = bool (*)(const SDL_Event& event);
    void setEventHook(EventHook hook) { eventHook_ = hook; }
    void setClickToCaptureEnabled(bool enabled) { clickToCapture_ = enabled; }

    // Resize the client area (keeps the window position/top-left corner).
    void setClientSize(int width, int height);

    // Borderless (desktop) fullscreen toggle — SDL_SetWindowFullscreen's
    // non-exclusive mode: the desktop resolution is kept.  On the way back to
    // windowed the decorations/resizable frame are restored explicitly and
    // the window is snapped to 1920x1080 (clamped to the display's usable
    // bounds, centered) instead of whatever size it had before.  The resize
    // events it generates flow through the normal poll() path (swapchain
    // OUT_OF_DATE handling covers the mode switch).
    void setFullscreen(bool enabled);
    bool isFullscreen() const { return fullscreen_; }

private:
    void setCaptured(bool captured);

    SDL_Window* window_ = nullptr;
    int width_ = 0, height_ = 0;             // logical (SDL window coordinates)
    int pixelWidth_ = 0, pixelHeight_ = 0;   // physical framebuffer pixels
    bool shouldClose_ = false;
    bool fullscreen_ = false;
    Input input_;
    EventHook eventHook_ = nullptr;
    bool clickToCapture_ = true;
};

} // namespace sr
