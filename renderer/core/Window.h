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
    };

    bool create(const char* title, int width, int height);
    void destroy();

    // Pump pending events and refresh input deltas.  Returns false on quit.
    bool poll();

    SDL_Window* sdlWindow() const { return window_; }
    int width() const { return width_; }
    int height() const { return height_; }
    bool shouldClose() const { return shouldClose_; }
    // Request a clean quit (e.g. a UI Exit button): the next poll() returns
    // false, so the caller's normal shutdown path runs.
    void requestClose() { shouldClose_ = true; }

    const Input& input() const { return input_; }
    void clearMouseDelta() { input_.mouseDX = 0.f; input_.mouseDY = 0.f; input_.resized = false; }

    // GUI integration: optional hook (e.g. ImGui_ImplSDL3_ProcessEvent)
    // invoked for every event before the default handling; returning true
    // consumes the event.  clickToCapture=false disables the viewer's
    // left-click mouse capture so UI clicks keep the cursor free.
    using EventHook = bool (*)(const SDL_Event& event);
    void setEventHook(EventHook hook) { eventHook_ = hook; }
    void setClickToCaptureEnabled(bool enabled) { clickToCapture_ = enabled; }

    // Resize the client area (keeps the window position/top-left corner).
    void setClientSize(int width, int height);

private:
    void setCaptured(bool captured);

    SDL_Window* window_ = nullptr;
    int width_ = 0, height_ = 0;
    bool shouldClose_ = false;
    Input input_;
    EventHook eventHook_ = nullptr;
    bool clickToCapture_ = true;
};

} // namespace sr
