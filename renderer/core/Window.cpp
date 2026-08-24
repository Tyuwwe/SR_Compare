#include "renderer/core/Window.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace sr {

namespace {

// Debug hook (SR_GUI_DEBUG_INPUT=1): confirm wheel/button/motion events
// actually reach the pump and the installed hook.
bool debugInputEnabled() {
    static const bool enabled = std::getenv("SR_GUI_DEBUG_INPUT") != nullptr;
    return enabled;
}

// Translate an SDL keycode to a Window::Input::keys index (virtual-key style:
// uppercase ASCII for letters/digits, kKey* for the modifiers the free-fly
// camera reads).  Returns -1 for keys we do not track.
int keyIndexFromSdl(SDL_Keycode k) {
    if (k >= SDLK_A && k <= SDLK_Z) return static_cast<int>(k - SDLK_A + 'A');
    if (k >= SDLK_0 && k <= SDLK_9) return static_cast<int>(k);
    switch (k) {
    case SDLK_SPACE: return kKeySpace;
    case SDLK_LSHIFT:
    case SDLK_RSHIFT: return kKeyShift;
    case SDLK_LCTRL:
    case SDLK_RCTRL: return kKeyControl;
    default: return -1;
    }
}

} // namespace

bool Window::create(const char* title, int width, int height) {
    width_ = width;
    height_ = height;

    if (!SDL_WasInit(SDL_INIT_VIDEO) && !SDL_Init(SDL_INIT_VIDEO)) {
        std::fprintf(stderr, "window: SDL_Init failed: %s\n", SDL_GetError());
        return false;
    }
    // The Vulkan surface is created later by SDL_Vulkan_CreateSurface.
    // HIGH_PIXEL_DENSITY: the drawable (framebuffer) size may exceed the
    // logical window size on scaled displays (macOS Retina; a no-op on
    // Windows, where both are reported in physical pixels) — swapchain code
    // must use pixelWidth()/pixelHeight(), not width()/height().
    window_ = SDL_CreateWindow(title, width, height,
                               SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE |
                                   SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!window_) {
        std::fprintf(stderr, "window: SDL_CreateWindow failed: %s\n", SDL_GetError());
        return false;
    }
    SDL_GetWindowSizeInPixels(window_, &pixelWidth_, &pixelHeight_);
    return true;
}

void Window::destroy() {
    if (window_) {
        SDL_DestroyWindow(window_);
        window_ = nullptr;
    }
    if (SDL_WasInit(SDL_INIT_VIDEO)) SDL_Quit();
}

bool Window::poll() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (debugInputEnabled() &&
            (event.type == SDL_EVENT_MOUSE_WHEEL ||
             event.type == SDL_EVENT_MOUSE_BUTTON_DOWN ||
             event.type == SDL_EVENT_MOUSE_BUTTON_UP ||
             event.type == SDL_EVENT_MOUSE_MOTION)) {
            std::fprintf(stderr, "[sdl-event] tick=%llu type=0x%04x hook=%d\n",
                         static_cast<unsigned long long>(SDL_GetTicks()), event.type,
                         eventHook_ ? 1 : 0);
        }
        // The hook (e.g. ImGui) observes every event; "consumed" only
        // suppresses the app-level INPUT handling below.  Window lifecycle
        // events (resize, close) must still reach our size tracking and quit
        // path — ImGui_ImplSDL3_ProcessEvent reports them as consumed, which
        // otherwise silently kills resize tracking and the close button.
        const bool consumed = eventHook_ && eventHook_(event);
        switch (event.type) {
        case SDL_EVENT_QUIT:
        case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            if (debugInputEnabled()) std::fprintf(stderr, "[sdl-event] close\n");
            shouldClose_ = true;
            break;
        case SDL_EVENT_WINDOW_RESIZED:
            if (event.window.data1 > 0 && event.window.data2 > 0 &&
                (event.window.data1 != width_ || event.window.data2 != height_)) {
                width_ = event.window.data1;
                height_ = event.window.data2;
                input_.resized = true;
            }
            break;
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
            // Framebuffer size changed without a logical-size change (display
            // scale change on a scaled-display platform): the swapchain must
            // be rebuilt against the new pixel extent.
            if (event.window.data1 > 0 && event.window.data2 > 0 &&
                (event.window.data1 != pixelWidth_ || event.window.data2 != pixelHeight_)) {
                pixelWidth_ = event.window.data1;
                pixelHeight_ = event.window.data2;
                input_.resized = true;
            }
            break;
        case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
            // Moved to a monitor with a different content scale (Windows
            // 100%/125%/150%/...): the UI re-scales from contentScale().
            input_.displayScaleChanged = true;
            SDL_GetWindowSizeInPixels(window_, &pixelWidth_, &pixelHeight_);
            break;
        case SDL_EVENT_KEY_DOWN:
        case SDL_EVENT_KEY_UP: {
            if (consumed) break;
            const int idx = keyIndexFromSdl(event.key.key);
            if (idx >= 0) input_.keys[idx] = (event.type == SDL_EVENT_KEY_DOWN);
            break;
        }
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
            if (consumed) break;
            if (event.button.button == SDL_BUTTON_LEFT) {
                if (clickToCapture_) setCaptured(true);
            } else if (event.button.button == SDL_BUTTON_RIGHT) {
                setCaptured(false);
            }
            break;
        case SDL_EVENT_MOUSE_MOTION:
            if (consumed) break;
            if (input_.mouseCaptured) {
                input_.mouseDX += event.motion.xrel;
                input_.mouseDY += event.motion.yrel;
            }
            break;
        case SDL_EVENT_WINDOW_FOCUS_LOST:
            setCaptured(false);
            break;
        default:
            break;
        }
    }
    return !shouldClose_;
}

float Window::contentScale() const {
    if (!window_) return 1.f;
    const float s = SDL_GetWindowDisplayScale(window_);
    // 0 = query failed; never let the UI scale collapse to zero.
    return s > 0.f ? s : 1.f;
}

void Window::setCaptured(bool captured) {
    if (captured == input_.mouseCaptured) return;
    input_.mouseCaptured = captured;
    // Relative mouse mode hides the cursor and reports relative motion
    // deltas (replacing the old hide-cursor + recenter loop).
    SDL_SetWindowRelativeMouseMode(window_, captured);
    if (captured) {
        input_.mouseDX = 0.f;
        input_.mouseDY = 0.f;
    }
}

void Window::setClientSize(int width, int height) {
    if (!window_) return;
    SDL_SetWindowSize(window_, width, height);
    width_ = width;
    height_ = height;
    SDL_GetWindowSizeInPixels(window_, &pixelWidth_, &pixelHeight_);
}

void Window::setFullscreen(bool enabled) {
    if (!window_ || fullscreen_ == enabled) return;
    // SDL3: bool fullscreen = borderless desktop mode (no exclusive mode
    // switch).  On failure keep the old state.
    if (!SDL_SetWindowFullscreen(window_, enabled)) {
        std::fprintf(stderr, "window: SDL_SetWindowFullscreen failed: %s\n", SDL_GetError());
        return;
    }
    fullscreen_ = enabled;
    if (!enabled) {
        // Back to windowed: restore the decorations and the resizable frame
        // explicitly (never rely on SDL's fullscreen bookkeeping), then snap
        // to a known-good size instead of whatever the window happened to be
        // before: 1920x1080, clamped to the display's usable bounds and
        // centered.
        SDL_SetWindowBordered(window_, true);
        SDL_SetWindowResizable(window_, true);
        int w = 1920, h = 1080;
        SDL_Rect usable{};
        if (SDL_GetDisplayUsableBounds(SDL_GetDisplayForWindow(window_), &usable)) {
            // Clamp to the work area only when it is meaningfully smaller —
            // a 1-2 px sliver (e.g. an auto-hidden taskbar) is not worth
            // shrinking the window over.
            constexpr int kSlack = 8;
            if (usable.w < w - kSlack) w = usable.w;
            if (usable.h < h - kSlack) h = usable.h;
        }
        SDL_SetWindowSize(window_, w, h);
        SDL_SetWindowPosition(window_, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
        width_ = w;
        height_ = h;
    }
    // Automation hook: scripted runs assert the resulting windowed state
    // (decorated + resizable flags, expected size) from this line.
    int w = 0, h = 0;
    SDL_GetWindowSize(window_, &w, &h);
    std::fprintf(stderr, "window: fullscreen=%d size=%dx%d flags=0x%llx\n", enabled ? 1 : 0, w,
                 h, static_cast<unsigned long long>(SDL_GetWindowFlags(window_)));
    // The OS resize events update width_/height_ asynchronously via poll();
    // fullscreen entry arrives as SDL_EVENT_WINDOW_RESIZED with the desktop
    // size, exit restores the windowed size.
}

} // namespace sr
