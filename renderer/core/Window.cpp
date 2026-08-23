#include "renderer/core/Window.h"

#include <SDL3/SDL.h>

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
    window_ = SDL_CreateWindow(title, width, height,
                               SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!window_) {
        std::fprintf(stderr, "window: SDL_CreateWindow failed: %s\n", SDL_GetError());
        return false;
    }
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
        if (eventHook_ && eventHook_(event)) continue;
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
        case SDL_EVENT_KEY_DOWN:
        case SDL_EVENT_KEY_UP: {
            const int idx = keyIndexFromSdl(event.key.key);
            if (idx >= 0) input_.keys[idx] = (event.type == SDL_EVENT_KEY_DOWN);
            break;
        }
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
            if (event.button.button == SDL_BUTTON_LEFT) {
                if (clickToCapture_) setCaptured(true);
            } else if (event.button.button == SDL_BUTTON_RIGHT) {
                setCaptured(false);
            }
            break;
        case SDL_EVENT_MOUSE_MOTION:
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
    // The OS resize events update width_/height_ asynchronously via poll();
    // fullscreen entry arrives as SDL_EVENT_WINDOW_RESIZED with the desktop
    // size, exit restores the windowed size.
}

} // namespace sr
