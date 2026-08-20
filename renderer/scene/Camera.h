#pragma once
// ============================================================================
// Free-fly + fixed-path camera.  Produces right-handed Vulkan view/projection
// matrices (camera looks down -Z, Y up) consumed by the scene UBO.
// ============================================================================
#include "renderer/core/Window.h"
#include "renderer/math/Math.h"

namespace sr {

class Camera {
public:
    Vec3 position{0.f, 3.f, 12.f};
    Vec3 forward{0.f, -0.2f, -1.f};
    Vec3 up{0.f, 1.f, 0.f};
    float fovY = 1.047197551f; // 60 degrees
    float nearPlane = 0.1f;
    float farPlane = 1000.f;

    Mat4 view() const;
    Mat4 proj(float aspect) const;

    void lookAt(const Vec3& target) { forward = normalize(target - position); }

    void setPose(const Vec3& pos, const Vec3& fwd, const Vec3& upv) {
        position = pos;
        forward = normalize(fwd);
        up = normalize(upv);
    }

    // Orbit camera used for the generated/fallback automation path.
    void setOrbit(float angle, float height, float radius, const Vec3& center);

    // WASD + mouse-look.  dt in seconds.
    void updateFreeFly(const Window::Input& input, float dt);

private:
    float yaw_ = 0.f;
    float pitch_ = 0.f;
    void syncAnglesFromForward();
};

} // namespace sr
