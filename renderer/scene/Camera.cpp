#include "renderer/scene/Camera.h"

#include <algorithm>
#include <cmath>

namespace sr {

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr float kMouseSensitivity = 0.0025f;
} // namespace

Mat4 Camera::view() const {
    return Mat4::lookAt(position, position + forward, up);
}

Mat4 Camera::proj(float aspect) const {
    return Mat4::perspective(fovY, aspect, nearPlane, farPlane);
}

void Camera::syncAnglesFromForward() {
    const Vec3 f = normalize(forward);
    pitch_ = std::asin(f.y);
    yaw_ = std::atan2(f.x, -f.z);
}

void Camera::setOrbit(float angle, float height, float radius, const Vec3& center) {
    position = center + Vec3{radius * std::cos(angle), height, radius * std::sin(angle)};
    forward = normalize(center - position);
    up = {0.f, 1.f, 0.f};
}

void Camera::updateFreeFly(const Window::Input& input, float dt) {
    dt = std::min(dt, 0.1f);

    syncAnglesFromForward();
    yaw_ += input.mouseDX * kMouseSensitivity;
    pitch_ -= input.mouseDY * kMouseSensitivity;
    constexpr float kPitchLimit = 1.55f;
    pitch_ = std::clamp(pitch_, -kPitchLimit, kPitchLimit);

    forward = normalize(Vec3{std::cos(pitch_) * std::sin(yaw_), std::sin(pitch_),
                             -std::cos(pitch_) * std::cos(yaw_)});
    const Vec3 worldUp{0.f, 1.f, 0.f};
    const Vec3 right = normalize(cross(forward, worldUp));

    const bool fast = input.keys[kKeyShift] != 0;
    const float speed = fast ? 16.f : 4.f;

    Vec3 move{0.f, 0.f, 0.f};
    if (input.keys['W']) move += forward;
    if (input.keys['S']) move -= forward;
    if (input.keys['D']) move += right;
    if (input.keys['A']) move -= right;
    if (input.keys['Q'] || input.keys[kKeySpace]) move += worldUp;
    if (input.keys['E'] || input.keys[kKeyControl]) move -= worldUp;
    if (dot(move, move) > 1e-6f) move = normalize(move);
    position += move * (speed * dt);
}

} // namespace sr
