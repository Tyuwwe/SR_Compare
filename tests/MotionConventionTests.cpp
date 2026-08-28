#include "renderer/math/Math.h"
#include "upscalers/InputAdapter.h"

#include <cmath>
#include <cstdio>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

bool near(float a, float b, float epsilon = 1e-5f) {
    return std::fabs(a - b) <= epsilon;
}

sr::Vec4 transformClip(const sr::Mat4& m, const sr::Vec3& p) {
    return {
        m.m[0] * p.x + m.m[4] * p.y + m.m[8] * p.z + m.m[12],
        m.m[1] * p.x + m.m[5] * p.y + m.m[9] * p.z + m.m[13],
        m.m[2] * p.x + m.m[6] * p.y + m.m[10] * p.z + m.m[14],
        m.m[3] * p.x + m.m[7] * p.y + m.m[11] * p.z + m.m[15],
    };
}

sr::Vec2 projectUv(const sr::Mat4& viewProj, const sr::Vec3& p) {
    const sr::Vec4 clip = transformClip(viewProj, p);
    return {clip.x / clip.w * 0.5f + 0.5f, clip.y / clip.w * 0.5f + 0.5f};
}

sr::Vec2 backwardMotion(const sr::Mat4& currentViewProj, const sr::Vec3& currentPoint,
                        const sr::Mat4& previousViewProj, const sr::Vec3& previousPoint) {
    const sr::Vec2 current = projectUv(currentViewProj, currentPoint);
    const sr::Vec2 previous = projectUv(previousViewProj, previousPoint);
    return {previous.x - current.x, previous.y - current.y};
}

} // namespace

int main() {
    using namespace sr;
    constexpr float kFov = 1.047197551f;
    const Mat4 projection = Mat4::perspective(kFov, 16.f / 9.f, 0.1f, 1000.f);
    const Mat4 view = Mat4::lookAt({0.f, 0.f, 0.f}, {0.f, 0.f, -1.f}, {0.f, 1.f, 0.f});
    const Mat4 vp = projection * view;
    const Vec3 point{0.f, 0.f, -5.f};

    const Vec2 stationary = backwardMotion(vp, point, vp, point);
    check(near(stationary.x, 0.f) && near(stationary.y, 0.f),
          "stationary camera/object produces zero motion");

    // Camera yaws right: a fixed center point moves left in the current image,
    // so its current->previous lookup points right (positive framebuffer U).
    const Mat4 yawedView =
        Mat4::lookAt({0.f, 0.f, 0.f}, {0.1f, 0.f, -1.f}, {0.f, 1.f, 0.f});
    const Vec2 yawMotion = backwardMotion(projection * yawedView, point, vp, point);
    check(yawMotion.x > 0.f, "camera yaw uses current-to-previous X direction");
    check(near(yawMotion.y, 0.f, 1e-4f), "horizontal camera yaw has no Y motion");

    // An object moving upward has a smaller framebuffer V now; the lookup back
    // to its old location therefore points down (positive V).
    const Vec2 objectMotion = backwardMotion(vp, {0.f, 1.f, -5.f}, vp, point);
    check(objectMotion.y > 0.f, "framebuffer UV uses positive-down Y orientation");

    const MotionScale pixelScale = motionUvToPixels(960, 540);
    const Vec2 tenPixelUv{10.f / 960.f, 10.f / 540.f};
    check(near(tenPixelUv.x * pixelScale.x, 10.f) &&
              near(tenPixelUv.y * pixelScale.y, 10.f),
          "UV-to-pixel adapter preserves a ten-pixel displacement");

    // FSR divides the supplied scale by render size internally, so passing
    // render dimensions preserves the canonical normalized displacement.
    const MotionScale fsrScale = fsrMotionVectorScale(960, 540);
    check(near(tenPixelUv.x * fsrScale.x / 960.f, tenPixelUv.x) &&
              near(tenPixelUv.y * fsrScale.y / 540.f, tenPixelUv.y),
          "FSR motion scale recovers backward UV motion");

    const MotionScale nssScale = nssMotionVectorScale(960, 540);
    check(near(tenPixelUv.x * nssScale.x, 10.f) &&
              near(tenPixelUv.y * nssScale.y, 10.f),
          "NSS receives backward input-pixel motion");

    const MotionScale xessScale = xessVelocityScale(960, 540);
    check(near(tenPixelUv.x * xessScale.x, 10.f) &&
              near(tenPixelUv.y * xessScale.y, 10.f),
          "XeSS receives backward input-pixel motion");

    const MotionScale dlssScale = dlssMotionVectorScale();
    check(near(tenPixelUv.x * dlssScale.x, tenPixelUv.x) &&
              near(tenPixelUv.y * dlssScale.y, tenPixelUv.y),
          "DLSS receives normalized backward motion unchanged");

    const MotionScale sgsrScale = motionUvToSgsrForwardNdc();
    check(near(tenPixelUv.x * sgsrScale.x, -2.f * tenPixelUv.x) &&
              near(tenPixelUv.y * sgsrScale.y, -2.f * tenPixelUv.y),
          "SGSR2 adapter converts backward UV to forward NDC");

    // UV motion must not change with render resolution; only the vendor pixel
    // conversion changes. The same normalized displacement at 1920x1080 is 2x pixels.
    const MotionScale fullHdScale = motionUvToPixels(1920, 1080);
    check(near(tenPixelUv.x * fullHdScale.x, 20.f) &&
              near(tenPixelUv.y * fullHdScale.y, 20.f),
          "canonical motion is render-resolution independent");

    if (failures == 0) std::printf("motion convention tests passed\n");
    return failures == 0 ? 0 : 1;
}
