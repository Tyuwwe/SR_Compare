#pragma once
// ============================================================================
// Fixed camera path: JSON playback + built-in orbit generator + exporter.
// File format (one object per sampled frame):
//   [ {"position":[x,y,z], "forward":[x,y,z], "up":[x,y,z]}, ... ]
// ============================================================================
#include "renderer/math/Math.h"

#include <vector>

namespace sr {

struct CameraKeyframe {
    Vec3 position;
    Vec3 forward;
    Vec3 up{0.f, 1.f, 0.f};
};

using CameraPath = std::vector<CameraKeyframe>;

bool loadCameraPath(const char* path, CameraPath& out);
bool saveCameraPath(const char* path, const CameraPath& in);

// Generate an orbit path around `center`.  `frames` keyframes, `rotations`
// full loops over the duration, height/radius vary linearly between min/max.
CameraPath generateOrbitPath(int frames, const Vec3& center, float radius, float heightMin,
                             float heightMax, float rotations);

} // namespace sr
