#pragma once
// ============================================================================
// PNG screenshot helper: converts a half-float RGBA16F buffer (read back from
// the final HDR image) into a tonemapped 8-bit sRGB PNG via stb_image_write.
// Tonemapping mirrors present.frag (Reinhard + gamma 2.2).
// ============================================================================
#include <cstdint>

namespace sr {

bool savePngFromHalfRgba(const char* path, const uint8_t* halfRgba, uint32_t width,
                         uint32_t height);

// Save an already-tonemapped RGBA8 buffer (e.g. the compare-mode composite,
// which includes the overlay text) without further processing.
bool savePngFromRgba8(const char* path, const uint8_t* rgba8, uint32_t width, uint32_t height);

} // namespace sr
