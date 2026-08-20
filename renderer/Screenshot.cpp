#include "renderer/Screenshot.h"

#include "renderer/core/PathUtil.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace sr {

namespace {

float halfToFloat(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3ff;
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

uint8_t tonemapByte(float v) {
    v = v / (1.f + v);
    v = std::pow(v, 1.f / 2.2f);
    v = std::clamp(v, 0.f, 1.f);
    return static_cast<uint8_t>(v * 255.f + 0.5f);
}

} // namespace

bool savePngFromHalfRgba(const char* path, const uint8_t* halfRgba, uint32_t width,
                         uint32_t height) {
    if (!path || !halfRgba || width == 0 || height == 0) return false;
    ensureParentDir(path);

    std::vector<uint8_t> rgba(static_cast<size_t>(width) * height * 4);
    const size_t pixelCount = static_cast<size_t>(width) * height;
    for (size_t i = 0; i < pixelCount; ++i) {
        const uint8_t* src = halfRgba + i * 8;
        uint16_t r, g, b;
        std::memcpy(&r, src, 2);
        std::memcpy(&g, src + 2, 2);
        std::memcpy(&b, src + 4, 2);
        rgba[i * 4 + 0] = tonemapByte(halfToFloat(r));
        rgba[i * 4 + 1] = tonemapByte(halfToFloat(g));
        rgba[i * 4 + 2] = tonemapByte(halfToFloat(b));
        rgba[i * 4 + 3] = 255;
    }

    stbi_write_png_compression_level = 4; // see savePngFromRgba8
    return stbi_write_png(path, static_cast<int>(width), static_cast<int>(height), 4,
                          rgba.data(), static_cast<int>(width) * 4) != 0;
}

bool savePngFromRgba8(const char* path, const uint8_t* rgba8, uint32_t width, uint32_t height) {
    if (!path || !rgba8 || width == 0 || height == 0) return false;
    ensureParentDir(path);
    // stb's deflate defaults to level 8, which takes seconds at 1080p+;
    // level 4 is ~3x faster for a modest size increase.
    stbi_write_png_compression_level = 4;
    return stbi_write_png(path, static_cast<int>(width), static_cast<int>(height), 4, rgba8,
                          static_cast<int>(width) * 4) != 0;
}

} // namespace sr
