#pragma once
// ============================================================================
// Color grading — CPU mirror of renderer/shaders/grading.glsl (log domain,
// pre-ACES), plus the .cube 3D LUT container and its GPU upload.  Used by the
// viewer present pass / compare compose (GPU) and by the CPU screenshot path
// (renderer/Screenshot.cpp) so PNG pixels match the swapchain.
//
// Parameter set (one shared set for GT and all upscaler columns):
//   temperatureK/tint — white balance (Tanner Helland black-body approx,
//                       6500K neutral; tint moves green/magenta)
//   contrast          — around the 18% grey pivot in the log domain
//   saturation        — Rec.709 luma mix in the log domain
//   lut               — optional .cube 3D LUT (17^3/33^3), applied in the log
//                       domain after contrast/saturation
// ============================================================================
#include "renderer/core/Vk.h"
#include "renderer/core/VulkanContext.h"
#include "renderer/math/Math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace sr {

// ACEScc constants shared with grading.glsl (simplified toe: plain clamp at
// 2^-15 — both sides do the same, so GPU/CPU stay identical).
inline constexpr float kLogPivot = 0.41359f; // ACEScc of 0.18
inline constexpr float kAcesccMin = 0.000030517578125f; // 2^-15

struct ColorGrading {
    float temperatureK = 6500.f; // 1000..40000 (Tanner Helland range)
    float tint = 0.f;            // -1..1, positive = magenta
    float contrast = 1.f;        // 1 = neutral
    float saturation = 1.f;      // 1 = neutral

    bool isNeutral() const {
        return temperatureK == 6500.f && tint == 0.f && contrast == 1.f && saturation == 1.f;
    }
};

// White-balance multiplier for a (temperature, tint) pair: Tanner Helland
// black-body RGB approximation, ratioed against 6500K and normalized so the
// green channel stays 1 at tint 0.  Deterministic; computed once per frame on
// the CPU and passed to the shaders as a vec3 (grading.glsl gradeB.rgb).
Vec3 whiteBalanceForTemperatureTint(float temperatureK, float tint);

// 3D LUT container: size^3 RGB triplets, red fastest (the .cube layout).
struct ColorLut {
    uint32_t size = 0;          // edge length (17 / 33 typical)
    std::vector<float> data;    // size^3 * 3

    bool valid() const { return size >= 2 && data.size() >= size * size * size * 3; }
    // Trilinear sample with the same texel-center mapping as grading.glsl.
    Vec3 sample(Vec3 c) const;
};

// Parses a .cube file (LUT_3D_SIZE 2..64; 17^3 and 33^3 are the common ones).
// Returns false on any parse/IO error.  DOMAIN_MIN/MAX beyond [0,1] are
// rejected (we apply the LUT in the normalized log domain).
bool loadCubeLut(const char* path, ColorLut& out);

// Procedural identity LUT (the default when no .cube file is given; with
// neutral grading the display output is bit-identical to no grading).
ColorLut makeIdentityLut(uint32_t size = 17);

// Procedural "gentle filmic" demo LUT: mild S-curve + slightly warm
// highlights / cool shadows in the log domain.  Not used by default —
// generateLutFile() writes it for experimentation.
ColorLut makeStylizedLut(uint32_t size = 17);
bool writeCubeLut(const char* path, const ColorLut& lut);

// CPU mirror of applyColorGrading() in grading.glsl.  `lut` may be null
// (treated as identity).  Keep the operator order in lockstep.
inline Vec3 applyColorGrading(Vec3 sceneLinear, const ColorGrading& g, const ColorLut* lut) {
    const Vec3 wb = whiteBalanceForTemperatureTint(g.temperatureK, g.tint);
    Vec3 c{sceneLinear.x * wb.x, sceneLinear.y * wb.y, sceneLinear.z * wb.z};
    const float kScale = 17.52f, kOffset = 9.72f;
    Vec3 l{(std::log2(std::max(c.x, kAcesccMin)) + kOffset) / kScale,
           (std::log2(std::max(c.y, kAcesccMin)) + kOffset) / kScale,
           (std::log2(std::max(c.z, kAcesccMin)) + kOffset) / kScale};
    l = (l - Vec3{kLogPivot, kLogPivot, kLogPivot}) * g.contrast +
        Vec3{kLogPivot, kLogPivot, kLogPivot};
    const float luma = l.x * 0.2126f + l.y * 0.7152f + l.z * 0.0722f;
    l.x = luma + (l.x - luma) * g.saturation;
    l.y = luma + (l.y - luma) * g.saturation;
    l.z = luma + (l.z - luma) * g.saturation;
    if (lut && lut->valid()) l = lut->sample(l);
    return {std::exp2(l.x * kScale - kOffset), std::exp2(l.y * kScale - kOffset),
            std::exp2(l.z * kScale - kOffset)};
}

// GPU half of the LUT: RGBA16F 3D image + linear sampler, uploaded through a
// staging buffer.  One instance per app (viewer / compare / GUI); the
// identity LUT is 17^3 (~39 KB).
class GradingLutGpu {
public:
    bool create(const VulkanContext& ctx, const ColorLut& lut);
    void destroy(const VulkanContext& ctx);

    VkImageView view() const { return view_; }
    VkSampler sampler() const { return sampler_; }
    bool valid() const { return image_ != VK_NULL_HANDLE; }

private:
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation memory_ = VK_NULL_HANDLE;
    VkImageView view_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;
};

} // namespace sr
