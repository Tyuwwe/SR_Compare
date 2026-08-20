#pragma once
// ============================================================================
// InputAdapter — per-algorithm input convention conversion helpers.
//
// Our renderer produces motion vectors with the following convention:
//   * R16G16_SFLOAT, pixel units, no jitter
//   * screen-space displacement: (currentNDC - previousNDC) * 0.5 * renderSize
//     => direction cur->prev ("forward motion"), +x = right, +y = down
//
// Algorithm-specific consumption (verified per integration; most temporal
// upscalers reproject with prev = uv + MV, i.e. they want cur->prev negated):
//   * TAA   : samples history at uv - MV -> correct as-is.
//   * FSR2/3: prev = uv + MV (see ffx_fsr2_reproject.h) -> motionVectorScale
//             {-1,-1} (A/B verified: +2.5 dB PSNR vs the wrong sign).
//   * XeSS  : wants cur->prev -> xessSetVelocityScale(-1,-1).
//   * DLSS  : wants cur->prev (same as FSR/UE velocity) -> mvecScale
//             {-1/renderW, -1/renderH}; in our static-geometry scenes DLSS
//             reprojects via clipToPrevClip, so the sign is untestable here.
//   * NSS   : backward pixel motion -> motionVectorScale {-1,-1}.
//   * SGSR2 : shader computes prev = uv - 0.5*clipMV -> takes cur->prev
//             clip-space MV, correct as-is (encode pass multiplies by
//             2/renderSize).
//
// Reactive mask (UpscalerResources::reactive/reactiveView) convention:
//   * R16_SFLOAT, render resolution, per-pixel accumulated translucent alpha
//     coverage (0 = fully opaque / no translucency; can exceed 1, consumers
//     clamp to [0,1]).  SHADER_READ_ONLY_OPTIMAL at dispatch; VK_NULL_HANDLE
//     when the scene has no translucency.
//   * Semantics: pixels under translucency must not trust reprojected history.
//   * Consumers:
//       TAA    : scales the history blend weight by (1 - clamp(mask,0,1))
//                (own shader; binding 6 + useReactive push flag).
//       FSR2/3 : dd.reactive (+ dd.transparencyAndComposition) — FsrUpscaler.cpp.
//       DLSS   : sl::kBufferTypeReactiveMaskHint (same 0..1 semantics) —
//                DlssUpscaler.cpp.
//       XeSS   : xess_vk_execute_params_t::responsivePixelMaskTexture with
//                XESS_INIT_FLAG_RESPONSIVE_PIXEL_MASK; a cleared render-res
//                fallback is bound when the mask is absent — XessUpscaler.cpp.
//   * Not consumed (no mask input in the vendor API):
//       NSS    : ffxApiDispatchDescNss has only color/depth/motionVectors/
//                output/outputTm1 + scalars (ffx-api/include/ffx_api/ffx_nss.h).
//       SGSR2  : fixed shader inputs (color/depth/velocity/history), no mask
//                concept in the reference shaders.
// ============================================================================
#include <cstdint>

namespace sr {

struct MotionScale {
    float x = 1.f;
    float y = 1.f;
};

// Scale render-resolution pixel motion into display-resolution pixels.
inline MotionScale motionScale(uint32_t renderWidth, uint32_t renderHeight, uint32_t displayWidth,
                               uint32_t displayHeight) {
    return {static_cast<float>(displayWidth) / static_cast<float>(renderWidth),
            static_cast<float>(displayHeight) / static_cast<float>(renderHeight)};
}

// Flip the Y sign (Vulkan y-down <-> GL / NSS y-up).
inline float flipMotionY(float y) {
    return -y;
}

} // namespace sr
