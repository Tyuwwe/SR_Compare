#pragma once
// ============================================================================
// InputAdapter — per-algorithm input convention conversion helpers.
//
// Our renderer produces motion vectors with one canonical convention:
//   * R16G16_SFLOAT, normalized framebuffer UV units, no jitter
//   * previousUV - currentUV (current->previous / backward motion)
//   * +x = right, +y = down; previousUV = currentUV + motionUV
//
// Algorithm-specific consumption (verified per integration; most temporal
// upscalers reproject with prev = uv + MV, i.e. they want current->previous):
//   * TAA   : samples history at uv + MV -> correct as-is.
//   * FSR2/3: motionVectorScale {renderWidth,renderHeight}; the SDK divides
//             by render size internally, recovering the stored backward UV.
//   * XeSS  : xessSetVelocityScale(renderWidth,renderHeight) converts UV to
//             the backward input-pixel displacement expected by XeSS.
//             jitterOffset is the framebuffer pixel displacement (same
//             sign as FSR2).  The official sample's -jitterY must not be
//             copied: it compensates clip.xy += on Y-up clip positions.
//   * DLSS  : wants normalized cur->prev -> mvecScale {1,1}.
//             jitterOffset is framebuffer pixels.
//   * NSS   : backward pixel motion -> motionVectorScale {renderWidth,
//             renderHeight}, matching the NSS user guide's UV-space example.
//             jitterOffset is negated (NssUpscaler.cpp): NSS wants the
//             unjitter sampling offset, not the content displacement —
//             Arm's reference sample applies jitter with proj[2][0/1] +=
//             on a clip.w = -z projection (content moves by -jitter
//             pixels) and reports the positive value.
//   * SGSR2 : official InputVelocity is forward NDC for dynamic objects only.
//             Static pixels remain zero and use clipToPrevClip + nearest depth;
//             reactive pixels encode forwardNDC = -2*motionUV.
//
// Reactive mask (UpscalerResources::reactive/reactiveView) convention:
//   * R16_SFLOAT, render resolution, per-pixel accumulated translucent alpha
//     coverage (0 = fully opaque / no translucency; can exceed 1, consumers
//     clamp to [0,1]).  The renderer feeds the 3x3-max DILATED +
//     MOTION-GATED copy (reactive_dilate.comp): the plateau absorbs the
//     sub-pixel straddle of consumers that sample it at the jittered
//     coordinate (FSR2/3), and coverage on (near-)static pixels is gated to
//     zero — reprojection is exact there, so dropping history would only
//     pass the phase-flipping aliased current frame through unfiltered
//     (shimmer).  SHADER_READ_ONLY_OPTIMAL at dispatch; VK_NULL_HANDLE
//     when the scene has no translucency.
//   * Semantics: pixels under MOVING translucency must not trust reprojected
//     history; static coverage keeps full history weight.
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
//       SGSR2  : uses coverage only to select pixels that need explicit
//                forward-NDC velocity; static opaque pixels retain matrix/depth
//                reconstruction from the reference shaders.
// FrameParams::preExposure convention:
//   The input color (UpscalerResources::color) is un-exposed scene-linear HDR.
//   preExposure is the display exposure multiplier (ACES input scale) that the
//   host applies to this frame's image at present/compose time.  With auto
//   exposure (UE4-style histogram + EV solver, see DeferredCore) it is the
//   latest CPU-harvested solver output — i.e. the value solved
//   kFramesInFlight frames ago (engine-style: earlier frames' luminance
//   drives the current frame's exposure; FSR2/DLSS expect the current or
//   previous frame's exposure here, both conventions are this value within
//   one frame).  Manual mode (--exposure) feeds the fixed override instead.
//   Deterministic: the solver smooths with a fixed 1/60 step per frame, so a
//   fixed camera path reproduces the same preExposure per frame index.
// ============================================================================
#include <cstdint>

namespace sr {

struct MotionScale {
    float x = 1.f;
    float y = 1.f;
};

// Convert canonical UV motion to input-resolution pixel motion.
inline MotionScale motionUvToPixels(uint32_t renderWidth, uint32_t renderHeight) {
    return {static_cast<float>(renderWidth), static_cast<float>(renderHeight)};
}

// Named wrappers keep each SDK integration tied to the tested shared contract,
// even where several SDKs currently require the same numerical conversion.
inline MotionScale fsrMotionVectorScale(uint32_t renderWidth, uint32_t renderHeight) {
    return motionUvToPixels(renderWidth, renderHeight);
}

inline MotionScale nssMotionVectorScale(uint32_t renderWidth, uint32_t renderHeight) {
    return motionUvToPixels(renderWidth, renderHeight);
}

inline MotionScale xessVelocityScale(uint32_t renderWidth, uint32_t renderHeight) {
    return motionUvToPixels(renderWidth, renderHeight);
}

inline MotionScale dlssMotionVectorScale() {
    return {1.f, 1.f};
}

// SGSR2 stores forward NDC while the canonical texture stores backward UV.
inline MotionScale motionUvToSgsrForwardNdc() {
    return {-2.f, -2.f};
}

} // namespace sr
