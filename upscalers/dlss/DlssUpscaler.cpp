// ============================================================================
// DLSS Super Resolution via Streamline.  See DlssUpscaler.h / SlContext.h.
//
// Input conventions (contract from upscalers/InputAdapter.h):
//   color  : render-res HDR, R16G16B16A16_SFLOAT
//   depth  : D32_SFLOAT, not inverted, near=0.1, far treated as infinite
//   motion : R16G16_SFLOAT, pixel units at render res, no jitter,
//            (currentNDC - previousNDC) * 0.5 * renderSize, +y down,
//            i.e. cur->prev.  NGX expects cur->prev (prev = uv + MV,
//            same as FSR/UE velocity), so mvecScale negates while
//            normalizing to [-1,1] ({-1/renderW, -1/renderH}).
//            NOTE: in our scenes all geometry is static and DLSS reprojects
//            camera motion via clipToPrevClip, so the MV sign is untestable
//            here; the sign follows the documented-majority convention.
//   jitter : pixel units, provided separately via Constants::jitterOffset
//            (matrices passed to SL never contain jitter).
//   reactive (optional, may be null) : R16_SFLOAT transparency-coverage mask
//            at render res (0 = opaque, 1 = fully transparent overlay),
//            tagged as kBufferTypeBiasCurrentColorHint +
//            kBufferTypeReactiveMaskHint + kBufferTypeTransparencyHint.
// ============================================================================
#include "upscalers/dlss/DlssUpscaler.h"

#include "renderer/core/Vk.h" // Vulkan headers must precede the SL headers
#include "upscalers/UpscalerFactory.h"
#include "upscalers/dlss/SlContext.h"

#include <sl.h>
#include <sl_consts.h>
#include <sl_core_types.h>
#include <sl_helpers.h>
#include <sl_matrix_helpers.h>

#include <cstdio>
#include <memory>

namespace sr {
namespace {

// ---------------------------------------------------------------------------
// Small matrix helpers.  sr::CameraParams stores column-major float[16]
// (GLM/Vulkan convention); sl::Constants wants row-major sl::float4x4
// (sl::matrixFullInvert / sl::matrixMul operate on that layout).
// ---------------------------------------------------------------------------

sl::float4x4 toRowMajor(const float colMajor[16]) {
    sl::float4x4 m;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) (&m.row[r].x)[c] = colMajor[c * 4 + r];
    return m;
}

// DLSS quality mode that brackets the requested render/display ratio.
// (Only used for NGX mode bookkeeping; the actual input resolution comes
// from the tagged input resource extent.)
sl::DLSSMode modeForRatio(float ratio) {
    if (ratio >= 0.74f) return sl::DLSSMode::eUltraQuality;
    if (ratio >= 0.64f) return sl::DLSSMode::eMaxQuality;
    if (ratio >= 0.55f) return sl::DLSSMode::eBalanced;
    if (ratio >= 0.45f) return sl::DLSSMode::eMaxPerformance;
    return sl::DLSSMode::eUltraPerformance;
}

// Wrap a renderer image as an sl::Resource (Vulkan flavor: view/desc/state
// are mandatory, device memory may stay null).
sl::Resource wrapImage(VkImage image, VkImageView view, VkImageLayout layout, uint32_t width,
                       uint32_t height, VkFormat format, VkImageUsageFlags usage) {
    sl::Resource r(sl::ResourceType::eTex2d, image, nullptr, view,
                   static_cast<uint32_t>(layout));
    r.width = width;
    r.height = height;
    r.nativeFormat = static_cast<uint32_t>(format);
    r.mipLevels = 1;
    r.arrayLayers = 1;
    r.flags = 0;
    r.usage = usage;
    return r;
}

} // namespace

DlssUpscaler::DlssUpscaler(sl::DLSSPreset preset, const char* displayName)
    : preset_(preset), displayName_(displayName) {}

DlssUpscaler::~DlssUpscaler() {
    shutdown();
    // If init() failed halfway (Streamline initialized but no reference held),
    // still give SL its shutdown while the Vulkan device is alive.
    sl_dlss::shutdownIfIdle();
}

const char* DlssUpscaler::name() const { return displayName_; }

uint32_t DlssUpscaler::capabilities() const { return Cap_Temporal | Cap_ML; }

bool DlssUpscaler::isAvailable(const VulkanEnv& env) {
    return sl_dlss::dlssSupported(env.physicalDevice);
}

bool DlssUpscaler::init(const VulkanEnv& env, const UpscalerDesc& desc) {
    shutdown();
    env_ = env;
    desc_ = desc;

    if (!sl_dlss::bindDevice(env)) return false;
    if (!sl_dlss::dlssSupported(env.physicalDevice)) return false;

    bool loaded = false;
    const sl::Result loadedRes = slIsFeatureLoaded(sl::kFeatureDLSS, loaded);
    if (loadedRes != sl::Result::eOk || !loaded) {
        std::fprintf(stderr, "[dlss] kFeatureDLSS not loaded: %s (%d), loaded=%d\n",
                     sl::getResultAsStr(loadedRes), static_cast<int>(loadedRes), loaded ? 1 : 0);
        return false;
    }

    // Sanity-check the render resolution against the mode we will run in.
    sl::DLSSOptions options = {};
    options.mode = modeForRatio(static_cast<float>(desc.renderWidth) /
                                static_cast<float>(desc.displayWidth));
    options.outputWidth = desc.displayWidth;
    options.outputHeight = desc.displayHeight;
    sl::DLSSOptimalSettings optimal = {};
    const sl::Result optRes = slDLSSGetOptimalSettings(options, optimal);
    if (optRes != sl::Result::eOk) {
        std::fprintf(stderr, "[dlss] slDLSSGetOptimalSettings failed: %s (%d)\n",
                     sl::getResultAsStr(optRes), static_cast<int>(optRes));
        return false;
    }
    std::fprintf(stderr,
                 "[dlss] %s: %ux%u -> %ux%u, optimal render %ux%u (allowed %ux%u..%ux%u)\n",
                 displayName_, desc.renderWidth, desc.renderHeight, desc.displayWidth,
                 desc.displayHeight, optimal.optimalRenderWidth, optimal.optimalRenderHeight,
                 optimal.renderWidthMin, optimal.renderHeightMin, optimal.renderWidthMax,
                 optimal.renderHeightMax);

    sl_dlss::addRef();
    addedRef_ = true;
    ready_ = true;
    return true;
}

void DlssUpscaler::dispatch(VkCommandBuffer cmd, const UpscalerResources& res,
                            const CameraParams& cam, const FrameParams& frame) {
    if (!ready_) return;
    const sl::ViewportHandle viewport(viewportId_);

    // Frame index 0 is invalid for NGX frame tracking; shift by one.
    const uint32_t frameIndex = static_cast<uint32_t>(frame.frameIndex) + 1;
    sl::FrameToken* token = nullptr;
    sl::Result r = slGetNewFrameToken(token, &frameIndex);
    if (r != sl::Result::eOk || !token) {
        std::fprintf(stderr, "[dlss] slGetNewFrameToken failed: %s (%d)\n", sl::getResultAsStr(r),
                     static_cast<int>(r));
        return;
    }

    // -- Per-frame camera constants -----------------------------------------
    const sl::float4x4 proj = toRowMajor(cam.proj);   // camera view -> clip
    const sl::float4x4 view = toRowMajor(cam.view);   // world -> camera view
    const sl::float4x4 prevViewProj = toRowMajor(cam.prevViewProj); // prev world -> clip

    sl::Constants consts = {};
    consts.cameraViewToClip = proj;
    sl::matrixFullInvert(consts.clipToCameraView, proj);

    sl::float4x4 viewInv; // camera view -> world (row major)
    sl::matrixFullInvert(viewInv, view);

    // clipToPrevClip (row major) = transpose(prevViewProj * viewInv * projInv)
    //                            = projInv^T * viewInv^T * prevViewProj^T
    sl::float4x4 tmp;
    sl::matrixMul(tmp, consts.clipToCameraView, viewInv);
    sl::matrixMul(consts.clipToPrevClip, tmp, prevViewProj);
    sl::matrixFullInvert(consts.prevClipToClip, consts.clipToPrevClip);

    consts.jitterOffset = {cam.jitterX, cam.jitterY};
    // Our MVs are cur->prev pixels; NGX expects cur->prev too (same as
    // FSR/UE velocity), so negate when normalizing to [-1,1]
    // (ProgrammingGuideDLSS: pixel-space MVs use {1/renderW, 1/renderH}).
    consts.mvecScale = {-1.0f / static_cast<float>(desc_.renderWidth),
                        -1.0f / static_cast<float>(desc_.renderHeight)};
    consts.cameraPinholeOffset = {0.0f, 0.0f};
    // Rows of (world -> camera view)^-1 in row-major layout are the camera
    // basis vectors in world space; translation sits in the last row.
    consts.cameraPos = {viewInv[3].x, viewInv[3].y, viewInv[3].z};
    consts.cameraRight = {viewInv[0].x, viewInv[0].y, viewInv[0].z};
    consts.cameraUp = {viewInv[1].x, viewInv[1].y, viewInv[1].z};
    consts.cameraFwd = {-viewInv[2].x, -viewInv[2].y, -viewInv[2].z}; // look down -Z
    consts.cameraNear = cam.cameraNear;
    consts.cameraFar = cam.cameraFar;
    consts.cameraFOV = cam.fovY;
    consts.cameraAspectRatio =
        static_cast<float>(desc_.displayWidth) / static_cast<float>(desc_.displayHeight);
    consts.motionVectorsInvalidValue = 0.0f;
    consts.depthInverted = desc_.invertedDepth ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    consts.cameraMotionIncluded = sl::Boolean::eTrue; // reprojection includes camera motion
    consts.motionVectors3D = sl::Boolean::eFalse;
    consts.reset = frame.resetHistory ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    consts.orthographicProjection = sl::Boolean::eFalse;
    consts.motionVectorsDilated = sl::Boolean::eFalse;
    consts.motionVectorsJittered = sl::Boolean::eFalse;

    r = slSetConstants(consts, *token, viewport);

    // -- DLSS options --------------------------------------------------------
    sl::DLSSOptions options = {};
    options.mode = modeForRatio(static_cast<float>(desc_.renderWidth) /
                                static_cast<float>(desc_.displayWidth));
    options.outputWidth = desc_.displayWidth;
    options.outputHeight = desc_.displayHeight;
    options.preExposure = frame.preExposure;
    options.exposureScale = 1.0f;
    options.colorBuffersHDR = desc_.hdr ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    // No exposure buffer is available -> let DLSS drive exposure itself.
    options.useAutoExposure = sl::Boolean::eTrue;
    // Pin every quality mode to the same preset so the plugin's preset is
    // honored regardless of the mode picked from the render ratio.
    options.dlaaPreset = preset_;
    options.qualityPreset = preset_;
    options.balancedPreset = preset_;
    options.performancePreset = preset_;
    options.ultraPerformancePreset = preset_;
    options.ultraQualityPreset = preset_;
    const sl::Result optionsRes = slDLSSSetOptions(viewport, options);

    // -- Resource tags -------------------------------------------------------
    // The renderer guarantees: color/motion/depth SHADER_READ_ONLY, output
    // GENERAL.  SL manages (and restores) the states of tagged resources.
    sl::Resource colorIn = wrapImage(res.color, res.colorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     desc_.renderWidth, desc_.renderHeight,
                                     VK_FORMAT_R16G16B16A16_SFLOAT,
                                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    sl::Resource depth = wrapImage(res.depth, res.depthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   desc_.renderWidth, desc_.renderHeight, VK_FORMAT_D32_SFLOAT,
                                   VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                       VK_IMAGE_USAGE_SAMPLED_BIT);
    sl::Resource mvec = wrapImage(res.motion, res.motionView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  desc_.renderWidth, desc_.renderHeight, VK_FORMAT_R16G16_SFLOAT,
                                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    sl::Resource colorOut = wrapImage(res.output, res.outputView, VK_IMAGE_LAYOUT_GENERAL,
                                      desc_.displayWidth, desc_.displayHeight,
                                      VK_FORMAT_R16G16B16A16_SFLOAT,
                                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                          VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    sl::ResourceTag tags[] = {
        {&colorIn, sl::kBufferTypeScalingInputColor, sl::ResourceLifecycle::eOnlyValidNow},
        {&colorOut, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eOnlyValidNow},
        {&depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eOnlyValidNow},
        {&mvec, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eOnlyValidNow},
        // Optional transparency-coverage mask slots (filled below when present).
        {nullptr, sl::kBufferTypeBiasCurrentColorHint, sl::ResourceLifecycle::eOnlyValidNow},
        {nullptr, sl::kBufferTypeReactiveMaskHint, sl::ResourceLifecycle::eOnlyValidNow},
        {nullptr, sl::kBufferTypeTransparencyHint, sl::ResourceLifecycle::eOnlyValidNow},
    };
    uint32_t tagCount = 4;

    // Transparency-coverage mask (R16_SFLOAT, render res, 0 = opaque,
    // 1 = fully transparent overlay).  Map it onto every transparency hint
    // SL documents for DLSS (sl_core_types.h):
    //   kBufferTypeBiasCurrentColorHint — lerp(history, current, bias),
    //     1 = fully reject history, exactly what transparent pixels need;
    //   kBufferTypeReactiveMaskHint — 0 = default composition, 1 = fully
    //     reactive (same 0..1 semantics as our mask);
    //   kBufferTypeTransparencyHint — 1 if pixel is in a transparent area.
    // (kBufferTypeTransparencyAndCompositionMaskHint's header comment
    // describes "pixel lock" semantics that do not match our mask — skipped.)
    // eOnlyValidNow: the mask is renderer-owned and only needs to stay alive
    // until slEvaluateFeature returns, like the color/depth tags above.
    // (sl::Resource has no default ctor; construct it with the raw handles —
    // null-safe since it is only tagged when both handles are valid.)
    sl::Resource reactive = wrapImage(res.reactive, res.reactiveView,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                      desc_.renderWidth, desc_.renderHeight, VK_FORMAT_R16_SFLOAT,
                                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    if (res.reactive != VK_NULL_HANDLE && res.reactiveView != VK_NULL_HANDLE) {
        tags[tagCount].resource = &reactive;
        tags[tagCount + 1].resource = &reactive;
        tags[tagCount + 2].resource = &reactive;
        tagCount += 3;
    }
    const sl::Result tagRes = slSetTagForFrame(*token, viewport, tags, tagCount, cmd);

    // -- Evaluate ------------------------------------------------------------
    const sl::BaseStructure* inputs[] = {&viewport};
    const sl::Result evalRes =
        slEvaluateFeature(sl::kFeatureDLSS, *token, inputs, 1u, static_cast<sl::CommandBuffer*>(cmd));

    // Log everything for the first few frames, afterwards only failures.
    const bool anyError = r != sl::Result::eOk || optionsRes != sl::Result::eOk ||
                          tagRes != sl::Result::eOk || evalRes != sl::Result::eOk;
    if (loggedFrames_ < 3 || anyError) {
        std::fprintf(stderr,
                     "[dlss] %s frame %u: constants=%s(%d) options=%s(%d) tag=%s(%d) eval=%s(%d)\n",
                     displayName_, frameIndex, sl::getResultAsStr(r), static_cast<int>(r),
                     sl::getResultAsStr(optionsRes), static_cast<int>(optionsRes),
                     sl::getResultAsStr(tagRes), static_cast<int>(tagRes),
                     sl::getResultAsStr(evalRes), static_cast<int>(evalRes));
    }
    ++loggedFrames_;

    if (evalRes == sl::Result::eOk && !vramQueried_) {
        vramQueried_ = true;
        sl::DLSSState dlssState = {};
        if (slDLSSGetState(viewport, dlssState) == sl::Result::eOk) {
            vramBytes_ = dlssState.estimatedVRAMUsageInBytes;
            std::fprintf(stderr, "[dlss] %s: estimated VRAM usage %.1f MB\n", displayName_,
                         static_cast<double>(vramBytes_) / (1024.0 * 1024.0));
        }
    }
}

void DlssUpscaler::shutdown() {
    if (ready_) {
        ready_ = false;
        // The renderer has waited idle before calling shutdown(), so no
        // slEvaluateFeature is in flight and resources can be freed immediately.
        const sl::Result res = slFreeResources(sl::kFeatureDLSS, sl::ViewportHandle(viewportId_));
        if (res != sl::Result::eOk) {
            std::fprintf(stderr, "[dlss] slFreeResources failed: %s (%d)\n", sl::getResultAsStr(res),
                         static_cast<int>(res));
        }
    }
    if (addedRef_) {
        addedRef_ = false;
        sl_dlss::release();
    }
    // NOTE: deliberately NOT calling sl_dlss::shutdownIfIdle() here — init()
    // calls shutdown() defensively at entry, and tearing Streamline down at
    // that point would orphan the proxy-initialized plugins (the device was
    // already created under the current slInit).  The destructor handles the
    // failed-init cleanup case below.
}

uint64_t DlssUpscaler::gpuMemoryBytes() const { return vramBytes_; }

std::unique_ptr<IUpscaler> createDlssKUpscaler() {
    return std::make_unique<DlssUpscaler>(sl::DLSSPreset::ePresetK, "DLSS-K");
}

std::unique_ptr<IUpscaler> createDlssMUpscaler() {
    return std::make_unique<DlssUpscaler>(sl::DLSSPreset::ePresetM, "DLSS-M");
}

} // namespace sr

SR_REGISTER_UPSCALER("dlss-k", &sr::createDlssKUpscaler);
SR_REGISTER_UPSCALER("dlss-m", &sr::createDlssMUpscaler);
