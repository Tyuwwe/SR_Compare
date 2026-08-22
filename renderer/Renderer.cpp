#include "renderer/Renderer.h"

#include "app/CliUtils.h"

#include "renderer/Screenshot.h"
#include "renderer/core/MemoryBudget.h"
#include "renderer/core/PathUtil.h"
#include "renderer/core/VkUtil.h"
#include "renderer/core/Vma.h"
#include "renderer/scene/SceneRegistry.h"
#include "upscalers/UpscalerFactory.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <vector>

namespace sr {

namespace {

VkRenderingAttachmentInfo makeColorAttachment(VkImageView view, VkImageLayout layout,
                                              VkAttachmentLoadOp loadOp,
                                              float r = 0.f, float g = 0.f, float b = 0.f) {
    VkRenderingAttachmentInfo a = {};
    a.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    a.imageView = view;
    a.imageLayout = layout;
    a.loadOp = loadOp;
    a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    if (loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
        a.clearValue.color = {{r, g, b, 1.f}};
    }
    return a;
}

VkRenderingAttachmentInfo makeDepthAttachment(VkImageView view, VkImageLayout layout,
                                              VkAttachmentLoadOp loadOp) {
    VkRenderingAttachmentInfo a = {};
    a.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    a.imageView = view;
    a.imageLayout = layout;
    a.loadOp = loadOp;
    a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    if (loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
        a.clearValue.depthStencil = {1.f, 0};
    }
    return a;
}

void beginRendering(VkCommandBuffer cmd, uint32_t width, uint32_t height, uint32_t colorCount,
                    const VkRenderingAttachmentInfo* colors, const VkRenderingAttachmentInfo* depth) {
    VkRenderingInfo ri = {};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea = {{0, 0}, {width, height}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = colorCount;
    ri.pColorAttachments = colors;
    ri.pDepthAttachment = depth;
    vkCmdBeginRendering(cmd, &ri);
}

} // namespace

void Renderer::ImageResource::destroy(const VulkanContext& ctx) {
    if (view) { vkDestroyImageView(ctx.device, view, nullptr); view = VK_NULL_HANDLE; }
    if (image && arena) { arena->destroyImage(ctx, image); image = VK_NULL_HANDLE; arena = nullptr; }
    else if (image) { vmaDestroyImage(ctx.allocator, image, memory); image = VK_NULL_HANDLE; memory = VK_NULL_HANDLE; }
}

bool Renderer::init(const RendererOptions& opts) {
    opts_ = opts;
    diagNoJitter_ = sr::envFlag("SR_NO_JITTER");
    // Diagnostics: dump every frame as <dir>/frame_%04d.png (synchronous
    // readback per frame; used for offline temporal-stability metrics).
    // _dupenv_s: plain getenv trips C4996 (this project builds /W4).
#ifdef _MSC_VER
    {
        char* d = nullptr;
        size_t dLen = 0;
        if (_dupenv_s(&d, &dLen, "SR_FRAME_DUMP_DIR") == 0 && d != nullptr) {
            frameDumpDir_ = d;
            std::free(d);
        }
    }
#else
    if (const char* d = std::getenv("SR_FRAME_DUMP_DIR")) frameDumpDir_ = d;
#endif
    // GPU occlusion culling (Phase 7a) defaults on; SR_OCCLUSION=0 opts out.
    occlusion_ = occlusionEnabledByDefault();

    if (!window_.create("sr_compare", static_cast<int>(opts.displayWidth),
                        static_cast<int>(opts.displayHeight)))
        return false;
    if (!ctx_.create(window_)) return false;
    if (!swapchain_.create(ctx_, opts.displayWidth, opts.displayHeight, opts.vsync,
                           /*allowMailbox=*/true, opts.hdr))
        return false;
    if (opts.hdr) {
        const char* mode = swapchain_.hdrMode() == HdrMode::Hdr10 ? "HDR10 (PQ)"
                           : swapchain_.hdrMode() == HdrMode::ScRgb ? "scRGB" : "SDR (fallback)";
        std::fprintf(stderr, "swapchain: --hdr -> %s\n", mode);
    }

    renderWidth_ = std::max(1u, static_cast<uint32_t>(static_cast<float>(opts.displayWidth) * opts.renderScale));
    renderHeight_ = std::max(1u, static_cast<uint32_t>(static_cast<float>(opts.displayHeight) * opts.renderScale));

    bool sceneOk = false;
    // Mip streaming (Phase 7b) only in interactive free-fly: fixed-frame bench
    // runs and one-shot screenshots need the full mip chain resident from
    // frame 0 for bit-reproducible output.  SR_TEX_STREAM=1 forces it on for
    // validation of the streaming path itself.
    const int streamOv = texStreamingOverride();
    scene_.streamingEnabled =
        streamOv > 0 || (streamOv == 0 && opts.frames < 0 && opts.screenshotPath.empty());
    if (!opts.scenePath.empty()) sceneOk = scene_.loadGltf(ctx_, opts.scenePath.c_str());
    if (!sceneOk) sceneOk = scene_.loadProcedural(ctx_);
    if (!sceneOk) return false;
    hasTransparency_ = deferred_.sceneHasTransparency(scene_);
    const LightingPreset preset = lightingPresetForScene(opts.scenePath);
    iblIntensity_ = preset.iblIntensity;
    // Froxel volumetric fog (Phase 5a): media parameters come from the scene
    // preset; --no-volfog / a preset with fog disabled both gate the passes.
    fogParams_ = preset.fog;
    volFogActive_ = opts_.volFog && fogParams_.enabled;
    // Sky atmosphere sun: preset angles, overridable via --sun-elev/--sun-az.
    // The override also rewrites the directional key light so the sun disk,
    // shadows and the sky stay consistent.
    sunElevationDeg_ = opts_.sunElevationDeg >= 0.f ? opts_.sunElevationDeg
                                                    : preset.sunElevationDeg;
    sunAzimuthDeg_ = opts_.sunAzimuthDeg >= 0.f ? opts_.sunAzimuthDeg : preset.sunAzimuthDeg;
    if (opts_.sunElevationDeg >= 0.f || opts_.sunAzimuthDeg >= 0.f) {
        const Vec3 dir = sunDirectionFromElevAzimuth(sunElevationDeg_, sunAzimuthDeg_);
        const Vec3 color = atmosphereSunColor(sunElevationDeg_, sunAzimuthDeg_);
        for (Light& l : scene_.lights) {
            if (l.type == LightType::Directional) {
                l.positionOrDirection = dir;
                l.color = color;
            }
        }
    }
    // Color grading (Phase 6c): CLI grading fields win, then the scene
    // preset's optional override, then the neutral defaults.
    grading_ = ColorGrading{};
    if (preset.gradeTemperatureK > 0.f) {
        grading_.temperatureK = preset.gradeTemperatureK;
        grading_.tint = preset.gradeTint;
        grading_.contrast = preset.gradeContrast;
        grading_.saturation = preset.gradeSaturation;
    }
    if (opts_.grading.temperatureK > 0.f) grading_.temperatureK = opts_.grading.temperatureK;
    if (opts_.grading.contrast > 0.f) grading_.contrast = opts_.grading.contrast;
    if (opts_.grading.saturation > 0.f) grading_.saturation = opts_.grading.saturation;
    // Tint is signed, so 0 doubles as "unset": a nonzero CLI value wins over
    // the preset (there is no CLI way to force 0 over a preset tint).
    if (opts_.grading.tint != 0.f) grading_.tint = opts_.grading.tint;
    // Reflection probe placements (Phase 4c-2): hand-placed per scene in the
    // registry; inert until a matching .probes bake file is loaded below.
    scene_.probes = reflectionProbesForScene(opts.scenePath);

    if (scene_.materials.empty()) {
        Material fallback;
        fallback.baseColor = {0.8f, 0.8f, 0.8f, 1.f};
        fallback.roughness = 0.6f;
        scene_.materials.push_back(fallback);
    }
    if (scene_.textures.empty()) {
        const uint8_t white[4] = {255, 255, 255, 255};
        Texture dummy;
        if (!scene_.uploadTexture(ctx_, 1, 1, white, dummy)) return false;
        scene_.textures.push_back(dummy);
    }

    if (opts.upscalerName != "none") {
        upscaler_ = createUpscaler(opts.upscalerName.c_str());
        if (upscaler_) {
            UpscalerDesc desc;
            desc.renderWidth = renderWidth_;
            desc.renderHeight = renderHeight_;
            desc.displayWidth = opts.displayWidth;
            desc.displayHeight = opts.displayHeight;
            desc.hdr = true;
            desc.invertedDepth = false;
            desc.infiniteFarPlane = true;
            if (!upscaler_->init(ctx_.toEnv(), desc)) upscaler_.reset();
        }
        if (!upscaler_) {
            std::fprintf(stderr, "warning: upscaler '%s' unavailable, falling back to native GT\n",
                         opts.upscalerName.c_str());
        }
    }

    if (!createRenderTargets()) return false;
    // Shared deferred pipeline: IBL maps + shaders + layouts + pipelines.  Env
    // source priority: CLI --env-map, then the preset's envFile, else the
    // procedural sky atmosphere for the preset sun direction.
    const std::string envPath =
        !opts_.envMapPath.empty() ? opts_.envMapPath : preset.envFile;
    if (!deferred_.init(ctx_, envPath.c_str(),
                        sunDirectionFromElevAzimuth(sunElevationDeg_, sunAzimuthDeg_)))
        return false;
    // Baked reflection probes (Phase 4c-2): no bake file -> count 0, rendering
    // identical to the global-env-only path.
    deferred_.loadProbes(ctx_, scene_.probes, probeFilePathForScene(opts_.scenePath));
    // CSM shadow targets + spot shadow atlas (fixed size; a failure degrades
    // to no shadows).  Created unconditionally like CompareApp/GuiApp so the
    // lighting set's shadow bindings are always written; --no-shadows only
    // zeroes shadowParams.z (sampling off) via fillLightingUBO.
    shadowsActive_ = deferred_.createShadowTargets(ctx_, shadow_);
    if (!shadowsActive_)
        std::fprintf(stderr, "warning: shadow target creation failed, shadows disabled\n");
    spotAtlasActive_ = deferred_.createShadowAtlas(ctx_, spotAtlas_);
    if (!spotAtlasActive_)
        std::fprintf(stderr, "warning: spot shadow atlas creation failed, spot shadows disabled\n");
    // Start both shadow maps in SHADER_READ_ONLY so the descriptor bindings
    // are layout-valid even on frames that render no shadows (--no-shadows or
    // a scene without casting lights).
    submitOneShot(ctx_, [&](VkCommandBuffer cmd) {
        if (shadowsActive_) {
            imageBarrier(cmd, shadow_.image, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_NONE,
                         VK_ACCESS_2_NONE, sync::kFragment, sync::kSampled,
                         VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, kShadowCascadeCount);
            shadow_.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        if (spotAtlasActive_) {
            imageBarrier(cmd, spotAtlas_.image, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_NONE,
                         VK_ACCESS_2_NONE, sync::kFragment, sync::kSampled,
                         VK_IMAGE_ASPECT_DEPTH_BIT);
            spotAtlas_.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
    });
    // Seed the graph's cross-frame tracker with the states the one-shot above
    // left behind; every other tracked target is still UNDEFINED.
    if (shadowsActive_)
        rg_.adopt(shadow_.image, {VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kFragment,
                                  sync::kSampled});
    if (spotAtlasActive_)
        rg_.adopt(spotAtlas_.image, {VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kFragment,
                                     sync::kSampled});
    if (!createShaders()) return false;
    if (!createSceneDescriptors()) return false;
    if (!createAutoExposureResources()) return false;
    if (!createCullResources()) return false;
    if (!createPipelines()) return false;
    if (!createSyncResources()) return false;
    if (!createScreenshotStaging()) return false;

    if (!opts.cameraPath.empty()) {
        if (!loadCameraPath(opts.cameraPath.c_str(), path_)) {
            std::fprintf(stderr, "warning: failed to load camera path %s\n", opts.cameraPath.c_str());
            path_.clear();
        }
    } else if (opts.frames >= 0) {
        // Orbit stays inside the procedural room (walls at |x| = 10, z = -10).
        path_ = generateOrbitPath(opts.frames, {0.f, 2.f, 0.f}, 6.5f, 2.f, 5.5f, 1.f);
    }

    if (path_.empty()) {
        // Interactive free-fly start: per-scene pose when registered (the
        // generic default sits inside Bistro's outer wall), else the Camera
        // constructor default.
        Vec3 pos, fwd;
        if (initialCameraPose(opts_.scenePath, pos, fwd))
            camera_.setPose(pos, fwd, {0.f, 1.f, 0.f});
    }

    return true;
}

bool Renderer::createRenderTargets() {
    const uint32_t dw = opts_.displayWidth;
    const uint32_t dh = opts_.displayHeight;

    auto createRT = [&](ImageResource& rt, uint32_t w, uint32_t h, VkFormat format,
                        VkImageUsageFlags usage, VkImageAspectFlags aspect) {
        rt.width = w;
        rt.height = h;
        rt.format = format;
        if (createImage(ctx_, w, h, format, usage, rt.image, rt.memory) != VK_SUCCESS) return false;
        rt.view = createImageView(ctx_, rt.image, format, aspect);
        return rt.view != VK_NULL_HANDLE;
    };

    // TRANSFER_SRC/DST: some upscalers (DLSS via Streamline) copy the input
    // color/depth/motion into internal buffers instead of just sampling them.
    if (!createRT(gbColor_, renderWidth_, renderHeight_, deferred::kHdrColorFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gbMotion_, renderWidth_, renderHeight_, deferred::kMotionFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gbDepth_, renderWidth_, renderHeight_, deferred::kDepthFormat,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_DEPTH_BIT))
        return false;
    // Translucent coverage mask (reactive/TC mask for upscalers).
    if (!createRT(gbReactive_, renderWidth_, renderHeight_, deferred::kReactiveFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    // Dilated coverage mask (reactive_dilate.comp output; see Renderer.h).
    if (!createRT(gbReactiveDilated_, renderWidth_, renderHeight_, deferred::kReactiveFormat,
                  VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    // Deferred GBuffer attachments (low-res input path / full-res GT path).
    const VkImageUsageFlags gbUsage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (!createRT(gbAlbedo_, renderWidth_, renderHeight_, deferred::kAlbedoFormat, gbUsage,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gbNormal_, renderWidth_, renderHeight_, deferred::kNormalFormat, gbUsage,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gbMaterial_, renderWidth_, renderHeight_, deferred::kMaterialFormat, gbUsage,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gbEmissive_, renderWidth_, renderHeight_, deferred::kEmissiveFormat, gbUsage,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtAlbedo_, dw, dh, deferred::kAlbedoFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtNormal_, dw, dh, deferred::kNormalFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtMaterial_, dw, dh, deferred::kMaterialFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtEmissive_, dw, dh, deferred::kEmissiveFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtDepth_, dw, dh, deferred::kDepthFormat,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_DEPTH_BIT))
        return false;
    // GTAO targets (working RG16F + filtered R16F) for both GBuffer paths.
    const VkImageUsageFlags aoUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    // The AO raw target and the SSR trace target of a path never overlap in
    // time (the raw AO is consumed by the temporal pass before lighting; the
    // trace target is written after the color pyramid), so each pair shares
    // one aliased memory block.  Falls back to plain VMA images on failure.
    auto createAliasedPair = [&](TransientImageArena& arena, ImageResource& raw,
                                 ImageResource& trace, uint32_t w, uint32_t h) -> bool {
        const VkImageUsageFlags usage = aoUsage;
        VkDeviceSize rawBytes = 0, traceBytes = 0, rawAlign = 0, traceAlign = 0;
        uint32_t rawBits = 0, traceBits = 0;
        bool ok = TransientImageArena::queryImageRequirements(ctx_, w, h, kAoRawFormat, usage,
                                                              rawBytes, rawAlign, rawBits) &&
                  TransientImageArena::queryImageRequirements(ctx_, w, h, kSsrTraceFormat, usage,
                                                              traceBytes, traceAlign, traceBits) &&
                  arena.init(ctx_, rawBits & traceBits, std::max(rawBytes, traceBytes),
                             std::max(rawAlign, traceAlign));
        if (ok) {
            raw.image = arena.createAliasedImage(ctx_, w, h, kAoRawFormat, usage);
            trace.image = arena.createAliasedImage(ctx_, w, h, kSsrTraceFormat, usage);
            ok = raw.image != VK_NULL_HANDLE && trace.image != VK_NULL_HANDLE;
        }
        if (ok) {
            raw.view = createImageView(ctx_, raw.image, kAoRawFormat, VK_IMAGE_ASPECT_COLOR_BIT);
            trace.view =
                createImageView(ctx_, trace.image, kSsrTraceFormat, VK_IMAGE_ASPECT_COLOR_BIT);
            ok = raw.view != VK_NULL_HANDLE && trace.view != VK_NULL_HANDLE;
        }
        if (!ok) {
            // Fallback: independent VMA allocations (pre-aliasing behavior).
            if (raw.image) { arena.destroyImage(ctx_, raw.image); raw.image = VK_NULL_HANDLE; }
            if (trace.image) { arena.destroyImage(ctx_, trace.image); trace.image = VK_NULL_HANDLE; }
            arena.destroy(ctx_);
            return createRT(raw, w, h, kAoRawFormat, usage, VK_IMAGE_ASPECT_COLOR_BIT) &&
                   createRT(trace, w, h, kSsrTraceFormat, usage, VK_IMAGE_ASPECT_COLOR_BIT);
        }
        raw.width = trace.width = w;
        raw.height = trace.height = h;
        raw.format = kAoRawFormat;
        trace.format = kSsrTraceFormat;
        raw.arena = &arena;
        trace.arena = &arena;
        return true;
    };
    if (!createAliasedPair(gbAoSsrArena_, gbAoRaw_, gbSsrTrace_, renderWidth_, renderHeight_))
        return false;
    if (!createRT(gbAo_, renderWidth_, renderHeight_, VK_FORMAT_R16_SFLOAT, aoUsage,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createAliasedPair(gtAoSsrArena_, gtAoRaw_, gtSsrTrace_, dw, dh))
        return false;
    if (!createRT(gtAo_, dw, dh, VK_FORMAT_R16_SFLOAT, aoUsage, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    // Bloom pyramids (Phase 6a): 5-level thresholded chains, one per path;
    // GENERAL-for-life, no host layout tracking.
    if (!deferred_.createBloomPyramid(ctx_, renderWidth_, renderHeight_, gbBloom_))
        return false;
    if (!deferred_.createBloomPyramid(ctx_, dw, dh, gtBloom_))
        return false;
    // GT-path motion RT (Phase 6b): the GT GBuffer pass writes per-object
    // motion like the LR path so GT motion blur matches.  Plus the per-path
    // motion-blur + DOF working targets (GENERAL for life).
    if (!createRT(gtMotion_, dw, dh, deferred::kMotionFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!deferred_.createPostFxTargets(ctx_, renderWidth_, renderHeight_, gbPostFx_))
        return false;
    if (!deferred_.createPostFxTargets(ctx_, dw, dh, gtPostFx_))
        return false;
    if (!createLensDirtTexture())
        return false;
    if (!createGradingLut())
        return false;
    if (!deferred_.createDepthPyramid(ctx_, renderWidth_, renderHeight_, gbPyramid_))
        return false;
    if (!deferred_.createDepthPyramid(ctx_, dw, dh, gtPyramid_))
        return false;
    // GTAO view-Z depth chains (XeGTAO DepthMIPFilter) + temporal history.
    if (!deferred_.createDepthPyramid(ctx_, renderWidth_, renderHeight_, gbPyramidAo_,
                                      /*aoFilter=*/true, camera_.nearPlane, camera_.farPlane))
        return false;
    if (!deferred_.createDepthPyramid(ctx_, dw, dh, gtPyramidAo_, /*aoFilter=*/true,
                                      camera_.nearPlane, camera_.farPlane))
        return false;
    if (!deferred_.createAoHistory(ctx_, renderWidth_, renderHeight_, gbAoHist_))
        return false;
    if (!deferred_.createAoHistory(ctx_, dw, dh, gtAoHist_))
        return false;
    // Opaque SSR trace targets + temporal history (Phase 2d), one per path.
    // The trace targets themselves were created above, aliased with the AO
    // raw targets (non-overlapping lifetimes).
    if (!deferred_.createSsrHistory(ctx_, renderWidth_, renderHeight_, gbSsrHist_))
        return false;
    if (!deferred_.createSsrHistory(ctx_, dw, dh, gtSsrHist_))
        return false;
    // Color mip chains for roughness-aware SSR (mip 0 = the lit-color copy).
    if (!deferred_.createColorPyramid(ctx_, renderWidth_, renderHeight_, gbColorPyramid_))
        return false;
    if (!deferred_.createColorPyramid(ctx_, dw, dh, gtColorPyramid_))
        return false;
    // Clustered shading grids (per-path resolution, per-slot buffers).
    if (!deferred_.createClusterGrid(ctx_, renderWidth_, renderHeight_, gbCluster_))
        return false;
    if (!deferred_.createClusterGrid(ctx_, dw, dh, gtCluster_))
        return false;
    // Froxel volumetric fog volumes (per-path resolution; Phase 5a).  A
    // creation failure degrades to no fog (same convention as shadows).
    if (volFogActive_) {
        if (!deferred_.createVolFogVolume(ctx_, renderWidth_, renderHeight_, gbFog_) ||
            !deferred_.createVolFogVolume(ctx_, dw, dh, gtFog_)) {
            std::fprintf(stderr, "warning: volumetric fog volume creation failed, fog disabled\n");
            deferred_.destroyVolFogVolume(ctx_, gbFog_);
            deferred_.destroyVolFogVolume(ctx_, gtFog_);
            volFogActive_ = false;
        }
    }
    if (!createRT(finalImage_, dw, dh, deferred::kHdrColorFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    return true;
}

bool Renderer::createLensDirtTexture() {
    // Procedural lens-dirt mask (UE4-style lens dirt needs no external asset):
    // a handful of soft radial blobs on a dim base, generated with a fixed-seed
    // LCG so the texture is identical on every run.  Peaks are kept <= 0.6 so
    // the default kLensDirtStrength stays subtle.  R8 is enough — the mask
    // scales the (already colored) bloom.
    constexpr uint32_t kSize = 512;
    constexpr uint32_t kBlobCount = 14;
    struct Blob { float x, y, r, i; };
    Blob blobs[kBlobCount];
    uint32_t rng = 0x9E3779B9u; // fixed seed: deterministic mask
    auto frand = [&rng]() {
        rng = rng * 1664525u + 1013904223u;
        return static_cast<float>((rng >> 8) & 0xFFFFFFu) * (1.0f / 16777216.0f);
    };
    for (Blob& b : blobs) {
        b.x = 0.12f + 0.76f * frand();
        b.y = 0.12f + 0.76f * frand();
        b.r = 0.03f + 0.11f * frand();
        b.i = 0.08f + 0.25f * frand();
    }
    std::vector<uint8_t> pixels(static_cast<size_t>(kSize) * kSize);
    for (uint32_t y = 0; y < kSize; ++y) {
        for (uint32_t x = 0; x < kSize; ++x) {
            const float u = (static_cast<float>(x) + 0.5f) / kSize;
            const float v = (static_cast<float>(y) + 0.5f) / kSize;
            float m = 0.05f; // faint uniform film so dirt never fully vanishes
            for (const Blob& b : blobs) {
                const float dx = u - b.x;
                const float dy = v - b.y;
                m += b.i * std::exp(-(dx * dx + dy * dy) / (b.r * b.r));
            }
            m = std::min(m, 0.6f);
            pixels[static_cast<size_t>(y) * kSize + x] =
                static_cast<uint8_t>(m * (255.0f / 0.6f) + 0.5f);
        }
    }

    lensDirt_.width = kSize;
    lensDirt_.height = kSize;
    lensDirt_.format = VK_FORMAT_R8_UNORM;
    if (createImage(ctx_, kSize, kSize, lensDirt_.format,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, lensDirt_.image,
                    lensDirt_.memory) != VK_SUCCESS)
        return false;
    lensDirt_.view = createImageView(ctx_, lensDirt_.image, lensDirt_.format,
                                     VK_IMAGE_ASPECT_COLOR_BIT);
    if (!lensDirt_.view) return false;

    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation stagingMemory = VK_NULL_HANDLE;
    if (createBuffer(ctx_, pixels.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging, stagingMemory) != VK_SUCCESS)
        return false;
    void* mapped = nullptr;
    if (vmaMapMemory(ctx_.allocator, stagingMemory, &mapped) != VK_SUCCESS) return false;
    std::memcpy(mapped, pixels.data(), pixels.size());
    vmaUnmapMemory(ctx_.allocator, stagingMemory);
    // copyBufferToImage transitions UNDEFINED -> TRANSFER_DST -> SHADER_READ_ONLY.
    submitUploadOneShot(
        ctx_,
        [&](VkCommandBuffer cmd) {
            copyBufferToImageTransferStage(cmd, staging, lensDirt_.image, kSize, kSize);
        },
        [&](VkCommandBuffer cmd) { transitionImageToShaderRead(cmd, lensDirt_.image); });
    vmaDestroyBuffer(ctx_.allocator, staging, stagingMemory);
    lensDirt_.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return true;
}

bool Renderer::createGradingLut() {
    // Phase 6c: --lut loads a .cube file; otherwise the procedural identity
    // LUT (bit-neutral output with default grading).  The CPU copy mirrors
    // the GPU one for the screenshot path (savePngFromHalfRgba).
    if (!opts_.lutPath.empty()) {
        if (!loadCubeLut(opts_.lutPath.c_str(), gradingLutCpu_)) {
            std::fprintf(stderr, "lut: falling back to identity (%s)\n", opts_.lutPath.c_str());
            gradingLutCpu_ = makeIdentityLut();
        } else {
            std::fprintf(stderr, "lut: loaded %s (%u^3)\n", opts_.lutPath.c_str(),
                         gradingLutCpu_.size);
        }
    } else {
        gradingLutCpu_ = makeIdentityLut();
    }
    return gradingLutGpu_.create(ctx_, gradingLutCpu_);
}

bool Renderer::loadShader(const char* name, VkShaderModule& out) {
    const std::string path = sr::resolveShaderPath(SR_SHADER_DIR, name);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "failed to open shader: %s\n", path.c_str());
        return false;
    }
    const std::streamoff size = file.tellg();
    if (size <= 0 || size % 4 != 0) return false;
    file.seekg(0);
    std::vector<uint32_t> code(static_cast<size_t>(size) / 4);
    file.read(reinterpret_cast<char*>(code.data()), size);

    VkShaderModuleCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = static_cast<size_t>(size);
    ci.pCode = code.data();
    return vkCreateShaderModule(ctx_.device, &ci, nullptr, &out) == VK_SUCCESS;
}

bool Renderer::createShaders() {
    // Deferred shaders live in DeferredCore; the viewer only needs present.frag.
    return loadShader("present.frag.spv", presentFrag_);
}

bool Renderer::createSceneDescriptors() {
    // Present set: 0 = final HDR image, 1 = accumulated bloom mip 0 of the
    // presented path (lens dirt), 2 = procedural lens-dirt mask, 3 = grading
    // LUT (Phase 6c).
    VkDescriptorSetLayoutBinding presentBindings[4] = {};
    for (uint32_t i = 0; i < 4; ++i) {
        presentBindings[i].binding = i;
        presentBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        presentBindings[i].descriptorCount = 1;
        presentBindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo presentLayoutCi = {};
    presentLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    presentLayoutCi.bindingCount = 4;
    presentLayoutCi.pBindings = presentBindings;
    if (vkCreateDescriptorSetLayout(ctx_.device, &presentLayoutCi, nullptr, &presentSetLayout_) != VK_SUCCESS)
        return false;

    VkDescriptorPoolSize sizes[5] = {};
    // Hi-Z / color downsample sets: one per mip per pyramid (sampler + storage image).
    const uint32_t hizSets = gbPyramid_.mipCount + gtPyramid_.mipCount + gbPyramidAo_.mipCount +
                             gtPyramidAo_.mipCount;
    const uint32_t colorSets = gbColorPyramid_.mipCount + gtColorPyramid_.mipCount;
    const uint32_t fogPaths = volFogActive_ ? 2 : 0; // froxel fog sets per path (Phase 5a)
    sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = deferred::kMaxTextures + 4 + // texture array + present (image + bloom + dirt + lut)
                               14 * kFramesInFlight * 2 + // lighting sets (GB/GT), shadow + atlas + 2 probe arrays
                               8 * kFramesInFlight * 2 +  // transparent sets (GB/GT), +SSR+shadow+fog volume
                               11 * kFramesInFlight * 2 + // opaque-SSR trace sets (GB/GT), +2 probe arrays
                               2 * 2 + 3 * 4 + 1 * 4 +     // ssao + temporal + blur samplers
                               3 * 4 +                     // ssr temporal samplers (GB/GT x2 sets)
                               20 +                        // bloom pyramid extract/down/up/comp (GB/GT)
                               17 * 2 +                    // post-fx MB/DOF sets (Phase 6b, GB/GT)
                               1 +                         // auto-exposure HDR source
                               2 +                         // reactive mask dilate: src mask + motion samplers
                               14 * fogPaths +             // volfog light/temporal/march/composite samplers
                               2 +                         // occlusion cull sets: Hi-Z chains (GB/GT, Phase 7a)
                               hizSets + colorSets;
    sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[1].descriptorCount = kFramesInFlight * 10 + // scene + lighting(+probe) + 2 transparent + 2 SSR(+probe) UBOs
                               kClusterSlots * fogPaths; // volfog light sets
    sizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    sizes[2].descriptorCount = kFramesInFlight;
    sizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    // ssao raw + temporal history + blur (GB/GT) + pyramid mips + SSR trace
    // targets + SSR temporal history write / scene-color RMW (GB/GT x2 sets)
    // + volfog inject/light/temporal/march/composite storage (per fog path)
    // + bloom pyramid dst mips (GB/GT)
    sizes[3].descriptorCount = 10 + 8 + hizSets + colorSets + kFramesInFlight * 2 + 20 +
                               11 * 2 + 2 * 2 + // post-fx MB/DOF storage + MB copy-back (Phase 6b, GB/GT)
                               1 +              // reactive mask dilate
                               8 * fogPaths;
    sizes[4].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[4].descriptorCount = 2 + // auto-exposure histogram + state
                               2 * kFramesInFlight * 2 + // lighting sets: cluster lights + grid SSBOs
                               2 * kClusterSlots * 2 +    // cluster assign sets (GB/GT paths)
                               kFramesInFlight +          // scene sets: instance SSBO (Phase 7a)
                               2 * 2 +                    // occlusion cull sets (GB/GT, Phase 7a)
                               2 * kClusterSlots * fogPaths; // volfog light sets: lights + grid SSBOs
    VkDescriptorPoolCreateInfo poolCi = {};
    poolCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCi.maxSets = kFramesInFlight * 7 + 12 + 20 + hizSets + colorSets + 1 + 4 +
                     kClusterSlots * 2 + // +12 ssao/temporal/blur, +20 bloom pyramid, +1 auto-exposure, +4 ssr temporal, +cluster assign
                     8 * 2 + 1 * 2 +      // post-fx MB/DOF sets + MB copy-back (Phase 6b, GB/GT)
                     1 +                  // reactive mask dilate
                     2 +                  // occlusion cull sets (GB/GT, Phase 7a)
                     8 * fogPaths;        // volfog inject/light/temporal/march/composite sets
    poolCi.poolSizeCount = 5;
    poolCi.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(ctx_.device, &poolCi, nullptr, &descriptorPool_) != VK_SUCCESS)
        return false;

    if (!deferred_.writeDepthPyramidSets(ctx_, descriptorPool_, gbDepth_.view, gbPyramid_))
        return false;
    if (!deferred_.writeDepthPyramidSets(ctx_, descriptorPool_, gtDepth_.view, gtPyramid_))
        return false;
    if (!deferred_.writeDepthPyramidSets(ctx_, descriptorPool_, gbDepth_.view, gbPyramidAo_))
        return false;
    if (!deferred_.writeDepthPyramidSets(ctx_, descriptorPool_, gtDepth_.view, gtPyramidAo_))
        return false;
    // Color chain set 0 samples the lit HDR target (gbColor_ / finalImage_).
    if (!deferred_.writeColorPyramidSets(ctx_, descriptorPool_, gbColor_.view, gbColorPyramid_))
        return false;
    if (!deferred_.writeColorPyramidSets(ctx_, descriptorPool_, finalImage_.view, gtColorPyramid_))
        return false;

    if (!deferred_.createMaterialUbo(ctx_, scene_, materialUbo_, materialUboMemory_,
                                     materialStride_))
        return false;

    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = descriptorPool_;
        alloc.descriptorSetCount = 1;
        const VkDescriptorSetLayout sceneLayout = deferred_.sceneSetLayout();
        alloc.pSetLayouts = &sceneLayout;
        if (vkAllocateDescriptorSets(ctx_.device, &alloc, &frames_[i].sceneSet) != VK_SUCCESS) return false;
    }

    {
        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = descriptorPool_;
        alloc.descriptorSetCount = 1;
        const VkDescriptorSetLayout texLayout = deferred_.textureSetLayout();
        alloc.pSetLayouts = &texLayout;
        if (vkAllocateDescriptorSets(ctx_.device, &alloc, &textureSet_) != VK_SUCCESS) return false;
        deferred_.writeTextureSet(ctx_, textureSet_, scene_);
    }

    {
        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = descriptorPool_;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &presentSetLayout_;
        if (vkAllocateDescriptorSets(ctx_.device, &alloc, &presentSet_) != VK_SUCCESS) return false;

        // Binding 1: accumulated bloom mip 0 of the PRESENTED path (upscaler
        // -> LR chain, native -> GT chain), for the lens-dirt term.  The
        // pyramid lives in GENERAL; the mip content is undefined when bloom
        // is off, but the dirt strength is forced to 0 then, so the shader
        // never samples it.
        const bool useUpscaler = opts_.upscalerName != "none" && upscaler_ != nullptr;
        VkDescriptorImageInfo info[4] = {};
        info[0].sampler = deferred_.textureSampler();
        info[0].imageView = finalImage_.view;
        info[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        info[1].sampler = deferred_.gbufferSampler(); // linear clamp
        info[1].imageView = useUpscaler ? gbBloom_.mipViews[0] : gtBloom_.mipViews[0];
        info[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        info[2].sampler = deferred_.gbufferSampler();
        info[2].imageView = lensDirt_.view;
        info[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        info[3].sampler = gradingLutGpu_.sampler();
        info[3].imageView = gradingLutGpu_.view();
        info[3].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = presentSet_;
        write.dstBinding = 0;
        write.descriptorCount = 4;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = info;
        vkUpdateDescriptorSets(ctx_.device, 1, &write, 0, nullptr);
    }

    return true;
}

bool Renderer::createCullResources() {
    // Phase 7a: the instance SSBO upper bound is the full instance list (the
    // candidate build skips blend/skinned/frustum-culled entries); one cull
    // channel per path, bound to that path's Hi-Z chain.
    const uint32_t capacity = static_cast<uint32_t>(scene_.instances.size());
    if (!deferred_.createInstanceBuffer(ctx_, capacity, instances_)) return false;
    if (!deferred_.createCullChannel(ctx_, capacity, gbCull_)) return false;
    if (!deferred_.createCullChannel(ctx_, capacity, gtCull_)) return false;
    if (!deferred_.writeCullSet(ctx_, descriptorPool_, instances_, gbPyramid_.chainView, gbCull_))
        return false;
    if (!deferred_.writeCullSet(ctx_, descriptorPool_, instances_, gtPyramid_.chainView, gtCull_))
        return false;
    // Scene set binding 3 (instance SSBO) for both frame slots.
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        if (instances_.buffer)
            deferred_.writeSceneInstanceBinding(ctx_, frames_[i].sceneSet, instances_.buffer);
    }
    cullInstCpu_.resize(capacity);
    cullCmdCpu_.resize(capacity);
    return true;
}

bool Renderer::createAutoExposureResources() {
    if (!opts_.autoExposure) return true;
    // Seed the smoothed EV with the (preset/manual) exposure so the frames
    // before the first readback match the old fixed-exposure look.
    const float initialEV = -std::log2(opts_.exposure);
    const bool useUpscaler = opts_.upscalerName != "none" && upscaler_ != nullptr;
    const ImageResource& src = useUpscaler ? gbColor_ : finalImage_;
    return deferred_.createExposureChannel(ctx_, descriptorPool_, src.view, src.width,
                                           src.height, initialEV, exposureChannel_);
}

bool Renderer::createPipelines() {
    // The GBuffer/GT/lighting pipelines live in DeferredCore; only the
    // swapchain present pipeline (fullscreen triangle + tonemap) remains here.
    VkPushConstantRange presentPush = {};
    presentPush.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    presentPush.offset = 0;
    presentPush.size = 80; // 5 x vec4: exposure + lensA + lensB + gradeA + gradeB (present.frag)
    VkPipelineLayoutCreateInfo presentLayoutCi = {};
    presentLayoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    presentLayoutCi.setLayoutCount = 1;
    presentLayoutCi.pSetLayouts = &presentSetLayout_;
    presentLayoutCi.pushConstantRangeCount = 1;
    presentLayoutCi.pPushConstantRanges = &presentPush;
    if (vkCreatePipelineLayout(ctx_.device, &presentLayoutCi, nullptr, &presentPipelineLayout_) != VK_SUCCESS)
        return false;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.f;

    VkPipelineMultisampleStateCreateInfo multisample = {};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineVertexInputStateCreateInfo emptyVertexInput = {};
    emptyVertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineDepthStencilStateCreateInfo noDepth = {};
    noDepth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

    VkDynamicState dynamicStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState = {};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineShaderStageCreateInfo presentStages[2] = {};
    presentStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    presentStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    presentStages[0].module = deferred_.fullscreenVert();
    presentStages[0].pName = "main";
    presentStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    presentStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    presentStages[1].module = presentFrag_;
    presentStages[1].pName = "main";

    VkPipelineColorBlendAttachmentState presentBlend = {};
    presentBlend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo presentColorBlend = {};
    presentColorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    presentColorBlend.attachmentCount = 1;
    presentColorBlend.pAttachments = &presentBlend;

    VkGraphicsPipelineCreateInfo presentCi = {};
    presentCi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    presentCi.stageCount = 2;
    presentCi.pStages = presentStages;
    presentCi.pVertexInputState = &emptyVertexInput;
    presentCi.pInputAssemblyState = &inputAssembly;
    presentCi.pViewportState = &viewportState;
    presentCi.pRasterizationState = &rasterizer;
    presentCi.pMultisampleState = &multisample;
    presentCi.pDepthStencilState = &noDepth;
    presentCi.pColorBlendState = &presentColorBlend;
    presentCi.pDynamicState = &dynamicState;
    presentCi.layout = presentPipelineLayout_;

    VkPipelineRenderingCreateInfo presentRendering = {};
    presentRendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    presentRendering.colorAttachmentCount = 1;
    const VkFormat presentFormat = swapchain_.format();
    presentRendering.pColorAttachmentFormats = &presentFormat;
    presentCi.pNext = &presentRendering;
    if (createGraphicsPipeline(ctx_, presentCi, presentPipeline_) != VK_SUCCESS)
        return false;

    return true;
}

bool Renderer::createSyncResources() {
    VkCommandBufferAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = ctx_.framePool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = kFramesInFlight;
    VkCommandBuffer cmds[kFramesInFlight] = {};
    if (vkAllocateCommandBuffers(ctx_.device, &alloc, cmds) != VK_SUCCESS) return false;
    for (uint32_t i = 0; i < kFramesInFlight; ++i) frames_[i].cmd = cmds[i];

    // Cluster assignment sets (per slot, both paths); the buffers were created
    // in createRenderTargets, the pool in createSceneDescriptors.
    if (!deferred_.writeClusterGridSets(ctx_, descriptorPool_, gbCluster_)) return false;
    if (!deferred_.writeClusterGridSets(ctx_, descriptorPool_, gtCluster_)) return false;

    VkSemaphoreCreateInfo semCi = {};
    semCi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fenceCi = {};
    fenceCi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCi.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    const VkDeviceSize uboSize = sizeof(SceneUBO);
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        if (vkCreateSemaphore(ctx_.device, &semCi, nullptr, &frames_[i].imageAvailable) != VK_SUCCESS) return false;
        if (vkCreateFence(ctx_.device, &fenceCi, nullptr, &frames_[i].fence) != VK_SUCCESS) return false;

        if (createBuffer(ctx_, uboSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         frames_[i].ubo, frames_[i].uboMemory) != VK_SUCCESS)
            return false;
        vmaMapMemory(ctx_.allocator, frames_[i].uboMemory, &frames_[i].uboMapped);

        if (createBuffer(ctx_, sizeof(LightingUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         frames_[i].lightingUbo, frames_[i].lightingUboMemory) != VK_SUCCESS)
            return false;
        vmaMapMemory(ctx_.allocator, frames_[i].lightingUboMemory,
                    &frames_[i].lightingUboMapped);

        VkDescriptorBufferInfo sceneBuf = {};
        sceneBuf.buffer = frames_[i].ubo;
        sceneBuf.offset = 0;
        sceneBuf.range = uboSize;
        VkDescriptorBufferInfo materialBuf = {};
        materialBuf.buffer = materialUbo_;
        materialBuf.offset = 0;
        materialBuf.range = sizeof(MaterialUBO);

        VkWriteDescriptorSet writes[2] = {};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = frames_[i].sceneSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo = &sceneBuf;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = frames_[i].sceneSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        writes[1].pBufferInfo = &materialBuf;
        vkUpdateDescriptorSets(ctx_.device, 2, writes, 0, nullptr);

        // Joint palette for skinned draws: per-slot buffer, matching the slot
        // advanceToFrame(frameIndex) writes (slot = frameIndex % frames).
        if (scene_.hasSkinnedMeshes())
            deferred_.writeSceneSkinBinding(ctx_, frames_[i].sceneSet, scene_.skinPalette(i));

        // Lighting sets: one per frame slot per path (low-res GB / full-res GT).
        const VkDescriptorSetLayout lightingLayout = deferred_.lightingSetLayout();
        VkDescriptorSetAllocateInfo lightAlloc = {};
        lightAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        lightAlloc.descriptorPool = descriptorPool_;
        lightAlloc.descriptorSetCount = 1;
        lightAlloc.pSetLayouts = &lightingLayout;
        if (vkAllocateDescriptorSets(ctx_.device, &lightAlloc, &frames_[i].lightingSetGb) != VK_SUCCESS)
            return false;
        if (vkAllocateDescriptorSets(ctx_.device, &lightAlloc, &frames_[i].lightingSetGt) != VK_SUCCESS)
            return false;

        const VkImageView shadowView = shadowsActive_ ? shadow_.arrayView : VK_NULL_HANDLE;
        const VkImageView spotAtlasView = spotAtlasActive_ ? spotAtlas_.view : VK_NULL_HANDLE;
        deferred_.writeLightingSet(ctx_, frames_[i].lightingSetGb, frames_[i].lightingUbo,
                                   gbAlbedo_.view, gbNormal_.view, gbMaterial_.view,
                                   gbEmissive_.view, gbDepth_.view, gbAo_.view, shadowView,
                                   spotAtlasView, gbCluster_.lightsBuffer[i], gbCluster_.gridBuffer[i]);
        deferred_.writeLightingSet(ctx_, frames_[i].lightingSetGt, frames_[i].lightingUbo,
                                   gtAlbedo_.view, gtNormal_.view, gtMaterial_.view,
                                   gtEmissive_.view, gtDepth_.view, gtAo_.view, shadowView,
                                   spotAtlasView, gtCluster_.lightsBuffer[i], gtCluster_.gridBuffer[i]);

        // Per-path transparent sets (each binds its own path's SSAO texture).
        const VkDescriptorSetLayout transparentLayout = deferred_.transparentSetLayout();
        VkDescriptorSetAllocateInfo transparentAlloc = {};
        transparentAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        transparentAlloc.descriptorPool = descriptorPool_;
        transparentAlloc.descriptorSetCount = 1;
        transparentAlloc.pSetLayouts = &transparentLayout;
        if (vkAllocateDescriptorSets(ctx_.device, &transparentAlloc,
                                     &frames_[i].transparentSetGb) != VK_SUCCESS)
            return false;
        if (vkAllocateDescriptorSets(ctx_.device, &transparentAlloc,
                                     &frames_[i].transparentSetGt) != VK_SUCCESS)
            return false;
        deferred_.writeTransparentSet(ctx_, frames_[i].transparentSetGb, frames_[i].lightingUbo,
                                      gbAo_.view, shadowView, gbColorPyramid_.chainView,
                                      gbPyramid_.chainView,
                                      volFogActive_ ? gbFog_.intView : VK_NULL_HANDLE);
        deferred_.writeTransparentSet(ctx_, frames_[i].transparentSetGt, frames_[i].lightingUbo,
                                      gtAo_.view, shadowView, gtColorPyramid_.chainView,
                                      gtPyramid_.chainView,
                                      volFogActive_ ? gtFog_.intView : VK_NULL_HANDLE);

        // Per-path opaque-SSR sets (binding 0 reuses this frame's lighting UBO).
        const VkDescriptorSetLayout ssrLayout = deferred_.ssrSetLayout();
        VkDescriptorSetAllocateInfo ssrAlloc = {};
        ssrAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ssrAlloc.descriptorPool = descriptorPool_;
        ssrAlloc.descriptorSetCount = 1;
        ssrAlloc.pSetLayouts = &ssrLayout;
        if (vkAllocateDescriptorSets(ctx_.device, &ssrAlloc, &frames_[i].ssrSetGb) != VK_SUCCESS)
            return false;
        if (vkAllocateDescriptorSets(ctx_.device, &ssrAlloc, &frames_[i].ssrSetGt) != VK_SUCCESS)
            return false;
        deferred_.writeSsrSet(ctx_, frames_[i].ssrSetGb, frames_[i].lightingUbo, gbAlbedo_.view,
                              gbNormal_.view, gbMaterial_.view, gbDepth_.view, gbAo_.view,
                              gbColorPyramid_.chainView, gbPyramid_.chainView, gbSsrTrace_.view);
        deferred_.writeSsrSet(ctx_, frames_[i].ssrSetGt, frames_[i].lightingUbo, gtAlbedo_.view,
                              gtNormal_.view, gtMaterial_.view, gtDepth_.view, gtAo_.view,
                              gtColorPyramid_.chainView, gtPyramid_.chainView, gtSsrTrace_.view);
    }

    // SSAO sets (static bindings; per-frame data goes through push constants).
    {
        VkDescriptorSetAllocateInfo ssaoAlloc = {};
        ssaoAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ssaoAlloc.descriptorPool = descriptorPool_;
        ssaoAlloc.descriptorSetCount = 1;
        const VkDescriptorSetLayout ssaoLayout = deferred_.ssaoSetLayout();
        ssaoAlloc.pSetLayouts = &ssaoLayout;
        if (vkAllocateDescriptorSets(ctx_.device, &ssaoAlloc, &ssaoSetGb_) != VK_SUCCESS) return false;
        if (vkAllocateDescriptorSets(ctx_.device, &ssaoAlloc, &ssaoSetGt_) != VK_SUCCESS) return false;
        deferred_.writeSsaoSet(ctx_, ssaoSetGb_, gbPyramidAo_.chainView, gbNormal_.view,
                               gbAoRaw_.view);
        deferred_.writeSsaoSet(ctx_, ssaoSetGt_, gtPyramidAo_.chainView, gtNormal_.view,
                               gtAoRaw_.view);
        if (!deferred_.writeAoHistorySets(ctx_, descriptorPool_, gbAoRaw_.view, gbDepth_.view,
                                          gbAo_.view, gbAoHist_))
            return false;
        if (!deferred_.writeAoHistorySets(ctx_, descriptorPool_, gtAoRaw_.view, gtDepth_.view,
                                          gtAo_.view, gtAoHist_))
            return false;
        if (!deferred_.writeSsrHistorySets(ctx_, descriptorPool_, gbSsrTrace_.view,
                                           gbDepth_.view, gbColor_.view, gbSsrHist_))
            return false;
        if (!deferred_.writeSsrHistorySets(ctx_, descriptorPool_, gtSsrTrace_.view,
                                           gtDepth_.view, finalImage_.view, gtSsrHist_))
            return false;
    }

    // Froxel volumetric fog sets (Phase 5a): static bindings of the per-path
    // volumes, this path's cluster grids + per-slot lighting UBOs, the shared
    // shadow maps, the GBuffer depth and the lit HDR target.
    if (volFogActive_) {
        const VkBuffer lightingUbos[kClusterSlots] = {frames_[0].lightingUbo,
                                                      frames_[1].lightingUbo};
        const VkImageView shadowView = shadowsActive_ ? shadow_.arrayView : VK_NULL_HANDLE;
        const VkImageView spotAtlasView = spotAtlasActive_ ? spotAtlas_.view : VK_NULL_HANDLE;
        if (!deferred_.writeVolFogSets(ctx_, descriptorPool_, gbFog_, gbCluster_, lightingUbos,
                                       shadowView, spotAtlasView, gbDepth_.view, gbColor_.view))
            return false;
        if (!deferred_.writeVolFogSets(ctx_, descriptorPool_, gtFog_, gtCluster_, lightingUbos,
                                       shadowView, spotAtlasView, gtDepth_.view, finalImage_.view))
            return false;
    }

    // Bloom pyramid sets (pool-owned; mip views already exist).
    if (!deferred_.writeBloomPyramidSets(ctx_, descriptorPool_, gbColor_.view, gbBloom_))
        return false;
    if (!deferred_.writeBloomPyramidSets(ctx_, descriptorPool_, finalImage_.view, gtBloom_))
        return false;
    // Post-fx sets (Phase 6b): lit HDR target + motion + depth per path.
    if (!deferred_.writePostFxSets(ctx_, descriptorPool_, gbColor_.view, gbMotion_.view,
                                   gbDepth_.view, gbPostFx_))
        return false;
    if (!deferred_.writePostFxSets(ctx_, descriptorPool_, finalImage_.view, gtMotion_.view,
                                   gtDepth_.view, gtPostFx_))
        return false;
    // Coverage mask dilate set: raw mask + motion -> dilated/gated target
    // (pool-owned set).
    if (!deferred_.writeReactiveDilateSet(ctx_, descriptorPool_, gbReactive_.view,
                                          gbReactiveDilated_.view, gbMotion_.view,
                                          reactiveDilateSet_))
        return false;

    // One renderFinished semaphore per swapchain image: the submit signals the
    // semaphore for the acquired image, and present waits on that same image's
    // semaphore.  Reusing per-image semaphores avoids reuse while a previous
    // present is still pending.
    renderFinished_.resize(swapchain_.imageCount(), VK_NULL_HANDLE);
    for (size_t i = 0; i < renderFinished_.size(); ++i) {
        if (vkCreateSemaphore(ctx_.device, &semCi, nullptr, &renderFinished_[i]) != VK_SUCCESS) return false;
    }

    if (!timestamps_.create(ctx_, kFramesInFlight)) return false;
    return true;
}

bool Renderer::recreateSwapchain(uint32_t width, uint32_t height, bool vsync) {
    // In-flight frames may still reference the old swapchain images, so wait
    // for all queued work to finish before destroying the previous semaphores
    // and swapchain (the image count may change after the recreate).
    vkDeviceWaitIdle(ctx_.device);

    for (VkSemaphore sem : renderFinished_) {
        if (sem) vkDestroySemaphore(ctx_.device, sem, nullptr);
    }
    renderFinished_.clear();

    if (!swapchain_.create(ctx_, width, height, vsync, /*allowMailbox=*/true, opts_.hdr))
        return false;

    // Rebuild one present semaphore per swapchain image: the new swapchain may
    // have a different image count than the previous one.
    VkSemaphoreCreateInfo semCi = {};
    semCi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    renderFinished_.resize(swapchain_.imageCount(), VK_NULL_HANDLE);
    for (size_t i = 0; i < renderFinished_.size(); ++i) {
        if (vkCreateSemaphore(ctx_.device, &semCi, nullptr, &renderFinished_[i]) != VK_SUCCESS) return false;
    }
    return true;
}

bool Renderer::createScreenshotStaging() {
    screenshotSize_ = static_cast<VkDeviceSize>(opts_.displayWidth) * opts_.displayHeight * 8; // RGBA16F
    if (createBuffer(ctx_, screenshotSize_, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     screenshotStaging_, screenshotStagingMemory_) != VK_SUCCESS)
        return false;
    vmaMapMemory(ctx_.allocator, screenshotStagingMemory_, &screenshotMapped_);
    return true;
}

void Renderer::updateSceneUBO(uint32_t frameIndex, bool jitter, uint32_t renderW, uint32_t renderH,
                              const Mat4& view, const Mat4& proj, const Mat4& projJittered,
                              const Mat4& prevViewProj) {
    FrameResources& fr = frames_[frameIndex % kFramesInFlight];
    SceneUBO ubo;
    deferred_.fillSceneUBO(ubo, scene_, camera_, view, proj, projJittered, prevViewProj,
                           renderW, renderH, jitterX_, jitterY_, jitter);
    std::memcpy(fr.uboMapped, &ubo, sizeof(ubo));
}

void Renderer::updateLightingUBO(uint32_t frameIndex, const Mat4& viewProj,
                                 const Mat4& invViewProj, const ShadowFrame* shadow,
                                 const std::vector<Light>* overrideLights) {
    FrameResources& fr = frames_[frameIndex % kFramesInFlight];
    LightingUBO ubo;
    deferred_.fillLightingUBO(ubo, scene_, camera_, viewProj, invViewProj, overrideLights, shadow,
                              iblIntensity_);
    ubo.shadowAtlasParams[3] = opts_.contactShadows ? 1.f : 0.f;
    std::memcpy(fr.lightingUboMapped, &ubo, sizeof(ubo));
    // Full point/spot light set for the clustered pass (same slot rule as the
    // UBO: the slot's fence passed before recording).
    const uint32_t slot = frameIndex % kFramesInFlight;
    const std::vector<Light>& lights = DeferredCore::effectiveLights(scene_, overrideLights);
    deferred_.fillClusterLights(gbCluster_.lightsMapped[slot], lights);
    deferred_.fillClusterLights(gtCluster_.lightsMapped[slot], lights);
}

void Renderer::applyCameraKeyframe(uint32_t frameIndex) {
    if (path_.empty()) return;
    const CameraKeyframe& kf = path_[frameIndex % path_.size()];
    camera_.setPose(kf.position, kf.forward, kf.up);
}

void Renderer::updateCamera(uint32_t frameIndex, float dt) {
    if (!path_.empty()) {
        applyCameraKeyframe(frameIndex);
    } else {
        camera_.updateFreeFly(window_.input(), dt);
    }
    window_.clearMouseDelta();
}

void Renderer::captureScreenshotIntoStaging(VkCommandBuffer cmd) {
    // finalImage_ is TRANSFER_SRC here (the enclosing graph pass declared it).
    VkBufferImageCopy region = {};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {opts_.displayWidth, opts_.displayHeight, 1};
    vkCmdCopyImageToBuffer(cmd, finalImage_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           screenshotStaging_, 1, &region);
}

void Renderer::saveScreenshot(const std::string& path) {
    if (!screenshotMapped_) return;
    // Same exposure and grading as the present pass of the captured frame, so
    // the CPU tonemap matches the on-screen (GPU) result bit-for-bit.
    if (!savePngFromHalfRgba(path.c_str(), static_cast<const uint8_t*>(screenshotMapped_),
                             opts_.displayWidth, opts_.displayHeight, displayExposure(), grading_,
                             &gradingLutCpu_)) {
        std::fprintf(stderr, "failed to save screenshot %s\n", path.c_str());
    }
}

// --- Offline reflection-probe baking (--bake-probes, Phase 4c-2) --------------
// For each registry-placed probe the scene is rendered from the probe position
// into the six cube faces at kBakeSize^2, reusing the real GBuffer + deferred
// lighting pipeline (GT variant: no motion/reactive attachments) at 90 deg
// FOV — so the bake sees the same punctual lights, materials, emissive and the
// global-env skybox the viewer shades with.  Documented quality
// simplifications: no shadow maps, no SSAO (constant 1), no SSR/transparency,
// and the source cube keeps a single mip (the prefilter shaders clamp their
// LOD).  The bake is an offline command: nothing here runs during bench.
//
// Face conventions: each face renders with a lookAt along kFaceDir (up from
// kFaceUp); the raster image is then mirrored horizontally vs the Vulkan
// cube-face layout the IBL shaders sample (cubeDir() in ibl_*.comp), so the
// readback flips X before writing the face.
bool Renderer::bakeProbes() {
    const std::vector<ReflectionProbe>& defs = scene_.probes;
    const std::string outPath = probeFilePathForScene(opts_.scenePath);
    if (defs.empty()) {
        std::fprintf(stderr, "bake-probes: scene has no probe placements\n");
        return true;
    }
    constexpr uint32_t S = ReflectionProbes::kBakeSize;
    std::fprintf(stderr, "bake-probes: %zu probe(s), %ux%u per face -> %s\n", defs.size(), S, S,
                 outPath.c_str());

    // Cube face capture orientations (see the comment above; readback flips X).
    static const Vec3 kFaceDir[6] = {{1.f, 0.f, 0.f}, {-1.f, 0.f, 0.f}, {0.f, 1.f, 0.f},
                                     {0.f, -1.f, 0.f}, {0.f, 0.f, 1.f},  {0.f, 0.f, -1.f}};
    static const Vec3 kFaceUp[6] = {{0.f, 1.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, -1.f},
                                    {0.f, 0.f, 1.f}, {0.f, 1.f, 0.f}, {0.f, 1.f, 0.f}};

    // --- bake-local targets (128^2 GBuffer + HDR + white AO stand-in) ----------
    // Phase 6b: the GT GBuffer pipeline now has five attachments, so the bake
    // keeps a (discarded) motion target too.
    ImageResource albedo, normal, material, emissive, depth, hdr, ao, motion;
    auto createRT = [&](ImageResource& rt, VkFormat format, VkImageUsageFlags usage,
                        VkImageAspectFlags aspect) {
        rt.width = S;
        rt.height = S;
        rt.format = format;
        if (createImage(ctx_, S, S, format, usage, rt.image, rt.memory) != VK_SUCCESS)
            return false;
        rt.view = createImageView(ctx_, rt.image, format, aspect);
        return rt.view != VK_NULL_HANDLE;
    };
    const VkImageUsageFlags gbUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    const bool targetsOk =
        createRT(albedo, deferred::kAlbedoFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT) &&
        createRT(normal, deferred::kNormalFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT) &&
        createRT(material, deferred::kMaterialFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT) &&
        createRT(emissive, deferred::kEmissiveFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT) &&
        createRT(motion, deferred::kMotionFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT) &&
        createRT(depth, deferred::kDepthFormat,
                 VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                 VK_IMAGE_ASPECT_DEPTH_BIT) &&
        createRT(hdr, deferred::kHdrColorFormat,
                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                 VK_IMAGE_ASPECT_COLOR_BIT) &&
        createRT(ao, VK_FORMAT_R16_SFLOAT,
                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                 VK_IMAGE_ASPECT_COLOR_BIT);
    if (!targetsOk) return false;
    // White AO (the bake skips the GTAO chain; AO=1 keeps the env term intact).
    submitOneShot(ctx_, [&](VkCommandBuffer cmd) {
        imageBarrier(cmd, ao.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_CLEAR_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT);
        const VkClearColorValue white = {{1.f, 1.f, 1.f, 1.f}};
        const VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, ao.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &white, 1,
                             &range);
        imageBarrier(cmd, ao.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_CLEAR_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT, sync::kFragment, sync::kSampled);
    });
    ao.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // --- bake-local descriptors (lighting set + cluster grid at 128^2) ---------
    VkDescriptorPoolSize sizes[3] = {};
    sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = 14; // lighting set image bindings
    sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[1].descriptorCount = 2; // lighting UBO + probe UBO
    sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[2].descriptorCount = 2 + 2 * kClusterSlots; // lighting SSBOs + cluster assign sets
    VkDescriptorPoolCreateInfo poolCi = {};
    poolCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCi.maxSets = 1 + kClusterSlots;
    poolCi.poolSizeCount = 3;
    poolCi.pPoolSizes = sizes;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(ctx_.device, &poolCi, nullptr, &pool) != VK_SUCCESS) return false;

    ClusterGrid cluster;
    bool bakeOk = deferred_.createClusterGrid(ctx_, S, S, cluster) &&
                  deferred_.writeClusterGridSets(ctx_, pool, cluster);

    VkBuffer lightingUbo = VK_NULL_HANDLE;
    VmaAllocation lightingUboMemory = VK_NULL_HANDLE;
    void* lightingUboMapped = nullptr;
    if (bakeOk &&
        createBuffer(ctx_, sizeof(LightingUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     lightingUbo, lightingUboMemory) == VK_SUCCESS) {
        vmaMapMemory(ctx_.allocator, lightingUboMemory, &lightingUboMapped);
    } else {
        bakeOk = false;
    }

    VkDescriptorSet lightingSet = VK_NULL_HANDLE;
    if (bakeOk) {
        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = pool;
        alloc.descriptorSetCount = 1;
        const VkDescriptorSetLayout layout = deferred_.lightingSetLayout();
        alloc.pSetLayouts = &layout;
        bakeOk = vkAllocateDescriptorSets(ctx_.device, &alloc, &lightingSet) == VK_SUCCESS;
    }
    if (bakeOk) {
        deferred_.writeLightingSet(ctx_, lightingSet, lightingUbo, albedo.view, normal.view,
                                   material.view, emissive.view, depth.view, ao.view,
                                   shadowsActive_ ? shadow_.arrayView : VK_NULL_HANDLE,
                                   spotAtlasActive_ ? spotAtlas_.view : VK_NULL_HANDLE,
                                   cluster.lightsBuffer[0], cluster.gridBuffer[0]);
    }

    // Readback staging for one face (RGBA16F 128^2).
    const VkDeviceSize faceBytes = static_cast<VkDeviceSize>(S) * S * 8;
    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation stagingMemory = VK_NULL_HANDLE;
    void* stagingMapped = nullptr;
    if (bakeOk &&
        createBuffer(ctx_, faceBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging, stagingMemory) == VK_SUCCESS) {
        vmaMapMemory(ctx_.allocator, stagingMemory, &stagingMapped);
    } else {
        bakeOk = false;
    }

    std::vector<uint16_t> all; // probe-major, face-major RGBA16F
    if (bakeOk)
        all.resize(static_cast<size_t>(defs.size()) * 6 * S * S * 4);

    auto transition = [&](VkCommandBuffer cmd, ImageResource& rt, VkImageLayout target,
                          VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                          VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
                          VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
        imageBarrier(cmd, rt.image, rt.layout, target, srcStage, srcAccess, dstStage, dstAccess,
                     aspect);
        rt.layout = target;
    };

    for (uint32_t pi = 0; bakeOk && pi < defs.size(); ++pi) {
        for (uint32_t face = 0; face < 6; ++face) {
            Camera cam;
            cam.position = defs[pi].position;
            cam.forward = kFaceDir[face];
            cam.up = kFaceUp[face];
            cam.fovY = 1.5707963f; // 90 deg, square face
            cam.nearPlane = camera_.nearPlane;
            cam.farPlane = camera_.farPlane;
            const Mat4 view = cam.view();
            const Mat4 proj = cam.proj(1.f);
            const Mat4 viewProj = Mat4::multiply(proj, view);

            scene_.advanceToFrame(0);
            scene_.updateLodSelection(cam.position, cam.fovY, lodEnabledByDefault());
            SceneUBO subo;
            deferred_.fillSceneUBO(subo, scene_, cam, view, proj, proj, Mat4::identity(), S, S,
                                   0.f, 0.f, false);
            std::memcpy(frames_[0].uboMapped, &subo, sizeof(subo));
            LightingUBO lubo;
            deferred_.fillLightingUBO(lubo, scene_, cam, viewProj, Mat4::inverse(viewProj),
                                      nullptr, nullptr, iblIntensity_);
            std::memcpy(lightingUboMapped, &lubo, sizeof(lubo));
            deferred_.fillClusterLights(cluster.lightsMapped[0],
                                        DeferredCore::effectiveLights(scene_, nullptr));

            submitOneShot(ctx_, [&](VkCommandBuffer cmd) {
                transition(cmd, albedo, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           sync::kSampleStages, sync::kSampled, sync::kColorAttach,
                           sync::kColorWrite);
                transition(cmd, normal, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           sync::kSampleStages, sync::kSampled, sync::kColorAttach,
                           sync::kColorWrite);
                transition(cmd, material, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           sync::kSampleStages, sync::kSampled, sync::kColorAttach,
                           sync::kColorWrite);
                transition(cmd, emissive, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           sync::kSampleStages, sync::kSampled, sync::kColorAttach,
                           sync::kColorWrite);
                transition(cmd, depth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                           sync::kSampleStages, sync::kSampled, sync::kDepthTests,
                           sync::kDepthReadWrite, VK_IMAGE_ASPECT_DEPTH_BIT);
                transition(cmd, motion, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           sync::kSampleStages, sync::kSampled, sync::kColorAttach,
                           sync::kColorWrite);
                {
                    // The bake shares the indirect GBuffer path (no cull pass;
                    // every candidate visible).  Slot 0 staging is safe here:
                    // each face renders in its own one-shot submit.
                    const uint32_t candidates = deferred_.buildInstanceList(
                        scene_, viewProj, instances_.capacity, cullInstCpu_.data(),
                        cullCmdCpu_.data(), cullRuns_);
                    if (candidates > 0) {
                        std::memcpy(instances_.stagingMapped[0], cullInstCpu_.data(),
                                    static_cast<size_t>(candidates) * sizeof(GpuInstance));
                        std::memcpy(gtCull_.cmdStagingMapped[0], cullCmdCpu_.data(),
                                    static_cast<size_t>(candidates) *
                                        sizeof(VkDrawIndexedIndirectCommand));
                    }
                    deferred_.recordInstanceUpload(cmd, 0, instances_, candidates);
                    deferred_.recordCommandUpload(cmd, 0, gtCull_, candidates, /*culled=*/false);
                    VkRenderingAttachmentInfo colors[5] = {
                        makeColorAttachment(albedo.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                            VK_ATTACHMENT_LOAD_OP_CLEAR),
                        makeColorAttachment(normal.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                            VK_ATTACHMENT_LOAD_OP_CLEAR, 0.f, 0.f, 1.f),
                        makeColorAttachment(material.view,
                                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                            VK_ATTACHMENT_LOAD_OP_CLEAR),
                        makeColorAttachment(emissive.view,
                                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                            VK_ATTACHMENT_LOAD_OP_CLEAR),
                        makeColorAttachment(motion.view,
                                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                            VK_ATTACHMENT_LOAD_OP_CLEAR)};
                    VkRenderingAttachmentInfo depthAtt = makeDepthAttachment(
                        depth.view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        VK_ATTACHMENT_LOAD_OP_CLEAR);
                    beginRendering(cmd, S, S, 5, colors, &depthAtt);
                    // GT pipeline variant (five attachments incl. motion; the
                    // bake discards motion — prevViewProj is identity anyway).
                    deferred_.recordGBufferDraws(cmd, scene_, /*gtPass=*/true,
                                                 frames_[0].sceneSet, textureSet_, materialStride_,
                                                 S, S, viewProj, gtCull_, cullRuns_.data(),
                                                 static_cast<uint32_t>(cullRuns_.size()));
                    vkCmdEndRendering(cmd);
                }
                transition(cmd, motion, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled);
                transition(cmd, albedo, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled);
                transition(cmd, normal, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled);
                transition(cmd, material, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled);
                transition(cmd, emissive, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled);
                transition(cmd, depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           sync::kDepthTests, sync::kDepthWrite, sync::kFragment, sync::kSampled,
                           VK_IMAGE_ASPECT_DEPTH_BIT);
                transition(cmd, hdr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kCopy,
                           sync::kTransferRead, sync::kColorAttach, sync::kColorWrite);
                deferred_.recordLightingPass(cmd, lightingSet, cluster, 0, view, proj, hdr.view,
                                             S, S);
                transition(cmd, hdr, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sync::kColorAttach,
                           sync::kColorWrite, sync::kCopy, sync::kTransferRead);
                VkBufferImageCopy region = {};
                region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.imageExtent = {S, S, 1};
                vkCmdCopyImageToBuffer(cmd, hdr.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       staging, 1, &region);
            });

            // Readback + horizontal flip (raster face -> Vulkan cube face).
            const uint16_t* srcPx = static_cast<const uint16_t*>(stagingMapped);
            uint16_t* dstPx =
                all.data() + (static_cast<size_t>(pi) * 6 + face) * S * S * 4;
            for (uint32_t y = 0; y < S; ++y)
                for (uint32_t x = 0; x < S; ++x)
                    for (uint32_t c = 0; c < 4; ++c)
                        dstPx[(static_cast<size_t>(y) * S + x) * 4 + c] =
                            srcPx[(static_cast<size_t>(y) * S + (S - 1 - x)) * 4 + c];
        }
        std::fprintf(stderr, "bake-probes: probe %u done\n", pi);
    }

    if (bakeOk && !saveProbeFile(outPath.c_str(), defs, S, all)) {
        std::fprintf(stderr, "bake-probes: failed to write %s\n", outPath.c_str());
        bakeOk = false;
    }

    // --- cleanup ---------------------------------------------------------------
    if (staging) {
        if (stagingMapped) vmaUnmapMemory(ctx_.allocator, stagingMemory);
        vmaDestroyBuffer(ctx_.allocator, staging, stagingMemory);
    }
    deferred_.destroyClusterGrid(ctx_, cluster);
    if (lightingUbo) {
        if (lightingUboMapped) vmaUnmapMemory(ctx_.allocator, lightingUboMemory);
        vmaDestroyBuffer(ctx_.allocator, lightingUbo, lightingUboMemory);
    }
    if (pool) vkDestroyDescriptorPool(ctx_.device, pool, nullptr);
    albedo.destroy(ctx_);
    normal.destroy(ctx_);
    material.destroy(ctx_);
    emissive.destroy(ctx_);
    motion.destroy(ctx_);
    depth.destroy(ctx_);
    hdr.destroy(ctx_);
    ao.destroy(ctx_);
    return bakeOk;
}

void Renderer::recordFrame(uint32_t frameIndex, uint32_t swapchainIndex) {
    const uint32_t slot = frameIndex % kFramesInFlight;
    FrameResources& fr = frames_[slot];
    VkCommandBuffer cmd = fr.cmd;

    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    timestamps_.resetForFrame(cmd, slot);
    timestamps_.frameBegin(cmd, slot);

    const bool useUpscaler = opts_.upscalerName != "none" && upscaler_ != nullptr;
    const bool jitter = useUpscaler && upscalerNeedsJitter(upscaler_.get());
    const float aspect = static_cast<float>(opts_.displayWidth) / static_cast<float>(opts_.displayHeight);
    const Mat4 view = camera_.view();
    const Mat4 proj = camera_.proj(aspect);
    Mat4 projJittered = proj;
    prevJitterX_ = jitterX_;
    prevJitterY_ = jitterY_;
    if (jitter) {
        const Vec2 h = halton23(frameIndex + 1);
        // SR_NO_JITTER keeps the low-res path but zeroes the offsets (and the
        // jitter reported to the upscaler) to isolate jitter from resolution.
        jitterX_ = diagNoJitter_ ? 0.f : h.x - 0.5f;
        jitterY_ = diagNoJitter_ ? 0.f : h.y - 0.5f;
        // Sub-pixel jitter must shift NDC by a constant (every temporal
        // upscaler assumes: jittered pixel pos = unjittered pos + jitter),
        // i.e. clip.xy += offset * clip.w.  clip.w = -z_view comes from row 3
        // (m[11] = -1), so the offset lands in column 2 (m[8]/m[9]) with a
        // negative sign.  Writing the translation column (m[12]/m[13])
        // instead would add a constant clip-space term, making the pixel
        // shift depth-dependent (jitterX / viewDepth) and breaking the
        // uniform-jitter contract the upscalers are told about.
        projJittered.m[8] -= jitterX_ * 2.f / static_cast<float>(renderWidth_);
        projJittered.m[9] -= jitterY_ * 2.f / static_cast<float>(renderHeight_);
    } else {
        jitterX_ = 0.f;
        jitterY_ = 0.f;
    }

    updateSceneUBO(frameIndex, jitter, renderWidth_, renderHeight_, view, proj, projJittered, prevViewProj_);

    const bool gbuffer = useUpscaler;

    // Lighting reconstructs world positions with the inverse of the exact
    // view-projection used for this pass (jittered for the low-res path).
    const Mat4 viewProjUsed = Mat4::multiply(gbuffer ? projJittered : proj, view);

    // --- Phase 7a: candidate build + staged upload -----------------------------
    // CPU frustum cull + LOD state (path-independent) -> the slot's staging;
    // the GPU upload/cull/draw consume it below.  The active path's channel
    // gets this frame's commands; the idle path keeps its stale (valid)
    // pyramid + prevViewProj pair.
    CullChannel& cull = gbuffer ? gbCull_ : gtCull_;
    const Mat4 cullViewProj = Mat4::multiply(proj, view); // un-jittered (sub-pixel)
    cullCandidates_ =
        deferred_.buildInstanceList(scene_, cullViewProj, instances_.capacity,
                                    cullInstCpu_.data(), cullCmdCpu_.data(), cullRuns_);
    if (cullCandidates_ > 0) {
        std::memcpy(instances_.stagingMapped[slot], cullInstCpu_.data(),
                    static_cast<size_t>(cullCandidates_) * sizeof(GpuInstance));
        std::memcpy(cull.cmdStagingMapped[slot], cullCmdCpu_.data(),
                    static_cast<size_t>(cullCandidates_) * sizeof(VkDrawIndexedIndirectCommand));
    }

    // Shadows (Phase 4b): CSM sun cascades (first shadow-casting directional,
    // same selection rule as fillLightingUBO) plus the spot shadow atlas —
    // shadow-casting spots are scored by intensity/distance^2 and the top
    // kShadowAtlasTiles get a tile (selectSpotShadowLights writes shadowIndex
    // into a lights copy that overrides both the UBO and the cluster SSBO).
    // The GT path samples the same maps — GT is the same lighting at native res.
    ShadowFrame shadowFrame;
    const ShadowFrame* shadow = nullptr;
    std::vector<Light> shadowLights;
    const std::vector<Light>* lightsOverride = nullptr;
    if (shadowsActive_ && opts_.shadows) {
        const std::vector<Light>& lights =
            scene_.lights.empty() ? defaultLights() : scene_.lights;
        for (uint32_t i = 0; i < lights.size(); ++i) {
            if (lights[i].type == LightType::Directional && lights[i].castShadow) {
                DeferredCore::computeCascadeVPs(camera_, aspect, lights[i].positionOrDirection,
                                                shadowFrame.cascadeVp, shadowFrame.splitDepth);
                shadowFrame.lightIndex = static_cast<int32_t>(i);
                break;
            }
        }
        if (spotAtlasActive_) {
            shadowLights = lights;
            shadowFrame.atlasTileCount =
                DeferredCore::selectSpotShadowLights(shadowLights, camera_.position);
            for (const Light& l : shadowLights) {
                if (l.shadowIndex >= 0)
                    shadowFrame.atlasVp[l.shadowIndex] = DeferredCore::computeSpotShadowVp(l);
            }
            lightsOverride = &shadowLights;
        }
        shadowFrame.debugCascades = opts_.shadowDebug;
        shadowFrame.frameIndex = frameIndex;
        if (shadowFrame.lightIndex >= 0 || shadowFrame.atlasTileCount > 0)
            shadow = &shadowFrame;
    }
    const Mat4 invViewProjUsed = Mat4::inverse(viewProjUsed);
    updateLightingUBO(frameIndex, viewProjUsed, invViewProjUsed, shadow, lightsOverride);

    ImageResource& tgtAlbedo = gbuffer ? gbAlbedo_ : gtAlbedo_;
    ImageResource& tgtNormal = gbuffer ? gbNormal_ : gtNormal_;
    ImageResource& tgtMaterial = gbuffer ? gbMaterial_ : gtMaterial_;
    ImageResource& tgtEmissive = gbuffer ? gbEmissive_ : gtEmissive_;
    ImageResource& tgtDepth = gbuffer ? gbDepth_ : gtDepth_;
    ImageResource& tgtAoRaw = gbuffer ? gbAoRaw_ : gtAoRaw_;
    ImageResource& tgtAo = gbuffer ? gbAo_ : gtAo_;
    ImageResource& litTarget = gbuffer ? gbColor_ : finalImage_;
    const uint32_t sceneW = gbuffer ? renderWidth_ : opts_.displayWidth;
    const uint32_t sceneH = gbuffer ? renderHeight_ : opts_.displayHeight;
    // Per-path GTAO temporal state (ping-pong index, reprojection matrix,
    // first-frame reset).
    DepthPyramid& aoChain = gbuffer ? gbPyramidAo_ : gtPyramidAo_;
    AoHistory& aoHist = gbuffer ? gbAoHist_ : gtAoHist_;
    Mat4& prevAoVp = gbuffer ? prevAoViewProjGb_ : prevAoViewProjGt_;
    uint32_t& aoFrames = gbuffer ? aoFramesGb_ : aoFramesGt_;

    ImageResource& tgtMotion = gbuffer ? gbMotion_ : gtMotion_; // Phase 6b: both paths write motion

    // --- Frame graph -------------------------------------------------------------
    // The frame records as a linear pass list: each pass declares its image
    // accesses and the graph derives the sync2 barriers from its cross-frame
    // state tracker (replacing the old hand-written transition() calls and the
    // per-resource layout members).  GENERAL-for-life resources (Hi-Z/color
    // pyramids, temporal histories, fog volumes, bloom/post-fx working
    // targets) keep their DeferredCore-internal barriers and stay untracked.
    rg_.clear();

    // Phase 7a uploads + the occlusion cull pass are scene work: keep them
    // inside the scene timestamp range.  The cull reprojects candidates
    // against LAST frame's pyramid of this path (1-frame latency; skipped on
    // the path's first frame or after a resize -> everything visible).
    const bool cullActive = occlusion_ && cull.prevValid && cullCandidates_ > 0;

    // --- GBuffer pass ---------------------------------------------------------
    // Both paths attach a motion RT (Phase 6b): the GT pass uses the GT
    // pipeline (same five formats) so GT motion blur matches the LR path.
    {
        RenderGraph::Pass& p = rg_.addPass("GBuffer");
        p.access(tgtAlbedo.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kColorAttach,
                 sync::kColorWrite);
        p.access(tgtNormal.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kColorAttach,
                 sync::kColorWrite);
        p.access(tgtMaterial.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kColorAttach,
                 sync::kColorWrite);
        p.access(tgtEmissive.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kColorAttach,
                 sync::kColorWrite);
        p.access(tgtMotion.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kColorAttach,
                 sync::kColorWrite);
        p.access(tgtDepth.image, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                 sync::kDepthTests, sync::kDepthReadWrite, VK_IMAGE_ASPECT_DEPTH_BIT);
        p.record = [&](VkCommandBuffer c) {
            timestamps_.sceneBegin(c, slot);
            deferred_.recordInstanceUpload(c, slot, instances_, cullCandidates_);
            deferred_.recordCommandUpload(c, slot, cull, cullCandidates_, cullActive);
            if (cullActive)
                deferred_.recordOcclusionCull(c, cull, cullCandidates_, cull.prevViewProj,
                                              (gbuffer ? gbPyramid_ : gtPyramid_).mipCount, sceneW,
                                              sceneH);
            VkRenderingAttachmentInfo colors[5] = {
                makeColorAttachment(tgtAlbedo.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(tgtNormal.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR, 0.f, 0.f, 1.f),
                makeColorAttachment(tgtMaterial.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(tgtEmissive.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(tgtMotion.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR)};
            const uint32_t colorCount = 5;
            VkRenderingAttachmentInfo depth =
                makeDepthAttachment(tgtDepth.view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR);
            beginRendering(c, sceneW, sceneH, colorCount, colors, &depth);
            deferred_.recordGBufferDraws(c, scene_, !gbuffer, fr.sceneSet, textureSet_,
                                         materialStride_, sceneW, sceneH, cullViewProj, cull,
                                         cullRuns_.data(), static_cast<uint32_t>(cullRuns_.size()));
            vkCmdEndRendering(c);
        };
    }

    // --- Shadow pass (sun CSM, one 2048^2 layer per cascade) -----------------
    // Runs once per frame and feeds both the LR and the GT lighting paths.
    if (shadow && shadow->lightIndex >= 0) {
        RenderGraph::Pass& p = rg_.addPass("ShadowCSM");
        p.access(shadow_.image, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                 sync::kDepthTests, sync::kDepthReadWrite, VK_IMAGE_ASPECT_DEPTH_BIT, 0,
                 kShadowCascadeCount);
        p.record = [&](VkCommandBuffer c) {
            deferred_.recordShadowPass(c, shadow_, scene_, shadow->cascadeVp, fr.sceneSet,
                                       textureSet_, materialStride_);
        };
        // Depth writes -> shadow-comparison samples in the lighting shaders
        // (barrier-only pass; the lighting pass re-declares the read).
        rg_.addPass("ShadowCSMResolve")
            .access(shadow_.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kFragment,
                    sync::kSampled, VK_IMAGE_ASPECT_DEPTH_BIT, 0, kShadowCascadeCount);
    }

    // --- Spot shadow atlas (Phase 4b, one 1024^2 tile per selected spot) -----
    // Same once-per-frame sharing between the LR and GT paths as the CSM pass.
    if (shadow && shadow->atlasTileCount > 0) {
        RenderGraph::Pass& p = rg_.addPass("SpotAtlas");
        p.access(spotAtlas_.image, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                 sync::kDepthTests, sync::kDepthReadWrite, VK_IMAGE_ASPECT_DEPTH_BIT);
        p.record = [&](VkCommandBuffer c) {
            deferred_.recordSpotShadowPass(c, spotAtlas_, scene_, shadow->atlasVp,
                                           shadow->atlasTileCount, fr.sceneSet, textureSet_,
                                           materialStride_);
        };
        rg_.addPass("SpotAtlasResolve")
            .access(spotAtlas_.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kFragment,
                    sync::kSampled, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    // --- Hi-Z + GTAO (between the GBuffer and the lighting pass) ---------------
    // Hi-Z pyramid for the SSR marchers (glass in the transparency pass and
    // the opaque-SSR compute pass) and next frame's occlusion cull (Phase 7a).
    // Skipped only when no consumer runs; the pyramid then just keeps its last
    // contents.
    if (hasTransparency_ || opts_.ssr || occlusion_) {
        RenderGraph::Pass& p = rg_.addPass("HiZ");
        p.access(tgtDepth.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kCompute,
                 sync::kSampled, VK_IMAGE_ASPECT_DEPTH_BIT);
        p.record = [&](VkCommandBuffer c) {
            deferred_.recordDepthPyramidPass(c, gbuffer ? gbPyramid_ : gtPyramid_);
        };
    }

    // --- GTAO (view-Z depth chain -> main pass -> temporal EMA -> denoise) ------
    // The AO depth chain is rebuilt every frame (the main pass samples it at
    // per-step LODs).  AoRaw ping-pongs between ssao main (storage write) and
    // the temporal pass (sampled); the accumulated history feeds the blur.
    {
        RenderGraph::Pass& p = rg_.addPass("AoChain");
        p.access(tgtDepth.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kCompute,
                 sync::kSampled, VK_IMAGE_ASPECT_DEPTH_BIT);
        p.record = [&](VkCommandBuffer c) { deferred_.recordDepthPyramidPass(c, aoChain); };
    }
    // XeGTAO's working depth has 5 levels: cap the step LOD at 4.
    const float aoMaxLod = static_cast<float>(std::min(aoChain.mipCount - 1, 4u));
    {
        RenderGraph::Pass& p = rg_.addPass("GTAO");
        p.access(tgtNormal.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kCompute,
                 sync::kSampled);
        p.access(tgtAoRaw.image, VK_IMAGE_LAYOUT_GENERAL, sync::kCompute, sync::kStorageWrite);
        p.record = [&](VkCommandBuffer c) {
            deferred_.recordSsaoPass(c, gbuffer ? ssaoSetGb_ : ssaoSetGt_, invViewProjUsed,
                                     frameIndex, camera_.nearPlane, camera_.farPlane, aoMaxLod,
                                     sceneW, sceneH);
        };
    }
    const uint32_t aoWrite = aoFrames & 1u;
    {
        RenderGraph::Pass& p = rg_.addPass("GTAOTemporal");
        p.access(tgtAoRaw.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kCompute,
                 sync::kSampled);
        p.access(tgtDepth.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kCompute,
                 sync::kSampled, VK_IMAGE_ASPECT_DEPTH_BIT);
        // Value captures: the lambda runs at execute() time, after the
        // bookkeeping below already advanced the temporal state.
        const Mat4 aoPrevVp = prevAoVp;
        const bool aoReset = aoFrames == 0;
        p.record = [&, aoPrevVp, aoReset](VkCommandBuffer c) {
            deferred_.recordSsaoTemporalPass(c, aoHist, aoWrite, invViewProjUsed, aoPrevVp, sceneW,
                                             sceneH, aoReset);
        };
    }
    prevAoVp = viewProjUsed;
    ++aoFrames;
    // Ao: sampled by the lighting fragment + opaque-SSR compute shaders,
    // rewritten by the blur pass.
    {
        RenderGraph::Pass& p = rg_.addPass("GTAOBlur");
        p.access(tgtAo.image, VK_IMAGE_LAYOUT_GENERAL, sync::kCompute, sync::kStorageWrite);
        p.record = [&](VkCommandBuffer c) {
            deferred_.recordSsaoBlurPass(c, aoHist.blurSet[aoWrite], sceneW, sceneH);
        };
    }

    // --- Lighting pass (deferred PBR + IBL, skybox on far-plane pixels) --------
    // The GBuffer reads are declared here (fragment scope); the opaque-SSR
    // compute pass re-declares them with compute scope later — read-after-read
    // in one layout needs no barrier, so the graph emits nothing extra.
    {
        RenderGraph::Pass& p = rg_.addPass("Lighting");
        p.access(tgtAlbedo.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kFragment,
                 sync::kSampled);
        p.access(tgtNormal.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kFragment,
                 sync::kSampled);
        p.access(tgtMaterial.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kFragment,
                 sync::kSampled);
        p.access(tgtEmissive.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kFragment,
                 sync::kSampled);
        p.access(tgtDepth.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kFragment,
                 sync::kSampled, VK_IMAGE_ASPECT_DEPTH_BIT);
        p.access(tgtAo.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kFragment,
                 sync::kSampled);
        if (shadow && shadow->lightIndex >= 0)
            p.access(shadow_.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kFragment,
                     sync::kSampled, VK_IMAGE_ASPECT_DEPTH_BIT, 0, kShadowCascadeCount);
        if (shadow && shadow->atlasTileCount > 0)
            p.access(spotAtlas_.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kFragment,
                     sync::kSampled, VK_IMAGE_ASPECT_DEPTH_BIT);
        p.access(litTarget.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kColorAttach,
                 sync::kColorWrite);
        p.record = [&](VkCommandBuffer c) {
            deferred_.recordLightingPass(c, gbuffer ? fr.lightingSetGb : fr.lightingSetGt,
                                         gbuffer ? gbCluster_ : gtCluster_, slot, view,
                                         gbuffer ? projJittered : proj, litTarget.view, sceneW,
                                         sceneH);
        };
    }

    // --- Froxel volumetric fog (Phase 5a): inject -> light -> temporal ->
    //     march, then composite over the lit HDR target (before tonemap) -----
    // The cluster assignment for this frame ran inside recordLightingPass and
    // the CSM/spot maps are already shader-readable.  The fog passes use the
    // UN-JITTERED camera matrices for both paths so the volume does not swim
    // under TAA jitter; the GT path is the same algorithm and parameters at
    // native resolution.
    if (volFogActive_) {
        // NOTE: record lambdas run at rg_.execute() time, AFTER this block has
        // closed — anything block-scope they use must be captured by value or
        // rebound from members inside the lambda (references would dangle).
        Mat4& prevFogVp = gbuffer ? prevFogViewProjGb_ : prevFogViewProjGt_;
        uint32_t& fogFrames = gbuffer ? fogFramesGb_ : fogFramesGt_;
        const Mat4 fogViewProj = Mat4::multiply(proj, view); // un-jittered
        const Mat4 fogPrevVp = prevFogVp;
        const uint32_t fogWrite = fogFrames & 1u;
        const bool fogReset = fogFrames == 0;
        rg_.addPass("VolFogAccumulate").record = [&, fogPrevVp, fogWrite, fogReset](VkCommandBuffer c) {
            VolFogVolume& fog = gbuffer ? gbFog_ : gtFog_;
            ClusterGrid& fogCluster = gbuffer ? gbCluster_ : gtCluster_;
            deferred_.recordVolFogAccumulate(c, fog, fogCluster, slot, view, proj, fogPrevVp,
                                             fogParams_, frameIndex, fogWrite, fogReset);
        };
        prevFogVp = fogViewProj;
        ++fogFrames;
        RenderGraph::Pass& p = rg_.addPass("VolFogComposite");
        p.access(litTarget.image, VK_IMAGE_LAYOUT_GENERAL, sync::kCompute, sync::kStorageReadWrite);
        p.access(tgtDepth.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kCompute,
                 sync::kSampled, VK_IMAGE_ASPECT_DEPTH_BIT);
        p.record = [&](VkCommandBuffer c) {
            const VolFogVolume& fog = gbuffer ? gbFog_ : gtFog_;
            deferred_.recordVolFogComposite(c, fog, proj, fogParams_.maxDistance, sceneW, sceneH);
        };
    }

    // --- Opaque SSR + transparency pass (over the lit scene) ------------------
    // Build the color mip chain from the lit target: mip 0 IS the opaque HDR
    // copy glass SSR used to make with a transfer; the chain is GENERAL for
    // life (untracked — the pass barriers internally).  The opaque-SSR compute
    // pass consumes the same chain, so it is built whenever either consumer
    // runs.
    if (hasTransparency_ || opts_.ssr) {
        RenderGraph::Pass& cp = rg_.addPass("ColorPyramid");
        cp.access(litTarget.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kCompute,
                  sync::kSampled);
        cp.record = [&](VkCommandBuffer c) {
            deferred_.recordColorPyramidPass(c, gbuffer ? gbColorPyramid_ : gtColorPyramid_);
        };
        if (opts_.ssr) {
            // Opaque SSR, Phase 2d: trace into the path's full-res RT (rgb =
            // composite delta, a = view |z|), then the temporal pass EMA-
            // accumulates the delta and fuses the composite (in-place RMW add
            // on the lit target) — first frame equals the old in-place pass.
            ImageResource& ssrTrace = gbuffer ? gbSsrTrace_ : gtSsrTrace_;
            Mat4& prevSsrVp = gbuffer ? prevSsrViewProjGb_ : prevSsrViewProjGt_;
            uint32_t& ssrFrames = gbuffer ? ssrFramesGb_ : ssrFramesGt_;
            {
                RenderGraph::Pass& p = rg_.addPass("SsrTrace");
                p.access(tgtAlbedo.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         sync::kCompute, sync::kSampled);
                p.access(tgtNormal.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         sync::kCompute, sync::kSampled);
                p.access(tgtMaterial.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         sync::kCompute, sync::kSampled);
                p.access(tgtDepth.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         sync::kCompute, sync::kSampled, VK_IMAGE_ASPECT_DEPTH_BIT);
                p.access(tgtAo.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kCompute,
                         sync::kSampled);
                p.access(ssrTrace.image, VK_IMAGE_LAYOUT_GENERAL, sync::kCompute,
                         sync::kStorageWrite);
                p.record = [&](VkCommandBuffer c) {
                    deferred_.recordSsrPass(c, gbuffer ? fr.ssrSetGb : fr.ssrSetGt, viewProjUsed,
                                            sceneW, sceneH);
                };
            }
            // Value captures: the lambda runs at execute() time, after the
            // bookkeeping below already advanced the ping-pong state.
            const uint32_t ssrWrite = ssrFrames & 1u;
            const Mat4 ssrPrevVp = prevSsrVp;
            const bool ssrReset = ssrFrames == 0;
            {
                RenderGraph::Pass& p = rg_.addPass("SsrTemporal");
                p.access(ssrTrace.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         sync::kCompute, sync::kSampled);
                p.access(tgtDepth.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         sync::kCompute, sync::kSampled, VK_IMAGE_ASPECT_DEPTH_BIT);
                p.access(litTarget.image, VK_IMAGE_LAYOUT_GENERAL, sync::kCompute,
                         sync::kStorageReadWrite);
                p.record = [&, ssrWrite, ssrPrevVp, ssrReset](VkCommandBuffer c) {
                    SsrHistory& ssrHist = gbuffer ? gbSsrHist_ : gtSsrHist_;
                    deferred_.recordSsrTemporalPass(c, ssrHist, ssrWrite, invViewProjUsed,
                                                    ssrPrevVp, sceneW, sceneH, ssrReset);
                };
            }
            prevSsrVp = viewProjUsed;
            ++ssrFrames;
        }
    }
    if (hasTransparency_) {
        RenderGraph::Pass& p = rg_.addPass("Transparency");
        // LOAD_OP_LOAD + alpha blending: the lit target is read AND written.
        p.access(litTarget.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kColorAttach,
                 sync::kColorReadWrite);
        if (gbuffer) {
            p.access(gbMotion_.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     sync::kColorAttach, sync::kColorReadWrite);
            p.access(gbReactive_.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     sync::kColorAttach, sync::kColorWrite);
        }
        p.access(tgtDepth.image, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                 sync::kDepthTests, sync::kDepthRead, VK_IMAGE_ASPECT_DEPTH_BIT);
        p.record = [&](VkCommandBuffer c) {
            VkRenderingAttachmentInfo tColors[3] = {};
            tColors[0] =
                makeColorAttachment(litTarget.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_LOAD);
            if (gbuffer) {
                tColors[1] =
                    makeColorAttachment(gbMotion_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                        VK_ATTACHMENT_LOAD_OP_LOAD);
                tColors[2] =
                    makeColorAttachment(gbReactive_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                        VK_ATTACHMENT_LOAD_OP_CLEAR);
            }
            VkRenderingAttachmentInfo tDepth =
                makeDepthAttachment(tgtDepth.view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_LOAD);
            beginRendering(c, sceneW, sceneH, gbuffer ? 3 : 1, tColors, &tDepth);
            deferred_.recordTransparentDraws(c, scene_, !gbuffer, fr.sceneSet, textureSet_,
                                             gbuffer ? fr.transparentSetGb : fr.transparentSetGt,
                                             materialStride_, sceneW, sceneH,
                                             Mat4::multiply(proj, view), camera_.position,
                                             proj.m[14] / proj.m[10], fogParams_.maxDistance,
                                             volFogActive_);
            vkCmdEndRendering(c);
        };
    }
    // Coverage mask conditioning (LR only): 3x3 max dilate + motion gate.
    // The dilated plateau absorbs the sub-pixel straddle of consumers
    // sampling at the jittered coordinate; the motion gate keeps the mask
    // from dropping history on static pixels (see reactive_dilate.comp).
    if (hasTransparency_ && gbuffer) {
        RenderGraph::Pass& p = rg_.addPass("ReactiveDilate");
        p.access(gbReactive_.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kCompute,
                 sync::kSampled);
        p.access(gbMotion_.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kCompute,
                 sync::kSampled);
        p.access(gbReactiveDilated_.image, VK_IMAGE_LAYOUT_GENERAL, sync::kCompute,
                 sync::kStorageWrite);
        p.record = [&](VkCommandBuffer c) {
            deferred_.recordReactiveDilatePass(c, reactiveDilateSet_, sceneW, sceneH);
        };
    }

    if (opts_.bloom) {
        // The helper transitions the lit target internally (GENERAL RMW
        // composite) and hands it back SHADER_READ_ONLY: entry via the
        // declaration, the in/out layout through the tracker's layoutRef, the
        // exit stage/access via leaves().
        RenderGraph::Pass& p = rg_.addPass("Bloom");
        p.access(litTarget.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kCompute,
                 sync::kSampled);
        p.leaves(litTarget.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 sync::kFragment | sync::kCompute, sync::kSampled);
        p.record = [&](VkCommandBuffer c) {
            const BloomPyramid& bloom = gbuffer ? gbBloom_ : gtBloom_;
            deferred_.recordBloomPyramidPass(c, bloom, litTarget.image,
                                             rg_.layoutRef(litTarget.image), sceneW, sceneH);
        };
    }
    // --- Motion blur + depth of field (Phase 6b) -------------------------------
    // HDR domain, after bloom and before upscale/present.  Same algorithm and
    // parameters on the LR and GT paths (the GT blurs identically, so compare
    // metrics stay fair); the blur-radius clamps scale with path height so the
    // display-space blur is resolution-independent.
    if (opts_.motionBlur || opts_.dof) {
        PostFxParams fx;
        fx.depthM10 = proj.m[10];
        fx.depthM14 = proj.m[14];
        fx.farPlane = camera_.farPlane;
        const float resScale = static_cast<float>(sceneH) / 1080.f;
        fx.maxBlurPx = std::max(8.f, kMotionBlurMaxPixels * resScale);
        fx.maxCocPx = std::max(2.f, opts_.dofMaxBlurPx * resScale);
        fx.aperture = kDofAperture * (kDofDefaultFstop / opts_.dofFstop);
        fx.focusDistance = opts_.dofFocus;
        fx.motionBlur = opts_.motionBlur;
        fx.dof = opts_.dof;
        RenderGraph::Pass& p = rg_.addPass("PostFx");
        // Same internal-transition convention as the bloom pass above.
        p.access(litTarget.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 sync::kFragment | sync::kCompute, sync::kSampled);
        p.access(tgtMotion.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kCompute,
                 sync::kSampled);
        p.access(tgtDepth.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kCompute,
                 sync::kSampled, VK_IMAGE_ASPECT_DEPTH_BIT);
        p.leaves(litTarget.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 sync::kFragment | sync::kCompute, sync::kSampled);
        p.record = [&, fx](VkCommandBuffer c) { // fx by value: runs at execute() time
            deferred_.recordPostFxPass(c, gbuffer ? gbPostFx_ : gtPostFx_, litTarget.image,
                                       rg_.layoutRef(litTarget.image), fx, frameIndex);
        };
    }
    // --- Auto exposure (lit HDR -> histogram -> smoothed EV) -------------------
    // Runs on this path's HDR source (gbColor_ for LR, finalImage_ for GT),
    // which is SHADER_READ_ONLY here with compute in the sampled-dst scope.
    // The solved state is copied to the slot's staging buffer and harvested by
    // the run loop once the slot's fence signals — so the exposure APPLIED to
    // this frame (present push, upscaler preExposure, CPU screenshot) is the
    // value solved kFramesInFlight frames ago.  This is the engine convention
    // (earlier frames' luminance drives the current exposure) and keeps the
    // applied value CPU-known, which the deterministic CPU screenshot path
    // requires.
    if (opts_.autoExposure) {
        ExposureSolvePush solve;
        solve.minEV = opts_.exposureMinEV;
        solve.maxEV = opts_.exposureMaxEV;
        solve.resetState = (frameIndex == 0) ? 1.f : 0.f; // snap on the first frame
        RenderGraph::Pass& p = rg_.addPass("AutoExposure");
        p.access(litTarget.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kCompute,
                 sync::kSampled);
        p.record = [&, solve](VkCommandBuffer c) { // solve by value: runs at execute() time
            deferred_.recordAutoExposurePass(c, exposureChannel_.gpu, solve,
                                             exposureChannel_.staging[slot]);
        };
        exposureChannel_.pending[slot] = true;
    }
    rg_.addPass("SceneEnd").record = [&](VkCommandBuffer c) { timestamps_.sceneEnd(c, slot); };

    // Phase 7a: from the next frame on, this path's cull pass reprojects
    // against the pyramid just built with this frame's exact view-projection.
    cull.prevViewProj = viewProjUsed;
    cull.prevValid = true;

    if (!gbuffer) {
        // GT has no upscaler pass; emit a zero-width range so every query slot
        // is written (readback with WAIT_BIT otherwise blocks forever).
        rg_.addPass("Upscale").record = [&](VkCommandBuffer c) {
            timestamps_.upscaleBegin(c, slot);
            timestamps_.upscaleEnd(c, slot);
        };
    }

    if (gbuffer) {
        // gbColor_/gbMotion_/gbDepth_ are already SHADER_READ_ONLY (lighting
        // wrote gbColor_ and sampled depth; the graph tracks the states).
        // finalImage_ was sampled by the present pass; the upscaler writes it
        // as a storage image (fragment or compute).
        RenderGraph::Pass& p = rg_.addPass("Upscale");
        p.access(finalImage_.image, VK_IMAGE_LAYOUT_GENERAL, sync::kSampleStages,
                 sync::kStorageWrite);
        p.access(gbColor_.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kSampleStages,
                 sync::kSampled);
        p.access(gbDepth_.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kSampleStages,
                 sync::kSampled, VK_IMAGE_ASPECT_DEPTH_BIT);
        p.access(gbMotion_.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kSampleStages,
                 sync::kSampled);
        if (hasTransparency_)
            p.access(gbReactiveDilated_.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     sync::kSampleStages, sync::kSampled);
        p.record = [&](VkCommandBuffer c) {
            UpscalerResources res;
            res.color = gbColor_.image;
            res.colorView = gbColor_.view;
            res.depth = gbDepth_.image;
            res.depthView = gbDepth_.view;
            res.motion = gbMotion_.image;
            res.motionView = gbMotion_.view;
            if (hasTransparency_) {
                res.reactive = gbReactiveDilated_.image;
                res.reactiveView = gbReactiveDilated_.view;
            }
            res.output = finalImage_.image;
            res.outputView = finalImage_.view;

            CameraParams cam;
            std::memcpy(cam.view, view.m, sizeof(cam.view));
            std::memcpy(cam.proj, proj.m, sizeof(cam.proj));
            std::memcpy(cam.prevViewProj, prevViewProj_.m, sizeof(cam.prevViewProj));
            cam.jitterX = jitterX_;
            cam.jitterY = jitterY_;
            cam.prevJitterX = prevJitterX_;
            cam.prevJitterY = prevJitterY_;
            cam.cameraNear = camera_.nearPlane;
            cam.cameraFar = camera_.farPlane;
            cam.fovY = camera_.fovY;

            FrameParams frame;
            frame.frameIndex = static_cast<int>(frameIndex);
            frame.deltaTime = deltaTime_;
            // Real exposure input (was hardcoded 1): the display exposure applied
            // to this frame — see InputAdapter.h for the preExposure convention.
            frame.preExposure = displayExposure();
            frame.resetHistory = (frameIndex == 0);

            timestamps_.upscaleBegin(c, slot);
            upscaler_->dispatch(c, res, cam, frame);
            timestamps_.upscaleEnd(c, slot);
        };
        rg_.addPass("PostUpscale")
            .access(finalImage_.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kFragment,
                    sync::kSampled);
    }
    // GT path: finalImage_ is already SHADER_READ_ONLY (post-lighting).

    // Acquired swapchain contents are undefined; the graph starts the image
    // from UNDEFINED (no source synchronization) like the old code did.
    const VkImage swapImage = swapchain_.image(swapchainIndex);
    const VkImageView swapView = swapchain_.view(swapchainIndex);
    rg_.discard(swapImage);
    {
        RenderGraph::Pass& p = rg_.addPass("Present");
        p.access(finalImage_.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kFragment,
                 sync::kSampled);
        p.access(swapImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kColorAttach,
                 sync::kColorWrite);
        p.record = [&](VkCommandBuffer c) {
            VkRenderingAttachmentInfo presentColor =
                makeColorAttachment(swapView, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ATTACHMENT_LOAD_OP_CLEAR);
            beginRendering(c, swapchain_.extent().width, swapchain_.extent().height, 1, &presentColor, nullptr);
            vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, presentPipeline_);
            VkViewport pvp = {0.f, 0.f, static_cast<float>(swapchain_.extent().width),
                              static_cast<float>(swapchain_.extent().height), 0.f, 1.f};
            vkCmdSetViewport(c, 0, 1, &pvp);
            VkRect2D psc = {{0, 0}, {swapchain_.extent().width, swapchain_.extent().height}};
            vkCmdSetScissor(c, 0, 1, &psc);
            vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, presentPipelineLayout_, 0, 1,
                                    &presentSet_, 0, nullptr);
            // Present: exposure + the terminal lens-effects chain (strengths zeroed
            // when disabled; dirt also needs the bloom pyramid, so --no-bloom forces
            // it off — the bound mip content is undefined then, see the present set)
            // + the log-domain grading set (Phase 6c) + the HDR output mode from the
            // swapchain (Phase 6c; 0 = SDR).
            const bool fx = opts_.lensFx;
            const Vec3 wb = whiteBalanceForTemperatureTint(grading_.temperatureK, grading_.tint);
            const float presentPush[20] = {
                displayExposure(), 0.f, 0.f, 0.f,
                fx ? kLensCaStrength : 0.f,
                fx ? kLensVignetteStrength : 0.f,
                fx ? kLensGrainStrength : 0.f,
                static_cast<float>(frameIndex),
                (fx && opts_.bloom) ? kLensDirtStrength : 0.f, 0.f, 0.f, 0.f,
                grading_.contrast, grading_.saturation,
                static_cast<float>(swapchain_.hdrMode()), opts_.hdrPaperWhiteNits,
                wb.x, wb.y, wb.z, static_cast<float>(gradingLutCpu_.size),
            };
            vkCmdPushConstants(c, presentPipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof(presentPush), presentPush);
            vkCmdDraw(c, 3, 1, 0, 0);
            vkCmdEndRendering(c);
        };
    }
    rg_.addPass("PresentOut")
        .access(swapImage, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_NONE,
                VK_ACCESS_2_NONE);

    const bool capturing = (!opts_.screenshotPath.empty() &&
                            static_cast<int>(frameIndex) == opts_.frames - 1) ||
                           !frameDumpDir_.empty();
    if (capturing) {
        RenderGraph::Pass& p = rg_.addPass("Screenshot");
        p.access(finalImage_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sync::kCopy,
                 sync::kTransferRead);
        p.record = [&](VkCommandBuffer c) { captureScreenshotIntoStaging(c); };
        rg_.addPass("ScreenshotBack")
            .access(finalImage_.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kFragment,
                    sync::kSampled);
    }

    rg_.addPass("FrameEnd").record = [&](VkCommandBuffer c) { timestamps_.frameEnd(c, slot); };

    rg_.execute(cmd);
    vkEndCommandBuffer(cmd);

    prevViewProj_ = Mat4::multiply(proj, view);
}

void Renderer::run() {
    uint32_t frameIndex = 0;
    auto lastTime = std::chrono::steady_clock::now();

    while (true) {
        if (!window_.poll()) break;

        // WM_SIZE sets input_.resized; consume it here and rebuild the swapchain
        // against the actual client size (the previous recreate paths used the
        // fixed opts_ display size, so interactive resizes were ignored).
        // TODO: the offscreen targets (finalImage_, screenshot staging, ...) are
        // still allocated from opts_.displayWidth/Height and are not rebuilt
        // here; the present pass scales them to the swapchain extent, so a
        // resize works at the presentation level only.
        if (window_.input().resized) {
            window_.clearMouseDelta();
            recreateSwapchain(static_cast<uint32_t>(window_.width()),
                              static_cast<uint32_t>(window_.height()), opts_.vsync);
        }

        if (opts_.frames >= 0 && static_cast<int>(frameIndex) >= opts_.frames) break;

        const auto now = std::chrono::steady_clock::now();
        const float dt = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;
        // Bench mode (opts_.frames >= 0) keeps the fixed 1/60 default so the
        // upscaler's frame-to-frame integration is reproducible across machines
        // and runs; interactive mode tracks real wall-clock time.
        if (opts_.frames < 0) deltaTime_ = dt;
        updateCamera(frameIndex, dt);

        const uint32_t slot = frameIndex % kFramesInFlight;

        // Ensure the previous work using this frame slot is complete before
        // reusing its fence / image-available semaphore / UBO.
        vkWaitForFences(ctx_.device, 1, &frames_[slot].fence, VK_TRUE, UINT64_MAX);
        vkResetFences(ctx_.device, 1, &frames_[slot].fence);

        // Frame-index driven scene animation (dynamic boxes / glTF clips).
        // After the slot fence: the palette/instance state of this slot is no
        // longer read by the GPU.  No-op for static scenes.
        scene_.advanceToFrame(frameIndex);
        // CPU LOD selection for this frame's camera (viewer: always enabled).
        // Before recording; both GT and upscaled paths share the result.
        scene_.updateLodSelection(camera_.position, camera_.fovY, lodEnabledByDefault());
        // Background fine-mip streaming tick (no-op when streaming is off).
        scene_.updateTextureStreaming(ctx_, camera_.position);
        // complete; harvest its GPU timings before the slot is reused.
        if (!opts_.frameTimesPath.empty() && frameIndex >= kFramesInFlight) {
            frameTimes_.push_back(timestamps_.read(ctx_, slot));
        }
        // Same completion event: harvest the auto-exposure state solved in
        // that frame.  Deterministic per frame index (fixed slot schedule).
        deferred_.harvestExposureChannel(exposureChannel_, slot);

        uint32_t swapIndex = 0;
        VkResult acq = swapchain_.acquireNext(ctx_, frames_[slot].imageAvailable, swapIndex);
        if (acq == VK_ERROR_OUT_OF_DATE_KHR || acq == VK_SUBOPTIMAL_KHR) {
            recreateSwapchain(opts_.displayWidth, opts_.displayHeight, opts_.vsync);
            continue;
        }
        if (acq != VK_SUCCESS) break;

        recordFrame(frameIndex, swapIndex);

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit = {};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &frames_[slot].imageAvailable;
        submit.pWaitDstStageMask = &waitStage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &frames_[slot].cmd;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &renderFinished_[swapIndex];
        VkResult submitRes = VK_SUCCESS;
        {
            // The texture-streaming worker submits one-shot uploads off-thread
            // (Phase 7b); all queue access is serialized via queueMutex.
            std::lock_guard<std::mutex> lk(ctx_.queueMutex);
            submitRes = vkQueueSubmit(ctx_.graphicsQueue, 1, &submit, frames_[slot].fence);
        }
        if (submitRes != VK_SUCCESS) break;

        VkResult pres = swapchain_.present(ctx_, swapIndex, renderFinished_[swapIndex]);
        if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) {
            recreateSwapchain(opts_.displayWidth, opts_.displayHeight, opts_.vsync);
        } else if (pres != VK_SUCCESS) {
            break;
        }

        if (!opts_.screenshotPath.empty() && static_cast<int>(frameIndex) == opts_.frames - 1) {
            vkWaitForFences(ctx_.device, 1, &frames_[slot].fence, VK_TRUE, UINT64_MAX);
            saveScreenshot(opts_.screenshotPath);
        }
        // Per-frame dump (SR_FRAME_DUMP_DIR): the fence wait serialises the
        // pipeline — diagnostics only, never enabled by default.
        if (!frameDumpDir_.empty()) {
            vkWaitForFences(ctx_.device, 1, &frames_[slot].fence, VK_TRUE, UINT64_MAX);
            char name[32];
            std::snprintf(name, sizeof(name), "frame_%04u.png", frameIndex);
            saveScreenshot(frameDumpDir_ + "/" + name);
        }

        ++frameIndex;
    }

    vkDeviceWaitIdle(ctx_.device);

    if (!opts_.frameTimesPath.empty()) {
        // Drain the last in-flight frames (all work is complete after WaitIdle).
        for (uint32_t f = static_cast<uint32_t>(frameTimes_.size()); f < frameIndex; ++f) {
            frameTimes_.push_back(timestamps_.read(ctx_, f % kFramesInFlight));
        }
        const MemoryBudgetInfo budget = queryMemoryBudget(ctx_);
        VkDeviceSize heapTotal = 0;
        VkDeviceSize vmaTotal = 0;
        for (uint32_t i = 0; i < budget.heapCount; ++i) {
            heapTotal += budget.heapUsage[i];
            vmaTotal += budget.vmaAllocationBytes[i];
        }
        const uint64_t algoBytes = upscaler_ ? upscaler_->gpuMemoryBytes() : 0;
        std::ofstream csv(opts_.frameTimesPath);
        csv << "frame,frameMs,sceneMs,upscaleMs,vramAlgoBytes,vramTotalBytes,vramVmaBytes\n";
        for (size_t i = 0; i < frameTimes_.size(); ++i) {
            csv << i << ',' << frameTimes_[i].frameMs << ',' << frameTimes_[i].sceneMs << ','
                << frameTimes_[i].upscaleMs << ',' << algoBytes << ',' << heapTotal << ','
                << vmaTotal << '\n';
        }
        std::fprintf(stdout, "frameTimes=%zu written to %s\n", frameTimes_.size(),
                     opts_.frameTimesPath.c_str());
    }

    if (opts_.frames >= 0 && frameIndex > 0) {
        const TimestampQuery::Timings t = timestamps_.read(ctx_, (frameIndex - 1) % kFramesInFlight);
        std::fprintf(stdout, "frames=%u lastFrameMs=%.3f sceneMs=%.3f upscaleMs=%.3f\n",
                     frameIndex, t.frameMs, t.sceneMs, t.upscaleMs);
    }
}

void Renderer::shutdown() {
    if (!ctx_.device) return;
    vkDeviceWaitIdle(ctx_.device);

    if (upscaler_) { upscaler_->shutdown(); upscaler_.reset(); }

    if (screenshotStaging_) { vmaDestroyBuffer(ctx_.allocator, screenshotStaging_, screenshotStagingMemory_); screenshotStaging_ = VK_NULL_HANDLE; screenshotStagingMemory_ = VK_NULL_HANDLE; }

    deferred_.destroyExposureChannel(ctx_, exposureChannel_);

    timestamps_.destroy(ctx_);

    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        FrameResources& fr = frames_[i];
        if (fr.ubo) { vmaDestroyBuffer(ctx_.allocator, fr.ubo, fr.uboMemory); fr.ubo = VK_NULL_HANDLE; fr.uboMemory = VK_NULL_HANDLE; }
        if (fr.lightingUbo) { vmaDestroyBuffer(ctx_.allocator, fr.lightingUbo, fr.lightingUboMemory); fr.lightingUbo = VK_NULL_HANDLE; fr.lightingUboMemory = VK_NULL_HANDLE; }
        if (fr.imageAvailable) { vkDestroySemaphore(ctx_.device, fr.imageAvailable, nullptr); fr.imageAvailable = VK_NULL_HANDLE; }
        if (fr.fence) { vkDestroyFence(ctx_.device, fr.fence, nullptr); fr.fence = VK_NULL_HANDLE; }
    }
    for (VkSemaphore sem : renderFinished_) {
        if (sem) vkDestroySemaphore(ctx_.device, sem, nullptr);
    }
    renderFinished_.clear();

    if (presentPipeline_) { vkDestroyPipeline(ctx_.device, presentPipeline_, nullptr); presentPipeline_ = VK_NULL_HANDLE; }
    if (presentPipelineLayout_) { vkDestroyPipelineLayout(ctx_.device, presentPipelineLayout_, nullptr); presentPipelineLayout_ = VK_NULL_HANDLE; }
    if (presentFrag_) { vkDestroyShaderModule(ctx_.device, presentFrag_, nullptr); presentFrag_ = VK_NULL_HANDLE; }

    if (descriptorPool_) { vkDestroyDescriptorPool(ctx_.device, descriptorPool_, nullptr); descriptorPool_ = VK_NULL_HANDLE; }
    if (presentSetLayout_) { vkDestroyDescriptorSetLayout(ctx_.device, presentSetLayout_, nullptr); presentSetLayout_ = VK_NULL_HANDLE; }

    if (materialUbo_) { vmaDestroyBuffer(ctx_.allocator, materialUbo_, materialUboMemory_); materialUbo_ = VK_NULL_HANDLE; materialUboMemory_ = VK_NULL_HANDLE; }

    gbColor_.destroy(ctx_);
    gbMotion_.destroy(ctx_);
    gbReactive_.destroy(ctx_);
    gbReactiveDilated_.destroy(ctx_);
    gbDepth_.destroy(ctx_);
    gbAlbedo_.destroy(ctx_);
    gbNormal_.destroy(ctx_);
    gbMaterial_.destroy(ctx_);
    gbEmissive_.destroy(ctx_);
    gtDepth_.destroy(ctx_);
    gtMotion_.destroy(ctx_);
    gtAlbedo_.destroy(ctx_);
    gtNormal_.destroy(ctx_);
    gtMaterial_.destroy(ctx_);
    gtEmissive_.destroy(ctx_);
    gbAoRaw_.destroy(ctx_);
    gbAo_.destroy(ctx_);
    gtAoRaw_.destroy(ctx_);
    gtAo_.destroy(ctx_);
    deferred_.destroyBloomPyramid(ctx_, gbBloom_);
    deferred_.destroyBloomPyramid(ctx_, gtBloom_);
    deferred_.destroyPostFxTargets(ctx_, gbPostFx_);
    deferred_.destroyPostFxTargets(ctx_, gtPostFx_);
    lensDirt_.destroy(ctx_);
    gradingLutGpu_.destroy(ctx_);
    deferred_.destroyColorPyramid(ctx_, gbColorPyramid_);
    deferred_.destroyColorPyramid(ctx_, gtColorPyramid_);
    deferred_.destroyClusterGrid(ctx_, gbCluster_);
    deferred_.destroyClusterGrid(ctx_, gtCluster_);
    deferred_.destroyVolFogVolume(ctx_, gbFog_);
    deferred_.destroyVolFogVolume(ctx_, gtFog_);
    deferred_.destroyDepthPyramid(ctx_, gbPyramid_);
    deferred_.destroyDepthPyramid(ctx_, gtPyramid_);
    deferred_.destroyDepthPyramid(ctx_, gbPyramidAo_);
    deferred_.destroyDepthPyramid(ctx_, gtPyramidAo_);
    deferred_.destroyCullChannel(ctx_, gbCull_);
    deferred_.destroyCullChannel(ctx_, gtCull_);
    deferred_.destroyInstanceBuffer(ctx_, instances_);
    deferred_.destroyAoHistory(ctx_, gbAoHist_);
    deferred_.destroyAoHistory(ctx_, gtAoHist_);
    deferred_.destroySsrHistory(ctx_, gbSsrHist_);
    deferred_.destroySsrHistory(ctx_, gtSsrHist_);
    gbSsrTrace_.destroy(ctx_);
    gtSsrTrace_.destroy(ctx_);
    // Backing blocks of the aliased AO-raw/SSR-trace pairs (images already
    // destroyed above via ImageResource::arena).
    gbAoSsrArena_.destroy(ctx_);
    gtAoSsrArena_.destroy(ctx_);
    finalImage_.destroy(ctx_);

    if (shadowsActive_) { deferred_.destroyShadowTargets(ctx_, shadow_); shadowsActive_ = false; }
    if (spotAtlasActive_) { deferred_.destroyShadowAtlas(ctx_, spotAtlas_); spotAtlasActive_ = false; }

    deferred_.destroy(ctx_);

    scene_.destroy(ctx_);
    swapchain_.destroy(ctx_);
    ctx_.destroy();
    window_.destroy();
}

} // namespace sr
