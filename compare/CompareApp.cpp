#include "compare/CompareApp.h"

#include "app/CliUtils.h"

#include "compare/Font5x7.h"
#include "renderer/ColorGrading.h"
#include "renderer/Screenshot.h"
#include "renderer/core/PathUtil.h"
#include "renderer/core/VkUtil.h"
#include "renderer/core/Vma.h"
#include "renderer/scene/SceneRegistry.h"
#include "upscalers/UpscalerFactory.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace sr {

namespace {

constexpr VkFormat kComposeFormat = VK_FORMAT_R8G8B8A8_UNORM;

// SceneUBO / MaterialUBO / LightingUBO / ScenePush and the GBuffer formats are
// shared with the viewer via renderer/deferred/DeferredCore.h.

// Compose pass push constants: column pixel size + text scale + text slot,
// the source-region window (normalized offset/size) and source dimensions,
// plus the terminal lens-effects chain (Phase 6a; same algorithm/defaults as
// the viewer present.frag — lens dirt excluded, the compare paths have no
// bloom chain) and the log-domain grading set (Phase 6c; identical for every
// column — compare stays SDR, HDR presentation is viewer/GUI-only).
struct ComposePush {
    float colSize[2];
    float textScale;
    float textSlot;
    float uvRect[4];  // normalized source region: offset xy, size zw
    float srcSize[2]; // source image pixels
    float nearest;    // != 0: sample nearest (magnification >= 1:1)
    float exposure;   // display-domain ACES input multiplier
    float lensA[4];   // x = chromatic aberration, y = vignette, z = film grain,
                      // w = frame index (grain hash seed)
    float gradeA[4];  // x = contrast, y = saturation (zw unused; SDR only)
    float gradeB[4];  // xyz = white balance, w = LUT size
};
static_assert(sizeof(ComposePush) == 96, "ComposePush size mismatch");

// Metric compute push constants (two uvec4s in the shaders).
struct MetricPush {
    uint32_t x = 0, y = 0, z = 0, w = 0;      // region offset xy, extent zw
    uint32_t x2 = 0, y2 = 0, z2 = 0, w2 = 0;  // x2 = blocks per row (region)
    float exposureTest = 1.f; // test image (upscaler column / LR path)
    float exposureRef = 1.f;  // reference image (GT column / GT path)
    float pad[2] = {};
};
static_assert(sizeof(MetricPush) == 48, "MetricPush std140/push size mismatch");

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

// Compute-shader write -> read barrier for the metric SSBO chain.
void computeBarrier(VkCommandBuffer cmd, VkAccessFlags dstAccess, VkPipelineStageFlags dstStage) {
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, dstStage, 0, 1, &barrier, 0,
                         nullptr, 0, nullptr);
}

char asciiUpper(char c) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
}

} // namespace

void CompareApp::ImageResource::destroy(const VulkanContext& ctx) {
    if (view) { vkDestroyImageView(ctx.device, view, nullptr); view = VK_NULL_HANDLE; }
    if (image) { vmaDestroyImage(ctx.allocator, image, memory); image = VK_NULL_HANDLE; memory = VK_NULL_HANDLE; }
}

void CompareApp::computeViewRegion(uint32_t srcW, uint32_t srcH, uint32_t colW, uint32_t colH,
                                   float zoom, float centerU, float centerV, float out[4]) {
    // Fill the column while keeping the source aspect: scale to cover, crop
    // the overflow; the zoom window shrinks the visible region around the
    // requested center (clamped to the image).
    const float s = std::max(static_cast<float>(colW) / static_cast<float>(srcW),
                             static_cast<float>(colH) / static_cast<float>(srcH));
    float sizeX = std::min(static_cast<float>(colW) / s, static_cast<float>(srcW));
    float sizeY = std::min(static_cast<float>(colH) / s, static_cast<float>(srcH));
    zoom = std::max(1.f, zoom);
    sizeX /= zoom;
    sizeY /= zoom;
    const float maxOffX = std::max(static_cast<float>(srcW) - sizeX, 0.f);
    const float maxOffY = std::max(static_cast<float>(srcH) - sizeY, 0.f);
    out[0] = std::clamp(centerU * static_cast<float>(srcW) - sizeX * 0.5f, 0.f, maxOffX);
    out[1] = std::clamp(centerV * static_cast<float>(srcH) - sizeY * 0.5f, 0.f, maxOffY);
    out[2] = sizeX;
    out[3] = sizeY;
}

bool CompareApp::init(const CompareOptions& opts) {
    opts_ = opts;
    if (opts_.metricInterval < 1) opts_.metricInterval = 1;
    opts_.zoom = std::clamp(opts_.zoom, 1.f, 16.f);
    opts_.zoomCenterU = std::clamp(opts_.zoomCenterU, 0.f, 1.f);
    opts_.zoomCenterV = std::clamp(opts_.zoomCenterV, 0.f, 1.f);
    // renderScale feeds a float->uint32 cast below; keep it in (0,1] so the
    // cast stays well-defined even if the caller passed an out-of-range value.
    opts_.renderScale = std::clamp(opts_.renderScale, 0.01f, 1.f);

    // Diagnostic env switches (see CompareApp.h for their meaning).
    diagNoJitter_ = sr::envFlag("SR_NO_JITTER");
    diagMetricStdout_ = sr::envFlag("SR_METRIC_STDOUT");
    // GPU occlusion culling (Phase 7a) defaults on; SR_OCCLUSION=0 opts out.
    occlusion_ = occlusionEnabledByDefault();

    if (!window_.create("sr_compare — compare", static_cast<int>(opts.displayWidth),
                        static_cast<int>(opts.displayHeight)))
        return false;
    if (!ctx_.create(window_)) return false;
    // Swapchain extent = physical framebuffer pixels (SDL_GetWindowSizeInPixels).
    if (!swapchain_.create(ctx_, static_cast<uint32_t>(window_.pixelWidth()),
                           static_cast<uint32_t>(window_.pixelHeight()), opts.vsync))
        return false;

    renderWidth_ = std::max(1u, static_cast<uint32_t>(static_cast<float>(opts_.displayWidth) * opts_.renderScale));
    renderHeight_ = std::max(1u, static_cast<uint32_t>(static_cast<float>(opts_.displayHeight) * opts_.renderScale));

    bool sceneOk = false;
    if (!opts.scenePath.empty()) sceneOk = scene_.loadGltf(ctx_, opts.scenePath.c_str());
    if (!sceneOk) sceneOk = scene_.loadProcedural(ctx_);
    if (!sceneOk) return false;
    hasTransparency_ = deferred_.sceneHasTransparency(scene_);
    const LightingPreset preset = lightingPresetForScene(opts.scenePath);
    iblIntensity_ = preset.iblIntensity;
    // Froxel volumetric fog (Phase 5a): media parameters from the scene
    // preset; --no-volfog gates the passes (same rule as the viewer).
    fogParams_ = preset.fog;
    volFogActive_ = opts_.volFog && fogParams_.enabled;
    // Reflection probe placements (Phase 4c-2); inert without a bake file.
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

    if (!initAlgorithms()) return false;
    // Shared deferred pipeline: IBL maps + shaders + layouts + pipelines.  Env
    // source priority: CLI --env-map, then the preset's envFile, else the
    // procedural sky atmosphere for the preset sun direction.
    const std::string envPath =
        !opts_.envMapPath.empty() ? opts_.envMapPath : preset.envFile;
    if (!deferred_.init(ctx_, envPath.c_str(),
                        sunDirectionFromElevAzimuth(preset.sunElevationDeg,
                                                    preset.sunAzimuthDeg)))
        return false;
    // Baked reflection probes (Phase 4c-2): count 0 without a bake file.
    deferred_.loadProbes(ctx_, scene_.probes, probeFilePathForScene(opts_.scenePath));
    // CSM shadow targets (fixed size; a failure degrades to no shadows).
    // Created before createSyncResources so the lighting/transparent sets can
    // bind the array view.
    shadowsActive_ = deferred_.createShadowTargets(ctx_, shadow_);
    if (!shadowsActive_)
        std::fprintf(stderr, "warning: shadow target creation failed, shadows disabled\n");
    // Spot shadow atlas (Phase 4b); a failure degrades to sun-only shadows.
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
    if (!createRenderTargets()) return false;
    if (!createFontAtlas()) return false;
    // Grading LUT (Phase 6c): compare uses the procedural identity LUT with
    // the neutral grading set (same parameters for every column, SDR only).
    if (!gradingLut_.create(ctx_, makeIdentityLut())) return false;
    if (!createMetricResources()) return false;
    if (!createShaders()) return false;
    if (!createDescriptors()) return false;
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

    refreshOverlayText();
    return true;
}

bool CompareApp::initAlgorithms() {
    std::vector<std::string> names = opts_.upscalerNames;
    if (names.empty()) names = listUpscalers(); // default: every registered plugin

    if (names.size() > kMaxAlgos) {
        std::fprintf(stderr, "warning: %zu upscalers requested, truncating to %u columns\n",
                     names.size(), kMaxAlgos);
        names.resize(kMaxAlgos);
    }

    const VulkanEnv env = ctx_.toEnv();
    for (const std::string& name : names) {
        std::unique_ptr<IUpscaler> up = createUpscaler(name.c_str());
        if (!up) {
            std::fprintf(stderr, "warning: unknown upscaler '%s', skipped\n", name.c_str());
            continue;
        }
        if (!up->isAvailable(env)) {
            std::fprintf(stderr, "warning: upscaler '%s' is not available, skipped\n", name.c_str());
            continue;
        }
        UpscalerDesc desc;
        desc.renderWidth = renderWidth_;
        desc.renderHeight = renderHeight_;
        desc.displayWidth = opts_.displayWidth;
        desc.displayHeight = opts_.displayHeight;
        desc.hdr = true;
        desc.invertedDepth = false;
        desc.infiniteFarPlane = true;
        if (!up->init(env, desc)) {
            std::fprintf(stderr, "warning: upscaler '%s' init failed, skipped\n", name.c_str());
            continue;
        }
        AlgoColumn col;
        col.id = name;
        col.upscaler = std::move(up);
        algos_.push_back(std::move(col));
    }

    if (algos_.empty()) {
        std::fprintf(stderr, "compare mode: no usable upscaler (need at least one column)\n");
        return false;
    }
    return true;
}

bool CompareApp::createRenderTargets() {
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

    // Shared low-resolution GBuffer inputs for every upscaler.  The deferred
    // pass writes albedo/normal/material/emissive/motion + depth, then the
    // lighting pass resolves them into gbColor_.
    // TRANSFER_SRC/DST: some upscalers (DLSS via Streamline) copy the inputs
    // into internal buffers instead of just sampling them.
    if (!createRT(gbColor_, renderWidth_, renderHeight_, deferred::kHdrColorFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                      VK_IMAGE_USAGE_STORAGE_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    // Unjittered LR color for spatial plugins when mixed with temporal ones
    // (raster jitter cannot be undone by resampling).
    if (!createRT(gbColorSpatial_, renderWidth_, renderHeight_, deferred::kHdrColorFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gbMotion_, renderWidth_, renderHeight_, deferred::kMotionFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    // Translucent coverage mask (reactive/TC mask for upscalers).
    if (!createRT(gbReactive_, renderWidth_, renderHeight_, deferred::kReactiveFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gbDepth_, renderWidth_, renderHeight_, deferred::kDepthFormat,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_DEPTH_BIT))
        return false;
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

    // GTAO targets (working RG16F + filtered R16F) for the low-res path.
    const VkImageUsageFlags aoUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (!createRT(gbAoRaw_, renderWidth_, renderHeight_, kAoRawFormat, aoUsage,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gbAo_, renderWidth_, renderHeight_, VK_FORMAT_R16_SFLOAT, aoUsage,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    // Hi-Z depth pyramids for the SSR march (LR / GT / GT-SSAA paths).
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
    if (!createRT(gbSsrTrace_, renderWidth_, renderHeight_, kSsrTraceFormat, aoUsage,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtSsrTrace_, dw, dh, kSsrTraceFormat, aoUsage, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!deferred_.createSsrHistory(ctx_, renderWidth_, renderHeight_, gbSsrHist_))
        return false;
    if (!deferred_.createSsrHistory(ctx_, dw, dh, gtSsrHist_))
        return false;
    // Color mip chains for roughness-aware SSR (mip 0 = the lit-color copy).
    if (!deferred_.createColorPyramid(ctx_, renderWidth_, renderHeight_, gbColorPyramid_))
        return false;
    if (!deferred_.createColorPyramid(ctx_, dw, dh, gtColorPyramid_))
        return false;
    // Clustered shading grids (per-path resolution, per-slot buffers).  The LR
    // grid serves both the jittered (temporal) and unjittered (spatial)
    // lighting variants — jitter only shifts cluster boundaries sub-pixel.
    if (!deferred_.createClusterGrid(ctx_, renderWidth_, renderHeight_, gbCluster_))
        return false;
    if (!deferred_.createClusterGrid(ctx_, dw, dh, gtCluster_))
        return false;
    // Froxel volumetric fog volumes (per-path resolution; Phase 5a).  The LR
    // volume serves both the jittered and unjittered (spatial) lighting
    // variants.  A creation failure degrades to no fog (same as shadows).
    if (volFogActive_) {
        if (!deferred_.createVolFogVolume(ctx_, renderWidth_, renderHeight_, gbFog_) ||
            !deferred_.createVolFogVolume(ctx_, dw, dh, gtFog_)) {
            std::fprintf(stderr, "warning: volumetric fog volume creation failed, fog disabled\n");
            deferred_.destroyVolFogVolume(ctx_, gbFog_);
            deferred_.destroyVolFogVolume(ctx_, gtFog_);
            volFogActive_ = false;
        }
    }

    // Native-resolution ground truth (same camera, no jitter).  The lighting
    // pass samples the GT GBuffer depth, so it needs the SAMPLED bit.
    if (!createRT(gtColor_, dw, dh, deferred::kHdrColorFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtDepth_, dw, dh, deferred::kDepthFormat,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                  VK_IMAGE_ASPECT_DEPTH_BIT))
        return false;
    if (!createRT(gtAlbedo_, dw, dh, deferred::kAlbedoFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtNormal_, dw, dh, deferred::kNormalFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtMaterial_, dw, dh, deferred::kMaterialFormat, gbUsage,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtEmissive_, dw, dh, deferred::kEmissiveFormat, gbUsage,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtAoRaw_, dw, dh, kAoRawFormat, aoUsage, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!createRT(gtAo_, dw, dh, VK_FORMAT_R16_SFLOAT, aoUsage, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    // GT motion RT (Phase 6b) + per-path post-fx working targets (LR/GT).
    if (!createRT(gtMotion_, dw, dh, deferred::kMotionFormat, gbUsage, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (!deferred_.createPostFxTargets(ctx_, renderWidth_, renderHeight_, gbPostFx_))
        return false;
    if (!deferred_.createPostFxTargets(ctx_, dw, dh, gtPostFx_))
        return false;

    // 200% SSAA ground truth: render at 2x, downsample into gtColor_.
    if (opts_.gtSsaa) {
        if (!createRT(gtSsaaColor_, dw * 2, dh * 2, deferred::kHdrColorFormat,
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                      VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
        if (!createRT(gtSsaaDepth_, dw * 2, dh * 2, deferred::kDepthFormat,
                      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                      VK_IMAGE_ASPECT_DEPTH_BIT))
            return false;
        if (!createRT(gtSsaaMotion_, dw * 2, dh * 2, deferred::kMotionFormat, gbUsage,
                      VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
        if (!deferred_.createPostFxTargets(ctx_, dw * 2, dh * 2, gtSsaaPostFx_))
            return false;
        if (!createRT(gtSsaaAlbedo_, dw * 2, dh * 2, deferred::kAlbedoFormat, gbUsage,
                      VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
        if (!createRT(gtSsaaNormal_, dw * 2, dh * 2, deferred::kNormalFormat, gbUsage,
                      VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
        if (!createRT(gtSsaaMaterial_, dw * 2, dh * 2, deferred::kMaterialFormat, gbUsage,
                      VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
        if (!createRT(gtSsaaEmissive_, dw * 2, dh * 2, deferred::kEmissiveFormat, gbUsage,
                      VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
        if (!createRT(gtSsaaAoRaw_, dw * 2, dh * 2, kAoRawFormat, aoUsage,
                      VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
        if (!createRT(gtSsaaAo_, dw * 2, dh * 2, VK_FORMAT_R16_SFLOAT, aoUsage,
                      VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
        if (!deferred_.createDepthPyramid(ctx_, dw * 2, dh * 2, gtSsaaPyramid_))
            return false;
        if (!deferred_.createDepthPyramid(ctx_, dw * 2, dh * 2, gtSsaaPyramidAo_,
                                          /*aoFilter=*/true, camera_.nearPlane, camera_.farPlane))
            return false;
        if (!deferred_.createAoHistory(ctx_, dw * 2, dh * 2, gtSsaaAoHist_))
            return false;
        if (!createRT(gtSsaaSsrTrace_, dw * 2, dh * 2, kSsrTraceFormat, aoUsage,
                      VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
        if (!deferred_.createSsrHistory(ctx_, dw * 2, dh * 2, gtSsaaSsrHist_))
            return false;
        if (!deferred_.createColorPyramid(ctx_, dw * 2, dh * 2, gtSsaaColorPyramid_))
            return false;
        if (!deferred_.createClusterGrid(ctx_, dw * 2, dh * 2, gtSsaaCluster_))
            return false;
        if (volFogActive_ && !deferred_.createVolFogVolume(ctx_, dw * 2, dh * 2, gtSsaaFog_)) {
            std::fprintf(stderr, "warning: SSAA volumetric fog volume creation failed, SSAA fog disabled\n");
            deferred_.destroyVolFogVolume(ctx_, gtSsaaFog_);
        }
    }

    // Per-algorithm display-resolution outputs (storage during dispatch).
    for (AlgoColumn& algo : algos_) {
        if (!createRT(algo.output, dw, dh, deferred::kHdrColorFormat,
                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                      VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
    }

    // Tonemapped composite (columns + overlay text); also the screenshot source.
    if (!createRT(composeImage_, dw, dh, kComposeFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    return true;
}

bool CompareApp::createFontAtlas() {
    // Rasterize the 5x7 bitmap font into an R8 atlas and upload it.
    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation stagingMemory = VK_NULL_HANDLE;
    const VkDeviceSize size = kFontAtlasW * kFontAtlasH;
    if (createBuffer(ctx_, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging, stagingMemory) != VK_SUCCESS)
        return false;
    void* mapped = nullptr;
    vmaMapMemory(ctx_.allocator, stagingMemory, &mapped);
    buildFontAtlas(static_cast<uint8_t*>(mapped));
    vmaUnmapMemory(ctx_.allocator, stagingMemory);

    if (createImage(ctx_, kFontAtlasW, kFontAtlasH, VK_FORMAT_R8_UNORM,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    fontAtlas_.image, fontAtlas_.memory) != VK_SUCCESS) {
        vmaDestroyBuffer(ctx_.allocator, staging, stagingMemory);
        return false;
    }
    fontAtlas_.width = kFontAtlasW;
    fontAtlas_.height = kFontAtlasH;
    fontAtlas_.format = VK_FORMAT_R8_UNORM;

    submitUploadOneShot(
        ctx_,
        [&](VkCommandBuffer cmd) {
            copyBufferToImageTransferStage(cmd, staging, fontAtlas_.image, kFontAtlasW,
                                           kFontAtlasH);
        },
        [&](VkCommandBuffer cmd) { transitionImageToShaderRead(cmd, fontAtlas_.image); });
    vmaDestroyBuffer(ctx_.allocator, staging, stagingMemory);

    fontAtlas_.view = createImageView(ctx_, fontAtlas_.image, VK_FORMAT_R8_UNORM,
                                      VK_IMAGE_ASPECT_COLOR_BIT);
    return fontAtlas_.view != VK_NULL_HANDLE;
}

bool CompareApp::createMetricResources() {
    const uint32_t dw = opts_.displayWidth;
    const uint32_t dh = opts_.displayHeight;
    blocksPerRow_ = (dw + 7) / 8;
    blockCount_ = blocksPerRow_ * ((dh + 7) / 8);

    const VkDeviceSize blocksSize = static_cast<VkDeviceSize>(blockCount_) * kMetricFloats * 4;
    for (AlgoColumn& algo : algos_) {
        if (createBuffer(ctx_, blocksSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, algo.blocksBuffer,
                         algo.blocksMemory) != VK_SUCCESS)
            return false;
    }

    // One device-local result record (8 floats) per algorithm.
    const VkDeviceSize resultSize = kMaxAlgos * kMetricFloats * 4;
    if (createBuffer(ctx_, resultSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, metricResultBuf_,
                     metricResultMemory_) != VK_SUCCESS)
        return false;

    // Per-frame-slot host-visible readback staging.
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        if (createBuffer(ctx_, resultSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         metricStaging_[i], metricStagingMemory_[i]) != VK_SUCCESS)
            return false;
        vmaMapMemory(ctx_.allocator, metricStagingMemory_[i],
                    &metricStagingMapped_[i]);
    }
    return true;
}

bool CompareApp::loadShader(const char* name, VkShaderModule& out) {
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

bool CompareApp::createShaders() {
    // Scene/lighting shaders live in DeferredCore (shared with the viewer).
    return loadShader("fullscreen.vert.spv", fullscreenVert_) &&
           loadShader("compare_compose.frag.spv", composeFrag_) &&
           loadShader("compare_copy.frag.spv", copyFrag_) &&
           loadShader("compare_metrics_blocks.comp.spv", metricBlocksComp_) &&
           loadShader("compare_metrics_reduce.comp.spv", metricReduceComp_);
}

bool CompareApp::createDescriptors() {
    // Scene/texture/lighting set layouts live in DeferredCore (shared with the
    // viewer); only the compare-specific layouts are created here.

    // Compose: column source + packed text UBO + font atlas + grading LUT.
    VkDescriptorSetLayoutBinding composeBindings[4] = {};
    for (uint32_t i = 0; i < 4; ++i) {
        composeBindings[i].binding = i;
        composeBindings[i].descriptorCount = 1;
        composeBindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        composeBindings[i].descriptorType =
            i == 1 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    }
    VkDescriptorSetLayoutCreateInfo composeLayoutCi = {};
    composeLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    composeLayoutCi.bindingCount = 4;
    composeLayoutCi.pBindings = composeBindings;
    if (vkCreateDescriptorSetLayout(ctx_.device, &composeLayoutCi, nullptr, &composeSetLayout_) != VK_SUCCESS)
        return false;

    VkDescriptorSetLayoutBinding copyBinding = {};
    copyBinding.binding = 0;
    copyBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    copyBinding.descriptorCount = 1;
    copyBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo copyLayoutCi = {};
    copyLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    copyLayoutCi.bindingCount = 1;
    copyLayoutCi.pBindings = &copyBinding;
    if (vkCreateDescriptorSetLayout(ctx_.device, &copyLayoutCi, nullptr, &copySetLayout_) != VK_SUCCESS)
        return false;

    // Metrics: test/ref images + per-block SSBO + result SSBO (one layout for
    // both compute passes).
    VkDescriptorSetLayoutBinding metricBindings[4] = {};
    for (uint32_t i = 0; i < 4; ++i) {
        metricBindings[i].binding = i;
        metricBindings[i].descriptorCount = 1;
        metricBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        metricBindings[i].descriptorType =
            i < 2 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    VkDescriptorSetLayoutCreateInfo metricLayoutCi = {};
    metricLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    metricLayoutCi.bindingCount = 4;
    metricLayoutCi.pBindings = metricBindings;
    if (vkCreateDescriptorSetLayout(ctx_.device, &metricLayoutCi, nullptr, &metricSetLayout_) != VK_SUCCESS)
        return false;

    // --- pool ----------------------------------------------------------------
    const uint32_t numAlgos = static_cast<uint32_t>(algos_.size());
    const uint32_t numColumns = 1 + numAlgos;
    // Hi-Z / color downsample sets: one per mip per pyramid (sampler + storage image).
    const uint32_t hizSets = gbPyramid_.mipCount + gtPyramid_.mipCount + gtSsaaPyramid_.mipCount +
                             gbPyramidAo_.mipCount + gtPyramidAo_.mipCount +
                             gtSsaaPyramidAo_.mipCount;
    const uint32_t colorSets =
        gbColorPyramid_.mipCount + gtColorPyramid_.mipCount + gtSsaaColorPyramid_.mipCount;
    VkDescriptorPoolSize sizes[5] = {};
    const uint32_t fogPaths = volFogActive_ ? 3 : 0; // froxel fog sets per path (Phase 5a)
    const uint32_t postFxPaths = opts_.gtSsaa ? 3 : 2; // MB/DOF set paths (Phase 6b: GB/GT[/SSAA])
    sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = deferred::kMaxTextures + numColumns * 3 + 2 + numAlgos * 2 +
                               14 * kFramesInFlight * 4 + // lighting sets (GB/GT/SSAA/spatial), shadow + atlas + 2 probe arrays
                               8 * kFramesInFlight * 4 +  // transparent sets + SSR + fog volume
                               11 * kFramesInFlight * 4 + // opaque-SSR trace sets (GB/GT/SSAA/spatial), +2 probe arrays
                               10 * 3 +                   // ssao + temporal + blur samplers (per path)
                               3 * 6 +                    // ssr temporal samplers (GB/GT/SSAA x2 sets)
                               17 * postFxPaths +         // post-fx MB/DOF samplers (Phase 6b)
                               2 +                        // auto-exposure HDR sources (LR + GT)
                               14 * fogPaths +            // volfog light/temporal/march/composite samplers
                               3 +                        // occlusion cull sets: Hi-Z chains (GB/GT/SSAA, Phase 7a)
                               hizSets + colorSets;
    sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[1].descriptorCount = kFramesInFlight * 2 + numColumns +
                               kFramesInFlight * 2 + // lighting UBOs
                               kFramesInFlight * 4 + // lighting probe UBOs
                               kFramesInFlight * 3 +  // transparent UBOs (GB/GT/SSAA)
                               kFramesInFlight * 4 +  // opaque-SSR UBOs (GB/GT/SSAA/spatial)
                               kFramesInFlight * 4 +  // opaque-SSR probe UBOs
                               3 * kFramesInFlight +   // spatial scene + lighting + transparent
                               kClusterSlots * fogPaths; // volfog light sets
    sizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    sizes[2].descriptorCount = kFramesInFlight * 3; // Gb + Gt + spatial scene sets
    sizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[3].descriptorCount = numAlgos * 2 + 4 + // metric blocks/result + auto-exposure (LR + GT)
                               2 * kFramesInFlight * 4 + // lighting sets: cluster lights + grid SSBOs
                               2 * kClusterSlots * 3 +    // cluster assign sets (GB/GT/SSAA paths)
                               kFramesInFlight * 3 +      // scene sets: instance SSBO (Phase 7a)
                               2 * 3 +                    // occlusion cull sets (GB/GT/SSAA, Phase 7a)
                               2 * kClusterSlots * fogPaths; // volfog light sets
    sizes[4].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    // ssao raw + temporal history + blur outputs (GB/GT/SSAA) + pyramid mips +
    // SSR trace targets + SSR temporal history write / scene-color RMW (x2 sets per path)
    // + volfog inject/light/temporal/march/composite storage (per fog path)
    sizes[4].descriptorCount = 15 + hizSets + colorSets + kFramesInFlight * 4 + 2 * 6 +
                               11 * postFxPaths + // post-fx MB/DOF storage (Phase 6b)
                               2 * postFxPaths +  // post-fx MB copy-back storage (Phase 6b)
                               8 * fogPaths;
    VkDescriptorPoolCreateInfo poolCi = {};
    poolCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCi.maxSets = kFramesInFlight * 2 + 2 + numColumns + 1 + numAlgos + kFramesInFlight * 3 +
                     kFramesInFlight * 3 + // transparent sets (GB/GT/SSAA)
                     kFramesInFlight * 4 + // opaque-SSR sets (GB/GT/SSAA/spatial)
                     15 +                   // ssao + temporal + blur sets (GB/GT/SSAA, static)
                     6 +                    // ssr temporal sets (GB/GT/SSAA x2, static)
                     2 +                    // auto-exposure sets (LR + GT)
                     3 * kFramesInFlight + // spatial scene + lighting + transparent
                     kClusterSlots * 3 +    // cluster assign sets (GB/GT/SSAA)
                     8 * postFxPaths +      // post-fx MB/DOF sets (Phase 6b)
                     1 * postFxPaths +      // post-fx MB copy-back set (Phase 6b)
                     8 * fogPaths +         // volfog sets (per fog path)
                     3 +                    // occlusion cull sets (GB/GT/SSAA, Phase 7a)
                     hizSets + colorSets;
    poolCi.poolSizeCount = 5;
    poolCi.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(ctx_.device, &poolCi, nullptr, &descriptorPool_) != VK_SUCCESS)
        return false;

    // Cluster assignment sets (per slot, per path); buffers from createRenderTargets.
    if (!deferred_.writeClusterGridSets(ctx_, descriptorPool_, gbCluster_)) return false;
    if (!deferred_.writeClusterGridSets(ctx_, descriptorPool_, gtCluster_)) return false;
    if (opts_.gtSsaa && !deferred_.writeClusterGridSets(ctx_, descriptorPool_, gtSsaaCluster_))
        return false;
    if (!deferred_.writeDepthPyramidSets(ctx_, descriptorPool_, gbDepth_.view, gbPyramid_))
        return false;
    if (!deferred_.writeDepthPyramidSets(ctx_, descriptorPool_, gtDepth_.view, gtPyramid_))
        return false;
    if (opts_.gtSsaa &&
        !deferred_.writeDepthPyramidSets(ctx_, descriptorPool_, gtSsaaDepth_.view,
                                         gtSsaaPyramid_))
        return false;
    if (!deferred_.writeDepthPyramidSets(ctx_, descriptorPool_, gbDepth_.view, gbPyramidAo_))
        return false;
    if (!deferred_.writeDepthPyramidSets(ctx_, descriptorPool_, gtDepth_.view, gtPyramidAo_))
        return false;
    if (opts_.gtSsaa &&
        !deferred_.writeDepthPyramidSets(ctx_, descriptorPool_, gtSsaaDepth_.view,
                                         gtSsaaPyramidAo_))
        return false;
    // Color chain set 0 samples the lit HDR target of each path.
    if (!deferred_.writeColorPyramidSets(ctx_, descriptorPool_, gbColor_.view, gbColorPyramid_))
        return false;
    if (!deferred_.writeColorPyramidSets(ctx_, descriptorPool_, gtColor_.view, gtColorPyramid_))
        return false;
    if (opts_.gtSsaa &&
        !deferred_.writeColorPyramidSets(ctx_, descriptorPool_, gtSsaaColor_.view,
                                         gtSsaaColorPyramid_))
        return false;
    // Post-fx sets (Phase 6b): lit HDR target + motion + depth per path.
    if (!deferred_.writePostFxSets(ctx_, descriptorPool_, gbColor_.view, gbMotion_.view,
                                   gbDepth_.view, gbPostFx_))
        return false;
    if (!deferred_.writePostFxSets(ctx_, descriptorPool_, gtColor_.view, gtMotion_.view,
                                   gtDepth_.view, gtPostFx_))
        return false;
    if (opts_.gtSsaa &&
        !deferred_.writePostFxSets(ctx_, descriptorPool_, gtSsaaColor_.view,
                                   gtSsaaMotion_.view, gtSsaaDepth_.view, gtSsaaPostFx_))
        return false;

    linearSampler_ = createSampler(ctx_, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    fontSampler_ = createSampler(ctx_, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    if (!linearSampler_ || !fontSampler_) return false;

    // --- material UBO (same data path as the viewer renderer) ----------------
    if (!deferred_.createMaterialUbo(ctx_, scene_, materialUbo_, materialUboMemory_,
                                     materialStride_))
        return false;

    // --- packed overlay text UBO (shared by all compose sets) ----------------
    const VkDeviceSize textSize = kMaxColumns * kTextCharsPerColumn * 4;
    if (createBuffer(ctx_, textSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     textUbo_, textUboMemory_) != VK_SUCCESS)
        return false;
    vmaMapMemory(ctx_.allocator, textUboMemory_, &textUboMapped_);

    // --- allocate sets ---------------------------------------------------------
    auto allocSet = [&](VkDescriptorSetLayout layout, VkDescriptorSet& set) {
        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool = descriptorPool_;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &layout;
        return vkAllocateDescriptorSets(ctx_.device, &alloc, &set) == VK_SUCCESS;
    };

    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        if (!allocSet(deferred_.sceneSetLayout(), frames_[i].sceneSetGb)) return false;
        if (!allocSet(deferred_.sceneSetLayout(), frames_[i].sceneSetGbSpatial)) return false;
        if (!allocSet(deferred_.sceneSetLayout(), frames_[i].sceneSetGt)) return false;
        if (!allocSet(deferred_.lightingSetLayout(), frames_[i].lightingSetGb)) return false;
        if (!allocSet(deferred_.lightingSetLayout(), frames_[i].lightingSetGbSpatial)) return false;
        if (!allocSet(deferred_.lightingSetLayout(), frames_[i].lightingSetGt)) return false;
        if (opts_.gtSsaa && !allocSet(deferred_.lightingSetLayout(), frames_[i].lightingSetSsaa))
            return false;
        if (!allocSet(deferred_.transparentSetLayout(), frames_[i].transparentSetGb)) return false;
        if (!allocSet(deferred_.transparentSetLayout(), frames_[i].transparentSetGbSpatial))
            return false;
        if (!allocSet(deferred_.transparentSetLayout(), frames_[i].transparentSetGt)) return false;
        if (opts_.gtSsaa &&
            !allocSet(deferred_.transparentSetLayout(), frames_[i].transparentSetSsaa))
            return false;
        if (!allocSet(deferred_.ssrSetLayout(), frames_[i].ssrSetGb)) return false;
        if (!allocSet(deferred_.ssrSetLayout(), frames_[i].ssrSetGbSpatial)) return false;
        if (!allocSet(deferred_.ssrSetLayout(), frames_[i].ssrSetGt)) return false;
        if (opts_.gtSsaa && !allocSet(deferred_.ssrSetLayout(), frames_[i].ssrSetSsaa))
            return false;
    }
    if (!allocSet(deferred_.textureSetLayout(), textureSet_)) return false;
    if (!allocSet(deferred_.ssaoSetLayout(), ssaoSetGb_)) return false;
    if (!allocSet(deferred_.ssaoSetLayout(), ssaoSetGt_)) return false;
    if (opts_.gtSsaa) {
        if (!allocSet(deferred_.ssaoSetLayout(), ssaoSetSsaa_)) return false;
    }
    if (!allocSet(composeSetLayout_, gtComposeSet_)) return false;
    if (!allocSet(copySetLayout_, copySet_)) return false;
    if (opts_.gtSsaa && !allocSet(copySetLayout_, gtDownsampleSet_)) return false;
    for (AlgoColumn& algo : algos_) {
        if (!allocSet(composeSetLayout_, algo.composeSet)) return false;
        if (!allocSet(metricSetLayout_, algo.metricSet)) return false;
    }

    // --- writes ---------------------------------------------------------------
    deferred_.writeTextureSet(ctx_, textureSet_, scene_);
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
    if (opts_.gtSsaa) {
        deferred_.writeSsaoSet(ctx_, ssaoSetSsaa_, gtSsaaPyramidAo_.chainView,
                               gtSsaaNormal_.view, gtSsaaAoRaw_.view);
        if (!deferred_.writeAoHistorySets(ctx_, descriptorPool_, gtSsaaAoRaw_.view,
                                          gtSsaaDepth_.view, gtSsaaAo_.view, gtSsaaAoHist_))
            return false;
    }
    if (!deferred_.writeSsrHistorySets(ctx_, descriptorPool_, gbSsrTrace_.view, gbDepth_.view,
                                       gbColor_.view, gbSsrHist_))
        return false;
    if (!deferred_.writeSsrHistorySets(ctx_, descriptorPool_, gtSsrTrace_.view, gtDepth_.view,
                                       gtColor_.view, gtSsrHist_))
        return false;
    if (opts_.gtSsaa) {
        if (!deferred_.writeSsrHistorySets(ctx_, descriptorPool_, gtSsaaSsrTrace_.view,
                                           gtSsaaDepth_.view, gtSsaaColor_.view, gtSsaaSsrHist_))
            return false;
    }

    auto writeComposeSet = [&](VkDescriptorSet set, VkImageView source) {
        VkDescriptorImageInfo srcInfo = {};
        srcInfo.sampler = linearSampler_;
        srcInfo.imageView = source;
        srcInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorBufferInfo textInfo = {};
        textInfo.buffer = textUbo_;
        textInfo.offset = 0;
        textInfo.range = textSize;
        VkDescriptorImageInfo fontInfo = {};
        fontInfo.sampler = fontSampler_;
        fontInfo.imageView = fontAtlas_.view;
        fontInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo lutInfo = {};
        lutInfo.sampler = gradingLut_.sampler();
        lutInfo.imageView = gradingLut_.view();
        lutInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet writes[4] = {};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = set;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &srcInfo;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = set;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[1].pBufferInfo = &textInfo;
        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = set;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].pImageInfo = &fontInfo;
        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = set;
        writes[3].dstBinding = 3;
        writes[3].descriptorCount = 1;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[3].pImageInfo = &lutInfo;
        vkUpdateDescriptorSets(ctx_.device, 4, writes, 0, nullptr);
    };
    writeComposeSet(gtComposeSet_, gtColor_.view);

    {
        VkDescriptorImageInfo info = {};
        info.sampler = linearSampler_;
        info.imageView = composeImage_.view;
        info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = copySet_;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &info;
        vkUpdateDescriptorSets(ctx_.device, 1, &write, 0, nullptr);
    }

    // SSAA downsample source (2x GT), sampled with the linear sampler: at an
    // exact 2:1 ratio bilinear weights land on a 2x2 box filter.
    if (opts_.gtSsaa) {
        VkDescriptorImageInfo info = {};
        info.sampler = linearSampler_;
        info.imageView = gtSsaaColor_.view;
        info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = gtDownsampleSet_;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &info;
        vkUpdateDescriptorSets(ctx_.device, 1, &write, 0, nullptr);
    }

    for (uint32_t i = 0; i < numAlgos; ++i) {
        AlgoColumn& algo = algos_[i];
        writeComposeSet(algo.composeSet, algo.output.view);

        VkDescriptorImageInfo testInfo = {};
        testInfo.sampler = linearSampler_;
        testInfo.imageView = algo.output.view;
        testInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo refInfo = {};
        refInfo.sampler = linearSampler_;
        refInfo.imageView = gtColor_.view;
        refInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorBufferInfo blocksInfo = {};
        blocksInfo.buffer = algo.blocksBuffer;
        blocksInfo.offset = 0;
        blocksInfo.range = VK_WHOLE_SIZE;
        VkDescriptorBufferInfo resultInfo = {};
        resultInfo.buffer = metricResultBuf_;
        resultInfo.offset = i * kMetricFloats * 4;
        resultInfo.range = kMetricFloats * 4;

        VkWriteDescriptorSet writes[4] = {};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = algo.metricSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &testInfo;
        writes[1] = writes[0];
        writes[1].dstBinding = 1;
        writes[1].pImageInfo = &refInfo;
        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = algo.metricSet;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &blocksInfo;
        writes[3] = writes[2];
        writes[3].dstBinding = 3;
        writes[3].pBufferInfo = &resultInfo;
        vkUpdateDescriptorSets(ctx_.device, 4, writes, 0, nullptr);
    }

    return true;
}

bool CompareApp::createAutoExposureResources() {
    if (!opts_.autoExposure) return true;
    // Seed both solvers with the manual/preset exposure so the frames before
    // the first readback match the old fixed-exposure look.
    const float initialEV = -std::log2(opts_.exposure);
    if (!deferred_.createExposureChannel(ctx_, descriptorPool_, gbColor_.view, renderWidth_,
                                         renderHeight_, initialEV, lrExposure_))
        return false;
    const ImageResource& gtSrc = opts_.gtSsaa ? gtSsaaColor_ : gtColor_;
    return deferred_.createExposureChannel(ctx_, descriptorPool_, gtSrc.view, gtSrc.width,
                                           gtSrc.height, initialEV, gtExposure_);
}

bool CompareApp::createCullResources() {
    // Phase 7a: shared instance SSBO + one cull channel per path (LR / GT /
    // GT-SSAA), each bound to that path's Hi-Z chain.  Capacity covers the
    // full instance list (the candidate build skips blend/skinned/culled).
    const uint32_t capacity = static_cast<uint32_t>(scene_.instances.size());
    if (!deferred_.createInstanceBuffer(ctx_, capacity, instances_)) return false;
    if (!deferred_.createCullChannel(ctx_, capacity, gbCull_)) return false;
    if (!deferred_.createCullChannel(ctx_, capacity, gtCull_)) return false;
    if (!deferred_.writeCullSet(ctx_, descriptorPool_, instances_, gbPyramid_.chainView, gbCull_))
        return false;
    if (!deferred_.writeCullSet(ctx_, descriptorPool_, instances_, gtPyramid_.chainView, gtCull_))
        return false;
    if (opts_.gtSsaa) {
        if (!deferred_.createCullChannel(ctx_, capacity, gtSsaaCull_)) return false;
        if (!deferred_.writeCullSet(ctx_, descriptorPool_, instances_, gtSsaaPyramid_.chainView,
                                    gtSsaaCull_))
            return false;
    }
    // Scene set binding 3 (instance SSBO) for every slot's GB/spatial/GT set.
    if (instances_.buffer) {
        for (uint32_t i = 0; i < kFramesInFlight; ++i) {
            deferred_.writeSceneInstanceBinding(ctx_, frames_[i].sceneSetGb, instances_.buffer);
            deferred_.writeSceneInstanceBinding(ctx_, frames_[i].sceneSetGbSpatial,
                                                instances_.buffer);
            deferred_.writeSceneInstanceBinding(ctx_, frames_[i].sceneSetGt, instances_.buffer);
        }
    }
    cullInstCpu_.resize(capacity);
    cullCmdCpu_.resize(capacity);
    return true;
}

bool CompareApp::createPipelines() {
    // The GBuffer/GT/lighting pipelines live in DeferredCore (shared with the
    // viewer); compare mode only builds compose/copy/downsample/metrics here.
    VkPushConstantRange composePushRange = {};
    composePushRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    composePushRange.offset = 0;
    composePushRange.size = sizeof(ComposePush);
    VkPipelineLayoutCreateInfo composeLayoutCi = {};
    composeLayoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    composeLayoutCi.setLayoutCount = 1;
    composeLayoutCi.pSetLayouts = &composeSetLayout_;
    composeLayoutCi.pushConstantRangeCount = 1;
    composeLayoutCi.pPushConstantRanges = &composePushRange;
    if (vkCreatePipelineLayout(ctx_.device, &composeLayoutCi, nullptr, &composePipelineLayout_) != VK_SUCCESS)
        return false;

    VkPushConstantRange copyPushRange = {};
    copyPushRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    copyPushRange.offset = 0;
    copyPushRange.size = 16; // vec4: hdr mode + paper white (copy.frag; SDR here)
    VkPipelineLayoutCreateInfo copyLayoutCi = {};
    copyLayoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    copyLayoutCi.setLayoutCount = 1;
    copyLayoutCi.pSetLayouts = &copySetLayout_;
    copyLayoutCi.pushConstantRangeCount = 1;
    copyLayoutCi.pPushConstantRanges = &copyPushRange;
    if (vkCreatePipelineLayout(ctx_.device, &copyLayoutCi, nullptr, &copyPipelineLayout_) != VK_SUCCESS)
        return false;

    VkPushConstantRange metricPushRange = {};
    metricPushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    metricPushRange.offset = 0;
    metricPushRange.size = sizeof(MetricPush);
    VkPipelineLayoutCreateInfo metricLayoutCi = {};
    metricLayoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    metricLayoutCi.setLayoutCount = 1;
    metricLayoutCi.pSetLayouts = &metricSetLayout_;
    metricLayoutCi.pushConstantRangeCount = 1;
    metricLayoutCi.pPushConstantRanges = &metricPushRange;
    if (vkCreatePipelineLayout(ctx_.device, &metricLayoutCi, nullptr, &metricPipelineLayout_) != VK_SUCCESS)
        return false;

    // --- shared graphics state for the fullscreen passes -----------------------
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

    VkPipelineColorBlendAttachmentState blendAttachments[1] = {};
    for (auto& b : blendAttachments) {
        b.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    }
    VkPipelineColorBlendStateCreateInfo colorBlend = {};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = blendAttachments;

    VkDynamicState dynamicStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState = {};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    // --- fullscreen pipelines (compose + swapchain copy) ----------------------
    VkPipelineVertexInputStateCreateInfo emptyVertexInput = {};
    emptyVertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineDepthStencilStateCreateInfo noDepth = {};
    noDepth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

    VkPipelineShaderStageCreateInfo fsStages[2] = {};
    fsStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fsStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    fsStages[0].module = fullscreenVert_;
    fsStages[0].pName = "main";
    fsStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fsStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fsStages[1].module = composeFrag_;
    fsStages[1].pName = "main";

    VkGraphicsPipelineCreateInfo fsCi = {};
    fsCi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    fsCi.stageCount = 2;
    fsCi.pStages = fsStages;
    fsCi.pVertexInputState = &emptyVertexInput;
    fsCi.pInputAssemblyState = &inputAssembly;
    fsCi.pViewportState = &viewportState;
    fsCi.pRasterizationState = &rasterizer;
    fsCi.pMultisampleState = &multisample;
    fsCi.pDepthStencilState = &noDepth;
    fsCi.pColorBlendState = &colorBlend;
    fsCi.pDynamicState = &dynamicState;

    VkPipelineRenderingCreateInfo composeRendering = {};
    composeRendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    composeRendering.colorAttachmentCount = 1;
    composeRendering.pColorAttachmentFormats = &kComposeFormat;
    fsCi.pNext = &composeRendering;
    fsCi.layout = composePipelineLayout_;
    if (createGraphicsPipeline(ctx_, fsCi, composePipeline_) != VK_SUCCESS)
        return false;

    fsStages[1].module = copyFrag_;
    VkPipelineRenderingCreateInfo copyRendering = {};
    copyRendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    copyRendering.colorAttachmentCount = 1;
    const VkFormat presentFormat = swapchain_.format();
    copyRendering.pColorAttachmentFormats = &presentFormat;
    fsCi.pNext = &copyRendering;
    fsCi.layout = copyPipelineLayout_;
    if (createGraphicsPipeline(ctx_, fsCi, copyPipeline_) != VK_SUCCESS)
        return false;

    // GT SSAA downsample: same passthrough fragment shader, HDR target.
    VkPipelineRenderingCreateInfo downsampleRendering = {};
    downsampleRendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    downsampleRendering.colorAttachmentCount = 1;
    downsampleRendering.pColorAttachmentFormats = &deferred::kHdrColorFormat;
    fsCi.pNext = &downsampleRendering;
    if (createGraphicsPipeline(ctx_, fsCi, downsamplePipeline_) != VK_SUCCESS)
        return false;

    // --- metric compute pipelines ----------------------------------------------
    VkComputePipelineCreateInfo compCi = {};
    compCi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compCi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compCi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compCi.stage.module = metricBlocksComp_;
    compCi.stage.pName = "main";
    compCi.layout = metricPipelineLayout_;
    if (createComputePipeline(ctx_, compCi, metricBlocksPipeline_) != VK_SUCCESS)
        return false;
    compCi.stage.module = metricReduceComp_;
    if (createComputePipeline(ctx_, compCi, metricReducePipeline_) != VK_SUCCESS)
        return false;

    return true;
}

bool CompareApp::createSyncResources() {
    VkCommandBufferAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = ctx_.framePool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = kFramesInFlight;
    VkCommandBuffer cmds[kFramesInFlight] = {};
    if (vkAllocateCommandBuffers(ctx_.device, &alloc, cmds) != VK_SUCCESS) return false;
    for (uint32_t i = 0; i < kFramesInFlight; ++i) frames_[i].cmd = cmds[i];

    VkSemaphoreCreateInfo semCi = {};
    semCi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fenceCi = {};
    fenceCi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCi.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    const VkDeviceSize uboSize = sizeof(SceneUBO);
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        FrameResources& fr = frames_[i];
        if (vkCreateSemaphore(ctx_.device, &semCi, nullptr, &fr.imageAvailable) != VK_SUCCESS) return false;
        if (vkCreateFence(ctx_.device, &fenceCi, nullptr, &fr.fence) != VK_SUCCESS) return false;

        // Scene UBOs per frame slot: jittered GB / unjittered spatial GB / unjittered GT.
        VkBuffer ubos[3] = {};
        VmaAllocation uboMems[3] = {};
        void* uboMaps[3] = {};
        VkDescriptorSet sets[3] = {fr.sceneSetGb, fr.sceneSetGbSpatial, fr.sceneSetGt};
        for (int k = 0; k < 3; ++k) {
            if (createBuffer(ctx_, uboSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             ubos[k], uboMems[k]) != VK_SUCCESS)
                return false;
            vmaMapMemory(ctx_.allocator, uboMems[k], &uboMaps[k]);

            VkDescriptorBufferInfo sceneBuf = {};
            sceneBuf.buffer = ubos[k];
            sceneBuf.offset = 0;
            sceneBuf.range = uboSize;
            VkDescriptorBufferInfo materialBuf = {};
            materialBuf.buffer = materialUbo_;
            materialBuf.offset = 0;
            materialBuf.range = sizeof(MaterialUBO);

            VkWriteDescriptorSet writes[2] = {};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = sets[k];
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].pBufferInfo = &sceneBuf;
            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = sets[k];
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            writes[1].pBufferInfo = &materialBuf;
            vkUpdateDescriptorSets(ctx_.device, 2, writes, 0, nullptr);

            // Joint palette for skinned draws (per-slot buffer, matching the
            // slot advanceToFrame(frameIndex) writes).
            if (scene_.hasSkinnedMeshes())
                deferred_.writeSceneSkinBinding(ctx_, sets[k], scene_.skinPalette(i));
        }
        fr.uboGb = ubos[0]; fr.uboGbMemory = uboMems[0]; fr.uboGbMapped = uboMaps[0];
        fr.uboGbSpatial = ubos[1]; fr.uboGbSpatialMemory = uboMems[1];
        fr.uboGbSpatialMapped = uboMaps[1];
        fr.uboGt = ubos[2]; fr.uboGtMemory = uboMems[2]; fr.uboGtMapped = uboMaps[2];

        // Lighting UBOs: GB (jittered invViewProj), spatial GB (unjittered),
        // GT (un-jittered; shared by the 1x and 2x SSAA GT lighting passes).
        if (createBuffer(ctx_, sizeof(LightingUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         fr.lightingUboGb, fr.lightingUboGbMemory) != VK_SUCCESS)
            return false;
        vmaMapMemory(ctx_.allocator, fr.lightingUboGbMemory,
                    &fr.lightingUboGbMapped);
        if (createBuffer(ctx_, sizeof(LightingUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         fr.lightingUboGbSpatial, fr.lightingUboGbSpatialMemory) != VK_SUCCESS)
            return false;
        vmaMapMemory(ctx_.allocator, fr.lightingUboGbSpatialMemory,
                    &fr.lightingUboGbSpatialMapped);
        if (createBuffer(ctx_, sizeof(LightingUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         fr.lightingUboGt, fr.lightingUboGtMemory) != VK_SUCCESS)
            return false;
        vmaMapMemory(ctx_.allocator, fr.lightingUboGtMemory,
                    &fr.lightingUboGtMapped);

        // Shadow map binding: the array view is always bound when the targets
        // exist; --no-shadows only zeroes shadowParams.z (sampling off) via
        // fillLightingUBO.  VK_NULL_HANDLE (creation failed) leaves binding
        // 11 unwritten, which is safe only while shadows stay off.
        const VkImageView shadowView = shadowsActive_ ? shadow_.arrayView : VK_NULL_HANDLE;
        const VkImageView spotAtlasView = spotAtlasActive_ ? spotAtlas_.view : VK_NULL_HANDLE;
        deferred_.writeLightingSet(ctx_, fr.lightingSetGb, fr.lightingUboGb, gbAlbedo_.view,
                                   gbNormal_.view, gbMaterial_.view, gbEmissive_.view,
                                   gbDepth_.view, gbAo_.view, shadowView, spotAtlasView,
                                   gbCluster_.lightsBuffer[i], gbCluster_.gridBuffer[i]);
        deferred_.writeLightingSet(ctx_, fr.lightingSetGbSpatial, fr.lightingUboGbSpatial,
                                   gbAlbedo_.view, gbNormal_.view, gbMaterial_.view,
                                   gbEmissive_.view, gbDepth_.view, gbAo_.view, shadowView,
                                   spotAtlasView,
                                   gbCluster_.lightsBuffer[i], gbCluster_.gridBuffer[i]);
        deferred_.writeLightingSet(ctx_, fr.lightingSetGt, fr.lightingUboGt, gtAlbedo_.view,
                                   gtNormal_.view, gtMaterial_.view, gtEmissive_.view,
                                   gtDepth_.view, gtAo_.view, shadowView, spotAtlasView,
                                   gtCluster_.lightsBuffer[i], gtCluster_.gridBuffer[i]);
        if (opts_.gtSsaa) {
            deferred_.writeLightingSet(ctx_, fr.lightingSetSsaa, fr.lightingUboGt,
                                       gtSsaaAlbedo_.view, gtSsaaNormal_.view,
                                       gtSsaaMaterial_.view, gtSsaaEmissive_.view,
                                       gtSsaaDepth_.view, gtSsaaAo_.view, shadowView, spotAtlasView,
                                       gtSsaaCluster_.lightsBuffer[i], gtSsaaCluster_.gridBuffer[i]);
        }

        // The transparency shader reads iblParams (identical in both lighting
        // UBOs) plus the path's own SSAO texture: one set per path.  Binding 8
        // is the path's ray-integrated froxel volume (volumetric fog on
        // translucency); the spatial LR variant shares gbFog_ (same rule as
        // the fog light sets below).
        deferred_.writeTransparentSet(ctx_, fr.transparentSetGb, fr.lightingUboGb, gbAo_.view,
                                      shadowView, gbColorPyramid_.chainView,
                                      gbPyramid_.chainView,
                                      volFogActive_ ? gbFog_.intView : VK_NULL_HANDLE);
        deferred_.writeTransparentSet(ctx_, fr.transparentSetGbSpatial, fr.lightingUboGbSpatial,
                                      gbAo_.view, shadowView, gbColorPyramid_.chainView,
                                      gbPyramid_.chainView,
                                      volFogActive_ ? gbFog_.intView : VK_NULL_HANDLE);
        deferred_.writeTransparentSet(ctx_, fr.transparentSetGt, fr.lightingUboGt, gtAo_.view,
                                      shadowView, gtColorPyramid_.chainView,
                                      gtPyramid_.chainView,
                                      volFogActive_ ? gtFog_.intView : VK_NULL_HANDLE);
        if (opts_.gtSsaa) {
            deferred_.writeTransparentSet(ctx_, fr.transparentSetSsaa, fr.lightingUboGt,
                                          gtSsaaAo_.view, shadowView,
                                          gtSsaaColorPyramid_.chainView,
                                          gtSsaaPyramid_.chainView,
                                          volFogActive_ ? gtSsaaFog_.intView : VK_NULL_HANDLE);
        }

        // Opaque-SSR trace sets: binding 0 reuses the path's lighting UBO; the
        // rest is the GBuffer + SSAO + pyramids + the path's trace target
        // (write-only storage; the temporal pass owns the lit-target RMW).
        deferred_.writeSsrSet(ctx_, fr.ssrSetGb, fr.lightingUboGb, gbAlbedo_.view,
                              gbNormal_.view, gbMaterial_.view, gbDepth_.view, gbAo_.view,
                              gbColorPyramid_.chainView, gbPyramid_.chainView, gbSsrTrace_.view);
        deferred_.writeSsrSet(ctx_, fr.ssrSetGbSpatial, fr.lightingUboGbSpatial, gbAlbedo_.view,
                              gbNormal_.view, gbMaterial_.view, gbDepth_.view, gbAo_.view,
                              gbColorPyramid_.chainView, gbPyramid_.chainView, gbSsrTrace_.view);
        deferred_.writeSsrSet(ctx_, fr.ssrSetGt, fr.lightingUboGt, gtAlbedo_.view,
                              gtNormal_.view, gtMaterial_.view, gtDepth_.view, gtAo_.view,
                              gtColorPyramid_.chainView, gtPyramid_.chainView, gtSsrTrace_.view);
        if (opts_.gtSsaa) {
            deferred_.writeSsrSet(ctx_, fr.ssrSetSsaa, fr.lightingUboGt, gtSsaaAlbedo_.view,
                                  gtSsaaNormal_.view, gtSsaaMaterial_.view, gtSsaaDepth_.view,
                                  gtSsaaAo_.view, gtSsaaColorPyramid_.chainView,
                                  gtSsaaPyramid_.chainView, gtSsaaSsrTrace_.view);
        }
    }

    // Froxel volumetric fog sets (Phase 5a), one writeVolFogSets per path.
    // The GB light sets bind the non-spatial lightingUboGb for both slots:
    // everything the fog light pass reads (sun index, shadow/atlas params,
    // cluster depth, iblParams) is identical in the spatial variant, so the
    // mixed-mode second LR record can share the accumulated volume.
    if (volFogActive_) {
        const VkImageView shadowView = shadowsActive_ ? shadow_.arrayView : VK_NULL_HANDLE;
        const VkImageView spotAtlasView = spotAtlasActive_ ? spotAtlas_.view : VK_NULL_HANDLE;
        const VkBuffer gbFogUbos[kClusterSlots] = {frames_[0].lightingUboGb,
                                                   frames_[1].lightingUboGb};
        const VkBuffer gtFogUbos[kClusterSlots] = {frames_[0].lightingUboGt,
                                                   frames_[1].lightingUboGt};
        if (!deferred_.writeVolFogSets(ctx_, descriptorPool_, gbFog_, gbCluster_, gbFogUbos,
                                       shadowView, spotAtlasView, gbDepth_.view, gbColor_.view))
            return false;
        if (!deferred_.writeVolFogSets(ctx_, descriptorPool_, gtFog_, gtCluster_, gtFogUbos,
                                       shadowView, spotAtlasView, gtDepth_.view, gtColor_.view))
            return false;
        if (opts_.gtSsaa && gtSsaaFog_.injectImage != VK_NULL_HANDLE &&
            !deferred_.writeVolFogSets(ctx_, descriptorPool_, gtSsaaFog_, gtSsaaCluster_,
                                       gtFogUbos, shadowView, spotAtlasView, gtSsaaDepth_.view,
                                       gtSsaaColor_.view))
            return false;
    }

    // One renderFinished semaphore per swapchain image (present sync).
    if (!recreateRenderFinishedSemaphores()) return false;
    return true;
}

bool CompareApp::recreateRenderFinishedSemaphores() {
    // Destroy the old set first: the swapchain may now expose a different image
    // count, so the previous array size (and its semaphores) no longer match.
    for (VkSemaphore sem : renderFinished_) {
        if (sem) vkDestroySemaphore(ctx_.device, sem, nullptr);
    }
    renderFinished_.clear();

    VkSemaphoreCreateInfo semCi = {};
    semCi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    renderFinished_.resize(swapchain_.imageCount(), VK_NULL_HANDLE);
    for (size_t i = 0; i < renderFinished_.size(); ++i) {
        if (vkCreateSemaphore(ctx_.device, &semCi, nullptr, &renderFinished_[i]) != VK_SUCCESS) return false;
    }
    return true;
}

bool CompareApp::createScreenshotStaging() {
    // Composite is RGBA8, so 4 bytes per pixel.
    const VkDeviceSize size = static_cast<VkDeviceSize>(opts_.displayWidth) * opts_.displayHeight * 4;
    if (createBuffer(ctx_, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     screenshotStaging_, screenshotStagingMemory_) != VK_SUCCESS)
        return false;
    vmaMapMemory(ctx_.allocator, screenshotStagingMemory_, &screenshotMapped_);
    return true;
}

void CompareApp::updateSceneUBO(void* mapped, bool jitter, uint32_t renderW, uint32_t renderH,
                                const Mat4& view, const Mat4& proj, const Mat4& projJittered,
                                const Mat4& prevViewProj) {
    SceneUBO ubo;
    deferred_.fillSceneUBO(ubo, scene_, camera_, view, proj, projJittered, prevViewProj,
                           renderW, renderH, jitterX_, jitterY_, jitter);
    std::memcpy(mapped, &ubo, sizeof(ubo));
}

void CompareApp::updateLightingUBO(void* mapped, const Mat4& viewProj,
                                   const ShadowFrame* shadow,
                                   const std::vector<Light>* overrideLights) {
    LightingUBO ubo;
    deferred_.fillLightingUBO(ubo, scene_, camera_, viewProj, Mat4::inverse(viewProj),
                              overrideLights, shadow, iblIntensity_);
    ubo.shadowAtlasParams[3] = opts_.contactShadows ? 1.f : 0.f;
    std::memcpy(mapped, &ubo, sizeof(ubo));
}

void CompareApp::updateClusterLights(uint32_t frameIndex, const std::vector<Light>& lights) {
    // Full point/spot set for the clustered pass; same per-slot rule as the
    // UBOs (the slot's fence passed before recording).  The list is identical
    // for every path — only the grid resolution differs.
    const uint32_t slot = frameIndex % kFramesInFlight;
    deferred_.fillClusterLights(gbCluster_.lightsMapped[slot], lights);
    deferred_.fillClusterLights(gtCluster_.lightsMapped[slot], lights);
    if (gtSsaaCluster_.lightsMapped[slot])
        deferred_.fillClusterLights(gtSsaaCluster_.lightsMapped[slot], lights);
}

void CompareApp::updateCamera(uint32_t frameIndex, float dt) {
    if (!path_.empty()) {
        const CameraKeyframe& kf = path_[frameIndex % path_.size()];
        camera_.setPose(kf.position, kf.forward, kf.up);
    } else {
        camera_.updateFreeFly(window_.input(), dt);
    }
    window_.clearMouseDelta();
}

void CompareApp::captureScreenshotIntoStaging(VkCommandBuffer cmd) {
    // composeImage_ is SHADER_READ_ONLY here (sampled by the swapchain copy).
    imageBarrier(cmd, composeImage_.image, composeLayout_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 sync::kFragment, sync::kSampled, sync::kCopy, sync::kTransferRead);
    VkBufferImageCopy region = {};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {opts_.displayWidth, opts_.displayHeight, 1};
    vkCmdCopyImageToBuffer(cmd, composeImage_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           screenshotStaging_, 1, &region);
    imageBarrier(cmd, composeImage_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 sync::kCopy, sync::kTransferRead, sync::kFragment, sync::kSampled);
    composeLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void CompareApp::saveScreenshot(const std::string& path) {
    if (!screenshotMapped_) return;
    if (!savePngFromRgba8(path.c_str(), static_cast<const uint8_t*>(screenshotMapped_),
                          opts_.displayWidth, opts_.displayHeight)) {
        std::fprintf(stderr, "failed to save screenshot %s\n", path.c_str());
    }
}

void CompareApp::harvestMetrics(uint32_t slot) {
    const auto* f = static_cast<const float*>(metricStagingMapped_[slot]);
    const uint32_t numAlgos = static_cast<uint32_t>(algos_.size());
    for (uint32_t i = 0; i < numAlgos; ++i) {
        const float* r = f + i * kMetricFloats;
        // PSNR = 10*log10(3*N / sumSqDiff), data range 1 (tonemapped domain).
        const double d2 = static_cast<double>(r[0]) + r[1] + r[2];
        const double n = static_cast<double>(r[3]);
        float psnr = 99.f; // cap for identical images (avoids inf in the overlay)
        if (n > 0.0 && d2 > 0.0) {
            psnr = static_cast<float>(10.0 * std::log10(3.0 * n / d2));
        }
        const double blocks = std::max(static_cast<double>(r[7]), 1.0);
        const float ssim =
            static_cast<float>((static_cast<double>(r[4]) + r[5] + r[6]) / (3.0 * blocks));
        algos_[i].psnr = psnr;
        algos_[i].ssim = ssim;
        algos_[i].hasMetric = true;
        if (diagMetricStdout_) {
            // The metric for frame k*metricInterval is harvested
            // kFramesInFlight frames later; k = metricHarvestCount_.
            std::fprintf(stdout, "metric: frame=%u algo=%-8s PSNR=%.2f SSIM=%.4f\n",
                         metricHarvestCount_ * static_cast<uint32_t>(opts_.metricInterval),
                         algos_[i].upscaler->name(), static_cast<double>(psnr),
                         static_cast<double>(ssim));
        }
    }
    ++metricHarvestCount_;
    refreshOverlayText();
}

void CompareApp::refreshOverlayText() {
    if (!textUboMapped_) return;
    auto* dst = static_cast<uint32_t*>(textUboMapped_);
    std::memset(dst, 0, kMaxColumns * kTextCharsPerColumn * 4);

    auto writeLine = [&](uint32_t slot, uint32_t line, const char* text) {
        uint32_t* row = dst + slot * kTextCharsPerColumn + line * 24;
        for (uint32_t c = 0; c < 24 && text[c] != '\0'; ++c) row[c] = asciiUpper(text[c]);
    };

    writeLine(0, 0, "Native (GT)");
    {
        char line[25];
        std::snprintf(line, sizeof(line), "ZOOM %.2fX", static_cast<double>(opts_.zoom));
        writeLine(0, 1, line);
        if (opts_.gtSsaa) writeLine(0, 2, "GT SSAA 2X");
    }
    for (uint32_t i = 0; i < algos_.size(); ++i) {
        const AlgoColumn& algo = algos_[i];
        const uint32_t slot = i + 1;
        char line[25];
        writeLine(slot, 0, algo.upscaler->name());
        writeLine(slot, 1, "FPS --");
        if (algo.hasMetric) {
            std::snprintf(line, sizeof(line), "PSNR %.2f dB", static_cast<double>(algo.psnr));
            writeLine(slot, 2, line);
            std::snprintf(line, sizeof(line), "SSIM %.4f", static_cast<double>(algo.ssim));
            writeLine(slot, 3, line);
        } else {
            writeLine(slot, 2, "PSNR --");
            writeLine(slot, 3, "SSIM --");
        }
    }
}

void CompareApp::recordFrame(uint32_t frameIndex, uint32_t swapchainIndex) {
    const uint32_t slot = frameIndex % kFramesInFlight;
    FrameResources& fr = frames_[slot];
    VkCommandBuffer cmd = fr.cmd;

    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    const uint32_t dw = opts_.displayWidth;
    const uint32_t dh = opts_.displayHeight;
    const float aspect = static_cast<float>(dw) / static_cast<float>(dh);
    const Mat4 view = camera_.view();
    const Mat4 proj = camera_.proj(aspect);
    Mat4 projJittered = proj;
    prevJitterX_ = jitterX_;
    prevJitterY_ = jitterY_;
    // Raster jitter cannot be undone by resampling.  Temporal plugins get a
    // Halton-jittered GBuffer; spatial plugins (FSR1 / SGSR1) get unjittered
    // LR color (a second deferred pass when mixed).
    bool hasTemporal = false;
    bool hasSpatial = false;
    for (const AlgoColumn& a : algos_) {
        if (!a.upscaler) continue;
        if (upscalerNeedsJitter(a.upscaler.get())) hasTemporal = true;
        else hasSpatial = true;
    }
    const bool temporal = hasTemporal;
    const bool mixed = hasTemporal && hasSpatial;
    const Vec2 h = halton23(frameIndex + 1);
    jitterX_ = (temporal && !diagNoJitter_) ? (h.x - 0.5f) : 0.f;
    jitterY_ = (temporal && !diagNoJitter_) ? (h.y - 0.5f) : 0.f;
    // Uniform NDC shift: clip.xy += offset * clip.w (clip.w = -z_view via
    // m[11] = -1), so the offset belongs in column 2 with a negative sign —
    // see Renderer.cpp for the full rationale.
    projJittered.m[8] -= jitterX_ * 2.f / static_cast<float>(renderWidth_);
    projJittered.m[9] -= jitterY_ * 2.f / static_cast<float>(renderHeight_);

    updateSceneUBO(fr.uboGbMapped, temporal, renderWidth_, renderHeight_, view, proj, projJittered,
                   prevViewProj_);
    const uint32_t gtW = opts_.gtSsaa ? dw * 2 : dw;
    const uint32_t gtH = opts_.gtSsaa ? dh * 2 : dh;
    updateSceneUBO(fr.uboGtMapped, false, gtW, gtH, view, proj, proj, prevViewProj_);

    // Shadows (Phase 4b): CSM sun cascades (first shadow-casting directional,
    // same selection rule as fillLightingUBO) plus the spot shadow atlas —
    // shadow-casting spots are scored by intensity/distance^2 and the top
    // kShadowAtlasTiles get a tile (selectSpotShadowLights writes shadowIndex
    // into a lights copy that overrides both the UBO and the cluster SSBO).
    // The GT and SSAA paths sample the same maps — GT is the same lighting at
    // native res.  --no-shadows leaves shadow null (shadowParams.z = 0).
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

    // Lighting reconstructs world positions with the inverse of the exact
    // view-projection used for each pass (jittered GB / unjittered spatial GB /
    // un-jittered GT; the GT matrix is resolution-independent and shared by
    // the 1x and 2x SSAA pass).
    updateLightingUBO(fr.lightingUboGbMapped, Mat4::multiply(projJittered, view), shadow,
                      lightsOverride);
    updateLightingUBO(fr.lightingUboGtMapped, Mat4::multiply(proj, view), shadow,
                      lightsOverride);
    updateClusterLights(frameIndex, DeferredCore::effectiveLights(scene_, lightsOverride));
    if (mixed) {
        updateSceneUBO(fr.uboGbSpatialMapped, false, renderWidth_, renderHeight_, view, proj, proj,
                       prevViewProj_);
        updateLightingUBO(fr.lightingUboGbSpatialMapped, Mat4::multiply(proj, view),
                          shadow, lightsOverride);
    }

    auto transition = [&](VkImage image, VkImageLayout& current, VkImageLayout target,
                          VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                          VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
                          VkImageAspectFlags aspect) {
        imageBarrier(cmd, image, current, target, srcStage, srcAccess, dstStage, dstAccess, aspect);
        current = target;
    };
    const Mat4 cullViewProj = Mat4::multiply(proj, view); // un-jittered (sub-pixel)

    // --- Phase 7a: candidate build + staged instance/command data -------------
    // One CPU build (frustum + LOD, path-independent); each path's channel
    // gets a copy of the same command list and zeroes its own occluded
    // instanceCounts against its own Hi-Z chain.
    cullCandidates_ =
        deferred_.buildInstanceList(scene_, cullViewProj, instances_.capacity,
                                    cullInstCpu_.data(), cullCmdCpu_.data(), cullRuns_);
    if (cullCandidates_ > 0) {
        std::memcpy(instances_.stagingMapped[slot], cullInstCpu_.data(),
                    static_cast<size_t>(cullCandidates_) * sizeof(GpuInstance));
        const size_t cmdBytes =
            static_cast<size_t>(cullCandidates_) * sizeof(VkDrawIndexedIndirectCommand);
        std::memcpy(gbCull_.cmdStagingMapped[slot], cullCmdCpu_.data(), cmdBytes);
        std::memcpy(gtCull_.cmdStagingMapped[slot], cullCmdCpu_.data(), cmdBytes);
        if (opts_.gtSsaa)
            std::memcpy(gtSsaaCull_.cmdStagingMapped[slot], cullCmdCpu_.data(), cmdBytes);
    }
    deferred_.recordInstanceUpload(cmd, slot, instances_, cullCandidates_);

    auto recordPostFx = [&](PostFxTargets& fxTargets, VkImage color, VkImageLayout& colorLayout,
                            uint32_t pathH) {
        // Phase 6b: same algorithm + parameters on every path; the blur clamps
        // scale with path height so the display-space blur matches the GT.
        PostFxParams fx;
        fx.depthM10 = proj.m[10];
        fx.depthM14 = proj.m[14];
        fx.farPlane = camera_.farPlane;
        const float resScale = static_cast<float>(pathH) / 1080.f;
        fx.maxBlurPx = std::max(8.f, kMotionBlurMaxPixels * resScale);
        fx.maxCocPx = std::max(2.f, opts_.dofMaxBlurPx * resScale);
        fx.aperture = kDofAperture * (kDofDefaultFstop / opts_.dofFstop);
        fx.focusDistance = opts_.dofFocus;
        fx.motionBlur = opts_.motionBlur;
        fx.dof = opts_.dof;
        deferred_.recordPostFxPass(cmd, fxTargets, color, colorLayout, fx, frameIndex);
    };

    auto recordLrGBuffer = [&](VkDescriptorSet sceneSet) {
        // Phase 7a: upload this frame's commands and run the occlusion cull
        // against the LR Hi-Z chain as it currently stands (previous frame's
        // pyramid — or, in mixed mode's second record, this frame's spatial
        // rebuild; gbCull_.prevViewProj always tracks the producing VP).
        const bool cullActive = occlusion_ && gbCull_.prevValid && cullCandidates_ > 0;
        deferred_.recordCommandUpload(cmd, slot, gbCull_, cullCandidates_, cullActive);
        if (cullActive)
            deferred_.recordOcclusionCull(cmd, gbCull_, cullCandidates_, gbCull_.prevViewProj,
                                          gbPyramid_.mipCount, renderWidth_, renderHeight_);
        // GBuffer targets: sampled by the lighting fragment shader (and the
        // upscalers) during the previous use -> attachment writes.
        transition(gbAlbedo_.image, gbAlbedoLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbNormal_.image, gbNormalLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbMaterial_.image, gbMaterialLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbEmissive_.image, gbEmissiveLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbMotion_.image, gbMotionLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbDepth_.image, gbDepthLayout_, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kDepthTests, sync::kDepthReadWrite,
                   VK_IMAGE_ASPECT_DEPTH_BIT);
        {
            VkRenderingAttachmentInfo colors[5] = {
                makeColorAttachment(gbAlbedo_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gbNormal_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR, 0.f, 0.f, 1.f),
                makeColorAttachment(gbMaterial_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gbEmissive_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gbMotion_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR)};
            VkRenderingAttachmentInfo depth =
                makeDepthAttachment(gbDepth_.view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR);
            beginRendering(cmd, renderWidth_, renderHeight_, 5, colors, &depth);
            deferred_.recordGBufferDraws(cmd, scene_, false, sceneSet, textureSet_, materialStride_,
                                         renderWidth_, renderHeight_, cullViewProj, gbCull_,
                                         cullRuns_.data(), static_cast<uint32_t>(cullRuns_.size()));
            vkCmdEndRendering(cmd);
        }
    };

    auto recordLrLighting = [&](VkDescriptorSet sceneSet, VkDescriptorSet lightingSet,
                                VkDescriptorSet transparentSet, VkDescriptorSet ssrSet,
                                const Mat4& ssaoViewProj, const Mat4& projUsed) {
        // dst scope includes compute: the opaque-SSR pass (ssr_opaque.comp)
        // samples the GBuffer after the lighting fragment shader.
        transition(gbAlbedo_.image, gbAlbedoLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbNormal_.image, gbNormalLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbMaterial_.image, gbMaterialLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbEmissive_.image, gbEmissiveLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbMotion_.image, gbMotionLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbDepth_.image, gbDepthLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kDepthTests, sync::kDepthWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_DEPTH_BIT);
        // Hi-Z pyramid for the SSR marchers (glass transparency + opaque-SSR
        // compute; both LR lighting variants share the same GBuffer depth) and
        // the occlusion cull (Phase 7a).  ssaoViewProj is the exact
        // view-projection of the GBuffer record that fed this depth — the cull
        // channel reprojects with it from the next record on.
        if (hasTransparency_ || opts_.ssr || occlusion_) {
            deferred_.recordDepthPyramidPass(cmd, gbPyramid_);
            gbCull_.prevViewProj = ssaoViewProj;
            gbCull_.prevValid = true;
        }

        // GTAO: view-Z depth chain (sampled at per-step LODs) -> main pass ->
        // temporal EMA -> denoise.  The chain is rebuilt every record.
        deferred_.recordDepthPyramidPass(cmd, gbPyramidAo_);
        const Mat4 invAoVp = Mat4::inverse(ssaoViewProj);
        const float aoMaxLodGb = static_cast<float>(std::min(gbPyramidAo_.mipCount - 1, 4u));
        transition(gbAoRaw_.image, gbAoRawLayout_, VK_IMAGE_LAYOUT_GENERAL, sync::kCompute,
                   sync::kSampled, sync::kCompute, sync::kStorageWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordSsaoPass(cmd, ssaoSetGb_, invAoVp, frameIndex, camera_.nearPlane,
                                 camera_.farPlane, aoMaxLodGb, renderWidth_, renderHeight_);
        transition(gbAoRaw_.image, gbAoRawLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kCompute, sync::kStorageWrite, sync::kCompute, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        const uint32_t aoWriteGb = aoFramesGb_ & 1u;
        deferred_.recordSsaoTemporalPass(cmd, gbAoHist_, aoWriteGb, invAoVp, prevAoViewProjGb_,
                                         renderWidth_, renderHeight_, /*reset=*/aoFramesGb_ == 0);
        prevAoViewProjGb_ = ssaoViewProj;
        ++aoFramesGb_;
        transition(gbAo_.image, gbAoLayout_, VK_IMAGE_LAYOUT_GENERAL, sync::kSampleStages,
                   sync::kSampled, sync::kCompute, sync::kStorageWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordSsaoBlurPass(cmd, gbAoHist_.blurSet[aoWriteGb], renderWidth_,
                                     renderHeight_);
        transition(gbAo_.image, gbAoLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kCompute, sync::kStorageWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);

        transition(gbColor_.image, gbColorLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordLightingPass(cmd, lightingSet, gbCluster_, frameIndex % kFramesInFlight,
                                     view, projUsed, gbColor_.view, renderWidth_, renderHeight_);

        // Froxel volumetric fog (Phase 5a): accumulate once per frame (mixed
        // mode records LR lighting twice; the spatial record wins the guard),
        // composite into the lit HDR target on every record.  Un-jittered
        // matrices so the volume does not swim under TAA jitter.
        if (volFogActive_) {
            if (fogAccumFrameGb_ != frameIndex) {
                deferred_.recordVolFogAccumulate(cmd, gbFog_, gbCluster_,
                                                 frameIndex % kFramesInFlight, view, proj,
                                                 prevFogViewProjGb_, fogParams_, frameIndex,
                                                 fogFramesGb_ & 1u, /*reset=*/fogFramesGb_ == 0);
                prevFogViewProjGb_ = cullViewProj;
                ++fogFramesGb_;
                fogAccumFrameGb_ = frameIndex;
            }
            transition(gbColor_.image, gbColorLayout_, VK_IMAGE_LAYOUT_GENERAL,
                       sync::kColorAttach, sync::kColorWrite, sync::kCompute,
                       sync::kStorageReadWrite, VK_IMAGE_ASPECT_COLOR_BIT);
            deferred_.recordVolFogComposite(cmd, gbFog_, proj, fogParams_.maxDistance,
                                            renderWidth_, renderHeight_);
            transition(gbColor_.image, gbColorLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                       sync::kCompute, sync::kStorageWrite, sync::kColorAttach,
                       sync::kColorReadWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        }

        if (hasTransparency_ || opts_.ssr) {
            // Color mip chain: mip 0 is the opaque HDR copy glass SSR used to
            // make with a transfer; the chain stays GENERAL for life.  The
            // opaque-SSR pass consumes the same chain, so it is built whenever
            // either consumer runs.
            transition(gbColor_.image, gbColorLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       sync::kColorAttach, sync::kColorWrite, sync::kCompute, sync::kSampled,
                       VK_IMAGE_ASPECT_COLOR_BIT);
            deferred_.recordColorPyramidPass(cmd, gbColorPyramid_);
            if (opts_.ssr) {
                // Opaque SSR, Phase 2d: trace into the LR RT, then temporal
                // EMA + fused composite (in-place RMW on gbColor_).
                transition(gbSsrTrace_.image, gbSsrTraceLayout_, VK_IMAGE_LAYOUT_GENERAL,
                           sync::kCompute, sync::kSampled, sync::kCompute, sync::kStorageWrite,
                           VK_IMAGE_ASPECT_COLOR_BIT);
                deferred_.recordSsrPass(cmd, ssrSet, ssaoViewProj, renderWidth_, renderHeight_,
                                      opts_.ssrStrength);
                transition(gbSsrTrace_.image, gbSsrTraceLayout_,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kCompute,
                           sync::kStorageWrite, sync::kCompute, sync::kSampled,
                           VK_IMAGE_ASPECT_COLOR_BIT);
                transition(gbColor_.image, gbColorLayout_, VK_IMAGE_LAYOUT_GENERAL,
                           sync::kCompute, sync::kSampled, sync::kCompute,
                           sync::kStorageReadWrite, VK_IMAGE_ASPECT_COLOR_BIT);
                deferred_.recordSsrTemporalPass(cmd, gbSsrHist_, ssrFramesGb_ & 1u, invAoVp,
                                                prevSsrViewProjGb_, renderWidth_, renderHeight_,
                                                /*reset=*/ssrFramesGb_ == 0);
                prevSsrViewProjGb_ = ssaoViewProj;
                ++ssrFramesGb_;
                transition(gbColor_.image, gbColorLayout_,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kCompute,
                           sync::kStorageWrite, sync::kColorAttach, sync::kColorReadWrite,
                           VK_IMAGE_ASPECT_COLOR_BIT);
            } else {
                transition(gbColor_.image, gbColorLayout_,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kCompute,
                           sync::kSampled, sync::kColorAttach, sync::kColorReadWrite,
                           VK_IMAGE_ASPECT_COLOR_BIT);
            }
        }

        // Overwrites motion (static glass = camera motion) and accumulates the
        // translucent coverage mask consumed by upscalers as the reactive / TC mask.
        if (hasTransparency_) {
            transition(gbMotion_.image, gbMotionLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                       sync::kFragment, sync::kSampled, sync::kColorAttach, sync::kColorReadWrite,
                       VK_IMAGE_ASPECT_COLOR_BIT);
            transition(gbReactive_.image, gbReactiveLayout_,
                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kSampleStages,
                       sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                       VK_IMAGE_ASPECT_COLOR_BIT);
            transition(gbDepth_.image, gbDepthLayout_,
                       VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, sync::kFragment,
                       sync::kSampled, sync::kDepthTests, sync::kDepthRead,
                       VK_IMAGE_ASPECT_DEPTH_BIT);
            {
                VkRenderingAttachmentInfo tColors[3] = {
                    makeColorAttachment(gbColor_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                        VK_ATTACHMENT_LOAD_OP_LOAD),
                    makeColorAttachment(gbMotion_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                        VK_ATTACHMENT_LOAD_OP_LOAD),
                    makeColorAttachment(gbReactive_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                        VK_ATTACHMENT_LOAD_OP_CLEAR)};
                VkRenderingAttachmentInfo tDepth =
                    makeDepthAttachment(gbDepth_.view,
                                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                                        VK_ATTACHMENT_LOAD_OP_LOAD);
                beginRendering(cmd, renderWidth_, renderHeight_, 3, tColors, &tDepth);
                deferred_.recordTransparentDraws(cmd, scene_, false, sceneSet, textureSet_,
                                                 transparentSet, materialStride_, renderWidth_,
                                                 renderHeight_, cullViewProj, camera_.position,
                                                 proj.m[14] / proj.m[10], fogParams_.maxDistance,
                                                 volFogActive_);
                vkCmdEndRendering(cmd);
            }
            transition(gbMotion_.image, gbMotionLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                       VK_IMAGE_ASPECT_COLOR_BIT);
            transition(gbReactive_.image, gbReactiveLayout_,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kColorAttach,
                       sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                       VK_IMAGE_ASPECT_COLOR_BIT);
            transition(gbDepth_.image, gbDepthLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       sync::kDepthTests, sync::kDepthRead, sync::kSampleStages, sync::kSampled,
                       VK_IMAGE_ASPECT_DEPTH_BIT);
        }

        transition(gbColor_.image, gbColorLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        recordPostFx(gbPostFx_, gbColor_.image, gbColorLayout_, renderHeight_);
    };

    // --- 1) low-resolution deferred (spatial copy then jittered when mixed) -----
    if (mixed) {
        recordLrGBuffer(fr.sceneSetGbSpatial);
    } else {
        recordLrGBuffer(fr.sceneSetGb);
    }

    // --- Shadow pass (sun CSM, one 2048^2 layer per cascade) -------------------
    // Runs once per frame and feeds the LR, GT and SSAA lighting paths alike.
    if (shadow && shadow->lightIndex >= 0) {
        imageBarrier(cmd, shadow_.image, shadow_.layout,
                     VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, sync::kFragment,
                     sync::kSampled, sync::kDepthTests, sync::kDepthReadWrite,
                     VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, kShadowCascadeCount);
        shadow_.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        deferred_.recordShadowPass(cmd, shadow_, scene_, shadow->cascadeVp, fr.sceneSetGb,
                                   textureSet_, materialStride_);
        imageBarrier(cmd, shadow_.image, shadow_.layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     sync::kDepthTests, sync::kDepthWrite, sync::kFragment, sync::kSampled,
                     VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, kShadowCascadeCount);
        shadow_.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    // --- Spot shadow atlas (Phase 4b, one 1024^2 tile per selected spot) -----
    // Same once-per-frame sharing across the LR/GT/SSAA paths as the CSM pass.
    if (shadow && shadow->atlasTileCount > 0) {
        imageBarrier(cmd, spotAtlas_.image, spotAtlas_.layout,
                     VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, sync::kFragment,
                     sync::kSampled, sync::kDepthTests, sync::kDepthReadWrite,
                     VK_IMAGE_ASPECT_DEPTH_BIT);
        spotAtlas_.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        deferred_.recordSpotShadowPass(cmd, spotAtlas_, scene_, shadow->atlasVp,
                                       shadow->atlasTileCount, fr.sceneSetGb, textureSet_,
                                       materialStride_);
        imageBarrier(cmd, spotAtlas_.image, spotAtlas_.layout,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kDepthTests, sync::kDepthWrite,
                     sync::kFragment, sync::kSampled, VK_IMAGE_ASPECT_DEPTH_BIT);
        spotAtlas_.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    if (mixed) {
        recordLrLighting(fr.sceneSetGbSpatial, fr.lightingSetGbSpatial, fr.transparentSetGbSpatial,
                         fr.ssrSetGbSpatial, cullViewProj, proj);
        transition(gbColor_.image, gbColorLayout_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kCopy, sync::kTransferRead,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbColorSpatial_.image, gbColorSpatialLayout_,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, sync::kSampleStages, sync::kSampled,
                   sync::kCopy, sync::kTransferWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        {
            VkImageCopy region = {};
            region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.srcSubresource.layerCount = 1;
            region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.dstSubresource.layerCount = 1;
            region.extent = {renderWidth_, renderHeight_, 1};
            vkCmdCopyImage(cmd, gbColor_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           gbColorSpatial_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        }
        transition(gbColorSpatial_.image, gbColorSpatialLayout_,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kCopy, sync::kTransferWrite,
                   sync::kSampleStages, sync::kSampled, VK_IMAGE_ASPECT_COLOR_BIT);
        recordLrGBuffer(fr.sceneSetGb);
        recordLrLighting(fr.sceneSetGb, fr.lightingSetGb, fr.transparentSetGb, fr.ssrSetGb,
                         Mat4::multiply(projJittered, view), projJittered);
    } else {
        recordLrLighting(fr.sceneSetGb, fr.lightingSetGb, fr.transparentSetGb, fr.ssrSetGb,
                         Mat4::multiply(projJittered, view), projJittered);
    }

    // --- Auto exposure: LR histogram of this frame's lit HDR (gbColor_) --------
    // Feeds the upscaler preExposure + the algorithm columns.  The applied
    // value is the solve harvested kFramesInFlight frames ago (engine-style
    // latency; deterministic under a fixed camera path).  gbColor_ is
    // SHADER_READ_ONLY here with compute in the sampled-dst scope.
    if (opts_.autoExposure) {
        ExposureSolvePush solve;
        solve.minEV = opts_.exposureMinEV;
        solve.maxEV = opts_.exposureMaxEV;
        solve.resetState = (frameIndex == 0) ? 1.f : 0.f; // snap on the first frame
        deferred_.recordAutoExposurePass(cmd, lrExposure_.gpu, solve,
                                         lrExposure_.staging[slot]);
        lrExposure_.pending[slot] = true;
    }

    // --- 2) per-algorithm dispatch into its own output texture -----------------
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
    frame.deltaTime = 1.f / 60.f;
    // Real exposure input (was hardcoded 1): the LR path's display exposure —
    // see InputAdapter.h for the preExposure convention.
    frame.preExposure = lrExposure();
    frame.resetHistory = (frameIndex == 0);

    for (AlgoColumn& algo : algos_) {
        transition(algo.output.image, algo.outputLayout, VK_IMAGE_LAYOUT_GENERAL,
                   sync::kSampleStages, sync::kSampled, sync::kSampleStages, sync::kStorageWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);

        CameraParams algoCam = cam;
        UpscalerResources res;
        if (mixed && !upscalerNeedsJitter(algo.upscaler.get())) {
            res.color = gbColorSpatial_.image;
            res.colorView = gbColorSpatial_.view;
            algoCam.jitterX = 0.f;
            algoCam.jitterY = 0.f;
            algoCam.prevJitterX = 0.f;
            algoCam.prevJitterY = 0.f;
        } else {
            res.color = gbColor_.image;
            res.colorView = gbColor_.view;
        }
        res.depth = gbDepth_.image;
        res.depthView = gbDepth_.view;
        res.motion = gbMotion_.image;
        res.motionView = gbMotion_.view;
        if (hasTransparency_) {
            res.reactive = gbReactive_.image;
            res.reactiveView = gbReactive_.view;
        }
        res.output = algo.output.image;
        res.outputView = algo.output.view;
        algo.upscaler->dispatch(cmd, res, algoCam, frame);

        transition(algo.output.image, algo.outputLayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kSampleStages, sync::kStorageWrite, sync::kFragment, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
    }

    // --- 3) native-resolution ground truth (same camera, no jitter) -------------
    if (opts_.gtSsaa) {
        // 200% SSAA: deferred render at 2x into gtSsaaColor_, then
        // box-downsample to display resolution (gtColor_), which stays the
        // metric/display ref.
        const uint32_t sw = dw * 2;
        const uint32_t sh = dh * 2;
        transition(gtSsaaAlbedo_.image, gtSsaaAlbedoLayout_,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kSampleStages, sync::kSampled,
                   sync::kColorAttach, sync::kColorWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaNormal_.image, gtSsaaNormalLayout_,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kSampleStages, sync::kSampled,
                   sync::kColorAttach, sync::kColorWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaMaterial_.image, gtSsaaMaterialLayout_,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kSampleStages, sync::kSampled,
                   sync::kColorAttach, sync::kColorWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaEmissive_.image, gtSsaaEmissiveLayout_,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kSampleStages, sync::kSampled,
                   sync::kColorAttach, sync::kColorWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaMotion_.image, gtSsaaMotionLayout_,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kSampleStages, sync::kSampled,
                   sync::kColorAttach, sync::kColorWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaDepth_.image, gtSsaaDepthLayout_,
                   VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, sync::kSampleStages,
                   sync::kSampled, sync::kDepthTests, sync::kDepthReadWrite,
                   VK_IMAGE_ASPECT_DEPTH_BIT);
        {
            // Phase 7a: SSAA-path occlusion cull against the 2x Hi-Z chain.
            const bool cullActive = occlusion_ && gtSsaaCull_.prevValid && cullCandidates_ > 0;
            deferred_.recordCommandUpload(cmd, slot, gtSsaaCull_, cullCandidates_, cullActive);
            if (cullActive)
                deferred_.recordOcclusionCull(cmd, gtSsaaCull_, cullCandidates_,
                                              gtSsaaCull_.prevViewProj, gtSsaaPyramid_.mipCount,
                                              sw, sh);
            VkRenderingAttachmentInfo colors[5] = {
                makeColorAttachment(gtSsaaAlbedo_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gtSsaaNormal_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR, 0.f, 0.f, 1.f),
                makeColorAttachment(gtSsaaMaterial_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gtSsaaEmissive_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gtSsaaMotion_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR)};
            VkRenderingAttachmentInfo depth =
                makeDepthAttachment(gtSsaaDepth_.view,
                                    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR);
            beginRendering(cmd, sw, sh, 5, colors, &depth);
            deferred_.recordGBufferDraws(cmd, scene_, true, fr.sceneSetGt, textureSet_,
                                         materialStride_, sw, sh, cullViewProj, gtSsaaCull_,
                                         cullRuns_.data(), static_cast<uint32_t>(cullRuns_.size()));
            vkCmdEndRendering(cmd);
        }
        // dst scope includes compute: the opaque-SSR pass samples the GBuffer.
        transition(gtSsaaAlbedo_.image, gtSsaaAlbedoLayout_,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kColorAttach, sync::kColorWrite,
                   sync::kSampleStages, sync::kSampled, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaNormal_.image, gtSsaaNormalLayout_,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kColorAttach, sync::kColorWrite,
                   sync::kSampleStages, sync::kSampled, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaMaterial_.image, gtSsaaMaterialLayout_,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kColorAttach, sync::kColorWrite,
                   sync::kSampleStages, sync::kSampled, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaEmissive_.image, gtSsaaEmissiveLayout_,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kColorAttach, sync::kColorWrite,
                   sync::kSampleStages, sync::kSampled, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaMotion_.image, gtSsaaMotionLayout_,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kColorAttach, sync::kColorWrite,
                   sync::kSampleStages, sync::kSampled, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaDepth_.image, gtSsaaDepthLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kDepthTests, sync::kDepthWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_DEPTH_BIT);
        if (hasTransparency_ || opts_.ssr || occlusion_) {
            deferred_.recordDepthPyramidPass(cmd, gtSsaaPyramid_);
            gtSsaaCull_.prevViewProj = cullViewProj; // GT paths never jitter
            gtSsaaCull_.prevValid = true;
        }

        // GTAO for the 2x GT path (un-jittered view-projection): view-Z depth
        // chain -> main pass -> temporal EMA -> denoise.  The AO output is
        // also sampled by the opaque-SSR compute pass.
        deferred_.recordDepthPyramidPass(cmd, gtSsaaPyramidAo_);
        const Mat4 invAoVpSsaa = Mat4::inverse(cullViewProj);
        const float aoMaxLodSsaa =
            static_cast<float>(std::min(gtSsaaPyramidAo_.mipCount - 1, 4u));
        transition(gtSsaaAoRaw_.image, gtSsaaAoRawLayout_, VK_IMAGE_LAYOUT_GENERAL,
                   sync::kCompute, sync::kSampled, sync::kCompute, sync::kStorageWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordSsaoPass(cmd, ssaoSetSsaa_, invAoVpSsaa, frameIndex, camera_.nearPlane,
                                 camera_.farPlane, aoMaxLodSsaa, sw, sh);
        transition(gtSsaaAoRaw_.image, gtSsaaAoRawLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kCompute, sync::kStorageWrite, sync::kCompute, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        const uint32_t aoWriteSsaa = aoFramesSsaa_ & 1u;
        deferred_.recordSsaoTemporalPass(cmd, gtSsaaAoHist_, aoWriteSsaa, invAoVpSsaa,
                                         prevAoViewProjSsaa_, sw, sh,
                                         /*reset=*/aoFramesSsaa_ == 0);
        prevAoViewProjSsaa_ = cullViewProj;
        ++aoFramesSsaa_;
        transition(gtSsaaAo_.image, gtSsaaAoLayout_, VK_IMAGE_LAYOUT_GENERAL, sync::kSampleStages,
                   sync::kSampled, sync::kCompute, sync::kStorageWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordSsaoBlurPass(cmd, gtSsaaAoHist_.blurSet[aoWriteSsaa], sw, sh);
        transition(gtSsaaAo_.image, gtSsaaAoLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kCompute, sync::kStorageWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);

        transition(gtSsaaColor_.image, gtSsaaColorLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordLightingPass(cmd, fr.lightingSetSsaa, gtSsaaCluster_,
                                     frameIndex % kFramesInFlight, view, proj,
                                     gtSsaaColor_.view, sw, sh);

        // Froxel volumetric fog (Phase 5a) on the 2x GT path.
        if (volFogActive_ && gtSsaaFog_.injectImage != VK_NULL_HANDLE) {
            if (fogAccumFrameSsaa_ != frameIndex) {
                deferred_.recordVolFogAccumulate(cmd, gtSsaaFog_, gtSsaaCluster_,
                                                 frameIndex % kFramesInFlight, view, proj,
                                                 prevFogViewProjSsaa_, fogParams_, frameIndex,
                                                 fogFramesSsaa_ & 1u,
                                                 /*reset=*/fogFramesSsaa_ == 0);
                prevFogViewProjSsaa_ = cullViewProj;
                ++fogFramesSsaa_;
                fogAccumFrameSsaa_ = frameIndex;
            }
            transition(gtSsaaColor_.image, gtSsaaColorLayout_, VK_IMAGE_LAYOUT_GENERAL,
                       sync::kColorAttach, sync::kColorWrite, sync::kCompute,
                       sync::kStorageReadWrite, VK_IMAGE_ASPECT_COLOR_BIT);
            deferred_.recordVolFogComposite(cmd, gtSsaaFog_, proj, fogParams_.maxDistance,
                                            sw, sh);
            transition(gtSsaaColor_.image, gtSsaaColorLayout_,
                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kCompute,
                       sync::kStorageWrite, sync::kColorAttach, sync::kColorReadWrite,
                       VK_IMAGE_ASPECT_COLOR_BIT);
        }

        if (hasTransparency_ || opts_.ssr) {
            // Same color mip chain build as the GB path (GENERAL for life);
            // the opaque-SSR pass consumes the same chain.
            transition(gtSsaaColor_.image, gtSsaaColorLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       sync::kColorAttach, sync::kColorWrite, sync::kCompute, sync::kSampled,
                       VK_IMAGE_ASPECT_COLOR_BIT);
            deferred_.recordColorPyramidPass(cmd, gtSsaaColorPyramid_);
            if (opts_.ssr) {
                // Opaque SSR at 2x before the box downsample: trace into the
                // SSAA RT, then temporal EMA + fused composite on gtSsaaColor_.
                transition(gtSsaaSsrTrace_.image, gtSsaaSsrTraceLayout_, VK_IMAGE_LAYOUT_GENERAL,
                           sync::kCompute, sync::kSampled, sync::kCompute,
                           sync::kStorageWrite, VK_IMAGE_ASPECT_COLOR_BIT);
                deferred_.recordSsrPass(cmd, fr.ssrSetSsaa, cullViewProj, sw, sh,
                                      opts_.ssrStrength);
                transition(gtSsaaSsrTrace_.image, gtSsaaSsrTraceLayout_,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kCompute,
                           sync::kStorageWrite, sync::kCompute, sync::kSampled,
                           VK_IMAGE_ASPECT_COLOR_BIT);
                transition(gtSsaaColor_.image, gtSsaaColorLayout_, VK_IMAGE_LAYOUT_GENERAL,
                           sync::kCompute, sync::kSampled, sync::kCompute,
                           sync::kStorageReadWrite, VK_IMAGE_ASPECT_COLOR_BIT);
                deferred_.recordSsrTemporalPass(cmd, gtSsaaSsrHist_, ssrFramesSsaa_ & 1u,
                                                invAoVpSsaa, prevSsrViewProjSsaa_, sw, sh,
                                                /*reset=*/ssrFramesSsaa_ == 0);
                prevSsrViewProjSsaa_ = cullViewProj;
                ++ssrFramesSsaa_;
                transition(gtSsaaColor_.image, gtSsaaColorLayout_,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kCompute,
                           sync::kStorageWrite, sync::kColorAttach, sync::kColorReadWrite,
                           VK_IMAGE_ASPECT_COLOR_BIT);
            } else {
                transition(gtSsaaColor_.image, gtSsaaColorLayout_,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kCompute,
                           sync::kSampled, sync::kColorAttach, sync::kColorReadWrite,
                           VK_IMAGE_ASPECT_COLOR_BIT);
            }
        }

        // Transparency pass: alpha-blended surfaces over the lit scene (GT
        // path: color only, no motion/mask outputs).
        if (hasTransparency_) {
            transition(gtSsaaDepth_.image, gtSsaaDepthLayout_,
                       VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, sync::kFragment,
                       sync::kSampled, sync::kDepthTests, sync::kDepthRead,
                       VK_IMAGE_ASPECT_DEPTH_BIT);
            {
                VkRenderingAttachmentInfo tColor =
                    makeColorAttachment(gtSsaaColor_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                        VK_ATTACHMENT_LOAD_OP_LOAD);
                VkRenderingAttachmentInfo tDepth =
                    makeDepthAttachment(gtSsaaDepth_.view,
                                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                                        VK_ATTACHMENT_LOAD_OP_LOAD);
                beginRendering(cmd, sw, sh, 1, &tColor, &tDepth);
                deferred_.recordTransparentDraws(cmd, scene_, true, fr.sceneSetGt, textureSet_,
                                                 fr.transparentSetSsaa, materialStride_, sw, sh,
                                                 cullViewProj, camera_.position,
                                                 proj.m[14] / proj.m[10], fogParams_.maxDistance,
                                                 volFogActive_ &&
                                                     gtSsaaFog_.injectImage != VK_NULL_HANDLE);
                vkCmdEndRendering(cmd);
            }
            transition(gtSsaaDepth_.image, gtSsaaDepthLayout_,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kDepthTests,
                       sync::kDepthRead, sync::kFragment, sync::kSampled, VK_IMAGE_ASPECT_DEPTH_BIT);
        }

        transition(gtSsaaColor_.image, gtSsaaColorLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);

        // Phase 6b: MB/DOF in the 2x domain, before the box downsample.
        recordPostFx(gtSsaaPostFx_, gtSsaaColor_.image, gtSsaaColorLayout_, sh);

        transition(gtColor_.image, gtColorLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        {
            VkRenderingAttachmentInfo color =
                makeColorAttachment(gtColor_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_DONT_CARE);
            beginRendering(cmd, dw, dh, 1, &color, nullptr);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, downsamplePipeline_);
            VkViewport viewport = {0.f, 0.f, static_cast<float>(dw), static_cast<float>(dh), 0.f,
                                   1.f};
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            VkRect2D scissor = {{0, 0}, {dw, dh}};
            vkCmdSetScissor(cmd, 0, 1, &scissor);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, copyPipelineLayout_, 0, 1,
                                    &gtDownsampleSet_, 0, nullptr);
            // Mode 0: passthrough — the downsample target is HDR linear, not
            // the swapchain (copy.frag HDR branch is for presentation only).
            const float copyPush[4] = {0.f, 0.f, 0.f, 0.f};
            vkCmdPushConstants(cmd, copyPipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof(copyPush), copyPush);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            vkCmdEndRendering(cmd);
        }
        transition(gtColor_.image, gtColorLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
    } else {
        transition(gtAlbedo_.image, gtAlbedoLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtNormal_.image, gtNormalLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtMaterial_.image, gtMaterialLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtEmissive_.image, gtEmissiveLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtMotion_.image, gtMotionLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtDepth_.image, gtDepthLayout_, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kDepthTests, sync::kDepthReadWrite,
                   VK_IMAGE_ASPECT_DEPTH_BIT);
        {
            // Phase 7a: GT-path occlusion cull against the 1x Hi-Z chain.
            const bool cullActive = occlusion_ && gtCull_.prevValid && cullCandidates_ > 0;
            deferred_.recordCommandUpload(cmd, slot, gtCull_, cullCandidates_, cullActive);
            if (cullActive)
                deferred_.recordOcclusionCull(cmd, gtCull_, cullCandidates_, gtCull_.prevViewProj,
                                              gtPyramid_.mipCount, dw, dh);
            VkRenderingAttachmentInfo colors[5] = {
                makeColorAttachment(gtAlbedo_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gtNormal_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR, 0.f, 0.f, 1.f),
                makeColorAttachment(gtMaterial_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gtEmissive_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gtMotion_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR)};
            VkRenderingAttachmentInfo depth =
                makeDepthAttachment(gtDepth_.view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR);
            beginRendering(cmd, dw, dh, 5, colors, &depth);
            deferred_.recordGBufferDraws(cmd, scene_, true, fr.sceneSetGt, textureSet_,
                                         materialStride_, dw, dh, cullViewProj, gtCull_,
                                         cullRuns_.data(), static_cast<uint32_t>(cullRuns_.size()));
            vkCmdEndRendering(cmd);
        }
        // dst scope includes compute: the opaque-SSR pass samples the GBuffer.
        transition(gtAlbedo_.image, gtAlbedoLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtNormal_.image, gtNormalLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtMaterial_.image, gtMaterialLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtEmissive_.image, gtEmissiveLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtMotion_.image, gtMotionLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtDepth_.image, gtDepthLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kDepthTests, sync::kDepthWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_DEPTH_BIT);
        if (hasTransparency_ || opts_.ssr || occlusion_) {
            deferred_.recordDepthPyramidPass(cmd, gtPyramid_);
            gtCull_.prevViewProj = cullViewProj; // GT paths never jitter
            gtCull_.prevValid = true;
        }

        // GTAO for the 1x GT path (un-jittered view-projection): view-Z depth
        // chain -> main pass -> temporal EMA -> denoise.  The AO output is
        // also sampled by the opaque-SSR compute pass.
        deferred_.recordDepthPyramidPass(cmd, gtPyramidAo_);
        const Mat4 invAoVpGt = Mat4::inverse(cullViewProj);
        const float aoMaxLodGt = static_cast<float>(std::min(gtPyramidAo_.mipCount - 1, 4u));
        transition(gtAoRaw_.image, gtAoRawLayout_, VK_IMAGE_LAYOUT_GENERAL, sync::kCompute,
                   sync::kSampled, sync::kCompute, sync::kStorageWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordSsaoPass(cmd, ssaoSetGt_, invAoVpGt, frameIndex, camera_.nearPlane,
                                 camera_.farPlane, aoMaxLodGt, dw, dh);
        transition(gtAoRaw_.image, gtAoRawLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kCompute, sync::kStorageWrite, sync::kCompute, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        const uint32_t aoWriteGt = aoFramesGt_ & 1u;
        deferred_.recordSsaoTemporalPass(cmd, gtAoHist_, aoWriteGt, invAoVpGt, prevAoViewProjGt_,
                                         dw, dh, /*reset=*/aoFramesGt_ == 0);
        prevAoViewProjGt_ = cullViewProj;
        ++aoFramesGt_;
        transition(gtAo_.image, gtAoLayout_, VK_IMAGE_LAYOUT_GENERAL, sync::kSampleStages,
                   sync::kSampled, sync::kCompute, sync::kStorageWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordSsaoBlurPass(cmd, gtAoHist_.blurSet[aoWriteGt], dw, dh);
        transition(gtAo_.image, gtAoLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kCompute, sync::kStorageWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);

        transition(gtColor_.image, gtColorLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordLightingPass(cmd, fr.lightingSetGt, gtCluster_,
                                     frameIndex % kFramesInFlight, view, proj,
                                     gtColor_.view, dw, dh);

        // Froxel volumetric fog (Phase 5a) on the 1x GT path.
        if (volFogActive_) {
            if (fogAccumFrameGt_ != frameIndex) {
                deferred_.recordVolFogAccumulate(cmd, gtFog_, gtCluster_,
                                                 frameIndex % kFramesInFlight, view, proj,
                                                 prevFogViewProjGt_, fogParams_, frameIndex,
                                                 fogFramesGt_ & 1u, /*reset=*/fogFramesGt_ == 0);
                prevFogViewProjGt_ = cullViewProj;
                ++fogFramesGt_;
                fogAccumFrameGt_ = frameIndex;
            }
            transition(gtColor_.image, gtColorLayout_, VK_IMAGE_LAYOUT_GENERAL,
                       sync::kColorAttach, sync::kColorWrite, sync::kCompute,
                       sync::kStorageReadWrite, VK_IMAGE_ASPECT_COLOR_BIT);
            deferred_.recordVolFogComposite(cmd, gtFog_, proj, fogParams_.maxDistance, dw, dh);
            transition(gtColor_.image, gtColorLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                       sync::kCompute, sync::kStorageWrite, sync::kColorAttach,
                       sync::kColorReadWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        }

        if (hasTransparency_ || opts_.ssr) {
            // Same color mip chain build as the GB path (GENERAL for life);
            // the opaque-SSR pass consumes the same chain.
            transition(gtColor_.image, gtColorLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       sync::kColorAttach, sync::kColorWrite, sync::kCompute, sync::kSampled,
                       VK_IMAGE_ASPECT_COLOR_BIT);
            deferred_.recordColorPyramidPass(cmd, gtColorPyramid_);
            if (opts_.ssr) {
                // Opaque SSR, Phase 2d: trace into the GT RT, then temporal
                // EMA + fused composite (in-place RMW on gtColor_).
                transition(gtSsrTrace_.image, gtSsrTraceLayout_, VK_IMAGE_LAYOUT_GENERAL,
                           sync::kCompute, sync::kSampled, sync::kCompute,
                           sync::kStorageWrite, VK_IMAGE_ASPECT_COLOR_BIT);
                deferred_.recordSsrPass(cmd, fr.ssrSetGt, cullViewProj, dw, dh,
                                      opts_.ssrStrength);
                transition(gtSsrTrace_.image, gtSsrTraceLayout_,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kCompute,
                           sync::kStorageWrite, sync::kCompute, sync::kSampled,
                           VK_IMAGE_ASPECT_COLOR_BIT);
                transition(gtColor_.image, gtColorLayout_, VK_IMAGE_LAYOUT_GENERAL,
                           sync::kCompute, sync::kSampled, sync::kCompute,
                           sync::kStorageReadWrite, VK_IMAGE_ASPECT_COLOR_BIT);
                deferred_.recordSsrTemporalPass(cmd, gtSsrHist_, ssrFramesGt_ & 1u, invAoVpGt,
                                                prevSsrViewProjGt_, dw, dh,
                                                /*reset=*/ssrFramesGt_ == 0);
                prevSsrViewProjGt_ = cullViewProj;
                ++ssrFramesGt_;
                transition(gtColor_.image, gtColorLayout_,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kCompute,
                           sync::kStorageWrite, sync::kColorAttach, sync::kColorReadWrite,
                           VK_IMAGE_ASPECT_COLOR_BIT);
            } else {
                transition(gtColor_.image, gtColorLayout_,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kCompute,
                           sync::kSampled, sync::kColorAttach, sync::kColorReadWrite,
                           VK_IMAGE_ASPECT_COLOR_BIT);
            }
        }

        // Transparency pass: alpha-blended surfaces over the lit scene (GT
        // path: color only, no motion/mask outputs).
        if (hasTransparency_) {
            transition(gtDepth_.image, gtDepthLayout_,
                       VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, sync::kFragment,
                       sync::kSampled, sync::kDepthTests, sync::kDepthRead,
                       VK_IMAGE_ASPECT_DEPTH_BIT);
            {
                VkRenderingAttachmentInfo tColor =
                    makeColorAttachment(gtColor_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                        VK_ATTACHMENT_LOAD_OP_LOAD);
                VkRenderingAttachmentInfo tDepth =
                    makeDepthAttachment(gtDepth_.view,
                                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                                        VK_ATTACHMENT_LOAD_OP_LOAD);
                beginRendering(cmd, dw, dh, 1, &tColor, &tDepth);
                deferred_.recordTransparentDraws(cmd, scene_, true, fr.sceneSetGt, textureSet_,
                                                 fr.transparentSetGt, materialStride_, dw, dh,
                                                 cullViewProj, camera_.position,
                                                 proj.m[14] / proj.m[10], fogParams_.maxDistance,
                                                 volFogActive_);
                vkCmdEndRendering(cmd);
            }
            transition(gtDepth_.image, gtDepthLayout_,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kDepthTests,
                       sync::kDepthRead, sync::kFragment, sync::kSampled, VK_IMAGE_ASPECT_DEPTH_BIT);
        }

        transition(gtColor_.image, gtColorLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        // Phase 6b: MB/DOF on the 1x GT HDR.
        recordPostFx(gtPostFx_, gtColor_.image, gtColorLayout_, dh);
    }

    // --- Auto exposure: GT histogram (own HDR source; gtSsaa uses its 2x HDR) --
    // The GT column gets its own solver, independent of the LR one — both are
    // separate render pipelines, so differing column exposures are the correct
    // engine behaviour.  Sources are SHADER_READ_ONLY here (SSAA's
    // gtSsaaColor_ stays sampled after the box downsample).
    if (opts_.autoExposure) {
        ExposureSolvePush solve;
        solve.minEV = opts_.exposureMinEV;
        solve.maxEV = opts_.exposureMaxEV;
        solve.resetState = (frameIndex == 0) ? 1.f : 0.f;
        deferred_.recordAutoExposurePass(cmd, gtExposure_.gpu, solve,
                                         gtExposure_.staging[slot]);
        gtExposure_.pending[slot] = true;
    }

    // --- 4) GPU metric reduction (every metricInterval frames) -------------------
    // Metric region: full image at zoom 1; the zoomed view window otherwise.
    const uint32_t numColumns = 1 + static_cast<uint32_t>(algos_.size());
    const uint32_t colW = dw / numColumns;
    uint32_t regX = 0, regY = 0, regW = dw, regH = dh;
    if (opts_.zoom > 1.f) {
        float rect[4];
        computeViewRegion(dw, dh, colW, dh, opts_.zoom, opts_.zoomCenterU, opts_.zoomCenterV,
                          rect);
        regX = std::min(static_cast<uint32_t>(rect[0]), dw - 1);
        regY = std::min(static_cast<uint32_t>(rect[1]), dh - 1);
        regW = std::max(1u, std::min(static_cast<uint32_t>(rect[2]), dw - regX));
        regH = std::max(1u, std::min(static_cast<uint32_t>(rect[3]), dh - regY));
    }
    const uint32_t regBlocksPerRow = (regW + 7) / 8;
    const uint32_t regBlockCount = regBlocksPerRow * ((regH + 7) / 8);
    if (frameIndex % static_cast<uint32_t>(opts_.metricInterval) == 0) {
        for (AlgoColumn& algo : algos_) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, metricBlocksPipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, metricPipelineLayout_, 0,
                                    1, &algo.metricSet, 0, nullptr);
            MetricPush push;
            push.x = regX;
            push.y = regY;
            push.z = regW;
            push.w = regH;
            push.x2 = regBlocksPerRow;
            // Each image is tonemapped with its own path's exposure (test =
            // LR auto exposure, ref = GT auto exposure): with auto exposure
            // on, the PSNR/SSIM numbers include the exposure difference the
            // user actually sees between the columns.
            push.exposureTest = lrExposure();
            push.exposureRef = gtExposure();
            vkCmdPushConstants(cmd, metricPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(push), &push);
            vkCmdDispatch(cmd, regBlocksPerRow, (regH + 7) / 8, 1);

            computeBarrier(cmd, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, metricReducePipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, metricPipelineLayout_, 0,
                                    1, &algo.metricSet, 0, nullptr);
            MetricPush reducePush;
            reducePush.x = regBlockCount;
            vkCmdPushConstants(cmd, metricPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(reducePush), &reducePush);
            vkCmdDispatch(cmd, 1, 1, 1);

            computeBarrier(cmd, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
        computeBarrier(cmd, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferCopy copyRegion = {};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = static_cast<VkDeviceSize>(algos_.size()) * kMetricFloats * 4;
        vkCmdCopyBuffer(cmd, metricResultBuf_, metricStaging_[slot], 1, &copyRegion);
        metricPending_[slot] = true;
    }

    // --- 5) compose columns + overlay into the offscreen composite ---------------
    transition(composeImage_.image, composeLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
               sync::kFragment, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
               VK_IMAGE_ASPECT_COLOR_BIT);
    {
        VkRenderingAttachmentInfo color =
            makeColorAttachment(composeImage_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_ATTACHMENT_LOAD_OP_CLEAR);
        beginRendering(cmd, dw, dh, 1, &color, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, composePipeline_);

        const float textScale = colW >= 300 ? 2.f : 1.f;
        for (uint32_t i = 0; i < numColumns; ++i) {
            const uint32_t x = i * colW;
            const uint32_t w = (i == numColumns - 1) ? (dw - x) : colW;
            VkViewport viewport = {static_cast<float>(x), 0.f, static_cast<float>(w),
                                   static_cast<float>(dh), 0.f, 1.f};
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            VkRect2D scissor = {{static_cast<int32_t>(x), 0}, {w, dh}};
            vkCmdSetScissor(cmd, 0, 1, &scissor);

            const VkDescriptorSet set = (i == 0) ? gtComposeSet_ : algos_[i - 1].composeSet;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, composePipelineLayout_,
                                    0, 1, &set, 0, nullptr);
            ComposePush push;
            push.colSize[0] = static_cast<float>(w);
            push.colSize[1] = static_cast<float>(dh);
            push.textScale = textScale;
            push.textSlot = static_cast<float>(i);
            // Aspect-preserving crop + zoom window over the display-res source.
            float rect[4];
            computeViewRegion(dw, dh, w, dh, opts_.zoom, opts_.zoomCenterU, opts_.zoomCenterV,
                              rect);
            push.uvRect[0] = rect[0] / static_cast<float>(dw);
            push.uvRect[1] = rect[1] / static_cast<float>(dh);
            push.uvRect[2] = rect[2] / static_cast<float>(dw);
            push.uvRect[3] = rect[3] / static_cast<float>(dh);
            push.srcSize[0] = static_cast<float>(dw);
            push.srcSize[1] = static_cast<float>(dh);
            // Nearest sampling only once the on-screen magnification passes
            // 1:1 (strictly): at exactly 1:1 linear sampling hits texel
            // centers anyway, and the lens chain stays active.  Pixel-peep
            // (>1:1) shows raw texels and skips the lens chain.
            push.nearest = (static_cast<float>(w) > rect[2]) ? 1.f : 0.f;
            // Per-column exposure: the GT column uses the GT path's solver,
            // algorithm columns the LR path's (auto mode; manual mode shares
            // opts_.exposure everywhere).
            push.exposure = (i == 0) ? gtExposure() : lrExposure();
            // Terminal lens chain: identical parameters for every column —
            // each column is an independent present of its own path.
            push.lensA[0] = opts_.lensFx ? kLensCaStrength : 0.f;
            push.lensA[1] = opts_.lensFx ? kLensVignetteStrength : 0.f;
            push.lensA[2] = opts_.lensFx ? kLensGrainStrength : 0.f;
            push.lensA[3] = static_cast<float>(frameIndex);
            // Grading (Phase 6c): neutral set, same for every column (compare
            // output stays SDR; HDR presentation is viewer/GUI-only).
            const Vec3 wb = whiteBalanceForTemperatureTint(grading_.temperatureK, grading_.tint);
            push.gradeA[0] = grading_.contrast;
            push.gradeA[1] = grading_.saturation;
            push.gradeA[2] = 0.f;
            push.gradeA[3] = 0.f;
            push.gradeB[0] = wb.x;
            push.gradeB[1] = wb.y;
            push.gradeB[2] = wb.z;
            push.gradeB[3] = 17.f; // makeIdentityLut() default edge length
            vkCmdPushConstants(cmd, composePipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof(push), &push);
            vkCmdDraw(cmd, 3, 1, 0, 0);
        }
        vkCmdEndRendering(cmd);
    }
    transition(composeImage_.image, composeLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled,
               VK_IMAGE_ASPECT_COLOR_BIT);

    // --- 6) copy the composite to the swapchain -----------------------------------
    const VkImage swapImage = swapchain_.image(swapchainIndex);
    const VkImageView swapView = swapchain_.view(swapchainIndex);
    imageBarrier(cmd, swapImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 sync::kColorAttach, sync::kColorWrite, sync::kColorAttach, sync::kColorWrite);
    {
        VkRenderingAttachmentInfo color =
            makeColorAttachment(swapView, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_ATTACHMENT_LOAD_OP_CLEAR);
        beginRendering(cmd, swapchain_.extent().width, swapchain_.extent().height, 1, &color,
                       nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, copyPipeline_);
        VkViewport viewport = {0.f, 0.f, static_cast<float>(swapchain_.extent().width),
                               static_cast<float>(swapchain_.extent().height), 0.f, 1.f};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        VkRect2D scissor = {{0, 0}, {swapchain_.extent().width, swapchain_.extent().height}};
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, copyPipelineLayout_, 0, 1,
                                &copySet_, 0, nullptr);
        // Compare presentation is SDR only (Phase 6c: HDR is viewer/GUI-only).
        const float copyPush[4] = {0.f, 0.f, 0.f, 0.f};
        vkCmdPushConstants(cmd, copyPipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(copyPush), copyPush);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);
    }
    imageBarrier(cmd, swapImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 sync::kColorAttach, sync::kColorWrite, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE);

    const bool capturing = !opts_.screenshotPath.empty() &&
                           static_cast<int>(frameIndex) == opts_.frames - 1;
    if (capturing) {
        captureScreenshotIntoStaging(cmd);
    }

    vkEndCommandBuffer(cmd);

    prevViewProj_ = Mat4::multiply(proj, view);
}

void CompareApp::run() {
    uint32_t frameIndex = 0;
    auto lastTime = std::chrono::steady_clock::now();

    while (true) {
        if (!window_.poll()) break;
        if (opts_.frames >= 0 && static_cast<int>(frameIndex) >= opts_.frames) break;

        const auto now = std::chrono::steady_clock::now();
        const float dt = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;
        updateCamera(frameIndex, dt);

        const uint32_t slot = frameIndex % kFramesInFlight;

        // Ensure the previous work using this frame slot is complete before
        // reusing its fence / image-available semaphore / UBOs / staging. The
        // fence is reset only after a successful acquire: an OUT_OF_DATE /
        // SUBOPTIMAL acquire leaves it signaled so the next iteration's wait
        // returns immediately instead of blocking forever.
        vkWaitForFences(ctx_.device, 1, &frames_[slot].fence, VK_TRUE, UINT64_MAX);

        // Frame-index driven scene animation (dynamic boxes / glTF clips).
        // After the slot fence: the palette/instance state of this slot is no
        // longer read by the GPU.  No-op for static scenes.
        scene_.advanceToFrame(frameIndex);
        // CPU LOD selection for this frame's camera.  One decision per frame,
        // shared by the GT and upscaled paths, so both sides of the
        // comparison always draw the same levels (fair + deterministic).
        scene_.updateLodSelection(camera_.position, camera_.fovY, lodEnabledByDefault());
        if (metricPending_[slot]) {
            harvestMetrics(slot);
            metricPending_[slot] = false;
        }
        // Same completion event: harvest the auto-exposure solves recorded in
        // that frame (deterministic per frame index: fixed slot schedule).
        deferred_.harvestExposureChannel(lrExposure_, slot);
        deferred_.harvestExposureChannel(gtExposure_, slot);

        uint32_t swapIndex = 0;
        VkResult acq = swapchain_.acquireNext(ctx_, frames_[slot].imageAvailable, swapIndex);
        if (acq == VK_ERROR_OUT_OF_DATE_KHR || acq == VK_SUBOPTIMAL_KHR) {
            // The surface size changed. Wait for in-flight work, then rebuild
            // the swapchain and the per-image present semaphores for the new
            // image count (the old count is not guaranteed to match).
            vkDeviceWaitIdle(ctx_.device);
            if (!swapchain_.create(ctx_, opts_.displayWidth, opts_.displayHeight, opts_.vsync)) break;
            if (!recreateRenderFinishedSemaphores()) break;
            continue;
        }
        if (acq != VK_SUCCESS) break;

        vkResetFences(ctx_.device, 1, &frames_[slot].fence);

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
        if (vkQueueSubmit(ctx_.graphicsQueue, 1, &submit, frames_[slot].fence) != VK_SUCCESS) break;

        VkResult pres = swapchain_.present(ctx_, swapIndex, renderFinished_[swapIndex]);
        if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR) {
            // Present reported the swapchain is no longer optimal. The queued
            // present may still reference the old semaphores, so wait idle
            // before destroying them and rebuilding for the new image count.
            vkDeviceWaitIdle(ctx_.device);
            if (!swapchain_.create(ctx_, opts_.displayWidth, opts_.displayHeight, opts_.vsync)) break;
            if (!recreateRenderFinishedSemaphores()) break;
        } else if (pres != VK_SUCCESS) {
            break;
        }

        if (!opts_.screenshotPath.empty() && static_cast<int>(frameIndex) == opts_.frames - 1) {
            vkWaitForFences(ctx_.device, 1, &frames_[slot].fence, VK_TRUE, UINT64_MAX);
            saveScreenshot(opts_.screenshotPath);
        }

        ++frameIndex;
    }

    vkDeviceWaitIdle(ctx_.device);

    // Drain metric readbacks still in flight so the final summary is current.
    for (uint32_t s = 0; s < kFramesInFlight; ++s) {
        if (metricPending_[s]) {
            harvestMetrics(s);
            metricPending_[s] = false;
        }
    }
    for (const AlgoColumn& algo : algos_) {
        if (algo.hasMetric) {
            std::fprintf(stdout, "compare: %-8s PSNR=%.2f dB SSIM=%.4f\n", algo.upscaler->name(),
                         static_cast<double>(algo.psnr), static_cast<double>(algo.ssim));
        } else {
            std::fprintf(stdout, "compare: %-8s (no metric harvested)\n", algo.upscaler->name());
        }
    }
    if (opts_.frames >= 0) {
        std::fprintf(stdout, "frames=%u done\n", frameIndex);
    }
}

void CompareApp::shutdown() {
    if (!ctx_.device) return;
    vkDeviceWaitIdle(ctx_.device);

    for (AlgoColumn& algo : algos_) {
        if (algo.upscaler) { algo.upscaler->shutdown(); algo.upscaler.reset(); }
        algo.output.destroy(ctx_);
        if (algo.blocksBuffer) {
            vmaDestroyBuffer(ctx_.allocator, algo.blocksBuffer, algo.blocksMemory);
            algo.blocksBuffer = VK_NULL_HANDLE;
            algo.blocksMemory = VK_NULL_HANDLE;
        }
    }
    algos_.clear();

    if (screenshotStaging_) {
        vmaDestroyBuffer(ctx_.allocator, screenshotStaging_, screenshotStagingMemory_);
        screenshotStaging_ = VK_NULL_HANDLE;
        screenshotStagingMemory_ = VK_NULL_HANDLE;
    }

    deferred_.destroyExposureChannel(ctx_, lrExposure_);
    deferred_.destroyExposureChannel(ctx_, gtExposure_);
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        if (metricStaging_[i]) {
            vmaDestroyBuffer(ctx_.allocator, metricStaging_[i], metricStagingMemory_[i]);
            metricStaging_[i] = VK_NULL_HANDLE;
            metricStagingMemory_[i] = VK_NULL_HANDLE;
            metricStagingMapped_[i] = nullptr;
        }
    }
    if (metricResultBuf_) {
        vmaDestroyBuffer(ctx_.allocator, metricResultBuf_, metricResultMemory_);
        metricResultBuf_ = VK_NULL_HANDLE;
        metricResultMemory_ = VK_NULL_HANDLE;
    }
    if (textUbo_) {
        vmaDestroyBuffer(ctx_.allocator, textUbo_, textUboMemory_);
        textUbo_ = VK_NULL_HANDLE;
        textUboMemory_ = VK_NULL_HANDLE;
        textUboMapped_ = nullptr;
    }

    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        FrameResources& fr = frames_[i];
        if (fr.uboGb) {
            vmaDestroyBuffer(ctx_.allocator, fr.uboGb, fr.uboGbMemory);
            fr.uboGb = VK_NULL_HANDLE;
            fr.uboGbMemory = VK_NULL_HANDLE;
        }
        if (fr.uboGbSpatial) {
            vmaDestroyBuffer(ctx_.allocator, fr.uboGbSpatial, fr.uboGbSpatialMemory);
            fr.uboGbSpatial = VK_NULL_HANDLE;
            fr.uboGbSpatialMemory = VK_NULL_HANDLE;
            fr.uboGbSpatialMapped = nullptr;
        }
        if (fr.uboGt) {
            vmaDestroyBuffer(ctx_.allocator, fr.uboGt, fr.uboGtMemory);
            fr.uboGt = VK_NULL_HANDLE;
            fr.uboGtMemory = VK_NULL_HANDLE;
        }
        if (fr.lightingUboGb) {
            vmaDestroyBuffer(ctx_.allocator, fr.lightingUboGb, fr.lightingUboGbMemory);
            fr.lightingUboGb = VK_NULL_HANDLE;
            fr.lightingUboGbMemory = VK_NULL_HANDLE;
        }
        if (fr.lightingUboGbSpatial) {
            vmaDestroyBuffer(ctx_.allocator, fr.lightingUboGbSpatial, fr.lightingUboGbSpatialMemory);
            fr.lightingUboGbSpatial = VK_NULL_HANDLE;
            fr.lightingUboGbSpatialMemory = VK_NULL_HANDLE;
            fr.lightingUboGbSpatialMapped = nullptr;
        }
        if (fr.lightingUboGt) {
            vmaDestroyBuffer(ctx_.allocator, fr.lightingUboGt, fr.lightingUboGtMemory);
            fr.lightingUboGt = VK_NULL_HANDLE;
            fr.lightingUboGtMemory = VK_NULL_HANDLE;
        }
        if (fr.imageAvailable) { vkDestroySemaphore(ctx_.device, fr.imageAvailable, nullptr); fr.imageAvailable = VK_NULL_HANDLE; }
        if (fr.fence) { vkDestroyFence(ctx_.device, fr.fence, nullptr); fr.fence = VK_NULL_HANDLE; }
    }
    for (VkSemaphore sem : renderFinished_) {
        if (sem) vkDestroySemaphore(ctx_.device, sem, nullptr);
    }
    renderFinished_.clear();

    if (composePipeline_) { vkDestroyPipeline(ctx_.device, composePipeline_, nullptr); composePipeline_ = VK_NULL_HANDLE; }
    if (copyPipeline_) { vkDestroyPipeline(ctx_.device, copyPipeline_, nullptr); copyPipeline_ = VK_NULL_HANDLE; }
    if (downsamplePipeline_) { vkDestroyPipeline(ctx_.device, downsamplePipeline_, nullptr); downsamplePipeline_ = VK_NULL_HANDLE; }
    if (metricBlocksPipeline_) { vkDestroyPipeline(ctx_.device, metricBlocksPipeline_, nullptr); metricBlocksPipeline_ = VK_NULL_HANDLE; }
    if (metricReducePipeline_) { vkDestroyPipeline(ctx_.device, metricReducePipeline_, nullptr); metricReducePipeline_ = VK_NULL_HANDLE; }
    if (composePipelineLayout_) { vkDestroyPipelineLayout(ctx_.device, composePipelineLayout_, nullptr); composePipelineLayout_ = VK_NULL_HANDLE; }
    if (copyPipelineLayout_) { vkDestroyPipelineLayout(ctx_.device, copyPipelineLayout_, nullptr); copyPipelineLayout_ = VK_NULL_HANDLE; }
    if (metricPipelineLayout_) { vkDestroyPipelineLayout(ctx_.device, metricPipelineLayout_, nullptr); metricPipelineLayout_ = VK_NULL_HANDLE; }
    if (fullscreenVert_) { vkDestroyShaderModule(ctx_.device, fullscreenVert_, nullptr); fullscreenVert_ = VK_NULL_HANDLE; }
    if (composeFrag_) { vkDestroyShaderModule(ctx_.device, composeFrag_, nullptr); composeFrag_ = VK_NULL_HANDLE; }
    if (copyFrag_) { vkDestroyShaderModule(ctx_.device, copyFrag_, nullptr); copyFrag_ = VK_NULL_HANDLE; }
    if (metricBlocksComp_) { vkDestroyShaderModule(ctx_.device, metricBlocksComp_, nullptr); metricBlocksComp_ = VK_NULL_HANDLE; }
    if (metricReduceComp_) { vkDestroyShaderModule(ctx_.device, metricReduceComp_, nullptr); metricReduceComp_ = VK_NULL_HANDLE; }

    if (descriptorPool_) { vkDestroyDescriptorPool(ctx_.device, descriptorPool_, nullptr); descriptorPool_ = VK_NULL_HANDLE; }
    if (composeSetLayout_) { vkDestroyDescriptorSetLayout(ctx_.device, composeSetLayout_, nullptr); composeSetLayout_ = VK_NULL_HANDLE; }
    if (copySetLayout_) { vkDestroyDescriptorSetLayout(ctx_.device, copySetLayout_, nullptr); copySetLayout_ = VK_NULL_HANDLE; }
    if (metricSetLayout_) { vkDestroyDescriptorSetLayout(ctx_.device, metricSetLayout_, nullptr); metricSetLayout_ = VK_NULL_HANDLE; }
    if (linearSampler_) { vkDestroySampler(ctx_.device, linearSampler_, nullptr); linearSampler_ = VK_NULL_HANDLE; }
    if (fontSampler_) { vkDestroySampler(ctx_.device, fontSampler_, nullptr); fontSampler_ = VK_NULL_HANDLE; }

    if (materialUbo_) {
        vmaDestroyBuffer(ctx_.allocator, materialUbo_, materialUboMemory_);
        materialUbo_ = VK_NULL_HANDLE;
        materialUboMemory_ = VK_NULL_HANDLE;
    }

    gbColor_.destroy(ctx_);
    gbColorSpatial_.destroy(ctx_);
    deferred_.destroyColorPyramid(ctx_, gbColorPyramid_);
    deferred_.destroyColorPyramid(ctx_, gtColorPyramid_);
    deferred_.destroyColorPyramid(ctx_, gtSsaaColorPyramid_);
    deferred_.destroyClusterGrid(ctx_, gbCluster_);
    deferred_.destroyClusterGrid(ctx_, gtCluster_);
    deferred_.destroyClusterGrid(ctx_, gtSsaaCluster_);
    deferred_.destroyVolFogVolume(ctx_, gbFog_);
    deferred_.destroyVolFogVolume(ctx_, gtFog_);
    deferred_.destroyVolFogVolume(ctx_, gtSsaaFog_);
    deferred_.destroyDepthPyramid(ctx_, gbPyramid_);
    deferred_.destroyDepthPyramid(ctx_, gtPyramid_);
    deferred_.destroyDepthPyramid(ctx_, gtSsaaPyramid_);
    deferred_.destroyCullChannel(ctx_, gbCull_);
    deferred_.destroyCullChannel(ctx_, gtCull_);
    deferred_.destroyCullChannel(ctx_, gtSsaaCull_);
    deferred_.destroyInstanceBuffer(ctx_, instances_);
    deferred_.destroyDepthPyramid(ctx_, gbPyramidAo_);
    deferred_.destroyDepthPyramid(ctx_, gtPyramidAo_);
    deferred_.destroyDepthPyramid(ctx_, gtSsaaPyramidAo_);
    deferred_.destroyAoHistory(ctx_, gbAoHist_);
    deferred_.destroyAoHistory(ctx_, gtAoHist_);
    deferred_.destroyAoHistory(ctx_, gtSsaaAoHist_);
    deferred_.destroySsrHistory(ctx_, gbSsrHist_);
    deferred_.destroySsrHistory(ctx_, gtSsrHist_);
    deferred_.destroySsrHistory(ctx_, gtSsaaSsrHist_);
    gbSsrTrace_.destroy(ctx_);
    gtSsrTrace_.destroy(ctx_);
    gtSsaaSsrTrace_.destroy(ctx_);
    gbAlbedo_.destroy(ctx_);
    gbNormal_.destroy(ctx_);
    gbMaterial_.destroy(ctx_);
    gbEmissive_.destroy(ctx_);
    gbMotion_.destroy(ctx_);
    gbReactive_.destroy(ctx_);
    gbDepth_.destroy(ctx_);
    gtColor_.destroy(ctx_);
    gtAlbedo_.destroy(ctx_);
    gtNormal_.destroy(ctx_);
    gtMaterial_.destroy(ctx_);
    gtEmissive_.destroy(ctx_);
    gtMotion_.destroy(ctx_);
    gtDepth_.destroy(ctx_);
    gtSsaaColor_.destroy(ctx_);
    gtSsaaAlbedo_.destroy(ctx_);
    gtSsaaNormal_.destroy(ctx_);
    gtSsaaMaterial_.destroy(ctx_);
    gtSsaaEmissive_.destroy(ctx_);
    gtSsaaMotion_.destroy(ctx_);
    gtSsaaDepth_.destroy(ctx_);
    deferred_.destroyPostFxTargets(ctx_, gbPostFx_);
    deferred_.destroyPostFxTargets(ctx_, gtPostFx_);
    deferred_.destroyPostFxTargets(ctx_, gtSsaaPostFx_);
    gbAoRaw_.destroy(ctx_);
    gbAo_.destroy(ctx_);
    gtAoRaw_.destroy(ctx_);
    gtAo_.destroy(ctx_);
    gtSsaaAoRaw_.destroy(ctx_);
    gtSsaaAo_.destroy(ctx_);
    composeImage_.destroy(ctx_);
    fontAtlas_.destroy(ctx_);
    gradingLut_.destroy(ctx_);

    if (shadowsActive_) { deferred_.destroyShadowTargets(ctx_, shadow_); shadowsActive_ = false; }
    if (spotAtlasActive_) { deferred_.destroyShadowAtlas(ctx_, spotAtlas_); spotAtlasActive_ = false; }
    deferred_.destroy(ctx_);
    scene_.destroy(ctx_);
    swapchain_.destroy(ctx_);
    ctx_.destroy();
    window_.destroy();
}

} // namespace sr
