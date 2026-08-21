#include "compare/CompareApp.h"

#include "app/CliUtils.h"

#include "compare/Font5x7.h"
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
// the source-region window (normalized offset/size) and source dimensions.
struct ComposePush {
    float colSize[2];
    float textScale;
    float textSlot;
    float uvRect[4];  // normalized source region: offset xy, size zw
    float srcSize[2]; // source image pixels
    float nearest;    // != 0: sample nearest (magnification >= 1:1)
    float exposure;   // display-domain ACES input multiplier
};
static_assert(sizeof(ComposePush) == 48, "ComposePush size mismatch");

// Metric compute push constants (two uvec4s in the shaders).
struct MetricPush {
    uint32_t x = 0, y = 0, z = 0, w = 0;      // region offset xy, extent zw
    uint32_t x2 = 0, y2 = 0, z2 = 0, w2 = 0;  // x2 = blocks per row (region)
    float exposure = 1.f;
    float pad[3] = {};
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

    if (!window_.create("sr_compare — compare", static_cast<int>(opts.displayWidth),
                        static_cast<int>(opts.displayHeight)))
        return false;
    if (!ctx_.create(window_)) return false;
    if (!swapchain_.create(ctx_, opts.displayWidth, opts.displayHeight, opts.vsync)) return false;

    renderWidth_ = std::max(1u, static_cast<uint32_t>(static_cast<float>(opts_.displayWidth) * opts_.renderScale));
    renderHeight_ = std::max(1u, static_cast<uint32_t>(static_cast<float>(opts_.displayHeight) * opts_.renderScale));

    bool sceneOk = false;
    if (!opts.scenePath.empty()) sceneOk = scene_.loadGltf(ctx_, opts.scenePath.c_str());
    if (!sceneOk) sceneOk = scene_.loadProcedural(ctx_);
    if (!sceneOk) return false;
    hasTransparency_ = deferred_.sceneHasTransparency(scene_);
    iblIntensity_ = lightingPresetForScene(opts.scenePath).iblIntensity;

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
    // Shared deferred pipeline: IBL maps + shaders + layouts + pipelines.
    if (!deferred_.init(ctx_, opts_.envMapPath.c_str())) return false;
    // CSM shadow targets (fixed size; a failure degrades to no shadows).
    // Created before createSyncResources so the lighting/transparent sets can
    // bind the array view.
    shadowsActive_ = deferred_.createShadowTargets(ctx_, shadow_);
    if (!shadowsActive_)
        std::fprintf(stderr, "warning: shadow target creation failed, shadows disabled\n");
    if (!createRenderTargets()) return false;
    if (!createFontAtlas()) return false;
    if (!createMetricResources()) return false;
    if (!createShaders()) return false;
    if (!createDescriptors()) return false;
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
    // Color mip chains for roughness-aware SSR (mip 0 = the lit-color copy).
    if (!deferred_.createColorPyramid(ctx_, renderWidth_, renderHeight_, gbColorPyramid_))
        return false;
    if (!deferred_.createColorPyramid(ctx_, dw, dh, gtColorPyramid_))
        return false;

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
        if (!deferred_.createColorPyramid(ctx_, dw * 2, dh * 2, gtSsaaColorPyramid_))
            return false;
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

    submitOneShot(ctx_, [&](VkCommandBuffer cmd) {
        copyBufferToImage(cmd, staging, fontAtlas_.image, kFontAtlasW, kFontAtlasH,
                          VK_FORMAT_R8_UNORM);
    });
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

    // Compose: column source + packed text UBO + font atlas.
    VkDescriptorSetLayoutBinding composeBindings[3] = {};
    for (uint32_t i = 0; i < 3; ++i) {
        composeBindings[i].binding = i;
        composeBindings[i].descriptorCount = 1;
        composeBindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        composeBindings[i].descriptorType =
            i == 1 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    }
    VkDescriptorSetLayoutCreateInfo composeLayoutCi = {};
    composeLayoutCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    composeLayoutCi.bindingCount = 3;
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
    const uint32_t hizSets =
        gbPyramid_.mipCount + gtPyramid_.mipCount + gtSsaaPyramid_.mipCount;
    const uint32_t colorSets =
        gbColorPyramid_.mipCount + gtColorPyramid_.mipCount + gtSsaaColorPyramid_.mipCount;
    VkDescriptorPoolSize sizes[5] = {};
    sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = deferred::kMaxTextures + numColumns * 2 + 2 + numAlgos * 2 +
                               11 * kFramesInFlight * 4 + // lighting sets (GB/GT/SSAA/spatial), +1 shadow
                               7 * kFramesInFlight * 4 +  // transparent sets + SSR
                               3 * 3 +                    // ssao + blur samplers (per path)
                               hizSets + colorSets;
    sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[1].descriptorCount = kFramesInFlight * 2 + numColumns +
                               kFramesInFlight * 2 + // lighting UBOs
                               kFramesInFlight * 3 +  // transparent UBOs (GB/GT/SSAA)
                               3 * kFramesInFlight;   // spatial scene + lighting + transparent
    sizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    sizes[2].descriptorCount = kFramesInFlight * 3; // Gb + Gt + spatial scene sets
    sizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[3].descriptorCount = numAlgos * 2;
    sizes[4].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    sizes[4].descriptorCount = 6 + hizSets + colorSets; // ssao raw + blurred outputs (GB/GT/SSAA) + pyramid mips
    VkDescriptorPoolCreateInfo poolCi = {};
    poolCi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCi.maxSets = kFramesInFlight * 2 + 2 + numColumns + 1 + numAlgos + kFramesInFlight * 3 +
                     kFramesInFlight * 3 + // transparent sets (GB/GT/SSAA)
                     6 +                    // ssao sets (GB/GT/SSAA, static)
                     3 * kFramesInFlight + // spatial scene + lighting + transparent
                     hizSets + colorSets;
    poolCi.poolSizeCount = 5;
    poolCi.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(ctx_.device, &poolCi, nullptr, &descriptorPool_) != VK_SUCCESS)
        return false;

    if (!deferred_.writeDepthPyramidSets(ctx_, descriptorPool_, gbDepth_.view, gbPyramid_))
        return false;
    if (!deferred_.writeDepthPyramidSets(ctx_, descriptorPool_, gtDepth_.view, gtPyramid_))
        return false;
    if (opts_.gtSsaa &&
        !deferred_.writeDepthPyramidSets(ctx_, descriptorPool_, gtSsaaDepth_.view,
                                         gtSsaaPyramid_))
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
    }
    if (!allocSet(deferred_.textureSetLayout(), textureSet_)) return false;
    if (!allocSet(deferred_.ssaoSetLayout(), ssaoSetGb_)) return false;
    if (!allocSet(deferred_.ssaoBlurSetLayout(), ssaoBlurSetGb_)) return false;
    if (!allocSet(deferred_.ssaoSetLayout(), ssaoSetGt_)) return false;
    if (!allocSet(deferred_.ssaoBlurSetLayout(), ssaoBlurSetGt_)) return false;
    if (opts_.gtSsaa) {
        if (!allocSet(deferred_.ssaoSetLayout(), ssaoSetSsaa_)) return false;
        if (!allocSet(deferred_.ssaoBlurSetLayout(), ssaoBlurSetSsaa_)) return false;
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
    deferred_.writeSsaoSet(ctx_, ssaoSetGb_, gbDepth_.view, gbNormal_.view, gbAoRaw_.view);
    deferred_.writeSsaoBlurSet(ctx_, ssaoBlurSetGb_, gbAoRaw_.view, gbAo_.view);
    deferred_.writeSsaoSet(ctx_, ssaoSetGt_, gtDepth_.view, gtNormal_.view, gtAoRaw_.view);
    deferred_.writeSsaoBlurSet(ctx_, ssaoBlurSetGt_, gtAoRaw_.view, gtAo_.view);
    if (opts_.gtSsaa) {
        deferred_.writeSsaoSet(ctx_, ssaoSetSsaa_, gtSsaaDepth_.view, gtSsaaNormal_.view,
                               gtSsaaAoRaw_.view);
        deferred_.writeSsaoBlurSet(ctx_, ssaoBlurSetSsaa_, gtSsaaAoRaw_.view, gtSsaaAo_.view);
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

        VkWriteDescriptorSet writes[3] = {};
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
        vkUpdateDescriptorSets(ctx_.device, 3, writes, 0, nullptr);
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

    VkPipelineLayoutCreateInfo copyLayoutCi = {};
    copyLayoutCi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    copyLayoutCi.setLayoutCount = 1;
    copyLayoutCi.pSetLayouts = &copySetLayout_;
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
        deferred_.writeLightingSet(ctx_, fr.lightingSetGb, fr.lightingUboGb, gbAlbedo_.view,
                                   gbNormal_.view, gbMaterial_.view, gbEmissive_.view,
                                   gbDepth_.view, gbAo_.view, shadowView);
        deferred_.writeLightingSet(ctx_, fr.lightingSetGbSpatial, fr.lightingUboGbSpatial,
                                   gbAlbedo_.view, gbNormal_.view, gbMaterial_.view,
                                   gbEmissive_.view, gbDepth_.view, gbAo_.view, shadowView);
        deferred_.writeLightingSet(ctx_, fr.lightingSetGt, fr.lightingUboGt, gtAlbedo_.view,
                                   gtNormal_.view, gtMaterial_.view, gtEmissive_.view,
                                   gtDepth_.view, gtAo_.view, shadowView);
        if (opts_.gtSsaa) {
            deferred_.writeLightingSet(ctx_, fr.lightingSetSsaa, fr.lightingUboGt,
                                       gtSsaaAlbedo_.view, gtSsaaNormal_.view,
                                       gtSsaaMaterial_.view, gtSsaaEmissive_.view,
                                       gtSsaaDepth_.view, gtSsaaAo_.view, shadowView);
        }

        // The transparency shader reads iblParams (identical in both lighting
        // UBOs) plus the path's own SSAO texture: one set per path.
        deferred_.writeTransparentSet(ctx_, fr.transparentSetGb, fr.lightingUboGb, gbAo_.view,
                                      shadowView, gbColorPyramid_.chainView,
                                      gbPyramid_.chainView);
        deferred_.writeTransparentSet(ctx_, fr.transparentSetGbSpatial, fr.lightingUboGbSpatial,
                                      gbAo_.view, shadowView, gbColorPyramid_.chainView,
                                      gbPyramid_.chainView);
        deferred_.writeTransparentSet(ctx_, fr.transparentSetGt, fr.lightingUboGt, gtAo_.view,
                                      shadowView, gtColorPyramid_.chainView,
                                      gtPyramid_.chainView);
        if (opts_.gtSsaa) {
            deferred_.writeTransparentSet(ctx_, fr.transparentSetSsaa, fr.lightingUboGt,
                                          gtSsaaAo_.view, shadowView,
                                          gtSsaaColorPyramid_.chainView,
                                          gtSsaaPyramid_.chainView);
        }
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

void CompareApp::updateLightingUBO(void* mapped, const Mat4& invViewProj,
                                   const ShadowFrame* shadow) {
    LightingUBO ubo;
    deferred_.fillLightingUBO(ubo, scene_, camera_, invViewProj, nullptr, shadow, iblIntensity_);
    std::memcpy(mapped, &ubo, sizeof(ubo));
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

    // CSM sun shadows: pick the first shadow-casting directional light (same
    // light-list selection rule as fillLightingUBO) and compute the cascades.
    // The GT and SSAA paths sample the same map — GT is the same lighting at
    // native res.  --no-shadows leaves shadow null (shadowParams.z = 0).
    ShadowFrame shadowFrame;
    const ShadowFrame* shadow = nullptr;
    if (shadowsActive_ && opts_.shadows) {
        const std::vector<Light>& lights =
            scene_.lights.empty() ? defaultLights() : scene_.lights;
        for (uint32_t i = 0; i < lights.size(); ++i) {
            if (lights[i].type == LightType::Directional && lights[i].castShadow) {
                DeferredCore::computeCascadeVPs(camera_, aspect, lights[i].positionOrDirection,
                                                shadowFrame.cascadeVp, shadowFrame.splitDepth);
                shadowFrame.lightIndex = static_cast<int32_t>(i);
                shadowFrame.debugCascades = opts_.shadowDebug;
                shadow = &shadowFrame;
                break;
            }
        }
    }

    // Lighting reconstructs world positions with the inverse of the exact
    // view-projection used for each pass (jittered GB / unjittered spatial GB /
    // un-jittered GT; the GT matrix is resolution-independent and shared by
    // the 1x and 2x SSAA pass).
    updateLightingUBO(fr.lightingUboGbMapped,
                      Mat4::inverse(Mat4::multiply(projJittered, view)), shadow);
    updateLightingUBO(fr.lightingUboGtMapped, Mat4::inverse(Mat4::multiply(proj, view)), shadow);
    if (mixed) {
        updateSceneUBO(fr.uboGbSpatialMapped, false, renderWidth_, renderHeight_, view, proj, proj,
                       prevViewProj_);
        updateLightingUBO(fr.lightingUboGbSpatialMapped, Mat4::inverse(Mat4::multiply(proj, view)),
                          shadow);
    }

    auto transition = [&](VkImage image, VkImageLayout& current, VkImageLayout target,
                          VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                          VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
                          VkImageAspectFlags aspect) {
        imageBarrier(cmd, image, current, target, srcStage, srcAccess, dstStage, dstAccess, aspect);
        current = target;
    };
    const Mat4 cullViewProj = Mat4::multiply(proj, view); // un-jittered (sub-pixel)

    auto recordLrGBuffer = [&](VkDescriptorSet sceneSet) {
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
                                         renderWidth_, renderHeight_, cullViewProj);
            vkCmdEndRendering(cmd);
        }
    };

    auto recordLrLighting = [&](VkDescriptorSet sceneSet, VkDescriptorSet lightingSet,
                                VkDescriptorSet transparentSet, const Mat4& ssaoViewProj) {
        transition(gbAlbedo_.image, gbAlbedoLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbNormal_.image, gbNormalLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbMaterial_.image, gbMaterialLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbEmissive_.image, gbEmissiveLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbMotion_.image, gbMotionLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbDepth_.image, gbDepthLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kDepthTests, sync::kDepthWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_DEPTH_BIT);
        // Hi-Z pyramid for the glass SSR marcher (both LR lighting variants
        // share the same GBuffer depth).
        if (hasTransparency_)
            deferred_.recordDepthPyramidPass(cmd, gbPyramid_);

        transition(gbAoRaw_.image, gbAoRawLayout_, VK_IMAGE_LAYOUT_GENERAL, sync::kCompute,
                   sync::kSampled, sync::kCompute, sync::kStorageWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordSsaoPass(cmd, ssaoSetGb_, ssaoViewProj, frameIndex, renderWidth_,
                                 renderHeight_);
        transition(gbAoRaw_.image, gbAoRawLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kCompute, sync::kStorageWrite, sync::kCompute, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gbAo_.image, gbAoLayout_, VK_IMAGE_LAYOUT_GENERAL, sync::kFragment,
                   sync::kSampled, sync::kCompute, sync::kStorageWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordSsaoBlurPass(cmd, ssaoBlurSetGb_, renderWidth_, renderHeight_);
        transition(gbAo_.image, gbAoLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kCompute, sync::kStorageWrite, sync::kFragment, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);

        transition(gbColor_.image, gbColorLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordLightingPass(cmd, lightingSet, gbColor_.view, renderWidth_, renderHeight_);

        if (hasTransparency_) {
            // Color mip chain: mip 0 is the opaque HDR copy glass SSR used to
            // make with a transfer; the chain stays GENERAL for life.
            transition(gbColor_.image, gbColorLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       sync::kColorAttach, sync::kColorWrite, sync::kCompute, sync::kSampled,
                       VK_IMAGE_ASPECT_COLOR_BIT);
            deferred_.recordColorPyramidPass(cmd, gbColorPyramid_);
            transition(gbColor_.image, gbColorLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                       sync::kCompute, sync::kSampled, sync::kColorAttach, sync::kColorReadWrite,
                       VK_IMAGE_ASPECT_COLOR_BIT);
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
                                                 renderHeight_, cullViewProj, camera_.position);
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
    };

    // --- 1) low-resolution deferred (spatial copy then jittered when mixed) -----
    if (mixed) {
        recordLrGBuffer(fr.sceneSetGbSpatial);
    } else {
        recordLrGBuffer(fr.sceneSetGb);
    }

    // --- Shadow pass (sun CSM, one 2048^2 layer per cascade) -------------------
    // Runs once per frame and feeds the LR, GT and SSAA lighting paths alike.
    if (shadow) {
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

    if (mixed) {
        recordLrLighting(fr.sceneSetGbSpatial, fr.lightingSetGbSpatial, fr.transparentSetGbSpatial,
                         cullViewProj);
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
        recordLrLighting(fr.sceneSetGb, fr.lightingSetGb, fr.transparentSetGb,
                         Mat4::multiply(projJittered, view));
    } else {
        recordLrLighting(fr.sceneSetGb, fr.lightingSetGb, fr.transparentSetGb,
                         Mat4::multiply(projJittered, view));
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
    frame.preExposure = 1.f;
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
        transition(gtSsaaDepth_.image, gtSsaaDepthLayout_,
                   VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, sync::kSampleStages,
                   sync::kSampled, sync::kDepthTests, sync::kDepthReadWrite,
                   VK_IMAGE_ASPECT_DEPTH_BIT);
        {
            VkRenderingAttachmentInfo colors[4] = {
                makeColorAttachment(gtSsaaAlbedo_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gtSsaaNormal_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR, 0.f, 0.f, 1.f),
                makeColorAttachment(gtSsaaMaterial_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gtSsaaEmissive_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR)};
            VkRenderingAttachmentInfo depth =
                makeDepthAttachment(gtSsaaDepth_.view,
                                    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR);
            beginRendering(cmd, sw, sh, 4, colors, &depth);
            deferred_.recordGBufferDraws(cmd, scene_, true, fr.sceneSetGt, textureSet_,
                                         materialStride_, sw, sh, cullViewProj);
            vkCmdEndRendering(cmd);
        }
        transition(gtSsaaAlbedo_.image, gtSsaaAlbedoLayout_,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kColorAttach, sync::kColorWrite,
                   sync::kFragment, sync::kSampled, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaNormal_.image, gtSsaaNormalLayout_,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kColorAttach, sync::kColorWrite,
                   sync::kFragment, sync::kSampled, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaMaterial_.image, gtSsaaMaterialLayout_,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kColorAttach, sync::kColorWrite,
                   sync::kFragment, sync::kSampled, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaEmissive_.image, gtSsaaEmissiveLayout_,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kColorAttach, sync::kColorWrite,
                   sync::kFragment, sync::kSampled, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaDepth_.image, gtSsaaDepthLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kDepthTests, sync::kDepthWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_DEPTH_BIT);
        if (hasTransparency_)
            deferred_.recordDepthPyramidPass(cmd, gtSsaaPyramid_);

        // SSAO for the 2x GT path (un-jittered view-projection).
        transition(gtSsaaAoRaw_.image, gtSsaaAoRawLayout_, VK_IMAGE_LAYOUT_GENERAL,
                   sync::kCompute, sync::kSampled, sync::kCompute, sync::kStorageWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordSsaoPass(cmd, ssaoSetSsaa_, cullViewProj, frameIndex, sw, sh);
        transition(gtSsaaAoRaw_.image, gtSsaaAoRawLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kCompute, sync::kStorageWrite, sync::kCompute, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtSsaaAo_.image, gtSsaaAoLayout_, VK_IMAGE_LAYOUT_GENERAL, sync::kFragment,
                   sync::kSampled, sync::kCompute, sync::kStorageWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordSsaoBlurPass(cmd, ssaoBlurSetSsaa_, sw, sh);
        transition(gtSsaaAo_.image, gtSsaaAoLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kCompute, sync::kStorageWrite, sync::kFragment, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);

        transition(gtSsaaColor_.image, gtSsaaColorLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordLightingPass(cmd, fr.lightingSetSsaa, gtSsaaColor_.view, sw, sh);

        if (hasTransparency_) {
            // Same color mip chain build as the GB path (GENERAL for life).
            transition(gtSsaaColor_.image, gtSsaaColorLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       sync::kColorAttach, sync::kColorWrite, sync::kCompute, sync::kSampled,
                       VK_IMAGE_ASPECT_COLOR_BIT);
            deferred_.recordColorPyramidPass(cmd, gtSsaaColorPyramid_);
            transition(gtSsaaColor_.image, gtSsaaColorLayout_,
                       VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, sync::kCompute, sync::kSampled,
                       sync::kColorAttach, sync::kColorReadWrite, VK_IMAGE_ASPECT_COLOR_BIT);
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
                                                 cullViewProj, camera_.position);
                vkCmdEndRendering(cmd);
            }
            transition(gtSsaaDepth_.image, gtSsaaDepthLayout_,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kDepthTests,
                       sync::kDepthRead, sync::kFragment, sync::kSampled, VK_IMAGE_ASPECT_DEPTH_BIT);
        }

        transition(gtSsaaColor_.image, gtSsaaColorLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);

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
        transition(gtDepth_.image, gtDepthLayout_, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kDepthTests, sync::kDepthReadWrite,
                   VK_IMAGE_ASPECT_DEPTH_BIT);
        {
            VkRenderingAttachmentInfo colors[4] = {
                makeColorAttachment(gtAlbedo_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gtNormal_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR, 0.f, 0.f, 1.f),
                makeColorAttachment(gtMaterial_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR),
                makeColorAttachment(gtEmissive_.view, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR)};
            VkRenderingAttachmentInfo depth =
                makeDepthAttachment(gtDepth_.view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR);
            beginRendering(cmd, dw, dh, 4, colors, &depth);
            deferred_.recordGBufferDraws(cmd, scene_, true, fr.sceneSetGt, textureSet_,
                                         materialStride_, dw, dh, cullViewProj);
            vkCmdEndRendering(cmd);
        }
        transition(gtAlbedo_.image, gtAlbedoLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtNormal_.image, gtNormalLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtMaterial_.image, gtMaterialLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtEmissive_.image, gtEmissiveLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kFragment, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtDepth_.image, gtDepthLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kDepthTests, sync::kDepthWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_DEPTH_BIT);
        if (hasTransparency_)
            deferred_.recordDepthPyramidPass(cmd, gtPyramid_);

        // SSAO for the 1x GT path (un-jittered view-projection).
        transition(gtAoRaw_.image, gtAoRawLayout_, VK_IMAGE_LAYOUT_GENERAL, sync::kCompute,
                   sync::kSampled, sync::kCompute, sync::kStorageWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordSsaoPass(cmd, ssaoSetGt_, cullViewProj, frameIndex, dw, dh);
        transition(gtAoRaw_.image, gtAoRawLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kCompute, sync::kStorageWrite, sync::kCompute, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        transition(gtAo_.image, gtAoLayout_, VK_IMAGE_LAYOUT_GENERAL, sync::kFragment,
                   sync::kSampled, sync::kCompute, sync::kStorageWrite, VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordSsaoBlurPass(cmd, ssaoBlurSetGt_, dw, dh);
        transition(gtAo_.image, gtAoLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kCompute, sync::kStorageWrite, sync::kFragment, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);

        transition(gtColor_.image, gtColorLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   sync::kSampleStages, sync::kSampled, sync::kColorAttach, sync::kColorWrite,
                   VK_IMAGE_ASPECT_COLOR_BIT);
        deferred_.recordLightingPass(cmd, fr.lightingSetGt, gtColor_.view, dw, dh);

        if (hasTransparency_) {
            // Same color mip chain build as the GB path (GENERAL for life).
            transition(gtColor_.image, gtColorLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       sync::kColorAttach, sync::kColorWrite, sync::kCompute, sync::kSampled,
                       VK_IMAGE_ASPECT_COLOR_BIT);
            deferred_.recordColorPyramidPass(cmd, gtColorPyramid_);
            transition(gtColor_.image, gtColorLayout_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                       sync::kCompute, sync::kSampled, sync::kColorAttach, sync::kColorReadWrite,
                       VK_IMAGE_ASPECT_COLOR_BIT);
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
                                                 cullViewProj, camera_.position);
                vkCmdEndRendering(cmd);
            }
            transition(gtDepth_.image, gtDepthLayout_,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kDepthTests,
                       sync::kDepthRead, sync::kFragment, sync::kSampled, VK_IMAGE_ASPECT_DEPTH_BIT);
        }

        transition(gtColor_.image, gtColorLayout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   sync::kColorAttach, sync::kColorWrite, sync::kSampleStages, sync::kSampled,
                   VK_IMAGE_ASPECT_COLOR_BIT);
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
            push.exposure = opts_.exposure;
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
            // Nearest sampling once the on-screen magnification passes 1:1.
            push.nearest = (static_cast<float>(w) >= rect[2]) ? 1.f : 0.f;
            push.exposure = opts_.exposure;
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

        // The metric readback recorded for this slot (if any) is complete now.
        if (metricPending_[slot]) {
            harvestMetrics(slot);
            metricPending_[slot] = false;
        }

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
    deferred_.destroyDepthPyramid(ctx_, gbPyramid_);
    deferred_.destroyDepthPyramid(ctx_, gtPyramid_);
    deferred_.destroyDepthPyramid(ctx_, gtSsaaPyramid_);
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
    gtDepth_.destroy(ctx_);
    gtSsaaColor_.destroy(ctx_);
    gtSsaaAlbedo_.destroy(ctx_);
    gtSsaaNormal_.destroy(ctx_);
    gtSsaaMaterial_.destroy(ctx_);
    gtSsaaEmissive_.destroy(ctx_);
    gtSsaaDepth_.destroy(ctx_);
    gbAoRaw_.destroy(ctx_);
    gbAo_.destroy(ctx_);
    gtAoRaw_.destroy(ctx_);
    gtAo_.destroy(ctx_);
    gtSsaaAoRaw_.destroy(ctx_);
    gtSsaaAo_.destroy(ctx_);
    composeImage_.destroy(ctx_);
    fontAtlas_.destroy(ctx_);

    if (shadowsActive_) { deferred_.destroyShadowTargets(ctx_, shadow_); shadowsActive_ = false; }
    deferred_.destroy(ctx_);
    scene_.destroy(ctx_);
    swapchain_.destroy(ctx_);
    ctx_.destroy();
    window_.destroy();
}

} // namespace sr
