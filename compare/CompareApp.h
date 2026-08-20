#pragma once
// ============================================================================
// CompareApp — split-screen quality comparison mode.
//
// Renders one low-resolution deferred GBuffer pass (Halton jittered) shared by
// every registered upscaler, one native-resolution ground-truth pass, and
// composites the results side by side: the leftmost column is the native 100%
// render ("NATIVE (GT)"), the remaining columns are the upscaler outputs.
// Shading is the shared DeferredCore PBR + IBL pipeline (renderer/deferred),
// identical to the viewer renderer.  Each column shows a live PSNR/SSIM
// overlay computed on the GPU (two-pass compute reduction, read back every N
// frames).
// ============================================================================
#include "renderer/core/Swapchain.h"
#include "renderer/core/VulkanContext.h"
#include "renderer/core/Window.h"
#include "renderer/deferred/DeferredCore.h"
#include "renderer/scene/Camera.h"
#include "renderer/scene/CameraPath.h"
#include "renderer/scene/Scene.h"
#include "upscalers/IUpscaler.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sr {

struct CompareOptions {
    uint32_t displayWidth = 1920;
    uint32_t displayHeight = 1080;
    float renderScale = 0.5f;
    bool vsync = true;
    int frames = -1;                    // -1 = interactive
    std::vector<std::string> upscalerNames; // empty = all registered & available
    std::string screenshotPath;
    std::string cameraPath;             // empty = orbit (automation) / free-fly
    std::string scenePath;              // empty = procedural
    int metricInterval = 15;            // frames between GPU metric readbacks
    bool gtSsaa = false;                // render GT at 2x and downsample to 1080p
    float zoom = 1.f;                   // compare-view zoom (1..16)
    float zoomCenterU = 0.5f;           // zoom window center, normalized source UV
    float zoomCenterV = 0.5f;
    std::string envMapPath = kDefaultEnvMapPath; // equirect HDR for IBL/skybox
};

class CompareApp {
public:
    bool init(const CompareOptions& opts);
    void run();
    void shutdown();

private:
    static constexpr uint32_t kFramesInFlight = 2;
    static constexpr uint32_t kMaxAlgos = 4;            // extra columns are truncated
    static constexpr uint32_t kMaxColumns = 1 + kMaxAlgos; // GT + algorithms
    static constexpr uint32_t kMetricFloats = 8;        // per-algorithm reduce record
    static constexpr uint32_t kTextCharsPerColumn = 72; // 3 lines x 24 chars

    struct ImageResource {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        uint32_t width = 0;
        uint32_t height = 0;
        void destroy(const VulkanContext& ctx);
    };

    struct FrameResources {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkBuffer uboGb = VK_NULL_HANDLE;       // jittered scene UBO (GBuffer pass)
        VkDeviceMemory uboGbMemory = VK_NULL_HANDLE;
        void* uboGbMapped = nullptr;
        VkBuffer uboGt = VK_NULL_HANDLE;       // un-jittered scene UBO (GT pass)
        VkDeviceMemory uboGtMemory = VK_NULL_HANDLE;
        void* uboGtMapped = nullptr;
        VkBuffer lightingUboGb = VK_NULL_HANDLE; // lighting UBO (jittered invViewProj)
        VkDeviceMemory lightingUboGbMemory = VK_NULL_HANDLE;
        void* lightingUboGbMapped = nullptr;
        VkBuffer lightingUboGt = VK_NULL_HANDLE; // lighting UBO (un-jittered invViewProj)
        VkDeviceMemory lightingUboGtMemory = VK_NULL_HANDLE;
        void* lightingUboGtMapped = nullptr;
        VkDescriptorSet sceneSetGb = VK_NULL_HANDLE;
        VkDescriptorSet sceneSetGt = VK_NULL_HANDLE;
        VkDescriptorSet lightingSetGb = VK_NULL_HANDLE;
        VkDescriptorSet lightingSetGt = VK_NULL_HANDLE;
        VkDescriptorSet lightingSetSsaa = VK_NULL_HANDLE; // 2x GT GBuffer (gtSsaa only)
        // IBL + per-path SSAO texture for the transparency pass.
        VkDescriptorSet transparentSetGb = VK_NULL_HANDLE;
        VkDescriptorSet transparentSetGt = VK_NULL_HANDLE;
        VkDescriptorSet transparentSetSsaa = VK_NULL_HANDLE; // gtSsaa only
    };

    struct AlgoColumn {
        std::string id;
        std::unique_ptr<IUpscaler> upscaler;
        ImageResource output; // display-resolution RGBA16F
        VkImageLayout outputLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkBuffer blocksBuffer = VK_NULL_HANDLE; // per-block metric records
        VkDeviceMemory blocksMemory = VK_NULL_HANDLE;
        VkDescriptorSet metricSet = VK_NULL_HANDLE;
        VkDescriptorSet composeSet = VK_NULL_HANDLE;
        float psnr = 0.f;
        float ssim = 0.f;
        bool hasMetric = false;
    };

    CompareOptions opts_;
    Window window_;
    VulkanContext ctx_;
    Swapchain swapchain_;
    Scene scene_;
    Camera camera_;
    CameraPath path_;
    std::vector<AlgoColumn> algos_;

    uint32_t renderWidth_ = 0;
    uint32_t renderHeight_ = 0;
    uint32_t blocksPerRow_ = 0;
    uint32_t blockCount_ = 0;

    Mat4 prevViewProj_ = Mat4::identity();
    float jitterX_ = 0.f;
    float jitterY_ = 0.f;
    float prevJitterX_ = 0.f;
    float prevJitterY_ = 0.f;
    bool hasTransparency_ = false; // any alphaMode BLEND material in the scene

    ImageResource gbColor_;
    ImageResource gbMotion_;
    ImageResource gbReactive_; // translucent coverage mask (reactive/TC input)
    ImageResource gbDepth_;
    // Deferred GBuffer attachments of the low-res input pass.
    ImageResource gbAlbedo_;
    ImageResource gbNormal_;
    ImageResource gbMaterial_;
    ImageResource gbEmissive_;
    ImageResource gtColor_;
    ImageResource gtDepth_;
    // Deferred GBuffer attachments of the native-res GT pass.
    ImageResource gtAlbedo_;
    ImageResource gtNormal_;
    ImageResource gtMaterial_;
    ImageResource gtEmissive_;
    ImageResource gtSsaaColor_; // 2x GT render target (gtSsaa only)
    ImageResource gtSsaaDepth_;
    // Deferred GBuffer attachments of the 2x GT pass (gtSsaa only).
    ImageResource gtSsaaAlbedo_;
    ImageResource gtSsaaNormal_;
    ImageResource gtSsaaMaterial_;
    ImageResource gtSsaaEmissive_;
    // SSAO (raw + blurred) per GBuffer path, R16_SFLOAT, path resolution.
    ImageResource gbAoRaw_;
    ImageResource gbAo_;
    ImageResource gtAoRaw_;
    ImageResource gtAo_;
    ImageResource gtSsaaAoRaw_; // gtSsaa only
    ImageResource gtSsaaAo_;
    ImageResource composeImage_; // RGBA8, tonemapped columns + overlay
    ImageResource fontAtlas_;

    // Shared deferred pipeline (shaders/layouts/pipelines/samplers + IBL maps).
    DeferredCore deferred_;

    VkDescriptorSetLayout composeSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout copySetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout metricSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout composePipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout copyPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout metricPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline composePipeline_ = VK_NULL_HANDLE;
    VkPipeline copyPipeline_ = VK_NULL_HANDLE;
    VkPipeline downsamplePipeline_ = VK_NULL_HANDLE; // 2x GT -> 1x GT (SSAA)
    VkPipeline metricBlocksPipeline_ = VK_NULL_HANDLE;
    VkPipeline metricReducePipeline_ = VK_NULL_HANDLE;
    VkShaderModule fullscreenVert_ = VK_NULL_HANDLE;
    VkShaderModule composeFrag_ = VK_NULL_HANDLE;
    VkShaderModule copyFrag_ = VK_NULL_HANDLE;
    VkShaderModule metricBlocksComp_ = VK_NULL_HANDLE;
    VkShaderModule metricReduceComp_ = VK_NULL_HANDLE;

    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    FrameResources frames_[kFramesInFlight] = {};
    std::vector<VkSemaphore> renderFinished_;
    VkDescriptorSet textureSet_ = VK_NULL_HANDLE;
    VkDescriptorSet copySet_ = VK_NULL_HANDLE;
    VkDescriptorSet gtComposeSet_ = VK_NULL_HANDLE;
    VkDescriptorSet gtDownsampleSet_ = VK_NULL_HANDLE; // 2x GT source (SSAA)
    // SSAO descriptor sets (static: the referenced textures never change).
    VkDescriptorSet ssaoSetGb_ = VK_NULL_HANDLE;
    VkDescriptorSet ssaoBlurSetGb_ = VK_NULL_HANDLE;
    VkDescriptorSet ssaoSetGt_ = VK_NULL_HANDLE;
    VkDescriptorSet ssaoBlurSetGt_ = VK_NULL_HANDLE;
    VkDescriptorSet ssaoSetSsaa_ = VK_NULL_HANDLE;     // gtSsaa only
    VkDescriptorSet ssaoBlurSetSsaa_ = VK_NULL_HANDLE; // gtSsaa only
    VkSampler linearSampler_ = VK_NULL_HANDLE;
    VkSampler fontSampler_ = VK_NULL_HANDLE;

    VkImageLayout gbColorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbMotionLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbReactiveLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbDepthLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbAlbedoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbNormalLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbMaterialLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbEmissiveLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtColorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtDepthLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtAlbedoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtNormalLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtMaterialLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtEmissiveLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaColorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaDepthLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaAlbedoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaNormalLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaMaterialLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaEmissiveLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbAoRawLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gbAoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtAoRawLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtAoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaAoRawLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout gtSsaaAoLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout composeLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    VkBuffer materialUbo_ = VK_NULL_HANDLE;
    VkDeviceMemory materialUboMemory_ = VK_NULL_HANDLE;
    uint32_t materialStride_ = 0;

    VkBuffer textUbo_ = VK_NULL_HANDLE; // packed ASCII overlay text (all columns)
    VkDeviceMemory textUboMemory_ = VK_NULL_HANDLE;
    void* textUboMapped_ = nullptr;

    VkBuffer metricResultBuf_ = VK_NULL_HANDLE; // kMaxAlgos * kMetricFloats floats
    VkDeviceMemory metricResultMemory_ = VK_NULL_HANDLE;
    VkBuffer metricStaging_[kFramesInFlight] = {};
    VkDeviceMemory metricStagingMemory_[kFramesInFlight] = {};
    void* metricStagingMapped_[kFramesInFlight] = {};
    bool metricPending_[kFramesInFlight] = {};

    VkBuffer screenshotStaging_ = VK_NULL_HANDLE;
    VkDeviceMemory screenshotStagingMemory_ = VK_NULL_HANDLE;
    void* screenshotMapped_ = nullptr;

    bool initAlgorithms();
    bool createRenderTargets();
    bool createShaders();
    bool createDescriptors();
    bool createPipelines();
    bool createSyncResources();
    bool recreateRenderFinishedSemaphores();
    bool createFontAtlas();
    bool createMetricResources();
    bool createScreenshotStaging();
    void updateSceneUBO(void* mapped, bool jitter, uint32_t renderW, uint32_t renderH,
                        const Mat4& view, const Mat4& proj, const Mat4& projJittered,
                        const Mat4& prevViewProj);
    void updateLightingUBO(void* mapped, const Mat4& invViewProj);
    void updateCamera(uint32_t frameIndex, float dt);
    void recordFrame(uint32_t frameIndex, uint32_t swapchainIndex);
    void captureScreenshotIntoStaging(VkCommandBuffer cmd);
    void saveScreenshot(const std::string& path);
    void harvestMetrics(uint32_t slot);
    void refreshOverlayText();
    bool loadShader(const char* name, VkShaderModule& out);
    // Visible source region (px) for a column of colW x colH: aspect-preserving
    // center-crop at zoom 1; a zoomed window centered on (centerU, centerV)
    // (normalized source UV) when zoom > 1.  out = {offX, offY, sizeX, sizeY}.
    static void computeViewRegion(uint32_t srcW, uint32_t srcH, uint32_t colW, uint32_t colH,
                                  float zoom, float centerU, float centerV, float out[4]);
};

} // namespace sr
