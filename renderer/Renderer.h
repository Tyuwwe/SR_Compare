#pragma once
// ============================================================================
// Renderer — owns the window, Vulkan context, swapchain, scene, camera and the
// frame loop.  Produces the upscaler inputs (HDR color + depth + motion) at
// render scale (with Halton jitter), runs the IUpscaler, and presents either
// the upscaled image or the native-resolution ground truth.
// ============================================================================
#include "renderer/core/Swapchain.h"
#include "renderer/core/TimestampQuery.h"
#include "renderer/core/Vk.h"
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

struct RendererOptions {
    uint32_t displayWidth = 1920;
    uint32_t displayHeight = 1080;
    float renderScale = 0.5f;
    bool vsync = true;
    int frames = -1;           // -1 = interactive
    std::string upscalerName = "taa";  // registered plugin name; "none" = native ground truth
    std::string screenshotPath;
    std::string frameTimesPath;  // non-empty: dump per-frame GPU timings + VRAM to CSV
    std::string cameraPath;    // empty = orbit (automation) / free-fly (interactive)
    std::string scenePath;     // empty = procedural
    // Equirect HDR environment map for IBL + skybox; empty or unreadable file
    // falls back to a procedural gradient (constant-ambient look).
    std::string envMapPath = kDefaultEnvMapPath;
    bool shadows = true;      // CSM sun shadows (CLI: --no-shadows)
    bool shadowDebug = false; // cascade tint overlay (CLI: --shadow-debug)
    float exposure = 1.f;     // display-domain ACES input multiplier
    bool bloom = true;        // HDR bloom before upscale (CLI: --no-bloom)
};

class Renderer {
public:
    bool init(const RendererOptions& opts);
    void run();
    void shutdown();

private:
    static constexpr uint32_t kFramesInFlight = 2;

    struct ImageResource {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        uint32_t width = 0;
        uint32_t height = 0;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        void destroy(const VulkanContext& ctx);
    };

    struct FrameResources {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkBuffer ubo = VK_NULL_HANDLE;
        VmaAllocation uboMemory = VK_NULL_HANDLE;
        void* uboMapped = nullptr;
        VkBuffer lightingUbo = VK_NULL_HANDLE;
        VmaAllocation lightingUboMemory = VK_NULL_HANDLE;
        void* lightingUboMapped = nullptr;
        VkDescriptorSet sceneSet = VK_NULL_HANDLE;
        VkDescriptorSet lightingSetGb = VK_NULL_HANDLE;
        VkDescriptorSet lightingSetGt = VK_NULL_HANDLE;
        VkDescriptorSet transparentSetGb = VK_NULL_HANDLE; // IBL + LR AO texture
        VkDescriptorSet transparentSetGt = VK_NULL_HANDLE; // IBL + GT AO texture
    };

    RendererOptions opts_;
    Window window_;
    VulkanContext ctx_;
    Swapchain swapchain_;
    Scene scene_;
    Camera camera_;
    CameraPath path_;
    std::unique_ptr<IUpscaler> upscaler_;

    uint32_t renderWidth_ = 0;
    uint32_t renderHeight_ = 0;

    Mat4 prevViewProj_ = Mat4::identity();
    float jitterX_ = 0.f;
    float jitterY_ = 0.f;
    float prevJitterX_ = 0.f;
    float prevJitterY_ = 0.f;
    bool hasTransparency_ = false; // any alphaMode BLEND material in the scene
    // Diagnostic env switch: SR_NO_JITTER=1 zeroes the Halton sub-pixel jitter
    // while keeping the low-res path (isolates jitter from resolution).
    bool diagNoJitter_ = false;
    // Wall-clock delta time (seconds) fed to the upscaler.  Interactive mode
    // updates it every frame; bench mode keeps the 1/60 default for
    // reproducible, machine-independent results.
    float deltaTime_ = 1.0f / 60.0f;

    ImageResource gbColor_;
    ImageResource gbMotion_;
    ImageResource gbDepth_;
    ImageResource gbReactive_; // R16F translucent coverage mask (render res)
    ImageResource gbAlbedo_;
    ImageResource gbNormal_;
    ImageResource gbMaterial_;
    ImageResource gbEmissive_;
    ImageResource gtDepth_;
    ImageResource gtAlbedo_;
    ImageResource gtNormal_;
    ImageResource gtMaterial_;
    ImageResource gtEmissive_;
    ImageResource finalImage_;
    // GTAO working target (RG16F: AO + view Z) and filtered R16F, GBuffer res.
    ImageResource gbAoRaw_;
    ImageResource gbAo_;
    ImageResource gtAoRaw_;
    ImageResource gtAo_;
    ImageResource gbBloomA_; // half-res ping-pong
    ImageResource gbBloomB_;
    ImageResource gtBloomA_;
    ImageResource gtBloomB_;
    ImageResource gbSsrSrc_; // opaque HDR copy for glass SSR
    ImageResource gtSsrSrc_;

    // Shared deferred pipeline (shaders/layouts/pipelines/samplers + IBL maps).
    DeferredCore deferred_;

    // CSM sun shadow targets (fixed 2048^2 x 4, resolution-independent).
    // shadowsActive_ is false when creation failed or --no-shadows; the UBO
    // then keeps shadowParams.z = 0 and the shaders short-circuit.
    ShadowTargets shadow_;
    bool shadowsActive_ = false;
    float iblIntensity_ = 1.f; // from lightingPresetForScene; not a CLI flag

    VkDescriptorSetLayout presentSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout presentPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline presentPipeline_ = VK_NULL_HANDLE;
    VkShaderModule presentFrag_ = VK_NULL_HANDLE;

    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    FrameResources frames_[kFramesInFlight] = {};
    std::vector<VkSemaphore> renderFinished_; // one per swapchain image (present sync)

    VkBuffer materialUbo_ = VK_NULL_HANDLE;
    VmaAllocation materialUboMemory_ = VK_NULL_HANDLE;
    uint32_t materialStride_ = 0;
    VkDescriptorSet textureSet_ = VK_NULL_HANDLE;
    VkDescriptorSet presentSet_ = VK_NULL_HANDLE;
    // SSAO descriptor sets (static: the referenced textures never change).
    VkDescriptorSet ssaoSetGb_ = VK_NULL_HANDLE;
    VkDescriptorSet ssaoBlurSetGb_ = VK_NULL_HANDLE;
    VkDescriptorSet ssaoSetGt_ = VK_NULL_HANDLE;
    VkDescriptorSet ssaoBlurSetGt_ = VK_NULL_HANDLE;
    VkDescriptorSet bloomExtractGb_ = VK_NULL_HANDLE;
    VkDescriptorSet bloomBlurHGb_ = VK_NULL_HANDLE;
    VkDescriptorSet bloomBlurVGb_ = VK_NULL_HANDLE;
    VkDescriptorSet bloomCompGb_ = VK_NULL_HANDLE;
    VkDescriptorSet bloomExtractGt_ = VK_NULL_HANDLE;
    VkDescriptorSet bloomBlurHGt_ = VK_NULL_HANDLE;
    VkDescriptorSet bloomBlurVGt_ = VK_NULL_HANDLE;
    VkDescriptorSet bloomCompGt_ = VK_NULL_HANDLE;

    VkBuffer screenshotStaging_ = VK_NULL_HANDLE;
    VmaAllocation screenshotStagingMemory_ = VK_NULL_HANDLE;
    void* screenshotMapped_ = nullptr;
    VkDeviceSize screenshotSize_ = 0;

    TimestampQuery timestamps_;
    std::vector<TimestampQuery::Timings> frameTimes_;  // per-frame, when frameTimesPath set

    bool createRenderTargets();
    bool createShaders();
    bool createSceneDescriptors();
    bool createPipelines();
    bool createSyncResources();
    bool createScreenshotStaging();
    bool recreateSwapchain(uint32_t width, uint32_t height, bool vsync);
    void updateSceneUBO(uint32_t frameIndex, bool jitter, uint32_t renderW, uint32_t renderH,
                        const Mat4& view, const Mat4& proj, const Mat4& projJittered,
                        const Mat4& prevViewProj);
    void updateLightingUBO(uint32_t frameIndex, const Mat4& invViewProj, const ShadowFrame* shadow);
    void updateCamera(uint32_t frameIndex, float dt);
    void applyCameraKeyframe(uint32_t frameIndex);
    void recordFrame(uint32_t frameIndex, uint32_t swapchainIndex);
    void captureScreenshotIntoStaging(VkCommandBuffer cmd);
    void saveScreenshot(const std::string& path);
    bool loadShader(const char* name, VkShaderModule& out);
};

} // namespace sr
