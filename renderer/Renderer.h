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
        VkDeviceMemory memory = VK_NULL_HANDLE;
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
        VkDeviceMemory uboMemory = VK_NULL_HANDLE;
        void* uboMapped = nullptr;
        VkBuffer lightingUbo = VK_NULL_HANDLE;
        VkDeviceMemory lightingUboMemory = VK_NULL_HANDLE;
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
    // SSAO (raw + blurred) for the LR and GT paths, R16_SFLOAT, GBuffer res.
    ImageResource gbAoRaw_;
    ImageResource gbAo_;
    ImageResource gtAoRaw_;
    ImageResource gtAo_;

    // Shared deferred pipeline (shaders/layouts/pipelines/samplers + IBL maps).
    DeferredCore deferred_;

    VkDescriptorSetLayout presentSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout presentPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline presentPipeline_ = VK_NULL_HANDLE;
    VkShaderModule presentFrag_ = VK_NULL_HANDLE;

    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    FrameResources frames_[kFramesInFlight] = {};
    std::vector<VkSemaphore> renderFinished_; // one per swapchain image (present sync)

    VkBuffer materialUbo_ = VK_NULL_HANDLE;
    VkDeviceMemory materialUboMemory_ = VK_NULL_HANDLE;
    uint32_t materialStride_ = 0;
    VkDescriptorSet textureSet_ = VK_NULL_HANDLE;
    VkDescriptorSet presentSet_ = VK_NULL_HANDLE;
    // SSAO descriptor sets (static: the referenced textures never change).
    VkDescriptorSet ssaoSetGb_ = VK_NULL_HANDLE;
    VkDescriptorSet ssaoBlurSetGb_ = VK_NULL_HANDLE;
    VkDescriptorSet ssaoSetGt_ = VK_NULL_HANDLE;
    VkDescriptorSet ssaoBlurSetGt_ = VK_NULL_HANDLE;

    VkBuffer screenshotStaging_ = VK_NULL_HANDLE;
    VkDeviceMemory screenshotStagingMemory_ = VK_NULL_HANDLE;
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
    void updateSceneUBO(uint32_t frameIndex, bool jitter, uint32_t renderW, uint32_t renderH,
                        const Mat4& view, const Mat4& proj, const Mat4& projJittered,
                        const Mat4& prevViewProj);
    void updateLightingUBO(uint32_t frameIndex, const Mat4& invViewProj);
    void updateCamera(uint32_t frameIndex, float dt);
    void applyCameraKeyframe(uint32_t frameIndex);
    void recordFrame(uint32_t frameIndex, uint32_t swapchainIndex);
    void captureScreenshotIntoStaging(VkCommandBuffer cmd);
    void saveScreenshot(const std::string& path);
    bool loadShader(const char* name, VkShaderModule& out);
};

} // namespace sr
