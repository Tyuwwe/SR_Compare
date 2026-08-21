#pragma once
// ============================================================================
// IFrameGen — interpolates one display-rate frame between two real frames.
// Independent of IUpscaler: runs after upscale, writes a display-res HDR
// image.  FSR3-FI and NFRU implement this.  Do not hijack the swapchain.
// ============================================================================
#include "upscalers/IUpscaler.h"

#include <memory>
#include <string>
#include <vector>

namespace sr {

struct FrameGenDesc {
    uint32_t renderWidth = 0, renderHeight = 0;
    uint32_t displayWidth = 0, displayHeight = 0;
    bool hdr = true;
    bool invertedDepth = false;
    bool infiniteFarPlane = true;
};

struct FrameGenResources {
    // Current-frame upscaled color (display res, SHADER_READ_ONLY).
    VkImage     color     = VK_NULL_HANDLE;
    VkImageView colorView = VK_NULL_HANDLE;
    // Render-res depth / motion (same convention as IUpscaler).
    VkImage     depth     = VK_NULL_HANDLE;
    VkImageView depthView = VK_NULL_HANDLE;
    VkImage     motion    = VK_NULL_HANDLE;
    VkImageView motionView = VK_NULL_HANDLE;
    // Interpolated output, display res, GENERAL.
    VkImage     output     = VK_NULL_HANDLE;
    VkImageView outputView = VK_NULL_HANDLE;
};

class IFrameGen {
public:
    virtual ~IFrameGen() = default;
    virtual const char* name() const = 0;
    virtual bool isAvailable(const VulkanEnv& env) = 0;
    virtual bool init(const VulkanEnv& env, const FrameGenDesc& desc) = 0;
    virtual void dispatch(VkCommandBuffer cmd, const FrameGenResources& res,
                          const CameraParams& cam, const FrameParams& frame) = 0;
    virtual void shutdown() = 0;
    virtual uint64_t gpuMemoryBytes() const = 0;
};

using FrameGenCreateFn = std::unique_ptr<IFrameGen> (*)();

bool registerFrameGen(const char* name, FrameGenCreateFn fn);
std::unique_ptr<IFrameGen> createFrameGen(const char* name);
std::vector<std::string> listFrameGens();

} // namespace sr
