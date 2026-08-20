#pragma once
// ============================================================================
// XeSS — Intel XeSS Super Resolution (Vulkan path) plugin.
// Temporal ML upscaler: consumes render-resolution HDR color, low-res motion
// vectors (pixel units, no jitter) and depth; XeSS dilates/up-samples the MVs
// internally.  On non-Intel GPUs it runs the cross-vendor DP4a kernel.
// Implements sr::IUpscaler.
// ============================================================================
#include "upscalers/IUpscaler.h"

namespace sr {

class XessUpscaler : public IUpscaler {
public:
    XessUpscaler() = default;
    ~XessUpscaler() override;

    const char* name() const override;
    uint32_t capabilities() const override;
    bool isAvailable(const VulkanEnv& env) override;
    bool init(const VulkanEnv& env, const UpscalerDesc& desc) override;
    void dispatch(VkCommandBuffer cmd, const UpscalerResources& res, const CameraParams& cam,
                  const FrameParams& frame) override;
    void shutdown() override;
    uint64_t gpuMemoryBytes() const override;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

} // namespace sr
