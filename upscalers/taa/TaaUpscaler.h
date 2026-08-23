#pragma once
// ============================================================================
// TAAU — the self-authored temporal baseline.  YCoCg-space variance clipping
// + depth disocclusion rejection + velocity-adaptive history weight, resolved
// from render-resolution HDR/depth/motion inputs to display resolution, then
// FidelityFX CAS (RCAS) sharpening.  Implements sr::IUpscaler.
// ============================================================================
#include "upscalers/IUpscaler.h"

namespace sr {

class TaaUpscaler : public IUpscaler {
public:
    TaaUpscaler() = default;
    ~TaaUpscaler() override;

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
