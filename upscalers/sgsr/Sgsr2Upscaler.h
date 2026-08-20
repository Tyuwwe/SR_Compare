#pragma once
// ============================================================================
// SGSR2 — Qualcomm Snapdragon Game Super Resolution 2 (temporal), 3-pass
// compute variant (Convert / Activate / Upscale) plus an internal motion
// encode pass adapting the renderer's pixel-unit motion vectors to the
// clip-space encoding SGSR2 expects.  Implements sr::IUpscaler.
// ============================================================================
#include "upscalers/IUpscaler.h"

namespace sr {

class Sgsr2Upscaler : public IUpscaler {
public:
    Sgsr2Upscaler() = default;
    ~Sgsr2Upscaler() override;

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
