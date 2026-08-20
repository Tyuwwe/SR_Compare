#pragma once
// ============================================================================
// SGSR1 — Qualcomm Snapdragon Game Super Resolution 1 (spatial).
// Single-pass fragment upscaler (12-tap Lanczos + adaptive sharpening),
// RGBA mode, color input only.  Implements sr::IUpscaler.
// ============================================================================
#include "upscalers/IUpscaler.h"

namespace sr {

class Sgsr1Upscaler : public IUpscaler {
public:
    Sgsr1Upscaler() = default;
    ~Sgsr1Upscaler() override;

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
