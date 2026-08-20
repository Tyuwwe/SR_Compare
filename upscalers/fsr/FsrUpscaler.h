#pragma once
// ============================================================================
// AMD FSR1 / FSR2 / FSR3.1 upscalers via the FidelityFX SDK host components
// and its Vulkan backend.  Implements sr::IUpscaler; registered as "fsr1",
// "fsr2", "fsr3".
// ============================================================================
#include "upscalers/IUpscaler.h"

namespace sr {

class Fsr1Upscaler : public IUpscaler {
public:
    Fsr1Upscaler() = default;
    ~Fsr1Upscaler() override;

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

class Fsr2Upscaler : public IUpscaler {
public:
    Fsr2Upscaler() = default;
    ~Fsr2Upscaler() override;

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

class Fsr3Upscaler : public IUpscaler {
public:
    Fsr3Upscaler() = default;
    ~Fsr3Upscaler() override;

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
