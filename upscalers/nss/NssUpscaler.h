#pragma once
// ============================================================================
// NSS — Arm Neural Super Sampling (Neural Graphics SDK for Game Engines).
//
// NSS runs a quantized int8 data-graph (VGF model) through the Vulkan ML
// extensions VK_ARM_tensors / VK_ARM_data_graph.  On PC there is no native
// hardware support, so the extensions are provided by Arm's ML Emulation
// Layer (VK_LAYER_ML_Graph_Emulation + VK_LAYER_ML_Tensor_Emulation), which
// must be discoverable by the Vulkan loader at process start (see
// run_with_nss.bat).
//
// Input/output adaptation:
//   * color/output are converted between our R16G16B16A16_SFLOAT and the
//     SDK-mandated R11G11B10 float via two small compute passes.
//   * depth (D32_SFLOAT) and motion (R16G16_SFLOAT) are passed through.
//     Motion is backward framebuffer UV; NSS expects backward pixel motion,
//     so motionVectorScale = (renderWidth, renderHeight) converts the units.
// ============================================================================
#include "upscalers/IUpscaler.h"

namespace sr {

class NssUpscaler : public IUpscaler {
public:
    NssUpscaler() = default;
    ~NssUpscaler() override;

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
