#pragma once
// ============================================================================
// DLSS Super Resolution upscaler via NVIDIA Streamline (Vulkan).
// One class, three registered plugins differing only in the DLSS preset:
//   "dlss-k" -> sl::DLSSPreset::ePresetK (transformer, quality/balanced default)
//   "dlss-l" -> sl::DLSSPreset::ePresetL (transformer, ultra-performance default)
//   "dlss-m" -> sl::DLSSPreset::ePresetM (transformer, performance default)
// ============================================================================
#include "upscalers/IUpscaler.h"

#include <memory>

// Forward declaration avoids pulling the (non self-contained) SL headers
// into every translation unit that includes this header.
namespace sl {
enum class DLSSPreset : uint32_t;
}

namespace sr {

class DlssUpscaler : public IUpscaler {
public:
    DlssUpscaler(sl::DLSSPreset preset, const char* displayName);
    ~DlssUpscaler() override;

    const char* name() const override;
    uint32_t capabilities() const override;
    bool isAvailable(const VulkanEnv& env) override;
    bool init(const VulkanEnv& env, const UpscalerDesc& desc) override;
    void dispatch(VkCommandBuffer cmd, const UpscalerResources& res, const CameraParams& cam,
                  const FrameParams& frame) override;
    void shutdown() override;
    uint64_t gpuMemoryBytes() const override;

private:
    sl::DLSSPreset preset_;
    const char* displayName_;
    VulkanEnv env_ = {};
    UpscalerDesc desc_ = {};
    uint32_t viewportId_ = 0;   // sl::ViewportHandle wrapper, built on demand
    bool addedRef_ = false;     // SlContext::addRef() succeeded
    bool ready_ = false;        // init succeeded, safe to dispatch
    bool vramQueried_ = false;
    uint64_t vramBytes_ = 0;
    int loggedFrames_ = 0;      // throttle per-frame slResult logging
};

std::unique_ptr<IUpscaler> createDlssKUpscaler();
std::unique_ptr<IUpscaler> createDlssLUpscaler();
std::unique_ptr<IUpscaler> createDlssMUpscaler();

} // namespace sr
