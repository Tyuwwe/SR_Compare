#pragma once
// ============================================================================
// IUpscaler — 超分插件统一契约
// 所有超分算法（TAA/FSR/XeSS/DLSS/SGSR/NSS）都实现此接口。
// 渲染器负责提供同一份低分辨率输入（color/depth/motion/jitter），
// 各插件经 InputAdapter 约定转换为自家格式后执行上采样。
// ============================================================================
#include <vulkan/vulkan.h>
#include <cstdint>
#include <mutex>

namespace sr {

// 矩阵均为列主序 float[16]（GLM/Vulkan 约定）
struct CameraParams {
    float view[16];            // 当前帧 view 矩阵
    float proj[16];            // 当前帧 projection（不含 jitter）
    float prevViewProj[16];    // 上一帧 proj*view（不含 jitter），用于重投影
    float jitterX = 0.f;       // 当前帧亚像素抖动，单位：像素，[-0.5, 0.5]
    float jitterY = 0.f;
    float prevJitterX = 0.f;
    float prevJitterY = 0.f;
    float cameraNear = 0.1f;
    float cameraFar = 1000.f;  // infiniteFarPlane 时忽略
    float fovY = 0.f;          // 垂直视场角，弧度
};

struct FrameParams {
    int   frameIndex = 0;
    float deltaTime = 0.f;         // 秒
    float preExposure = 1.f;       // 无自动曝光时为 1
    bool  resetHistory = false;    // 切场景/改分辨率/切算法首帧置真
};

struct UpscalerDesc {
    uint32_t renderWidth = 0, renderHeight = 0;    // 输入（渲染）分辨率
    uint32_t displayWidth = 0, displayHeight = 0;  // 输出（目标）分辨率
    bool hdr = true;
    bool invertedDepth = false;   // 项目约定：不使用 reverse-Z 时为 false
    bool infiniteFarPlane = true;
};

struct UpscalerResources {
    // 输入：渲染分辨率。dispatch 时资源状态由渲染器保证：
    //   color/motion -> SHADER_READ_ONLY_OPTIMAL，depth -> DEPTH_STENCIL_READ_ONLY 或 SHADER_READ_ONLY，
    //   output -> GENERAL（storage）。算法内部不得销毁这些资源。
    VkImage     color      = VK_NULL_HANDLE;  // R16G16B16A16_SFLOAT，HDR 线性
    VkImageView colorView  = VK_NULL_HANDLE;
    VkImage     depth      = VK_NULL_HANDLE;  // D32_SFLOAT
    VkImageView depthView  = VK_NULL_HANDLE;
    VkImage     motion     = VK_NULL_HANDLE;  // R16G16_SFLOAT，像素单位、不含 jitter
    VkImageView motionView = VK_NULL_HANDLE;
    // 半透明覆盖度 mask（R16_SFLOAT，渲染分辨率，0=无透明，逐层累加 alpha）。
    // 供支持 reactive/transparency/bias/responsive mask 的算法使用
    //（TAA: 历史权重衰减；FSR2/3: reactive+TC；DLSS: bias/reactive/transparency hint；
    //  XeSS: responsive pixel mask）。场景无半透明时为 null。
    VkImage     reactive      = VK_NULL_HANDLE;
    VkImageView reactiveView  = VK_NULL_HANDLE;
    VkImage     output     = VK_NULL_HANDLE;  // R16G16B16A16_SFLOAT，输出分辨率
    VkImageView outputView = VK_NULL_HANDLE;
};

// 算法初始化所需的 Vulkan 环境（全部来自渲染器，算法不得自行创建 device）
struct VulkanEnv {
    VkInstance       instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice         device = VK_NULL_HANDLE;
    VkQueue          graphicsQueue = VK_NULL_HANDLE;
    uint32_t         graphicsQueueFamily = 0;
    VkCommandPool    commandPool = VK_NULL_HANDLE;  // 一次性命令提交用
    // 非空时：对 graphicsQueue 的 vkQueueSubmit/vkQueueWaitIdle 必须持锁
    // （GUI 异步加载期间工作线程与主线程会并发提交同一队列）。
    std::mutex*      queueMutex = nullptr;
    PFN_vkGetInstanceProcAddr getInstanceProcAddr = nullptr;
};

enum UpscalerCapability : uint32_t {
    Cap_Spatial  = 1u << 0,   // 纯空间，无时域依赖
    Cap_Temporal = 1u << 1,   // 需要 depth/MV/jitter/history
    Cap_ML       = 1u << 2,   // 神经网络推理
};

class IUpscaler {
public:
    virtual ~IUpscaler() = default;

    virtual const char* name() const = 0;                 // "TAA", "FSR2", "DLSS-K" ...
    virtual uint32_t capabilities() const = 0;            // UpscalerCapability 位或

    // 硬件/DLL/扩展可用性检查（不创建资源）；不可用则 init 不会被调用
    virtual bool isAvailable(const VulkanEnv& env) = 0;

    virtual bool init(const VulkanEnv& env, const UpscalerDesc& desc) = 0;

    // 在渲染器提供的 command buffer 内记录命令（录制状态）。
    // 算法自行管理内部资源与 barrier；输入/输出资源布局约定见 UpscalerResources。
    virtual void dispatch(VkCommandBuffer cmd,
                          const UpscalerResources& res,
                          const CameraParams& cam,
                          const FrameParams& frame) = 0;

    virtual void shutdown() = 0;

    // 算法内部分配的显存估计（字节），用于 benchmark 显存指标
    virtual uint64_t gpuMemoryBytes() const = 0;
};

// Spatial-only plugins (FSR1 / SGSR1) must not receive Halton jitter: they
// have no history to converge, so a per-frame sub-pixel offset just shakes.
inline bool upscalerNeedsJitter(const IUpscaler* u) {
    return u && (u->capabilities() & Cap_Temporal) != 0;
}

} // namespace sr
