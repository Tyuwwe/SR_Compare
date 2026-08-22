#pragma once
// ============================================================================
// RenderGraph — a deliberately minimal render-graph skeleton (Phase: pass
// dependency explicitation).  Not a Frostbite/UE-style graph: passes execute
// strictly in registration order (the hosts' chains are linear), there is no
// pass culling or reordering, and buffers stay hand-synced inside the pass
// record lambdas.  What it does provide:
//
//   * Pass registration: name + declared image reads/writes + a record lambda.
//   * Automatic barrier derivation: a persistent cross-frame per-image state
//     tracker (layout/stage/access) turns the declared accesses into the exact
//     VkImageMemoryBarrier2 set, replacing the hosts' hand-written layout
//     variables and imageBarrier call sites.  Hazard rule: a barrier is
//     emitted when the layout changes, when the new access writes, or when the
//     previous access wrote (read-after-read in the same layout is free).
//     One batched vkCmdPipelineBarrier2 per pass.
//   * Transient aliasing: TransientImageArena binds several intra-frame
//     intermediate targets with provably non-overlapping lifetimes onto one
//     VMA-backed memory block (every image binds at the same block base).
//
// A pass whose record lambda performs internal transitions of a tracked image
// (the DeferredCore helpers taking VkImageLayout&) must declare its exit state
// with Pass::leaves(); the tracker otherwise assumes the last declared access.
// Images touched outside the graph (swapchain acquire, upscaler internals,
// one-shot init transitions) are synced via adopt()/discard().
//
// Debug: SR_RG_TRACE=1 logs every derived barrier per pass to stderr.
// ============================================================================
#include "renderer/core/Vk.h"
#include "renderer/core/VulkanContext.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace sr {

// Persistent access state of one image (cross-frame).
struct RgImageState {
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 access = VK_ACCESS_2_NONE;
};

// One declared image access of a pass.
struct RgAccess {
    VkImage image = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 access = VK_ACCESS_2_NONE;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    uint32_t baseLayer = 0; // whole-image transitions by default
    uint32_t layerCount = 1;
};

class RenderGraph {
public:
    struct Pass {
        std::string name;
        std::vector<RgAccess> accesses; // entry requirements, in declaration order
        RgAccess exitOverride;          // set by leaves()
        bool hasExitOverride = false;
        std::function<void(VkCommandBuffer)> record;

        // Declares a required access at pass entry; the matching barrier is
        // derived from the tracked previous state.  read/write differ only in
        // intent documentation — the hazard rule keys off the access mask.
        Pass& access(VkImage image, VkImageLayout layout, VkPipelineStageFlags2 stage,
                     VkAccessFlags2 accessMask,
                     VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                     uint32_t baseLayer = 0, uint32_t layerCount = 1);
        // Records the image state after the record lambda ran, for passes that
        // transition tracked images internally (DeferredCore helpers).  Only
        // state bookkeeping — no barrier is emitted.
        Pass& leaves(VkImage image, VkImageLayout layout, VkPipelineStageFlags2 stage,
                     VkAccessFlags2 accessMask);
    };

    // Registration order is execution order.
    Pass& addPass(std::string name);
    // Drops the pass list; image states persist across frames.
    void clear() { passes_.clear(); }

    // Derives and records each pass's barriers, then its record lambda.
    void execute(VkCommandBuffer cmd);

    // --- cross-frame state management -----------------------------------------
    // Seeds/overrides the tracked state after work recorded outside the graph
    // (init-time one-shot transitions, upscaler-internal copies).
    void adopt(VkImage image, RgImageState state);
    // Marks the contents undefined (swapchain acquire, resize): the next pass
    // gets an UNDEFINED-sourced transition with no source synchronization.
    void discard(VkImage image);
    RgImageState state(VkImage image) const;
    // Mutable layout reference into the tracked state, for legacy helpers that
    // still take VkImageLayout& (recordBloomPyramidPass / recordPostFxPass).
    // Pair the pass with leaves() so the stage/access stay in sync.
    VkImageLayout& layoutRef(VkImage image);

private:
    std::vector<Pass> passes_;
    std::unordered_map<VkImage, RgImageState> states_;
};

// --- Transient image aliasing --------------------------------------------------
// One VMA-allocated device-local block; every image created from the arena is
// bound at the same block base, so ALL images of one arena alias each other.  The host
// must guarantee their lifetimes never overlap within or across frames (e.g.
// the GTAO raw target dies before the SSR trace target is written).  Images
// are persistent objects with per-image views/descriptors; only the memory is
// shared.  VRAM reporting stays inside VMA (the block is one VmaAllocation).
class TransientImageArena {
public:
    // Requirements probe for sizing the block before init() (creates and
    // destroys a throwaway image).
    static bool queryImageRequirements(const VulkanContext& ctx, uint32_t width, uint32_t height,
                                       VkFormat format, VkImageUsageFlags usage,
                                       VkDeviceSize& outBytes, VkDeviceSize& outAlignment,
                                       uint32_t& outMemoryTypeBits);
    // bytes must cover the largest single image; alignment the largest
    // alignment; memoryTypeBits is the intersection of all member images'
    // type bits.  On any failure the caller falls back to plain createImage
    // allocation.
    bool init(const VulkanContext& ctx, uint32_t memoryTypeBits, VkDeviceSize bytes,
              VkDeviceSize alignment);
    void destroy(const VulkanContext& ctx);
    // Creates an image bound to the shared block base.  Returns
    // VK_NULL_HANDLE on failure (caller falls back to createImage).
    VkImage createAliasedImage(const VulkanContext& ctx, uint32_t width, uint32_t height,
                               VkFormat format, VkImageUsageFlags usage);
    void destroyImage(const VulkanContext& ctx, VkImage image);
    bool valid() const { return allocation_ != VK_NULL_HANDLE; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE; // backing block
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize baseOffset_ = 0; // the VMA suballocation's offset within memory_
    VkDeviceSize blockBytes_ = 0;
    uint32_t memoryTypeBits_ = 0;
    std::vector<VkImage> images_;
};

} // namespace sr
