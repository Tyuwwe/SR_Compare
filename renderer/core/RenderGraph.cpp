#include "renderer/core/RenderGraph.h"

#include "renderer/core/Vma.h"

#include <cstdio>
#include <cstdlib>

namespace sr {

namespace {

bool isWriteAccess(VkAccessFlags2 access) {
    constexpr VkAccessFlags2 kWrites =
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT |
        VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_HOST_WRITE_BIT;
    return (access & kWrites) != 0;
}

bool traceEnabled() {
    // Same _dupenv_s pattern as app/CliUtils.h envFlag (plain getenv trips
    // C4996 under /W4); duplicated here to keep renderer/ free of app/.
    static const bool on = [] {
#ifdef _MSC_VER
        char* value = nullptr;
        size_t len = 0;
        const bool set = _dupenv_s(&value, &len, "SR_RG_TRACE") == 0 && value != nullptr;
        std::free(value);
        return set;
#else
        return std::getenv("SR_RG_TRACE") != nullptr;
#endif
    }();
    return on;
}

const char* layoutName(VkImageLayout layout) {
    // if-chain instead of a switch: /W4 C4061 fires on enum switches even
    // with a default arm.
    if (layout == VK_IMAGE_LAYOUT_UNDEFINED) return "UNDEFINED";
    if (layout == VK_IMAGE_LAYOUT_GENERAL) return "GENERAL";
    if (layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) return "COLOR_ATTACHMENT";
    if (layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) return "DEPTH_ATTACHMENT";
    if (layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL) return "DEPTH_READ_ONLY";
    if (layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) return "SHADER_READ";
    if (layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) return "TRANSFER_SRC";
    if (layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) return "TRANSFER_DST";
    if (layout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) return "PRESENT_SRC";
    return "?";
}

} // namespace

RenderGraph::Pass& RenderGraph::Pass::access(VkImage image, VkImageLayout layout,
                                             VkPipelineStageFlags2 stage,
                                             VkAccessFlags2 accessMask,
                                             VkImageAspectFlags aspect, uint32_t baseLayer,
                                             uint32_t layerCount) {
    RgAccess a;
    a.image = image;
    a.layout = layout;
    a.stage = stage;
    a.access = accessMask;
    a.aspect = aspect;
    a.baseLayer = baseLayer;
    a.layerCount = layerCount;
    accesses.push_back(a);
    return *this;
}

RenderGraph::Pass& RenderGraph::Pass::leaves(VkImage image, VkImageLayout layout,
                                             VkPipelineStageFlags2 stage,
                                             VkAccessFlags2 accessMask) {
    exitOverride.image = image;
    exitOverride.layout = layout;
    exitOverride.stage = stage;
    exitOverride.access = accessMask;
    hasExitOverride = true;
    return *this;
}

RenderGraph::Pass& RenderGraph::addPass(std::string name) {
    passes_.push_back(Pass{});
    passes_.back().name = std::move(name);
    return passes_.back();
}

void RenderGraph::adopt(VkImage image, RgImageState state) {
    states_[image] = state;
}

void RenderGraph::discard(VkImage image) {
    states_[image] = RgImageState{};
}

RgImageState RenderGraph::state(VkImage image) const {
    const auto it = states_.find(image);
    return it != states_.end() ? it->second : RgImageState{};
}

VkImageLayout& RenderGraph::layoutRef(VkImage image) {
    return states_[image].layout;
}

void RenderGraph::execute(VkCommandBuffer cmd) {
    const bool trace = traceEnabled();
    for (Pass& pass : passes_) {
        // Derive the barrier set from the tracked state vs the declared
        // accesses.  Passes may declare the same image once; the DeferredCore
        // chains never need two states of one image within a single pass.
        VkImageMemoryBarrier2 barriers[16];
        uint32_t barrierCount = 0;
        auto flush = [&]() {
            if (barrierCount == 0) return;
            VkDependencyInfo dep = {};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = barrierCount;
            dep.pImageMemoryBarriers = barriers;
            vkCmdPipelineBarrier2(cmd, &dep);
            barrierCount = 0;
        };
        for (const RgAccess& a : pass.accesses) {
            RgImageState& cur = states_[a.image];
            const bool layoutChange = cur.layout != a.layout;
            // Hazard rule: any write on either side (or a layout change)
            // needs synchronization; read-after-read in one layout is free.
            const bool hazard = isWriteAccess(a.access) || isWriteAccess(cur.access);
            if (!layoutChange && !hazard) {
                // Unordered reads: accumulate the consumer scope so a later
                // hazard barrier still covers every reader since the last one.
                cur.stage |= a.stage;
                cur.access |= a.access;
                continue;
            }
            if (barrierCount == 16) flush(); // chunk long declaration lists
            {
                VkImageMemoryBarrier2& b = barriers[barrierCount++];
                b = {};
                b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                // UNDEFINED discards the contents: no source synchronization.
                b.srcStageMask = cur.layout == VK_IMAGE_LAYOUT_UNDEFINED
                                     ? VK_PIPELINE_STAGE_2_NONE
                                     : cur.stage;
                b.srcAccessMask =
                    cur.layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_ACCESS_2_NONE : cur.access;
                b.dstStageMask = a.stage;
                b.dstAccessMask = a.access;
                b.oldLayout = cur.layout;
                b.newLayout = a.layout;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = a.image;
                b.subresourceRange.aspectMask = a.aspect;
                b.subresourceRange.baseMipLevel = 0;
                b.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
                b.subresourceRange.baseArrayLayer = a.baseLayer;
                b.subresourceRange.layerCount = a.layerCount;
                if (trace) {
                    std::fprintf(stderr,
                                 "[rg] %-16s %s -> %s (src stage %llx access %llx)\n",
                                 pass.name.c_str(), layoutName(cur.layout), layoutName(a.layout),
                                 static_cast<unsigned long long>(b.srcStageMask),
                                 static_cast<unsigned long long>(b.srcAccessMask));
                }
            }
            cur.layout = a.layout;
            cur.stage = a.stage;
            cur.access = a.access;
        }
        flush();
        if (pass.record) pass.record(cmd);
        // Passes with internal transitions of a tracked image (the DeferredCore
        // helpers taking VkImageLayout&) report the resulting state here.
        if (pass.hasExitOverride) {
            RgImageState& cur = states_[pass.exitOverride.image];
            cur.layout = pass.exitOverride.layout;
            cur.stage = pass.exitOverride.stage;
            cur.access = pass.exitOverride.access;
        }
    }
}

// --- TransientImageArena --------------------------------------------------------

bool TransientImageArena::queryImageRequirements(const VulkanContext& ctx, uint32_t width,
                                                 uint32_t height, VkFormat format,
                                                 VkImageUsageFlags usage, VkDeviceSize& outBytes,
                                                 VkDeviceSize& outAlignment,
                                                 uint32_t& outMemoryTypeBits) {
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = {width, height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    VkImage probe = VK_NULL_HANDLE;
    if (vkCreateImage(ctx.device, &ci, nullptr, &probe) != VK_SUCCESS) return false;
    VkMemoryRequirements req = {};
    vkGetImageMemoryRequirements(ctx.device, probe, &req);
    vkDestroyImage(ctx.device, probe, nullptr);
    outBytes = req.size;
    outAlignment = req.alignment;
    outMemoryTypeBits = req.memoryTypeBits;
    return true;
}

bool TransientImageArena::init(const VulkanContext& ctx, uint32_t memoryTypeBits,
                               VkDeviceSize bytes, VkDeviceSize alignment) {
    if (bytes == 0 || memoryTypeBits == 0) return false;
    device_ = ctx.device;
    blockBytes_ = bytes;
    memoryTypeBits_ = memoryTypeBits;

    // Raw vmaAllocateMemory has no resource to derive VMA_MEMORY_USAGE_AUTO
    // preferences from (it would fail with VK_ERROR_FEATURE_NOT_PRESENT), so
    // require DEVICE_LOCAL explicitly — the same effective choice
    // VkUtil::createImage makes through vmaCreateImage.  The returned
    // allocation may be a suballocation of a larger VMA block, hence
    // baseOffset_.
    VkMemoryRequirements req = {};
    req.size = bytes;
    req.alignment = alignment > 0 ? alignment : 1;
    req.memoryTypeBits = memoryTypeBits;
    VmaAllocationCreateInfo allocCi = {};
    allocCi.usage = VMA_MEMORY_USAGE_UNKNOWN;
    allocCi.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VmaAllocationInfo info = {};
    const VkResult res =
        vmaAllocateMemory(ctx.allocator, &req, &allocCi, &allocation_, &info);
    if (res != VK_SUCCESS) {
        if (traceEnabled()) {
            std::fprintf(stderr, "[rg] arena init failed (%d): %llu bytes, align %llu, typeBits 0x%x\n",
                         static_cast<int>(res), static_cast<unsigned long long>(bytes),
                         static_cast<unsigned long long>(alignment), memoryTypeBits);
            for (uint32_t i = 0; i < ctx.memoryProperties.memoryTypeCount; ++i)
                std::fprintf(stderr, "[rg]   memType %u: flags 0x%x heap %u (size %llu MB)%s\n", i,
                             ctx.memoryProperties.memoryTypes[i].propertyFlags,
                             ctx.memoryProperties.memoryTypes[i].heapIndex,
                             static_cast<unsigned long long>(
                                 ctx.memoryProperties.memoryHeaps[ctx.memoryProperties.memoryTypes[i].heapIndex].size /
                                 (1024 * 1024)),
                             (memoryTypeBits & (1u << i)) ? "  <allowed>" : "");
        }
        return false;
    }
    memory_ = info.deviceMemory;
    baseOffset_ = info.offset;
    if (traceEnabled())
        std::fprintf(stderr, "[rg] arena init: %llu bytes, typeBits 0x%x\n",
                     static_cast<unsigned long long>(bytes), memoryTypeBits);
    return true;
}

void TransientImageArena::destroy(const VulkanContext& ctx) {
    for (VkImage image : images_)
        if (image) vkDestroyImage(ctx.device, image, nullptr);
    images_.clear();
    if (allocation_) {
        vmaFreeMemory(ctx.allocator, allocation_);
        allocation_ = VK_NULL_HANDLE;
        memory_ = VK_NULL_HANDLE;
    }
    device_ = VK_NULL_HANDLE;
}

VkImage TransientImageArena::createAliasedImage(const VulkanContext& ctx, uint32_t width,
                                                uint32_t height, VkFormat format,
                                                VkImageUsageFlags usage) {
    if (!allocation_) return VK_NULL_HANDLE;
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = {width, height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    VkImage image = VK_NULL_HANDLE;
    if (vkCreateImage(ctx.device, &ci, nullptr, &image) != VK_SUCCESS) {
        if (traceEnabled()) std::fprintf(stderr, "[rg] arena image creation failed\n");
        return VK_NULL_HANDLE;
    }

    VkMemoryRequirements req = {};
    vkGetImageMemoryRequirements(ctx.device, image, &req);
    // Every image of the arena binds at the same block base: they alias by
    // contract (the host guarantees non-overlapping lifetimes).  The block was
    // allocated with the maximum image alignment, so baseOffset_ satisfies
    // every image's alignment requirement.
    if (req.size > blockBytes_ || (req.memoryTypeBits & memoryTypeBits_) == 0 ||
        (req.alignment > 0 && (baseOffset_ % req.alignment) != 0) ||
        vkBindImageMemory(ctx.device, image, memory_, baseOffset_) != VK_SUCCESS) {
        if (traceEnabled())
            std::fprintf(stderr,
                         "[rg] arena bind failed: size %llu (block %llu) typeBits 0x%x/0x%x\n",
                         static_cast<unsigned long long>(req.size),
                         static_cast<unsigned long long>(blockBytes_), req.memoryTypeBits,
                         memoryTypeBits_);
        vkDestroyImage(ctx.device, image, nullptr);
        return VK_NULL_HANDLE;
    }
    images_.push_back(image);
    return image;
}

void TransientImageArena::destroyImage(const VulkanContext& ctx, VkImage image) {
    for (auto it = images_.begin(); it != images_.end(); ++it) {
        if (*it == image) {
            vkDestroyImage(ctx.device, image, nullptr);
            images_.erase(it);
            return;
        }
    }
}

} // namespace sr
