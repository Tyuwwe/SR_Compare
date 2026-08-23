#pragma once
// ============================================================================
// GpuProfiler — per-pass GPU timestamp zones for the profiler overlay.
//
// Record side: scoped zones emit vkCmdWriteTimestamp pairs into a per
// frame-in-flight slot of one big query pool.  Zones nest (the depth counter
// tracks the RAII scope stack), so a pass group and its sub-passes read as a
// hierarchy.  Use the SR_GPU_ZONE macro:
//
//     profiler.beginFrame(cmd, slot, frameIndex);
//     { SR_GPU_ZONE(profiler, cmd, "lighting"); ... }
//     // ... later, after the slot's fence signaled:
//     profiler.harvest(ctx, slot);
//
// Read side: harvest() appends one Frame (zones in record order, inclusive +
// self milliseconds) to a fixed-length history ring consumed by the UI.
//
// Overhead: when disabled (panel closed), beginFrame/beginZone/endZone are
// branch-only no-ops — no query reset, no timestamp writes.  Readback uses
// the same fence-synchronized slot reuse as TimestampQuery (2 frames of
// latency, non-blocking).
// ============================================================================
#include "renderer/core/Vk.h"
#include "renderer/core/VulkanContext.h"

#include <cstdint>
#include <string>
#include <vector>

namespace sr {

class GpuProfiler {
public:
    struct Zone {
        std::string name;
        uint32_t depth = 0; // RAII nesting level (0 = top-level pass group)
        float ms = 0.f;     // inclusive (children included)
        float selfMs = 0.f; // inclusive minus direct children (stacking uses this)
    };
    struct Frame {
        std::vector<Zone> zones; // record (begin) order
        float totalMs = 0.f;     // first zone begin -> last zone end
        uint64_t frameIndex = 0;
    };

    static constexpr uint32_t kMaxZonesPerFrame = 96;
    static constexpr uint32_t kHistoryLen = 240;

    bool create(const VulkanContext& ctx, uint32_t framesInFlight);
    void destroy(const VulkanContext& ctx);

    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool enabled() const { return enabled_; }

    // Start of a recorded frame: resets the slot's query range.  Safe to call
    // unconditionally; a disabled profiler marks the slot inactive so a later
    // harvest() ignores it.
    void beginFrame(VkCommandBuffer cmd, uint32_t slot, uint64_t frameIndex);
    // End of the recorded frame: later beginZone/endZone calls (e.g. from a
    // second present recorded on another command buffer) become no-ops.
    void endFrame() { recordingSlot_ = -1; }
    void beginZone(VkCommandBuffer cmd, const char* name);
    void endZone(VkCommandBuffer cmd);

    // Append the frame recorded into `slot` to the history ring.  Call only
    // after that frame's fence has signaled.  Returns false when the slot has
    // no recorded frame (profiler was off) or the results are not ready.
    bool harvest(const VulkanContext& ctx, uint32_t slot);

    // History ring, oldest first: count() entries, at(0) = oldest.
    uint32_t count() const { return historyCount_; }
    const Frame& at(uint32_t i) const {
        return history_[(historyHead_ + i) % kHistoryLen];
    }
    const Frame& latest() const { return at(historyCount_ - 1); }

    // RAII scope for SR_GPU_ZONE.  Copying the name happens only when enabled.
    class ZoneScope {
    public:
        ZoneScope(GpuProfiler& profiler, VkCommandBuffer cmd, const char* name)
            : profiler_(profiler), cmd_(cmd), active_(profiler.enabled()) {
            if (active_) profiler_.beginZone(cmd_, name);
        }
        ~ZoneScope() {
            if (active_) profiler_.endZone(cmd_);
        }
        ZoneScope(const ZoneScope&) = delete;
        ZoneScope& operator=(const ZoneScope&) = delete;

    private:
        GpuProfiler& profiler_;
        VkCommandBuffer cmd_;
        bool active_;
    };

private:
    struct PendingZone {
        std::string name;
        uint32_t depth = 0;
        uint32_t query = 0; // begin timestamp index (end = query + 1)
    };
    struct PendingFrame {
        std::vector<PendingZone> zones;
        uint32_t depth = 0;        // current nesting level during recording
        uint32_t skippedDepth = 0; // zones dropped by the kMaxZonesPerFrame cap
        uint64_t frameIndex = 0;
        bool active = false;
    };

    VkQueryPool pool_ = VK_NULL_HANDLE;
    uint32_t framesInFlight_ = 0;
    float periodNs_ = 0.f;
    bool enabled_ = false;
    std::vector<PendingFrame> pending_; // one per frame-in-flight slot
    int32_t recordingSlot_ = -1;        // slot of the frame being recorded (-1 = off)
    Frame history_[kHistoryLen];
    uint32_t historyHead_ = 0; // oldest entry
    uint32_t historyCount_ = 0;
};

#define SR_GPU_CONCAT_INNER(a, b) a##b
#define SR_GPU_CONCAT(a, b) SR_GPU_CONCAT_INNER(a, b)
// Scoped GPU zone; compiles to a disabled-check + two timestamp writes.
#define SR_GPU_ZONE(profiler, cmd, name) \
    sr::GpuProfiler::ZoneScope SR_GPU_CONCAT(srGpuZone_, __LINE__)(profiler, cmd, name)

} // namespace sr
