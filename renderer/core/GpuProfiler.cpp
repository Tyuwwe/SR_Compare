#include "renderer/core/GpuProfiler.h"

#include <algorithm>

namespace sr {

bool GpuProfiler::create(const VulkanContext& ctx, uint32_t framesInFlight) {
    framesInFlight_ = framesInFlight;
    periodNs_ = ctx.properties.limits.timestampPeriod;
    pending_.resize(framesInFlight);

    VkQueryPoolCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = framesInFlight * kMaxZonesPerFrame * 2;
    if (vkCreateQueryPool(ctx.device, &info, nullptr, &pool_) != VK_SUCCESS) return false;
    return true;
}

void GpuProfiler::destroy(const VulkanContext& ctx) {
    if (pool_) vkDestroyQueryPool(ctx.device, pool_, nullptr);
    pool_ = VK_NULL_HANDLE;
    recordingSlot_ = -1;
    for (PendingFrame& p : pending_) {
        p.zones.clear();
        p.active = false;
    }
    historyHead_ = 0;
    historyCount_ = 0;
}

void GpuProfiler::beginFrame(VkCommandBuffer cmd, uint32_t slot, uint64_t frameIndex) {
    if (!enabled_) {
        recordingSlot_ = -1;
        pending_[slot].active = false;
        return;
    }
    PendingFrame& p = pending_[slot];
    p.zones.clear();
    p.depth = 0;
    p.skippedDepth = 0;
    p.frameIndex = frameIndex;
    p.active = true;
    recordingSlot_ = static_cast<int32_t>(slot);
    vkCmdResetQueryPool(cmd, pool_, slot * kMaxZonesPerFrame * 2, kMaxZonesPerFrame * 2);
}

void GpuProfiler::beginZone(VkCommandBuffer cmd, const char* name) {
    if (recordingSlot_ < 0) return;
    PendingFrame& p = pending_[static_cast<uint32_t>(recordingSlot_)];
    if (p.zones.size() >= kMaxZonesPerFrame) {
        // Cap hit: keep the depth books balanced so endZone stays paired.
        ++p.skippedDepth;
        return;
    }
    PendingZone z;
    z.name = name;
    z.depth = p.depth;
    z.query = static_cast<uint32_t>(recordingSlot_) * kMaxZonesPerFrame * 2 +
              static_cast<uint32_t>(p.zones.size()) * 2;
    ++p.depth;
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool_, z.query);
    p.zones.push_back(std::move(z));
}

void GpuProfiler::endZone(VkCommandBuffer cmd) {
    if (recordingSlot_ < 0) return;
    PendingFrame& p = pending_[static_cast<uint32_t>(recordingSlot_)];
    if (p.skippedDepth > 0) {
        --p.skippedDepth;
        return;
    }
    if (p.depth == 0 || p.zones.empty()) return;
    --p.depth;
    // The zone this end pairs with is the most recent still-open one: with a
    // pure RAII scope stack it is the last zone whose depth equals the current
    // (pre-decrement) level.
    for (size_t i = p.zones.size(); i-- > 0;) {
        if (p.zones[i].depth == p.depth) {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool_,
                                p.zones[i].query + 1);
            return;
        }
    }
}

bool GpuProfiler::harvest(const VulkanContext& ctx, uint32_t slot) {
    PendingFrame& p = pending_[slot];
    if (!p.active) return false;
    p.active = false;
    if (p.zones.empty()) return false;

    const uint32_t count = static_cast<uint32_t>(p.zones.size()) * 2;
    std::vector<uint64_t> values(count);
    const VkResult res =
        vkGetQueryPoolResults(ctx.device, pool_, slot * kMaxZonesPerFrame * 2, count,
                              values.size() * sizeof(uint64_t), values.data(), sizeof(uint64_t),
                              VK_QUERY_RESULT_64_BIT);
    if (res != VK_SUCCESS) return false;

    Frame frame;
    frame.frameIndex = p.frameIndex;
    frame.zones.resize(p.zones.size());
    const auto toMs = [&](uint64_t a, uint64_t b) {
        if (b <= a) return 0.f;
        return static_cast<float>(static_cast<double>(b - a) * static_cast<double>(periodNs_) / 1e6);
    };
    uint64_t tFirst = UINT64_MAX, tLast = 0;
    for (size_t i = 0; i < p.zones.size(); ++i) {
        Zone& z = frame.zones[i];
        z.name = p.zones[i].name;
        z.depth = p.zones[i].depth;
        const uint64_t t0 = values[i * 2];
        const uint64_t t1 = values[i * 2 + 1];
        z.ms = toMs(t0, t1);
        tFirst = std::min(tFirst, t0);
        tLast = std::max(tLast, t1);
    }
    frame.totalMs = toMs(tFirst, tLast);

    // Self time: inclusive minus direct children (the run of zones at depth+1
    // following a zone, before the depth drops back to its level).
    for (size_t i = 0; i < frame.zones.size(); ++i) {
        Zone& z = frame.zones[i];
        float childMs = 0.f;
        for (size_t j = i + 1; j < frame.zones.size() && frame.zones[j].depth > z.depth; ++j) {
            if (frame.zones[j].depth == z.depth + 1) childMs += frame.zones[j].ms;
        }
        z.selfMs = z.ms - childMs;
        if (z.selfMs < 0.f) z.selfMs = 0.f;
    }

    if (historyCount_ < kHistoryLen) {
        history_[(historyHead_ + historyCount_) % kHistoryLen] = std::move(frame);
        ++historyCount_;
    } else {
        history_[historyHead_] = std::move(frame);
        historyHead_ = (historyHead_ + 1) % kHistoryLen;
    }
    return true;
}

} // namespace sr
