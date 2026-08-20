#include "renderer/core/TimestampQuery.h"

namespace sr {

bool TimestampQuery::create(const VulkanContext& ctx, uint32_t framesInFlight) {
    framesInFlight_ = framesInFlight;
    periodNs_ = ctx.properties.limits.timestampPeriod;

    VkQueryPoolCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = framesInFlight * kQueriesPerSlot;
    if (vkCreateQueryPool(ctx.device, &info, nullptr, &pool_) != VK_SUCCESS) return false;
    return true;
}

void TimestampQuery::destroy(const VulkanContext& ctx) {
    if (pool_) vkDestroyQueryPool(ctx.device, pool_, nullptr);
    pool_ = VK_NULL_HANDLE;
}

void TimestampQuery::resetForFrame(VkCommandBuffer cmd, uint32_t slot) {
    vkCmdResetQueryPool(cmd, pool_, slot * kQueriesPerSlot, kQueriesPerSlot);
}

void TimestampQuery::frameBegin(VkCommandBuffer cmd, uint32_t slot) {
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool_, slot * kQueriesPerSlot + 0);
}
void TimestampQuery::sceneBegin(VkCommandBuffer cmd, uint32_t slot) {
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool_, slot * kQueriesPerSlot + 1);
}
void TimestampQuery::sceneEnd(VkCommandBuffer cmd, uint32_t slot) {
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool_, slot * kQueriesPerSlot + 2);
}
void TimestampQuery::upscaleBegin(VkCommandBuffer cmd, uint32_t slot) {
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool_, slot * kQueriesPerSlot + 3);
}
void TimestampQuery::upscaleEnd(VkCommandBuffer cmd, uint32_t slot) {
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool_, slot * kQueriesPerSlot + 4);
}
void TimestampQuery::frameEnd(VkCommandBuffer cmd, uint32_t slot) {
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool_, slot * kQueriesPerSlot + 5);
}

TimestampQuery::Timings TimestampQuery::read(const VulkanContext& ctx, uint32_t slot) const {
    uint64_t values[kQueriesPerSlot] = {};
    const VkResult res = vkGetQueryPoolResults(
        ctx.device, pool_, slot * kQueriesPerSlot, kQueriesPerSlot, sizeof(values), values,
        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
    Timings t;
    if (res != VK_SUCCESS) return t;

    const auto toMs = [&](uint64_t a, uint64_t b) {
        if (b <= a) return 0.0;
        return static_cast<double>(b - a) * static_cast<double>(periodNs_) / 1e6;
    };
    t.frameMs = toMs(values[0], values[5]);
    t.sceneMs = toMs(values[1], values[2]);
    t.upscaleMs = toMs(values[3], values[4]);
    return t;
}

} // namespace sr
