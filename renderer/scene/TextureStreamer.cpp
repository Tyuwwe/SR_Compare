// ============================================================================
// Phase 7b: BC7 KTX2 direct upload + background fine-mip streaming.
// See the "Texture compression / mip streaming" comment block in Scene.h for
// the design (determinism boundary, placeholder semantics, budget).
// ============================================================================
#include "renderer/scene/Scene.h"

#include "renderer/core/VkUtil.h"
#include "renderer/core/Vma.h"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

namespace sr {

// Out-of-line special members (TextureStreamer is complete below).
Scene::Scene() = default;
Scene::~Scene() { stopTextureStreaming(); }
Scene::Scene(Scene&&) noexcept = default;
Scene& Scene::operator=(Scene&&) noexcept = default;

namespace {

uint32_t mipExtent(uint32_t dim, uint32_t level) {
    const uint32_t v = dim >> level;
    return v ? v : 1;
}

// Uploads levels [firstLevel, mipLevels) of `img` (already-validated BC7
// KTX2) through one staging buffer + one-shot submit.  Buffer offsets are
// 16-byte aligned (BC7 block size; satisfies the compressed-copy alignment
// rules).  Uploaded levels end in SHADER_READ_ONLY.
bool uploadKtx2Levels(const VulkanContext& ctx, const Ktx2Image& img, VkImage image,
                      uint32_t firstLevel, VkCommandPool pool) {
    VkDeviceSize stagingSize = 0;
    for (uint32_t l = firstLevel; l < img.mipLevels; ++l)
        stagingSize += (img.levels[l].length + 15ull) & ~15ull;

    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation stagingMemory = VK_NULL_HANDLE;
    if (createBuffer(ctx, stagingSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging, stagingMemory) != VK_SUCCESS)
        return false;

    void* mapped = nullptr;
    vmaMapMemory(ctx.allocator, stagingMemory, &mapped);
    auto* bytes = static_cast<uint8_t*>(mapped);
    std::vector<VkBufferImageCopy> regions(img.mipLevels - firstLevel);
    VkDeviceSize off = 0;
    for (uint32_t l = firstLevel; l < img.mipLevels; ++l) {
        std::memcpy(bytes + off, img.data.data() + img.levels[l].offset,
                    static_cast<size_t>(img.levels[l].length));
        VkBufferImageCopy& r = regions[l - firstLevel];
        r.bufferOffset = off;
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, 1};
        r.imageExtent = {mipExtent(img.width, l), mipExtent(img.height, l), 1};
        off += (img.levels[l].length + 15ull) & ~15ull;
    }
    vmaUnmapMemory(ctx.allocator, stagingMemory);

    submitOneShot(
        ctx,
        [&](VkCommandBuffer cmd) {
            imageBarrier(cmd, image, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE,
                         VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_COPY_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, firstLevel,
                         static_cast<uint32_t>(regions.size()));
            vkCmdCopyBufferToImage(cmd, staging, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   static_cast<uint32_t>(regions.size()), regions.data());
            imageBarrier(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, sync::kSampleStages,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                         firstLevel, static_cast<uint32_t>(regions.size()));
        },
        pool);

    vmaDestroyBuffer(ctx.allocator, staging, stagingMemory);
    return true;
}

} // namespace

// --- background streamer -------------------------------------------------------

struct Scene::TextureStreamer {
    const VulkanContext* ctx = nullptr;
    // Set in start(), which runs lazily on the first per-frame tick — after the
    // scene has been moved into its final home (GUI async install), so the
    // pointer stays valid for the thread's lifetime.
    Scene* scene = nullptr;
    std::vector<Scene::TextureStreamJob> pending;
    // One representative world position per texture (start()), nearest-first.
    std::vector<Vec3> texCentroid;
    std::vector<char> texHasCentroid;
    std::thread thread;
    std::mutex mtx;
    std::condition_variable cv;
    bool stop = false;
    uint64_t budget = 0; // bytes the worker may still upload this tick
    Vec3 cameraPos{0.f, 0.f, 0.f};

    void start(Scene& scene);
    void run();
    void stopAndJoin();
};

void Scene::TextureStreamer::start(Scene& s) {
    scene = &s;
    // One representative world position per texture: the average centre of the
    // world AABBs of the instances whose material references it.  Computed once
    // here (instances are final after load); nearest-first ordering uses it.
    const size_t n = s.textures.size();
    std::vector<Vec3> sum(n, Vec3{0.f, 0.f, 0.f});
    std::vector<uint32_t> count(n, 0);
    for (const MeshInstance& inst : s.instances) {
        const Material& m = s.materials[inst.materialIndex];
        const int32_t slots[5] = {m.texIndex, m.normalTexIndex, m.mrTexIndex, m.aoTexIndex,
                                  m.emissiveTexIndex};
        const Vec3 c = (inst.aabbMin + inst.aabbMax) * 0.5f;
        for (int32_t slot : slots) {
            if (slot < 0 || static_cast<size_t>(slot) >= n) continue;
            sum[static_cast<size_t>(slot)] += c;
            ++count[static_cast<size_t>(slot)];
        }
    }
    texCentroid.resize(n);
    texHasCentroid.assign(n, 0);
    for (size_t t = 0; t < n; ++t) {
        if (count[t] > 0) {
            texCentroid[t] = sum[t] * (1.f / static_cast<float>(count[t]));
            texHasCentroid[t] = 1;
        }
    }
    thread = std::thread([this] { run(); });
}

void Scene::TextureStreamer::run() {
    // Worker-private pool; queue submits are serialized inside submitOneShot
    // via ctx.queueMutex (same pattern as the GUI async load worker).
    VkCommandPoolCreateInfo poolCi = {};
    poolCi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolCi.queueFamilyIndex = ctx->graphicsQueueFamily;
    VkCommandPool pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(ctx->device, &poolCi, nullptr, &pool) != VK_SUCCESS || !pool) {
        // No fallback to ctx.oneShotPool: it is owned by the main thread.
        // Textures keep their low-mip placeholders.
        return;
    }

    for (;;) {
        TextureStreamJob job;
        uint32_t level = 0;
        {
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [&] { return stop || (budget > 0 && !pending.empty()); });
            if (stop) break;
            // Nearest first under the latest camera position.
            auto best = pending.begin();
            float bestDist = 0.f;
            for (auto it = pending.begin(); it != pending.end(); ++it) {
                // Textures referenced by no instance sort last (distance +inf).
                const size_t t = it->textureIndex;
                const float d = (t < texHasCentroid.size() && texHasCentroid[t])
                                    ? length(texCentroid[t] - cameraPos)
                                    : 3.4e38f;
                if (it == pending.begin() || d < bestDist) {
                    best = it;
                    bestDist = d;
                }
            }
            level = best->nextLevel;
            job = *best; // copy: string + level table (few entries)
            if (level == 0)
                pending.erase(best);
            else
                best->nextLevel = level - 1;
            budget -= std::min(budget, job.levels[level].length);
        }

        std::vector<uint8_t> bytes;
        Texture* tex = &scene->textures[job.textureIndex];
        if (readKtx2Level(job.path.c_str(), job.levels[level].offset, job.levels[level].length,
                          bytes)) {
            VkBuffer staging = VK_NULL_HANDLE;
            VmaAllocation stagingMemory = VK_NULL_HANDLE;
            if (createBuffer(*ctx, bytes.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             staging, stagingMemory) == VK_SUCCESS) {
                void* mapped = nullptr;
                vmaMapMemory(ctx->allocator, stagingMemory, &mapped);
                std::memcpy(mapped, bytes.data(), bytes.size());
                vmaUnmapMemory(ctx->allocator, stagingMemory);
                submitOneShot(
                    *ctx,
                    [&](VkCommandBuffer cmd) {
                        // Placeholder state is SHADER_READ_ONLY (undefined
                        // contents); net layout across this buffer is
                        // unchanged, so frames recorded before/after this
                        // submit see a consistent image.
                        imageBarrier(cmd, tex->image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, sync::kSampleStages,
                                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                     VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                     VK_IMAGE_ASPECT_COLOR_BIT, level, 1);
                        VkBufferImageCopy r = {};
                        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
                        r.imageExtent = {mipExtent(job.width, level),
                                         mipExtent(job.height, level), 1};
                        vkCmdCopyBufferToImage(cmd, staging, tex->image,
                                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
                        imageBarrier(cmd, tex->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                     sync::kSampleStages, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                     VK_IMAGE_ASPECT_COLOR_BIT, level, 1);
                    },
                    pool);
                vmaDestroyBuffer(ctx->allocator, staging, stagingMemory);
            }
        }
        // A failed read/upload drops the level silently: the placeholder stays.
    }

    if (pool) vkDestroyCommandPool(ctx->device, pool, nullptr);
}

void Scene::TextureStreamer::stopAndJoin() {
    {
        std::lock_guard<std::mutex> lk(mtx);
        stop = true;
    }
    cv.notify_all();
    if (thread.joinable()) thread.join();
}

// --- Scene entry points ----------------------------------------------------------

bool Scene::uploadTextureCompressed(const VulkanContext& ctx, const Ktx2Image& img, bool srgb,
                                    Texture& out, const char* sourcePath, VkCommandPool pool) {
    // Block payload is identical for UNORM and SRGB; only sampling differs.
    const VkFormat format = srgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK;

    // Streaming: large textures start with only the coarse tail resident.
    const bool stream = streamingEnabled && img.mipLevels > kStreamResidentBaseMip + 1;
    const uint32_t residentFirst = stream ? kStreamResidentBaseMip : 0;

    if (createImage(ctx, img.width, img.height, format,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, out.image,
                    out.memory, img.mipLevels) != VK_SUCCESS)
        return false;

    if (stream) {
        // Levels [0, residentFirst): placeholder — layout valid for sampling,
        // contents undefined until the streamer fills them (see Scene.h).
        submitOneShot(
            ctx,
            [&](VkCommandBuffer cmd) {
                imageBarrier(cmd, out.image, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_NONE,
                             VK_ACCESS_2_NONE, sync::kSampleStages,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 0,
                             residentFirst);
            },
            pool);
    }

    if (!uploadKtx2Levels(ctx, img, out.image, residentFirst, pool)) {
        vmaDestroyImage(ctx.allocator, out.image, out.memory);
        out.image = VK_NULL_HANDLE;
        out.memory = VK_NULL_HANDLE;
        return false;
    }

    out.view = createImageView(ctx, out.image, format, VK_IMAGE_ASPECT_COLOR_BIT, 0,
                               img.mipLevels);
    out.width = img.width;
    out.height = img.height;
    out.mipLevels = img.mipLevels;
    if (out.view == VK_NULL_HANDLE) return false;

    if (stream) {
        TextureStreamJob job;
        job.textureIndex = static_cast<uint32_t>(textures.size()); // pushed by caller next
        job.path = sourcePath;
        job.width = img.width;
        job.height = img.height;
        job.nextLevel = residentFirst - 1;
        job.levels.assign(img.levels.begin(), img.levels.begin() + residentFirst);
        for (const auto& l : job.levels) job.bytesRemaining += l.length;
        pendingStreamTextures_.push_back(std::move(job));
    }
    return true;
}

void Scene::updateTextureStreaming(const VulkanContext& ctx, const Vec3& cameraPos) {
    if (pendingStreamTextures_.empty() && !streamer_) return;
    if (!streamer_) {
        auto s = std::make_unique<TextureStreamer>();
        s->ctx = &ctx;
        s->pending = std::move(pendingStreamTextures_);
        pendingStreamTextures_.clear();
        s->cameraPos = cameraPos;
        streamer_ = std::move(s);
        streamer_->start(*this);
    }
    {
        std::lock_guard<std::mutex> lk(streamer_->mtx);
        streamer_->cameraPos = cameraPos;
        streamer_->budget += kStreamBudgetBytesPerFrame;
        // Cap so a hidden window / breakpoint does not accumulate unbounded work.
        streamer_->budget = std::min(streamer_->budget, 4 * kStreamBudgetBytesPerFrame);
    }
    streamer_->cv.notify_one();
}

void Scene::stopTextureStreaming() {
    if (streamer_) {
        streamer_->stopAndJoin();
        streamer_.reset();
    }
    pendingStreamTextures_.clear();
}

} // namespace sr
