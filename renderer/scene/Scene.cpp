#include "renderer/scene/Scene.h"

#include "renderer/core/VkUtil.h"

#include <algorithm>
#include <cstring>

namespace sr {

namespace {

bool createStagedBuffer(const VulkanContext& ctx, const void* data, VkDeviceSize size,
                        VkBufferUsageFlags usage, VkBuffer& buffer, VkDeviceMemory& memory,
                        VkCommandPool pool) {
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    if (createBuffer(ctx, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging, stagingMemory) != VK_SUCCESS)
        return false;

    void* mapped = nullptr;
    vkMapMemory(ctx.device, stagingMemory, 0, size, 0, &mapped);
    std::memcpy(mapped, data, static_cast<size_t>(size));
    vkUnmapMemory(ctx.device, stagingMemory);

    if (createBuffer(ctx, size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, memory) != VK_SUCCESS) {
        vkDestroyBuffer(ctx.device, staging, nullptr);
        vkFreeMemory(ctx.device, stagingMemory, nullptr);
        return false;
    }

    VkBufferCopy region = {};
    region.size = size;
    submitOneShot(ctx, [&](VkCommandBuffer cmd) {
        vkCmdCopyBuffer(cmd, staging, buffer, 1, &region);
    }, pool);

    vkDestroyBuffer(ctx.device, staging, nullptr);
    vkFreeMemory(ctx.device, stagingMemory, nullptr);
    return true;
}

} // namespace

bool Scene::uploadMesh(const VulkanContext& ctx, const std::vector<Vertex>& vertices,
                       const std::vector<uint32_t>& indices, Mesh& out, VkCommandPool pool) {
    if (vertices.empty() || indices.empty()) return false;

    const VkDeviceSize vbSize = vertices.size() * sizeof(Vertex);
    const VkDeviceSize ibSize = indices.size() * sizeof(uint32_t);

    if (!createStagedBuffer(ctx, vertices.data(), vbSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            out.vertexBuffer, out.vertexMemory, pool))
        return false;
    if (!createStagedBuffer(ctx, indices.data(), ibSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                            out.indexBuffer, out.indexMemory, pool))
        return false;

    Vec3 lo = vertices[0].position, hi = vertices[0].position;
    for (const auto& v : vertices) {
        lo.x = std::min(lo.x, v.position.x); lo.y = std::min(lo.y, v.position.y);
        lo.z = std::min(lo.z, v.position.z);
        hi.x = std::max(hi.x, v.position.x); hi.y = std::max(hi.y, v.position.y);
        hi.z = std::max(hi.z, v.position.z);
    }
    out.aabbMin = lo;
    out.aabbMax = hi;

    // Accumulate into the scene-wide merged buffers (see buildMergedBuffers).
    // Indices stay mesh-local; the draw call adds vertexOffset at draw time.
    out.firstIndex = static_cast<uint32_t>(mergedIndices_.size());
    out.vertexOffset = static_cast<int32_t>(mergedVerts_.size());
    mergedVerts_.insert(mergedVerts_.end(), vertices.begin(), vertices.end());
    // No reserve(): exact-fit reserve would defeat geometric growth and turn
    // the accumulation into O(n^2) copying (measured +17s on Bistro).
    mergedIndices_.insert(mergedIndices_.end(), indices.begin(), indices.end());

    out.indexCount = static_cast<uint32_t>(indices.size());
    out.indexType = VK_INDEX_TYPE_UINT32;
    return true;
}

bool Scene::buildMergedBuffers(const VulkanContext& ctx, VkCommandPool pool) {
    if (mergedVerts_.empty() || mergedIndices_.empty()) return false;
    const VkDeviceSize vbSize = mergedVerts_.size() * sizeof(Vertex);
    const VkDeviceSize ibSize = mergedIndices_.size() * sizeof(uint32_t);
    const bool ok =
        createStagedBuffer(ctx, mergedVerts_.data(), vbSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                           mergedVertexBuffer, mergedVertexMemory, pool) &&
        createStagedBuffer(ctx, mergedIndices_.data(), ibSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                           mergedIndexBuffer, mergedIndexMemory, pool);
    std::vector<Vertex>().swap(mergedVerts_);
    std::vector<uint32_t>().swap(mergedIndices_);
    return ok;
}

void Scene::finalizeInstances() {
    for (auto& inst : instances) {
        inst.normalModel = Mat4::transpose(Mat4::inverse(inst.model));
        // Transform the 8 local-AABB corners; the result is conservative for
        // any affine model matrix.
        const Mesh& mesh = meshes[inst.meshIndex];
        Vec3 lo, hi;
        for (int c = 0; c < 8; ++c) {
            const Vec3 corner{(c & 1) ? mesh.aabbMax.x : mesh.aabbMin.x,
                              (c & 2) ? mesh.aabbMax.y : mesh.aabbMin.y,
                              (c & 4) ? mesh.aabbMax.z : mesh.aabbMin.z};
            const Vec3 w = transformPoint(inst.model, corner);
            if (c == 0) { lo = w; hi = w; }
            else {
                lo.x = std::min(lo.x, w.x); lo.y = std::min(lo.y, w.y);
                lo.z = std::min(lo.z, w.z);
                hi.x = std::max(hi.x, w.x); hi.y = std::max(hi.y, w.y);
                hi.z = std::max(hi.z, w.z);
            }
        }
        inst.aabbMin = lo;
        inst.aabbMax = hi;
    }
    std::sort(instances.begin(), instances.end(), [](const MeshInstance& a, const MeshInstance& b) {
        if (a.materialIndex != b.materialIndex) return a.materialIndex < b.materialIndex;
        return a.meshIndex < b.meshIndex;
    });
}

bool Scene::uploadTexture(const VulkanContext& ctx, uint32_t width, uint32_t height,
                          const uint8_t* rgba8, Texture& out, bool srgb, VkCommandPool pool) {
    // SRGB so that sampled base colors land in linear space for shading; data
    // textures (normal/MR/AO) stay UNORM.  Full mip chain fixes minification
    // aliasing at grazing angles (floors, roads).
    const VkFormat format = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    uint32_t mipLevels = 1;
    for (uint32_t s = std::max(width, height); s > 1; s >>= 1) ++mipLevels;

    if (createImage(ctx, width, height, format,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                    out.image, out.memory, mipLevels) != VK_SUCCESS)
        return false;

    const VkDeviceSize size = static_cast<VkDeviceSize>(width) * height * 4;
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    if (createBuffer(ctx, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging, stagingMemory) != VK_SUCCESS) {
        vkDestroyImage(ctx.device, out.image, nullptr);
        vkFreeMemory(ctx.device, out.memory, nullptr);
        out.image = VK_NULL_HANDLE;
        out.memory = VK_NULL_HANDLE;
        return false;
    }

    void* mapped = nullptr;
    vkMapMemory(ctx.device, stagingMemory, 0, size, 0, &mapped);
    std::memcpy(mapped, rgba8, static_cast<size_t>(size));
    vkUnmapMemory(ctx.device, stagingMemory);

    submitOneShot(ctx, [&](VkCommandBuffer cmd) {
        // Level 0: UNDEFINED -> TRANSFER_DST, upload, -> TRANSFER_SRC.
        imageBarrier(cmd, out.image, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);
        VkBufferImageCopy region = {};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {width, height, 1};
        vkCmdCopyBufferToImage(cmd, staging, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                               &region);
        imageBarrier(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1);

        // Downsample level by level; each finished level goes SHADER_READ_ONLY.
        for (uint32_t level = 1; level < mipLevels; ++level) {
            imageBarrier(cmd, out.image, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, level, 1);
            VkImageBlit blit = {};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1};
            blit.srcOffsets[0] = {0, 0, 0};
            blit.srcOffsets[1] = {static_cast<int32_t>(std::max(1u, width >> (level - 1))),
                                  static_cast<int32_t>(std::max(1u, height >> (level - 1))), 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1};
            blit.dstOffsets[0] = {0, 0, 0};
            blit.dstOffsets[1] = {static_cast<int32_t>(std::max(1u, width >> level)),
                                  static_cast<int32_t>(std::max(1u, height >> level)), 1};
            vkCmdBlitImage(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, out.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
            imageBarrier(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT,
                         level - 1, 1);
            // The new level becomes the source for the next blit.
            imageBarrier(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, level, 1);
        }
        imageBarrier(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT,
                     mipLevels - 1, 1);
    }, pool);

    vkDestroyBuffer(ctx.device, staging, nullptr);
    vkFreeMemory(ctx.device, stagingMemory, nullptr);

    out.view = createImageView(ctx, out.image, format, VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels);
    out.width = width;
    out.height = height;
    out.mipLevels = mipLevels;
    return out.view != VK_NULL_HANDLE;
}

void Scene::destroy(const VulkanContext& ctx) {
    for (auto& m : meshes) {
        if (m.vertexBuffer) vkDestroyBuffer(ctx.device, m.vertexBuffer, nullptr);
        if (m.vertexMemory) vkFreeMemory(ctx.device, m.vertexMemory, nullptr);
        if (m.indexBuffer) vkDestroyBuffer(ctx.device, m.indexBuffer, nullptr);
        if (m.indexMemory) vkFreeMemory(ctx.device, m.indexMemory, nullptr);
    }
    meshes.clear();

    for (auto& t : textures) {
        if (t.view) vkDestroyImageView(ctx.device, t.view, nullptr);
        if (t.image) vkDestroyImage(ctx.device, t.image, nullptr);
        if (t.memory) vkFreeMemory(ctx.device, t.memory, nullptr);
    }
    textures.clear();

    if (mergedVertexBuffer) vkDestroyBuffer(ctx.device, mergedVertexBuffer, nullptr);
    if (mergedVertexMemory) vkFreeMemory(ctx.device, mergedVertexMemory, nullptr);
    if (mergedIndexBuffer) vkDestroyBuffer(ctx.device, mergedIndexBuffer, nullptr);
    if (mergedIndexMemory) vkFreeMemory(ctx.device, mergedIndexMemory, nullptr);
    mergedVertexBuffer = VK_NULL_HANDLE;
    mergedVertexMemory = VK_NULL_HANDLE;
    mergedIndexBuffer = VK_NULL_HANDLE;
    mergedIndexMemory = VK_NULL_HANDLE;
    mergedVerts_.clear();
    mergedIndices_.clear();

    materials.clear();
    instances.clear();
    lights.clear();
}

} // namespace sr
