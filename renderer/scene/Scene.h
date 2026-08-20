#pragma once
// ============================================================================
// Scene data + GPU resources.  Meshes/textures are uploaded through the Vulkan
// context; the actual construction lives in ProceduralScene / GltfLoader.
// ============================================================================
#include "renderer/core/Vk.h"
#include "renderer/core/VulkanContext.h"
#include "renderer/math/Math.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace sr {

struct Vertex {
    Vec3 position;
    Vec3 normal;
    Vec2 uv;
    Vec4 tangent{1.f, 0.f, 0.f, 1.f}; // xyz = tangent, w = bitangent sign
};

struct Mesh {
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexMemory = VK_NULL_HANDLE;
    uint32_t indexCount = 0;
    VkIndexType indexType = VK_INDEX_TYPE_UINT32;
    Vec3 aabbMin{0.f, 0.f, 0.f}; // local-space bounds (for frustum culling)
    Vec3 aabbMax{0.f, 0.f, 0.f};
    // Location inside the merged scene-wide buffers (viewer draw path).
    uint32_t firstIndex = 0;
    int32_t vertexOffset = 0;
};

struct Texture {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mipLevels = 1;
};

struct Material {
    Vec4 baseColor{1.f, 1.f, 1.f, 1.f};
    float metallic = 0.f;
    float roughness = 0.5f;
    int32_t texIndex = -1;         // -1 = untextured (use baseColor)
    int32_t normalTexIndex = -1;   // tangent-space normal map (linear)
    int32_t mrTexIndex = -1;       // glTF metallicRoughness (G=rough, B=metal; linear)
    int32_t aoTexIndex = -1;       // occlusion (R channel; linear)
    int32_t emissiveTexIndex = -1; // emissive (sRGB)
    Vec3 emissiveFactor{0.f, 0.f, 0.f};
    float occlusionStrength = 1.f;
    float alphaCutoff = 0.f;       // > 0: alphaMode MASK, discard below cutoff
    bool blend = false;            // alphaMode BLEND: drawn in the transparency pass
};

struct MeshInstance {
    uint32_t meshIndex = 0;
    uint32_t materialIndex = 0;
    Mat4 model;
    Mat4 prevModel; // reserved per-object previous transform (static == model for now)
    Mat4 normalModel;    // cached transpose(inverse(model)); only the upper 3x3 matters
    Vec3 aabbMin{0.f, 0.f, 0.f}; // world-space bounds (for frustum culling)
    Vec3 aabbMax{0.f, 0.f, 0.f};
};

struct Light {
    Vec3 position;
    Vec3 color{1.f, 1.f, 1.f};
    float intensity = 1.f;
};

class Scene {
public:
    std::vector<Mesh> meshes;
    std::vector<Texture> textures;
    std::vector<Material> materials;
    std::vector<MeshInstance> instances;
    std::vector<Light> lights;

    // Scene-wide merged vertex/index buffers.  The viewer binds these once per
    // pass and draws with per-mesh firstIndex/vertexOffset, which avoids the
    // (very expensive) per-draw vkCmdBindVertexBuffers/IndexBuffer calls.
    // Compare mode keeps using the per-mesh buffers above.
    VkBuffer mergedVertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory mergedVertexMemory = VK_NULL_HANDLE;
    VkBuffer mergedIndexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory mergedIndexMemory = VK_NULL_HANDLE;

    // Optional load progress (glTF).  total == 0 means the stage is
    // indeterminate (parse / finalize).  The callback fires from whichever
    // thread runs the load (the GUI worker thread during async rebuilds), so
    // it must only touch thread-safe state (atomics / locked data).
    enum class LoadStage { Parse, Textures, Meshes, Finalize };
    using LoadProgressFn = std::function<void(LoadStage, size_t done, size_t total)>;

    // pool != VK_NULL_HANDLE: upload one-shot command buffers come from this
    // pool instead of ctx.oneShotPool (async loader: worker-private pool, so
    // command-pool access stays single-threaded per Vulkan rules).
    bool loadProcedural(const VulkanContext& ctx, VkCommandPool pool = VK_NULL_HANDLE);
    bool loadGltf(const VulkanContext& ctx, const char* path,
                  VkCommandPool pool = VK_NULL_HANDLE, const LoadProgressFn& progress = {});
    void destroy(const VulkanContext& ctx);

    // For a fully static scene, previous per-object transform == current.
    void updatePrevTransforms() {
        for (auto& inst : instances) inst.prevModel = inst.model;
    }

    // Fills cached normal matrices + world AABBs and sorts instances by
    // (material, mesh) so the draw loop can skip redundant state changes.
    // Call once after all meshes/instances are built (static scenes only).
    void finalizeInstances();

    // --- upload helpers (used by the loaders above) ---
    bool uploadMesh(const VulkanContext& ctx, const std::vector<Vertex>& vertices,
                    const std::vector<uint32_t>& indices, Mesh& out,
                    VkCommandPool pool = VK_NULL_HANDLE);
    // Uploads RGBA8 texels and generates the full mip chain (per-level blits).
    // srgb=true (default): base color / emissive; false: normal/MR/AO data.
    bool uploadTexture(const VulkanContext& ctx, uint32_t width, uint32_t height,
                       const uint8_t* rgba8, Texture& out, bool srgb = true,
                       VkCommandPool pool = VK_NULL_HANDLE);

    // Uploads the accumulated merged vertex/index data.  Call once after all
    // meshes are uploaded; frees the CPU-side accumulation.
    bool buildMergedBuffers(const VulkanContext& ctx, VkCommandPool pool = VK_NULL_HANDLE);

private:
    // CPU-side accumulation for the merged buffers (cleared by buildMergedBuffers).
    std::vector<Vertex> mergedVerts_;
    std::vector<uint32_t> mergedIndices_;
};

} // namespace sr
