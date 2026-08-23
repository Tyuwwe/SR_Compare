#pragma once
// ============================================================================
// Runtime LOD chain generation for static meshes (meshoptimizer) with a disk
// cache.  Implemented in LodBuilder.cpp; used by the glTF loader.
// ============================================================================
#include <cstdint>
#include <string>
#include <vector>

namespace sr {

class Scene;
struct Vertex;

// CPU copy of one uploaded static mesh, stashed by the glTF loader for LOD
// generation (vertices/indices are moved in after the GPU upload, so the
// steady-state memory cost is zero).
struct LodSourceMesh {
    uint32_t meshIndex = 0;
    std::vector<Vertex> verts;
    std::vector<uint32_t> indices;
};

// Generates up to kMaxMeshLods-1 simplified index ranges per source mesh and
// appends them via Scene::appendMeshLod (call before buildMergedBuffers).
// Meshes flagged in skipMeshes (indexed by scene mesh index, e.g. members of
// an authored MSFT_lod chain) are left at LOD0.
//
// Results are cached next to the glTF as "<path>.lodcache", keyed by a hash
// of the source geometry (positions + indices) and a format version, so the
// expensive simplification runs once per asset instead of every startup.
// Cache read/write failures silently fall back to regeneration.
void generateMeshLods(Scene& scene, std::vector<LodSourceMesh>& sources,
                      const std::string& gltfPath, const std::vector<char>& skipMeshes);

} // namespace sr
