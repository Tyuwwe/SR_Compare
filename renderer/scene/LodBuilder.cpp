// ============================================================================
// LOD chains: generation (meshopt_simplify), disk cache, per-frame selection.
//
// Generation is index-only: meshopt_simplify re-indexes the original vertex
// set, so every level shares the mesh's vertex buffer and a level is just an
// extra index range in the merged scene buffers.  Selection is a pure
// function of camera + world AABBs (with hysteresis), evaluated once per
// frame at a fixed 1080p reference height so the GT and upscaled paths of a
// compare/bench frame always draw the same levels.
// ============================================================================
#if defined(_MSC_VER)
// std::fopen is portable; the _s variants are MSVC-only (same pattern the
// rest of the project relies on for file I/O).
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "renderer/scene/LodBuilder.h"

#include "renderer/scene/Scene.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <meshoptimizer.h>

namespace sr {

namespace {

// Selection thresholds: projected bounding-sphere radius in pixels at the
// 1080p reference height.  Below kLodSwitchRadiusPx[k] the instance moves
// from level k to k+1; below kLodCullRadiusPx it is not drawn at all (the
// "max distance" cull, expressed as screen size so it adapts to mesh scale).
constexpr float kLodReferenceHeightPx = 1080.f;
constexpr float kLodSwitchRadiusPx[kMaxMeshLods - 1] = {200.f, 80.f, 32.f};
constexpr float kLodCullRadiusPx = 10.f;
// Hysteresis band around every threshold: a boundary-hugging camera keeps its
// current level instead of oscillating (visible pop-in) frame to frame.
constexpr float kLodHysteresis = 0.15f;

// Generation parameters: target index-count ratio and error cap per extra
// level (relative to the full LOD0, not chained, so error does not
// accumulate).  Border locking keeps open edges (UV seams, mesh boundaries)
// from collapsing into cracks at reduced levels.
constexpr float kLodTargetRatio[kMaxMeshLods - 1] = {0.5f, 0.25f, 0.125f};
constexpr float kLodTargetError[kMaxMeshLods - 1] = {0.5f, 0.5f, 0.5f};
// Below this many source triangles the draw is cheap already; decimation
// would only add cache/memory overhead.
constexpr size_t kLodMinSourceIndices = 768;

constexpr char kCacheMagic[8] = {'S', 'R', 'L', 'O', 'D', 'C', '0', '1'};

// FNV-1a 64 over all source geometry: any asset change invalidates the cache.
uint64_t hashBytes(uint64_t h, const void* data, size_t size) {
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

uint64_t hashSources(const std::vector<LodSourceMesh>& sources) {
    uint64_t h = 1469598103934665603ull;
    for (const LodSourceMesh& src : sources) {
        const uint64_t counts = (static_cast<uint64_t>(src.verts.size()) << 32) |
                                static_cast<uint64_t>(src.indices.size());
        h = hashBytes(h, &counts, sizeof(counts));
        if (!src.verts.empty())
            h = hashBytes(h, &src.verts[0].position, src.verts.size() * sizeof(Vec3));
        if (!src.indices.empty())
            h = hashBytes(h, src.indices.data(), src.indices.size() * sizeof(uint32_t));
    }
    return h;
}

bool readCache(Scene& scene, const std::vector<LodSourceMesh>& sources, uint64_t hash,
               const std::string& cachePath) {
    FILE* f = std::fopen(cachePath.c_str(), "rb");
    if (!f) return false;
    bool ok = false;
    char magic[8] = {};
    uint64_t fileHash = 0;
    uint32_t count = 0;
    if (std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, kCacheMagic, 8) == 0 &&
        std::fread(&fileHash, sizeof(fileHash), 1, f) == 1 && fileHash == hash &&
        std::fread(&count, sizeof(count), 1, f) == 1 && count == sources.size()) {
        ok = true;
        for (uint32_t s = 0; s < count && ok; ++s) {
            uint32_t meshIndex = 0, extraCount = 0;
            ok = std::fread(&meshIndex, sizeof(meshIndex), 1, f) == 1 &&
                 std::fread(&extraCount, sizeof(extraCount), 1, f) == 1 &&
                 meshIndex == sources[s].meshIndex && extraCount < kMaxMeshLods;
            for (uint32_t e = 0; e < extraCount && ok; ++e) {
                uint32_t indexCount = 0;
                ok = std::fread(&indexCount, sizeof(indexCount), 1, f) == 1 && indexCount >= 3 &&
                     indexCount <= sources[s].indices.size();
                if (!ok) break;
                std::vector<uint32_t> indices(indexCount);
                ok = std::fread(indices.data(), sizeof(uint32_t), indexCount, f) == indexCount;
                if (ok) scene.appendMeshLod(meshIndex, indices);
            }
        }
    }
    std::fclose(f);
    return ok;
}

void writeCache(const std::vector<LodSourceMesh>& sources,
                const std::vector<std::vector<std::vector<uint32_t>>>& extraLods, uint64_t hash,
                const std::string& cachePath) {
    FILE* f = std::fopen(cachePath.c_str(), "wb");
    if (!f) return; // read-only asset dir: regenerate next time, no failure
    std::fwrite(kCacheMagic, 1, 8, f);
    std::fwrite(&hash, sizeof(hash), 1, f);
    const uint32_t count = static_cast<uint32_t>(sources.size());
    std::fwrite(&count, sizeof(count), 1, f);
    for (size_t s = 0; s < sources.size(); ++s) {
        std::fwrite(&sources[s].meshIndex, sizeof(uint32_t), 1, f);
        const uint32_t extraCount = static_cast<uint32_t>(extraLods[s].size());
        std::fwrite(&extraCount, sizeof(extraCount), 1, f);
        for (const std::vector<uint32_t>& lod : extraLods[s]) {
            const uint32_t indexCount = static_cast<uint32_t>(lod.size());
            std::fwrite(&indexCount, sizeof(indexCount), 1, f);
            std::fwrite(lod.data(), sizeof(uint32_t), lod.size(), f);
        }
    }
    std::fclose(f);
}

} // namespace

void generateMeshLods(Scene& scene, std::vector<LodSourceMesh>& sources,
                      const std::string& gltfPath, const std::vector<char>& skipMeshes) {
    if (sources.empty()) return;
    const uint64_t hash = hashSources(sources);
    const std::string cachePath = gltfPath + ".lodcache";
    if (readCache(scene, sources, hash, cachePath)) return;

    std::vector<std::vector<std::vector<uint32_t>>> extraLods(sources.size());
    for (size_t s = 0; s < sources.size(); ++s) {
        const LodSourceMesh& src = sources[s];
        const uint32_t meshIndex = src.meshIndex;
        if (meshIndex < skipMeshes.size() && skipMeshes[meshIndex]) continue;
        if (src.indices.size() < kLodMinSourceIndices || src.verts.empty()) continue;

        // Each level decimates the original LOD0 indices; the destination
        // overwrites nothing (meshopt writes a fresh index list).
        size_t prevCount = src.indices.size();
        for (uint32_t level = 0; level < kMaxMeshLods - 1; ++level) {
            const size_t target = std::max<size_t>(
                3, static_cast<size_t>(static_cast<double>(src.indices.size()) *
                                       static_cast<double>(kLodTargetRatio[level])));
            std::vector<uint32_t> out(src.indices.size());
            float resultError = 0.f;
            const size_t result = meshopt_simplify(
                out.data(), src.indices.data(), src.indices.size(), &src.verts[0].position.x,
                src.verts.size(), sizeof(Vertex), target, kLodTargetError[level],
                meshopt_SimplifyLockBorder, &resultError);
            out.resize(result);
            // Keep the level only if it actually shrinks the draw; otherwise
            // stop the chain (a level that saves <20% is not worth the memory).
            if (result < 3 || result > (prevCount * 4) / 5) break;
            scene.appendMeshLod(meshIndex, out);
            extraLods[s].push_back(std::move(out));
            prevCount = result;
        }
    }
    writeCache(sources, extraLods, hash, cachePath);
}

void Scene::appendMeshLod(uint32_t meshIndex, const std::vector<uint32_t>& indices) {
    if (meshIndex >= meshes.size() || indices.empty()) return;
    Mesh& m = meshes[meshIndex];
    if (m.lodCount >= kMaxMeshLods) return;
    LodDraw d;
    d.firstIndex = static_cast<uint32_t>(mergedIndices_.size());
    d.indexCount = static_cast<uint32_t>(indices.size());
    d.vertexOffset = m.vertexOffset; // shared vertices: only the index list differs
    mergedIndices_.insert(mergedIndices_.end(), indices.begin(), indices.end());
    m.lods[m.lodCount++] = d;
}

void Scene::buildLodDraws() {
    for (auto& inst : instances) {
        inst.lodLevel = 0;
        inst.lodCulled = false;
        if (inst.skinIndex >= 0) continue; // skinned path: single level, own buffers
        if (inst.authoredLodCount > 0) {
            // MSFT_lod chain: each level is a different mesh's LOD0 range.
            uint32_t n = 0;
            for (uint32_t k = 0; k < inst.authoredLodCount && n < kMaxMeshLods; ++k) {
                const uint32_t mi = inst.authoredLodMeshes[k];
                if (mi >= meshes.size()) continue;
                const Mesh& m = meshes[mi];
                if (m.indexCount == 0) continue;
                inst.lodDraws[n++] = {m.firstIndex, m.indexCount, m.vertexOffset};
            }
            if (n > 0) {
                inst.lodDrawCount = n;
                continue;
            }
        }
        const Mesh& m = meshes[inst.meshIndex];
        inst.lodDrawCount = m.lodCount;
        for (uint32_t k = 0; k < m.lodCount; ++k) inst.lodDraws[k] = m.lods[k];
    }
}

void Scene::updateLodSelection(const Vec3& cameraPos, float fovY, bool enabled) {
    // pixels-per-unit at distance 1 for the fixed reference height.
    const float projFactor = kLodReferenceHeightPx / (2.f * std::tan(fovY * 0.5f));
    for (auto& inst : instances) {
        if (inst.skinIndex >= 0) continue; // skinned meshes stay at LOD0 (Scene.h)
        if (!enabled) {
            inst.lodLevel = 0;
            inst.lodCulled = false;
            continue;
        }
        const Vec3 center = (inst.aabbMin + inst.aabbMax) * 0.5f;
        const float radius = length(inst.aabbMax - center);
        if (radius < 1e-6f) continue; // degenerate bounds: leave previous state
        const float dist = std::max(length(center - cameraPos), 1e-3f);
        const float screenR = radius * projFactor / dist;

        const uint32_t maxLevel = inst.lodDrawCount - 1;
        uint32_t lvl = std::min(inst.lodLevel, maxLevel);
        while (lvl < maxLevel && screenR < kLodSwitchRadiusPx[lvl] * (1.f - kLodHysteresis)) ++lvl;
        while (lvl > 0 && screenR >= kLodSwitchRadiusPx[lvl - 1] * (1.f + kLodHysteresis)) --lvl;
        inst.lodLevel = lvl;

        if (!inst.lodCulled && screenR < kLodCullRadiusPx * (1.f - kLodHysteresis))
            inst.lodCulled = true;
        else if (inst.lodCulled && screenR > kLodCullRadiusPx * (1.f + kLodHysteresis))
            inst.lodCulled = false;
    }
}

} // namespace sr
