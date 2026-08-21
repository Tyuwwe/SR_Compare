// ============================================================================
// Scene animation: glTF node-tree evaluation (frame-index driven, never wall
// clock — bench determinism), joint palette updates and the procedural
// dynamic-box drivers.  All of this is inert for scenes without animations,
// skins or drivers (Sponza / Bistro keep the fully static path).
// ============================================================================
#include "renderer/scene/Scene.h"

#include "renderer/core/VkUtil.h"
#include "renderer/core/Vma.h"

#include <algorithm>
#include <cfloat>
#include <cstring>

namespace sr {

namespace {

constexpr float kTwoPi = 6.28318530717958647692f;

Mat4 trsToMat4(const Vec3& t, const Vec4& q, const Vec3& s) {
    // Quaternion (xyzw) to rotation matrix, columns scaled by s, translation
    // in column 3.  Column-major storage (m[col * 4 + row]).
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    Mat4 r;
    r.m[0] = (1.f - 2.f * (y * y + z * z)) * s.x;
    r.m[1] = (2.f * (x * y + z * w)) * s.x;
    r.m[2] = (2.f * (x * z - y * w)) * s.x;
    r.m[3] = 0.f;
    r.m[4] = (2.f * (x * y - z * w)) * s.y;
    r.m[5] = (1.f - 2.f * (x * x + z * z)) * s.y;
    r.m[6] = (2.f * (y * z + x * w)) * s.y;
    r.m[7] = 0.f;
    r.m[8] = (2.f * (x * z + y * w)) * s.z;
    r.m[9] = (2.f * (y * z - x * w)) * s.z;
    r.m[10] = (1.f - 2.f * (x * x + y * y)) * s.z;
    r.m[11] = 0.f;
    r.m[12] = t.x;
    r.m[13] = t.y;
    r.m[14] = t.z;
    r.m[15] = 1.f;
    return r;
}

Vec4 lerpVec4(const Vec4& a, const Vec4& b, float f) {
    return {a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f, a.z + (b.z - a.z) * f,
            a.w + (b.w - a.w) * f};
}

Vec4 normalizeQuat(const Vec4& q) {
    const float l = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (l < 1e-8f) return {0.f, 0.f, 0.f, 1.f};
    const float inv = 1.f / l;
    return {q.x * inv, q.y * inv, q.z * inv, q.w * inv};
}

// Samples one channel at time t (already wrapped into the clip).  Rotation
// channels interpolate as normalized lerp with hemisphere correction — good
// enough for keyframe-dense clips and cheaper than slerp.
Vec4 sampleSampler(const AnimSampler& s, float t, bool rotation) {
    if (s.times.empty()) return {0.f, 0.f, 0.f, 1.f};
    if (s.times.size() == 1 || t <= s.times.front()) return s.values.front();
    if (t >= s.times.back()) return s.values.back();
    const auto it = std::upper_bound(s.times.begin(), s.times.end(), t);
    const size_t i = static_cast<size_t>(it - s.times.begin()) - 1;
    if (s.step) return s.values[i];
    const float t0 = s.times[i], t1 = s.times[i + 1];
    const float f = (t1 > t0) ? (t - t0) / (t1 - t0) : 0.f;
    Vec4 b = s.values[i + 1];
    if (rotation) {
        const float d = s.values[i].x * b.x + s.values[i].y * b.y + s.values[i].z * b.z +
                        s.values[i].w * b.w;
        if (d < 0.f) b = {-b.x, -b.y, -b.z, -b.w}; // take the short arc
        return normalizeQuat(lerpVec4(s.values[i], b, f));
    }
    return lerpVec4(s.values[i], b, f);
}

// Evaluates every node's local TRS at clip time `time` (seconds, wrapped).
void evaluatePose(const std::vector<SceneNode>& nodes, const std::vector<AnimSampler>& samplers,
                  const std::vector<AnimChannel>& channels, float animDuration, float time,
                  std::vector<Vec3>& outT, std::vector<Vec4>& outR, std::vector<Vec3>& outS) {
    const size_t n = nodes.size();
    outT.resize(n);
    outR.resize(n);
    outS.resize(n);
    for (size_t i = 0; i < n; ++i) {
        outT[i] = nodes[i].translation;
        outR[i] = nodes[i].rotation;
        outS[i] = nodes[i].scale;
    }
    float t = time;
    if (animDuration > 0.f) {
        t = std::fmod(time, animDuration);
        if (t < 0.f) t += animDuration;
    }
    for (const AnimChannel& ch : channels) {
        const Vec4 v = sampleSampler(samplers[ch.sampler], t, ch.path == AnimPath::Rotation);
        switch (ch.path) {
        case AnimPath::Translation: outT[ch.node] = {v.x, v.y, v.z}; break;
        case AnimPath::Rotation: outR[ch.node] = v; break;
        case AnimPath::Scale: outS[ch.node] = {v.x, v.y, v.z}; break;
        }
    }
}

// Local TRS -> world matrices, parents first (nodeTopoOrder guarantees it).
void computeGlobals(const std::vector<SceneNode>& nodes, const std::vector<uint32_t>& topoOrder,
                    const std::vector<Vec3>& t, const std::vector<Vec4>& r,
                    const std::vector<Vec3>& s, std::vector<Mat4>& out) {
    out.resize(nodes.size());
    for (const uint32_t i : topoOrder) {
        const Mat4 local = trsToMat4(t[i], r[i], s[i]);
        const int32_t parent = nodes[i].parent;
        out[i] = parent >= 0 ? Mat4::multiply(out[static_cast<size_t>(parent)], local) : local;
    }
}

void unionAabbCorner(Vec3& lo, Vec3& hi, const Vec3& p) {
    lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
    hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
}

void unionTransformedAabb(Vec3& lo, Vec3& hi, const Mat4& m, const Vec3& aabbMin,
                          const Vec3& aabbMax) {
    for (int c = 0; c < 8; ++c) {
        const Vec3 corner{(c & 1) ? aabbMax.x : aabbMin.x, (c & 2) ? aabbMax.y : aabbMin.y,
                          (c & 4) ? aabbMax.z : aabbMin.z};
        unionAabbCorner(lo, hi, transformPoint(m, corner));
    }
}

Mat4 driverModel(const DynamicBoxDriver& d, float t) {
    const float yaw = d.baseYaw + d.yawRate * t;
    Vec3 pos = d.basePos;
    if (d.slideAmp.x != 0.f || d.slideAmp.y != 0.f || d.slideAmp.z != 0.f) {
        const float s = std::sin(kTwoPi * t / d.slidePeriod + d.slidePhase);
        pos += d.slideAmp * s;
    }
    return Mat4::multiply(Mat4::translation(pos),
                          Mat4::multiply(Mat4::rotationY(yaw), Mat4::scale(d.scale)));
}

} // namespace

void Scene::advanceToFrame(uint32_t frameIndex) {
    if (!hasDynamicContent()) return;
    const float t = static_cast<float>(frameIndex) * kAnimDt;
    // Frame 0 has no past: prev == cur yields zero motion vectors.
    const float tPrev = frameIndex > 0 ? static_cast<float>(frameIndex - 1) * kAnimDt : t;

    if (!animChannels.empty()) {
        std::vector<Vec3> curT, prevT, curS, prevS;
        std::vector<Vec4> curR, prevR;
        evaluatePose(nodes, animSamplers, animChannels, animDuration, t, curT, curR, curS);
        evaluatePose(nodes, animSamplers, animChannels, animDuration, tPrev, prevT, prevR, prevS);
        std::vector<Mat4> curG, prevG;
        computeGlobals(nodes, nodeTopoOrder, curT, curR, curS, curG);
        computeGlobals(nodes, nodeTopoOrder, prevT, prevR, prevS, prevG);

        // Rigid instances driven by animated nodes: real per-object prevModel.
        for (auto& inst : instances) {
            if (inst.nodeIndex < 0) continue;
            const size_t ni = static_cast<size_t>(inst.nodeIndex);
            inst.model = curG[ni];
            inst.prevModel = prevG[ni];
            inst.normalModel = Mat4::transpose(Mat4::inverse(inst.model));
        }

        // Joint palettes for the current in-flight slot (double-buffered, see
        // kSkinPaletteSlots).  palette[j] = jointGlobal * inverseBind, so the
        // skinned vertex shader needs no per-node model matrix at all (glTF
        // spec: a skinned mesh ignores its node's own transform).
        void* mapped = skinPaletteMapped[frameIndex % kSkinPaletteSlots];
        if (mapped) {
            Mat4* pal = static_cast<Mat4*>(mapped);
            for (const Skin& skin : skins) {
                for (size_t j = 0; j < skin.joints.size(); ++j) {
                    const size_t node = skin.joints[j];
                    pal[skin.paletteCur + j] = Mat4::multiply(curG[node], skin.inverseBind[j]);
                    pal[skin.palettePrev + j] = Mat4::multiply(prevG[node], skin.inverseBind[j]);
                }
            }
        }
    }

    for (const DynamicBoxDriver& d : dynamicDrivers) {
        MeshInstance& inst = instances[d.instanceIndex];
        inst.model = driverModel(d, t);
        inst.prevModel = driverModel(d, tPrev);
        inst.normalModel = Mat4::transpose(Mat4::inverse(inst.model));
    }
}

bool Scene::createSkinPalettes(const VulkanContext& ctx) {
    if (skins.empty()) return true;
    uint32_t total = 0;
    for (auto& skin : skins) {
        skin.paletteCur = total;
        total += static_cast<uint32_t>(skin.joints.size());
    }
    for (auto& skin : skins) skin.palettePrev = total + skin.paletteCur;
    skinPaletteJointCount = total;

    const VkDeviceSize size = static_cast<VkDeviceSize>(total) * 2 * sizeof(Mat4);
    for (uint32_t slot = 0; slot < kSkinPaletteSlots; ++slot) {
        if (createBuffer(ctx, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         skinPaletteBuffer[slot], skinPaletteMemory[slot]) != VK_SUCCESS)
            return false;
        if (vmaMapMemory(ctx.allocator, skinPaletteMemory[slot], &skinPaletteMapped[slot]) !=
            VK_SUCCESS)
            return false;
    }

    // Bind pose fill (identical for both slots): without animations the
    // palettes never change, and frame 0 reads prev == cur anyway.
    std::vector<Vec3> t;
    std::vector<Vec4> r;
    std::vector<Vec3> s;
    evaluatePose(nodes, animSamplers, animChannels, 0.f, 0.f, t, r, s);
    std::vector<Mat4> globals;
    computeGlobals(nodes, nodeTopoOrder, t, r, s, globals);
    for (uint32_t slot = 0; slot < kSkinPaletteSlots; ++slot) {
        Mat4* pal = static_cast<Mat4*>(skinPaletteMapped[slot]);
        for (const Skin& skin : skins) {
            for (size_t j = 0; j < skin.joints.size(); ++j) {
                const Mat4 m = Mat4::multiply(globals[skin.joints[j]], skin.inverseBind[j]);
                pal[skin.paletteCur + j] = m;
                pal[skin.palettePrev + j] = m;
            }
        }
    }
    return true;
}

void Scene::computeAnimatedBounds() {
    // Static instances keep the tight bounds finalizeInstances() gave them.
    // Dynamic ones get the union over the whole animation so the (static,
    // load-time) culling bounds stay valid for every frame — recomputing
    // AABBs per frame would be wasted work for a culler this coarse.
    std::vector<Vec3> lo(instances.size(), Vec3{FLT_MAX, FLT_MAX, FLT_MAX});
    std::vector<Vec3> hi(instances.size(), Vec3{-FLT_MAX, -FLT_MAX, -FLT_MAX});
    std::vector<char> touched(instances.size(), 0);
    for (size_t i = 0; i < instances.size(); ++i) {
        if (instances[i].nodeIndex >= 0 || instances[i].skinIndex >= 0) touched[i] = 1;
    }

    constexpr uint32_t kSteps = 16;
    const uint32_t steps = animDuration > 0.f ? kSteps : 1;
    for (uint32_t step = 0; step < steps; ++step) {
        const float t = animDuration > 0.f
                            ? animDuration * static_cast<float>(step) / static_cast<float>(kSteps)
                            : 0.f;
        std::vector<Vec3> pt, ps;
        std::vector<Vec4> pr;
        evaluatePose(nodes, animSamplers, animChannels, animDuration, t, pt, pr, ps);
        std::vector<Mat4> globals;
        computeGlobals(nodes, nodeTopoOrder, pt, pr, ps, globals);

        for (size_t i = 0; i < instances.size(); ++i) {
            if (!touched[i]) continue;
            const MeshInstance& inst = instances[i];
            if (inst.skinIndex >= 0) {
                // Skinned: corners through every joint's palette matrix (a
                // superset of the true deformation hull).
                const Skin& skin = skins[static_cast<size_t>(inst.skinIndex)];
                const Mesh& mesh = skinnedMeshes[inst.meshIndex];
                for (size_t j = 0; j < skin.joints.size(); ++j) {
                    const Mat4 m = Mat4::multiply(globals[skin.joints[j]], skin.inverseBind[j]);
                    unionTransformedAabb(lo[i], hi[i], m, mesh.aabbMin, mesh.aabbMax);
                }
            } else {
                const Mesh& mesh = meshes[inst.meshIndex];
                unionTransformedAabb(lo[i], hi[i], globals[static_cast<size_t>(inst.nodeIndex)],
                                     mesh.aabbMin, mesh.aabbMax);
            }
        }
    }

    for (size_t i = 0; i < instances.size(); ++i) {
        if (!touched[i]) continue;
        instances[i].aabbMin = lo[i];
        instances[i].aabbMax = hi[i];
    }
}

} // namespace sr
