#include "renderer/scene/Scene.h"

#include "renderer/scene/SceneRegistry.h"

#include <cmath>
#include <cstdint>

namespace sr {

namespace {

constexpr float kPi = 3.14159265358979323846f;

Mat4 composeTransform(const Vec3& translate, float rotY, const Vec3& scale) {
    return Mat4::multiply(Mat4::translation(translate),
                          Mat4::multiply(Mat4::rotationY(rotY), Mat4::scale(scale)));
}

// Unit cube centered at origin, half-extent 0.5, explicit per-face normals.
void makeCube(std::vector<Vertex>& verts, std::vector<uint32_t>& idx) {
    struct Face {
        Vec3 n;
        Vec3 p0, p1, p2, p3;
    };
    const Face faces[6] = {
        {{1.f, 0.f, 0.f}, {0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, -0.5f}},
        {{-1.f, 0.f, 0.f}, {-0.5f, -0.5f, 0.5f}, {-0.5f, -0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f}, {-0.5f, 0.5f, 0.5f}},
        {{0.f, 1.f, 0.f}, {-0.5f, 0.5f, -0.5f}, {-0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, -0.5f}},
        {{0.f, -1.f, 0.f}, {-0.5f, -0.5f, 0.5f}, {-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, 0.5f}},
        {{0.f, 0.f, 1.f}, {-0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}},
        {{0.f, 0.f, -1.f}, {0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}},
    };
    const Vec2 uvs[4] = {{0.f, 0.f}, {1.f, 0.f}, {1.f, 1.f}, {0.f, 1.f}};
    for (const auto& f : faces) {
        const Vec3* p = &f.p0;
        const Vec3 tan = normalize(p[1] - p[0]); // +u direction of the face UVs
        const uint32_t base = static_cast<uint32_t>(verts.size());
        for (int i = 0; i < 4; ++i) {
            Vertex v;
            v.position = p[i];
            v.normal = f.n;
            v.uv = uvs[i];
            v.tangent = {tan.x, tan.y, tan.z, 1.f};
            verts.push_back(v);
        }
        const uint32_t order[6] = {0, 1, 2, 0, 2, 3};
        for (int i = 0; i < 6; ++i) idx.push_back(base + order[i]);
    }
}

} // namespace

bool Scene::loadProcedural(const VulkanContext& ctx, VkCommandPool pool) {
    destroy(ctx);

    // Checkerboard texture (RGBA8), used by the floor.
    constexpr uint32_t kTexSize = 128;
    constexpr uint32_t kTile = 16;
    std::vector<uint8_t> pixels(kTexSize * kTexSize * 4);
    for (uint32_t y = 0; y < kTexSize; ++y) {
        for (uint32_t x = 0; x < kTexSize; ++x) {
            const bool bright = ((x / kTile) + (y / kTile)) % 2 == 0;
            const uint8_t v = bright ? 210 : 70;
            const size_t o = (static_cast<size_t>(y) * kTexSize + x) * 4;
            pixels[o + 0] = v;
            pixels[o + 1] = v;
            pixels[o + 2] = v;
            pixels[o + 3] = 255;
        }
    }
    Texture checker;
    if (!uploadTexture(ctx, kTexSize, kTexSize, pixels.data(), checker, true, pool)) return false;
    textures.push_back(checker);

    // Shared unit-cube geometry.
    std::vector<Vertex> cubeVerts;
    std::vector<uint32_t> cubeIdx;
    makeCube(cubeVerts, cubeIdx);
    Mesh cube;
    if (!uploadMesh(ctx, cubeVerts, cubeIdx, cube, pool)) return false;
    meshes.push_back(cube);
    const uint32_t cubeMesh = 0;

    // Materials.
    Material floorMat;
    floorMat.baseColor = {0.9f, 0.9f, 0.9f, 1.f};
    floorMat.roughness = 0.85f;
    floorMat.texIndex = 0;
    materials.push_back(floorMat);
    const uint32_t floorMatIdx = 0;

    Material wallMat;
    wallMat.baseColor = {0.55f, 0.58f, 0.62f, 1.f};
    wallMat.roughness = 0.9f;
    materials.push_back(wallMat);
    const uint32_t wallMatIdx = 1;

    const Vec4 boxColors[] = {
        {0.85f, 0.25f, 0.25f, 1.f}, {0.25f, 0.7f, 0.3f, 1.f}, {0.25f, 0.45f, 0.85f, 1.f},
        {0.9f, 0.8f, 0.2f, 1.f},    {0.35f, 0.8f, 0.8f, 1.f}, {0.85f, 0.4f, 0.85f, 1.f},
        {0.9f, 0.55f, 0.2f, 1.f},   {0.85f, 0.85f, 0.85f, 1.f},
    };
    const uint32_t boxMatStart = static_cast<uint32_t>(materials.size());
    for (int i = 0; i < 8; ++i) {
        Material m;
        m.baseColor = boxColors[i];
        m.metallic = (i % 3 == 0) ? 0.7f : 0.05f;
        m.roughness = 0.25f + 0.09f * static_cast<float>(i % 4);
        materials.push_back(m);
    }

    // Static geometry: floor, three walls and a field of boxes.
    auto addInstance = [&](const Mat4& model, uint32_t material) {
        MeshInstance inst;
        inst.meshIndex = cubeMesh;
        inst.materialIndex = material;
        inst.model = model;
        inst.prevModel = model;
        instances.push_back(inst);
    };

    addInstance(composeTransform({0.f, -0.1f, 0.f}, 0.f, {20.f, 0.2f, 20.f}), floorMatIdx);
    addInstance(composeTransform({0.f, 5.f, -10.f}, 0.f, {20.f, 10.f, 0.2f}), wallMatIdx);
    addInstance(composeTransform({-10.f, 5.f, 0.f}, 0.f, {0.2f, 10.f, 20.f}), wallMatIdx);
    addInstance(composeTransform({10.f, 5.f, 0.f}, 0.f, {0.2f, 10.f, 20.f}), wallMatIdx);

    int material = 0;
    for (int gz = 0; gz < 5; ++gz) {
        for (int gx = 0; gx < 5; ++gx) {
            const float cx = -7.f + static_cast<float>(gx) * 3.5f;
            const float cz = -7.f + static_cast<float>(gz) * 3.5f;
            const float h = 1.0f + 0.35f * static_cast<float>((gx + gz) % 3);
            const float rot = 0.25f * static_cast<float>((gx * 5 + gz) % 7);
            const uint32_t mat = boxMatStart + static_cast<uint32_t>(material % 8);
            addInstance(composeTransform({cx, h * 0.5f, cz}, rot, {1.1f, h, 1.1f}), mat);
            ++material;
        }
    }

    lights = lightsFromPreset(lightingPresetForScene(""));

    updatePrevTransforms();
    finalizeInstances();
    buildMergedBuffers(ctx, pool);
    return true;
}

} // namespace sr
