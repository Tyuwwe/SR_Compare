#include "renderer/scene/Scene.h"

#include "renderer/core/VkUtil.h"

#include <cmath>
#include <set>
#include <string>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#define CGLTF_IMPLEMENTATION
#include <cgltf.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace sr {

namespace {

bool readVec3(const cgltf_accessor* acc, size_t index, Vec3& out) {
    float f[3] = {0.f, 0.f, 0.f};
    if (!cgltf_accessor_read_float(acc, index, f, 3)) return false;
    out = {f[0], f[1], f[2]};
    return true;
}

bool readVec2(const cgltf_accessor* acc, size_t index, Vec2& out) {
    float f[2] = {0.f, 0.f};
    if (!cgltf_accessor_read_float(acc, index, f, 2)) return false;
    out = {f[0], f[1]};
    return true;
}

bool readVec4(const cgltf_accessor* acc, size_t index, Vec4& out) {
    float f[4] = {0.f, 0.f, 0.f, 1.f};
    if (!cgltf_accessor_read_float(acc, index, f, 4)) return false;
    out = {f[0], f[1], f[2], f[3]};
    return true;
}

// Per-triangle tangent accumulation for meshes without a TANGENT attribute.
// Orthogonalized against the normal; mirrored-UV handedness is ignored (w=1).
void computeTangents(std::vector<Vertex>& verts, const std::vector<uint32_t>& idx) {
    for (auto& v : verts) v.tangent = {0.f, 0.f, 0.f, 1.f};
    for (size_t t = 0; t + 2 < idx.size(); t += 3) {
        Vertex& v0 = verts[idx[t]];
        Vertex& v1 = verts[idx[t + 1]];
        Vertex& v2 = verts[idx[t + 2]];
        const Vec3 dp1 = v1.position - v0.position;
        const Vec3 dp2 = v2.position - v0.position;
        const float du1 = v1.uv.x - v0.uv.x, dv1 = v1.uv.y - v0.uv.y;
        const float du2 = v2.uv.x - v0.uv.x, dv2 = v2.uv.y - v0.uv.y;
        const float det = du1 * dv2 - dv1 * du2;
        if (std::fabs(det) < 1e-12f) continue;
        const float r = 1.f / det;
        const Vec3 tan = (dp1 * dv2 - dp2 * dv1) * r;
        v0.tangent.x += tan.x; v0.tangent.y += tan.y; v0.tangent.z += tan.z;
        v1.tangent.x += tan.x; v1.tangent.y += tan.y; v1.tangent.z += tan.z;
        v2.tangent.x += tan.x; v2.tangent.y += tan.y; v2.tangent.z += tan.z;
    }
    for (auto& v : verts) {
        Vec3 t{v.tangent.x, v.tangent.y, v.tangent.z};
        t = t - v.normal * dot(v.normal, t); // Gram-Schmidt
        t = normalize(t);
        if (length(t) < 1e-6f) {
            // Degenerate UVs: pick any axis orthogonal to the normal.
            const Vec3 ref = std::fabs(v.normal.y) < 0.9f ? Vec3{0.f, 1.f, 0.f} : Vec3{1.f, 0.f, 0.f};
            t = normalize(cross(ref, v.normal));
        }
        v.tangent = {t.x, t.y, t.z, 1.f};
    }
}

int32_t findImageIndex(cgltf_data* data, const cgltf_image* image) {
    if (!image) return -1;
    for (int i = 0; i < static_cast<int>(data->images_count); ++i) {
        if (&data->images[i] == image) return i;
    }
    return -1;
}

int32_t findMaterialIndex(cgltf_data* data, const cgltf_material* mat) {
    if (!mat) return -1;
    for (int i = 0; i < static_cast<int>(data->materials_count); ++i) {
        if (&data->materials[i] == mat) return i;
    }
    return -1;
}

std::string dirOfPath(const char* path) {
    std::string p(path);
    const size_t slash = p.find_last_of("/\\");
    return slash == std::string::npos ? std::string() : p.substr(0, slash + 1);
}

} // namespace

bool Scene::loadGltf(const VulkanContext& ctx, const char* path, VkCommandPool pool,
                     const LoadProgressFn& progress) {
    destroy(ctx);
    auto report = [&](LoadStage stage, size_t done, size_t total) {
        if (progress) progress(stage, done, total);
    };
    report(LoadStage::Parse, 0, 0); // indeterminate

    cgltf_options options = {};
    cgltf_data* data = nullptr;
    if (cgltf_parse_file(&options, path, &data) != cgltf_result_success) return false;
    if (cgltf_load_buffers(&options, data, path) != cgltf_result_success) {
        cgltf_free(data);
        return false;
    }

    const std::string dir = dirOfPath(path);

    // Exact texture-upload total for the progress callback: the unique
    // (image, color space) pairs the material pass below will request.
    uint32_t totalTexUploads = 0;
    {
        std::set<std::pair<int32_t, bool>> unique;
        auto addTex = [&](const cgltf_texture* tex, bool srgb) {
            if (!tex || !tex->image) return;
            const int32_t idx = findImageIndex(data, tex->image);
            if (idx >= 0) unique.insert({idx, srgb});
        };
        for (int i = 0; i < static_cast<int>(data->materials_count); ++i) {
            const cgltf_material* m = &data->materials[i];
            addTex(m->pbr_metallic_roughness.base_color_texture.texture, true);
            addTex(m->pbr_metallic_roughness.metallic_roughness_texture.texture, false);
            addTex(m->normal_texture.texture, false);
            addTex(m->occlusion_texture.texture, false);
            addTex(m->emissive_texture.texture, true);
        }
        totalTexUploads = static_cast<uint32_t>(unique.size());
    }
    uint32_t texUploadsDone = 0;
    // First-attempt tracking so failed decodes retried by a later material do
    // not double-count against the total.
    std::vector<char> texAttempted(data->images_count * 2, 0);

    // Images decode on demand (CPU pixels cached per image) and upload as two
    // variants: sRGB (base color / emissive) and linear (normal / MR / AO).
    std::vector<std::vector<uint8_t>> imagePixels(data->images_count);
    std::vector<int> imageW(data->images_count, 0), imageH(data->images_count, 0);
    std::vector<int32_t> texSrgb(data->images_count, -1), texLinear(data->images_count, -1);

    auto loadImagePixels = [&](int i) -> bool {
        if (imagePixels[i].size()) return true;
        const cgltf_image* img = &data->images[i];
        int w = 0, h = 0, comp = 0;
        stbi_uc* pixels = nullptr;
        if (img->buffer_view) {
            const cgltf_buffer_view* bv = img->buffer_view;
            const auto* src = static_cast<const stbi_uc*>(bv->buffer->data) + bv->offset;
            pixels = stbi_load_from_memory(src, static_cast<int>(bv->size), &w, &h, &comp, 4);
        } else if (img->uri && img->uri[0] != '\0' && std::string(img->uri).rfind("data:", 0) != 0) {
            const std::string file = dir + img->uri;
            pixels = stbi_load(file.c_str(), &w, &h, &comp, 4);
        }
        if (!pixels) return false;
        imagePixels[i].assign(pixels, pixels + static_cast<size_t>(w) * h * 4);
        stbi_image_free(pixels);
        imageW[i] = w;
        imageH[i] = h;
        return true;
    };

    // Returns the Scene texture index for `image` in the requested color
    // space, uploading it on first use; -1 on failure.
    auto textureFor = [&](const cgltf_texture* tex, bool srgb) -> int32_t {
        if (!tex || !tex->image) return -1;
        const int32_t imgIdx = findImageIndex(data, tex->image);
        if (imgIdx < 0) return -1;
        std::vector<int32_t>& cache = srgb ? texSrgb : texLinear;
        int32_t& slot = cache[imgIdx];
        if (slot >= 0) return slot;
        const size_t attemptIdx = static_cast<size_t>(imgIdx) * 2 + (srgb ? 1u : 0u);
        if (!texAttempted[attemptIdx]) {
            texAttempted[attemptIdx] = 1;
            ++texUploadsDone;
            report(LoadStage::Textures, texUploadsDone, totalTexUploads);
        }
        if (!loadImagePixels(imgIdx)) return -1;
        Texture t;
        if (!uploadTexture(ctx, static_cast<uint32_t>(imageW[imgIdx]),
                           static_cast<uint32_t>(imageH[imgIdx]), imagePixels[imgIdx].data(), t,
                           srgb, pool))
            return -1;
        textures.push_back(t);
        slot = static_cast<int32_t>(textures.size() - 1);
        return slot;
    };

    // Materials (one per cgltf material, preserving index order).
    std::vector<int32_t> materialRemap(data->materials_count, -1);
    for (int i = 0; i < static_cast<int>(data->materials_count); ++i) {
        const cgltf_material* m = &data->materials[i];
        Material mat;
        const cgltf_pbr_metallic_roughness& pbr = m->pbr_metallic_roughness;
        mat.baseColor = {pbr.base_color_factor[0], pbr.base_color_factor[1],
                         pbr.base_color_factor[2], pbr.base_color_factor[3]};
        mat.metallic = pbr.metallic_factor;
        mat.roughness = pbr.roughness_factor;
        mat.texIndex = textureFor(pbr.base_color_texture.texture, true);
        mat.mrTexIndex = textureFor(pbr.metallic_roughness_texture.texture, false);
        mat.normalTexIndex = textureFor(m->normal_texture.texture, false);
        mat.aoTexIndex = textureFor(m->occlusion_texture.texture, false);
        if (m->occlusion_texture.texture) mat.occlusionStrength = m->occlusion_texture.scale;
        mat.emissiveTexIndex = textureFor(m->emissive_texture.texture, true);
        mat.emissiveFactor = {m->emissive_factor[0], m->emissive_factor[1], m->emissive_factor[2]};
        if (m->has_emissive_strength) {
            const float s = m->emissive_strength.emissive_strength;
            mat.emissiveFactor = mat.emissiveFactor * s;
        }
        if (m->alpha_mode == cgltf_alpha_mode_mask) mat.alphaCutoff = m->alpha_cutoff;
        if (m->alpha_mode == cgltf_alpha_mode_blend) mat.blend = true;
        materials.push_back(mat);
        materialRemap[i] = static_cast<int32_t>(materials.size() - 1);
    }

    // Meshes: each primitive becomes a Mesh + one instance per node that uses it.
    size_t totalPrims = 0;
    for (int m = 0; m < static_cast<int>(data->meshes_count); ++m)
        totalPrims += data->meshes[m].primitives_count;
    size_t primsDone = 0;
    for (int m = 0; m < static_cast<int>(data->meshes_count); ++m) {
        const cgltf_mesh* mesh = &data->meshes[m];
        for (int p = 0; p < static_cast<int>(mesh->primitives_count); ++p) {
            ++primsDone;
            report(LoadStage::Meshes, primsDone, totalPrims);
            const cgltf_primitive* prim = &mesh->primitives[p];

            const cgltf_accessor* pos = nullptr;
            const cgltf_accessor* normal = nullptr;
            const cgltf_accessor* uv = nullptr;
            const cgltf_accessor* tangent = nullptr;
            const cgltf_accessor* indices = prim->indices;
            for (int a = 0; a < static_cast<int>(prim->attributes_count); ++a) {
                const cgltf_attribute& attr = prim->attributes[a];
                if (attr.type == cgltf_attribute_type_position) pos = attr.data;
                else if (attr.type == cgltf_attribute_type_normal) normal = attr.data;
                else if (attr.type == cgltf_attribute_type_texcoord) uv = attr.data;
                else if (attr.type == cgltf_attribute_type_tangent) tangent = attr.data;
            }
            if (!pos) continue;

            const size_t vertexCount = pos->count;
            std::vector<Vertex> verts(vertexCount);
            for (size_t v = 0; v < vertexCount; ++v) {
                Vec3 n{0.f, 1.f, 0.f};
                Vec2 t{0.f, 0.f};
                readVec3(pos, v, verts[v].position);
                if (normal) readVec3(normal, v, n);
                if (uv) readVec2(uv, v, t);
                verts[v].normal = n;
                verts[v].uv = t;
                if (tangent) readVec4(tangent, v, verts[v].tangent);
            }

            std::vector<uint32_t> idx;
            if (indices) {
                idx.resize(indices->count);
                for (size_t i = 0; i < indices->count; ++i) {
                    idx[i] = static_cast<uint32_t>(cgltf_accessor_read_index(indices, i));
                }
            } else {
                idx.resize(vertexCount);
                for (size_t i = 0; i < vertexCount; ++i) idx[i] = static_cast<uint32_t>(i);
            }

            if (!tangent && uv) computeTangents(verts, idx);

            Mesh out;
            if (!uploadMesh(ctx, verts, idx, out, pool)) continue;
            meshes.push_back(out);
            const uint32_t meshIndex = static_cast<uint32_t>(meshes.size() - 1);

            int32_t matIdx = -1;
            const int32_t cgltfMat = findMaterialIndex(data, prim->material);
            if (cgltfMat >= 0) matIdx = materialRemap[cgltfMat];

            // Instances come from nodes that reference this mesh.
            for (int n = 0; n < static_cast<int>(data->nodes_count); ++n) {
                const cgltf_node* node = &data->nodes[n];
                if (node->mesh != mesh) continue;
                cgltf_float world[16];
                cgltf_node_transform_world(node, world);

                MeshInstance inst;
                inst.meshIndex = meshIndex;
                inst.materialIndex = matIdx >= 0 ? static_cast<uint32_t>(matIdx) : 0;
                std::memcpy(inst.model.m, world, sizeof(world));
                inst.prevModel = inst.model;
                instances.push_back(inst);
            }
        }
    }

    // Default lights when the file provides none (keeps the scene visible).
    if (lights.empty()) {
        lights.push_back({{4.f, 7.f, 4.f}, {1.f, 0.9f, 0.75f}, 140.f});
        lights.push_back({{-4.f, 3.f, -3.f}, {0.6f, 0.7f, 1.f}, 55.f});
    }

    cgltf_free(data);
    report(LoadStage::Finalize, 0, 0); // indeterminate
    updatePrevTransforms();
    finalizeInstances();
    buildMergedBuffers(ctx, pool);
    return !meshes.empty();
}

} // namespace sr
