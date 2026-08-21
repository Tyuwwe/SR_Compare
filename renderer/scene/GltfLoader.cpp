#include "renderer/scene/Scene.h"

#include "renderer/core/VkUtil.h"
#include "renderer/scene/LodBuilder.h"
#include "renderer/scene/SceneRegistry.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
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

// cgltf keeps unparsed extensions as raw JSON (name + data); MSFT_lod carries
// {"ids":[n0,n1,...]} — nodes that are the next-coarser LODs of this node.
std::vector<int32_t> parseMsftLodIds(const cgltf_node* node, cgltf_size nodeCount) {
    std::vector<int32_t> ids;
    for (size_t e = 0; e < node->extensions_count; ++e) {
        const cgltf_extension& ext = node->extensions[e];
        if (!ext.name || std::strcmp(ext.name, "MSFT_lod") != 0 || !ext.data) continue;
        const char* p = std::strstr(ext.data, "\"ids\"");
        if (!p) continue;
        p = std::strchr(p, '[');
        if (!p) continue;
        ++p;
        while (*p && *p != ']') {
            while (*p && *p != ']' && (std::isspace(static_cast<unsigned char>(*p)) || *p == ','))
                ++p;
            if (!*p || *p == ']') break;
            char* end = nullptr;
            const long v = std::strtol(p, &end, 10);
            if (end == p) break; // malformed: stop instead of looping forever
            p = end;
            if (v >= 0 && static_cast<cgltf_size>(v) < nodeCount)
                ids.push_back(static_cast<int32_t>(v));
        }
    }
    return ids;
}

// Splits a node's baked matrix into TRS so animation channels can override
// individual components.  Translation = column 3; scale = column lengths
// (sign from the determinant); rotation = orthonormalized basis -> quaternion.
void decomposeMatrix(const cgltf_float m[16], Vec3& t, Vec4& q, Vec3& s) {
    t = {m[12], m[13], m[14]};
    const Vec3 c0{m[0], m[1], m[2]}, c1{m[4], m[5], m[6]}, c2{m[8], m[9], m[10]};
    s = {length(c0), length(c1), length(c2)};
    const float det = dot(c0, cross(c1, c2));
    if (det < 0.f) s.x = -s.x;
    const float r00 = c0.x / s.x, r10 = c0.y / s.x, r20 = c0.z / s.x;
    const float r01 = c1.x / s.y, r11 = c1.y / s.y, r21 = c1.z / s.y;
    const float r02 = c2.x / s.z, r12 = c2.y / s.z, r22 = c2.z / s.z;
    // Standard trace-based matrix->quaternion (largest-component branch).
    const float trace = r00 + r11 + r22;
    if (trace > 0.f) {
        const float w = std::sqrt(trace + 1.f) * 0.5f;
        const float inv = 0.25f / w;
        q = {(r21 - r12) * inv, (r02 - r20) * inv, (r10 - r01) * inv, w};
    } else if (r00 > r11 && r00 > r22) {
        const float x = std::sqrt(1.f + r00 - r11 - r22) * 0.5f;
        const float inv = 0.25f / x;
        q = {x, (r01 + r10) * inv, (r02 + r20) * inv, (r21 - r12) * inv};
    } else if (r11 > r22) {
        const float y = std::sqrt(1.f + r11 - r00 - r22) * 0.5f;
        const float inv = 0.25f / y;
        q = {(r01 + r10) * inv, y, (r12 + r21) * inv, (r02 - r20) * inv};
    } else {
        const float z = std::sqrt(1.f + r22 - r00 - r11) * 0.5f;
        const float inv = 0.25f / z;
        q = {(r02 + r20) * inv, (r12 + r21) * inv, z, (r10 - r01) * inv};
    }
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
        } else if (m->name) {
            // Bistro street/lantern materials ship with strength 1; without a
            // reconvert they stay dim.  Skip this path when the glTF already
            // authored KHR_materials_emissive_strength.
            std::string lower(m->name);
            for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (lower.find("lantern") != std::string::npos ||
                lower.find("streetlight") != std::string::npos ||
                lower.find("emissive") != std::string::npos) {
                mat.emissiveFactor = mat.emissiveFactor * 8.f;
            }
        }
        if (m->alpha_mode == cgltf_alpha_mode_mask) mat.alphaCutoff = m->alpha_cutoff;
        if (m->alpha_mode == cgltf_alpha_mode_blend) mat.blend = true;
        materials.push_back(mat);
        materialRemap[i] = static_cast<int32_t>(materials.size() - 1);
    }

    // --- node tree / animations / skins -----------------------------------------
    // Kept only when the file actually uses them; static files (Sponza,
    // Bistro) take the old bake-the-world-matrix path untouched.
    const bool keepNodes = data->animations_count > 0 || data->skins_count > 0;
    std::vector<int32_t> skinRemap;
    if (keepNodes) {
        const size_t nodeCount = data->nodes_count;
        nodes.resize(nodeCount);
        for (size_t i = 0; i < nodeCount; ++i) {
            const cgltf_node* node = &data->nodes[i];
            SceneNode& sn = nodes[i];
            if (node->has_matrix) {
                decomposeMatrix(node->matrix, sn.translation, sn.rotation, sn.scale);
            } else {
                sn.translation = {node->translation[0], node->translation[1], node->translation[2]};
                sn.rotation = {node->rotation[0], node->rotation[1], node->rotation[2],
                               node->rotation[3]};
                sn.scale = {node->scale[0], node->scale[1], node->scale[2]};
            }
        }
        for (size_t i = 0; i < nodeCount; ++i) {
            for (size_t c = 0; c < data->nodes[i].children_count; ++c) {
                const size_t child = static_cast<size_t>(data->nodes[i].children[c] - data->nodes);
                nodes[child].parent = static_cast<int32_t>(i);
            }
        }
        // Parents-before-children order for the per-frame global evaluation.
        for (size_t i = 0; i < nodeCount; ++i) {
            if (nodes[i].parent < 0) nodeTopoOrder.push_back(static_cast<uint32_t>(i));
        }
        for (size_t cursor = 0; cursor < nodeTopoOrder.size(); ++cursor) {
            const int32_t cur = static_cast<int32_t>(nodeTopoOrder[cursor]);
            for (size_t i = 0; i < nodeCount; ++i) {
                if (nodes[i].parent == cur) nodeTopoOrder.push_back(static_cast<uint32_t>(i));
            }
        }

        // Only the first animation is played (no multi-clip blending).
        if (data->animations_count > 0) {
            const cgltf_animation* anim = &data->animations[0];
            for (size_t s = 0; s < anim->samplers_count; ++s) {
                const cgltf_animation_sampler* cs = &anim->samplers[s];
                AnimSampler smp;
                smp.step = cs->interpolation == cgltf_interpolation_type_step;
                // CUBICSPLINE carries 3 values per key (in-tangent, point,
                // out-tangent); keep the points and lerp between them.
                const size_t stride = cs->interpolation == cgltf_interpolation_type_cubic_spline
                                          ? 3
                                          : 1;
                const size_t comps = cs->output->type == cgltf_type_vec4 ? 4 : 3;
                const size_t keyCount = cs->input->count;
                smp.times.resize(keyCount);
                smp.values.resize(keyCount);
                for (size_t k = 0; k < keyCount; ++k) {
                    float t = 0.f;
                    cgltf_accessor_read_float(cs->input, k, &t, 1);
                    smp.times[k] = t;
                    float f[4] = {0.f, 0.f, 0.f, 1.f};
                    cgltf_accessor_read_float(cs->output, k * stride, f, comps);
                    smp.values[k] = {f[0], f[1], f[2], f[3]};
                }
                if (!smp.times.empty()) animDuration = std::max(animDuration, smp.times.back());
                animSamplers.push_back(std::move(smp));
            }
            for (size_t c = 0; c < anim->channels_count; ++c) {
                const cgltf_animation_channel* cc = &anim->channels[c];
                if (!cc->target_node || !cc->sampler) continue;
                AnimChannel ch;
                ch.node = static_cast<uint32_t>(cc->target_node - data->nodes);
                ch.sampler = static_cast<uint32_t>(cc->sampler - anim->samplers);
                switch (cc->target_path) {
                case cgltf_animation_path_type_rotation: ch.path = AnimPath::Rotation; break;
                case cgltf_animation_path_type_scale: ch.path = AnimPath::Scale; break;
                default: ch.path = AnimPath::Translation; break;
                }
                animChannels.push_back(ch);
            }
        }

        skinRemap.assign(data->skins_count, -1);
        for (size_t s = 0; s < data->skins_count; ++s) {
            const cgltf_skin* cs = &data->skins[s];
            Skin skin;
            skin.joints.resize(cs->joints_count);
            skin.inverseBind.resize(cs->joints_count, Mat4::identity());
            for (size_t j = 0; j < cs->joints_count; ++j) {
                skin.joints[j] = static_cast<uint32_t>(cs->joints[j] - data->nodes);
                if (cs->inverse_bind_matrices) {
                    cgltf_accessor_read_float(cs->inverse_bind_matrices, j,
                                              skin.inverseBind[j].m, 16);
                }
            }
            skinRemap[s] = static_cast<int32_t>(skins.size());
            skins.push_back(std::move(skin));
        }
    }

    // Meshes: each primitive becomes a Mesh + one instance per node that uses it.
    // Skinned primitives (node has a skin AND the primitive has JOINTS_0 /
    // WEIGHTS_0) become a SkinnedVertex mesh instead; dedup maps keep both
    // variants shared across nodes referencing the same primitive.
    std::vector<std::vector<int32_t>> skinnedMeshIdx;
    if (keepNodes) {
        skinnedMeshIdx.resize(data->meshes_count);
        for (size_t m = 0; m < data->meshes_count; ++m)
            skinnedMeshIdx[m].assign(data->meshes[m].primitives_count, -1);
    }

    // (mesh, primitive) -> scene mesh index for static uploads, needed to
    // resolve MSFT_lod chains whose LOD nodes may be processed later.
    std::vector<std::vector<int32_t>> staticMeshIdx(data->meshes_count);
    for (size_t m = 0; m < data->meshes_count; ++m)
        staticMeshIdx[m].assign(data->meshes[m].primitives_count, -1);

    // MSFT_lod: nodes referenced by another node's "ids" are LOD variants of
    // it; they get no own instance, the master instance switches to their
    // meshes instead.  Skinned masters keep a single level (see Scene.h).
    std::vector<char> lodAuxNode(data->nodes_count, 0);
    std::vector<std::vector<int32_t>> lodIdsPerNode(data->nodes_count);
    for (size_t n = 0; n < data->nodes_count; ++n) {
        const cgltf_node* node = &data->nodes[n];
        if (!node->mesh || node->skin) continue;
        lodIdsPerNode[n] = parseMsftLodIds(node, data->nodes_count);
        for (const int32_t id : lodIdsPerNode[n]) {
            if (data->nodes[id].mesh) lodAuxNode[static_cast<size_t>(id)] = 1;
        }
    }
    // Instances whose authored chain must be resolved after ALL meshes are
    // uploaded (LOD nodes can reference meshes later in the file).
    struct PendingLodChain {
        uint32_t instanceIndex;
        uint32_t node;
        uint32_t prim;
    };
    std::vector<PendingLodChain> pendingLodChains;
    // CPU geometry stash for runtime LOD generation (moved, not copied).
    std::vector<LodSourceMesh> lodSources;

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
            const cgltf_accessor* jointsAcc = nullptr;
            const cgltf_accessor* weightsAcc = nullptr;
            const cgltf_accessor* indices = prim->indices;
            for (int a = 0; a < static_cast<int>(prim->attributes_count); ++a) {
                const cgltf_attribute& attr = prim->attributes[a];
                if (attr.type == cgltf_attribute_type_position) pos = attr.data;
                else if (attr.type == cgltf_attribute_type_normal) normal = attr.data;
                else if (attr.type == cgltf_attribute_type_texcoord && attr.index == 0) uv = attr.data;
                else if (attr.type == cgltf_attribute_type_tangent) tangent = attr.data;
                else if (attr.type == cgltf_attribute_type_joints && attr.index == 0) jointsAcc = attr.data;
                else if (attr.type == cgltf_attribute_type_weights && attr.index == 0) weightsAcc = attr.data;
            }
            if (!pos) continue;

            const size_t vertexCount = pos->count;

            int32_t matIdx = -1;
            const int32_t cgltfMat = findMaterialIndex(data, prim->material);
            if (cgltfMat >= 0) matIdx = materialRemap[cgltfMat];

            // A primitive is drawn skinned for a node only when that node has
            // a skin AND the primitive actually carries JOINTS_0/WEIGHTS_0.
            const bool canSkin = keepNodes && jointsAcc && weightsAcc;
            bool needStatic = true;
            if (canSkin) {
                needStatic = false;
                for (int n = 0; n < static_cast<int>(data->nodes_count); ++n) {
                    const cgltf_node* node = &data->nodes[n];
                    if (node->mesh == mesh && !node->skin) needStatic = true;
                }
            }

            uint32_t meshIndex = 0;
            if (needStatic) {
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
                meshIndex = static_cast<uint32_t>(meshes.size() - 1);
                staticMeshIdx[static_cast<size_t>(m)][static_cast<size_t>(p)] =
                    static_cast<int32_t>(meshIndex);
                // Stash the CPU copy for LOD generation (move: uploadMesh
                // already fed the merged buffers, so this costs nothing).
                lodSources.push_back({meshIndex, std::move(verts), std::move(idx)});
            }

            // Instances come from nodes that reference this mesh.
            for (int n = 0; n < static_cast<int>(data->nodes_count); ++n) {
                const cgltf_node* node = &data->nodes[n];
                if (node->mesh != mesh) continue;
                if (lodAuxNode[static_cast<size_t>(n)]) continue; // drawn via the LOD master

                MeshInstance inst;
                inst.materialIndex = matIdx >= 0 ? static_cast<uint32_t>(matIdx) : 0;

                if (canSkin && node->skin) {
                    // Skinned instance: positions come from the joint palette,
                    // so the per-instance model stays identity (glTF spec: a
                    // skinned mesh ignores its node's own transform).
                    int32_t& sidx = skinnedMeshIdx[static_cast<size_t>(m)][static_cast<size_t>(p)];
                    if (sidx < 0) {
                        std::vector<SkinnedVertex> sverts(vertexCount);
                        for (size_t v = 0; v < vertexCount; ++v) {
                            Vec3 nrm{0.f, 1.f, 0.f};
                            Vec2 t{0.f, 0.f};
                            readVec3(pos, v, sverts[v].position);
                            if (normal) readVec3(normal, v, nrm);
                            if (uv) readVec2(uv, v, t);
                            sverts[v].normal = nrm;
                            sverts[v].uv = t;
                            if (tangent) readVec4(tangent, v, sverts[v].tangent);
                            cgltf_uint j[4] = {0, 0, 0, 0};
                            cgltf_accessor_read_uint(jointsAcc, v, j, 4);
                            for (int k = 0; k < 4; ++k)
                                sverts[v].joints[k] = static_cast<uint16_t>(j[k]);
                            Vec4 w{1.f, 0.f, 0.f, 0.f};
                            readVec4(weightsAcc, v, w);
                            const float sum = w.x + w.y + w.z + w.w;
                            if (sum > 1e-6f) {
                                const float inv = 1.f / sum;
                                w = {w.x * inv, w.y * inv, w.z * inv, w.w * inv};
                            }
                            sverts[v].weights = w;
                        }
                        std::vector<uint32_t> sidxBuf;
                        if (indices) {
                            sidxBuf.resize(indices->count);
                            for (size_t i = 0; i < indices->count; ++i)
                                sidxBuf[i] = static_cast<uint32_t>(cgltf_accessor_read_index(indices, i));
                        } else {
                            sidxBuf.resize(vertexCount);
                            for (size_t i = 0; i < vertexCount; ++i)
                                sidxBuf[i] = static_cast<uint32_t>(i);
                        }
                        Mesh sout;
                        if (!uploadSkinnedMesh(ctx, sverts, sidxBuf, sout, pool)) break;
                        skinnedMeshes.push_back(sout);
                        sidx = static_cast<int32_t>(skinnedMeshes.size() - 1);
                    }
                    inst.meshIndex = static_cast<uint32_t>(sidx);
                    inst.skinIndex = skinRemap[static_cast<size_t>(node->skin - data->skins)];
                    inst.model = Mat4::identity();
                    inst.prevModel = Mat4::identity();
                    instances.push_back(inst);
                    continue;
                }

                cgltf_float world[16];
                cgltf_node_transform_world(node, world);
                inst.meshIndex = meshIndex;
                std::memcpy(inst.model.m, world, sizeof(world));
                inst.prevModel = inst.model;
                if (keepNodes) inst.nodeIndex = n; // per-frame advance rewrites model/prevModel
                instances.push_back(inst);
                if (!lodIdsPerNode[static_cast<size_t>(n)].empty()) {
                    pendingLodChains.push_back({static_cast<uint32_t>(instances.size() - 1),
                                                static_cast<uint32_t>(n), static_cast<uint32_t>(p)});
                }
            }
        }
    }

    // Resolve authored MSFT_lod chains now that every mesh exists.  Chain
    // members are excluded from runtime simplification (authored wins).
    std::vector<char> lodSkipMeshes;
    if (!pendingLodChains.empty()) lodSkipMeshes.assign(meshes.size(), 0);
    for (const PendingLodChain& pc : pendingLodChains) {
        MeshInstance& inst = instances[pc.instanceIndex];
        uint32_t cnt = 0;
        inst.authoredLodMeshes[cnt++] = inst.meshIndex;
        for (const int32_t id : lodIdsPerNode[pc.node]) {
            if (cnt >= kMaxMeshLods) break;
            const cgltf_node* aux = &data->nodes[static_cast<size_t>(id)];
            if (!aux->mesh) continue;
            const size_t am = static_cast<size_t>(aux->mesh - data->meshes);
            // LOD meshes are expected to mirror the master's primitive count;
            // clamp so a shorter aux mesh still yields a usable level.
            const size_t ap = std::min<size_t>(pc.prim, aux->mesh->primitives_count - 1);
            const int32_t mi = staticMeshIdx[am][ap];
            if (mi < 0) continue;
            inst.authoredLodMeshes[cnt++] = static_cast<uint32_t>(mi);
        }
        inst.authoredLodCount = cnt;
        if (cnt > 1) {
            for (uint32_t k = 0; k < cnt; ++k) lodSkipMeshes[inst.authoredLodMeshes[k]] = 1;
        } else {
            inst.authoredLodCount = 0; // no usable aux mesh: fall back to generated LODs
        }
    }

    // KHR_lights_punctual: cgltf exposes data->lights via node->light.
    // Directional, point and spot lights are supported.
    for (int n = 0; n < static_cast<int>(data->nodes_count); ++n) {
        const cgltf_node* node = &data->nodes[n];
        if (!node->light) continue;
        const cgltf_light* gl = node->light;

        cgltf_float world[16];
        cgltf_node_transform_world(node, world);
        Mat4 worldMat;
        std::memcpy(worldMat.m, world, sizeof(world));

        Light l;
        l.color = {gl->color[0], gl->color[1], gl->color[2]};
        if (gl->type == cgltf_light_type_directional) {
            l.type = LightType::Directional;
            // glTF convention: the light shines along the node's local -Z, so
            // the direction *towards* the light (the shader's L) is local +Z
            // rotated into world space.
            l.positionOrDirection =
                normalize(transformDirection(worldMat, Vec3{0.f, 0.f, 1.f}));
            // glTF directional intensity is in lux (a real sun is ~100k);
            // scale to engine units where a sun reads as ~1-10, comparable to
            // the IBL environment brightness.
            l.intensity = gl->intensity / 10000.f;
        } else {
            l.type = gl->type == cgltf_light_type_spot ? LightType::Spot : LightType::Point;
            l.positionOrDirection = transformPoint(worldMat, Vec3{0.f, 0.f, 0.f});
            // glTF point/spot intensity is in candela, which maps 1:1 onto the
            // engine's inverse-square point-light units.
            l.intensity = gl->intensity;
            l.range = gl->range; // 0 = infinite in both glTF and the engine
            if (l.type == LightType::Spot) {
                // The cone shines along the node's local -Z (same convention
                // as directional, opposite sign).  cgltf fills
                // inner/outer_cone_angle with the spec defaults (0 / pi/4);
                // clamp so cos(inner) > cos(outer) keeps a non-zero penumbra
                // band in the shader's smoothstep.
                l.spotDirection =
                    normalize(transformDirection(worldMat, Vec3{0.f, 0.f, -1.f}));
                l.innerConeAngle = gl->spot_inner_cone_angle;
                l.outerConeAngle = std::max(gl->spot_outer_cone_angle, gl->spot_inner_cone_angle + 1e-3f);
                // Authored spots are shadow-casting by default (DCC authoring
                // intent); the Phase 4b atlas selection caps how many actually
                // render a map per frame.  Points stay non-casting this phase.
                l.castShadow = true;
            }
        }
        lights.push_back(l);
    }

    // Merge authored KHR_lights with the scene lighting preset.  Empty files
    // take the full preset (sun + optional fill).  Bistro's golden-hour sun
    // replaces authored directionals so the FBX key light cannot override it;
    // authored point lights (lanterns, strings) are kept.
    const LightingPreset preset = lightingPresetForScene(path);
    if (lights.empty()) {
        lights = lightsFromPreset(preset);
    } else if (preset.preferPresetSun) {
        std::vector<Light> merged = lightsFromPreset(preset);
        for (const Light& l : lights) {
            if (l.type != LightType::Directional) merged.push_back(l);
        }
        lights = std::move(merged);
    } else {
        bool hasDir = false;
        for (const Light& l : lights) {
            if (l.type == LightType::Directional) {
                hasDir = true;
                break;
            }
        }
        if (!hasDir && preset.sunEnabled) {
            lights.insert(lights.begin(),
                          makeSunLight(preset.sunElevationDeg, preset.sunAzimuthDeg,
                                       preset.sunIntensity, preset.sunColor));
        }
    }

    cgltf_free(data);
    report(LoadStage::Finalize, 0, 0); // indeterminate
    updatePrevTransforms();
    finalizeInstances();
    if (keepNodes) computeAnimatedBounds();
    // Runtime LOD generation (cached next to the glTF) + per-instance draw
    // tables; must run after the merged index accumulation is complete and
    // before it is uploaded.
    generateMeshLods(*this, lodSources, path, lodSkipMeshes);
    buildLodDraws();
    buildMergedBuffers(ctx, pool);
    if (keepNodes && !skins.empty() && !createSkinPalettes(ctx)) return false;
    return !meshes.empty() || !skinnedMeshes.empty();
}

} // namespace sr
