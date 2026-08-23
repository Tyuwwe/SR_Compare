#pragma once
// ============================================================================
// Scene data + GPU resources.  Meshes/textures are uploaded through the Vulkan
// context; the actual construction lives in ProceduralScene / GltfLoader.
// ============================================================================
#include "renderer/core/Vk.h"
#include "renderer/core/VulkanContext.h"
#include "renderer/math/Math.h"
#include "renderer/scene/Ktx2.h"

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace sr {

struct Vertex {
    Vec3 position;
    Vec3 normal;
    Vec2 uv;
    Vec4 tangent{1.f, 0.f, 0.f, 1.f}; // xyz = tangent, w = bitangent sign
};

// --- LOD -------------------------------------------------------------------
// Per-mesh LOD chains (2020-standard: discrete levels switched by projected
// screen size).  Two sources, unified into the same per-instance draw-range
// table at load time:
//  - authored: glTF MSFT_lod node extension (chain of sibling nodes/meshes);
//  - generated: meshopt_simplify index-only decimation (LodBuilder.cpp).
// Skinned meshes deliberately keep a single level: simplification would have
// to preserve skin weights/bounds and skinned meshes are rare, so the win
// does not pay for the complexity.
constexpr uint32_t kMaxMeshLods = 4; // LOD0 + up to 3 coarser levels

// Hosts default LOD selection to on; SR_LOD=0 disables it (bench A/B
// comparisons against pre-LOD behavior in the same binary).
inline bool lodEnabledByDefault() {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996) // std::getenv is portable; _dupenv_s is MSVC-only
#endif
    const char* v = std::getenv("SR_LOD");
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    return !v || v[0] != '0';
}

// Hosts default GPU occlusion culling (Phase 7a, Hi-Z reproject + indirect
// draws) to on; SR_OCCLUSION=0 disables the cull pass (the GBuffer stays on
// the indirect path with every command visible, so A/B runs share shaders).
inline bool occlusionEnabledByDefault() {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996) // std::getenv is portable; _dupenv_s is MSVC-only
#endif
    const char* v = std::getenv("SR_OCCLUSION");
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    return !v || v[0] != '0';
}

// --- Texture compression / mip streaming (Phase 7b) ---------------------------
// Textures prefer a sibling .ktx2 (BC7, pre-baked full mip chain; see
// scripts/transcode_textures.py) over the referenced .png/.jpg.  BC7 cuts
// texture VRAM ~4x vs RGBA8.  sRGB vs UNORM is decided by material usage at
// image-creation time (baseColor/emissive sRGB, normal/MR/AO linear) — the
// block payload is identical either way.  Normal maps stay BC7 (not BC5) to
// keep one code path; BC5 is a possible follow-up.
//
// Mip streaming: interactive hosts (viewer free-fly, GUI) upload only the
// coarse tail (mip >= kStreamResidentBaseMip) of large KTX2 textures at load
// and fill in finer levels from a background thread, nearest-to-camera first,
// capped at kStreamBudgetBytesPerFrame per frame tick.  Un-uploaded levels
// sit in SHADER_READ_ONLY with undefined contents (fresh device-local VRAM is
// zeroed by the OS) — transient low-quality placeholders, never validation
// errors.  Bench/compare/screenshot runs disable streaming (full upload at
// load) so every frame is bit-reproducible; the PNG fallback path never
// streams (no pre-baked chain to read back).  SR_TEX_STREAM: 0 = force off,
// 1 = force on (debug/validation of the streaming path), unset = default.
inline int texStreamingOverride() {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996) // std::getenv is portable; _dupenv_s is MSVC-only
#endif
    const char* v = std::getenv("SR_TEX_STREAM");
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    if (!v) return 0;
    return v[0] == '1' ? 1 : -1;
}
inline constexpr uint32_t kStreamResidentBaseMip = 4; // resident: 1/16 resolution tail
inline constexpr uint64_t kStreamBudgetBytesPerFrame = 8ull * 1024 * 1024;

// One draw range into the merged scene buffers (index span + vertex offset).
struct LodDraw {
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    int32_t vertexOffset = 0;
};

struct Mesh {
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VmaAllocation vertexMemory = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VmaAllocation indexMemory = VK_NULL_HANDLE;
    uint32_t indexCount = 0;
    VkIndexType indexType = VK_INDEX_TYPE_UINT32;
    Vec3 aabbMin{0.f, 0.f, 0.f}; // local-space bounds (for frustum culling)
    Vec3 aabbMax{0.f, 0.f, 0.f};
    // Location inside the merged scene-wide buffers (viewer draw path).
    uint32_t firstIndex = 0;
    int32_t vertexOffset = 0;
    // LOD chain as ranges into the merged index buffer; generated levels
    // share the original vertices (meshopt_simplify only re-indexes), so
    // vertexOffset stays the mesh's own.  lods[0] mirrors the fields above.
    LodDraw lods[kMaxMeshLods] = {};
    uint32_t lodCount = 1;
};

struct Texture {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation memory = VK_NULL_HANDLE;
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
    // Opaque-mirror glass (still routed through the transparency pass so it
    // gets SSR + motion/mask outputs): the shader kills transmission and
    // writes alpha = 1.  Set by GltfLoader's per-scene mirror table.
    bool mirror = false;
};

struct MeshInstance {
    uint32_t meshIndex = 0;      // meshes[], or skinnedMeshes[] when skinIndex >= 0
    uint32_t materialIndex = 0;
    Mat4 model;
    Mat4 prevModel; // previous-frame transform; rewritten per frame for dynamic instances
    Mat4 normalModel;    // cached transpose(inverse(model)); only the upper 3x3 matters
    Vec3 aabbMin{0.f, 0.f, 0.f}; // world-space bounds (for frustum culling)
    Vec3 aabbMax{0.f, 0.f, 0.f};
    int32_t nodeIndex = -1; // scene node driving this instance (-1 = static)
    int32_t skinIndex = -1; // skins[] entry (-1 = rigid); skinned draws use skinnedMeshes

    // Per-frame LOD state (updateLodSelection; deterministic: pure function of
    // camera + instance bounds with hysteresis, no wall-clock input).
    uint32_t lodLevel = 0; // index into lodDraws
    bool lodCulled = false; // projected size below the cull radius: skip draw
    // Draw range per LOD level.  Mirrors meshes[meshIndex].lods unless the
    // glTF authored an MSFT_lod chain, in which case each level points at the
    // LOD node's own mesh.  Unused by the skinned draw path.
    LodDraw lodDraws[kMaxMeshLods] = {};
    uint32_t lodDrawCount = 1;
    // Authored MSFT_lod chain (scene mesh indices per level, [0] = this
    // instance's mesh); consumed once by buildLodDraws(), then stale.
    uint32_t authoredLodMeshes[kMaxMeshLods] = {};
    uint32_t authoredLodCount = 0;
};

// Skinned vertex: same layout as Vertex plus JOINTS0/WEIGHTS0.  Joints index
// into the owning skin's joint list (never the palette directly).
struct SkinnedVertex {
    Vec3 position;
    Vec3 normal;
    Vec2 uv;
    Vec4 tangent{1.f, 0.f, 0.f, 1.f};
    uint16_t joints[4] = {0, 0, 0, 0};
    Vec4 weights{1.f, 0.f, 0.f, 0.f}; // normalized on load
};

// --- glTF animation / skinning state ------------------------------------------
// Populated only when the loaded file has animations or skins; scenes without
// them (Sponza, Bistro, procedural) keep zero overhead and the exact old
// static behaviour.

struct SceneNode {
    int32_t parent = -1;
    Vec3 translation{0.f, 0.f, 0.f};
    Vec4 rotation{0.f, 0.f, 0.f, 1.f}; // quaternion xyzw
    Vec3 scale{1.f, 1.f, 1.f};
};

enum class AnimPath : uint32_t { Translation = 0, Rotation = 1, Scale = 2 };

struct AnimSampler {
    std::vector<float> times;
    std::vector<Vec4> values; // translation/scale in xyz, rotation quaternion in xyzw
    bool step = false;        // STEP interpolation (default: LINEAR; CUBICSPLINE unsupported)
};

struct AnimChannel {
    uint32_t node = 0;
    AnimPath path = AnimPath::Translation;
    uint32_t sampler = 0;
};

struct Skin {
    std::vector<uint32_t> joints;   // scene node indices
    std::vector<Mat4> inverseBind;  // one per joint
    // Mat4 offsets into the joint palette buffer (see Scene::skinPaletteBuffer).
    uint32_t paletteCur = 0;
    uint32_t palettePrev = 0;
};

// Procedural test-scene motion driver: a box yawing around its own centre and
// sliding on a sinusoidal path.  Analytic (not keyframed) so the pose is a
// pure function of the frame index — bench determinism depends on that.
struct DynamicBoxDriver {
    uint32_t instanceIndex = 0;
    Vec3 basePos{0.f, 0.f, 0.f};
    float baseYaw = 0.f;
    Vec3 scale{1.f, 1.f, 1.f};
    float yawRate = 0.f;            // rad/s around Y
    Vec3 slideAmp{0.f, 0.f, 0.f};   // sinusoidal translation amplitude (m)
    float slidePeriod = 1.f;        // seconds
    float slidePhase = 0.f;
};

// Fixed animation timestep: poses are sampled at frame * kAnimDt (never wall
// clock) so a fixed camera path is bit-reproducible across runs.
constexpr float kAnimDt = 1.f / 60.f;

// Joint palettes are double-buffered to match the hosts' kFramesInFlight: the
// palette written for frame N lives in slot N % kSkinPaletteSlots and is only
// overwritten once that slot's fence has passed (same rule as the per-slot
// UBOs), so in-flight frames never read a half-updated palette.
constexpr uint32_t kSkinPaletteSlots = 2;

// Punctual light, modelled after KHR_lights_punctual.  Plain POD: the GPU
// packing (std430 LightGPU) happens in DeferredCore::fillLightingUBO and
// DeferredCore::fillClusterLights.
enum class LightType : uint32_t { Directional = 0, Point = 1, Spot = 2 };

struct Light {
    LightType type = LightType::Point;
    // Point/spot: world-space position.  Directional: unit direction *towards*
    // the light (i.e. the shader's L), so a sun shining straight down is (0,1,0).
    Vec3 positionOrDirection{0.f, 1.f, 0.f};
    Vec3 color{1.f, 1.f, 1.f};
    float intensity = 1.f;
    float range = 0.f;       // point/spot only, 0 = infinite (pure inverse-square)
    bool castShadow = false; // CSM sun (directional) or spot shadow atlas eligibility
    // Spot only (KHR_lights_punctual): unit direction the cone points (world
    // space) and the inner/outer half-angles in radians (umbra/penumbra).
    // Shading fades smoothly from innerCos to outerCos; the cluster assignment
    // pass bounds the cone with a conservative sphere of height `range`.
    Vec3 spotDirection{0.f, -1.f, 0.f};
    float innerConeAngle = 0.f;                    // full intensity inside
    float outerConeAngle = 3.14159265f / 4.f;      // zero intensity outside
    int32_t shadowIndex = -1; // spot shadow atlas tile (Phase 4b), -1 =
                              // unshadowed; rewritten per frame by
                              // DeferredCore::selectSpotShadowLights
};

// Elevation (deg above horizon) / azimuth (deg, 0 = +Z, 90 = +X) to a unit
// direction *towards* the sun.  Matches the GUI lighting sliders.
inline Vec3 sunDirectionFromElevAzimuth(float elevationDeg, float azimuthDeg) {
    const float el = elevationDeg * (3.14159265f / 180.f);
    const float az = azimuthDeg * (3.14159265f / 180.f);
    return normalize(Vec3{std::cos(el) * std::sin(az), std::sin(el),
                          std::cos(el) * std::cos(az)});
}

inline Light makeSunLight(float elevationDeg, float azimuthDeg, float intensity, Vec3 color) {
    Light sun;
    sun.type = LightType::Directional;
    sun.positionOrDirection = sunDirectionFromElevAzimuth(elevationDeg, azimuthDeg);
    sun.color = color;
    sun.intensity = intensity;
    sun.castShadow = true;
    return sun;
}

inline Light defaultFillLight() {
    Light fill;
    fill.type = LightType::Point;
    fill.positionOrDirection = {-4.f, 5.f, -3.f};
    fill.color = {0.6f, 0.7f, 1.f};
    fill.intensity = 55.f; // candela-like units (inverse-square falloff)
    return fill;
}

// Shared fallback lighting for scenes without authored lights (procedural
// scene, glTF files without KHR_lights_punctual, and the DeferredCore UBO
// filler safety net).  Single source of truth so the three hosts and all
// loaders stay identical.  Angles match the GUI lighting-slider defaults.
inline const std::vector<Light>& defaultLights() {
    static const std::vector<Light> lights = [] {
        std::vector<Light> v;
        v.push_back(makeSunLight(65.3f, 49.4f, 3.f, {1.f, 0.95f, 0.85f}));
        v.push_back(defaultFillLight());
        return v;
    }();
    return lights;
}

// --- Froxel volumetric fog (Phase 5a; Frostbite 2015 / UE4.16) ----------------
// Per-scene participating-media parameters, filled from the lighting preset
// (SceneRegistry) and consumed by the DeferredCore froxel passes.  Plain POD
// scene data; the CLI/GUI toggles gate on `enabled` without rebuilding.
// References: Hillaire, "Physically Based Sky, Atmosphere and Cloud Rendering
// in Frostbite" (SIGGRAPH 2015); Wronski, "Volumetric Fog" (SIGGRAPH 2014);
// UE4 Volumetric Fog documentation.
struct VolFogParams {
    bool enabled = false;     // presets opt in; hosts AND this with the CLI/GUI toggle
    float density = 0.01f;    // base extinction sigma_t at baseHeight (1/m)
    float heightFalloff = 0.15f; // exponential density decay per metre above baseHeight
    float baseHeight = 0.f;   // world Y of the densest fog (m)
    float anisotropy = 0.6f;  // Schlick/Henyey-Greenstein g (forward scattering)
    Vec3 albedo{0.9f, 0.9f, 0.9f}; // scattering albedo (fog colour)
    float noiseStrength = 0.5f; // 0..1 static 3D noise modulation of density
    float noiseScale = 0.08f;   // noise feature scale (1/m)
    float maxDistance = 200.f;  // froxel far range (m of view depth); matches
                                // kShadowMaxDistance so fog god rays stay
                                // inside the CSM coverage
    float ambient = 0.4f;     // isotropic ambient light scale (x IBL intensity)
};

// --- Reflection probes (UE4 reflection-capture style, Phase 4c-2) ------------
// A baked local specular/diffuse capture: the scene is rendered into a small
// cubemap from `position` once (offline --bake-probes command), prefiltered
// like the global IBL maps, and at runtime pixels inside the AABB sample it
// with parallax-corrected box projection instead of the global environment
// (fallback chain: SSR hit -> local probe -> global env).  Placement is scene
// data: filled from the scene registry (SceneRegistry::reflectionProbesForScene).
// Without a matching .probes bake file the probe list is inert.
constexpr uint32_t kMaxReflectionProbes = 8;

struct ReflectionProbe {
    Vec3 position{0.f, 0.f, 0.f}; // capture origin (parallax pivot)
    Vec3 boxMin{0.f, 0.f, 0.f};   // influence box (world AABB)
    Vec3 boxMax{0.f, 0.f, 0.f};
};

class Scene {
public:
    // Movable but not copyable (owns GPU handles; the GUI installs async-loaded
    // scenes via move).  All special members are out-of-line: TextureStreamer
    // is incomplete here and unique_ptr member cleanup must not be
    // instantiated in other TUs.
    Scene();
    ~Scene();
    Scene(Scene&&) noexcept;
    Scene& operator=(Scene&&) noexcept;
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;

    std::vector<Mesh> meshes;
    std::vector<Texture> textures;
    std::vector<Material> materials;
    std::vector<MeshInstance> instances;
    std::vector<Light> lights;
    // Reflection probe placements (see above); populated by the hosts from the
    // scene registry after load.  Capped at kMaxReflectionProbes on use.
    std::vector<ReflectionProbe> probes;

    // Scene-wide merged vertex/index buffers.  The viewer binds these once per
    // pass and draws with per-mesh firstIndex/vertexOffset, which avoids the
    // (very expensive) per-draw vkCmdBindVertexBuffers/IndexBuffer calls.
    // Compare mode keeps using the per-mesh buffers above.
    VkBuffer mergedVertexBuffer = VK_NULL_HANDLE;
    VmaAllocation mergedVertexMemory = VK_NULL_HANDLE;
    VkBuffer mergedIndexBuffer = VK_NULL_HANDLE;
    VmaAllocation mergedIndexMemory = VK_NULL_HANDLE;

    // --- animation / skinning (see the struct comments above) -----------------
    std::vector<SceneNode> nodes;         // node tree (local TRS), kept only for animated scenes
    std::vector<uint32_t> nodeTopoOrder;  // parents before children
    std::vector<AnimSampler> animSamplers;
    std::vector<AnimChannel> animChannels; // the first glTF animation, flattened
    float animDuration = 0.f;              // seconds; looping wraps via fmod
    std::vector<Skin> skins;
    // Skinned meshes keep their own (non-merged) buffers: their vertex format
    // differs and they are few, so the merged-buffer optimization is skipped.
    std::vector<Mesh> skinnedMeshes;
    std::vector<DynamicBoxDriver> dynamicDrivers;

    // Joint palette SSBOs (one per in-flight slot, see kSkinPaletteSlots).
    // Layout per buffer: [current-frame joints][previous-frame joints], the
    // previous block starting at skinPaletteJointCount mat4s.  Persistently
    // mapped; advanceToFrame() rewrites the current frame's slot.
    VkBuffer skinPaletteBuffer[kSkinPaletteSlots] = {};
    VmaAllocation skinPaletteMemory[kSkinPaletteSlots] = {};
    void* skinPaletteMapped[kSkinPaletteSlots] = {};
    uint32_t skinPaletteJointCount = 0;

    bool hasDynamicContent() const { return !animChannels.empty() || !dynamicDrivers.empty(); }
    bool hasSkinnedMeshes() const { return !skinnedMeshes.empty(); }
    VkBuffer skinPalette(uint32_t slot) const { return skinPaletteBuffer[slot % kSkinPaletteSlots]; }

    // Advances animations / dynamic drivers to the given frame (time =
    // frameIndex * kAnimDt).  Rewrites model/prevModel/normalModel of driven
    // instances and the current slot's joint palettes.  No-op for fully
    // static scenes.  Must be called once per frame before recording.
    void advanceToFrame(uint32_t frameIndex);

    // Per-frame LOD selection: picks lodLevel/lodCulled per instance from the
    // world-AABB bounding sphere projected at a FIXED 1080p reference height
    // (so GT and upscaled paths of the same frame always agree).  Called once
    // per frame by the hosts right after advanceToFrame(); with enabled=false
    // everything stays at LOD0 / never culled.
    void updateLodSelection(const Vec3& cameraPos, float fovY, bool enabled);

    // Fills per-instance lodDraws from mesh LOD chains (or the authored
    // MSFT_lod chain when present).  Call once after all meshes and generated
    // LOD levels exist, before buildMergedBuffers().
    void buildLodDraws();

    // Appends one generated LOD level (index buffer over the mesh's existing
    // vertices) to the mesh's chain and the merged index accumulation.
    // Used by LodBuilder before buildMergedBuffers().
    void appendMeshLod(uint32_t meshIndex, const std::vector<uint32_t>& indices);

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
    // Skinned meshes stay out of the merged scene buffers (different vertex
    // format); the returned Mesh addresses its own buffers from offset 0.
    bool uploadSkinnedMesh(const VulkanContext& ctx, const std::vector<SkinnedVertex>& vertices,
                           const std::vector<uint32_t>& indices, Mesh& out,
                           VkCommandPool pool = VK_NULL_HANDLE);
    // Allocates + persistently maps the per-slot joint palette SSBOs and fills
    // them with the bind pose.  Call once after skins/nodes are loaded.
    bool createSkinPalettes(const VulkanContext& ctx);
    // Recomputes conservative world AABBs for animation-driven and skinned
    // instances by sampling the whole animation (static instances already got
    // tight bounds from finalizeInstances).  Called by the loaders.
    void computeAnimatedBounds();
    // Uploads RGBA8 texels and generates the full mip chain (per-level blits).
    // srgb=true (default): base color / emissive; false: normal/MR/AO data.
    bool uploadTexture(const VulkanContext& ctx, uint32_t width, uint32_t height,
                       const uint8_t* rgba8, Texture& out, bool srgb = true,
                       VkCommandPool pool = VK_NULL_HANDLE);
    // Uploads a pre-baked BC7 KTX2 (no blits: block data copies directly).
    // With streamingEnabled and a long enough mip chain, only the coarse tail
    // (mip >= kStreamResidentBaseMip) is uploaded now and a stream job for the
    // fine levels is queued for updateTextureStreaming(); otherwise the full
    // chain uploads here (bench/compare determinism).
    bool uploadTextureCompressed(const VulkanContext& ctx, const Ktx2Image& img, bool srgb,
                                 Texture& out, const char* sourcePath,
                                 VkCommandPool pool = VK_NULL_HANDLE);

    // Per-frame streaming tick (interactive hosts only): reprioritizes pending
    // fine-mip uploads by distance to the camera and hands the background
    // thread one frame's byte budget.  Starts the worker lazily on first call
    // (after the scene has been moved into its final home).  No-op when
    // streaming is disabled or nothing is pending.
    void updateTextureStreaming(const VulkanContext& ctx, const Vec3& cameraPos);

    // Set by the host before loadGltf: interactive viewer/GUI = true,
    // bench/compare/screenshot = false (see the Phase 7b comment block above).
    bool streamingEnabled = false;

    // Uploads the accumulated merged vertex/index data.  Call once after all
    // meshes are uploaded; frees the CPU-side accumulation.
    bool buildMergedBuffers(const VulkanContext& ctx, VkCommandPool pool = VK_NULL_HANDLE);

private:
    // CPU-side accumulation for the merged buffers (cleared by buildMergedBuffers).
    std::vector<Vertex> mergedVerts_;
    std::vector<uint32_t> mergedIndices_;

    // Queued by uploadTextureCompressed when streaming is enabled; consumed by
    // the background worker started in updateTextureStreaming.
    struct TextureStreamJob {
        uint32_t textureIndex = 0;   // into textures
        std::string path;            // source .ktx2 (levels re-read on demand)
        uint32_t width = 0, height = 0;
        uint32_t nextLevel = 0;      // next fine level to upload (descends to 0)
        std::vector<Ktx2Image::Level> levels;
        uint64_t bytesRemaining = 0;
    };
    std::vector<TextureStreamJob> pendingStreamTextures_;
    // Pimpl keeps Scene movable (mutex/thread members are not).  Defined in
    // TextureStreamer.cpp.
    struct TextureStreamer;
    std::unique_ptr<TextureStreamer> streamer_;
    // Stops + joins the streaming worker (idempotent).  Called by destroy() and
    // the destructor; must complete before any texture image is destroyed.
    void stopTextureStreaming();
};

} // namespace sr
