#version 450
// Deferred GBuffer fragment shader for the native ground-truth path.
// Identical to gbuffer.frag; Phase 6b added the motion output (RT4) so the GT
// path runs the same motion-blur post pass as the LR path (per-object motion
// of dynamic objects must blur the GT identically, or the compare metrics
// would fault the upscalers for a post effect only one side has).

layout(set = 0, binding = 1) uniform MaterialUBO {
    vec4 baseColor;
    vec4 factors;   // x = metallic, y = roughness, z = occlusionStrength, w = alphaCutoff
    vec4 emissive;
    vec4 tex0;      // baseColor, normal, mr, ao
    vec4 tex1;      // x = emissive
} material;

layout(set = 1, binding = 0) uniform sampler2D uTextures[1024];

// Vertex-stage SceneUBO, also visible to the fragment stage (see gbuffer.frag).
layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 viewProj;
    mat4 viewProjNoJitter;
    mat4 prevViewProj;
    vec4 cameraPos;
    vec4 ambient;
    vec4 renderSizeJitter;
    vec4 clipPlane;    // planar mirror plane (xyz normal toward the camera, w offset)
    mat4 reflViewProj; // mirrored-view view-projection (main view, reflParams.x = 1)
    vec4 reflParams;   // x = reflection image available, y = clip behind clipPlane,
                       // zw = mirrored proj m[10]/m[14] (viewZ = w / (depth + z))
} scene;

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vUV;
layout(location = 3) in vec4 vTangent;
layout(location = 4) in vec4 vCurrentClip;
layout(location = 5) in vec4 vPreviousClip;

layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outMaterial;
layout(location = 3) out vec3 outEmissive; // B10G11R11_UFLOAT: no alpha channel
layout(location = 4) out vec2 outMotion;   // RG16F, previousUV - currentUV, no jitter

int texIndex(float f) { return int(floor(f + 0.5)); }

void main() {
    vec4 base = material.baseColor;
    const int baseTex = texIndex(material.tex0.x);
    if (baseTex >= 0) base *= texture(uTextures[baseTex], vUV);

    if (material.factors.w > 0.0 && base.a < material.factors.w) discard;
    // Planar-reflection view (PlanarReflection.h): the mirrored camera sits
    // behind the pane, so everything on the far side of the mirror plane
    // (the shop interior) must not occlude the reflected street.
    if (scene.reflParams.y > 0.5 &&
        dot(scene.clipPlane.xyz, vWorldPos) + scene.clipPlane.w < -0.02) discard;

    float metallic = material.factors.x;
    float roughness = material.factors.y;
    const int mrTex = texIndex(material.tex0.z);
    if (mrTex >= 0) {
        const vec4 mr = texture(uTextures[mrTex], vUV);
        roughness *= mr.g;
        metallic *= mr.b;
    }

    float ao = 1.0;
    const int aoTex = texIndex(material.tex0.w);
    if (aoTex >= 0) {
        ao = mix(1.0, texture(uTextures[aoTex], vUV).r, material.factors.z);
    }

    vec3 N = normalize(vNormal);
    // Cull mode is NONE: flip the geometric normal toward the viewer before
    // normal mapping so backfaces store viewer-facing normals (see
    // gbuffer.frag for why this must happen here, not in lighting.frag).
    if (dot(N, scene.cameraPos.xyz - vWorldPos) < 0.0) N = -N;
    const int normalTex = texIndex(material.tex0.y);
    if (normalTex >= 0) {
        vec3 T = normalize(vTangent.xyz - N * dot(N, vTangent.xyz));
        vec3 B = cross(N, T) * vTangent.w;
        vec3 n = texture(uTextures[normalTex], vUV).xyz * 2.0 - 1.0;
        N = normalize(mat3(T, B, N) * n);
    }

    vec3 emissive = material.emissive.rgb;
    const int emTex = texIndex(material.tex1.x);
    if (emTex >= 0) emissive *= texture(uTextures[emTex], vUV).rgb;

    outAlbedo = vec4(base.rgb, base.a);
    outNormal = vec4(N * 0.5 + 0.5, 1.0); // A2B10G10R10 is unsigned: remap [-1,1] to [0,1]
    outMaterial = vec4(clamp(metallic, 0.0, 1.0), clamp(roughness, 0.0, 1.0),
                       clamp(ao, 0.0, 1.0), 1.0);
    outEmissive = emissive;
    // Same canonical motion as gbuffer.frag: current -> previous in
    // framebuffer-oriented normalized UV, unclamped, no jitter.
    vec2 currentUV = vCurrentClip.xy / vCurrentClip.w * 0.5 + 0.5;
    vec2 previousUV = vPreviousClip.xy / vPreviousClip.w * 0.5 + 0.5;
    outMotion = previousUV - currentUV;
}
