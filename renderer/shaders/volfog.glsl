#ifndef SR_VOLFOG_GLSL
#define SR_VOLFOG_GLSL
// ============================================================================
// Shared helpers for the froxel volumetric fog passes (volfog_*.comp).
//
// References:
//  - Hillaire, "Physically Based Sky, Atmosphere and Cloud Rendering in
//    Frostbite" (SIGGRAPH 2015): unified participating media in a
//    camera-frustum voxel grid (froxels), per-froxel density injection +
//    light accumulation + analytic ray integration.
//  - Epic Games, UE4.16 Volumetric Fog documentation / Wronski, "Volumetric
//    Fog" (SIGGRAPH 2014): exponential depth slicing, per-column front-to-back
//    integration, temporal reprojection of the lit volume.
//
// Conventions: the froxel grid is aligned with the camera frustum, XY =
// screen / kFroxelTileSize (host constant), Z = exponential slices over
// [near, fogFar] — the same slicing family as the cluster grid (DOOM Eternal
// 24 slices), so froxel <-> cluster mapping stays a pure depth formula.
// ============================================================================

// Slice coordinate (froxel index space, centres at i + 0.5) -> view depth.
float froxelViewZ(float sliceCoord, float gridZ, float nearZ, float farZ) {
    return nearZ * pow(farZ / nearZ, sliceCoord / gridZ);
}

// View depth -> normalized volume W for texture() sampling: texel z-centres
// sit at (i + 0.5) / gridZ in UV space, and slice coordinate s maps to
// w = s / gridZ, so the log() terms cancel the grid resolution entirely.
float froxelW(float viewZ, float nearZ, float farZ) {
    return clamp(log(max(viewZ, nearZ) / nearZ) / log(farZ / nearZ), 0.0, 1.0);
}

// World-space centre of a froxel.  projParams: x = proj[0][0],
// y = proj[1][1] (carries the Vulkan y-flip sign, which reconstructs the
// correct view-space Y), z = near, w = fog far range.  Same unprojection
// math as cluster_assign.comp's per-cluster AABB corners.
vec3 froxelWorldPos(uvec3 c, vec3 gridDim, vec4 projParams, mat4 invView) {
    const vec2 ndc = (vec2(c.xy) + 0.5) / gridDim.xy * 2.0 - 1.0;
    const float viewZ = froxelViewZ(float(c.z) + 0.5, gridDim.z, projParams.z, projParams.w);
    const vec3 viewPos = vec3(ndc.x * viewZ / projParams.x, ndc.y * viewZ / projParams.y, -viewZ);
    return (invView * vec4(viewPos, 1.0)).xyz;
}

// Schlick approximation of Henyey-Greenstein (k = g), normalized to 4 pi.
// cosTheta = dot(view direction camera->froxel, direction froxel->light), so
// g > 0 peaks when looking towards the light (forward scattering).
// The 1/(4pi) keeps the scale consistent with the surface lights, whose
// intensities are PI-scaled on the CPU (DeferredCore::packLightGpu): an
// isotropic unit-albedo medium then scatters intensity/4 per unit optical
// depth, against a Lambertian surface's ~intensity*NdL.
float volFogPhase(float g, float cosTheta) {
    const float k = clamp(g, -0.95, 0.95);
    const float d = 1.0 - k * cosTheta;
    return (1.0 - k * k) / (12.566371 * d * d);
}

// Deterministic 3D value noise (lowbias32 integer hash lattice, trilinear
// smoothstep interpolation).  Integer hashing is bit-reproducible across
// vendors — unlike fract(sin()) hashes — which the GT/upscaler determinism
// contract relies on.  Static noise: no wind animation this phase; the
// inject pass offsets the domain by the frame index so the temporal filter
// has a per-frame realisation to smooth.
uint volFogHash(uvec3 p) {
    uint x = p.x * 0x7feb352dU ^ p.y * 0x846ca68bU ^ p.z * 0x2c1b3c6dU;
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

float volFogHash01(uvec3 p) { return float(volFogHash(p)) * (1.0 / 4294967296.0); }

float volFogNoise3(vec3 p) {
    const ivec3 ip = ivec3(floor(p));
    const vec3 f0 = fract(p);
    const vec3 f = f0 * f0 * (3.0 - 2.0 * f0);
    const uvec3 i = uvec3(ip & 0xFFFF);
    const float n000 = volFogHash01(i);
    const float n100 = volFogHash01(i + uvec3(1, 0, 0));
    const float n010 = volFogHash01(i + uvec3(0, 1, 0));
    const float n110 = volFogHash01(i + uvec3(1, 1, 0));
    const float n001 = volFogHash01(i + uvec3(0, 0, 1));
    const float n101 = volFogHash01(i + uvec3(1, 0, 1));
    const float n011 = volFogHash01(i + uvec3(0, 1, 1));
    const float n111 = volFogHash01(i + uvec3(1, 1, 1));
    return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
               mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z);
}

#endif // SR_VOLFOG_GLSL
