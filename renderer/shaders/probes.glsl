// Baked local reflection probes (UE4 reflection-capture style, Phase 4c-2).
// Shared by lighting.frag (IBL split-sum terms), ssr_opaque.comp (the
// recomputed specIbl of the composite delta — both MUST use the same fallback
// chain or the SSR replacement would not cancel lighting's term exactly) and
// transparent*.frag (the glass inline-SSR miss fallback).
//
// Fallback chain: SSR hit -> local probe -> global env.  A pixel inside one or
// two probe boxes samples the baked cube arrays with parallax-corrected box
// projection; the two best probes blend by their box-edge distance weights;
// pixels outside every box keep the global environment maps.  With probe
// count 0 (no bake file) every helper degenerates to the global-env path.
//
// The includer defines the binding numbers (they differ between the lighting
// and the SSR descriptor set layouts) and, when its probe descriptors do not
// live in set 0, SR_PROBE_SET (the transparency pass binds them in set 2):
//   #define SR_PROBE_UBO_BINDING  / SR_PROBE_SPEC_BINDING / SR_PROBE_DIFF_BINDING
//   [#define SR_PROBE_SET 2]
#ifndef SR_PROBES_GLSL
#define SR_PROBES_GLSL

#ifndef SR_PROBE_SET
#define SR_PROBE_SET 0
#endif

const int kMaxProbes = 8; // must match kMaxReflectionProbes (Scene.h)

layout(set = SR_PROBE_SET, binding = SR_PROBE_UBO_BINDING) uniform ProbeUBO {
    vec4 info;              // x = active probe count, y = prefilter max lod,
                            // z = box-edge blend distance (m), w unused
    vec4 boxMin[kMaxProbes];
    vec4 boxMax[kMaxProbes];
    vec4 position[kMaxProbes]; // xyz = capture position (parallax pivot)
} probes;
layout(set = SR_PROBE_SET, binding = SR_PROBE_SPEC_BINDING) uniform samplerCubeArray probePrefilter;
layout(set = SR_PROBE_SET, binding = SR_PROBE_DIFF_BINDING) uniform samplerCubeArray probeIrradiance;

// Parallax-corrected sampling direction: intersect the reflection ray with
// the probe's box and re-aim from the capture position to the hit point.
vec3 probeBoxProject(int i, vec3 worldPos, vec3 R) {
    const vec3 rsafe = sign(R) * max(abs(R), vec3(1e-6));
    const vec3 inv = 1.0 / rsafe;
    const vec3 t0 = (probes.boxMin[i].xyz - worldPos) * inv;
    const vec3 t1 = (probes.boxMax[i].xyz - worldPos) * inv;
    const vec3 tmax = max(t0, t1);
    const float t = max(min(tmax.x, min(tmax.y, tmax.z)), 0.0);
    return normalize(worldPos + R * t - probes.position[i].xyz);
}

// Per-probe influence: interior distance to the closest box face, ramped over
// info.z metres so entering/leaving a box fades instead of popping.
float probeInfluence(int i, vec3 worldPos) {
    const vec3 d = min(worldPos - probes.boxMin[i].xyz, probes.boxMax[i].xyz - worldPos);
    return clamp(min(d.x, min(d.y, d.z)) / probes.info.z, 0.0, 1.0);
}

// Picks the best two containing probes by box-edge distance.
// Returns the total probe weight in [0,1] (0 = outside every box: caller uses
// the global env); i0/i1 are the probe indices (-1 = none) and w0/w1 their
// normalized blend (w0 + w1 = 1 when any probe contains the point).
float probeSelect(vec3 worldPos, out int i0, out int i1, out float w0, out float w1) {
    i0 = -1;
    i1 = -1;
    w0 = 0.0;
    w1 = 0.0;
    float f0 = 0.0, f1 = 0.0;
    const int n = int(probes.info.x + 0.5);
    for (int i = 0; i < kMaxProbes; ++i) {
        if (i >= n) break;
        const float f = probeInfluence(i, worldPos);
        if (f > f0) {
            f1 = f0; i1 = i0; f0 = f; i0 = i;
        } else if (f > f1) {
            f1 = f; i1 = i;
        }
    }
    const float sum = f0 + f1;
    if (sum > 1e-6) {
        w0 = f0 / sum;
        w1 = f1 / sum;
    }
    return clamp(sum, 0.0, 1.0);
}

// Blended parallax-corrected prefiltered specular of probes i0/i1.
vec3 probeSpecular(vec3 worldPos, vec3 R, float roughness, int i0, int i1, float w0, float w1) {
    const float lod = roughness * probes.info.y;
    vec3 c = vec3(0.0);
    if (i0 >= 0)
        c += w0 * textureLod(probePrefilter, vec4(probeBoxProject(i0, worldPos, R), float(i0)), lod).rgb;
    if (i1 >= 0)
        c += w1 * textureLod(probePrefilter, vec4(probeBoxProject(i1, worldPos, R), float(i1)), lod).rgb;
    return c;
}

// Blended diffuse irradiance (no parallax correction, same as global IBL).
vec3 probeDiffuse(vec3 N, int i0, int i1, float w0, float w1) {
    vec3 c = vec3(0.0);
    if (i0 >= 0) c += w0 * texture(probeIrradiance, vec4(N, float(i0))).rgb;
    if (i1 >= 0) c += w1 * texture(probeIrradiance, vec4(N, float(i1))).rgb;
    return c;
}

#endif
