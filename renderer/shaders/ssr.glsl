// Screen-space reflection: hierarchical-Z (Hi-Z) march in clip space against
// the opaque depth pyramid, matching the tracing model used by UE's
// ScreenSpaceReflections pass (SSRTReflections.usf) rather than a world-space
// quadratic march.
//
// Why clip-space: 1/w is linear in screen along a 3D ray, so view-Z and UV
// stay perspective-correct.  The previous world-space t² march compared
// Euclidean camera distances against a nonlinear depth buffer and produced
// stretched / false hits on the Bistro shop windows.
//
// Why Hi-Z: the march advances one pyramid cell at the current mip while the
// ray stays in front of the cell's farthest surface (ascending to coarser
// mips as it accelerates), and descends one mip without advancing when the
// hull is crossed — a finer cell's max can only be smaller, so the descent
// always reaches mip 0, where binary refinement pins the exact front-to-back
// transition.  Long rays across empty screen regions cost a handful of
// coarse steps instead of 96 full-resolution samples.
//
// UE composite (DiffuseIndirectComposite) applies EnvBRDF on the hit colour
// and a screen-edge vignette; the caller does the EnvBRDF multiply.  We return
// rgb + confidence (0 = miss).
//
// Ref: McGuire & Mara, "Efficient GPU Screen-Space Ray Tracing", JCGT 2014
//      Stachowiak, Stochastic SSR, SIGGRAPH 2015
//      https://eukky.github.io/GameDev/Render/StochasticScreenSpaceReflection.html
#ifndef SR_SSR_GLSL
#define SR_SSR_GLSL

// Optional trace telemetry (compile with -DSR_SSR_DEBUG=<mode>; the release
// build never defines it, so the statements below vanish).  Filled by the
// last traceSsr call of the invocation; transparent.frag maps it to colours.
#ifdef SR_SSR_DEBUG
struct SsrDebug {
    int rejThick;   // crossings rejected by the thickness window
    int rejBack;    // crossings rejected as back-side (-N) hits
    int rejSelf;    // crossings rejected as self-hits
    int rejSky;     // crossings that bisected onto a sky texel
    int iters;      // hierarchy iterations consumed
    int endState;   // 0 none, 1 hit, 2 early-out, 3 left screen, 4 ray end, 5 budget exhausted
    float lastOver; // view-Z overshoot of the last thickness-rejected crossing
    float lastThick;// window that rejected it
    float hitDist;  // world distance of the accepted hit
    vec2 hitUv;
    int exitTests;  // exit bisections evaluated
    float exitOver; // behind-side overshoot of the last exit test
    float exitUnder;// in-front-side undershoot of the last exit test
    int exitSky;    // exit tests skipped because a side was sky
    int runs;       // behind-runs started
    float rimZ;     // rim depth of the last run
    float maxBulge; // largest front bulge seen in any run
    float minOver;  // smallest overshoot seen in any run
    float minOverBulge; // 2*bulge at the sample with the smallest overshoot
    float overAtMaxBulge; // overshoot at the sample with the largest bulge
    float uvAtMaxBulge;   // screen u of that sample
    float rimAtMaxBulge;  // rim depth of that run
    float sceneAtMaxBulge;// front depth at that sample
};
SsrDebug gSsrDbg;
#define SSR_DBG(stmt) stmt
#else
#define SSR_DBG(stmt)
#endif

float ssrViewZ(mat4 viewProj, mat4 invViewProj, vec2 uv, float depth) {
    vec4 w = invViewProj * vec4(uv * 2.0 - 1.0, depth, 1.0);
    w /= w.w;
    return abs((viewProj * vec4(w.xyz, 1.0)).w);
}

// Liang-Barsky clip of the UV segment to [0,1]².  Returns the clipped end
// parameter in the original 0..1 range (1 = unclipped).
float ssrClipToScreen(vec2 startUv, vec2 endUv) {
    vec2 d = endUv - startUv;
    float t0 = 0.0;
    float t1 = 1.0;
    // Four clip planes: left, right, bottom, top.
    vec4 pp = vec4(-d.x, d.x, -d.y, d.y);
    vec4 qq = vec4(startUv.x, 1.0 - startUv.x, startUv.y, 1.0 - startUv.y);
    for (int k = 0; k < 4; ++k) {
        float pk = pp[k];
        float qk = qq[k];
        if (abs(pk) < 1e-8) {
            if (qk < 0.0) return 0.0;
        } else {
            float t = qk / pk;
            if (pk < 0.0) t0 = max(t0, t);
            else t1 = min(t1, t);
        }
    }
    return (t0 < t1) ? clamp(t1, 0.0, 1.0) : 0.0;
}

// hizTex: max-reduced depth pyramid (mip 0 = opaque depth copy, R32F);
// hizMipCount = textureQueryLevels(hizTex) at the call site.
// colorTex: box-filtered mip chain of the lit opaque HDR color (mip 0 =
// sharp copy); colorMipCount = textureQueryLevels(colorTex).  The hit colour
// is read at a roughness-driven LOD so one ray approximates the widened GGX
// lobe (rough reflections average a larger screen footprint).
// escalateEvery: coarsen one mip per this many in-front advances (1 = classic
// Hi-Z, one per step); callers pass 1.  maxMipCap caps the coarsest level the
// march may reach (glass 1: coarser cells tunnel through thin slatted
// reflectors; opaque the full chain for long ground rays).
// cfg packs the per-caller tuning:
//   x = nearField (m): mip coarsening is additionally gated on the ray's
//      WORLD distance travelled — a max-reduced cell mixing a thin NEAR
//      reflector (chair slats, mullions, lamps) with deeper background
//      reports the background depth, so coarse steps tunnel through the
//      reflector and the torn hit field stacks into layered bands in the
//      mirror.  The near field therefore marches at mip 0 where such
//      reflectors live; 0 restores plain advance-count gating (opaque).
//   y/z = thickness base / view-Z slope of the hit-acceptance window.
//   w = grazing widening amount [0..1]: the window is multiplied by
//      mix(1, 1/max(|R·N|, 0.25), w).  The opaque path keeps the historical
//      full widening (w = 1, tolerates coarse landings of long ground rays);
//      glass passes 0 — at grazing the widened window accepts rays passing
//      metres behind a reflector, and because a view-Z miss of w maps to a
//      w/|R·N| slide along the mirror, small objects smeared into long
//      horizontal streaks (the "stretched lamp" artifact).
// extentK: solid-occluder handling (0 = off, legacy behaviour).  The base
//   window alone tunnels through anything thicker than a few centimetres
//   (hanging lamps read as crescents, chairs and awnings tear into stripes
//   of far background / fallback).  With extentK > 0 a crossing whose
//   overshoot exceeds the window starts a BEHIND-RUN instead of being final;
//   how the run resolves depends on which way the ray travels in view Z:
//   - Ray moving TOWARD the camera (view Z decreasing): when it re-emerges
//     in front of the depth surface the exit is bisected.  A CONTINUOUS
//     back-to-front crossing (both sides of the boundary within the window)
//     means the ray passed through the occluder's front surface from inside,
//     i.e. for a convex solid it had to enter through the back face the
//     depth buffer cannot hold — a hit at the exit texel.  A jump to the
//     background means the projection merely left the silhouette: the ray
//     passed behind a thin object and continues.  Exact for convex solids,
//     never extrudes anything.
//   - Any direction, sample by sample: a convex occluder is symmetric about
//     its silhouette, so its unseen back face bulges away from the camera
//     by as much as its visible front face bulges toward it.  The run
//     remembers the depth of the silhouette it entered (the rim: for a
//     convex object the front and back faces meet there) and accepts the
//     first sample whose overshoot past the front fits under twice the
//     front bulge (rimZ - frontZ) at that texel — i.e. the ray is inside
//     the object.  This is local: the ray's projection can run on behind a
//     second surface (the hanging lamp sits under the awning) without the
//     first object's size leaking into the second, because the run restarts
//     whenever the front depth jumps or drops behind the rim.  Rays that
//     pass behind an object by more than its own thickness, and behind
//     wires / slats (no bulge), keep marching.  (Toward-camera rays get the
//     exit test as well; the two agree on convex solids.)
//   - A run still open when the march ends (screen edge, ray end, budget)
//     is scored by a chord model instead — its screen length in world metres
//     (capped at kExtentCap) against its mid-run overshoot — so walls /
//     awnings / ground beyond the screen edge block rays.
// outHitDist: world distance from worldPos to the accepted hit (0 on miss);
//   the glass pass reprojects the reflection through the mirror image of
//   the hit point.
vec4 traceSsrEx(sampler2D colorTex, int colorMipCount, sampler2D hizTex, int hizMipCount,
                mat4 viewProj, mat4 invViewProj, vec3 cameraPos, vec3 worldPos,
                vec3 N, vec3 R, float roughness, vec2 renderSize, int escalateEvery,
                int maxMipCap, vec4 cfg, int maxIters, float extentK, out float outHitDist) {
    outHitDist = 0.0;
    const int kMaxIters = maxIters; // hierarchy steps (each descends or advances);
    const float kExtentCap = 0.35;  // m; largest occluder thickness the open-run chord model assumes
    const int kRefine = 8;
    const float kMaxDist = 100.0;
    const float kBias = 0.08;
    const float kNearW = 0.02;
    const float nearField = cfg.x;
    const float thickBase = cfg.y;
    const float thickSlope = cfg.z;
    const float thickWiden = cfg.w;
    SSR_DBG(gSsrDbg.rejThick = 0; gSsrDbg.rejBack = 0; gSsrDbg.rejSelf = 0; gSsrDbg.rejSky = 0;
            gSsrDbg.iters = 0; gSsrDbg.endState = 2; gSsrDbg.lastOver = 0.0;
            gSsrDbg.lastThick = 0.0; gSsrDbg.hitDist = 0.0; gSsrDbg.hitUv = vec2(0.0);
            gSsrDbg.exitTests = 0; gSsrDbg.exitOver = -1.0; gSsrDbg.exitUnder = -1.0; gSsrDbg.exitSky = 0;
            gSsrDbg.runs = 0; gSsrDbg.rimZ = 0.0; gSsrDbg.maxBulge = 0.0; gSsrDbg.minOver = 1e9; gSsrDbg.minOverBulge = 0.0;)

    vec3 origin = worldPos + N * kBias;
    vec4 startCS = viewProj * vec4(origin, 1.0);
    vec4 endCS = viewProj * vec4(origin + R * kMaxDist, 1.0);

    // Clip to the near plane so 1/w stays finite (ray toward / behind camera).
    // s0/s1 track the ray's world-space distance range through the clips so
    // the march can gate mip coarsening on distance travelled (see below).
    float s0 = 0.0, s1 = kMaxDist;
    if (startCS.w < kNearW && endCS.w < kNearW) return vec4(0.0);
    if (endCS.w < kNearW) {
        float tn = (kNearW - startCS.w) / (endCS.w - startCS.w);
        if (tn <= 0.0) return vec4(0.0);
        endCS = mix(startCS, endCS, clamp(tn, 0.0, 1.0));
        s1 = mix(s0, s1, clamp(tn, 0.0, 1.0));
    }
    if (startCS.w < kNearW) {
        float tn = (kNearW - startCS.w) / (endCS.w - startCS.w);
        startCS = mix(startCS, endCS, clamp(tn, 0.0, 1.0));
        s0 = mix(s0, s1, clamp(tn, 0.0, 1.0));
    }

    vec3 startNdc = startCS.xyz / startCS.w;
    vec3 endNdc = endCS.xyz / endCS.w;
    vec2 startUv = startNdc.xy * 0.5 + 0.5;
    vec2 endUv = endNdc.xy * 0.5 + 0.5;
    if (any(lessThan(startUv, vec2(-0.02))) || any(greaterThan(startUv, vec2(1.02))))
        return vec4(0.0);
    startUv = clamp(startUv, vec2(0.0), vec2(1.0));

    // Bail only when the segment misses the screen entirely.  Head-on panes
    // reflect the ray back toward/behind the camera, so endNdc lands far
    // off-screen and tClip (relative to the full segment) is tiny even though
    // the clipped piece still crosses plenty of on-screen geometry — rejecting
    // on a small tClip kills exactly those reflections.  Truly degenerate
    // segments are caught by the pixel-length check below.
    float tClip = ssrClipToScreen(startUv, endUv);
    if (tClip <= 0.0) return vec4(0.0);
    endUv = mix(startUv, endUv, tClip);
    float invW0 = 1.0 / startCS.w;
    float invW1 = mix(invW0, 1.0 / endCS.w, tClip);
    // World-distance range of the clipped segment: s(t) = mix(sA, sB, t).
    const float sA = s0;
    const float sB = mix(s0, s1, tClip);

    vec2 startPx = startUv * renderSize;
    vec2 endPx = endUv * renderSize;
    vec2 dPx = endPx - startPx;
    float pixelDist = length(dPx);
    // A reflection ray that projects to less than a pixel cannot resolve a
    // hit; miss.
    if (pixelDist < 0.25) return vec4(0.0);

    const int maxMip = min(hizMipCount - 1, maxMipCap);
    const ivec2 size0 = ivec2(renderSize);
    // |R·N|: 1 head-on, ~0 at grazing.  Scales the self-hit radius below and
    // (via thickWiden) optionally widens the thickness window for the opaque
    // caller; the glass LOD term uses it too.
    const float grazing = abs(dot(R, N));
    // Self-hit rejection.  Legacy (opaque) callers keep the historical
    // Euclidean radius around the ray origin: 0.22 m head-on, shrinking with
    // |R·N| so grazing rays may still strike nearby frames and walls.  The
    // solid-occluder (glass) callers use plane clearance instead: a hit is
    // "the reflector itself" only when its surface point lies on the pane
    // plane (frame seam, coplanar decals).  The radius discarded everything
    // hugging the glass — the café lantern's back hangs 0.16 m off the pane,
    // and the half of its mirror image nearest the glass was rejected as a
    // self hit (the crescent-shaped lamp reflection).
    const float selfRadius = extentK > 0.0 ? kBias : kBias + 0.14 * grazing;
    const float selfPlane = extentK > 0.0 ? 0.02 : -1.0e30;

    // t parameterises the clipped segment: uv = mix(startUv, endUv, t) and
    // 1/w = mix(invW0, invW1, t), so every sample stays perspective-correct.
    float t = 0.0;
    float tFront = 0.0; // last sampled position still in front of the hull
    int level = 0;
    int advances = 0; // in-front steps taken; gates mip coarsening
    bool hit = false;
    vec2 hitUv = endUv;
    float hitDist = kMaxDist;
    float hitSide = 0.0;
    // Estimated (back-face) hits: confidence fades with the acceptance margin
    // and the colour comes from one mip coarser, so the reflection of a solid
    // occluder's unseen side reads as a smooth body instead of a ragged patch
    // of its front-face texture.
    float hitSoft = 1.0;
    float hitLodBias = 0.0;
    // Silhouette-extent candidate (see extentK above).
    bool candValid = false;
    vec2 candUv = vec2(0.0);
    float candDist = 0.0;
    float candOver = 0.0;    // overshoot at the run's first crossing
    float candOverLast = 0.0;// overshoot at the run's latest behind sample
    vec2 candUvLast = vec2(0.0);
    float candDistLast = 0.0;
    float candT = 0.0;       // segment parameter where the run started
    float candTBehind = 0.0; // last sample known to be behind the surface
    float candPxWorld = 0.0; // world metres per screen pixel at the candidate depth
    float candRimZ = 0.0;    // front depth at the silhouette the run entered
    float candSceneZLast = 0.0; // front depth at the run's latest sample
    // View Z decreases along the segment when 1/w grows (ray toward the camera).
    const bool towardCamera = invW1 > invW0;
    // Direction of one screen pixel along the march, in UV units.
    const vec2 pxStepUv = (dPx / max(pixelDist, 1e-6)) / renderSize;

    SSR_DBG(gSsrDbg.endState = 5;)
    for (int it = 0; it < kMaxIters && t <= 1.0; ++it) {
        SSR_DBG(gSsrDbg.iters = it + 1; if (t > 1.0) gSsrDbg.endState = 4;)
        vec2 px = startPx + dPx * t;
        vec2 uv = px / renderSize;
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { SSR_DBG(gSsrDbg.endState = 3;) break; }

        ivec2 mipSize = max(size0 >> level, ivec2(1));
        ivec2 ip = clamp(ivec2(px) >> level, ivec2(0), mipSize - 1);
        float cellDepth = texelFetch(hizTex, ip, level).r;

        float invW = mix(invW0, invW1, t);
        float rayZ = abs(1.0 / invW);
        // Sky (cleared depth 1.0) max-reduces to the far plane and dominates
        // any mixed cell; report +inf so sky cells are skipped at full speed.
        // The accepted Hi-Z tradeoff: thin silhouettes against the sky are
        // only resolved once the cell no longer contains a sky pixel.
        float cellZ = cellDepth < 0.9999 ? ssrViewZ(viewProj, invViewProj, uv, cellDepth)
                                         : 3.0e38;

        if (rayZ <= cellZ) {
            // Ray moving away from the camera: the bulge test below already
            // had every sample of the run; re-emerging means it passed
            // behind the occluder.
            if (candValid && !towardCamera) candValid = false;
            if (candValid) {
                // Ray moving toward the camera: bisect the exit between the
                // last behind sample and this in-front sample on mip 0 and
                // classify it (see extentK above).
                float lo = candTBehind; // behind side
                float hi = t;           // in-front side
                vec2 loUv = mix(startUv, endUv, lo);
                vec2 hiUv2 = uv;
                for (int r = 0; r < kRefine; ++r) {
                    const float mid = 0.5 * (lo + hi);
                    const vec2 mUv = mix(startUv, endUv, mid);
                    const float mDepth = texelFetch(hizTex, ivec2(clamp(mUv * renderSize, vec2(0.0),
                                                                          renderSize - 1.0)), 0).r;
                    const float mRayZ = abs(1.0 / mix(invW0, invW1, mid));
                    const bool behind = mDepth < 0.9999 &&
                                        mRayZ > ssrViewZ(viewProj, invViewProj, mUv, mDepth);
                    if (behind) { lo = mid; loUv = mUv; } else { hi = mid; hiUv2 = mUv; }
                }
                const float loDepth = texelFetch(hizTex, ivec2(clamp(loUv * renderSize, vec2(0.0),
                                                                     renderSize - 1.0)), 0).r;
                const float hiDepth2 = texelFetch(hizTex, ivec2(clamp(hiUv2 * renderSize, vec2(0.0),
                                                                      renderSize - 1.0)), 0).r;
                bool exitHit = false;
                SSR_DBG(++gSsrDbg.exitTests; if (!(loDepth < 0.9999 && hiDepth2 < 0.9999)) ++gSsrDbg.exitSky;)
                if (loDepth < 0.9999 && hiDepth2 < 0.9999) {
                    const float loRayZ = abs(1.0 / mix(invW0, invW1, lo));
                    const float loSceneZ = ssrViewZ(viewProj, invViewProj, loUv, loDepth);
                    const float hiRayZ = abs(1.0 / mix(invW0, invW1, hi));
                    const float hiSceneZ = ssrViewZ(viewProj, invViewProj, hiUv2, hiDepth2);
                    // Continuous crossing: the surface sits within the window
                    // on BOTH sides of the texel boundary (a silhouette exit
                    // leaves the in-front side metres in front of the
                    // background).  Twice the entry window: curved surfaces
                    // change depth faster per texel near their rims.
                    const float win = 2.0 * (thickBase + thickSlope * loRayZ);
                    SSR_DBG(gSsrDbg.exitOver = loRayZ - loSceneZ; gSsrDbg.exitUnder = hiSceneZ - hiRayZ;)
                    if (loRayZ - loSceneZ <= win && hiSceneZ - hiRayZ <= win) {
                        vec4 hw = invViewProj * vec4(loUv * 2.0 - 1.0, loDepth, 1.0);
                        hw /= hw.w;
                        const float d = length(hw.xyz - worldPos);
                        const float side = dot(hw.xyz - worldPos, N);
                        if (side >= -0.02 && d >= selfRadius && side >= selfPlane) {
                            hit = true;
                            hitUv = loUv;
                            hitDist = d;
                            exitHit = true;
                        }
                    }
                }
                if (exitHit) {
                    SSR_DBG(gSsrDbg.endState = 1; gSsrDbg.hitDist = hitDist; gSsrDbg.hitUv = hitUv;)
                    break;
                }
                candValid = false;
            }
            // In front of the farthest surface of this cell: safe to skip it,
            // then coarsen so long empty runs cost one iteration per cell.
            // Coarsening is additionally gated on world distance travelled:
            // a max-reduced cell mixing a thin NEAR reflector with deeper
            // background reports the background depth, so coarse steps tunnel
            // through slats / mullions / lamps and the hit field tears into
            // stacked bands.  The near field therefore marches at mip 0
            // (nearField = 0 restores plain advance-count gating).
            tFront = t;
            t += float(1 << level) / pixelDist;
            ++advances;
            if (advances % escalateEvery == 0 && mix(sA, sB, t) >= nearField)
                level = min(level + 1, maxMip);
            continue;
        }
        if (level > 0) {
            // Crossed the coarse hull: re-test the same position one mip
            // finer (a finer cell's max can only be smaller, so the descent
            // always terminates at mip 0).
            --level;
            continue;
        }

        // Mip-0 crossing between tFront and t.  When the crossing was
        // detected right after a COARSE advance, the span [tFront, t] covers
        // up to 2^level px of non-monotonic depth (a thin reflector in front
        // of a wall): binary search assumes monotonicity, converges onto one
        // arbitrary texel of the span, and collapses many neighbouring rays
        // onto that same texel — round reflectors (the hanging lamp) read as
        // axis-aligned rectangles.  Scan the span at 1px steps for the FIRST
        // front-to-back transition and refine only that 1px interval.
        float lo = tFront;
        float hi = t;
        vec2 hiUv = uv;
        const float spanPx = (hi - lo) * pixelDist;
        if (spanPx > 1.5) {
            const int kScan = 16; // px; wider spans fall back to plain bisect
            const float stepT = 1.0 / pixelDist;
            float tScan = lo;
            for (int s = 0; s < kScan; ++s) {
                const float tPrev = tScan;
                tScan = min(tScan + stepT, hi);
                const vec2 sUv = mix(startUv, endUv, tScan);
                const float sDepth = texelFetch(hizTex, ivec2(clamp(sUv * renderSize, vec2(0.0),
                                                                      renderSize - 1.0)), 0).r;
                if (sDepth < 0.9999) {
                    const float sRayZ = abs(1.0 / mix(invW0, invW1, tScan));
                    const float sSceneZ = ssrViewZ(viewProj, invViewProj, sUv, sDepth);
                    if (sRayZ > sSceneZ) {
                        lo = tPrev;
                        hi = tScan;
                        hiUv = sUv;
                        break;
                    }
                }
                if (tScan >= hi) break;
            }
        }
        for (int r = 0; r < kRefine; ++r) {
            float mid = 0.5 * (lo + hi);
            vec2 mUv = mix(startUv, endUv, mid);
            float mRayZ = abs(1.0 / mix(invW0, invW1, mid));
            float mDepth = texelFetch(hizTex, ivec2(clamp(mUv * renderSize, vec2(0.0),
                                                          renderSize - 1.0)), 0).r;
            if (mDepth >= 0.9999) {
                lo = mid;
                continue;
            }
            float mSceneZ = ssrViewZ(viewProj, invViewProj, mUv, mDepth);
            if (mRayZ > mSceneZ) {
                hi = mid;
                hiUv = mUv;
            } else {
                lo = mid;
            }
        }
        float hiDepth = texelFetch(hizTex, ivec2(clamp(hiUv * renderSize, vec2(0.0),
                                                        renderSize - 1.0)), 0).r;
        bool reject = hiDepth >= 0.9999;
        SSR_DBG(if (reject) ++gSsrDbg.rejSky;)
        float dist = kMaxDist;
        float hiOver = 0.0;
        float hiSceneZ = 0.0;
        bool thickReject = false;
        if (!reject) {
            float hiRayZ = abs(1.0 / mix(invW0, invW1, hi));
            hiSceneZ = ssrViewZ(viewProj, invViewProj, hiUv, hiDepth);
            hiOver = hiRayZ - hiSceneZ;
            // Hit-acceptance window (view-Z overshoot past the front surface).
            // The opaque path keeps the historical grazing widening (tolerates
            // coarse landings of long ground rays); glass passes thickWiden=0:
            // a view-Z miss of w maps to a w/|R·N| slide along the mirror, so
            // the widened window painted small reflectors into metre-long
            // horizontal streaks (see the cfg comment above).
            const float thickness = (thickBase + thickSlope * hiRayZ) *
                                    mix(1.0, 1.0 / max(grazing, 0.25), thickWiden);
            if (hiOver < 0.0 || hiOver > thickness) {
                reject = true;
                thickReject = extentK > 0.0 && hiOver > 0.0;
                SSR_DBG(++gSsrDbg.rejThick; gSsrDbg.lastOver = hiOver; gSsrDbg.lastThick = thickness;)
            }
        }
        if (!reject || thickReject) {
            vec4 hw = invViewProj * vec4(hiUv * 2.0 - 1.0, hiDepth, 1.0);
            hw /= hw.w;
            dist = length(hw.xyz - worldPos);
            hitSide = dot(hw.xyz - worldPos, N);
            if (thickReject) {
                // Candidate bookkeeping (front side, off the reflector itself).
                if (hitSide >= -0.02 && dist >= selfRadius && hitSide >= selfPlane) {
                    // A new object starts where the front depth jumps, or
                    // where it falls clearly behind the rim this run entered
                    // at (past the far rim of a convex occluder).  The rim
                    // tolerance absorbs intra-object depth noise: the café
                    // lanterns are alpha-masked cages around an emissive
                    // bulb, so texels inside the silhouette alternate between
                    // the cage front and the bulb 0.1-0.15 m deeper through
                    // the holes; restarting the run on those texels reset
                    // the rim to the bulb depth and the cage's bulge no
                    // longer covered the back-face hits (the lamp lost the
                    // third of its mirror image nearest the glass).
                    if (candValid) {
                        const bool jump = abs(hiSceneZ - candSceneZLast) > 0.35 + 0.02 * hiSceneZ;
                        const bool pastRim = hiSceneZ > candRimZ + 0.2;
                        if (jump || pastRim) candValid = false;
                    }
                    if (!candValid) {
                        candValid = true;
                        candT = hi;
                        candOver = hiOver;
                        candUv = hiUv;
                        candDist = dist;
                        candRimZ = hiSceneZ;
                        SSR_DBG(++gSsrDbg.runs; gSsrDbg.rimZ = hiSceneZ;)
                        vec4 hw2 = invViewProj * vec4((hiUv + pxStepUv) * 2.0 - 1.0, hiDepth, 1.0);
                        hw2 /= hw2.w;
                        candPxWorld = length(hw2.xyz - hw.xyz);
                    }
                    candTBehind = hi;
                    candOverLast = hiOver;
                    candUvLast = hiUv;
                    candDistLast = dist;
                    candSceneZLast = hiSceneZ;
                    // The rim is the deepest front texel of the run: a run
                    // often starts behind a nearer flat object in front of
                    // the occluder (the window mullion in front of the
                    // lantern), and the silhouette texel is the deepest one
                    // of the occluder itself.
                    candRimZ = max(candRimZ, hiSceneZ);
                    // Symmetric convex occluder: the back face at this texel
                    // lies as far behind the rim as the front face lies in
                    // front of it.  Inside -> hit here, at this texel.
                    // kRimSlack: a curved silhouette is depth-vertical, so
                    // the deepest texel the march can sample sits inside
                    // the true rim (~sqrt(2 r tau) for a sphere: 0.07 m for
                    // the 0.26 m café lantern at half-res texels) and the
                    // symmetric estimate is short by twice that.
                    const float kRimSlack = 0.1;
                    const float bulge = max(candRimZ - hiSceneZ, 0.0);
                    SSR_DBG(if (bulge > gSsrDbg.maxBulge) { gSsrDbg.overAtMaxBulge = hiOver; gSsrDbg.uvAtMaxBulge = hiUv.x; gSsrDbg.rimAtMaxBulge = candRimZ; gSsrDbg.sceneAtMaxBulge = hiSceneZ; }
                            gSsrDbg.maxBulge = max(gSsrDbg.maxBulge, bulge);
                            if (hiOver < gSsrDbg.minOver) { gSsrDbg.minOver = hiOver; gSsrDbg.minOverBulge = 2.0 * bulge; })
                    const float margin = extentK * (2.0 * bulge + kRimSlack) + thickBase - hiOver;
                    if (bulge > 0.0 && margin >= 0.0) {
                        hit = true;
                        hitUv = hiUv;
                        hitDist = dist;
                        hitSoft = smoothstep(0.0, 0.12, margin);
                        hitLodBias = 1.0;
                    }
                }
            }
        }
        if (!reject) {
            // A reflection can only show what sits on the reflector's FRONT
            // (+N) side.  Glass writes no depth, so a ray marched from a shop
            // window crosses the pane's own screen region and meets the
            // INTERIOR behind it in the depth buffer; those back-side (-N)
            // crossings read as reflections of interior furniture and the
            // rectangular ceiling lights (triangular/rectangular blobs tiling
            // the mirror).  Reject them like any other invalid crossing; the
            // march then resumes and can still find real front-side content
            // (street, awning, the hanging lamp).  Valid opaque-path hits are
            // on the front side as well, so the test is safe for both callers.
            if (hitSide < -0.02) { reject = true; SSR_DBG(++gSsrDbg.rejBack;) }
            // Reject hits on the reflecting surface itself, not hits that are
            // simply closer to the camera (street furniture in front of a
            // shop window is a valid mirror image).  The fixed 0.22 m radius
            // discarded exactly the grazing-angle case: the ray runs almost
            // parallel to the pane and legitimately strikes frames and walls
            // within a few tens of centimetres.  Scale the radius by the
            // clearance rate |R·N| — 0.22 m head-on (old behaviour), down to
            // the origin bias at exact grazing — and keep probing on reject.
            if (dist < selfRadius || hitSide < selfPlane) { reject = true; SSR_DBG(++gSsrDbg.rejSelf;) }
        }
        if (hit) {
            SSR_DBG(gSsrDbg.endState = 1; gSsrDbg.hitDist = hitDist; gSsrDbg.hitUv = hitUv;)
            break;
        }
        if (reject) {
            // Same contract as Phase 1a's lastDiff reset: a rejected crossing
            // (sky, thickness overshoot, self-hit) must not void the rest of
            // the ray — at grazing the ray hugs surfaces with internal depth
            // jumps, and later real crossings must stay detectable.  Resume
            // one mip-0 pixel past the rejected crossing.
            tFront = t;
            t += 1.0 / pixelDist;
            continue;
        }
        hit = true;
        hitUv = hiUv;
        hitDist = dist;
        SSR_DBG(gSsrDbg.endState = 1; gSsrDbg.hitDist = dist; gSsrDbg.hitUv = hiUv;)
        break;
    }
    SSR_DBG(if (!hit && gSsrDbg.endState == 5 && t > 1.0) gSsrDbg.endState = 4;)
    if (!hit && candValid) {
        // March ended (screen edge / ray end / budget) still behind the
        // occluder: score the open run with the chord model.
        const float runPx = max((min(t, 1.0) - candT) * pixelDist, 1.0);
        const float chord = min(runPx * candPxWorld, kExtentCap);
        if (0.5 * (candOver + candOverLast) <= extentK * chord + thickBase) {
            hit = true;
            hitUv = 0.5 * (candUv + candUvLast);
            hitDist = 0.5 * (candDist + candDistLast);
            SSR_DBG(gSsrDbg.endState = 1; gSsrDbg.hitDist = hitDist; gSsrDbg.hitUv = hitUv;)
        }
    }

    if (!hit) return vec4(0.0);
    outHitDist = hitDist;

    // Roughness → mip LOD: each chain level doubles the footprint of the 2x2
    // box average, approximating a reflection cone whose aperture grows with
    // the GGX lobe.  The mapping is NONLINEAR (perceptual roughness squared,
    // i.e. linear in GGX alpha) and capped two levels short of the chain tail:
    // the last box-filter levels are only a handful of texels (the tail ends
    // near 1x1), and sampling them overlays the reflection with stacked
    // square blobs — the "blocky reflection" artifact on mid/high-roughness
    // and grazing pixels.  Trilinear sampling between levels keeps the blur
    // free of banding.  This remains the standard "colour mip chain" half of
    // stochastic-free SSR (UE samples its blurred scene-colour pyramid the
    // same way once denoising is skipped).
    //
    // Grazing term: at |R·N| → 0 the ray runs nearly parallel to the surface,
    // so a one-texel hit-position step spans metres of surface and the marched
    // hit point (hence the sampled colour) is far less stable frame to frame
    // than the roughness footprint alone implies — the dominant SSR flicker
    // source on ground planes.  Widening the cone by (1-|R·N|)² (up to +1 LOD
    // at exact grazing, 0 head-on) pulls those samples from a coarser,
    // temporally stable average; head-on mirror reflections are untouched.
    // The cap applies to the SUM, so grazing rough pixels can no longer reach
    // the blocky chain tail either.
    const float maxLod = float(max(colorMipCount - 3, 1));
    const float graze = 1.0 - grazing;
    const float lod = min(roughness * roughness * maxLod + graze * graze + hitLodBias, maxLod);
    vec3 col = textureLod(colorTex, hitUv, lod).rgb;
    // UE tames fireflies with rcp(1+lum) because SSSR is stochastic; this
    // path is a single ray and the mip chain already averages out the hot
    // texels at higher LODs, so keep linear HDR.

    vec2 edge = smoothstep(vec2(0.0), vec2(0.05), hitUv) *
                smoothstep(vec2(0.0), vec2(0.05), 1.0 - hitUv);
    // Fade with world-space hit distance, not the screen-segment parameter:
    // grazing-angle hits cluster near the clipped segment end (hitT → 1),
    // so 1 - hitT² faded exactly the far-field reflections this pass exists
    // to keep, reading as "reflection lost" on shop windows.
    float distFade = 1.0 - smoothstep(0.4 * kMaxDist, 0.9 * kMaxDist, hitDist);
    float conf = edge.x * edge.y * distFade * hitSoft;
    return vec4(col, conf);
}

// Legacy entry point (extent heuristic off): bit-identical to the pre-extent
// tracer; the opaque pass keeps using it.
vec4 traceSsr(sampler2D colorTex, int colorMipCount, sampler2D hizTex, int hizMipCount,
              mat4 viewProj, mat4 invViewProj, vec3 cameraPos, vec3 worldPos,
              vec3 N, vec3 R, float roughness, vec2 renderSize, int escalateEvery,
              int maxMipCap, vec4 cfg, int maxIters) {
    float unusedDist;
    return traceSsrEx(colorTex, colorMipCount, hizTex, hizMipCount, viewProj, invViewProj,
                      cameraPos, worldPos, N, R, roughness, renderSize, escalateEvery,
                      maxMipCap, cfg, maxIters, 0.0, unusedDist);
}

#endif
