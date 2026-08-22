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
vec4 traceSsr(sampler2D colorTex, int colorMipCount, sampler2D hizTex, int hizMipCount,
              mat4 viewProj, mat4 invViewProj, vec3 cameraPos, vec3 worldPos,
              vec3 N, vec3 R, float roughness, vec2 renderSize) {
    const int kMaxIters = 128; // hierarchy steps (each descends or advances)
    const int kRefine = 8;
    const float kMaxDist = 100.0;
    const float kBias = 0.08;
    const float kNearW = 0.02;

    vec3 origin = worldPos + N * kBias;
    vec4 startCS = viewProj * vec4(origin, 1.0);
    vec4 endCS = viewProj * vec4(origin + R * kMaxDist, 1.0);

    // Clip to the near plane so 1/w stays finite (ray toward / behind camera).
    if (startCS.w < kNearW && endCS.w < kNearW) return vec4(0.0);
    if (endCS.w < kNearW) {
        float tn = (kNearW - startCS.w) / (endCS.w - startCS.w);
        if (tn <= 0.0) return vec4(0.0);
        endCS = mix(startCS, endCS, clamp(tn, 0.0, 1.0));
    }
    if (startCS.w < kNearW) {
        float tn = (kNearW - startCS.w) / (endCS.w - startCS.w);
        startCS = mix(startCS, endCS, clamp(tn, 0.0, 1.0));
    }

    vec3 startNdc = startCS.xyz / startCS.w;
    vec3 endNdc = endCS.xyz / endCS.w;
    vec2 startUv = startNdc.xy * 0.5 + 0.5;
    vec2 endUv = endNdc.xy * 0.5 + 0.5;
    if (any(lessThan(startUv, vec2(-0.02))) || any(greaterThan(startUv, vec2(1.02))))
        return vec4(0.0);
    startUv = clamp(startUv, vec2(0.0), vec2(1.0));

    float tClip = ssrClipToScreen(startUv, endUv);
    if (tClip <= 0.02) return vec4(0.0);
    endUv = mix(startUv, endUv, tClip);
    float invW0 = 1.0 / startCS.w;
    float invW1 = mix(invW0, 1.0 / endCS.w, tClip);

    vec2 startPx = startUv * renderSize;
    vec2 endPx = endUv * renderSize;
    vec2 dPx = endPx - startPx;
    float pixelDist = length(dPx);
    // Head-on panes project the reflection ray almost onto one pixel (R ≈ V);
    // still take a few samples so a slight offset can hit on-screen geometry.
    if (pixelDist < 0.25) return vec4(0.0);

    const int maxMip = hizMipCount - 1;
    const ivec2 size0 = ivec2(renderSize);
    // |R·N|: 1 head-on, ~0 at grazing.  Scales the self-hit radius and the
    // thickness window below; grazing incidence is where both fixed
    // thresholds used to kill valid reflections (Bistro shop windows).
    const float grazing = abs(dot(R, N));

    // t parameterises the clipped segment: uv = mix(startUv, endUv, t) and
    // 1/w = mix(invW0, invW1, t), so every sample stays perspective-correct.
    float t = 0.0;
    float tFront = 0.0; // last sampled position still in front of the hull
    int level = 0;
    bool hit = false;
    vec2 hitUv = endUv;
    float hitDist = kMaxDist;

    for (int it = 0; it < kMaxIters && t <= 1.0; ++it) {
        vec2 px = startPx + dPx * t;
        vec2 uv = px / renderSize;
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) break;

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
            // In front of the farthest surface of this cell: safe to skip it,
            // then coarsen so long empty runs cost one iteration per cell.
            tFront = t;
            t += float(1 << level) / pixelDist;
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

        // Mip-0 crossing between tFront and t: binary-refine the exact
        // front-to-back transition, then apply the acceptance tests.
        float lo = tFront;
        float hi = t;
        vec2 hiUv = uv;
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
        float dist = kMaxDist;
        if (!reject) {
            float hiRayZ = abs(1.0 / mix(invW0, invW1, hi));
            float hiSceneZ = ssrViewZ(viewProj, invViewProj, hiUv, hiDepth);
            float hiOver = hiRayZ - hiSceneZ;
            // Widen the view-Z window at grazing incidence: the ray's view-Z
            // then advances slowly, so even after binary refinement the hit
            // can sit a coarse step past a thin reflector (window frame,
            // mullion) and the angle-independent window rejected it.
            float thickness = (0.35 + 0.03 * hiRayZ) / max(grazing, 0.25);
            if (hiOver < 0.0 || hiOver > thickness) reject = true;
        }
        if (!reject) {
            vec4 hw = invViewProj * vec4(hiUv * 2.0 - 1.0, hiDepth, 1.0);
            hw /= hw.w;
            dist = length(hw.xyz - worldPos);
            // Reject hits on the reflecting surface itself, not hits that are
            // simply closer to the camera (street furniture in front of a
            // shop window is a valid mirror image).  The fixed 0.22 m radius
            // discarded exactly the grazing-angle case: the ray runs almost
            // parallel to the pane and legitimately strikes frames and walls
            // within a few tens of centimetres.  Scale the radius by the
            // clearance rate |R·N| — 0.22 m head-on (old behaviour), down to
            // the origin bias at exact grazing — and keep probing on reject.
            if (dist < kBias + 0.14 * grazing) reject = true;
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
        break;
    }

    if (!hit) return vec4(0.0);

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
    const float lod = min(roughness * roughness * maxLod + graze * graze, maxLod);
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
    float conf = edge.x * edge.y * distFade;
    return vec4(col, conf);
}

#endif
