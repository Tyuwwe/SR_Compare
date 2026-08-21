// Screen-space reflection: McGuire clip-space DDA against the opaque depth,
// matching the tracing model used by UE's ScreenSpaceReflections pass
// (SSRTReflections.usf) rather than a world-space quadratic march.
//
// Why clip-space: 1/w is linear in screen along a 3D ray, so view-Z and UV
// stay perspective-correct.  The previous world-space t² march compared
// Euclidean camera distances against a nonlinear depth buffer and produced
// stretched / false hits on the Bistro shop windows.
//
// UE composite (DiffuseIndirectComposite) applies EnvBRDF on the hit colour
// and a screen-edge vignette; the caller does the EnvBRDF multiply.  We return
// rgb + confidence (0 = miss).  Hierarchical Z is approximated by a coarse
// pixel stride plus binary refinement (no extra mip chain).
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

vec4 traceSsr(sampler2D colorTex, sampler2D depthTex, mat4 viewProj, mat4 invViewProj,
              vec3 cameraPos, vec3 worldPos, vec3 N, vec3 R, vec2 renderSize) {
    const int kMaxSteps = 96;
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

    int steps = int(clamp(max(pixelDist, 8.0), 8.0, float(kMaxSteps)));
    vec2 stepPx = dPx / float(steps);

    float lastDiff = -1.0;
    bool hit = false;
    vec2 hitUv = endUv;
    float hitT = 1.0;

    for (int i = 1; i <= steps; ++i) {
        float t = float(i) / float(steps);
        vec2 px = startPx + stepPx * float(i);
        vec2 uv = px / renderSize;
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) break;

        float invW = mix(invW0, invW1, t);
        float rayZ = abs(1.0 / invW);

        ivec2 ip = ivec2(clamp(px, vec2(0.0), renderSize - 1.0));
        float sceneDepth = texelFetch(depthTex, ip, 0).r;
        if (sceneDepth >= 0.9999) {
            lastDiff = -1.0;
            continue;
        }
        float sceneZ = ssrViewZ(viewProj, invViewProj, uv, sceneDepth);
        float diff = rayZ - sceneZ;
        // Any front-to-back crossing is a candidate.  Thickness is tested
        // AFTER binary refine: a coarse 4 px step can overshoot by metres.
        if (lastDiff <= 0.0 && diff > 0.0) {
            float lo = t - 1.0 / float(steps);
            float hi = t;
            vec2 hiUv = uv;
            for (int r = 0; r < kRefine; ++r) {
                float mid = 0.5 * (lo + hi);
                vec2 mUv = mix(startUv, endUv, mid);
                float mRayZ = abs(1.0 / mix(invW0, invW1, mid));
                float mDepth = texelFetch(depthTex, ivec2(clamp(mUv * renderSize, vec2(0.0),
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
            float hiDepth = texelFetch(depthTex, ivec2(clamp(hiUv * renderSize, vec2(0.0),
                                                            renderSize - 1.0)), 0).r;
            if (hiDepth >= 0.9999) {
                lastDiff = diff;
                continue;
            }
            float hiRayZ = abs(1.0 / mix(invW0, invW1, hi));
            float hiSceneZ = ssrViewZ(viewProj, invViewProj, hiUv, hiDepth);
            float hiOver = hiRayZ - hiSceneZ;
            float thickness = 0.35 + 0.03 * hiRayZ;
            if (hiOver < 0.0 || hiOver > thickness) {
                lastDiff = diff;
                continue;
            }
            vec4 hw = invViewProj * vec4(hiUv * 2.0 - 1.0, hiDepth, 1.0);
            hw /= hw.w;
            // Reject hits on the reflecting surface itself, not hits that are
            // simply closer to the camera (street furniture in front of a
            // shop window is a valid mirror image).
            if (length(hw.xyz - worldPos) < 0.22) {
                lastDiff = diff;
                continue;
            }
            hit = true;
            hitUv = hiUv;
            hitT = hi;
            break;
        }
        lastDiff = diff;
    }

    if (!hit) return vec4(0.0);

    vec3 col = texture(colorTex, hitUv).rgb;
    // UE tames fireflies with rcp(1+lum) because SSSR is stochastic; this
    // path is a single sharp ray (shop glass), so keep linear HDR.

    vec2 edge = smoothstep(vec2(0.0), vec2(0.05), hitUv) *
                smoothstep(vec2(0.0), vec2(0.05), 1.0 - hitUv);
    float conf = edge.x * edge.y * (1.0 - hitT * hitT);
    return vec4(col, conf);
}

#endif
