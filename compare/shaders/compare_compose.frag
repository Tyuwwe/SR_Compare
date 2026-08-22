#version 450
#extension GL_GOOGLE_include_directive : require
#include "tonemap.glsl"
// Compare-mode column compose: samples the column's HDR source (native GT or
// an upscaler output) through a source-region window (aspect-preserving crop
// + optional zoom/pan, computed on the CPU), applies the shared display
// transform (Hill fitted ACES + gamma 2.2, same as the viewer present pass /
// CPU screenshot path) and blits
// a 5x7 bitmap-font text overlay (algorithm name + FPS + live PSNR/SSIM)
// in the top-left corner.
//
// The region window keeps the source aspect ratio: the CPU passes a
// normalized (offset, size) rect that fills the column and crops the
// overflow.  When the on-screen magnification exceeds 1:1 the source is
// sampled with nearest filtering (texelFetch) so individual source pixels
// stay distinguishable for pixel-level algorithm comparison.
//
// Text comes in as packed ASCII codes in a UBO (5 slots x 96 chars); glyph
// pixels are sampled from an R8 atlas (16x6 grid of 8x8 cells, ASCII 32..127).

layout(set = 0, binding = 0) uniform sampler2D uSource;
layout(std140, set = 0, binding = 1) uniform TextUBO {
    uvec4 text[5][24]; // 96 chars per column slot, row-major (4 lines x 24)
} uText;
layout(set = 0, binding = 2) uniform sampler2D uFont;

layout(push_constant) uniform Push {
    vec4 colSize;  // xy = column size in pixels, z = scale, w = textSlot
    vec4 uvRect;   // xy = source region offset (normalized), zw = region size
    vec4 params;   // xy = source image size in pixels, z = 1 => nearest sampling,
                   // w = display exposure
    // Terminal lens-effects chain (same constants/algorithm as the viewer's
    // present.frag; lens dirt is viewer-only because the compare/GUI paths
    // have no HDR bloom chain).  All effects are skipped in nearest
    // (pixel-peep) mode so magnified pixels stay unmodified.  w = frame index
    // (grain hash seed).
    vec4 lensA;    // x = chromatic aberration, y = vignette, z = film grain
} pc;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

// Same deterministic integer hash as present.frag (film grain seed).
float hash13(uvec3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    return float(v.x & 0x00FFFFFFu) * (1.0 / 16777216.0);
}

void main() {
    const vec2 suv = pc.uvRect.xy + vUV * pc.uvRect.zw;
    // Lens chain runs at the same strength for every column (each column is
    // an independent present).  Off in nearest mode (pixel-level inspection).
    const bool lensFx = pc.params.z <= 0.5 && pc.lensA.xyz != vec3(0.0);
    const vec2 d = vUV - 0.5;
    vec3 c;
    if (pc.params.z > 0.5) {
        // Nearest: round the source-texel coordinate, clamp to the image.
        vec2 t = suv * pc.params.xy - 0.5;
        ivec2 ip = ivec2(clamp(floor(t + 0.5), vec2(0.0), pc.params.xy - 1.0));
        c = texelFetch(uSource, ip, 0).rgb;
    } else if (lensFx && pc.lensA.x > 0.0) {
        // Chromatic aberration: radial RGB split, squared-distance falloff.
        const vec2 off = d * dot(d, d) * pc.lensA.x;
        c.r = texture(uSource, suv - off).r;
        c.g = texture(uSource, suv).g;
        c.b = texture(uSource, suv + off).b;
    } else {
        c = texture(uSource, suv).rgb;
    }
    c = tonemapToDisplay(c, pc.params.w);
    if (lensFx) {
        if (pc.lensA.y > 0.0) c *= 1.0 - pc.lensA.y * dot(d, d) * 2.0;
        if (pc.lensA.z > 0.0) {
            const float n = hash13(uvec3(uvec2(gl_FragCoord.xy), uint(pc.lensA.w))) - 0.5;
            c += pc.lensA.z * n;
        }
    }

    const uint scale = uint(pc.colSize.z);
    const uint slot = uint(pc.colSize.w);
    vec2 px = vUV * pc.colSize.xy;

    const float padX = 6.0;
    const float padY = 6.0;
    const float lineH = 8.0 * float(scale);   // 7px glyph + 1px leading
    const float bandW = 24.0 * 6.0 * float(scale) + padX;
    const float bandH = 4.0 * lineH + padY;

    if (px.x < bandW && px.y < bandH) {
        c *= 0.35; // dim backdrop so white text stays readable on bright scenes
        vec2 local = px - vec2(padX, padY);
        if (local.x >= 0.0 && local.y >= 0.0) {
            int ci = int(local.x / (6.0 * float(scale)));
            int line = int(local.y / lineH);
            if (ci < 24 && line < 4) {
                int charIdx = line * 24 + ci;
                uint code = uText.text[slot][charIdx >> 2][charIdx & 3];
                if (code != 0u && code >= 32u) {
                    vec2 cell = vec2(mod(local.x, 6.0 * float(scale)), mod(local.y, lineH));
                    if (cell.x < 5.0 * float(scale) && cell.y < 7.0 * float(scale)) {
                        vec2 g = floor(cell / float(scale));
                        uint g0 = code - 32u;
                        vec2 uv = (vec2(float(g0 % 16u) * 8.0, float(g0 / 16u) * 8.0) + g + 0.5)
                                  / vec2(128.0, 48.0);
                        float a = texture(uFont, uv).r;
                        c = mix(c, vec3(1.0), a);
                    }
                }
            }
        }
    }

    outColor = vec4(c, 1.0);
}
