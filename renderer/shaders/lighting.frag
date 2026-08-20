#version 450
// Deferred lighting pass: reconstructs world position from depth, shades with
// Cook-Torrance GGX + Lambert for the punctual lights (directional/point),
// adds image-based lighting (irradiance + prefiltered specular + BRDF LUT,
// split-sum) and emissive.  Pixels at the far plane render the environment as
// skybox.  Output is HDR linear color (R16G16B16A16F); present.frag tone-maps.

// GPU mirror of scene::Light; must match LightGPU in DeferredCore.h.
struct LightGPU {
    vec4 posOrDir; // xyz = position (point) / direction-to-light (directional), w = type (0 = dir, 1 = point)
    vec4 color;    // rgb + w = intensity (PI-scaled on the CPU)
    vec4 params;   // x = range (0 = infinite), y = castShadow (reserved, C2), zw = reserved
};

layout(set = 0, binding = 0) uniform LightingUBO {
    mat4 invViewProj;   // matches the GBuffer pass (jittered for the low-res path)
    vec4 cameraPos;
    LightGPU lights[8];
    vec4 lightCounts;   // x = active light count, yzw reserved
    vec4 ambient;       // unused when IBL is active (kept for reference)
    vec4 iblParams;     // x = env intensity, y = prefilter max lod, z = skybox enabled, w = unused
} u;

layout(set = 0, binding = 1) uniform sampler2D gbAlbedo;
layout(set = 0, binding = 2) uniform sampler2D gbNormal;
layout(set = 0, binding = 3) uniform sampler2D gbMaterial;
layout(set = 0, binding = 4) uniform sampler2D gbEmissive;
layout(set = 0, binding = 5) uniform sampler2D gbDepth;
layout(set = 0, binding = 6) uniform samplerCube iblIrradiance;
layout(set = 0, binding = 7) uniform samplerCube iblPrefilter;
layout(set = 0, binding = 8) uniform sampler2D iblBrdfLut;
layout(set = 0, binding = 9) uniform samplerCube envCube; // base environment (skybox)
layout(set = 0, binding = 10) uniform sampler2D ssaoTex;  // screen-space AO (ssao.comp + blur)

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdH = max(dot(N, H), 0.0);
    float d = NdH * NdH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float geometrySchlickGGX(float NdX, float roughness) {
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0; // direct lighting
    return NdX / (NdX * (1.0 - k) + k);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    return geometrySchlickGGX(max(dot(N, V), 0.0), roughness) *
           geometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    vec3 F1 = max(vec3(1.0 - roughness), F0);
    return F0 + (F1 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Shades one punctual light.  Directional: no falloff, L = posOrDir (unit
// direction towards the light).  Point: inverse-square falloff with a
// Frostbite-style window (saturate(1 - (d/range)^4)^2) that reaches zero
// exactly at range; range = 0 keeps the legacy pure inverse-square falloff.
vec3 shadeLight(vec3 N, vec3 V, vec3 worldPos, vec3 albedo, float metallic,
                float roughness, vec3 F0, LightGPU light) {
    vec3 radiance = light.color.rgb * light.color.w;
    vec3 L;
    if (light.posOrDir.w < 0.5) {
        L = light.posOrDir.xyz;
    } else {
        vec3 toLight = light.posOrDir.xyz - worldPos;
        float dist = length(toLight);
        L = toLight / dist;
        float range = light.params.x;
        float atten = 1.0 / max(dist * dist, 1e-4);
        if (range > 0.0) {
            float w = clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0);
            atten *= w * w;
        }
        radiance *= atten;
    }

    vec3 H = normalize(V + L);
    float NdL = max(dot(N, L), 0.0);
    if (NdL <= 0.0) return vec3(0.0);

    float D = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
    vec3 specular = (D * G * F) / max(4.0 * max(dot(N, V), 0.0) * NdL, 1e-4);

    vec3 kd = (vec3(1.0) - F) * (1.0 - metallic);
    return (kd * albedo / PI + specular) * radiance * NdL;
}

void main() {
    ivec2 pix = ivec2(gl_FragCoord.xy);
    float depth = texelFetch(gbDepth, pix, 0).r;

    // World position from the inverted view-projection (NDC: y down, z [0,1]).
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(gbDepth, 0));
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 wp = u.invViewProj * clip;
    wp /= wp.w;
    vec3 viewDir = normalize(wp.xyz - u.cameraPos.xyz);

    if (depth == 1.0) {
        // Skybox: sample the base environment map along the view ray.
        vec3 sky = u.iblParams.z > 0.5
                       ? textureLod(envCube, viewDir, 0.0).rgb * u.iblParams.x
                       : u.ambient.rgb * u.ambient.w;
        outColor = vec4(sky, 1.0);
        return;
    }

    vec4 albedoA = texelFetch(gbAlbedo, pix, 0);
    vec3 albedo = albedoA.rgb;
    // A2B10G10R10 stores the normal remapped to [0,1]; unpack back to [-1,1].
    vec3 N = normalize(texelFetch(gbNormal, pix, 0).xyz * 2.0 - 1.0);
    vec4 matParams = texelFetch(gbMaterial, pix, 0);
    float metallic = matParams.r;
    float roughness = max(matParams.g, 0.04); // floor: avoids GGX singularity
    float ao = matParams.b;
    // Screen-space AO (R16F, same resolution as this GBuffer path) multiplies
    // the IBL environment term together with the material AO texture.
    float ssao = texelFetch(ssaoTex, pix, 0).r;
    vec3 emissive = texelFetch(gbEmissive, pix, 0).rgb;

    vec3 V = -viewDir;
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    vec3 color = vec3(0.0);
    int lightCount = int(u.lightCounts.x + 0.5);
    for (int i = 0; i < lightCount; ++i)
        color += shadeLight(N, V, wp.xyz, albedo, metallic, roughness, F0, u.lights[i]);

    // IBL (split-sum): cosine irradiance * albedo + prefiltered GGX * LUT.
    float NdV = max(dot(N, V), 0.0);
    vec3 F = fresnelSchlickRoughness(NdV, F0, roughness);
    vec3 kd = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 irradiance = texture(iblIrradiance, N).rgb;
    vec3 diffuseIbl = irradiance * albedo;

    vec3 R = reflect(-V, N);
    vec3 prefiltered = textureLod(iblPrefilter, R, roughness * u.iblParams.y).rgb;
    vec2 brdf = texture(iblBrdfLut, vec2(NdV, roughness)).rg;
    vec3 specularIbl = prefiltered * (F * brdf.x + brdf.y);

    color += (kd * diffuseIbl + specularIbl) * ao * ssao * u.iblParams.x;
    color += emissive;

    outColor = vec4(color, 1.0);
}
