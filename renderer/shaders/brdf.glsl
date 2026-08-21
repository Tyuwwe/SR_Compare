// Shared microfacet BRDF used by lighting.frag, transparent*.frag and the
// split-sum LUT.  Specular is Heitz height-correlated GGX (not the old
// independent Schlick-Smith G).  Diffuse is Hammon 2017 (GDC), which replaces
// Lambert: a GGX-compatible facing/rough term plus a cheap multi-scatter
// bounce, no albedo/PI Lambertian disk.
#ifndef SR_BRDF_GLSL
#define SR_BRDF_GLSL

#ifndef PI
#define PI 3.14159265359
#endif

// Trowbridge-Reitz GGX NDF.  `roughness` is perceptual; alpha = r^2.
float D_GGX(float NdH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float d = NdH * NdH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Heitz 2014 height-correlated Smith G2 / (4 NdV NdL).  Filament / UE4 V term.
// Specular lobe is D * V * F (V already has the Cook-Torrance 4-NdV-NdL).
float V_SmithGGXCorrelated(float NdV, float NdL, float roughness) {
    float a2 = roughness * roughness;
    a2 *= a2;
    float gv = NdL * sqrt(NdV * NdV * (1.0 - a2) + a2);
    float gl = NdV * sqrt(NdL * NdL * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-5);
}

vec3 F_Schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 F_SchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    vec3 F1 = max(vec3(1.0 - roughness), F0);
    return F0 + (F1 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Hammon 2017 "PBR Diffuse Lighting for GGX+Smith Microsurfaces".
// Returns the diffuse BRDF (already includes 1/PI in the single-scatter
// term).  Caller still multiplies by NdL * radiance and by (1-F)*(1-metallic).
//
// The paper's (0.5+NoH)/NoH is singular at NoH→0.  Foliage (noisy normals,
// two-sided) hits that constantly and paints pure white; floor NoH at 0.1
// (Hammon's fit was in the [0.1, 1] range anyway) and clamp the mix.
vec3 Fd_Hammon(vec3 albedo, float NdV, float NdL, float NdH, float LdV, float roughness) {
    float facing = 0.5 + 0.5 * LdV;
    float nh = max(NdH, 0.1);
    float rough = facing * (0.9 - 0.4 * facing) * ((0.5 + nh) / nh);
    float smoothTerm = 1.05 * (1.0 - pow(1.0 - NdL, 5.0)) * (1.0 - pow(1.0 - NdV, 5.0));
    float single = (1.0 / PI) * clamp(mix(smoothTerm, rough, roughness), 0.0, 4.0);
    float multi = 0.1159 * roughness;
    return albedo * (single + albedo * multi);
}

// Fdez-Agüera 2019 "Multiple-Scattering Physically-Based Discriminants"
// (JCGT; the form Filament ships): multi-scatter compensation for the
// split-sum IBL specular lobe.  The BRDF LUT integrates only the first
// microfacet bounce; the missing fraction Ems = 1 - Ess (Ess = single-scatter
// directional albedo at F = 1, i.e. LUT scale + bias) re-emerges scaled by
// the hemispheric average Fresnel Favg = F0 + (1 - F0)/21 through the
// geometric series Fms = FssEss * Favg / (1 - Ems * Favg).  Brightens rough
// metals (and, mildly, rough dielectrics) back toward the ground truth.
//
// lighting.frag, transparent*.frag AND ssr_opaque.comp must all go through
// this helper: the opaque-SSR pass subtracts the exact IBL term lighting.frag
// added before compositing the traced reflection, so any divergence reads as
// energy drift on SSR hits.
vec3 specularIblMultiScatter(vec3 F, vec3 F0, vec2 brdf) {
    const vec3 FssEss = F * brdf.x + brdf.y;
    const float Ess = clamp(brdf.x + brdf.y, 0.0, 1.0);
    const float Ems = 1.0 - Ess;
    const vec3 Favg = F0 + (vec3(1.0) - F0) * (1.0 / 21.0);
    const vec3 Fms = FssEss * Favg / max(vec3(1.0) - Ems * Favg, vec3(1e-4));
    return FssEss + Fms * Ems;
}

// Direct-light BRDF split into diffuse / specular (no radiance, no NdL).
void evalBrdf(vec3 N, vec3 V, vec3 L, vec3 albedo, float metallic, float roughness, vec3 F0,
              out vec3 diffuse, out vec3 specular) {
    vec3 H = normalize(V + L);
    float NdV = max(dot(N, V), 0.0);
    float NdL = max(dot(N, L), 0.0);
    float NdH = max(dot(N, H), 0.0);
    float HdV = max(dot(H, V), 0.0);
    float LdV = max(dot(L, V), 0.0);

    float D = D_GGX(NdH, roughness);
    float Vis = V_SmithGGXCorrelated(NdV, NdL, roughness);
    vec3 F = F_Schlick(HdV, F0);
    // Correlated Smith V * D spikes on grazing backfaces / tiny roughness
    // (Bistro cutout leaves).  16 is well above a sun-disk highlight.
    specular = min(D * Vis, 16.0) * F;

    vec3 kd = (vec3(1.0) - F) * (1.0 - metallic);
    diffuse = kd * Fd_Hammon(albedo, NdV, NdL, NdH, LdV, roughness);
}

#endif
