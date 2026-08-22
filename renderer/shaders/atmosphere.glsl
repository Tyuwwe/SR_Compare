// ============================================================================
// Hillaire 2020 sky/atmosphere model — shared constants + helpers.
// Reference: Sébastien Hillaire, "A Scalable and Production Ready Sky and
// Atmosphere Rendering Technique" (EGSR 2020) and its reference implementation
// (sebh/UnrealEngineSkyAtmosphere, MIT; UE4 SkyAtmosphere.usf).
// Units: kilometres.  Y-up planet frame, earth centre at the origin.
// ============================================================================

const float ATM_BOTTOM_RADIUS = 6360.0; // km, Earth mean radius
const float ATM_TOP_RADIUS = 6460.0;    // km, 100 km shell
const float ATM_PLANET_OFFSET = 0.01;   // km, offsets shadow rays off the ground

// Rayleigh / Mie / ozone coefficients (1/km) — UE4 SkyAtmosphere Earth defaults.
const vec3 ATM_RAYLEIGH_SCATTERING = vec3(0.005802, 0.013558, 0.033100);
const float ATM_RAYLEIGH_DENSITY_SCALE = -1.0 / 8.0; // 8 km scale height
const float ATM_MIE_SCATTERING = 0.003996;
const float ATM_MIE_EXTINCTION = 0.004440; // scattering + absorption
const float ATM_MIE_DENSITY_SCALE = -1.0 / 1.2;      // 1.2 km scale height
const float ATM_MIE_PHASE_G = 0.8;
const vec3 ATM_OZONE_ABSORPTION = vec3(0.000650, 0.001881, 0.000085);
// Ozone tent profile: width 25 km, peak at 25 km (UE4 SkyAtmosphere).
const float ATM_OZONE_LAYER_WIDTH = 25.0;
const float ATM_OZONE_LINEAR_0 = 1.0 / 15.0;
const float ATM_OZONE_CONST_0 = -2.0 / 3.0;
const float ATM_OZONE_LINEAR_1 = -1.0 / 15.0;
const float ATM_OZONE_CONST_1 = 8.0 / 3.0;

const float ATM_GROUND_ALBEDO = 0.3;

const float ATM_PI = 3.14159265359;

// Nearest positive intersection of ray (ro, rd) with a sphere at the origin;
// -1 when there is none.
float atmRaySphereNearest(vec3 ro, vec3 rd, float radius) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0) return -1.0;
    float t = -b - sqrt(disc);
    if (t < 0.0) t = -b + sqrt(disc); // ro inside the sphere
    return t;
}

// Participating-media sample at planet-relative position p (Hillaire 2020,
// sampleMediumRGB).  sigmaS = scattering (Rayleigh + Mie), sigmaT = extinction
// (adds Mie absorption and ozone absorption).
void atmSampleMedium(vec3 p, out vec3 sigmaSRay, out vec3 sigmaSMie, out vec3 sigmaT) {
    float h = length(p) - ATM_BOTTOM_RADIUS; // altitude in km
    float dMie = exp(ATM_MIE_DENSITY_SCALE * h);
    float dRay = exp(ATM_RAYLEIGH_DENSITY_SCALE * h);
    float dOzo = clamp(h < ATM_OZONE_LAYER_WIDTH
                           ? ATM_OZONE_LINEAR_0 * h + ATM_OZONE_CONST_0
                           : ATM_OZONE_LINEAR_1 * h + ATM_OZONE_CONST_1,
                       0.0, 1.0);
    sigmaSRay = ATM_RAYLEIGH_SCATTERING * dRay;
    sigmaSMie = vec3(ATM_MIE_SCATTERING * dMie);
    sigmaT = sigmaSRay + vec3(ATM_MIE_EXTINCTION * dMie) + ATM_OZONE_ABSORPTION * dOzo;
}

float atmRayleighPhase(float cosTheta) {
    return 3.0 / (16.0 * ATM_PI) * (1.0 + cosTheta * cosTheta);
}

// Henyey-Greenstein.  cosTheta = dot(dirToSun, viewRay): peaks looking at the
// sun (matches the reference's CornetteShanks call chain with -cosTheta).
float atmMiePhase(float cosTheta) {
    float g = ATM_MIE_PHASE_G;
    float denom = 1.0 + g * g - 2.0 * g * cosTheta;
    return (1.0 - g * g) / (4.0 * ATM_PI * denom * sqrt(denom));
}

// --- Transmittance LUT parameterisation -------------------------------------
// uv -> (viewHeight r, view-zenith cos mu) and back; Bruneton 2017 unit-sphere
// mapping as used by Hillaire 2020 (UvToLutTransmittanceParams).
void atmUvToLutTransmittanceParams(vec2 uv, out float r, out float mu) {
    float h = sqrt(ATM_TOP_RADIUS * ATM_TOP_RADIUS - ATM_BOTTOM_RADIUS * ATM_BOTTOM_RADIUS);
    float rho = h * uv.y;
    r = sqrt(rho * rho + ATM_BOTTOM_RADIUS * ATM_BOTTOM_RADIUS);
    float dMin = ATM_TOP_RADIUS - r;
    float dMax = rho + h;
    float d = dMin + uv.x * (dMax - dMin);
    mu = d == 0.0 ? 1.0 : (h * h - rho * rho - d * d) / (2.0 * r * d);
    mu = clamp(mu, -1.0, 1.0);
}

vec2 atmLutTransmittanceParamsToUv(float r, float mu) {
    float h = sqrt(ATM_TOP_RADIUS * ATM_TOP_RADIUS - ATM_BOTTOM_RADIUS * ATM_BOTTOM_RADIUS);
    float rho = sqrt(max(r * r - ATM_BOTTOM_RADIUS * ATM_BOTTOM_RADIUS, 0.0));
    // Distance to the top atmosphere boundary along (r, mu).
    float disc = r * r * (mu * mu - 1.0) + ATM_TOP_RADIUS * ATM_TOP_RADIUS;
    float d = max(-r * mu + sqrt(max(disc, 0.0)), 0.0);
    float dMin = ATM_TOP_RADIUS - r;
    float dMax = rho + h;
    float xMu = (dMax == dMin) ? 0.0 : (d - dMin) / (dMax - dMin);
    float xR = rho / h;
    return vec2(xMu, xR);
}

// Transmittance from planet-relative point p towards the sun (sunDir: unit
// direction towards the sun).
vec3 atmTransmittanceToSun(sampler2D lut, vec3 p, vec3 sunDir) {
    float r = length(p);
    float mu = dot(p / r, sunDir);
    return textureLod(lut, atmLutTransmittanceParamsToUv(r, mu), 0.0).rgb;
}

// 0 when the ray from p towards the sun hits the planet (earth shadow).
float atmEarthShadow(vec3 p, vec3 sunDir) {
    vec3 up = p / length(p);
    return atmRaySphereNearest(p + up * ATM_PLANET_OFFSET, sunDir, ATM_BOTTOM_RADIUS) >= 0.0
               ? 0.0
               : 1.0;
}

// Multi-scatter LUT sample (Hillaire 2020 equation 10, Psi_ms): uv.x = sun
// zenith cosine remapped to [0,1], uv.y = altitude fraction.
vec3 atmMultiScattering(sampler2D lut, vec3 p, vec3 sunDir) {
    float r = length(p);
    float mu = dot(p / r, sunDir);
    vec2 uv = vec2(mu * 0.5 + 0.5,
                   clamp((r - ATM_BOTTOM_RADIUS) / (ATM_TOP_RADIUS - ATM_BOTTOM_RADIUS),
                         0.0, 1.0));
    return textureLod(lut, uv, 0.0).rgb;
}
