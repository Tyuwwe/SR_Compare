#pragma once
// ============================================================================
// Minimal column-major math library (matching IUpscaler.h conventions).
//   - Mat4 stores float[16] column-major: m[col * 4 + row].
//   - Camera matrices use the Vulkan/GLM right-handed convention
//     (camera looks down -Z, clip space Y is down, depth is [0,1]).
// Kept intentionally small; no external math dependency.
// ============================================================================
#include <cmath>
#include <cstdint>

namespace sr {

struct Vec2 {
    float x = 0.f, y = 0.f;
};

struct Vec3 {
    float x = 0.f, y = 0.f, z = 0.f;

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(float s) const { return {x / s, y / s, z / s}; }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
};

struct Vec4 {
    float x = 0.f, y = 0.f, z = 0.f, w = 0.f;
};

inline Vec3 operator*(float s, const Vec3& v) { return {v.x * s, v.y * s, v.z * s}; }

inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length(const Vec3& v) { return std::sqrt(dot(v, v)); }
inline Vec3 normalize(const Vec3& v) {
    const float l = length(v);
    return l > 1e-8f ? v / l : Vec3{};
}

struct Mat4 {
    float m[16] = {
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        0.f, 0.f, 0.f, 1.f,
    };

    static Mat4 identity() { return Mat4{}; }

    static Mat4 perspective(float fovY, float aspect, float znear, float zfar) {
        Mat4 r;
        const float f = 1.f / std::tan(fovY * 0.5f);
        r.m[0] = f / aspect;
        r.m[5] = -f;                          // flip Y for Vulkan NDC
        r.m[10] = zfar / (znear - zfar);
        r.m[11] = -1.f;
        r.m[14] = znear * zfar / (znear - zfar); // = -(near*far)/(far-near)
        r.m[15] = 0.f;
        return r;
    }

    static Mat4 lookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
        const Vec3 f = normalize(center - eye);
        const Vec3 s = normalize(cross(f, up));
        const Vec3 u = cross(s, f);
        Mat4 r;
        // Column-major: row 0 = s, row 1 = u, row 2 = -f (gluLookAt layout).
        r.m[0] = s.x;  r.m[4] = s.y;  r.m[8] = s.z;   r.m[12] = -dot(s, eye);
        r.m[1] = u.x;  r.m[5] = u.y;  r.m[9] = u.z;   r.m[13] = -dot(u, eye);
        r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z; r.m[14] = dot(f, eye);
        r.m[15] = 1.f;
        return r;
    }

    static Mat4 translation(const Vec3& t) {
        Mat4 r;
        r.m[12] = t.x; r.m[13] = t.y; r.m[14] = t.z;
        return r;
    }

    static Mat4 scale(const Vec3& s) {
        Mat4 r;
        r.m[0] = s.x; r.m[5] = s.y; r.m[10] = s.z;
        return r;
    }

    static Mat4 rotationY(float radians) {
        const float c = std::cos(radians), s = std::sin(radians);
        Mat4 r;
        r.m[0] = c;   r.m[2] = s;
        r.m[8] = -s;  r.m[10] = c;
        return r;
    }

    static Mat4 multiply(const Mat4& a, const Mat4& b) {
        Mat4 r;
        for (int c = 0; c < 4; ++c) {
            for (int row = 0; row < 4; ++row) {
                float v = 0.f;
                for (int k = 0; k < 4; ++k)
                    v += a.m[k * 4 + row] * b.m[c * 4 + k];
                r.m[c * 4 + row] = v;
            }
        }
        return r;
    }

    static Mat4 transpose(const Mat4& a) {
        Mat4 r;
        for (int c = 0; c < 4; ++c)
            for (int row = 0; row < 4; ++row)
                r.m[c * 4 + row] = a.m[row * 4 + c];
        return r;
    }

    // General 4x4 inverse (Gauss-Jordan with partial pivoting).
    static Mat4 inverse(const Mat4& a) {
        float aug[4][8] = {};
        for (int c = 0; c < 4; ++c) {
            for (int row = 0; row < 4; ++row) {
                aug[row][c] = a.m[c * 4 + row];
                aug[row][c + 4] = (row == c) ? 1.f : 0.f;
            }
        }
        for (int col = 0; col < 4; ++col) {
            int pivot = col;
            float best = std::fabs(aug[col][col]);
            for (int r = col + 1; r < 4; ++r) {
                if (std::fabs(aug[r][col]) > best) { best = std::fabs(aug[r][col]); pivot = r; }
            }
            if (best < 1e-12f) return identity();
            if (pivot != col)
                for (int k = 0; k < 8; ++k) { const float t = aug[col][k]; aug[col][k] = aug[pivot][k]; aug[pivot][k] = t; }
            const float d = aug[col][col];
            for (int k = 0; k < 8; ++k) aug[col][k] /= d;
            for (int r = 0; r < 4; ++r) {
                if (r == col) continue;
                const float f = aug[r][col];
                if (std::fabs(f) < 1e-12f) continue;
                for (int k = 0; k < 8; ++k) aug[r][k] -= aug[col][k] * f;
            }
        }
        Mat4 r;
        for (int c = 0; c < 4; ++c)
            for (int row = 0; row < 4; ++row)
                r.m[c * 4 + row] = aug[row][c + 4];
        return r;
    }
};

inline Mat4 operator*(const Mat4& a, const Mat4& b) { return Mat4::multiply(a, b); }

// Transform a point (w=1) by a matrix.
inline Vec3 transformPoint(const Mat4& m, const Vec3& p) {
    const float x = m.m[0] * p.x + m.m[4] * p.y + m.m[8] * p.z + m.m[12];
    const float y = m.m[1] * p.x + m.m[5] * p.y + m.m[9] * p.z + m.m[13];
    const float z = m.m[2] * p.x + m.m[6] * p.y + m.m[10] * p.z + m.m[14];
    const float w = m.m[3] * p.x + m.m[7] * p.y + m.m[11] * p.z + m.m[15];
    return {x / w, y / w, z / w};
}

// Transform a direction (w=0): upper 3x3 only, no translation.  Not
// normalized (caller decides; non-uniform scale skews directions).
inline Vec3 transformDirection(const Mat4& m, const Vec3& v) {
    return {m.m[0] * v.x + m.m[4] * v.y + m.m[8] * v.z,
            m.m[1] * v.x + m.m[5] * v.y + m.m[9] * v.z,
            m.m[2] * v.x + m.m[6] * v.y + m.m[10] * v.z};
}

// Halton(base, index), index >= 1.
inline float halton(uint32_t index, uint32_t base) {
    float f = 1.f, r = 0.f;
    uint32_t i = index;
    while (i > 0) {
        f /= static_cast<float>(base);
        r += f * static_cast<float>(i % base);
        i /= base;
    }
    return r;
}

// Halton(2,3) 2D sample in [0,1)^2 for the given frame (frame >= 1).
inline Vec2 halton23(uint32_t frameIndex) {
    return {halton(frameIndex, 2), halton(frameIndex, 3)};
}

} // namespace sr
