#include "renderer/ColorGrading.h"

#include "renderer/core/VkUtil.h"
#include "renderer/core/Vma.h"

#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

namespace sr {

namespace {

// Tanner Helland black-body RGB approximation (1000..40000 K), 0..255.
void tannerHelland(float kelvin, float& r, float& g, float& b) {
    const double t = std::clamp(static_cast<double>(kelvin), 1000.0, 40000.0) / 100.0;
    const auto clamp255 = [](double v) { return std::clamp(v, 0.0, 255.0); };
    // Red.
    r = static_cast<float>(t <= 66.0 ? 255.0 : clamp255(329.698727446 * std::pow(t - 60.0, -0.1332047592)));
    // Green.
    g = static_cast<float>(t <= 66.0 ? clamp255(99.4708025861 * std::log(t) - 161.1195681661)
                                     : clamp255(288.1221695283 * std::pow(t - 60.0, -0.0755148492)));
    // Blue.
    b = static_cast<float>(t >= 66.0 ? 255.0
                                     : (t <= 19.0 ? 0.0
                                                  : clamp255(138.5177312231 * std::log(t - 10.0) -
                                                             305.0447927307)));
}

uint16_t floatToHalf(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int exp = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    const uint32_t mant = bits & 0x007fffffu;
    if (exp <= 0) return static_cast<uint16_t>(sign); // flush to zero (LUT values are >= 0)
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u); // inf
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

} // namespace

Vec3 whiteBalanceForTemperatureTint(float temperatureK, float tint) {
    float r, g, b;
    tannerHelland(temperatureK, r, g, b);
    float r6, g6, b6;
    tannerHelland(6500.f, r6, g6, b6);
    // Ratio against the 6500K neutral point, normalized to green = 1.
    Vec3 wb{(r / r6) / (g / g6), 1.f, (b / b6) / (g / g6)};
    // Tint shifts along the green/magenta axis (positive = magenta).
    wb.y *= std::pow(2.f, -0.25f * std::clamp(tint, -1.f, 1.f));
    return wb;
}

Vec3 ColorLut::sample(Vec3 c) const {
    const float n = static_cast<float>(size);
    const auto axis = [&](float v) {
        v = std::clamp(v, 0.f, 1.f) * (n - 1.f);
        const float i0 = std::floor(v);
        return std::pair<uint32_t, float>{static_cast<uint32_t>(i0), v - i0};
    };
    const auto [rx, fx] = axis(c.x);
    const auto [ry, fy] = axis(c.y);
    const auto [rz, fz] = axis(c.z);
    const uint32_t m = size - 1;
    const auto at = [&](uint32_t r, uint32_t g_, uint32_t b_) {
        const size_t idx = (static_cast<size_t>(b_) * size * size + static_cast<size_t>(g_) * size +
                            r) * 3;
        return Vec3{data[idx], data[idx + 1], data[idx + 2]};
    };
    Vec3 acc{0.f, 0.f, 0.f};
    for (uint32_t dz = 0; dz <= 1; ++dz) {
        for (uint32_t dy = 0; dy <= 1; ++dy) {
            for (uint32_t dx = 0; dx <= 1; ++dx) {
                const float w = (dx ? fx : 1.f - fx) * (dy ? fy : 1.f - fy) * (dz ? fz : 1.f - fz);
                if (w == 0.f) continue;
                acc += at(std::min(rx + dx, m), std::min(ry + dy, m), std::min(rz + dz, m)) * w;
            }
        }
    }
    return acc;
}

bool loadCubeLut(const char* path, ColorLut& out) {
    std::ifstream file(path);
    if (!file) {
        std::fprintf(stderr, "lut: cannot open %s\n", path);
        return false;
    }
    ColorLut lut;
    std::vector<float> values;
    std::string line;
    while (std::getline(file, line)) {
        // Strip comments and trim.
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        std::istringstream in(line);
        std::string tok;
        if (!(in >> tok)) continue;
        if (tok == "LUT_3D_SIZE") {
            uint32_t n = 0;
            if (!(in >> n) || n < 2 || n > 64) {
                std::fprintf(stderr, "lut: %s: unsupported LUT_3D_SIZE\n", path);
                return false;
            }
            lut.size = n;
            continue;
        }
        if (tok == "TITLE" || tok == "LUT_1D_SIZE") continue;
        if (tok == "DOMAIN_MIN" || tok == "DOMAIN_MAX") {
            float a = 0.f, b_ = 0.f, c = 0.f;
            in >> a >> b_ >> c;
            const bool isMin = tok == "DOMAIN_MIN";
            if ((isMin && (a != 0.f || b_ != 0.f || c != 0.f)) ||
                (!isMin && (a != 1.f || b_ != 1.f || c != 1.f))) {
                std::fprintf(stderr, "lut: %s: non-default %s unsupported\n", path, tok.c_str());
                return false;
            }
            continue;
        }
        // Data row: 3 floats (the first token is the red component).
        const char* begin = tok.c_str();
        char* end = nullptr;
        const float rv = std::strtof(begin, &end);
        if (end == begin) {
            std::fprintf(stderr, "lut: %s: unrecognized keyword '%s'\n", path, tok.c_str());
            return false;
        }
        float gv = 0.f, bv = 0.f;
        if (!(in >> gv >> bv)) {
            std::fprintf(stderr, "lut: %s: malformed data row\n", path);
            return false;
        }
        values.push_back(rv);
        values.push_back(gv);
        values.push_back(bv);
    }
    if (lut.size == 0 || values.size() < static_cast<size_t>(lut.size) * lut.size * lut.size * 3) {
        std::fprintf(stderr, "lut: %s: expected %u^3 data rows, got %zu values\n", path, lut.size,
                     values.size() / 3);
        return false;
    }
    lut.data = std::move(values);
    out = std::move(lut);
    return true;
}

ColorLut makeIdentityLut(uint32_t size) {
    ColorLut lut;
    lut.size = size;
    lut.data.reserve(static_cast<size_t>(size) * size * size * 3);
    const float n = static_cast<float>(size - 1);
    for (uint32_t b = 0; b < size; ++b)
        for (uint32_t g = 0; g < size; ++g)
            for (uint32_t r = 0; r < size; ++r) {
                lut.data.push_back(static_cast<float>(r) / n);
                lut.data.push_back(static_cast<float>(g) / n);
                lut.data.push_back(static_cast<float>(b) / n);
            }
    return lut;
}

ColorLut makeStylizedLut(uint32_t size) {
    // Gentle filmic demo look in the (normalized log) LUT domain: a mild
    // S-curve for contrast plus slightly warm highlights / cool shadows.
    ColorLut lut = makeIdentityLut(size);
    for (size_t i = 0; i + 2 < lut.data.size(); i += 3) {
        const float luma = lut.data[i] * 0.2126f + lut.data[i + 1] * 0.7152f + lut.data[i + 2] * 0.0722f;
        // S-curve around the mid pivot (smoothstep-ish, monotonic).
        const float t = std::clamp((luma - 0.25f) / 0.5f, 0.f, 1.f);
        const float s = t * t * (3.f - 2.f * t);
        const float curved = 0.25f + s * 0.5f;
        const float k = luma > 1e-6f ? curved / luma : 1.f;
        lut.data[i] = std::clamp(lut.data[i] * k + 0.012f * luma, 0.f, 1.f);        // warm gain
        lut.data[i + 1] = std::clamp(lut.data[i + 1] * k, 0.f, 1.f);
        lut.data[i + 2] = std::clamp(lut.data[i + 2] * k + 0.012f * (1.f - luma), 0.f, 1.f); // cool shadows
    }
    return lut;
}

bool writeCubeLut(const char* path, const ColorLut& lut) {
    if (!lut.valid()) return false;
    std::ofstream file(path);
    if (!file) return false;
    file << "# sr_compare procedural LUT\nLUT_3D_SIZE " << lut.size << "\n";
    char buf[128];
    for (size_t i = 0; i + 2 < lut.data.size(); i += 3) {
        std::snprintf(buf, sizeof(buf), "%.6f %.6f %.6f\n", static_cast<double>(lut.data[i]),
                      static_cast<double>(lut.data[i + 1]), static_cast<double>(lut.data[i + 2]));
        file << buf;
    }
    return file.good();
}

bool GradingLutGpu::create(const VulkanContext& ctx, const ColorLut& lut) {
    if (!lut.valid()) return false;
    const uint32_t n = lut.size;
    // RGBA16F: half-float linear filtering is universally supported, and the
    // LUT values live in [0,1] where 16F has plenty of precision.
    std::vector<uint16_t> pixels(static_cast<size_t>(n) * n * n * 4);
    for (size_t i = 0, j = 0; i + 2 < lut.data.size(); i += 3, j += 4) {
        pixels[j + 0] = floatToHalf(lut.data[i]);
        pixels[j + 1] = floatToHalf(lut.data[i + 1]);
        pixels[j + 2] = floatToHalf(lut.data[i + 2]);
        pixels[j + 3] = floatToHalf(1.f);
    }

    const VkFormat format = VK_FORMAT_R16G16B16A16_SFLOAT;
    if (createImage3D(ctx, n, n, n, format,
                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, image_,
                      memory_) != VK_SUCCESS)
        return false;
    view_ = createImageView(ctx, image_, format, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1,
                            VK_IMAGE_VIEW_TYPE_3D);
    if (!view_) return false;
    sampler_ = createSampler(ctx, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    if (!sampler_) return false;

    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation stagingMemory = VK_NULL_HANDLE;
    if (createBuffer(ctx, pixels.size() * sizeof(uint16_t), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging, stagingMemory) != VK_SUCCESS)
        return false;
    void* mapped = nullptr;
    if (vmaMapMemory(ctx.allocator, stagingMemory, &mapped) != VK_SUCCESS) return false;
    std::memcpy(mapped, pixels.data(), pixels.size() * sizeof(uint16_t));
    vmaUnmapMemory(ctx.allocator, stagingMemory);

    submitUploadOneShot(
        ctx,
        [&](VkCommandBuffer cmd) {
            imageBarrier(cmd, image_, VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE,
                         VK_ACCESS_2_NONE, sync::kCopy, sync::kTransferWrite);
            VkBufferImageCopy region = {};
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {n, n, n};
            vkCmdCopyBufferToImage(cmd, staging, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                   &region);
        },
        [&](VkCommandBuffer cmd) {
            imageBarrier(cmd, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sync::kCopy,
                         sync::kTransferWrite, sync::kSampleStages, sync::kSampled);
        });
    vmaDestroyBuffer(ctx.allocator, staging, stagingMemory);
    return true;
}

void GradingLutGpu::destroy(const VulkanContext& ctx) {
    if (sampler_) { vkDestroySampler(ctx.device, sampler_, nullptr); sampler_ = VK_NULL_HANDLE; }
    if (view_) { vkDestroyImageView(ctx.device, view_, nullptr); view_ = VK_NULL_HANDLE; }
    if (image_) { vmaDestroyImage(ctx.allocator, image_, memory_); image_ = VK_NULL_HANDLE; memory_ = VK_NULL_HANDLE; }
}

} // namespace sr
