#if defined(_MSC_VER)
// std::fopen is portable; the _s variants are MSVC-only (same pattern as
// LodBuilder.cpp / VulkanContext.cpp).
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "renderer/scene/Ktx2.h"

#include <cstdio>
#include <cstring>

namespace sr {

namespace {

// «AB»KTX 20«BB»\r\n\x1A\n
constexpr uint8_t kKtx2Identifier[12] = {0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32,
                                         0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
constexpr uint32_t kVkFormatBc7Unorm = 145; // VK_FORMAT_BC7_UNORM_BLOCK
constexpr uint32_t kVkFormatBc7Srgb = 146;  // VK_FORMAT_BC7_SRGB_BLOCK

// KTX2 is little-endian by spec; the hosts we run on are too, but read via
// memcpy to keep this portable and alignment-safe.
uint32_t readU32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}
uint64_t readU64(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

// BC7: 16 bytes per 4x4 block.
uint64_t bc7LevelBytes(uint32_t width, uint32_t height) {
    const uint64_t bw = (width + 3) / 4;
    const uint64_t bh = (height + 3) / 4;
    return bw * bh * 16;
}

} // namespace

bool loadKtx2File(const char* path, Ktx2Image& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    if (std::fseek(f, 0, SEEK_END) != 0) { std::fclose(f); return false; }
    const long size = std::ftell(f);
    if (size < 0 || std::fseek(f, 0, SEEK_SET) != 0) { std::fclose(f); return false; }

    // Header (80 bytes) + level index (24 bytes per level).
    if (size < 80) { std::fclose(f); return false; }
    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (std::fread(data.data(), 1, data.size(), f) != data.size()) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);

    if (std::memcmp(data.data(), kKtx2Identifier, sizeof(kKtx2Identifier)) != 0) return false;

    const uint8_t* h = data.data() + 12;
    const uint32_t vkFormat = readU32(h + 0);
    const uint32_t typeSize = readU32(h + 4);
    const uint32_t pixelWidth = readU32(h + 8);
    const uint32_t pixelHeight = readU32(h + 12);
    const uint32_t pixelDepth = readU32(h + 16);
    const uint32_t layerCount = readU32(h + 20);
    const uint32_t faceCount = readU32(h + 24);
    const uint32_t levelCount = readU32(h + 28);
    const uint32_t supercompression = readU32(h + 32);

    if (vkFormat != kVkFormatBc7Unorm && vkFormat != kVkFormatBc7Srgb) return false;
    if (typeSize != 1 || pixelDepth != 0 || layerCount != 0 || faceCount != 1) return false;
    if (supercompression != 0) return false; // no Zstd/UASTC: blocks upload as-is
    if (pixelWidth == 0 || pixelHeight == 0 || levelCount == 0) return false;

    const uint64_t indexEnd = 80ull + static_cast<uint64_t>(levelCount) * 24ull;
    if (indexEnd > static_cast<uint64_t>(size)) return false;

    Ktx2Image img;
    img.width = pixelWidth;
    img.height = pixelHeight;
    img.mipLevels = levelCount;
    img.levels.resize(levelCount);
    for (uint32_t l = 0; l < levelCount; ++l) {
        const uint8_t* e = data.data() + 80 + l * 24;
        img.levels[l].offset = readU64(e + 0);
        img.levels[l].length = readU64(e + 8);
        // uncompressedByteLength (e + 16) equals length when unsupercompressed.
        const uint64_t end = img.levels[l].offset + img.levels[l].length;
        if (end > static_cast<uint64_t>(size)) return false;
        const uint32_t lw = pixelWidth >> l ? pixelWidth >> l : 1;
        const uint32_t lh = pixelHeight >> l ? pixelHeight >> l : 1;
        if (img.levels[l].length != bc7LevelBytes(lw, lh)) return false;
    }
    img.data = std::move(data);
    out = std::move(img);
    return true;
}

bool readKtx2Level(const char* path, uint64_t offset, uint64_t length,
                   std::vector<uint8_t>& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    if (std::fseek(f, static_cast<long>(offset), SEEK_SET) != 0) {
        std::fclose(f);
        return false;
    }
    out.resize(static_cast<size_t>(length));
    const bool ok = std::fread(out.data(), 1, out.size(), f) == out.size();
    std::fclose(f);
    if (!ok) out.clear();
    return ok;
}

} // namespace sr
