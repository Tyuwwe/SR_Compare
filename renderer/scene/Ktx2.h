#pragma once
// ============================================================================
// Minimal KTX2 container reader for directly-uploadable BC7 textures.
//
// Scope is deliberately narrow: uncompressed (supercompressionScheme == 0)
// single-face 2D images whose vkFormat is BC7 UNORM/SRGB, produced offline by
// scripts/transcode_textures.py (AMD CompressonatorCLI).  BC7 block payloads
// are format-agnostic between UNORM and SRGB (the transfer function is a
// sampling-time property), so the loader picks the VkFormat from the material
// usage and only validates that the file carries BC7 blocks.
// ============================================================================

#include <cstdint>
#include <vector>

namespace sr {

struct Ktx2Image {
    struct Level {
        uint64_t offset = 0; // byte offset into `data`
        uint64_t length = 0;
    };
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mipLevels = 0;
    std::vector<Level> levels;  // index 0 = largest mip
    std::vector<uint8_t> data;  // whole file; level L at data[offset, offset+length)
};

// Reads + validates the file; returns false (out untouched) for anything that
// is not a plain BC7 KTX2 — the caller then falls back to the PNG path.
bool loadKtx2File(const char* path, Ktx2Image& out);

// Reads a single level's bytes straight from the file (mip streaming: the
// coarse tail is uploaded at load, fine levels are re-read on demand).
bool readKtx2Level(const char* path, uint64_t offset, uint64_t length,
                   std::vector<uint8_t>& out);

} // namespace sr
