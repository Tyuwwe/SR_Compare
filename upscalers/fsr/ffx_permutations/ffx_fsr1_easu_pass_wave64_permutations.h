#include "ffx_fsr1_easu_pass_wave64_b3758cdfcf3fe567f62b3cd45cbbdf89.h"
#include "ffx_fsr1_easu_pass_wave64_024f4fdefb2394e934274a005e95bcbe.h"
#include "ffx_fsr1_easu_pass_wave64_2d8685ca856679ab3d82d54889435989.h"
#include "ffx_fsr1_easu_pass_wave64_140795acc7b520bfb43966f003b3e921.h"

typedef union ffx_fsr1_easu_pass_wave64_PermutationKey {
    struct {
        uint32_t FFX_FSR1_OPTION_APPLY_RCAS : 1;
        uint32_t FFX_FSR1_OPTION_RCAS_PASSTHROUGH_ALPHA : 1;
        uint32_t FFX_FSR1_OPTION_SRGB_CONVERSIONS : 1;
    };
    uint32_t index;
} ffx_fsr1_easu_pass_wave64_PermutationKey;

typedef struct ffx_fsr1_easu_pass_wave64_PermutationInfo {
    const uint32_t       blobSize;
    const unsigned char* blobData;


    const uint32_t  numConstantBuffers;
    const char**    constantBufferNames;
    const uint32_t* constantBufferBindings;
    const uint32_t* constantBufferCounts;
    const uint32_t* constantBufferSpaces;

    const uint32_t  numSRVTextures;
    const char**    srvTextureNames;
    const uint32_t* srvTextureBindings;
    const uint32_t* srvTextureCounts;
    const uint32_t* srvTextureSpaces;

    const uint32_t  numUAVTextures;
    const char**    uavTextureNames;
    const uint32_t* uavTextureBindings;
    const uint32_t* uavTextureCounts;
    const uint32_t* uavTextureSpaces;

    const uint32_t  numSRVBuffers;
    const char**    srvBufferNames;
    const uint32_t* srvBufferBindings;
    const uint32_t* srvBufferCounts;
    const uint32_t* srvBufferSpaces;

    const uint32_t  numUAVBuffers;
    const char**    uavBufferNames;
    const uint32_t* uavBufferBindings;
    const uint32_t* uavBufferCounts;
    const uint32_t* uavBufferSpaces;

    const uint32_t  numSamplers;
    const char**    samplerNames;
    const uint32_t* samplerBindings;
    const uint32_t* samplerCounts;
    const uint32_t* samplerSpaces;

    const uint32_t  numRTAccelerationStructures;
    const char**    rtAccelerationStructureNames;
    const uint32_t* rtAccelerationStructureBindings;
    const uint32_t* rtAccelerationStructureCounts;
    const uint32_t* rtAccelerationStructureSpaces;

    const uint32_t  numRTTextures;
    const char**    rtTextureNames;
    const uint32_t* rtTextureBindings;
    const uint32_t* rtTextureCounts;
    const uint32_t* rtTextureSpaces;

    const uint32_t  numSRVTensors;
    const char**    srvTensorNames;
    const uint32_t* srvTensorBindings;
    const uint32_t* srvTensorCounts;
    const uint32_t* srvTensorSpaces;

    const uint32_t  numUAVTensors;
    const char**    uavTensorNames;
    const uint32_t* uavTensorBindings;
    const uint32_t* uavTensorCounts;
    const uint32_t* uavTensorSpaces;
} ffx_fsr1_easu_pass_wave64_PermutationInfo;

static const uint32_t g_ffx_fsr1_easu_pass_wave64_IndirectionTable[] = {
    2,
    1,
    2,
    1,
    3,
    0,
    3,
    0,
};

static const ffx_fsr1_easu_pass_wave64_PermutationInfo g_ffx_fsr1_easu_pass_wave64_PermutationInfo[] = {
    { g_ffx_fsr1_easu_pass_wave64_b3758cdfcf3fe567f62b3cd45cbbdf89_size, g_ffx_fsr1_easu_pass_wave64_b3758cdfcf3fe567f62b3cd45cbbdf89_data, 1, g_ffx_fsr1_easu_pass_wave64_b3758cdfcf3fe567f62b3cd45cbbdf89_CBVResourceNames, g_ffx_fsr1_easu_pass_wave64_b3758cdfcf3fe567f62b3cd45cbbdf89_CBVResourceBindings, g_ffx_fsr1_easu_pass_wave64_b3758cdfcf3fe567f62b3cd45cbbdf89_CBVResourceCounts, g_ffx_fsr1_easu_pass_wave64_b3758cdfcf3fe567f62b3cd45cbbdf89_CBVResourceSets, 1, g_ffx_fsr1_easu_pass_wave64_b3758cdfcf3fe567f62b3cd45cbbdf89_TextureSRVResourceNames, g_ffx_fsr1_easu_pass_wave64_b3758cdfcf3fe567f62b3cd45cbbdf89_TextureSRVResourceBindings, g_ffx_fsr1_easu_pass_wave64_b3758cdfcf3fe567f62b3cd45cbbdf89_TextureSRVResourceCounts, g_ffx_fsr1_easu_pass_wave64_b3758cdfcf3fe567f62b3cd45cbbdf89_TextureSRVResourceSets, 1, g_ffx_fsr1_easu_pass_wave64_b3758cdfcf3fe567f62b3cd45cbbdf89_TextureUAVResourceNames, g_ffx_fsr1_easu_pass_wave64_b3758cdfcf3fe567f62b3cd45cbbdf89_TextureUAVResourceBindings, g_ffx_fsr1_easu_pass_wave64_b3758cdfcf3fe567f62b3cd45cbbdf89_TextureUAVResourceCounts, g_ffx_fsr1_easu_pass_wave64_b3758cdfcf3fe567f62b3cd45cbbdf89_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr1_easu_pass_wave64_b3758cdfcf3fe567f62b3cd45cbbdf89_SamplerResourceNames, g_ffx_fsr1_easu_pass_wave64_b3758cdfcf3fe567f62b3cd45cbbdf89_SamplerResourceBindings, g_ffx_fsr1_easu_pass_wave64_b3758cdfcf3fe567f62b3cd45cbbdf89_SamplerResourceCounts, g_ffx_fsr1_easu_pass_wave64_b3758cdfcf3fe567f62b3cd45cbbdf89_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr1_easu_pass_wave64_024f4fdefb2394e934274a005e95bcbe_size, g_ffx_fsr1_easu_pass_wave64_024f4fdefb2394e934274a005e95bcbe_data, 1, g_ffx_fsr1_easu_pass_wave64_024f4fdefb2394e934274a005e95bcbe_CBVResourceNames, g_ffx_fsr1_easu_pass_wave64_024f4fdefb2394e934274a005e95bcbe_CBVResourceBindings, g_ffx_fsr1_easu_pass_wave64_024f4fdefb2394e934274a005e95bcbe_CBVResourceCounts, g_ffx_fsr1_easu_pass_wave64_024f4fdefb2394e934274a005e95bcbe_CBVResourceSets, 1, g_ffx_fsr1_easu_pass_wave64_024f4fdefb2394e934274a005e95bcbe_TextureSRVResourceNames, g_ffx_fsr1_easu_pass_wave64_024f4fdefb2394e934274a005e95bcbe_TextureSRVResourceBindings, g_ffx_fsr1_easu_pass_wave64_024f4fdefb2394e934274a005e95bcbe_TextureSRVResourceCounts, g_ffx_fsr1_easu_pass_wave64_024f4fdefb2394e934274a005e95bcbe_TextureSRVResourceSets, 1, g_ffx_fsr1_easu_pass_wave64_024f4fdefb2394e934274a005e95bcbe_TextureUAVResourceNames, g_ffx_fsr1_easu_pass_wave64_024f4fdefb2394e934274a005e95bcbe_TextureUAVResourceBindings, g_ffx_fsr1_easu_pass_wave64_024f4fdefb2394e934274a005e95bcbe_TextureUAVResourceCounts, g_ffx_fsr1_easu_pass_wave64_024f4fdefb2394e934274a005e95bcbe_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr1_easu_pass_wave64_024f4fdefb2394e934274a005e95bcbe_SamplerResourceNames, g_ffx_fsr1_easu_pass_wave64_024f4fdefb2394e934274a005e95bcbe_SamplerResourceBindings, g_ffx_fsr1_easu_pass_wave64_024f4fdefb2394e934274a005e95bcbe_SamplerResourceCounts, g_ffx_fsr1_easu_pass_wave64_024f4fdefb2394e934274a005e95bcbe_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr1_easu_pass_wave64_2d8685ca856679ab3d82d54889435989_size, g_ffx_fsr1_easu_pass_wave64_2d8685ca856679ab3d82d54889435989_data, 1, g_ffx_fsr1_easu_pass_wave64_2d8685ca856679ab3d82d54889435989_CBVResourceNames, g_ffx_fsr1_easu_pass_wave64_2d8685ca856679ab3d82d54889435989_CBVResourceBindings, g_ffx_fsr1_easu_pass_wave64_2d8685ca856679ab3d82d54889435989_CBVResourceCounts, g_ffx_fsr1_easu_pass_wave64_2d8685ca856679ab3d82d54889435989_CBVResourceSets, 1, g_ffx_fsr1_easu_pass_wave64_2d8685ca856679ab3d82d54889435989_TextureSRVResourceNames, g_ffx_fsr1_easu_pass_wave64_2d8685ca856679ab3d82d54889435989_TextureSRVResourceBindings, g_ffx_fsr1_easu_pass_wave64_2d8685ca856679ab3d82d54889435989_TextureSRVResourceCounts, g_ffx_fsr1_easu_pass_wave64_2d8685ca856679ab3d82d54889435989_TextureSRVResourceSets, 1, g_ffx_fsr1_easu_pass_wave64_2d8685ca856679ab3d82d54889435989_TextureUAVResourceNames, g_ffx_fsr1_easu_pass_wave64_2d8685ca856679ab3d82d54889435989_TextureUAVResourceBindings, g_ffx_fsr1_easu_pass_wave64_2d8685ca856679ab3d82d54889435989_TextureUAVResourceCounts, g_ffx_fsr1_easu_pass_wave64_2d8685ca856679ab3d82d54889435989_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr1_easu_pass_wave64_2d8685ca856679ab3d82d54889435989_SamplerResourceNames, g_ffx_fsr1_easu_pass_wave64_2d8685ca856679ab3d82d54889435989_SamplerResourceBindings, g_ffx_fsr1_easu_pass_wave64_2d8685ca856679ab3d82d54889435989_SamplerResourceCounts, g_ffx_fsr1_easu_pass_wave64_2d8685ca856679ab3d82d54889435989_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr1_easu_pass_wave64_140795acc7b520bfb43966f003b3e921_size, g_ffx_fsr1_easu_pass_wave64_140795acc7b520bfb43966f003b3e921_data, 1, g_ffx_fsr1_easu_pass_wave64_140795acc7b520bfb43966f003b3e921_CBVResourceNames, g_ffx_fsr1_easu_pass_wave64_140795acc7b520bfb43966f003b3e921_CBVResourceBindings, g_ffx_fsr1_easu_pass_wave64_140795acc7b520bfb43966f003b3e921_CBVResourceCounts, g_ffx_fsr1_easu_pass_wave64_140795acc7b520bfb43966f003b3e921_CBVResourceSets, 1, g_ffx_fsr1_easu_pass_wave64_140795acc7b520bfb43966f003b3e921_TextureSRVResourceNames, g_ffx_fsr1_easu_pass_wave64_140795acc7b520bfb43966f003b3e921_TextureSRVResourceBindings, g_ffx_fsr1_easu_pass_wave64_140795acc7b520bfb43966f003b3e921_TextureSRVResourceCounts, g_ffx_fsr1_easu_pass_wave64_140795acc7b520bfb43966f003b3e921_TextureSRVResourceSets, 1, g_ffx_fsr1_easu_pass_wave64_140795acc7b520bfb43966f003b3e921_TextureUAVResourceNames, g_ffx_fsr1_easu_pass_wave64_140795acc7b520bfb43966f003b3e921_TextureUAVResourceBindings, g_ffx_fsr1_easu_pass_wave64_140795acc7b520bfb43966f003b3e921_TextureUAVResourceCounts, g_ffx_fsr1_easu_pass_wave64_140795acc7b520bfb43966f003b3e921_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr1_easu_pass_wave64_140795acc7b520bfb43966f003b3e921_SamplerResourceNames, g_ffx_fsr1_easu_pass_wave64_140795acc7b520bfb43966f003b3e921_SamplerResourceBindings, g_ffx_fsr1_easu_pass_wave64_140795acc7b520bfb43966f003b3e921_SamplerResourceCounts, g_ffx_fsr1_easu_pass_wave64_140795acc7b520bfb43966f003b3e921_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
};

