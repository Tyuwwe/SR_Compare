#include "ffx_fsr2_tcr_autogen_pass_wave64_6f7d5c9125035ff61532027f60faf2f6.h"
#include "ffx_fsr2_tcr_autogen_pass_wave64_6c9802e77f283daa7665ce50565bdb5f.h"

typedef union ffx_fsr2_tcr_autogen_pass_wave64_PermutationKey {
    struct {
        uint32_t FFX_FSR2_OPTION_JITTERED_MOTION_VECTORS : 1;
        uint32_t FFX_FSR2_OPTION_INVERTED_DEPTH : 1;
        uint32_t FFX_FSR2_OPTION_REPROJECT_USE_LANCZOS_TYPE : 1;
        uint32_t FFX_FSR2_OPTION_HDR_COLOR_INPUT : 1;
        uint32_t FFX_FSR2_OPTION_LOW_RESOLUTION_MOTION_VECTORS : 1;
        uint32_t FFX_FSR2_OPTION_APPLY_SHARPENING : 1;
    };
    uint32_t index;
} ffx_fsr2_tcr_autogen_pass_wave64_PermutationKey;

typedef struct ffx_fsr2_tcr_autogen_pass_wave64_PermutationInfo {
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
} ffx_fsr2_tcr_autogen_pass_wave64_PermutationInfo;

static const uint32_t g_ffx_fsr2_tcr_autogen_pass_wave64_IndirectionTable[] = {
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
};

static const ffx_fsr2_tcr_autogen_pass_wave64_PermutationInfo g_ffx_fsr2_tcr_autogen_pass_wave64_PermutationInfo[] = {
    { g_ffx_fsr2_tcr_autogen_pass_wave64_6f7d5c9125035ff61532027f60faf2f6_size, g_ffx_fsr2_tcr_autogen_pass_wave64_6f7d5c9125035ff61532027f60faf2f6_data, 2, g_ffx_fsr2_tcr_autogen_pass_wave64_6f7d5c9125035ff61532027f60faf2f6_CBVResourceNames, g_ffx_fsr2_tcr_autogen_pass_wave64_6f7d5c9125035ff61532027f60faf2f6_CBVResourceBindings, g_ffx_fsr2_tcr_autogen_pass_wave64_6f7d5c9125035ff61532027f60faf2f6_CBVResourceCounts, g_ffx_fsr2_tcr_autogen_pass_wave64_6f7d5c9125035ff61532027f60faf2f6_CBVResourceSets, 7, g_ffx_fsr2_tcr_autogen_pass_wave64_6f7d5c9125035ff61532027f60faf2f6_TextureSRVResourceNames, g_ffx_fsr2_tcr_autogen_pass_wave64_6f7d5c9125035ff61532027f60faf2f6_TextureSRVResourceBindings, g_ffx_fsr2_tcr_autogen_pass_wave64_6f7d5c9125035ff61532027f60faf2f6_TextureSRVResourceCounts, g_ffx_fsr2_tcr_autogen_pass_wave64_6f7d5c9125035ff61532027f60faf2f6_TextureSRVResourceSets, 4, g_ffx_fsr2_tcr_autogen_pass_wave64_6f7d5c9125035ff61532027f60faf2f6_TextureUAVResourceNames, g_ffx_fsr2_tcr_autogen_pass_wave64_6f7d5c9125035ff61532027f60faf2f6_TextureUAVResourceBindings, g_ffx_fsr2_tcr_autogen_pass_wave64_6f7d5c9125035ff61532027f60faf2f6_TextureUAVResourceCounts, g_ffx_fsr2_tcr_autogen_pass_wave64_6f7d5c9125035ff61532027f60faf2f6_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr2_tcr_autogen_pass_wave64_6c9802e77f283daa7665ce50565bdb5f_size, g_ffx_fsr2_tcr_autogen_pass_wave64_6c9802e77f283daa7665ce50565bdb5f_data, 2, g_ffx_fsr2_tcr_autogen_pass_wave64_6c9802e77f283daa7665ce50565bdb5f_CBVResourceNames, g_ffx_fsr2_tcr_autogen_pass_wave64_6c9802e77f283daa7665ce50565bdb5f_CBVResourceBindings, g_ffx_fsr2_tcr_autogen_pass_wave64_6c9802e77f283daa7665ce50565bdb5f_CBVResourceCounts, g_ffx_fsr2_tcr_autogen_pass_wave64_6c9802e77f283daa7665ce50565bdb5f_CBVResourceSets, 7, g_ffx_fsr2_tcr_autogen_pass_wave64_6c9802e77f283daa7665ce50565bdb5f_TextureSRVResourceNames, g_ffx_fsr2_tcr_autogen_pass_wave64_6c9802e77f283daa7665ce50565bdb5f_TextureSRVResourceBindings, g_ffx_fsr2_tcr_autogen_pass_wave64_6c9802e77f283daa7665ce50565bdb5f_TextureSRVResourceCounts, g_ffx_fsr2_tcr_autogen_pass_wave64_6c9802e77f283daa7665ce50565bdb5f_TextureSRVResourceSets, 4, g_ffx_fsr2_tcr_autogen_pass_wave64_6c9802e77f283daa7665ce50565bdb5f_TextureUAVResourceNames, g_ffx_fsr2_tcr_autogen_pass_wave64_6c9802e77f283daa7665ce50565bdb5f_TextureUAVResourceBindings, g_ffx_fsr2_tcr_autogen_pass_wave64_6c9802e77f283daa7665ce50565bdb5f_TextureUAVResourceCounts, g_ffx_fsr2_tcr_autogen_pass_wave64_6c9802e77f283daa7665ce50565bdb5f_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
};

