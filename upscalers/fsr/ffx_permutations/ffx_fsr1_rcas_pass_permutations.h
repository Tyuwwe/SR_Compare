#include "ffx_fsr1_rcas_pass_34602b8311d5e77d7acfc832570da625.h"
#include "ffx_fsr1_rcas_pass_e447e283fb4ce5ec94bd4e5cb5775368.h"

typedef union ffx_fsr1_rcas_pass_PermutationKey {
    struct {
        uint32_t FFX_FSR1_OPTION_APPLY_RCAS : 1;
        uint32_t FFX_FSR1_OPTION_RCAS_PASSTHROUGH_ALPHA : 1;
        uint32_t FFX_FSR1_OPTION_SRGB_CONVERSIONS : 1;
    };
    uint32_t index;
} ffx_fsr1_rcas_pass_PermutationKey;

typedef struct ffx_fsr1_rcas_pass_PermutationInfo {
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
} ffx_fsr1_rcas_pass_PermutationInfo;

static const uint32_t g_ffx_fsr1_rcas_pass_IndirectionTable[] = {
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
};

static const ffx_fsr1_rcas_pass_PermutationInfo g_ffx_fsr1_rcas_pass_PermutationInfo[] = {
    { g_ffx_fsr1_rcas_pass_34602b8311d5e77d7acfc832570da625_size, g_ffx_fsr1_rcas_pass_34602b8311d5e77d7acfc832570da625_data, 1, g_ffx_fsr1_rcas_pass_34602b8311d5e77d7acfc832570da625_CBVResourceNames, g_ffx_fsr1_rcas_pass_34602b8311d5e77d7acfc832570da625_CBVResourceBindings, g_ffx_fsr1_rcas_pass_34602b8311d5e77d7acfc832570da625_CBVResourceCounts, g_ffx_fsr1_rcas_pass_34602b8311d5e77d7acfc832570da625_CBVResourceSets, 1, g_ffx_fsr1_rcas_pass_34602b8311d5e77d7acfc832570da625_TextureSRVResourceNames, g_ffx_fsr1_rcas_pass_34602b8311d5e77d7acfc832570da625_TextureSRVResourceBindings, g_ffx_fsr1_rcas_pass_34602b8311d5e77d7acfc832570da625_TextureSRVResourceCounts, g_ffx_fsr1_rcas_pass_34602b8311d5e77d7acfc832570da625_TextureSRVResourceSets, 1, g_ffx_fsr1_rcas_pass_34602b8311d5e77d7acfc832570da625_TextureUAVResourceNames, g_ffx_fsr1_rcas_pass_34602b8311d5e77d7acfc832570da625_TextureUAVResourceBindings, g_ffx_fsr1_rcas_pass_34602b8311d5e77d7acfc832570da625_TextureUAVResourceCounts, g_ffx_fsr1_rcas_pass_34602b8311d5e77d7acfc832570da625_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr1_rcas_pass_e447e283fb4ce5ec94bd4e5cb5775368_size, g_ffx_fsr1_rcas_pass_e447e283fb4ce5ec94bd4e5cb5775368_data, 1, g_ffx_fsr1_rcas_pass_e447e283fb4ce5ec94bd4e5cb5775368_CBVResourceNames, g_ffx_fsr1_rcas_pass_e447e283fb4ce5ec94bd4e5cb5775368_CBVResourceBindings, g_ffx_fsr1_rcas_pass_e447e283fb4ce5ec94bd4e5cb5775368_CBVResourceCounts, g_ffx_fsr1_rcas_pass_e447e283fb4ce5ec94bd4e5cb5775368_CBVResourceSets, 1, g_ffx_fsr1_rcas_pass_e447e283fb4ce5ec94bd4e5cb5775368_TextureSRVResourceNames, g_ffx_fsr1_rcas_pass_e447e283fb4ce5ec94bd4e5cb5775368_TextureSRVResourceBindings, g_ffx_fsr1_rcas_pass_e447e283fb4ce5ec94bd4e5cb5775368_TextureSRVResourceCounts, g_ffx_fsr1_rcas_pass_e447e283fb4ce5ec94bd4e5cb5775368_TextureSRVResourceSets, 1, g_ffx_fsr1_rcas_pass_e447e283fb4ce5ec94bd4e5cb5775368_TextureUAVResourceNames, g_ffx_fsr1_rcas_pass_e447e283fb4ce5ec94bd4e5cb5775368_TextureUAVResourceBindings, g_ffx_fsr1_rcas_pass_e447e283fb4ce5ec94bd4e5cb5775368_TextureUAVResourceCounts, g_ffx_fsr1_rcas_pass_e447e283fb4ce5ec94bd4e5cb5775368_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
};

