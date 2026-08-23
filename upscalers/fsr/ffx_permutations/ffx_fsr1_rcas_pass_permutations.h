#include "ffx_fsr1_rcas_pass_1621451aaefed13f2a8493ca308aa1ad.h"
#include "ffx_fsr1_rcas_pass_6cfbf7d0346d3759e0e30733319707ed.h"

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
    { g_ffx_fsr1_rcas_pass_1621451aaefed13f2a8493ca308aa1ad_size, g_ffx_fsr1_rcas_pass_1621451aaefed13f2a8493ca308aa1ad_data, 1, g_ffx_fsr1_rcas_pass_1621451aaefed13f2a8493ca308aa1ad_CBVResourceNames, g_ffx_fsr1_rcas_pass_1621451aaefed13f2a8493ca308aa1ad_CBVResourceBindings, g_ffx_fsr1_rcas_pass_1621451aaefed13f2a8493ca308aa1ad_CBVResourceCounts, g_ffx_fsr1_rcas_pass_1621451aaefed13f2a8493ca308aa1ad_CBVResourceSets, 1, g_ffx_fsr1_rcas_pass_1621451aaefed13f2a8493ca308aa1ad_TextureSRVResourceNames, g_ffx_fsr1_rcas_pass_1621451aaefed13f2a8493ca308aa1ad_TextureSRVResourceBindings, g_ffx_fsr1_rcas_pass_1621451aaefed13f2a8493ca308aa1ad_TextureSRVResourceCounts, g_ffx_fsr1_rcas_pass_1621451aaefed13f2a8493ca308aa1ad_TextureSRVResourceSets, 1, g_ffx_fsr1_rcas_pass_1621451aaefed13f2a8493ca308aa1ad_TextureUAVResourceNames, g_ffx_fsr1_rcas_pass_1621451aaefed13f2a8493ca308aa1ad_TextureUAVResourceBindings, g_ffx_fsr1_rcas_pass_1621451aaefed13f2a8493ca308aa1ad_TextureUAVResourceCounts, g_ffx_fsr1_rcas_pass_1621451aaefed13f2a8493ca308aa1ad_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr1_rcas_pass_6cfbf7d0346d3759e0e30733319707ed_size, g_ffx_fsr1_rcas_pass_6cfbf7d0346d3759e0e30733319707ed_data, 1, g_ffx_fsr1_rcas_pass_6cfbf7d0346d3759e0e30733319707ed_CBVResourceNames, g_ffx_fsr1_rcas_pass_6cfbf7d0346d3759e0e30733319707ed_CBVResourceBindings, g_ffx_fsr1_rcas_pass_6cfbf7d0346d3759e0e30733319707ed_CBVResourceCounts, g_ffx_fsr1_rcas_pass_6cfbf7d0346d3759e0e30733319707ed_CBVResourceSets, 1, g_ffx_fsr1_rcas_pass_6cfbf7d0346d3759e0e30733319707ed_TextureSRVResourceNames, g_ffx_fsr1_rcas_pass_6cfbf7d0346d3759e0e30733319707ed_TextureSRVResourceBindings, g_ffx_fsr1_rcas_pass_6cfbf7d0346d3759e0e30733319707ed_TextureSRVResourceCounts, g_ffx_fsr1_rcas_pass_6cfbf7d0346d3759e0e30733319707ed_TextureSRVResourceSets, 1, g_ffx_fsr1_rcas_pass_6cfbf7d0346d3759e0e30733319707ed_TextureUAVResourceNames, g_ffx_fsr1_rcas_pass_6cfbf7d0346d3759e0e30733319707ed_TextureUAVResourceBindings, g_ffx_fsr1_rcas_pass_6cfbf7d0346d3759e0e30733319707ed_TextureUAVResourceCounts, g_ffx_fsr1_rcas_pass_6cfbf7d0346d3759e0e30733319707ed_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
};

