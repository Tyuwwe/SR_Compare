#include "ffx_frameinterpolation_reconstruct_previous_depth_pass_40a51b287350c8e179ab993e0d8b0af3.h"
#include "ffx_frameinterpolation_reconstruct_previous_depth_pass_fc55203c6e6ae4bc3b47251814a4043d.h"

typedef union ffx_frameinterpolation_reconstruct_previous_depth_pass_PermutationKey {
    struct {
        uint32_t FFX_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH : 1;
        uint32_t FFX_FRAMEINTERPOLATION_OPTION_LOW_RES_MOTION_VECTORS : 1;
        uint32_t FFX_FRAMEINTERPOLATION_OPTION_JITTER_MOTION_VECTORS : 1;
    };
    uint32_t index;
} ffx_frameinterpolation_reconstruct_previous_depth_pass_PermutationKey;

typedef struct ffx_frameinterpolation_reconstruct_previous_depth_pass_PermutationInfo {
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
} ffx_frameinterpolation_reconstruct_previous_depth_pass_PermutationInfo;

static const uint32_t g_ffx_frameinterpolation_reconstruct_previous_depth_pass_IndirectionTable[] = {
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
};

static const ffx_frameinterpolation_reconstruct_previous_depth_pass_PermutationInfo g_ffx_frameinterpolation_reconstruct_previous_depth_pass_PermutationInfo[] = {
    { g_ffx_frameinterpolation_reconstruct_previous_depth_pass_40a51b287350c8e179ab993e0d8b0af3_size, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_40a51b287350c8e179ab993e0d8b0af3_data, 1, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_40a51b287350c8e179ab993e0d8b0af3_CBVResourceNames, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_40a51b287350c8e179ab993e0d8b0af3_CBVResourceBindings, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_40a51b287350c8e179ab993e0d8b0af3_CBVResourceCounts, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_40a51b287350c8e179ab993e0d8b0af3_CBVResourceSets, 3, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_40a51b287350c8e179ab993e0d8b0af3_TextureSRVResourceNames, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_40a51b287350c8e179ab993e0d8b0af3_TextureSRVResourceBindings, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_40a51b287350c8e179ab993e0d8b0af3_TextureSRVResourceCounts, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_40a51b287350c8e179ab993e0d8b0af3_TextureSRVResourceSets, 1, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_40a51b287350c8e179ab993e0d8b0af3_TextureUAVResourceNames, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_40a51b287350c8e179ab993e0d8b0af3_TextureUAVResourceBindings, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_40a51b287350c8e179ab993e0d8b0af3_TextureUAVResourceCounts, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_40a51b287350c8e179ab993e0d8b0af3_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_40a51b287350c8e179ab993e0d8b0af3_SamplerResourceNames, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_40a51b287350c8e179ab993e0d8b0af3_SamplerResourceBindings, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_40a51b287350c8e179ab993e0d8b0af3_SamplerResourceCounts, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_40a51b287350c8e179ab993e0d8b0af3_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_frameinterpolation_reconstruct_previous_depth_pass_fc55203c6e6ae4bc3b47251814a4043d_size, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_fc55203c6e6ae4bc3b47251814a4043d_data, 1, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_fc55203c6e6ae4bc3b47251814a4043d_CBVResourceNames, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_fc55203c6e6ae4bc3b47251814a4043d_CBVResourceBindings, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_fc55203c6e6ae4bc3b47251814a4043d_CBVResourceCounts, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_fc55203c6e6ae4bc3b47251814a4043d_CBVResourceSets, 3, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_fc55203c6e6ae4bc3b47251814a4043d_TextureSRVResourceNames, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_fc55203c6e6ae4bc3b47251814a4043d_TextureSRVResourceBindings, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_fc55203c6e6ae4bc3b47251814a4043d_TextureSRVResourceCounts, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_fc55203c6e6ae4bc3b47251814a4043d_TextureSRVResourceSets, 1, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_fc55203c6e6ae4bc3b47251814a4043d_TextureUAVResourceNames, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_fc55203c6e6ae4bc3b47251814a4043d_TextureUAVResourceBindings, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_fc55203c6e6ae4bc3b47251814a4043d_TextureUAVResourceCounts, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_fc55203c6e6ae4bc3b47251814a4043d_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_fc55203c6e6ae4bc3b47251814a4043d_SamplerResourceNames, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_fc55203c6e6ae4bc3b47251814a4043d_SamplerResourceBindings, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_fc55203c6e6ae4bc3b47251814a4043d_SamplerResourceCounts, g_ffx_frameinterpolation_reconstruct_previous_depth_pass_fc55203c6e6ae4bc3b47251814a4043d_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
};

