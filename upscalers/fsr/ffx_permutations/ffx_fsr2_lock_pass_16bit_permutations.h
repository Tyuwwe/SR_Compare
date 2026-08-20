#include "ffx_fsr2_lock_pass_16bit_0c87f93a28f340b0df03c4084fbb8082.h"
#include "ffx_fsr2_lock_pass_16bit_de0d10c520af48900f5ade39a2a356d5.h"

typedef union ffx_fsr2_lock_pass_16bit_PermutationKey {
    struct {
        uint32_t FFX_FSR2_OPTION_JITTERED_MOTION_VECTORS : 1;
        uint32_t FFX_FSR2_OPTION_INVERTED_DEPTH : 1;
        uint32_t FFX_FSR2_OPTION_REPROJECT_USE_LANCZOS_TYPE : 1;
        uint32_t FFX_FSR2_OPTION_HDR_COLOR_INPUT : 1;
        uint32_t FFX_FSR2_OPTION_LOW_RESOLUTION_MOTION_VECTORS : 1;
        uint32_t FFX_FSR2_OPTION_APPLY_SHARPENING : 1;
    };
    uint32_t index;
} ffx_fsr2_lock_pass_16bit_PermutationKey;

typedef struct ffx_fsr2_lock_pass_16bit_PermutationInfo {
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
} ffx_fsr2_lock_pass_16bit_PermutationInfo;

static const uint32_t g_ffx_fsr2_lock_pass_16bit_IndirectionTable[] = {
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
};

static const ffx_fsr2_lock_pass_16bit_PermutationInfo g_ffx_fsr2_lock_pass_16bit_PermutationInfo[] = {
    { g_ffx_fsr2_lock_pass_16bit_0c87f93a28f340b0df03c4084fbb8082_size, g_ffx_fsr2_lock_pass_16bit_0c87f93a28f340b0df03c4084fbb8082_data, 1, g_ffx_fsr2_lock_pass_16bit_0c87f93a28f340b0df03c4084fbb8082_CBVResourceNames, g_ffx_fsr2_lock_pass_16bit_0c87f93a28f340b0df03c4084fbb8082_CBVResourceBindings, g_ffx_fsr2_lock_pass_16bit_0c87f93a28f340b0df03c4084fbb8082_CBVResourceCounts, g_ffx_fsr2_lock_pass_16bit_0c87f93a28f340b0df03c4084fbb8082_CBVResourceSets, 1, g_ffx_fsr2_lock_pass_16bit_0c87f93a28f340b0df03c4084fbb8082_TextureSRVResourceNames, g_ffx_fsr2_lock_pass_16bit_0c87f93a28f340b0df03c4084fbb8082_TextureSRVResourceBindings, g_ffx_fsr2_lock_pass_16bit_0c87f93a28f340b0df03c4084fbb8082_TextureSRVResourceCounts, g_ffx_fsr2_lock_pass_16bit_0c87f93a28f340b0df03c4084fbb8082_TextureSRVResourceSets, 2, g_ffx_fsr2_lock_pass_16bit_0c87f93a28f340b0df03c4084fbb8082_TextureUAVResourceNames, g_ffx_fsr2_lock_pass_16bit_0c87f93a28f340b0df03c4084fbb8082_TextureUAVResourceBindings, g_ffx_fsr2_lock_pass_16bit_0c87f93a28f340b0df03c4084fbb8082_TextureUAVResourceCounts, g_ffx_fsr2_lock_pass_16bit_0c87f93a28f340b0df03c4084fbb8082_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr2_lock_pass_16bit_de0d10c520af48900f5ade39a2a356d5_size, g_ffx_fsr2_lock_pass_16bit_de0d10c520af48900f5ade39a2a356d5_data, 1, g_ffx_fsr2_lock_pass_16bit_de0d10c520af48900f5ade39a2a356d5_CBVResourceNames, g_ffx_fsr2_lock_pass_16bit_de0d10c520af48900f5ade39a2a356d5_CBVResourceBindings, g_ffx_fsr2_lock_pass_16bit_de0d10c520af48900f5ade39a2a356d5_CBVResourceCounts, g_ffx_fsr2_lock_pass_16bit_de0d10c520af48900f5ade39a2a356d5_CBVResourceSets, 1, g_ffx_fsr2_lock_pass_16bit_de0d10c520af48900f5ade39a2a356d5_TextureSRVResourceNames, g_ffx_fsr2_lock_pass_16bit_de0d10c520af48900f5ade39a2a356d5_TextureSRVResourceBindings, g_ffx_fsr2_lock_pass_16bit_de0d10c520af48900f5ade39a2a356d5_TextureSRVResourceCounts, g_ffx_fsr2_lock_pass_16bit_de0d10c520af48900f5ade39a2a356d5_TextureSRVResourceSets, 2, g_ffx_fsr2_lock_pass_16bit_de0d10c520af48900f5ade39a2a356d5_TextureUAVResourceNames, g_ffx_fsr2_lock_pass_16bit_de0d10c520af48900f5ade39a2a356d5_TextureUAVResourceBindings, g_ffx_fsr2_lock_pass_16bit_de0d10c520af48900f5ade39a2a356d5_TextureUAVResourceCounts, g_ffx_fsr2_lock_pass_16bit_de0d10c520af48900f5ade39a2a356d5_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
};

