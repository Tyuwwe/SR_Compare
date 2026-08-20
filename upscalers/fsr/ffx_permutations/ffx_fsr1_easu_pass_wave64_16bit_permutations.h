#include "ffx_fsr1_easu_pass_wave64_16bit_f3b1d0495006bbf703c827284ccb4a1b.h"
#include "ffx_fsr1_easu_pass_wave64_16bit_3b04a8f1454f008dc4ea186f9fcbc446.h"
#include "ffx_fsr1_easu_pass_wave64_16bit_c0e92167f0d2a0330e58770555c33c41.h"
#include "ffx_fsr1_easu_pass_wave64_16bit_339671eb24d159c72f82ad1b2001c57f.h"

typedef union ffx_fsr1_easu_pass_wave64_16bit_PermutationKey {
    struct {
        uint32_t FFX_FSR1_OPTION_APPLY_RCAS : 1;
        uint32_t FFX_FSR1_OPTION_RCAS_PASSTHROUGH_ALPHA : 1;
        uint32_t FFX_FSR1_OPTION_SRGB_CONVERSIONS : 1;
    };
    uint32_t index;
} ffx_fsr1_easu_pass_wave64_16bit_PermutationKey;

typedef struct ffx_fsr1_easu_pass_wave64_16bit_PermutationInfo {
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
} ffx_fsr1_easu_pass_wave64_16bit_PermutationInfo;

static const uint32_t g_ffx_fsr1_easu_pass_wave64_16bit_IndirectionTable[] = {
    1,
    0,
    1,
    0,
    3,
    2,
    3,
    2,
};

static const ffx_fsr1_easu_pass_wave64_16bit_PermutationInfo g_ffx_fsr1_easu_pass_wave64_16bit_PermutationInfo[] = {
    { g_ffx_fsr1_easu_pass_wave64_16bit_f3b1d0495006bbf703c827284ccb4a1b_size, g_ffx_fsr1_easu_pass_wave64_16bit_f3b1d0495006bbf703c827284ccb4a1b_data, 1, g_ffx_fsr1_easu_pass_wave64_16bit_f3b1d0495006bbf703c827284ccb4a1b_CBVResourceNames, g_ffx_fsr1_easu_pass_wave64_16bit_f3b1d0495006bbf703c827284ccb4a1b_CBVResourceBindings, g_ffx_fsr1_easu_pass_wave64_16bit_f3b1d0495006bbf703c827284ccb4a1b_CBVResourceCounts, g_ffx_fsr1_easu_pass_wave64_16bit_f3b1d0495006bbf703c827284ccb4a1b_CBVResourceSets, 1, g_ffx_fsr1_easu_pass_wave64_16bit_f3b1d0495006bbf703c827284ccb4a1b_TextureSRVResourceNames, g_ffx_fsr1_easu_pass_wave64_16bit_f3b1d0495006bbf703c827284ccb4a1b_TextureSRVResourceBindings, g_ffx_fsr1_easu_pass_wave64_16bit_f3b1d0495006bbf703c827284ccb4a1b_TextureSRVResourceCounts, g_ffx_fsr1_easu_pass_wave64_16bit_f3b1d0495006bbf703c827284ccb4a1b_TextureSRVResourceSets, 1, g_ffx_fsr1_easu_pass_wave64_16bit_f3b1d0495006bbf703c827284ccb4a1b_TextureUAVResourceNames, g_ffx_fsr1_easu_pass_wave64_16bit_f3b1d0495006bbf703c827284ccb4a1b_TextureUAVResourceBindings, g_ffx_fsr1_easu_pass_wave64_16bit_f3b1d0495006bbf703c827284ccb4a1b_TextureUAVResourceCounts, g_ffx_fsr1_easu_pass_wave64_16bit_f3b1d0495006bbf703c827284ccb4a1b_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr1_easu_pass_wave64_16bit_f3b1d0495006bbf703c827284ccb4a1b_SamplerResourceNames, g_ffx_fsr1_easu_pass_wave64_16bit_f3b1d0495006bbf703c827284ccb4a1b_SamplerResourceBindings, g_ffx_fsr1_easu_pass_wave64_16bit_f3b1d0495006bbf703c827284ccb4a1b_SamplerResourceCounts, g_ffx_fsr1_easu_pass_wave64_16bit_f3b1d0495006bbf703c827284ccb4a1b_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr1_easu_pass_wave64_16bit_3b04a8f1454f008dc4ea186f9fcbc446_size, g_ffx_fsr1_easu_pass_wave64_16bit_3b04a8f1454f008dc4ea186f9fcbc446_data, 1, g_ffx_fsr1_easu_pass_wave64_16bit_3b04a8f1454f008dc4ea186f9fcbc446_CBVResourceNames, g_ffx_fsr1_easu_pass_wave64_16bit_3b04a8f1454f008dc4ea186f9fcbc446_CBVResourceBindings, g_ffx_fsr1_easu_pass_wave64_16bit_3b04a8f1454f008dc4ea186f9fcbc446_CBVResourceCounts, g_ffx_fsr1_easu_pass_wave64_16bit_3b04a8f1454f008dc4ea186f9fcbc446_CBVResourceSets, 1, g_ffx_fsr1_easu_pass_wave64_16bit_3b04a8f1454f008dc4ea186f9fcbc446_TextureSRVResourceNames, g_ffx_fsr1_easu_pass_wave64_16bit_3b04a8f1454f008dc4ea186f9fcbc446_TextureSRVResourceBindings, g_ffx_fsr1_easu_pass_wave64_16bit_3b04a8f1454f008dc4ea186f9fcbc446_TextureSRVResourceCounts, g_ffx_fsr1_easu_pass_wave64_16bit_3b04a8f1454f008dc4ea186f9fcbc446_TextureSRVResourceSets, 1, g_ffx_fsr1_easu_pass_wave64_16bit_3b04a8f1454f008dc4ea186f9fcbc446_TextureUAVResourceNames, g_ffx_fsr1_easu_pass_wave64_16bit_3b04a8f1454f008dc4ea186f9fcbc446_TextureUAVResourceBindings, g_ffx_fsr1_easu_pass_wave64_16bit_3b04a8f1454f008dc4ea186f9fcbc446_TextureUAVResourceCounts, g_ffx_fsr1_easu_pass_wave64_16bit_3b04a8f1454f008dc4ea186f9fcbc446_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr1_easu_pass_wave64_16bit_3b04a8f1454f008dc4ea186f9fcbc446_SamplerResourceNames, g_ffx_fsr1_easu_pass_wave64_16bit_3b04a8f1454f008dc4ea186f9fcbc446_SamplerResourceBindings, g_ffx_fsr1_easu_pass_wave64_16bit_3b04a8f1454f008dc4ea186f9fcbc446_SamplerResourceCounts, g_ffx_fsr1_easu_pass_wave64_16bit_3b04a8f1454f008dc4ea186f9fcbc446_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr1_easu_pass_wave64_16bit_c0e92167f0d2a0330e58770555c33c41_size, g_ffx_fsr1_easu_pass_wave64_16bit_c0e92167f0d2a0330e58770555c33c41_data, 1, g_ffx_fsr1_easu_pass_wave64_16bit_c0e92167f0d2a0330e58770555c33c41_CBVResourceNames, g_ffx_fsr1_easu_pass_wave64_16bit_c0e92167f0d2a0330e58770555c33c41_CBVResourceBindings, g_ffx_fsr1_easu_pass_wave64_16bit_c0e92167f0d2a0330e58770555c33c41_CBVResourceCounts, g_ffx_fsr1_easu_pass_wave64_16bit_c0e92167f0d2a0330e58770555c33c41_CBVResourceSets, 1, g_ffx_fsr1_easu_pass_wave64_16bit_c0e92167f0d2a0330e58770555c33c41_TextureSRVResourceNames, g_ffx_fsr1_easu_pass_wave64_16bit_c0e92167f0d2a0330e58770555c33c41_TextureSRVResourceBindings, g_ffx_fsr1_easu_pass_wave64_16bit_c0e92167f0d2a0330e58770555c33c41_TextureSRVResourceCounts, g_ffx_fsr1_easu_pass_wave64_16bit_c0e92167f0d2a0330e58770555c33c41_TextureSRVResourceSets, 1, g_ffx_fsr1_easu_pass_wave64_16bit_c0e92167f0d2a0330e58770555c33c41_TextureUAVResourceNames, g_ffx_fsr1_easu_pass_wave64_16bit_c0e92167f0d2a0330e58770555c33c41_TextureUAVResourceBindings, g_ffx_fsr1_easu_pass_wave64_16bit_c0e92167f0d2a0330e58770555c33c41_TextureUAVResourceCounts, g_ffx_fsr1_easu_pass_wave64_16bit_c0e92167f0d2a0330e58770555c33c41_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr1_easu_pass_wave64_16bit_c0e92167f0d2a0330e58770555c33c41_SamplerResourceNames, g_ffx_fsr1_easu_pass_wave64_16bit_c0e92167f0d2a0330e58770555c33c41_SamplerResourceBindings, g_ffx_fsr1_easu_pass_wave64_16bit_c0e92167f0d2a0330e58770555c33c41_SamplerResourceCounts, g_ffx_fsr1_easu_pass_wave64_16bit_c0e92167f0d2a0330e58770555c33c41_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr1_easu_pass_wave64_16bit_339671eb24d159c72f82ad1b2001c57f_size, g_ffx_fsr1_easu_pass_wave64_16bit_339671eb24d159c72f82ad1b2001c57f_data, 1, g_ffx_fsr1_easu_pass_wave64_16bit_339671eb24d159c72f82ad1b2001c57f_CBVResourceNames, g_ffx_fsr1_easu_pass_wave64_16bit_339671eb24d159c72f82ad1b2001c57f_CBVResourceBindings, g_ffx_fsr1_easu_pass_wave64_16bit_339671eb24d159c72f82ad1b2001c57f_CBVResourceCounts, g_ffx_fsr1_easu_pass_wave64_16bit_339671eb24d159c72f82ad1b2001c57f_CBVResourceSets, 1, g_ffx_fsr1_easu_pass_wave64_16bit_339671eb24d159c72f82ad1b2001c57f_TextureSRVResourceNames, g_ffx_fsr1_easu_pass_wave64_16bit_339671eb24d159c72f82ad1b2001c57f_TextureSRVResourceBindings, g_ffx_fsr1_easu_pass_wave64_16bit_339671eb24d159c72f82ad1b2001c57f_TextureSRVResourceCounts, g_ffx_fsr1_easu_pass_wave64_16bit_339671eb24d159c72f82ad1b2001c57f_TextureSRVResourceSets, 1, g_ffx_fsr1_easu_pass_wave64_16bit_339671eb24d159c72f82ad1b2001c57f_TextureUAVResourceNames, g_ffx_fsr1_easu_pass_wave64_16bit_339671eb24d159c72f82ad1b2001c57f_TextureUAVResourceBindings, g_ffx_fsr1_easu_pass_wave64_16bit_339671eb24d159c72f82ad1b2001c57f_TextureUAVResourceCounts, g_ffx_fsr1_easu_pass_wave64_16bit_339671eb24d159c72f82ad1b2001c57f_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr1_easu_pass_wave64_16bit_339671eb24d159c72f82ad1b2001c57f_SamplerResourceNames, g_ffx_fsr1_easu_pass_wave64_16bit_339671eb24d159c72f82ad1b2001c57f_SamplerResourceBindings, g_ffx_fsr1_easu_pass_wave64_16bit_339671eb24d159c72f82ad1b2001c57f_SamplerResourceCounts, g_ffx_fsr1_easu_pass_wave64_16bit_339671eb24d159c72f82ad1b2001c57f_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
};

