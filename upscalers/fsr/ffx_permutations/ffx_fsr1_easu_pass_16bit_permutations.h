#include "ffx_fsr1_easu_pass_16bit_7121b998ce71162ca5548e095017e55a.h"
#include "ffx_fsr1_easu_pass_16bit_39f27c3bff734346bdd995f4d6893563.h"
#include "ffx_fsr1_easu_pass_16bit_b70725536b1b420e175f26b9323dbbfe.h"
#include "ffx_fsr1_easu_pass_16bit_713031a267493fb92af22bc0d778cf46.h"

typedef union ffx_fsr1_easu_pass_16bit_PermutationKey {
    struct {
        uint32_t FFX_FSR1_OPTION_APPLY_RCAS : 1;
        uint32_t FFX_FSR1_OPTION_RCAS_PASSTHROUGH_ALPHA : 1;
        uint32_t FFX_FSR1_OPTION_SRGB_CONVERSIONS : 1;
    };
    uint32_t index;
} ffx_fsr1_easu_pass_16bit_PermutationKey;

typedef struct ffx_fsr1_easu_pass_16bit_PermutationInfo {
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
} ffx_fsr1_easu_pass_16bit_PermutationInfo;

static const uint32_t g_ffx_fsr1_easu_pass_16bit_IndirectionTable[] = {
    2,
    1,
    2,
    1,
    3,
    0,
    3,
    0,
};

static const ffx_fsr1_easu_pass_16bit_PermutationInfo g_ffx_fsr1_easu_pass_16bit_PermutationInfo[] = {
    { g_ffx_fsr1_easu_pass_16bit_7121b998ce71162ca5548e095017e55a_size, g_ffx_fsr1_easu_pass_16bit_7121b998ce71162ca5548e095017e55a_data, 1, g_ffx_fsr1_easu_pass_16bit_7121b998ce71162ca5548e095017e55a_CBVResourceNames, g_ffx_fsr1_easu_pass_16bit_7121b998ce71162ca5548e095017e55a_CBVResourceBindings, g_ffx_fsr1_easu_pass_16bit_7121b998ce71162ca5548e095017e55a_CBVResourceCounts, g_ffx_fsr1_easu_pass_16bit_7121b998ce71162ca5548e095017e55a_CBVResourceSets, 1, g_ffx_fsr1_easu_pass_16bit_7121b998ce71162ca5548e095017e55a_TextureSRVResourceNames, g_ffx_fsr1_easu_pass_16bit_7121b998ce71162ca5548e095017e55a_TextureSRVResourceBindings, g_ffx_fsr1_easu_pass_16bit_7121b998ce71162ca5548e095017e55a_TextureSRVResourceCounts, g_ffx_fsr1_easu_pass_16bit_7121b998ce71162ca5548e095017e55a_TextureSRVResourceSets, 1, g_ffx_fsr1_easu_pass_16bit_7121b998ce71162ca5548e095017e55a_TextureUAVResourceNames, g_ffx_fsr1_easu_pass_16bit_7121b998ce71162ca5548e095017e55a_TextureUAVResourceBindings, g_ffx_fsr1_easu_pass_16bit_7121b998ce71162ca5548e095017e55a_TextureUAVResourceCounts, g_ffx_fsr1_easu_pass_16bit_7121b998ce71162ca5548e095017e55a_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr1_easu_pass_16bit_7121b998ce71162ca5548e095017e55a_SamplerResourceNames, g_ffx_fsr1_easu_pass_16bit_7121b998ce71162ca5548e095017e55a_SamplerResourceBindings, g_ffx_fsr1_easu_pass_16bit_7121b998ce71162ca5548e095017e55a_SamplerResourceCounts, g_ffx_fsr1_easu_pass_16bit_7121b998ce71162ca5548e095017e55a_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr1_easu_pass_16bit_39f27c3bff734346bdd995f4d6893563_size, g_ffx_fsr1_easu_pass_16bit_39f27c3bff734346bdd995f4d6893563_data, 1, g_ffx_fsr1_easu_pass_16bit_39f27c3bff734346bdd995f4d6893563_CBVResourceNames, g_ffx_fsr1_easu_pass_16bit_39f27c3bff734346bdd995f4d6893563_CBVResourceBindings, g_ffx_fsr1_easu_pass_16bit_39f27c3bff734346bdd995f4d6893563_CBVResourceCounts, g_ffx_fsr1_easu_pass_16bit_39f27c3bff734346bdd995f4d6893563_CBVResourceSets, 1, g_ffx_fsr1_easu_pass_16bit_39f27c3bff734346bdd995f4d6893563_TextureSRVResourceNames, g_ffx_fsr1_easu_pass_16bit_39f27c3bff734346bdd995f4d6893563_TextureSRVResourceBindings, g_ffx_fsr1_easu_pass_16bit_39f27c3bff734346bdd995f4d6893563_TextureSRVResourceCounts, g_ffx_fsr1_easu_pass_16bit_39f27c3bff734346bdd995f4d6893563_TextureSRVResourceSets, 1, g_ffx_fsr1_easu_pass_16bit_39f27c3bff734346bdd995f4d6893563_TextureUAVResourceNames, g_ffx_fsr1_easu_pass_16bit_39f27c3bff734346bdd995f4d6893563_TextureUAVResourceBindings, g_ffx_fsr1_easu_pass_16bit_39f27c3bff734346bdd995f4d6893563_TextureUAVResourceCounts, g_ffx_fsr1_easu_pass_16bit_39f27c3bff734346bdd995f4d6893563_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr1_easu_pass_16bit_39f27c3bff734346bdd995f4d6893563_SamplerResourceNames, g_ffx_fsr1_easu_pass_16bit_39f27c3bff734346bdd995f4d6893563_SamplerResourceBindings, g_ffx_fsr1_easu_pass_16bit_39f27c3bff734346bdd995f4d6893563_SamplerResourceCounts, g_ffx_fsr1_easu_pass_16bit_39f27c3bff734346bdd995f4d6893563_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr1_easu_pass_16bit_b70725536b1b420e175f26b9323dbbfe_size, g_ffx_fsr1_easu_pass_16bit_b70725536b1b420e175f26b9323dbbfe_data, 1, g_ffx_fsr1_easu_pass_16bit_b70725536b1b420e175f26b9323dbbfe_CBVResourceNames, g_ffx_fsr1_easu_pass_16bit_b70725536b1b420e175f26b9323dbbfe_CBVResourceBindings, g_ffx_fsr1_easu_pass_16bit_b70725536b1b420e175f26b9323dbbfe_CBVResourceCounts, g_ffx_fsr1_easu_pass_16bit_b70725536b1b420e175f26b9323dbbfe_CBVResourceSets, 1, g_ffx_fsr1_easu_pass_16bit_b70725536b1b420e175f26b9323dbbfe_TextureSRVResourceNames, g_ffx_fsr1_easu_pass_16bit_b70725536b1b420e175f26b9323dbbfe_TextureSRVResourceBindings, g_ffx_fsr1_easu_pass_16bit_b70725536b1b420e175f26b9323dbbfe_TextureSRVResourceCounts, g_ffx_fsr1_easu_pass_16bit_b70725536b1b420e175f26b9323dbbfe_TextureSRVResourceSets, 1, g_ffx_fsr1_easu_pass_16bit_b70725536b1b420e175f26b9323dbbfe_TextureUAVResourceNames, g_ffx_fsr1_easu_pass_16bit_b70725536b1b420e175f26b9323dbbfe_TextureUAVResourceBindings, g_ffx_fsr1_easu_pass_16bit_b70725536b1b420e175f26b9323dbbfe_TextureUAVResourceCounts, g_ffx_fsr1_easu_pass_16bit_b70725536b1b420e175f26b9323dbbfe_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr1_easu_pass_16bit_b70725536b1b420e175f26b9323dbbfe_SamplerResourceNames, g_ffx_fsr1_easu_pass_16bit_b70725536b1b420e175f26b9323dbbfe_SamplerResourceBindings, g_ffx_fsr1_easu_pass_16bit_b70725536b1b420e175f26b9323dbbfe_SamplerResourceCounts, g_ffx_fsr1_easu_pass_16bit_b70725536b1b420e175f26b9323dbbfe_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr1_easu_pass_16bit_713031a267493fb92af22bc0d778cf46_size, g_ffx_fsr1_easu_pass_16bit_713031a267493fb92af22bc0d778cf46_data, 1, g_ffx_fsr1_easu_pass_16bit_713031a267493fb92af22bc0d778cf46_CBVResourceNames, g_ffx_fsr1_easu_pass_16bit_713031a267493fb92af22bc0d778cf46_CBVResourceBindings, g_ffx_fsr1_easu_pass_16bit_713031a267493fb92af22bc0d778cf46_CBVResourceCounts, g_ffx_fsr1_easu_pass_16bit_713031a267493fb92af22bc0d778cf46_CBVResourceSets, 1, g_ffx_fsr1_easu_pass_16bit_713031a267493fb92af22bc0d778cf46_TextureSRVResourceNames, g_ffx_fsr1_easu_pass_16bit_713031a267493fb92af22bc0d778cf46_TextureSRVResourceBindings, g_ffx_fsr1_easu_pass_16bit_713031a267493fb92af22bc0d778cf46_TextureSRVResourceCounts, g_ffx_fsr1_easu_pass_16bit_713031a267493fb92af22bc0d778cf46_TextureSRVResourceSets, 1, g_ffx_fsr1_easu_pass_16bit_713031a267493fb92af22bc0d778cf46_TextureUAVResourceNames, g_ffx_fsr1_easu_pass_16bit_713031a267493fb92af22bc0d778cf46_TextureUAVResourceBindings, g_ffx_fsr1_easu_pass_16bit_713031a267493fb92af22bc0d778cf46_TextureUAVResourceCounts, g_ffx_fsr1_easu_pass_16bit_713031a267493fb92af22bc0d778cf46_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr1_easu_pass_16bit_713031a267493fb92af22bc0d778cf46_SamplerResourceNames, g_ffx_fsr1_easu_pass_16bit_713031a267493fb92af22bc0d778cf46_SamplerResourceBindings, g_ffx_fsr1_easu_pass_16bit_713031a267493fb92af22bc0d778cf46_SamplerResourceCounts, g_ffx_fsr1_easu_pass_16bit_713031a267493fb92af22bc0d778cf46_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
};

