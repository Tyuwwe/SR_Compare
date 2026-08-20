#include "ffx_fsr1_easu_pass_cfcc2e50985a6669f5aaff712c5061f0.h"
#include "ffx_fsr1_easu_pass_486bd8404d1d9cb86b159774ae102ad3.h"
#include "ffx_fsr1_easu_pass_0c216412a066e680c4065e6790a2df3c.h"
#include "ffx_fsr1_easu_pass_81fd7740d976e7a5248c157b142f74d5.h"

typedef union ffx_fsr1_easu_pass_PermutationKey {
    struct {
        uint32_t FFX_FSR1_OPTION_APPLY_RCAS : 1;
        uint32_t FFX_FSR1_OPTION_RCAS_PASSTHROUGH_ALPHA : 1;
        uint32_t FFX_FSR1_OPTION_SRGB_CONVERSIONS : 1;
    };
    uint32_t index;
} ffx_fsr1_easu_pass_PermutationKey;

typedef struct ffx_fsr1_easu_pass_PermutationInfo {
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
} ffx_fsr1_easu_pass_PermutationInfo;

static const uint32_t g_ffx_fsr1_easu_pass_IndirectionTable[] = {
    3,
    1,
    3,
    1,
    2,
    0,
    2,
    0,
};

static const ffx_fsr1_easu_pass_PermutationInfo g_ffx_fsr1_easu_pass_PermutationInfo[] = {
    { g_ffx_fsr1_easu_pass_cfcc2e50985a6669f5aaff712c5061f0_size, g_ffx_fsr1_easu_pass_cfcc2e50985a6669f5aaff712c5061f0_data, 1, g_ffx_fsr1_easu_pass_cfcc2e50985a6669f5aaff712c5061f0_CBVResourceNames, g_ffx_fsr1_easu_pass_cfcc2e50985a6669f5aaff712c5061f0_CBVResourceBindings, g_ffx_fsr1_easu_pass_cfcc2e50985a6669f5aaff712c5061f0_CBVResourceCounts, g_ffx_fsr1_easu_pass_cfcc2e50985a6669f5aaff712c5061f0_CBVResourceSets, 1, g_ffx_fsr1_easu_pass_cfcc2e50985a6669f5aaff712c5061f0_TextureSRVResourceNames, g_ffx_fsr1_easu_pass_cfcc2e50985a6669f5aaff712c5061f0_TextureSRVResourceBindings, g_ffx_fsr1_easu_pass_cfcc2e50985a6669f5aaff712c5061f0_TextureSRVResourceCounts, g_ffx_fsr1_easu_pass_cfcc2e50985a6669f5aaff712c5061f0_TextureSRVResourceSets, 1, g_ffx_fsr1_easu_pass_cfcc2e50985a6669f5aaff712c5061f0_TextureUAVResourceNames, g_ffx_fsr1_easu_pass_cfcc2e50985a6669f5aaff712c5061f0_TextureUAVResourceBindings, g_ffx_fsr1_easu_pass_cfcc2e50985a6669f5aaff712c5061f0_TextureUAVResourceCounts, g_ffx_fsr1_easu_pass_cfcc2e50985a6669f5aaff712c5061f0_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr1_easu_pass_cfcc2e50985a6669f5aaff712c5061f0_SamplerResourceNames, g_ffx_fsr1_easu_pass_cfcc2e50985a6669f5aaff712c5061f0_SamplerResourceBindings, g_ffx_fsr1_easu_pass_cfcc2e50985a6669f5aaff712c5061f0_SamplerResourceCounts, g_ffx_fsr1_easu_pass_cfcc2e50985a6669f5aaff712c5061f0_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr1_easu_pass_486bd8404d1d9cb86b159774ae102ad3_size, g_ffx_fsr1_easu_pass_486bd8404d1d9cb86b159774ae102ad3_data, 1, g_ffx_fsr1_easu_pass_486bd8404d1d9cb86b159774ae102ad3_CBVResourceNames, g_ffx_fsr1_easu_pass_486bd8404d1d9cb86b159774ae102ad3_CBVResourceBindings, g_ffx_fsr1_easu_pass_486bd8404d1d9cb86b159774ae102ad3_CBVResourceCounts, g_ffx_fsr1_easu_pass_486bd8404d1d9cb86b159774ae102ad3_CBVResourceSets, 1, g_ffx_fsr1_easu_pass_486bd8404d1d9cb86b159774ae102ad3_TextureSRVResourceNames, g_ffx_fsr1_easu_pass_486bd8404d1d9cb86b159774ae102ad3_TextureSRVResourceBindings, g_ffx_fsr1_easu_pass_486bd8404d1d9cb86b159774ae102ad3_TextureSRVResourceCounts, g_ffx_fsr1_easu_pass_486bd8404d1d9cb86b159774ae102ad3_TextureSRVResourceSets, 1, g_ffx_fsr1_easu_pass_486bd8404d1d9cb86b159774ae102ad3_TextureUAVResourceNames, g_ffx_fsr1_easu_pass_486bd8404d1d9cb86b159774ae102ad3_TextureUAVResourceBindings, g_ffx_fsr1_easu_pass_486bd8404d1d9cb86b159774ae102ad3_TextureUAVResourceCounts, g_ffx_fsr1_easu_pass_486bd8404d1d9cb86b159774ae102ad3_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr1_easu_pass_486bd8404d1d9cb86b159774ae102ad3_SamplerResourceNames, g_ffx_fsr1_easu_pass_486bd8404d1d9cb86b159774ae102ad3_SamplerResourceBindings, g_ffx_fsr1_easu_pass_486bd8404d1d9cb86b159774ae102ad3_SamplerResourceCounts, g_ffx_fsr1_easu_pass_486bd8404d1d9cb86b159774ae102ad3_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr1_easu_pass_0c216412a066e680c4065e6790a2df3c_size, g_ffx_fsr1_easu_pass_0c216412a066e680c4065e6790a2df3c_data, 1, g_ffx_fsr1_easu_pass_0c216412a066e680c4065e6790a2df3c_CBVResourceNames, g_ffx_fsr1_easu_pass_0c216412a066e680c4065e6790a2df3c_CBVResourceBindings, g_ffx_fsr1_easu_pass_0c216412a066e680c4065e6790a2df3c_CBVResourceCounts, g_ffx_fsr1_easu_pass_0c216412a066e680c4065e6790a2df3c_CBVResourceSets, 1, g_ffx_fsr1_easu_pass_0c216412a066e680c4065e6790a2df3c_TextureSRVResourceNames, g_ffx_fsr1_easu_pass_0c216412a066e680c4065e6790a2df3c_TextureSRVResourceBindings, g_ffx_fsr1_easu_pass_0c216412a066e680c4065e6790a2df3c_TextureSRVResourceCounts, g_ffx_fsr1_easu_pass_0c216412a066e680c4065e6790a2df3c_TextureSRVResourceSets, 1, g_ffx_fsr1_easu_pass_0c216412a066e680c4065e6790a2df3c_TextureUAVResourceNames, g_ffx_fsr1_easu_pass_0c216412a066e680c4065e6790a2df3c_TextureUAVResourceBindings, g_ffx_fsr1_easu_pass_0c216412a066e680c4065e6790a2df3c_TextureUAVResourceCounts, g_ffx_fsr1_easu_pass_0c216412a066e680c4065e6790a2df3c_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr1_easu_pass_0c216412a066e680c4065e6790a2df3c_SamplerResourceNames, g_ffx_fsr1_easu_pass_0c216412a066e680c4065e6790a2df3c_SamplerResourceBindings, g_ffx_fsr1_easu_pass_0c216412a066e680c4065e6790a2df3c_SamplerResourceCounts, g_ffx_fsr1_easu_pass_0c216412a066e680c4065e6790a2df3c_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr1_easu_pass_81fd7740d976e7a5248c157b142f74d5_size, g_ffx_fsr1_easu_pass_81fd7740d976e7a5248c157b142f74d5_data, 1, g_ffx_fsr1_easu_pass_81fd7740d976e7a5248c157b142f74d5_CBVResourceNames, g_ffx_fsr1_easu_pass_81fd7740d976e7a5248c157b142f74d5_CBVResourceBindings, g_ffx_fsr1_easu_pass_81fd7740d976e7a5248c157b142f74d5_CBVResourceCounts, g_ffx_fsr1_easu_pass_81fd7740d976e7a5248c157b142f74d5_CBVResourceSets, 1, g_ffx_fsr1_easu_pass_81fd7740d976e7a5248c157b142f74d5_TextureSRVResourceNames, g_ffx_fsr1_easu_pass_81fd7740d976e7a5248c157b142f74d5_TextureSRVResourceBindings, g_ffx_fsr1_easu_pass_81fd7740d976e7a5248c157b142f74d5_TextureSRVResourceCounts, g_ffx_fsr1_easu_pass_81fd7740d976e7a5248c157b142f74d5_TextureSRVResourceSets, 1, g_ffx_fsr1_easu_pass_81fd7740d976e7a5248c157b142f74d5_TextureUAVResourceNames, g_ffx_fsr1_easu_pass_81fd7740d976e7a5248c157b142f74d5_TextureUAVResourceBindings, g_ffx_fsr1_easu_pass_81fd7740d976e7a5248c157b142f74d5_TextureUAVResourceCounts, g_ffx_fsr1_easu_pass_81fd7740d976e7a5248c157b142f74d5_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr1_easu_pass_81fd7740d976e7a5248c157b142f74d5_SamplerResourceNames, g_ffx_fsr1_easu_pass_81fd7740d976e7a5248c157b142f74d5_SamplerResourceBindings, g_ffx_fsr1_easu_pass_81fd7740d976e7a5248c157b142f74d5_SamplerResourceCounts, g_ffx_fsr1_easu_pass_81fd7740d976e7a5248c157b142f74d5_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
};

