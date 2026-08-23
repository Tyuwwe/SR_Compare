#include "ffx_fsr1_rcas_pass_wave64_16bit_0971aa320dc55ab1dc730dfac9bc952d.h"
#include "ffx_fsr1_rcas_pass_wave64_16bit_f0b03c5be2e5fc1877741cb60e9a36fd.h"

typedef union ffx_fsr1_rcas_pass_wave64_16bit_PermutationKey {
    struct {
        uint32_t FFX_FSR1_OPTION_APPLY_RCAS : 1;
        uint32_t FFX_FSR1_OPTION_RCAS_PASSTHROUGH_ALPHA : 1;
        uint32_t FFX_FSR1_OPTION_SRGB_CONVERSIONS : 1;
    };
    uint32_t index;
} ffx_fsr1_rcas_pass_wave64_16bit_PermutationKey;

typedef struct ffx_fsr1_rcas_pass_wave64_16bit_PermutationInfo {
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
} ffx_fsr1_rcas_pass_wave64_16bit_PermutationInfo;

static const uint32_t g_ffx_fsr1_rcas_pass_wave64_16bit_IndirectionTable[] = {
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
};

static const ffx_fsr1_rcas_pass_wave64_16bit_PermutationInfo g_ffx_fsr1_rcas_pass_wave64_16bit_PermutationInfo[] = {
    { g_ffx_fsr1_rcas_pass_wave64_16bit_0971aa320dc55ab1dc730dfac9bc952d_size, g_ffx_fsr1_rcas_pass_wave64_16bit_0971aa320dc55ab1dc730dfac9bc952d_data, 1, g_ffx_fsr1_rcas_pass_wave64_16bit_0971aa320dc55ab1dc730dfac9bc952d_CBVResourceNames, g_ffx_fsr1_rcas_pass_wave64_16bit_0971aa320dc55ab1dc730dfac9bc952d_CBVResourceBindings, g_ffx_fsr1_rcas_pass_wave64_16bit_0971aa320dc55ab1dc730dfac9bc952d_CBVResourceCounts, g_ffx_fsr1_rcas_pass_wave64_16bit_0971aa320dc55ab1dc730dfac9bc952d_CBVResourceSets, 1, g_ffx_fsr1_rcas_pass_wave64_16bit_0971aa320dc55ab1dc730dfac9bc952d_TextureSRVResourceNames, g_ffx_fsr1_rcas_pass_wave64_16bit_0971aa320dc55ab1dc730dfac9bc952d_TextureSRVResourceBindings, g_ffx_fsr1_rcas_pass_wave64_16bit_0971aa320dc55ab1dc730dfac9bc952d_TextureSRVResourceCounts, g_ffx_fsr1_rcas_pass_wave64_16bit_0971aa320dc55ab1dc730dfac9bc952d_TextureSRVResourceSets, 1, g_ffx_fsr1_rcas_pass_wave64_16bit_0971aa320dc55ab1dc730dfac9bc952d_TextureUAVResourceNames, g_ffx_fsr1_rcas_pass_wave64_16bit_0971aa320dc55ab1dc730dfac9bc952d_TextureUAVResourceBindings, g_ffx_fsr1_rcas_pass_wave64_16bit_0971aa320dc55ab1dc730dfac9bc952d_TextureUAVResourceCounts, g_ffx_fsr1_rcas_pass_wave64_16bit_0971aa320dc55ab1dc730dfac9bc952d_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr1_rcas_pass_wave64_16bit_f0b03c5be2e5fc1877741cb60e9a36fd_size, g_ffx_fsr1_rcas_pass_wave64_16bit_f0b03c5be2e5fc1877741cb60e9a36fd_data, 1, g_ffx_fsr1_rcas_pass_wave64_16bit_f0b03c5be2e5fc1877741cb60e9a36fd_CBVResourceNames, g_ffx_fsr1_rcas_pass_wave64_16bit_f0b03c5be2e5fc1877741cb60e9a36fd_CBVResourceBindings, g_ffx_fsr1_rcas_pass_wave64_16bit_f0b03c5be2e5fc1877741cb60e9a36fd_CBVResourceCounts, g_ffx_fsr1_rcas_pass_wave64_16bit_f0b03c5be2e5fc1877741cb60e9a36fd_CBVResourceSets, 1, g_ffx_fsr1_rcas_pass_wave64_16bit_f0b03c5be2e5fc1877741cb60e9a36fd_TextureSRVResourceNames, g_ffx_fsr1_rcas_pass_wave64_16bit_f0b03c5be2e5fc1877741cb60e9a36fd_TextureSRVResourceBindings, g_ffx_fsr1_rcas_pass_wave64_16bit_f0b03c5be2e5fc1877741cb60e9a36fd_TextureSRVResourceCounts, g_ffx_fsr1_rcas_pass_wave64_16bit_f0b03c5be2e5fc1877741cb60e9a36fd_TextureSRVResourceSets, 1, g_ffx_fsr1_rcas_pass_wave64_16bit_f0b03c5be2e5fc1877741cb60e9a36fd_TextureUAVResourceNames, g_ffx_fsr1_rcas_pass_wave64_16bit_f0b03c5be2e5fc1877741cb60e9a36fd_TextureUAVResourceBindings, g_ffx_fsr1_rcas_pass_wave64_16bit_f0b03c5be2e5fc1877741cb60e9a36fd_TextureUAVResourceCounts, g_ffx_fsr1_rcas_pass_wave64_16bit_f0b03c5be2e5fc1877741cb60e9a36fd_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
};

