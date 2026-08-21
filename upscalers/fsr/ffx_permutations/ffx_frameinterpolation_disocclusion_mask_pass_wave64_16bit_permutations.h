#include "ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_481b0372dbdbd712039e0286cb357731.h"
#include "ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_3a7d1afd277e3d4d432a09bd04798830.h"

typedef union ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_PermutationKey {
    struct {
        uint32_t FFX_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH : 1;
        uint32_t FFX_FRAMEINTERPOLATION_OPTION_LOW_RES_MOTION_VECTORS : 1;
        uint32_t FFX_FRAMEINTERPOLATION_OPTION_JITTER_MOTION_VECTORS : 1;
    };
    uint32_t index;
} ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_PermutationKey;

typedef struct ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_PermutationInfo {
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
} ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_PermutationInfo;

static const uint32_t g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_IndirectionTable[] = {
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
};

static const ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_PermutationInfo g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_PermutationInfo[] = {
    { g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_481b0372dbdbd712039e0286cb357731_size, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_481b0372dbdbd712039e0286cb357731_data, 1, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_481b0372dbdbd712039e0286cb357731_CBVResourceNames, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_481b0372dbdbd712039e0286cb357731_CBVResourceBindings, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_481b0372dbdbd712039e0286cb357731_CBVResourceCounts, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_481b0372dbdbd712039e0286cb357731_CBVResourceSets, 7, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_481b0372dbdbd712039e0286cb357731_TextureSRVResourceNames, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_481b0372dbdbd712039e0286cb357731_TextureSRVResourceBindings, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_481b0372dbdbd712039e0286cb357731_TextureSRVResourceCounts, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_481b0372dbdbd712039e0286cb357731_TextureSRVResourceSets, 1, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_481b0372dbdbd712039e0286cb357731_TextureUAVResourceNames, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_481b0372dbdbd712039e0286cb357731_TextureUAVResourceBindings, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_481b0372dbdbd712039e0286cb357731_TextureUAVResourceCounts, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_481b0372dbdbd712039e0286cb357731_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_481b0372dbdbd712039e0286cb357731_SamplerResourceNames, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_481b0372dbdbd712039e0286cb357731_SamplerResourceBindings, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_481b0372dbdbd712039e0286cb357731_SamplerResourceCounts, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_481b0372dbdbd712039e0286cb357731_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_3a7d1afd277e3d4d432a09bd04798830_size, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_3a7d1afd277e3d4d432a09bd04798830_data, 1, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_3a7d1afd277e3d4d432a09bd04798830_CBVResourceNames, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_3a7d1afd277e3d4d432a09bd04798830_CBVResourceBindings, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_3a7d1afd277e3d4d432a09bd04798830_CBVResourceCounts, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_3a7d1afd277e3d4d432a09bd04798830_CBVResourceSets, 7, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_3a7d1afd277e3d4d432a09bd04798830_TextureSRVResourceNames, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_3a7d1afd277e3d4d432a09bd04798830_TextureSRVResourceBindings, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_3a7d1afd277e3d4d432a09bd04798830_TextureSRVResourceCounts, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_3a7d1afd277e3d4d432a09bd04798830_TextureSRVResourceSets, 1, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_3a7d1afd277e3d4d432a09bd04798830_TextureUAVResourceNames, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_3a7d1afd277e3d4d432a09bd04798830_TextureUAVResourceBindings, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_3a7d1afd277e3d4d432a09bd04798830_TextureUAVResourceCounts, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_3a7d1afd277e3d4d432a09bd04798830_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_3a7d1afd277e3d4d432a09bd04798830_SamplerResourceNames, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_3a7d1afd277e3d4d432a09bd04798830_SamplerResourceBindings, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_3a7d1afd277e3d4d432a09bd04798830_SamplerResourceCounts, g_ffx_frameinterpolation_disocclusion_mask_pass_wave64_16bit_3a7d1afd277e3d4d432a09bd04798830_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
};

