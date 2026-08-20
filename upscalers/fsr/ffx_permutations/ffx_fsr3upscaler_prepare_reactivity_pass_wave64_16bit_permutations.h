#include "ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_3e402c3200b0f0f08c088948c92e9c0b.h"

typedef union ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_PermutationKey {
    struct {
        uint32_t FFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS : 1;
        uint32_t FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH : 1;
        uint32_t FFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE : 1;
        uint32_t FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT : 1;
        uint32_t FFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS : 1;
        uint32_t FFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING : 1;
    };
    uint32_t index;
} ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_PermutationKey;

typedef struct ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_PermutationInfo {
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
} ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_PermutationInfo;

static const uint32_t g_ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_IndirectionTable[] = {
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
};

static const ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_PermutationInfo g_ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_PermutationInfo[] = {
    { g_ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_3e402c3200b0f0f08c088948c92e9c0b_size, g_ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_3e402c3200b0f0f08c088948c92e9c0b_data, 1, g_ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_3e402c3200b0f0f08c088948c92e9c0b_CBVResourceNames, g_ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_3e402c3200b0f0f08c088948c92e9c0b_CBVResourceBindings, g_ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_3e402c3200b0f0f08c088948c92e9c0b_CBVResourceCounts, g_ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_3e402c3200b0f0f08c088948c92e9c0b_CBVResourceSets, 9, g_ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_3e402c3200b0f0f08c088948c92e9c0b_TextureSRVResourceNames, g_ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_3e402c3200b0f0f08c088948c92e9c0b_TextureSRVResourceBindings, g_ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_3e402c3200b0f0f08c088948c92e9c0b_TextureSRVResourceCounts, g_ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_3e402c3200b0f0f08c088948c92e9c0b_TextureSRVResourceSets, 3, g_ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_3e402c3200b0f0f08c088948c92e9c0b_TextureUAVResourceNames, g_ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_3e402c3200b0f0f08c088948c92e9c0b_TextureUAVResourceBindings, g_ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_3e402c3200b0f0f08c088948c92e9c0b_TextureUAVResourceCounts, g_ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_3e402c3200b0f0f08c088948c92e9c0b_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_3e402c3200b0f0f08c088948c92e9c0b_SamplerResourceNames, g_ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_3e402c3200b0f0f08c088948c92e9c0b_SamplerResourceBindings, g_ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_3e402c3200b0f0f08c088948c92e9c0b_SamplerResourceCounts, g_ffx_fsr3upscaler_prepare_reactivity_pass_wave64_16bit_3e402c3200b0f0f08c088948c92e9c0b_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
};

