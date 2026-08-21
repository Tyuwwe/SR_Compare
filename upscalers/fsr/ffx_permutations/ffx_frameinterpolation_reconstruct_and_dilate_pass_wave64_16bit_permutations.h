#include "ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_c139bdf043966087210e32d923dbef0e.h"
#include "ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_a7a890a23c6cd3467dfc853b5056423a.h"
#include "ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_e0e996e56973b26982bd9954f13b0d3e.h"
#include "ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_4b81bf052015263d9f389e2a3df232c0.h"

typedef union ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_PermutationKey {
    struct {
        uint32_t FFX_FRAMEINTERPOLATION_OPTION_LOW_RES_MOTION_VECTORS : 1;
        uint32_t FFX_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH : 1;
        uint32_t FFX_FRAMEINTERPOLATION_OPTION_JITTER_MOTION_VECTORS : 1;
    };
    uint32_t index;
} ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_PermutationKey;

typedef struct ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_PermutationInfo {
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
} ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_PermutationInfo;

static const uint32_t g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_IndirectionTable[] = {
    3,
    1,
    2,
    0,
    3,
    1,
    2,
    0,
};

static const ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_PermutationInfo g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_PermutationInfo[] = {
    { g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_c139bdf043966087210e32d923dbef0e_size, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_c139bdf043966087210e32d923dbef0e_data, 1, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_c139bdf043966087210e32d923dbef0e_CBVResourceNames, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_c139bdf043966087210e32d923dbef0e_CBVResourceBindings, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_c139bdf043966087210e32d923dbef0e_CBVResourceCounts, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_c139bdf043966087210e32d923dbef0e_CBVResourceSets, 2, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_c139bdf043966087210e32d923dbef0e_TextureSRVResourceNames, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_c139bdf043966087210e32d923dbef0e_TextureSRVResourceBindings, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_c139bdf043966087210e32d923dbef0e_TextureSRVResourceCounts, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_c139bdf043966087210e32d923dbef0e_TextureSRVResourceSets, 3, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_c139bdf043966087210e32d923dbef0e_TextureUAVResourceNames, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_c139bdf043966087210e32d923dbef0e_TextureUAVResourceBindings, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_c139bdf043966087210e32d923dbef0e_TextureUAVResourceCounts, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_c139bdf043966087210e32d923dbef0e_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_a7a890a23c6cd3467dfc853b5056423a_size, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_a7a890a23c6cd3467dfc853b5056423a_data, 1, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_a7a890a23c6cd3467dfc853b5056423a_CBVResourceNames, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_a7a890a23c6cd3467dfc853b5056423a_CBVResourceBindings, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_a7a890a23c6cd3467dfc853b5056423a_CBVResourceCounts, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_a7a890a23c6cd3467dfc853b5056423a_CBVResourceSets, 2, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_a7a890a23c6cd3467dfc853b5056423a_TextureSRVResourceNames, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_a7a890a23c6cd3467dfc853b5056423a_TextureSRVResourceBindings, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_a7a890a23c6cd3467dfc853b5056423a_TextureSRVResourceCounts, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_a7a890a23c6cd3467dfc853b5056423a_TextureSRVResourceSets, 3, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_a7a890a23c6cd3467dfc853b5056423a_TextureUAVResourceNames, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_a7a890a23c6cd3467dfc853b5056423a_TextureUAVResourceBindings, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_a7a890a23c6cd3467dfc853b5056423a_TextureUAVResourceCounts, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_a7a890a23c6cd3467dfc853b5056423a_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_e0e996e56973b26982bd9954f13b0d3e_size, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_e0e996e56973b26982bd9954f13b0d3e_data, 1, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_e0e996e56973b26982bd9954f13b0d3e_CBVResourceNames, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_e0e996e56973b26982bd9954f13b0d3e_CBVResourceBindings, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_e0e996e56973b26982bd9954f13b0d3e_CBVResourceCounts, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_e0e996e56973b26982bd9954f13b0d3e_CBVResourceSets, 2, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_e0e996e56973b26982bd9954f13b0d3e_TextureSRVResourceNames, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_e0e996e56973b26982bd9954f13b0d3e_TextureSRVResourceBindings, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_e0e996e56973b26982bd9954f13b0d3e_TextureSRVResourceCounts, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_e0e996e56973b26982bd9954f13b0d3e_TextureSRVResourceSets, 3, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_e0e996e56973b26982bd9954f13b0d3e_TextureUAVResourceNames, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_e0e996e56973b26982bd9954f13b0d3e_TextureUAVResourceBindings, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_e0e996e56973b26982bd9954f13b0d3e_TextureUAVResourceCounts, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_e0e996e56973b26982bd9954f13b0d3e_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_4b81bf052015263d9f389e2a3df232c0_size, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_4b81bf052015263d9f389e2a3df232c0_data, 1, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_4b81bf052015263d9f389e2a3df232c0_CBVResourceNames, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_4b81bf052015263d9f389e2a3df232c0_CBVResourceBindings, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_4b81bf052015263d9f389e2a3df232c0_CBVResourceCounts, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_4b81bf052015263d9f389e2a3df232c0_CBVResourceSets, 2, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_4b81bf052015263d9f389e2a3df232c0_TextureSRVResourceNames, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_4b81bf052015263d9f389e2a3df232c0_TextureSRVResourceBindings, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_4b81bf052015263d9f389e2a3df232c0_TextureSRVResourceCounts, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_4b81bf052015263d9f389e2a3df232c0_TextureSRVResourceSets, 3, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_4b81bf052015263d9f389e2a3df232c0_TextureUAVResourceNames, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_4b81bf052015263d9f389e2a3df232c0_TextureUAVResourceBindings, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_4b81bf052015263d9f389e2a3df232c0_TextureUAVResourceCounts, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_16bit_4b81bf052015263d9f389e2a3df232c0_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
};

