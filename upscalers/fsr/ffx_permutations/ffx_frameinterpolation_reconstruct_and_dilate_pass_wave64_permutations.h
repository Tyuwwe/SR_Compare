#include "ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_7e24c88c994de392aa063eda6741e271.h"
#include "ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_f02deabac9d1e75c59e9bacf66313af8.h"
#include "ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_2865c2abeb7c6f86e5285da59b7c28d9.h"
#include "ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_6a31904809988224db586aeca599121a.h"

typedef union ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_PermutationKey {
    struct {
        uint32_t FFX_FRAMEINTERPOLATION_OPTION_LOW_RES_MOTION_VECTORS : 1;
        uint32_t FFX_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH : 1;
        uint32_t FFX_FRAMEINTERPOLATION_OPTION_JITTER_MOTION_VECTORS : 1;
    };
    uint32_t index;
} ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_PermutationKey;

typedef struct ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_PermutationInfo {
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
} ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_PermutationInfo;

static const uint32_t g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_IndirectionTable[] = {
    2,
    3,
    0,
    1,
    2,
    3,
    0,
    1,
};

static const ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_PermutationInfo g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_PermutationInfo[] = {
    { g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_7e24c88c994de392aa063eda6741e271_size, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_7e24c88c994de392aa063eda6741e271_data, 1, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_7e24c88c994de392aa063eda6741e271_CBVResourceNames, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_7e24c88c994de392aa063eda6741e271_CBVResourceBindings, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_7e24c88c994de392aa063eda6741e271_CBVResourceCounts, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_7e24c88c994de392aa063eda6741e271_CBVResourceSets, 2, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_7e24c88c994de392aa063eda6741e271_TextureSRVResourceNames, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_7e24c88c994de392aa063eda6741e271_TextureSRVResourceBindings, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_7e24c88c994de392aa063eda6741e271_TextureSRVResourceCounts, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_7e24c88c994de392aa063eda6741e271_TextureSRVResourceSets, 3, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_7e24c88c994de392aa063eda6741e271_TextureUAVResourceNames, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_7e24c88c994de392aa063eda6741e271_TextureUAVResourceBindings, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_7e24c88c994de392aa063eda6741e271_TextureUAVResourceCounts, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_7e24c88c994de392aa063eda6741e271_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_f02deabac9d1e75c59e9bacf66313af8_size, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_f02deabac9d1e75c59e9bacf66313af8_data, 1, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_f02deabac9d1e75c59e9bacf66313af8_CBVResourceNames, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_f02deabac9d1e75c59e9bacf66313af8_CBVResourceBindings, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_f02deabac9d1e75c59e9bacf66313af8_CBVResourceCounts, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_f02deabac9d1e75c59e9bacf66313af8_CBVResourceSets, 2, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_f02deabac9d1e75c59e9bacf66313af8_TextureSRVResourceNames, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_f02deabac9d1e75c59e9bacf66313af8_TextureSRVResourceBindings, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_f02deabac9d1e75c59e9bacf66313af8_TextureSRVResourceCounts, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_f02deabac9d1e75c59e9bacf66313af8_TextureSRVResourceSets, 3, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_f02deabac9d1e75c59e9bacf66313af8_TextureUAVResourceNames, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_f02deabac9d1e75c59e9bacf66313af8_TextureUAVResourceBindings, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_f02deabac9d1e75c59e9bacf66313af8_TextureUAVResourceCounts, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_f02deabac9d1e75c59e9bacf66313af8_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_2865c2abeb7c6f86e5285da59b7c28d9_size, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_2865c2abeb7c6f86e5285da59b7c28d9_data, 1, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_2865c2abeb7c6f86e5285da59b7c28d9_CBVResourceNames, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_2865c2abeb7c6f86e5285da59b7c28d9_CBVResourceBindings, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_2865c2abeb7c6f86e5285da59b7c28d9_CBVResourceCounts, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_2865c2abeb7c6f86e5285da59b7c28d9_CBVResourceSets, 2, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_2865c2abeb7c6f86e5285da59b7c28d9_TextureSRVResourceNames, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_2865c2abeb7c6f86e5285da59b7c28d9_TextureSRVResourceBindings, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_2865c2abeb7c6f86e5285da59b7c28d9_TextureSRVResourceCounts, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_2865c2abeb7c6f86e5285da59b7c28d9_TextureSRVResourceSets, 3, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_2865c2abeb7c6f86e5285da59b7c28d9_TextureUAVResourceNames, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_2865c2abeb7c6f86e5285da59b7c28d9_TextureUAVResourceBindings, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_2865c2abeb7c6f86e5285da59b7c28d9_TextureUAVResourceCounts, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_2865c2abeb7c6f86e5285da59b7c28d9_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_6a31904809988224db586aeca599121a_size, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_6a31904809988224db586aeca599121a_data, 1, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_6a31904809988224db586aeca599121a_CBVResourceNames, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_6a31904809988224db586aeca599121a_CBVResourceBindings, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_6a31904809988224db586aeca599121a_CBVResourceCounts, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_6a31904809988224db586aeca599121a_CBVResourceSets, 2, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_6a31904809988224db586aeca599121a_TextureSRVResourceNames, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_6a31904809988224db586aeca599121a_TextureSRVResourceBindings, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_6a31904809988224db586aeca599121a_TextureSRVResourceCounts, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_6a31904809988224db586aeca599121a_TextureSRVResourceSets, 3, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_6a31904809988224db586aeca599121a_TextureUAVResourceNames, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_6a31904809988224db586aeca599121a_TextureUAVResourceBindings, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_6a31904809988224db586aeca599121a_TextureUAVResourceCounts, g_ffx_frameinterpolation_reconstruct_and_dilate_pass_wave64_6a31904809988224db586aeca599121a_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
};

