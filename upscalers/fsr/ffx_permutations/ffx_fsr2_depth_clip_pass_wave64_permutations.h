#include "ffx_fsr2_depth_clip_pass_wave64_5b7e5a95f238eb8ced331480c730590d.h"
#include "ffx_fsr2_depth_clip_pass_wave64_35983f8bbfbec1e91a8744e343d7dfcf.h"
#include "ffx_fsr2_depth_clip_pass_wave64_1d3c607d5d0db5dfa13b1ae683c05876.h"
#include "ffx_fsr2_depth_clip_pass_wave64_17d7b1eb4a4417e3e191fe75dcf39786.h"
#include "ffx_fsr2_depth_clip_pass_wave64_17635b54db49d5bd07853604476c4825.h"
#include "ffx_fsr2_depth_clip_pass_wave64_f1395a3fd21af2b2816fa255f6dd9360.h"
#include "ffx_fsr2_depth_clip_pass_wave64_8d43fef06a31b9006bd6dd12f0d0e691.h"
#include "ffx_fsr2_depth_clip_pass_wave64_6659e8300a1ae5d1a2561b1a76360282.h"

typedef union ffx_fsr2_depth_clip_pass_wave64_PermutationKey {
    struct {
        uint32_t FFX_FSR2_OPTION_LOW_RESOLUTION_MOTION_VECTORS : 1;
        uint32_t FFX_FSR2_OPTION_JITTERED_MOTION_VECTORS : 1;
        uint32_t FFX_FSR2_OPTION_INVERTED_DEPTH : 1;
        uint32_t FFX_FSR2_OPTION_REPROJECT_USE_LANCZOS_TYPE : 1;
        uint32_t FFX_FSR2_OPTION_HDR_COLOR_INPUT : 1;
        uint32_t FFX_FSR2_OPTION_APPLY_SHARPENING : 1;
    };
    uint32_t index;
} ffx_fsr2_depth_clip_pass_wave64_PermutationKey;

typedef struct ffx_fsr2_depth_clip_pass_wave64_PermutationInfo {
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
} ffx_fsr2_depth_clip_pass_wave64_PermutationInfo;

static const uint32_t g_ffx_fsr2_depth_clip_pass_wave64_IndirectionTable[] = {
    1,
    6,
    2,
    5,
    7,
    3,
    0,
    4,
    1,
    6,
    2,
    5,
    7,
    3,
    0,
    4,
    1,
    6,
    2,
    5,
    7,
    3,
    0,
    4,
    1,
    6,
    2,
    5,
    7,
    3,
    0,
    4,
    1,
    6,
    2,
    5,
    7,
    3,
    0,
    4,
    1,
    6,
    2,
    5,
    7,
    3,
    0,
    4,
    1,
    6,
    2,
    5,
    7,
    3,
    0,
    4,
    1,
    6,
    2,
    5,
    7,
    3,
    0,
    4,
};

static const ffx_fsr2_depth_clip_pass_wave64_PermutationInfo g_ffx_fsr2_depth_clip_pass_wave64_PermutationInfo[] = {
    { g_ffx_fsr2_depth_clip_pass_wave64_5b7e5a95f238eb8ced331480c730590d_size, g_ffx_fsr2_depth_clip_pass_wave64_5b7e5a95f238eb8ced331480c730590d_data, 1, g_ffx_fsr2_depth_clip_pass_wave64_5b7e5a95f238eb8ced331480c730590d_CBVResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_5b7e5a95f238eb8ced331480c730590d_CBVResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_5b7e5a95f238eb8ced331480c730590d_CBVResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_5b7e5a95f238eb8ced331480c730590d_CBVResourceSets, 9, g_ffx_fsr2_depth_clip_pass_wave64_5b7e5a95f238eb8ced331480c730590d_TextureSRVResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_5b7e5a95f238eb8ced331480c730590d_TextureSRVResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_5b7e5a95f238eb8ced331480c730590d_TextureSRVResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_5b7e5a95f238eb8ced331480c730590d_TextureSRVResourceSets, 2, g_ffx_fsr2_depth_clip_pass_wave64_5b7e5a95f238eb8ced331480c730590d_TextureUAVResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_5b7e5a95f238eb8ced331480c730590d_TextureUAVResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_5b7e5a95f238eb8ced331480c730590d_TextureUAVResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_5b7e5a95f238eb8ced331480c730590d_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr2_depth_clip_pass_wave64_5b7e5a95f238eb8ced331480c730590d_SamplerResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_5b7e5a95f238eb8ced331480c730590d_SamplerResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_5b7e5a95f238eb8ced331480c730590d_SamplerResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_5b7e5a95f238eb8ced331480c730590d_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr2_depth_clip_pass_wave64_35983f8bbfbec1e91a8744e343d7dfcf_size, g_ffx_fsr2_depth_clip_pass_wave64_35983f8bbfbec1e91a8744e343d7dfcf_data, 1, g_ffx_fsr2_depth_clip_pass_wave64_35983f8bbfbec1e91a8744e343d7dfcf_CBVResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_35983f8bbfbec1e91a8744e343d7dfcf_CBVResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_35983f8bbfbec1e91a8744e343d7dfcf_CBVResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_35983f8bbfbec1e91a8744e343d7dfcf_CBVResourceSets, 9, g_ffx_fsr2_depth_clip_pass_wave64_35983f8bbfbec1e91a8744e343d7dfcf_TextureSRVResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_35983f8bbfbec1e91a8744e343d7dfcf_TextureSRVResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_35983f8bbfbec1e91a8744e343d7dfcf_TextureSRVResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_35983f8bbfbec1e91a8744e343d7dfcf_TextureSRVResourceSets, 2, g_ffx_fsr2_depth_clip_pass_wave64_35983f8bbfbec1e91a8744e343d7dfcf_TextureUAVResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_35983f8bbfbec1e91a8744e343d7dfcf_TextureUAVResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_35983f8bbfbec1e91a8744e343d7dfcf_TextureUAVResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_35983f8bbfbec1e91a8744e343d7dfcf_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr2_depth_clip_pass_wave64_35983f8bbfbec1e91a8744e343d7dfcf_SamplerResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_35983f8bbfbec1e91a8744e343d7dfcf_SamplerResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_35983f8bbfbec1e91a8744e343d7dfcf_SamplerResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_35983f8bbfbec1e91a8744e343d7dfcf_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr2_depth_clip_pass_wave64_1d3c607d5d0db5dfa13b1ae683c05876_size, g_ffx_fsr2_depth_clip_pass_wave64_1d3c607d5d0db5dfa13b1ae683c05876_data, 1, g_ffx_fsr2_depth_clip_pass_wave64_1d3c607d5d0db5dfa13b1ae683c05876_CBVResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_1d3c607d5d0db5dfa13b1ae683c05876_CBVResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_1d3c607d5d0db5dfa13b1ae683c05876_CBVResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_1d3c607d5d0db5dfa13b1ae683c05876_CBVResourceSets, 9, g_ffx_fsr2_depth_clip_pass_wave64_1d3c607d5d0db5dfa13b1ae683c05876_TextureSRVResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_1d3c607d5d0db5dfa13b1ae683c05876_TextureSRVResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_1d3c607d5d0db5dfa13b1ae683c05876_TextureSRVResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_1d3c607d5d0db5dfa13b1ae683c05876_TextureSRVResourceSets, 2, g_ffx_fsr2_depth_clip_pass_wave64_1d3c607d5d0db5dfa13b1ae683c05876_TextureUAVResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_1d3c607d5d0db5dfa13b1ae683c05876_TextureUAVResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_1d3c607d5d0db5dfa13b1ae683c05876_TextureUAVResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_1d3c607d5d0db5dfa13b1ae683c05876_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr2_depth_clip_pass_wave64_1d3c607d5d0db5dfa13b1ae683c05876_SamplerResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_1d3c607d5d0db5dfa13b1ae683c05876_SamplerResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_1d3c607d5d0db5dfa13b1ae683c05876_SamplerResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_1d3c607d5d0db5dfa13b1ae683c05876_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr2_depth_clip_pass_wave64_17d7b1eb4a4417e3e191fe75dcf39786_size, g_ffx_fsr2_depth_clip_pass_wave64_17d7b1eb4a4417e3e191fe75dcf39786_data, 1, g_ffx_fsr2_depth_clip_pass_wave64_17d7b1eb4a4417e3e191fe75dcf39786_CBVResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_17d7b1eb4a4417e3e191fe75dcf39786_CBVResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_17d7b1eb4a4417e3e191fe75dcf39786_CBVResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_17d7b1eb4a4417e3e191fe75dcf39786_CBVResourceSets, 9, g_ffx_fsr2_depth_clip_pass_wave64_17d7b1eb4a4417e3e191fe75dcf39786_TextureSRVResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_17d7b1eb4a4417e3e191fe75dcf39786_TextureSRVResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_17d7b1eb4a4417e3e191fe75dcf39786_TextureSRVResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_17d7b1eb4a4417e3e191fe75dcf39786_TextureSRVResourceSets, 2, g_ffx_fsr2_depth_clip_pass_wave64_17d7b1eb4a4417e3e191fe75dcf39786_TextureUAVResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_17d7b1eb4a4417e3e191fe75dcf39786_TextureUAVResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_17d7b1eb4a4417e3e191fe75dcf39786_TextureUAVResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_17d7b1eb4a4417e3e191fe75dcf39786_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr2_depth_clip_pass_wave64_17d7b1eb4a4417e3e191fe75dcf39786_SamplerResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_17d7b1eb4a4417e3e191fe75dcf39786_SamplerResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_17d7b1eb4a4417e3e191fe75dcf39786_SamplerResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_17d7b1eb4a4417e3e191fe75dcf39786_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr2_depth_clip_pass_wave64_17635b54db49d5bd07853604476c4825_size, g_ffx_fsr2_depth_clip_pass_wave64_17635b54db49d5bd07853604476c4825_data, 1, g_ffx_fsr2_depth_clip_pass_wave64_17635b54db49d5bd07853604476c4825_CBVResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_17635b54db49d5bd07853604476c4825_CBVResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_17635b54db49d5bd07853604476c4825_CBVResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_17635b54db49d5bd07853604476c4825_CBVResourceSets, 9, g_ffx_fsr2_depth_clip_pass_wave64_17635b54db49d5bd07853604476c4825_TextureSRVResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_17635b54db49d5bd07853604476c4825_TextureSRVResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_17635b54db49d5bd07853604476c4825_TextureSRVResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_17635b54db49d5bd07853604476c4825_TextureSRVResourceSets, 2, g_ffx_fsr2_depth_clip_pass_wave64_17635b54db49d5bd07853604476c4825_TextureUAVResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_17635b54db49d5bd07853604476c4825_TextureUAVResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_17635b54db49d5bd07853604476c4825_TextureUAVResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_17635b54db49d5bd07853604476c4825_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr2_depth_clip_pass_wave64_17635b54db49d5bd07853604476c4825_SamplerResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_17635b54db49d5bd07853604476c4825_SamplerResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_17635b54db49d5bd07853604476c4825_SamplerResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_17635b54db49d5bd07853604476c4825_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr2_depth_clip_pass_wave64_f1395a3fd21af2b2816fa255f6dd9360_size, g_ffx_fsr2_depth_clip_pass_wave64_f1395a3fd21af2b2816fa255f6dd9360_data, 1, g_ffx_fsr2_depth_clip_pass_wave64_f1395a3fd21af2b2816fa255f6dd9360_CBVResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_f1395a3fd21af2b2816fa255f6dd9360_CBVResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_f1395a3fd21af2b2816fa255f6dd9360_CBVResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_f1395a3fd21af2b2816fa255f6dd9360_CBVResourceSets, 9, g_ffx_fsr2_depth_clip_pass_wave64_f1395a3fd21af2b2816fa255f6dd9360_TextureSRVResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_f1395a3fd21af2b2816fa255f6dd9360_TextureSRVResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_f1395a3fd21af2b2816fa255f6dd9360_TextureSRVResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_f1395a3fd21af2b2816fa255f6dd9360_TextureSRVResourceSets, 2, g_ffx_fsr2_depth_clip_pass_wave64_f1395a3fd21af2b2816fa255f6dd9360_TextureUAVResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_f1395a3fd21af2b2816fa255f6dd9360_TextureUAVResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_f1395a3fd21af2b2816fa255f6dd9360_TextureUAVResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_f1395a3fd21af2b2816fa255f6dd9360_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr2_depth_clip_pass_wave64_f1395a3fd21af2b2816fa255f6dd9360_SamplerResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_f1395a3fd21af2b2816fa255f6dd9360_SamplerResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_f1395a3fd21af2b2816fa255f6dd9360_SamplerResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_f1395a3fd21af2b2816fa255f6dd9360_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr2_depth_clip_pass_wave64_8d43fef06a31b9006bd6dd12f0d0e691_size, g_ffx_fsr2_depth_clip_pass_wave64_8d43fef06a31b9006bd6dd12f0d0e691_data, 1, g_ffx_fsr2_depth_clip_pass_wave64_8d43fef06a31b9006bd6dd12f0d0e691_CBVResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_8d43fef06a31b9006bd6dd12f0d0e691_CBVResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_8d43fef06a31b9006bd6dd12f0d0e691_CBVResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_8d43fef06a31b9006bd6dd12f0d0e691_CBVResourceSets, 9, g_ffx_fsr2_depth_clip_pass_wave64_8d43fef06a31b9006bd6dd12f0d0e691_TextureSRVResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_8d43fef06a31b9006bd6dd12f0d0e691_TextureSRVResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_8d43fef06a31b9006bd6dd12f0d0e691_TextureSRVResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_8d43fef06a31b9006bd6dd12f0d0e691_TextureSRVResourceSets, 2, g_ffx_fsr2_depth_clip_pass_wave64_8d43fef06a31b9006bd6dd12f0d0e691_TextureUAVResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_8d43fef06a31b9006bd6dd12f0d0e691_TextureUAVResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_8d43fef06a31b9006bd6dd12f0d0e691_TextureUAVResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_8d43fef06a31b9006bd6dd12f0d0e691_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr2_depth_clip_pass_wave64_8d43fef06a31b9006bd6dd12f0d0e691_SamplerResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_8d43fef06a31b9006bd6dd12f0d0e691_SamplerResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_8d43fef06a31b9006bd6dd12f0d0e691_SamplerResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_8d43fef06a31b9006bd6dd12f0d0e691_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr2_depth_clip_pass_wave64_6659e8300a1ae5d1a2561b1a76360282_size, g_ffx_fsr2_depth_clip_pass_wave64_6659e8300a1ae5d1a2561b1a76360282_data, 1, g_ffx_fsr2_depth_clip_pass_wave64_6659e8300a1ae5d1a2561b1a76360282_CBVResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_6659e8300a1ae5d1a2561b1a76360282_CBVResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_6659e8300a1ae5d1a2561b1a76360282_CBVResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_6659e8300a1ae5d1a2561b1a76360282_CBVResourceSets, 9, g_ffx_fsr2_depth_clip_pass_wave64_6659e8300a1ae5d1a2561b1a76360282_TextureSRVResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_6659e8300a1ae5d1a2561b1a76360282_TextureSRVResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_6659e8300a1ae5d1a2561b1a76360282_TextureSRVResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_6659e8300a1ae5d1a2561b1a76360282_TextureSRVResourceSets, 2, g_ffx_fsr2_depth_clip_pass_wave64_6659e8300a1ae5d1a2561b1a76360282_TextureUAVResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_6659e8300a1ae5d1a2561b1a76360282_TextureUAVResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_6659e8300a1ae5d1a2561b1a76360282_TextureUAVResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_6659e8300a1ae5d1a2561b1a76360282_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr2_depth_clip_pass_wave64_6659e8300a1ae5d1a2561b1a76360282_SamplerResourceNames, g_ffx_fsr2_depth_clip_pass_wave64_6659e8300a1ae5d1a2561b1a76360282_SamplerResourceBindings, g_ffx_fsr2_depth_clip_pass_wave64_6659e8300a1ae5d1a2561b1a76360282_SamplerResourceCounts, g_ffx_fsr2_depth_clip_pass_wave64_6659e8300a1ae5d1a2561b1a76360282_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
};

