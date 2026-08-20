#include "ffx_fsr2_depth_clip_pass_16bit_3043ef697fac62c95f55a5c42d1b5359.h"
#include "ffx_fsr2_depth_clip_pass_16bit_6fbe4b2a6af87d8cfc8e40bcec927dfd.h"
#include "ffx_fsr2_depth_clip_pass_16bit_db708acee64203ff7d8bad121c4ff251.h"
#include "ffx_fsr2_depth_clip_pass_16bit_db780fd7344b2fe8d5c701651e91915d.h"
#include "ffx_fsr2_depth_clip_pass_16bit_31c00726e3b9216fdf28738ce9bfb9be.h"
#include "ffx_fsr2_depth_clip_pass_16bit_27acfa4e3aae81ef81806d5670d15088.h"
#include "ffx_fsr2_depth_clip_pass_16bit_238049d8470ad55a5e9089cd82827f99.h"
#include "ffx_fsr2_depth_clip_pass_16bit_a55979f142703ab9a599806ee34505a2.h"

typedef union ffx_fsr2_depth_clip_pass_16bit_PermutationKey {
    struct {
        uint32_t FFX_FSR2_OPTION_LOW_RESOLUTION_MOTION_VECTORS : 1;
        uint32_t FFX_FSR2_OPTION_JITTERED_MOTION_VECTORS : 1;
        uint32_t FFX_FSR2_OPTION_INVERTED_DEPTH : 1;
        uint32_t FFX_FSR2_OPTION_REPROJECT_USE_LANCZOS_TYPE : 1;
        uint32_t FFX_FSR2_OPTION_HDR_COLOR_INPUT : 1;
        uint32_t FFX_FSR2_OPTION_APPLY_SHARPENING : 1;
    };
    uint32_t index;
} ffx_fsr2_depth_clip_pass_16bit_PermutationKey;

typedef struct ffx_fsr2_depth_clip_pass_16bit_PermutationInfo {
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
} ffx_fsr2_depth_clip_pass_16bit_PermutationInfo;

static const uint32_t g_ffx_fsr2_depth_clip_pass_16bit_IndirectionTable[] = {
    4,
    1,
    7,
    2,
    6,
    3,
    5,
    0,
    4,
    1,
    7,
    2,
    6,
    3,
    5,
    0,
    4,
    1,
    7,
    2,
    6,
    3,
    5,
    0,
    4,
    1,
    7,
    2,
    6,
    3,
    5,
    0,
    4,
    1,
    7,
    2,
    6,
    3,
    5,
    0,
    4,
    1,
    7,
    2,
    6,
    3,
    5,
    0,
    4,
    1,
    7,
    2,
    6,
    3,
    5,
    0,
    4,
    1,
    7,
    2,
    6,
    3,
    5,
    0,
};

static const ffx_fsr2_depth_clip_pass_16bit_PermutationInfo g_ffx_fsr2_depth_clip_pass_16bit_PermutationInfo[] = {
    { g_ffx_fsr2_depth_clip_pass_16bit_3043ef697fac62c95f55a5c42d1b5359_size, g_ffx_fsr2_depth_clip_pass_16bit_3043ef697fac62c95f55a5c42d1b5359_data, 1, g_ffx_fsr2_depth_clip_pass_16bit_3043ef697fac62c95f55a5c42d1b5359_CBVResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_3043ef697fac62c95f55a5c42d1b5359_CBVResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_3043ef697fac62c95f55a5c42d1b5359_CBVResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_3043ef697fac62c95f55a5c42d1b5359_CBVResourceSets, 9, g_ffx_fsr2_depth_clip_pass_16bit_3043ef697fac62c95f55a5c42d1b5359_TextureSRVResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_3043ef697fac62c95f55a5c42d1b5359_TextureSRVResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_3043ef697fac62c95f55a5c42d1b5359_TextureSRVResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_3043ef697fac62c95f55a5c42d1b5359_TextureSRVResourceSets, 2, g_ffx_fsr2_depth_clip_pass_16bit_3043ef697fac62c95f55a5c42d1b5359_TextureUAVResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_3043ef697fac62c95f55a5c42d1b5359_TextureUAVResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_3043ef697fac62c95f55a5c42d1b5359_TextureUAVResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_3043ef697fac62c95f55a5c42d1b5359_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr2_depth_clip_pass_16bit_3043ef697fac62c95f55a5c42d1b5359_SamplerResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_3043ef697fac62c95f55a5c42d1b5359_SamplerResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_3043ef697fac62c95f55a5c42d1b5359_SamplerResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_3043ef697fac62c95f55a5c42d1b5359_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr2_depth_clip_pass_16bit_6fbe4b2a6af87d8cfc8e40bcec927dfd_size, g_ffx_fsr2_depth_clip_pass_16bit_6fbe4b2a6af87d8cfc8e40bcec927dfd_data, 1, g_ffx_fsr2_depth_clip_pass_16bit_6fbe4b2a6af87d8cfc8e40bcec927dfd_CBVResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_6fbe4b2a6af87d8cfc8e40bcec927dfd_CBVResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_6fbe4b2a6af87d8cfc8e40bcec927dfd_CBVResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_6fbe4b2a6af87d8cfc8e40bcec927dfd_CBVResourceSets, 9, g_ffx_fsr2_depth_clip_pass_16bit_6fbe4b2a6af87d8cfc8e40bcec927dfd_TextureSRVResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_6fbe4b2a6af87d8cfc8e40bcec927dfd_TextureSRVResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_6fbe4b2a6af87d8cfc8e40bcec927dfd_TextureSRVResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_6fbe4b2a6af87d8cfc8e40bcec927dfd_TextureSRVResourceSets, 2, g_ffx_fsr2_depth_clip_pass_16bit_6fbe4b2a6af87d8cfc8e40bcec927dfd_TextureUAVResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_6fbe4b2a6af87d8cfc8e40bcec927dfd_TextureUAVResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_6fbe4b2a6af87d8cfc8e40bcec927dfd_TextureUAVResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_6fbe4b2a6af87d8cfc8e40bcec927dfd_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr2_depth_clip_pass_16bit_6fbe4b2a6af87d8cfc8e40bcec927dfd_SamplerResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_6fbe4b2a6af87d8cfc8e40bcec927dfd_SamplerResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_6fbe4b2a6af87d8cfc8e40bcec927dfd_SamplerResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_6fbe4b2a6af87d8cfc8e40bcec927dfd_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr2_depth_clip_pass_16bit_db708acee64203ff7d8bad121c4ff251_size, g_ffx_fsr2_depth_clip_pass_16bit_db708acee64203ff7d8bad121c4ff251_data, 1, g_ffx_fsr2_depth_clip_pass_16bit_db708acee64203ff7d8bad121c4ff251_CBVResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_db708acee64203ff7d8bad121c4ff251_CBVResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_db708acee64203ff7d8bad121c4ff251_CBVResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_db708acee64203ff7d8bad121c4ff251_CBVResourceSets, 9, g_ffx_fsr2_depth_clip_pass_16bit_db708acee64203ff7d8bad121c4ff251_TextureSRVResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_db708acee64203ff7d8bad121c4ff251_TextureSRVResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_db708acee64203ff7d8bad121c4ff251_TextureSRVResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_db708acee64203ff7d8bad121c4ff251_TextureSRVResourceSets, 2, g_ffx_fsr2_depth_clip_pass_16bit_db708acee64203ff7d8bad121c4ff251_TextureUAVResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_db708acee64203ff7d8bad121c4ff251_TextureUAVResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_db708acee64203ff7d8bad121c4ff251_TextureUAVResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_db708acee64203ff7d8bad121c4ff251_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr2_depth_clip_pass_16bit_db708acee64203ff7d8bad121c4ff251_SamplerResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_db708acee64203ff7d8bad121c4ff251_SamplerResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_db708acee64203ff7d8bad121c4ff251_SamplerResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_db708acee64203ff7d8bad121c4ff251_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr2_depth_clip_pass_16bit_db780fd7344b2fe8d5c701651e91915d_size, g_ffx_fsr2_depth_clip_pass_16bit_db780fd7344b2fe8d5c701651e91915d_data, 1, g_ffx_fsr2_depth_clip_pass_16bit_db780fd7344b2fe8d5c701651e91915d_CBVResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_db780fd7344b2fe8d5c701651e91915d_CBVResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_db780fd7344b2fe8d5c701651e91915d_CBVResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_db780fd7344b2fe8d5c701651e91915d_CBVResourceSets, 9, g_ffx_fsr2_depth_clip_pass_16bit_db780fd7344b2fe8d5c701651e91915d_TextureSRVResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_db780fd7344b2fe8d5c701651e91915d_TextureSRVResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_db780fd7344b2fe8d5c701651e91915d_TextureSRVResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_db780fd7344b2fe8d5c701651e91915d_TextureSRVResourceSets, 2, g_ffx_fsr2_depth_clip_pass_16bit_db780fd7344b2fe8d5c701651e91915d_TextureUAVResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_db780fd7344b2fe8d5c701651e91915d_TextureUAVResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_db780fd7344b2fe8d5c701651e91915d_TextureUAVResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_db780fd7344b2fe8d5c701651e91915d_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr2_depth_clip_pass_16bit_db780fd7344b2fe8d5c701651e91915d_SamplerResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_db780fd7344b2fe8d5c701651e91915d_SamplerResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_db780fd7344b2fe8d5c701651e91915d_SamplerResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_db780fd7344b2fe8d5c701651e91915d_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr2_depth_clip_pass_16bit_31c00726e3b9216fdf28738ce9bfb9be_size, g_ffx_fsr2_depth_clip_pass_16bit_31c00726e3b9216fdf28738ce9bfb9be_data, 1, g_ffx_fsr2_depth_clip_pass_16bit_31c00726e3b9216fdf28738ce9bfb9be_CBVResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_31c00726e3b9216fdf28738ce9bfb9be_CBVResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_31c00726e3b9216fdf28738ce9bfb9be_CBVResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_31c00726e3b9216fdf28738ce9bfb9be_CBVResourceSets, 9, g_ffx_fsr2_depth_clip_pass_16bit_31c00726e3b9216fdf28738ce9bfb9be_TextureSRVResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_31c00726e3b9216fdf28738ce9bfb9be_TextureSRVResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_31c00726e3b9216fdf28738ce9bfb9be_TextureSRVResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_31c00726e3b9216fdf28738ce9bfb9be_TextureSRVResourceSets, 2, g_ffx_fsr2_depth_clip_pass_16bit_31c00726e3b9216fdf28738ce9bfb9be_TextureUAVResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_31c00726e3b9216fdf28738ce9bfb9be_TextureUAVResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_31c00726e3b9216fdf28738ce9bfb9be_TextureUAVResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_31c00726e3b9216fdf28738ce9bfb9be_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr2_depth_clip_pass_16bit_31c00726e3b9216fdf28738ce9bfb9be_SamplerResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_31c00726e3b9216fdf28738ce9bfb9be_SamplerResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_31c00726e3b9216fdf28738ce9bfb9be_SamplerResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_31c00726e3b9216fdf28738ce9bfb9be_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr2_depth_clip_pass_16bit_27acfa4e3aae81ef81806d5670d15088_size, g_ffx_fsr2_depth_clip_pass_16bit_27acfa4e3aae81ef81806d5670d15088_data, 1, g_ffx_fsr2_depth_clip_pass_16bit_27acfa4e3aae81ef81806d5670d15088_CBVResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_27acfa4e3aae81ef81806d5670d15088_CBVResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_27acfa4e3aae81ef81806d5670d15088_CBVResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_27acfa4e3aae81ef81806d5670d15088_CBVResourceSets, 9, g_ffx_fsr2_depth_clip_pass_16bit_27acfa4e3aae81ef81806d5670d15088_TextureSRVResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_27acfa4e3aae81ef81806d5670d15088_TextureSRVResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_27acfa4e3aae81ef81806d5670d15088_TextureSRVResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_27acfa4e3aae81ef81806d5670d15088_TextureSRVResourceSets, 2, g_ffx_fsr2_depth_clip_pass_16bit_27acfa4e3aae81ef81806d5670d15088_TextureUAVResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_27acfa4e3aae81ef81806d5670d15088_TextureUAVResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_27acfa4e3aae81ef81806d5670d15088_TextureUAVResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_27acfa4e3aae81ef81806d5670d15088_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr2_depth_clip_pass_16bit_27acfa4e3aae81ef81806d5670d15088_SamplerResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_27acfa4e3aae81ef81806d5670d15088_SamplerResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_27acfa4e3aae81ef81806d5670d15088_SamplerResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_27acfa4e3aae81ef81806d5670d15088_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr2_depth_clip_pass_16bit_238049d8470ad55a5e9089cd82827f99_size, g_ffx_fsr2_depth_clip_pass_16bit_238049d8470ad55a5e9089cd82827f99_data, 1, g_ffx_fsr2_depth_clip_pass_16bit_238049d8470ad55a5e9089cd82827f99_CBVResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_238049d8470ad55a5e9089cd82827f99_CBVResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_238049d8470ad55a5e9089cd82827f99_CBVResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_238049d8470ad55a5e9089cd82827f99_CBVResourceSets, 9, g_ffx_fsr2_depth_clip_pass_16bit_238049d8470ad55a5e9089cd82827f99_TextureSRVResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_238049d8470ad55a5e9089cd82827f99_TextureSRVResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_238049d8470ad55a5e9089cd82827f99_TextureSRVResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_238049d8470ad55a5e9089cd82827f99_TextureSRVResourceSets, 2, g_ffx_fsr2_depth_clip_pass_16bit_238049d8470ad55a5e9089cd82827f99_TextureUAVResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_238049d8470ad55a5e9089cd82827f99_TextureUAVResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_238049d8470ad55a5e9089cd82827f99_TextureUAVResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_238049d8470ad55a5e9089cd82827f99_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr2_depth_clip_pass_16bit_238049d8470ad55a5e9089cd82827f99_SamplerResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_238049d8470ad55a5e9089cd82827f99_SamplerResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_238049d8470ad55a5e9089cd82827f99_SamplerResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_238049d8470ad55a5e9089cd82827f99_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr2_depth_clip_pass_16bit_a55979f142703ab9a599806ee34505a2_size, g_ffx_fsr2_depth_clip_pass_16bit_a55979f142703ab9a599806ee34505a2_data, 1, g_ffx_fsr2_depth_clip_pass_16bit_a55979f142703ab9a599806ee34505a2_CBVResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_a55979f142703ab9a599806ee34505a2_CBVResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_a55979f142703ab9a599806ee34505a2_CBVResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_a55979f142703ab9a599806ee34505a2_CBVResourceSets, 9, g_ffx_fsr2_depth_clip_pass_16bit_a55979f142703ab9a599806ee34505a2_TextureSRVResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_a55979f142703ab9a599806ee34505a2_TextureSRVResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_a55979f142703ab9a599806ee34505a2_TextureSRVResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_a55979f142703ab9a599806ee34505a2_TextureSRVResourceSets, 2, g_ffx_fsr2_depth_clip_pass_16bit_a55979f142703ab9a599806ee34505a2_TextureUAVResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_a55979f142703ab9a599806ee34505a2_TextureUAVResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_a55979f142703ab9a599806ee34505a2_TextureUAVResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_a55979f142703ab9a599806ee34505a2_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, g_ffx_fsr2_depth_clip_pass_16bit_a55979f142703ab9a599806ee34505a2_SamplerResourceNames, g_ffx_fsr2_depth_clip_pass_16bit_a55979f142703ab9a599806ee34505a2_SamplerResourceBindings, g_ffx_fsr2_depth_clip_pass_16bit_a55979f142703ab9a599806ee34505a2_SamplerResourceCounts, g_ffx_fsr2_depth_clip_pass_16bit_a55979f142703ab9a599806ee34505a2_SamplerResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
};

