#include "ffx_fsr3upscaler_prepare_inputs_pass_16bit_85f9a88087a4647f3e2f63b4797cbaa4.h"
#include "ffx_fsr3upscaler_prepare_inputs_pass_16bit_d942d423b7c299c51529f6f67f667ae1.h"
#include "ffx_fsr3upscaler_prepare_inputs_pass_16bit_6b70cfcaff78bb315c4aed2f9d0c7499.h"
#include "ffx_fsr3upscaler_prepare_inputs_pass_16bit_3b62d65eea3c936034b5ec9b29ee198d.h"
#include "ffx_fsr3upscaler_prepare_inputs_pass_16bit_5236e1f8ac117a94245e80289dfe7697.h"
#include "ffx_fsr3upscaler_prepare_inputs_pass_16bit_ec1b4806d442cf288f9cb99b31548d2d.h"
#include "ffx_fsr3upscaler_prepare_inputs_pass_16bit_42a3c2c1aa26c9b694269f25a4569b07.h"
#include "ffx_fsr3upscaler_prepare_inputs_pass_16bit_95938a78a305c1001ea4ec748b7c26de.h"

typedef union ffx_fsr3upscaler_prepare_inputs_pass_16bit_PermutationKey {
    struct {
        uint32_t FFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS : 1;
        uint32_t FFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS : 1;
        uint32_t FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH : 1;
        uint32_t FFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE : 1;
        uint32_t FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT : 1;
        uint32_t FFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING : 1;
    };
    uint32_t index;
} ffx_fsr3upscaler_prepare_inputs_pass_16bit_PermutationKey;

typedef struct ffx_fsr3upscaler_prepare_inputs_pass_16bit_PermutationInfo {
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
} ffx_fsr3upscaler_prepare_inputs_pass_16bit_PermutationInfo;

static const uint32_t g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_IndirectionTable[] = {
    4,
    1,
    3,
    6,
    7,
    0,
    5,
    2,
    4,
    1,
    3,
    6,
    7,
    0,
    5,
    2,
    4,
    1,
    3,
    6,
    7,
    0,
    5,
    2,
    4,
    1,
    3,
    6,
    7,
    0,
    5,
    2,
    4,
    1,
    3,
    6,
    7,
    0,
    5,
    2,
    4,
    1,
    3,
    6,
    7,
    0,
    5,
    2,
    4,
    1,
    3,
    6,
    7,
    0,
    5,
    2,
    4,
    1,
    3,
    6,
    7,
    0,
    5,
    2,
};

static const ffx_fsr3upscaler_prepare_inputs_pass_16bit_PermutationInfo g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_PermutationInfo[] = {
    { g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_85f9a88087a4647f3e2f63b4797cbaa4_size, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_85f9a88087a4647f3e2f63b4797cbaa4_data, 1, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_85f9a88087a4647f3e2f63b4797cbaa4_CBVResourceNames, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_85f9a88087a4647f3e2f63b4797cbaa4_CBVResourceBindings, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_85f9a88087a4647f3e2f63b4797cbaa4_CBVResourceCounts, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_85f9a88087a4647f3e2f63b4797cbaa4_CBVResourceSets, 3, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_85f9a88087a4647f3e2f63b4797cbaa4_TextureSRVResourceNames, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_85f9a88087a4647f3e2f63b4797cbaa4_TextureSRVResourceBindings, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_85f9a88087a4647f3e2f63b4797cbaa4_TextureSRVResourceCounts, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_85f9a88087a4647f3e2f63b4797cbaa4_TextureSRVResourceSets, 5, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_85f9a88087a4647f3e2f63b4797cbaa4_TextureUAVResourceNames, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_85f9a88087a4647f3e2f63b4797cbaa4_TextureUAVResourceBindings, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_85f9a88087a4647f3e2f63b4797cbaa4_TextureUAVResourceCounts, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_85f9a88087a4647f3e2f63b4797cbaa4_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_d942d423b7c299c51529f6f67f667ae1_size, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_d942d423b7c299c51529f6f67f667ae1_data, 1, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_d942d423b7c299c51529f6f67f667ae1_CBVResourceNames, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_d942d423b7c299c51529f6f67f667ae1_CBVResourceBindings, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_d942d423b7c299c51529f6f67f667ae1_CBVResourceCounts, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_d942d423b7c299c51529f6f67f667ae1_CBVResourceSets, 3, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_d942d423b7c299c51529f6f67f667ae1_TextureSRVResourceNames, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_d942d423b7c299c51529f6f67f667ae1_TextureSRVResourceBindings, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_d942d423b7c299c51529f6f67f667ae1_TextureSRVResourceCounts, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_d942d423b7c299c51529f6f67f667ae1_TextureSRVResourceSets, 5, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_d942d423b7c299c51529f6f67f667ae1_TextureUAVResourceNames, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_d942d423b7c299c51529f6f67f667ae1_TextureUAVResourceBindings, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_d942d423b7c299c51529f6f67f667ae1_TextureUAVResourceCounts, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_d942d423b7c299c51529f6f67f667ae1_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_6b70cfcaff78bb315c4aed2f9d0c7499_size, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_6b70cfcaff78bb315c4aed2f9d0c7499_data, 1, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_6b70cfcaff78bb315c4aed2f9d0c7499_CBVResourceNames, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_6b70cfcaff78bb315c4aed2f9d0c7499_CBVResourceBindings, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_6b70cfcaff78bb315c4aed2f9d0c7499_CBVResourceCounts, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_6b70cfcaff78bb315c4aed2f9d0c7499_CBVResourceSets, 3, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_6b70cfcaff78bb315c4aed2f9d0c7499_TextureSRVResourceNames, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_6b70cfcaff78bb315c4aed2f9d0c7499_TextureSRVResourceBindings, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_6b70cfcaff78bb315c4aed2f9d0c7499_TextureSRVResourceCounts, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_6b70cfcaff78bb315c4aed2f9d0c7499_TextureSRVResourceSets, 5, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_6b70cfcaff78bb315c4aed2f9d0c7499_TextureUAVResourceNames, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_6b70cfcaff78bb315c4aed2f9d0c7499_TextureUAVResourceBindings, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_6b70cfcaff78bb315c4aed2f9d0c7499_TextureUAVResourceCounts, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_6b70cfcaff78bb315c4aed2f9d0c7499_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_3b62d65eea3c936034b5ec9b29ee198d_size, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_3b62d65eea3c936034b5ec9b29ee198d_data, 1, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_3b62d65eea3c936034b5ec9b29ee198d_CBVResourceNames, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_3b62d65eea3c936034b5ec9b29ee198d_CBVResourceBindings, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_3b62d65eea3c936034b5ec9b29ee198d_CBVResourceCounts, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_3b62d65eea3c936034b5ec9b29ee198d_CBVResourceSets, 3, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_3b62d65eea3c936034b5ec9b29ee198d_TextureSRVResourceNames, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_3b62d65eea3c936034b5ec9b29ee198d_TextureSRVResourceBindings, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_3b62d65eea3c936034b5ec9b29ee198d_TextureSRVResourceCounts, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_3b62d65eea3c936034b5ec9b29ee198d_TextureSRVResourceSets, 5, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_3b62d65eea3c936034b5ec9b29ee198d_TextureUAVResourceNames, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_3b62d65eea3c936034b5ec9b29ee198d_TextureUAVResourceBindings, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_3b62d65eea3c936034b5ec9b29ee198d_TextureUAVResourceCounts, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_3b62d65eea3c936034b5ec9b29ee198d_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_5236e1f8ac117a94245e80289dfe7697_size, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_5236e1f8ac117a94245e80289dfe7697_data, 1, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_5236e1f8ac117a94245e80289dfe7697_CBVResourceNames, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_5236e1f8ac117a94245e80289dfe7697_CBVResourceBindings, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_5236e1f8ac117a94245e80289dfe7697_CBVResourceCounts, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_5236e1f8ac117a94245e80289dfe7697_CBVResourceSets, 3, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_5236e1f8ac117a94245e80289dfe7697_TextureSRVResourceNames, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_5236e1f8ac117a94245e80289dfe7697_TextureSRVResourceBindings, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_5236e1f8ac117a94245e80289dfe7697_TextureSRVResourceCounts, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_5236e1f8ac117a94245e80289dfe7697_TextureSRVResourceSets, 5, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_5236e1f8ac117a94245e80289dfe7697_TextureUAVResourceNames, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_5236e1f8ac117a94245e80289dfe7697_TextureUAVResourceBindings, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_5236e1f8ac117a94245e80289dfe7697_TextureUAVResourceCounts, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_5236e1f8ac117a94245e80289dfe7697_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_ec1b4806d442cf288f9cb99b31548d2d_size, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_ec1b4806d442cf288f9cb99b31548d2d_data, 1, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_ec1b4806d442cf288f9cb99b31548d2d_CBVResourceNames, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_ec1b4806d442cf288f9cb99b31548d2d_CBVResourceBindings, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_ec1b4806d442cf288f9cb99b31548d2d_CBVResourceCounts, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_ec1b4806d442cf288f9cb99b31548d2d_CBVResourceSets, 3, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_ec1b4806d442cf288f9cb99b31548d2d_TextureSRVResourceNames, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_ec1b4806d442cf288f9cb99b31548d2d_TextureSRVResourceBindings, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_ec1b4806d442cf288f9cb99b31548d2d_TextureSRVResourceCounts, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_ec1b4806d442cf288f9cb99b31548d2d_TextureSRVResourceSets, 5, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_ec1b4806d442cf288f9cb99b31548d2d_TextureUAVResourceNames, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_ec1b4806d442cf288f9cb99b31548d2d_TextureUAVResourceBindings, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_ec1b4806d442cf288f9cb99b31548d2d_TextureUAVResourceCounts, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_ec1b4806d442cf288f9cb99b31548d2d_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_42a3c2c1aa26c9b694269f25a4569b07_size, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_42a3c2c1aa26c9b694269f25a4569b07_data, 1, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_42a3c2c1aa26c9b694269f25a4569b07_CBVResourceNames, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_42a3c2c1aa26c9b694269f25a4569b07_CBVResourceBindings, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_42a3c2c1aa26c9b694269f25a4569b07_CBVResourceCounts, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_42a3c2c1aa26c9b694269f25a4569b07_CBVResourceSets, 3, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_42a3c2c1aa26c9b694269f25a4569b07_TextureSRVResourceNames, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_42a3c2c1aa26c9b694269f25a4569b07_TextureSRVResourceBindings, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_42a3c2c1aa26c9b694269f25a4569b07_TextureSRVResourceCounts, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_42a3c2c1aa26c9b694269f25a4569b07_TextureSRVResourceSets, 5, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_42a3c2c1aa26c9b694269f25a4569b07_TextureUAVResourceNames, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_42a3c2c1aa26c9b694269f25a4569b07_TextureUAVResourceBindings, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_42a3c2c1aa26c9b694269f25a4569b07_TextureUAVResourceCounts, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_42a3c2c1aa26c9b694269f25a4569b07_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
    { g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_95938a78a305c1001ea4ec748b7c26de_size, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_95938a78a305c1001ea4ec748b7c26de_data, 1, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_95938a78a305c1001ea4ec748b7c26de_CBVResourceNames, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_95938a78a305c1001ea4ec748b7c26de_CBVResourceBindings, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_95938a78a305c1001ea4ec748b7c26de_CBVResourceCounts, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_95938a78a305c1001ea4ec748b7c26de_CBVResourceSets, 3, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_95938a78a305c1001ea4ec748b7c26de_TextureSRVResourceNames, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_95938a78a305c1001ea4ec748b7c26de_TextureSRVResourceBindings, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_95938a78a305c1001ea4ec748b7c26de_TextureSRVResourceCounts, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_95938a78a305c1001ea4ec748b7c26de_TextureSRVResourceSets, 5, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_95938a78a305c1001ea4ec748b7c26de_TextureUAVResourceNames, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_95938a78a305c1001ea4ec748b7c26de_TextureUAVResourceBindings, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_95938a78a305c1001ea4ec748b7c26de_TextureUAVResourceCounts, g_ffx_fsr3upscaler_prepare_inputs_pass_16bit_95938a78a305c1001ea4ec748b7c26de_TextureUAVResourceSets, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, },
};

