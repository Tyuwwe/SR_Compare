# Arm Neural Super Sampling (Neural Graphics SDK for Game Engines).
# The plugin talks to the prebuilt SDK runtime (ngsdk_windows_x64.dll) through
# the FFX-style C API, loaded dynamically at init time so the executable works
# without the DLL present.  The ML extensions (VK_ARM_tensors /
# VK_ARM_data_graph) are provided on PC by Arm's ML Emulation Layer; see
# run_with_nss.bat for the required runtime environment.
add_library(upscaler_nss STATIC NssUpscaler.cpp NfruFrameGen.cpp)
sr_common_setup(upscaler_nss)
target_compile_definitions(upscaler_nss PRIVATE SR_SHADER_DIR="${SR_SHADER_OUT_DIR}/")

# Bake runtime locations into the plugin so `--upscaler nss` works without
# run_with_nss.bat: the emulation-layer manifests are added to
# VK_ADD_LAYER_PATH from the instance-requirements hook (which runs before
# vkCreateInstance), and the SDK runtime DLL is loaded by absolute path.
target_compile_definitions(upscaler_nss PRIVATE
    SR_NSS_RUNTIME_DIR="${SR_THIRD_PARTY_DIR}/arm-nss/prebuilt_binaries"
    SR_NSS_EMU_GRAPH_DIR="${CMAKE_SOURCE_DIR}/build-nss-emu/build/graph"
    SR_NSS_EMU_TENSOR_DIR="${CMAKE_SOURCE_DIR}/build-nss-emu/build/tensor")

# SDK public headers (FFX API + NSS descriptors). ffx_api_vk.h also includes
# the SDK host backend header, hence both include roots.
target_include_directories(upscaler_nss SYSTEM PRIVATE
    ${SR_THIRD_PARTY_DIR}/arm-nss/ffx-api/include
    ${SR_THIRD_PARTY_DIR}/arm-nss/sdk/include)

sr_compile_shader(NSS_CONVERT_IN_SPV nss_convert_in.comp)
sr_compile_shader(NSS_CONVERT_OUT_SPV nss_convert_out.comp)
add_custom_target(nss_shaders DEPENDS ${NSS_CONVERT_IN_SPV} ${NSS_CONVERT_OUT_SPV})
add_dependencies(upscaler_nss nss_shaders)
