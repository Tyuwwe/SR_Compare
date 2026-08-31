# DLSS 5 Neural Rendering (DLSSNR) technology-validation spike.
# Standalone exe: does NOT link renderer/upscalers.
#
# Modes vulkan/d3d12 load nvngx_dlssnr.dll at runtime via LoadLibrary +
# GetProcAddress (no NGX import lib - 310.x NGX DLLs only export the
# "snippet" ABI, so the spike implements the NVSDK_NGX_Parameter vtable
# itself). Mode "glue" is the official NGX route and needs nvsdk_ngx_d.lib
# from the NVIDIA DLSS SDK (not committed; point DLSS5_NGX_SDK_LIB_DIR at the
# SDK's lib/Windows_x86_64/x64 directory). See the investigation notes at the
# top of main.cpp.
add_executable(dlss5_spike main.cpp)
sr_common_setup(dlss5_spike)
target_include_directories(dlss5_spike SYSTEM PRIVATE
    ${SR_THIRD_PARTY_DIR}/ngx/include
    ${SR_THIRD_PARTY_DIR}/streamline/include)
target_link_libraries(dlss5_spike PRIVATE d3d12 dxgi version)

set(DLSS5_NGX_SDK_LIB_DIR "" CACHE PATH
    "Directory containing nvsdk_ngx_d.lib (NVIDIA DLSS SDK lib/Windows_x86_64/x64)")
if(NOT DLSS5_NGX_SDK_LIB_DIR AND
   EXISTS "C:/Users/Tyuwwe/AppData/Local/Temp/dlsssdk/lib/Windows_x86_64/x64/nvsdk_ngx_d.lib")
    # Local convenience default: sparse checkout of github.com/NVIDIA/DLSS.
    set(DLSS5_NGX_SDK_LIB_DIR "C:/Users/Tyuwwe/AppData/Local/Temp/dlsssdk/lib/Windows_x86_64/x64")
endif()
if(DLSS5_NGX_SDK_LIB_DIR)
    find_library(DLSS5_NVSDK_NGX_D_LIB nvsdk_ngx_d PATHS "${DLSS5_NGX_SDK_LIB_DIR}" NO_DEFAULT_PATH)
endif()
if(DLSS5_NVSDK_NGX_D_LIB)
    target_link_libraries(dlss5_spike PRIVATE "${DLSS5_NVSDK_NGX_D_LIB}")
    target_compile_definitions(dlss5_spike PRIVATE DLSS5_HAVE_GLUE=1)
    message(STATUS "dlss5_spike: NGX glue enabled (${DLSS5_NVSDK_NGX_D_LIB})")
else()
    message(STATUS "dlss5_spike: nvsdk_ngx_d.lib not found, --api glue disabled")
endif()
