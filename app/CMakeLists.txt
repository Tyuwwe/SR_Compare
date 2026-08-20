add_executable(sr_compare main.cpp)
sr_common_setup(sr_compare)
target_link_libraries(sr_compare PRIVATE renderer upscalers compare bench gui)

# ---------------------------------------------------------------------------
# Packaged layout next to the exe (so the app runs when copied elsewhere):
#   sr_compare.exe
#   *.dll                       (upscaler runtimes, copied by their modules)
#   ngsdk_windows_x64.dll       (Arm NSS runtime)
#   nss-emu/graph|tensor/       (Arm ML emulation layer for PC)
#   shaders/*.spv               (all SPIR-V, resolved exe-relative)
#   vcruntime140*.dll, msvcp140.dll  (CRT, avoids requiring the redist installer)
# ---------------------------------------------------------------------------
set(SR_EXE_DIR "$<TARGET_FILE_DIR:sr_compare>")

add_custom_command(TARGET sr_compare POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${SR_SHADER_OUT_DIR}" "${SR_EXE_DIR}/shaders")

# One-click GUI launcher next to the exe (part of the packaged layout).
add_custom_command(TARGET sr_compare POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/cmake/run_gui.bat" "${SR_EXE_DIR}/run_gui.bat")

if(EXISTS "${SR_THIRD_PARTY_DIR}/arm-nss/prebuilt_binaries/ngsdk_windows_x64.dll")
    add_custom_command(TARGET sr_compare POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SR_THIRD_PARTY_DIR}/arm-nss/prebuilt_binaries/ngsdk_windows_x64.dll"
            "${SR_EXE_DIR}/ngsdk_windows_x64.dll")
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/build-nss-emu/build/graph/VkLayer_Graph.dll")
    # Only the layer DLL + manifest are needed at runtime (the emulation
    # shaders are compiled into the DLL).
    add_custom_command(TARGET sr_compare POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SR_EXE_DIR}/nss-emu/graph" "${SR_EXE_DIR}/nss-emu/tensor"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/build-nss-emu/build/graph/VkLayer_Graph.dll"
            "${CMAKE_SOURCE_DIR}/build-nss-emu/build/graph/VkLayer_Graph.json"
            "${SR_EXE_DIR}/nss-emu/graph/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/build-nss-emu/build/tensor/VkLayer_Tensor.dll"
            "${CMAKE_SOURCE_DIR}/build-nss-emu/build/tensor/VkLayer_Tensor.json"
            "${SR_EXE_DIR}/nss-emu/tensor/")
endif()

# CRT redist DLLs (optional; target machines without the VC++ redist installed
# would otherwise fail to start).  Located relative to the compiler.
get_filename_component(_vc_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
get_filename_component(_vc_msvc "${_vc_bin}/../../.." ABSOLUTE)
file(GLOB _vc_crt_dirs "${_vc_msvc}/../../../Redist/MSVC/*/x64/Microsoft.VC*.CRT")
if(_vc_crt_dirs)
    list(GET _vc_crt_dirs 0 _vc_crt)
    add_custom_command(TARGET sr_compare POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_vc_crt}/vcruntime140.dll"
            "${_vc_crt}/vcruntime140_1.dll"
            "${_vc_crt}/msvcp140.dll"
            "${SR_EXE_DIR}/")
    message(STATUS "CRT redist copied from ${_vc_crt}")
endif()
