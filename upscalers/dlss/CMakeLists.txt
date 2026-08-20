# NVIDIA DLSS Super Resolution via the Streamline SDK (Vulkan path).
# Convention: plugin targets are named `upscaler_<dirname>` and are linked
# whole-archive into the umbrella `upscalers` lib by ../CMakeLists.txt.
#
# Registers three plugins sharing one process-wide Streamline instance:
#   "dlss-k" -> sl::DLSSPreset::ePresetK
#   "dlss-l" -> sl::DLSSPreset::ePresetL
#   "dlss-m" -> sl::DLSSPreset::ePresetM
add_library(upscaler_dlss STATIC DlssUpscaler.cpp SlContext.cpp)
sr_common_setup(upscaler_dlss)

set(SL_SDK_DIR ${SR_THIRD_PARTY_DIR}/streamline)
target_include_directories(upscaler_dlss SYSTEM PRIVATE ${SL_SDK_DIR}/include)
# Import library for sl.interposer.dll (exports the sl* API).  PUBLIC so the
# dependency propagates to the final executable through the umbrella lib.
target_link_libraries(upscaler_dlss PUBLIC ${SL_SDK_DIR}/lib/x64/sl.interposer.lib)

# Deploy the Streamline runtime next to sr_compare.exe.  Development-signed
# binaries are used while iterating (they may draw an on-screen NDA notice);
# for report numbers swap in the production set from bin/x64 instead.
# A $<TARGET_FILE_DIR:sr_compare> genex would create a target dependency cycle
# (sr_compare -> upscalers -> upscaler_dlss), so the well-known layout of the
# app target's output directory is used instead (multi-config generators put
# the exe under app/<Config>/).
# Production DLLs (no "DO NOT DISTRIBUTE" watermark) for report-quality
# captures; swap to bin/x64/development for debugging.
file(GLOB SL_RUNTIME_DLLS "${SL_SDK_DIR}/bin/x64/*.dll")
add_custom_command(TARGET upscaler_dlss POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/app/$<CONFIG>/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${SL_RUNTIME_DLLS}
        "${CMAKE_BINARY_DIR}/app/$<CONFIG>/")
