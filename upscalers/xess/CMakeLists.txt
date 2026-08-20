# Intel XeSS (Vulkan path) upscaler plugin.
# Convention: plugin targets are named `upscaler_<dirname>` and are linked
# whole-archive into the umbrella `upscalers` lib by ../CMakeLists.txt.
add_library(upscaler_xess STATIC XessUpscaler.cpp)
sr_common_setup(upscaler_xess)

set(XESS_SDK_DIR ${SR_THIRD_PARTY_DIR}/xess)
target_include_directories(upscaler_xess SYSTEM PRIVATE ${XESS_SDK_DIR}/inc)
target_link_libraries(upscaler_xess PRIVATE ${XESS_SDK_DIR}/lib/libxess.lib)

# Deploy libxess.dll next to the sr_compare executable.  A
# $<TARGET_FILE_DIR:sr_compare> genex would create a target dependency cycle
# (sr_compare -> upscalers -> upscaler_xess), so the well-known layout of the
# app target's output directory is used instead (multi-config generators put
# the exe under app/<Config>/).
add_custom_command(TARGET upscaler_xess POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/app/$<CONFIG>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${XESS_SDK_DIR}/bin/libxess.dll"
        "${CMAKE_BINARY_DIR}/app/$<CONFIG>/")
