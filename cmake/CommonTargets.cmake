# ============================================================================
# Common target setup: C++17, warnings (/W4 zero-warning target for our code),
# project-root include (renderer/..., upscalers/...), third-party and Vulkan
# headers as SYSTEM/external so their warnings are suppressed on MSVC.
# ============================================================================
function(sr_common_setup TARGET)
    target_compile_features(${TARGET} PUBLIC cxx_std_17)

    if(MSVC)
        target_compile_options(${TARGET} PRIVATE /W4 /utf-8 /external:W0)
    else()
        target_compile_options(${TARGET} PRIVATE -Wall -Wextra)
    endif()

    target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR})

    # Single-header dependencies (cgltf / stb) — external, warnings silenced.
    target_include_directories(${TARGET} SYSTEM PRIVATE
        ${SR_THIRD_PARTY_DIR}/cgltf
        ${SR_THIRD_PARTY_DIR}/stb)

    # Vulkan SDK headers as external too.
    if(Vulkan_INCLUDE_DIRS)
        target_include_directories(${TARGET} SYSTEM PRIVATE ${Vulkan_INCLUDE_DIRS})
    endif()

    target_link_libraries(${TARGET} PUBLIC Vulkan::Vulkan)
endfunction()
