# ============================================================================
# GLSL -> SPIR-V compile helpers.
# Every shader module in this project (scene, TAA, and future upscalers) is
# compiled at build time by the Vulkan SDK's glslangValidator into a .spv file
# under ${CMAKE_BINARY_DIR}/shaders.  At runtime the .spv file is loaded by name
# (see SR_SHADER_DIR compile definition), so all modules share one mechanism.
# ============================================================================

find_program(SR_GLSLANG_VALIDATOR
    NAMES glslangValidator glslangValidator.exe
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")

if(NOT SR_GLSLANG_VALIDATOR)
    message(FATAL_ERROR "glslangValidator not found. Set VULKAN_SDK or add it to PATH.")
endif()

set(SR_SHADER_OUT_DIR "${CMAKE_BINARY_DIR}/shaders" CACHE INTERNAL "SPIR-V output directory")
file(MAKE_DIRECTORY "${SR_SHADER_OUT_DIR}")

# Shared GLSL include root (present/compose/metrics #include "tonemap.glsl").
set(SR_SHADER_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/renderer/shaders")
set(SR_TONE_MAP_GLSL "${SR_SHADER_INCLUDE_DIR}/tonemap.glsl")

# sr_compile_shader(<out_var> <glsl_source> [INCLUDE_DIR <dir>] [extra_depends...])
# Creates a custom command that compiles <glsl_source> to
# ${SR_SHADER_OUT_DIR}/<basename>.spv and returns the .spv path in <out_var>.
# Callers collect the outputs and add_dependencies() their target on them.
# INCLUDE_DIR adds one extra glslangValidator -I root (e.g. FidelityFX GPU
# headers for the TAA CAS pass).
#
# GLSL #include'd headers are NOT discovered automatically: callers must pass
# them as extra arguments after <glsl_source> so they appear in the custom
# command's DEPENDS and trigger a recompile when a header changes.
function(sr_compile_shader OUT_VAR GLSL_SOURCE)
    cmake_parse_arguments(SR "" "INCLUDE_DIR" "" ${ARGN})
    get_filename_component(SRC_ABS "${GLSL_SOURCE}" ABSOLUTE)
    get_filename_component(FNAME "${GLSL_SOURCE}" NAME)
    set(SPV "${SR_SHADER_OUT_DIR}/${FNAME}.spv")

    # Outputs are flattened to a basename, so two source directories can map to
    # the same .spv; reject that collision instead of silently overwriting.
    get_property(_sr_used_outputs GLOBAL PROPERTY SR_SHADER_OUTPUTS)
    list(FIND _sr_used_outputs "${SPV}" _sr_dup)
    if(_sr_dup GREATER_EQUAL 0)
        message(FATAL_ERROR "Shader output collision on ${SPV}: two sources share the same basename. Rename one of them.")
    endif()
    set_property(GLOBAL APPEND PROPERTY SR_SHADER_OUTPUTS "${SPV}")

    set(_sr_extra_inc "")
    if(SR_INCLUDE_DIR)
        set(_sr_extra_inc "-I${SR_INCLUDE_DIR}")
    endif()

    add_custom_command(
        OUTPUT "${SPV}"
        COMMAND "${SR_GLSLANG_VALIDATOR}" -V "-I${SR_SHADER_INCLUDE_DIR}" ${_sr_extra_inc} "${SRC_ABS}" -o "${SPV}" --target-env vulkan1.3
        DEPENDS "${SRC_ABS}" ${SR_UNPARSED_ARGUMENTS}
        COMMENT "Compiling shader ${FNAME} -> SPIR-V"
        VERBATIM)
    set(${OUT_VAR} "${SPV}" PARENT_SCOPE)
endfunction()
