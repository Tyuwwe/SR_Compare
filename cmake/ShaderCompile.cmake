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

# sr_compile_shader(<out_var> <glsl_source>)
# Creates a custom command that compiles <glsl_source> to
# ${SR_SHADER_OUT_DIR}/<basename>.spv and returns the .spv path in <out_var>.
# Callers collect the outputs and add_dependencies() their target on them.
function(sr_compile_shader OUT_VAR GLSL_SOURCE)
    get_filename_component(SRC_ABS "${GLSL_SOURCE}" ABSOLUTE)
    get_filename_component(FNAME "${GLSL_SOURCE}" NAME)
    set(SPV "${SR_SHADER_OUT_DIR}/${FNAME}.spv")
    add_custom_command(
        OUTPUT "${SPV}"
        COMMAND "${SR_GLSLANG_VALIDATOR}" -V "${SRC_ABS}" -o "${SPV}" --target-env vulkan1.3
        DEPENDS "${SRC_ABS}"
        COMMENT "Compiling shader ${FNAME} -> SPIR-V"
        VERBATIM)
    set(${OUT_VAR} "${SPV}" PARENT_SCOPE)
endfunction()
