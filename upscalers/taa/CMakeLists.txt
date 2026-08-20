# TAA baseline upscaler (implements sr::IUpscaler).
# Convention: plugin targets are named `upscaler_<dirname>` and are linked
# whole-archive into the umbrella `upscalers` lib by ../CMakeLists.txt.
add_library(upscaler_taa STATIC TaaUpscaler.cpp)
sr_common_setup(upscaler_taa)
target_compile_definitions(upscaler_taa PRIVATE SR_SHADER_DIR="${SR_SHADER_OUT_DIR}/")

sr_compile_shader(TAA_SPV taa.comp)
add_custom_target(taa_shaders DEPENDS ${TAA_SPV})
add_dependencies(upscaler_taa taa_shaders)
