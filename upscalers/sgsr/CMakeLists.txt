# Snapdragon SGSR1 (spatial) + SGSR2 (temporal, 3-pass CS) upscalers.
# Convention: plugin targets are named `upscaler_<dirname>` and are linked
# whole-archive into the umbrella `upscalers` lib by ../CMakeLists.txt.
add_library(upscaler_sgsr STATIC Sgsr1Upscaler.cpp Sgsr2Upscaler.cpp)
sr_common_setup(upscaler_sgsr)
target_compile_definitions(upscaler_sgsr PRIVATE SR_SHADER_DIR="${SR_SHADER_OUT_DIR}/")

sr_compile_shader(SGSR1_VERT_SPV sgsr1.vert)
sr_compile_shader(SGSR1_FRAG_SPV sgsr1.frag)
sr_compile_shader(SGSR2_MV_SPV sgsr2_mv_encode.comp)
sr_compile_shader(SGSR2_CONVERT_SPV sgsr2_convert.comp)
sr_compile_shader(SGSR2_ACTIVATE_SPV sgsr2_activate.comp)
sr_compile_shader(SGSR2_UPSCALE_SPV sgsr2_upscale.comp)
add_custom_target(sgsr_shaders DEPENDS
    ${SGSR1_VERT_SPV} ${SGSR1_FRAG_SPV}
    ${SGSR2_MV_SPV} ${SGSR2_CONVERT_SPV} ${SGSR2_ACTIVATE_SPV} ${SGSR2_UPSCALE_SPV})
add_dependencies(upscaler_sgsr sgsr_shaders)
