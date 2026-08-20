# Split-screen quality comparison mode.  Links `upscalers` so the default
# column set matches `viewer --list-upscalers`.
add_library(compare STATIC CompareMode.cpp CompareApp.cpp)
sr_common_setup(compare)
target_link_libraries(compare PUBLIC renderer upscalers)
target_compile_definitions(compare PRIVATE SR_SHADER_DIR="${SR_SHADER_OUT_DIR}/")

# Compare-specific GLSL (column compose / swapchain copy / metric reduction).
# Scene shaders are reused from the renderer target (renderer_shaders).
set(COMPARE_SHADERS
    shaders/compare_compose.frag
    shaders/compare_copy.frag
    shaders/compare_metrics_blocks.comp
    shaders/compare_metrics_reduce.comp
)
set(COMPARE_SPV "")
foreach(SHADER ${COMPARE_SHADERS})
    sr_compile_shader(SPV ${SHADER})
    list(APPEND COMPARE_SPV ${SPV})
endforeach()
add_custom_target(compare_shaders DEPENDS ${COMPARE_SPV})
add_dependencies(compare compare_shaders renderer_shaders)
