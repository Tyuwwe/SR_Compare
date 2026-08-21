# Vulkan renderer core + scene + deferred GBuffer/lighting + TAA input generation.
add_library(renderer STATIC
    core/VulkanContext.cpp
    core/VmaImpl.cpp
    core/Window.cpp
    core/VkUtil.cpp
    core/Swapchain.cpp
    core/TimestampQuery.cpp
    core/MemoryBudget.cpp
    scene/Scene.cpp
    scene/Animation.cpp
    scene/ProceduralScene.cpp
    scene/GltfLoader.cpp
    scene/Camera.cpp
    scene/CameraPath.cpp
    scene/SceneRegistry.cpp
    ibl/Ibl.cpp
    deferred/DeferredCore.cpp
    Renderer.cpp
    Screenshot.cpp
)

sr_common_setup(renderer)
target_compile_definitions(renderer PRIVATE SR_SHADER_DIR="${SR_SHADER_OUT_DIR}/")

# Compile the renderer's GLSL to SPIR-V at build time.  The deferred
# gbuffer/lighting shaders are shared by viewer, compare and gui (DeferredCore).
set(RENDERER_SHADERS
    shaders/fullscreen.vert
    shaders/present.frag
    shaders/gbuffer.vert
    shaders/gbuffer_skinned.vert
    shaders/gbuffer.frag
    shaders/gbuffer_gt.frag
    shaders/lighting.frag
    shaders/transparent.vert
    shaders/transparent.frag
    shaders/transparent_gt.frag
    shaders/shadow_depth.vert
    shaders/shadow_depth_skinned.vert
    shaders/shadow_depth.frag
    shaders/ssao.comp
    shaders/ssao_blur.comp
    shaders/ssao_temporal.comp
    shaders/ssr_opaque.comp
    shaders/ssr_temporal.comp
    shaders/hiz_downsample.comp
    shaders/color_downsample.comp
    shaders/bloom_extract.comp
    shaders/bloom_blur.comp
    shaders/bloom_composite.comp
    shaders/exposure_histogram.comp
    shaders/exposure_solve.comp
    shaders/ibl_equirect.comp
    shaders/ibl_irradiance.comp
    shaders/ibl_prefilter.comp
    shaders/ibl_brdf_lut.comp
)
set(RENDERER_SPV "")
foreach(SHADER ${RENDERER_SHADERS})
    if(SHADER STREQUAL "shaders/present.frag")
        sr_compile_shader(SPV ${SHADER} "${SR_TONE_MAP_GLSL}")
    elseif(SHADER STREQUAL "shaders/transparent.frag" OR SHADER STREQUAL "shaders/transparent_gt.frag" OR SHADER STREQUAL "shaders/ssr_opaque.comp")
        sr_compile_shader(SPV ${SHADER} "${SR_SHADER_INCLUDE_DIR}/ssr.glsl" "${SR_SHADER_INCLUDE_DIR}/brdf.glsl")
    elseif(SHADER STREQUAL "shaders/lighting.frag" OR SHADER STREQUAL "shaders/ibl_brdf_lut.comp")
        sr_compile_shader(SPV ${SHADER} "${SR_SHADER_INCLUDE_DIR}/brdf.glsl")
    else()
        sr_compile_shader(SPV ${SHADER})
    endif()
    list(APPEND RENDERER_SPV ${SPV})
endforeach()
add_custom_target(renderer_shaders DEPENDS ${RENDERER_SPV})
add_dependencies(renderer renderer_shaders)
