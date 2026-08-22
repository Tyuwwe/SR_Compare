# Vulkan renderer core + scene + deferred GBuffer/lighting + TAA input generation.
add_library(renderer STATIC
    core/VulkanContext.cpp
    core/VmaImpl.cpp
    core/Window.cpp
    core/VkUtil.cpp
    core/Swapchain.cpp
    core/TimestampQuery.cpp
    core/GpuProfiler.cpp
    core/MemoryBudget.cpp
    core/RenderGraph.cpp
    core/EngineConfig.cpp
    scene/Scene.cpp
    scene/Animation.cpp
    scene/ProceduralScene.cpp
    scene/GltfLoader.cpp
    scene/Ktx2.cpp
    scene/TextureStreamer.cpp
    scene/LodBuilder.cpp
    scene/Camera.cpp
    scene/CameraPath.cpp
    scene/SceneRegistry.cpp
    ibl/Ibl.cpp
    ibl/Probes.cpp
    ibl/SkyAtmosphere.cpp
    deferred/DeferredCore.cpp
    ColorGrading.cpp
    Renderer.cpp
    Screenshot.cpp
)

# meshoptimizer (zeux): meshopt_simplify for runtime LOD chain generation.
# Separate target so the project's /W4 zero-warning setup does not apply to
# third-party sources; its headers propagate as SYSTEM (external) includes.
add_library(meshoptimizer STATIC
    ${SR_THIRD_PARTY_DIR}/meshoptimizer/allocator.cpp
    ${SR_THIRD_PARTY_DIR}/meshoptimizer/simplifier.cpp
)
target_include_directories(meshoptimizer SYSTEM PUBLIC ${SR_THIRD_PARTY_DIR}/meshoptimizer)

sr_common_setup(renderer)
target_link_libraries(renderer PUBLIC meshoptimizer SDL3::SDL3)
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
    shaders/cluster_assign.comp
    shaders/occlusion_cull.comp
    shaders/ssao.comp
    shaders/ssao_blur.comp
    shaders/reactive_dilate.comp
    shaders/ssao_temporal.comp
    shaders/ssr_opaque.comp
    shaders/ssr_temporal.comp
    shaders/hiz_downsample.comp
    shaders/color_downsample.comp
    shaders/bloom_extract.comp
    shaders/bloom_downsample.comp
    shaders/bloom_upsample.comp
    shaders/bloom_composite.comp
    shaders/exposure_histogram.comp
    shaders/exposure_solve.comp
    shaders/ibl_equirect.comp
    shaders/ibl_irradiance.comp
    shaders/ibl_prefilter.comp
    shaders/ibl_brdf_lut.comp
    shaders/sky_transmittance.comp
    shaders/sky_multiscatter.comp
    shaders/sky_render.comp
    shaders/volfog_inject.comp
    shaders/volfog_light.comp
    shaders/volfog_temporal.comp
    shaders/volfog_march.comp
    shaders/volfog_composite.comp
    shaders/motion_blur_tilemax.comp
    shaders/motion_blur_neighborhood.comp
    shaders/motion_blur_gather.comp
    shaders/dof_coc.comp
    shaders/dof_gather.comp
    shaders/dof_composite.comp
    shaders/postfx_copyback.comp
)
set(RENDERER_SPV "")
foreach(SHADER ${RENDERER_SHADERS})
    if(SHADER STREQUAL "shaders/present.frag")
        sr_compile_shader(SPV ${SHADER} "${SR_TONE_MAP_GLSL}" "${SR_SHADER_INCLUDE_DIR}/grading.glsl")
    elseif(SHADER STREQUAL "shaders/transparent.frag" OR SHADER STREQUAL "shaders/transparent_gt.frag")
        sr_compile_shader(SPV ${SHADER} "${SR_SHADER_INCLUDE_DIR}/ssr.glsl" "${SR_SHADER_INCLUDE_DIR}/brdf.glsl" "${SR_SHADER_INCLUDE_DIR}/volfog.glsl")
    elseif(SHADER STREQUAL "shaders/ssr_opaque.comp")
        sr_compile_shader(SPV ${SHADER} "${SR_SHADER_INCLUDE_DIR}/ssr.glsl" "${SR_SHADER_INCLUDE_DIR}/brdf.glsl" "${SR_SHADER_INCLUDE_DIR}/probes.glsl")
    elseif(SHADER STREQUAL "shaders/lighting.frag")
        sr_compile_shader(SPV ${SHADER} "${SR_SHADER_INCLUDE_DIR}/brdf.glsl" "${SR_SHADER_INCLUDE_DIR}/ssr.glsl" "${SR_SHADER_INCLUDE_DIR}/probes.glsl")
    elseif(SHADER STREQUAL "shaders/ibl_brdf_lut.comp")
        sr_compile_shader(SPV ${SHADER} "${SR_SHADER_INCLUDE_DIR}/brdf.glsl")
    elseif(SHADER MATCHES "^shaders/volfog_.*\\.comp$")
        sr_compile_shader(SPV ${SHADER} "${SR_SHADER_INCLUDE_DIR}/volfog.glsl")
    elseif(SHADER MATCHES "^shaders/sky_.*\\.comp$")
        sr_compile_shader(SPV ${SHADER} "${SR_SHADER_INCLUDE_DIR}/atmosphere.glsl")
    else()
        sr_compile_shader(SPV ${SHADER})
    endif()
    list(APPEND RENDERER_SPV ${SPV})
endforeach()
add_custom_target(renderer_shaders DEPENDS ${RENDERER_SPV})
add_dependencies(renderer renderer_shaders)
