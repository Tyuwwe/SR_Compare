# Vulkan renderer core + scene + deferred GBuffer/lighting + TAA input generation.
add_library(renderer STATIC
    core/VulkanContext.cpp
    core/Window.cpp
    core/VkUtil.cpp
    core/Swapchain.cpp
    core/TimestampQuery.cpp
    core/MemoryBudget.cpp
    scene/Scene.cpp
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
    shaders/gbuffer.frag
    shaders/gbuffer_gt.frag
    shaders/lighting.frag
    shaders/transparent.vert
    shaders/transparent.frag
    shaders/transparent_gt.frag
    shaders/ssao.comp
    shaders/ssao_blur.comp
    shaders/ibl_equirect.comp
    shaders/ibl_irradiance.comp
    shaders/ibl_prefilter.comp
    shaders/ibl_brdf_lut.comp
)
set(RENDERER_SPV "")
foreach(SHADER ${RENDERER_SHADERS})
    sr_compile_shader(SPV ${SHADER})
    list(APPEND RENDERER_SPV ${SPV})
endforeach()
add_custom_target(renderer_shaders DEPENDS ${RENDERER_SPV})
add_dependencies(renderer renderer_shaders)
