# Dear ImGui (docking branch) from third_party/imgui, built as-is with
# warnings silenced (third-party code; our /W4 zero-warning rule does not
# apply to it).
set(IMGUI_DIR ${SR_THIRD_PARTY_DIR}/imgui)
add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backends/imgui_impl_win32.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui SYSTEM PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
if(Vulkan_INCLUDE_DIRS)
    target_include_directories(imgui SYSTEM PUBLIC ${Vulkan_INCLUDE_DIRS})
endif()
target_link_libraries(imgui PUBLIC Vulkan::Vulkan)
if(MSVC)
    target_compile_options(imgui PRIVATE /W0 /utf-8)
endif()

# Interactive GUI mode (`sr_compare gui`).  Rendering is organized like
# compare mode and reuses its shaders (compare_shaders) plus the renderer's.
add_library(gui STATIC GuiApp.cpp BenchRunner.cpp GpuProfilerWindow.cpp)
sr_common_setup(gui)
target_compile_definitions(gui PRIVATE SR_SHADER_DIR="${SR_SHADER_OUT_DIR}/")
target_link_libraries(gui PUBLIC renderer upscalers imgui)
add_dependencies(gui renderer_shaders compare_shaders)
