cmake_minimum_required(VERSION 3.24)
project(xess_probe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Vulkan REQUIRED)
add_executable(xess_probe main.cpp)
target_include_directories(xess_probe SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/xess/inc)
target_link_libraries(xess_probe PRIVATE Vulkan::Vulkan ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/xess/lib/libxess.lib)
add_custom_command(TARGET xess_probe POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/xess/bin/libxess.dll
        $<TARGET_FILE_DIR:xess_probe>)
