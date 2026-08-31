cmake_minimum_required(VERSION 3.24)
project(sr_compare LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

option(SR_WITH_FSR  "AMD FSR1/2/3"        ON)
option(SR_WITH_SGSR "Snapdragon SGSR1/2"  ON)
option(SR_WITH_XESS "Intel XeSS"          ON)
option(SR_WITH_DLSS "NVIDIA DLSS (Streamline)" ON)
option(SR_WITH_NSS  "Arm Neural Super Sampling" ON)

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "Only x64 is supported")
endif()

find_package(Vulkan REQUIRED)

# SDK 根目录（scripts/fetch_sdks.py 下载至此）
set(SR_THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# SDL3 windowing/input layer (fetch_sdks.py task "sdl3": headers + prebuilt
# VC x64 import lib + runtime DLL under third_party/sdl3).
if(NOT EXISTS "${SR_THIRD_PARTY_DIR}/sdl3/lib/x64/SDL3.lib")
    message(FATAL_ERROR
        "SDL3 not found under third_party/sdl3 -- fetch it first:\n"
        "  python scripts/fetch_sdks.py --only sdl3")
endif()
add_library(SDL3::SDL3 SHARED IMPORTED GLOBAL)
set_target_properties(SDL3::SDL3 PROPERTIES
    IMPORTED_LOCATION "${SR_THIRD_PARTY_DIR}/sdl3/lib/x64/SDL3.dll"
    IMPORTED_IMPLIB "${SR_THIRD_PARTY_DIR}/sdl3/lib/x64/SDL3.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${SR_THIRD_PARTY_DIR}/sdl3/include"
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${SR_THIRD_PARTY_DIR}/sdl3/include")

# GLSL -> SPIR-V 编译辅助（renderer / upscalers 共用）
include(cmake/ShaderCompile.cmake)

# 目标级通用配置（C++17 / /W4 / SYSTEM includes）
include(cmake/CommonTargets.cmake)

add_subdirectory(renderer)
add_subdirectory(upscalers)
add_subdirectory(compare)
add_subdirectory(bench)
add_subdirectory(gui)
add_subdirectory(app)
add_subdirectory(spike/dlss5)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
