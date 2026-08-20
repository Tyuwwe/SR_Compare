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
