@echo off
rem Configure the NSS Vulkan ML emulation layer (Ninja + MSVC).
rem Deps must exist in build-nss-emu\deps (run nss_emu_fetch_deps.bat once
rem after a fresh copy of the tree).
setlocal
set "ROOT=%~dp0.."
set "EMU=%ROOT%\build-nss-emu"
if not exist "%EMU%\deps\SPIRV-Tools" (
    echo deps missing - run scripts\nss_emu_fetch_deps.bat first 1>&2
    exit /b 1
)
call "%~dp0find_cmake.bat" || exit /b 1
call "%~dp0devcmd.bat" "%CMAKE%" -S "%ROOT%\third_party\arm-nss\emulation-layer" -B "%EMU%\build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="%ROOT%\third_party\arm-nss\emulation-layer\cmake\toolchain\windows-msvc.cmake" -DMSVC=ON -DVULKAN_HEADERS_PATH="%EMU%\deps\Vulkan-Headers" -DSPIRV_TOOLS_PATH="%EMU%\deps\SPIRV-Tools" -DSPIRV-Headers_SOURCE_DIR="%EMU%\deps\SPIRV-Headers" -DSPIRV_CROSS_PATH="%EMU%\deps\SPIRV-Cross" -DGLSLANG_PATH="%EMU%\deps\glslang" -DJSON_PATH="%EMU%\deps\json" -DVMEL_TESTS_ENABLE=OFF
