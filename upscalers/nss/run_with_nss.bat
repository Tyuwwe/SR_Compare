@echo off
rem ==========================================================================
rem Launch sr_compare with Arm NSS support on PC:
rem   * VK_ADD_LAYER_PATH points the Vulkan loader at the built ML Emulation
rem     Layer manifests (VkLayer_Graph.json / VkLayer_Tensor.json), which sit
rem     next to their DLLs in build-nss-emu\build\{graph,tensor}.
rem   * PATH is extended with the prebuilt Neural Graphics SDK runtime
rem     (ngsdk_windows_x64.dll), loaded by the nss plugin at init time.
rem Both must be set BEFORE the process starts (the Vulkan instance is created
rem at startup), which is why this wrapper exists instead of in-process setup.
rem
rem Usage: run_with_nss.bat [sr_compare arguments...]
rem   e.g. run_with_nss.bat viewer --upscaler nss --frames 30 --render-scale 0.5 --screenshot out.png
rem ==========================================================================
setlocal
set "ROOT=%~dp0..\.."
set "BUILD_DIR=%ROOT%\build-nss"
if defined SR_BUILD_DIR set "BUILD_DIR=%ROOT%\%SR_BUILD_DIR%"

set "VK_ADD_LAYER_PATH=%ROOT%\build-nss-emu\build\graph;%ROOT%\build-nss-emu\build\tensor"
set "PATH=%ROOT%\third_party\arm-nss\prebuilt_binaries;%PATH%"

set "EXE=%BUILD_DIR%\app\Release\sr_compare.exe"
if not exist "%EXE%" (
    echo run_with_nss.bat: %EXE% not found - build first, or set SR_BUILD_DIR 1>&2
    exit /b 1
)
"%EXE%" %*
