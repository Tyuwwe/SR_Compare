@echo off
rem Fetch the NSS Vulkan ML emulation layer's dependencies into
rem build-nss-emu\deps and apply the sr_test patches (upscalers\nss\patches).
rem Needs git + network.  Only required when build-nss-emu\deps is missing
rem (e.g. after a fresh copy of the source tree without build directories).
setlocal
set "ROOT=%~dp0.."
set "DEPS=%ROOT%\build-nss-emu\deps"
set "PATCHES=%ROOT%\upscalers\nss\patches"

if not exist "%DEPS%" mkdir "%DEPS%"

call :clone SPIRV-Headers https://github.com/KhronosGroup/SPIRV-Headers ad9184e || goto :fail
call :clone SPIRV-Tools   https://github.com/KhronosGroup/SPIRV-Tools   0539c81 || goto :fail
call :clone SPIRV-Cross   https://github.com/KhronosGroup/SPIRV-Cross   1a61695 || goto :fail
call :clone Vulkan-Headers https://github.com/KhronosGroup/Vulkan-Headers 8864cdc || goto :fail
call :clone glslang       https://github.com/KhronosGroup/glslang       275822a || goto :fail
call :clone json          https://github.com/nlohmann/json              9cca280 || goto :fail

rem SPIRV-Tools: backport TensorARM constant support (patch lives in-tree).
git -C "%DEPS%\SPIRV-Tools" apply --check "%PATCHES%\spirv-tools-tensorarm-constants.patch" >nul 2>&1
if %errorlevel%==0 (
    git -C "%DEPS%\SPIRV-Tools" apply "%PATCHES%\spirv-tools-tensorarm-constants.patch" || goto :fail
    echo applied spirv-tools-tensorarm-constants.patch
) else (
    echo SPIRV-Tools patch already applied or not applicable, skipping
)

rem Emulation layer sources ship in third_party\arm-nss\emulation-layer.
git -C "%ROOT%\third_party\arm-nss\emulation-layer" apply --check "%PATCHES%\emulation-layer-diagnostics.patch" >nul 2>&1
if %errorlevel%==0 (
    git -C "%ROOT%\third_party\arm-nss\emulation-layer" apply "%PATCHES%\emulation-layer-diagnostics.patch" || goto :fail
    echo applied emulation-layer-diagnostics.patch
) else (
    echo emulation-layer patch already applied or not applicable, skipping
)

echo deps ready in %DEPS%
exit /b 0

:clone
if exist "%DEPS%\%~1\.git" (
    echo %~1 already present, skipping
    exit /b 0
)
git clone %~2 "%DEPS%\%~1" || exit /b 1
git -C "%DEPS%\%~1" checkout %~3 || exit /b 1
exit /b 0

:fail
echo nss_emu_fetch_deps failed 1>&2
exit /b 1
