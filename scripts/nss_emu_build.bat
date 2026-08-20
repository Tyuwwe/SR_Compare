@echo off
rem Build the NSS Vulkan ML emulation layer (after nss_emu_configure.bat).
setlocal
set "ROOT=%~dp0.."
call "%~dp0find_cmake.bat" || exit /b 1
call "%~dp0devcmd.bat" "%CMAKE%" --build "%ROOT%\build-nss-emu\build"
