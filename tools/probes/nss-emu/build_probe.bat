@echo off
rem Build the NSS emu constant probe against the already-built SPIRV-Tools in
rem build-nss-emu.  Usage: build_probe.bat
setlocal
set "ROOT=%~dp0..\.."
set "EMU=%ROOT%\build-nss-emu"
call "%ROOT%\scripts\devcmd.bat" cl /MD /std:c++17 /EHsc /I "%EMU%\deps\SPIRV-Tools\include" /I "%EMU%\deps\SPIRV-Tools" /I "%EMU%\deps\SPIRV-Headers\include" /I "%EMU%\build\spirv-tools" "%~dp0const_probe.cpp" /Fe:"%~dp0const_probe.exe" /Fo:"%~dp0" /link /LIBPATH:"%EMU%\build\spirv-tools\source\opt" /LIBPATH:"%EMU%\build\spirv-tools\source" SPIRV-Tools-opt.lib SPIRV-Tools.lib
