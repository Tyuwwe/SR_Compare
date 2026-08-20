@echo off
rem Locate a cmake executable.  Sets CMAKE for the caller.
rem Preference: cmake on PATH, then the CMake bundled with Visual Studio.
where cmake >nul 2>&1
if %errorlevel%==0 (
    set "CMAKE=cmake"
    exit /b 0
)
if not defined VSINSTALL (
    call "%~dp0find_vs.bat" || exit /b 1
)
set "CMAKE=%VSINSTALL%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if not exist "%CMAKE%" (
    echo find_cmake: cmake not on PATH and VS-bundled CMake missing at "%CMAKE%" 1>&2
    exit /b 1
)
exit /b 0
