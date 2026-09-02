@echo off
rem One-click reflection-probe bake for every scene with registry placements.
rem Writes/refresh the .probes file next to each scene (assets/...); stale or
rem missing bake files leave the probes disabled (glass falls back to the
rem global env), so rerun this after probe placements change in
rem renderer/scene/SceneRegistry.cpp.
rem
rem Requires a built viewer: build_all.bat first, or this script builds the
rem sr_compare target itself when the exe is missing.
setlocal
set "ROOT=%~dp0.."
set "EXE=%ROOT%\build\app\Release\sr_compare.exe"

if not exist "%EXE%" (
    echo sr_compare.exe not found, building...
    call "%ROOT%\build_all.bat" --target sr_compare || exit /b 1
)

set "FAILED="
call :bake sponza          || set "FAILED=%FAILED% sponza"
call :bake bistro_exterior || set "FAILED=%FAILED% bistro_exterior"
call :bake bistro_interior || set "FAILED=%FAILED% bistro_interior"
call :bake ssrlab          || set "FAILED=%FAILED% ssrlab"
call :bake procedural      || set "FAILED=%FAILED% procedural(boxes)"

if not "%FAILED%" == "" (
    echo.
    echo bake FAILED for:%FAILED%
    exit /b 1
)
echo.
echo all probe bakes succeeded.
exit /b 0

:bake
echo.
echo === baking probes: %1 ===
"%EXE%" viewer --scene %1 --bake-probes
exit /b %errorlevel%
