@echo off
rem One-click full build: configure on first run, then build Release.
rem Extra args are forwarded to the build step, e.g.:
rem   build_all.bat --target sr_compare
setlocal
set "ROOT=%~dp0"
call "%ROOT%scripts\find_cmake.bat" || exit /b 1
if not exist "%ROOT%build\CMakeCache.txt" (
    echo configuring...
    "%CMAKE%" -S "%ROOT%" -B "%ROOT%build" || exit /b 1
)
"%CMAKE%" --build "%ROOT%build" --config Release %* || exit /b 1
echo done: %ROOT%build\app\Release\sr_compare.exe
