@echo off
rem Run a command inside the MSVC x64 developer environment.
rem Usage: devcmd.bat <command...>
setlocal
if not defined VSINSTALL (
    call "%~dp0find_vs.bat" || exit /b 1
)
call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
%*
