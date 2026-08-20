@echo off
rem Locate the latest Visual Studio installation that has the C++ workload.
rem Sets VSINSTALL for the caller.  Requires no hardcoded paths: vswhere lives
rem at a fixed installer location on any machine with VS 2017+.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo find_vs: vswhere.exe not found at "%VSWHERE%" 1>&2
    echo find_vs: install Visual Studio with the "Desktop development with C++" workload 1>&2
    exit /b 1
)
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
if not defined VSINSTALL (
    echo find_vs: no Visual Studio with the C++ workload found 1>&2
    exit /b 1
)
exit /b 0
