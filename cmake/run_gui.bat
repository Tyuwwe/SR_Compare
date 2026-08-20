@echo off
rem One-click GUI launcher.  Lives next to sr_compare.exe in the packaged
rem folder (copied there by the build); works after moving the folder anywhere.
start "" "%~dp0sr_compare.exe" gui
