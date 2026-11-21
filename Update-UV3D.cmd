@echo off
setlocal
title UV3D - Update from GitHub main
cd /d "%~dp0"
echo UV3D - updating from the latest main branch...
echo Close UV3D before running this script.
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Update-UV3D.ps1" %*
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" (
  echo Update FAILED, code %RC%. See workspace\logs\update.log.
) else (
  echo Update finished successfully.
)
echo.
pause
exit /b %RC%
