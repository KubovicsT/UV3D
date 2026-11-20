@echo off
setlocal
set "UV3D_ROOT=%~dp0"
set "UV3D_RUNTIME=%UV3D_ROOT%runtime"
set "TEMP=%UV3D_RUNTIME%\tmp"
set "TMP=%UV3D_RUNTIME%\tmp"
set "PIP_CACHE_DIR=%UV3D_RUNTIME%\pip-cache"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH="
if not exist "%UV3D_RUNTIME%\python\python.exe" (
 echo UV3D runtime not installed. Run Install-UV3D.ps1 first.
 pause
 exit /b 1
)
pushd "%UV3D_ROOT%"
"%UV3D_RUNTIME%\python\python.exe" -m uv3d
set "exitcode=%ERRORLEVEL%"
popd
exit /b %exitcode%
