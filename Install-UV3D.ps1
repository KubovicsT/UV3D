param([switch]$NoLaunch)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runtime = Join-Path $Root 'runtime'
$Workspace = Join-Path $Root 'workspace'
$Downloads = Join-Path $Runtime 'downloads'
$GitDir = Join-Path $Runtime 'git'
$PythonDir = Join-Path $Runtime 'python'
$Git = Join-Path $GitDir 'cmd\git.exe'
$Python = Join-Path $PythonDir 'python.exe'
foreach ($dir in @($Runtime,$Workspace,$Downloads,(Join-Path $Runtime 'tmp'),(Join-Path $Runtime 'pip-cache'),(Join-Path $Runtime 'site-packages'),(Join-Path $Workspace 'logs'),(Join-Path $Workspace 'jobs'))) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}
$env:TMP = Join-Path $Runtime 'tmp'; $env:TEMP = $env:TMP
$env:PIP_CACHE_DIR = Join-Path $Runtime 'pip-cache'
$env:PIP_CONFIG_FILE = 'NUL'
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
function Download-Verified([string]$Url, [string]$File, [string]$Algorithm, [string]$Hash) {
    if (Test-Path $File) { Remove-Item -Force $File }
    Write-Host "Downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $File -UseBasicParsing
    $got = (Get-FileHash -Algorithm $Algorithm -Path $File).Hash.ToLowerInvariant()
    if ($got -ne $Hash.ToLowerInvariant()) {
        Remove-Item -Force $File
        throw "Download checksum mismatch for $Url. Installation stopped."
    }
}
if (-not (Test-Path $Git)) {
    $asset = Join-Path $Downloads 'mingit.zip'
    Download-Verified 'https://github.com/git-for-windows/git/releases/download/v2.55.0.windows.4/MinGit-2.55.0.4-64-bit.zip' $asset 'SHA256' '4e03f94c2ffbf70be337e005cee02661c732dbfc81031a078bda9299b9a7d644'
    Expand-Archive -LiteralPath $asset -DestinationPath $GitDir -Force
    if (-not (Test-Path $Git)) { throw 'Bundled Git executable missing after extraction.' }
}
if (-not (Test-Path (Join-Path $Root '.git'))) {
    Write-Host 'Making the extracted UV3D folder a Git main-branch checkout...'
    # Intended only for a freshly extracted UV3D source folder. This hard reset
    # may overwrite untracked files with names also in the repository.
    & $Git -C $Root init -b main
    if ($LASTEXITCODE -ne 0) { throw 'Git init failed.' }
    & $Git -C $Root remote add origin 'https://github.com/KubovicsT/UV3D.git'
    if ($LASTEXITCODE -ne 0) { throw 'Setting Git origin failed.' }
    & $Git -C $Root fetch origin main
    if ($LASTEXITCODE -ne 0) { throw 'Initial Git fetch failed.' }
    & $Git -C $Root reset --hard origin/main
    if ($LASTEXITCODE -ne 0) { throw 'Initial checkout failed.' }
}
if (-not (Test-Path $Python)) {
    $asset = Join-Path $Downloads 'python-3.12.10-embed-amd64.zip'
    # Official Python.org MD5 of this exact artifact, for corruption detection.
    Download-Verified 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip' $asset 'MD5' 'fe8ef205f2e9c3ba44d0cf9954e1abd3'
    Expand-Archive -LiteralPath $asset -DestinationPath $PythonDir -Force
}
# Never install Python/PySide globally or into AppData.
@('python312.zip', '.', '..\site-packages', '..\..', 'import site') | Set-Content -Encoding ascii (Join-Path $PythonDir 'python312._pth')
$pip = Join-Path $Downloads 'pip.pyz'
if (-not (Test-Path $pip)) {
    Write-Host 'Downloading official pip standalone zipapp...'
    Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/pip/pip.pyz' -OutFile $pip -UseBasicParsing
}
$req = Join-Path $Root 'requirements.txt'
$stamp = Join-Path $Runtime 'requirements-installed.sha256'
$hash = (Get-FileHash -Path $req -Algorithm SHA256).Hash
if ((-not (Test-Path $stamp)) -or ((Get-Content $stamp -Raw).Trim() -ne $hash)) {
    Write-Host 'Installing application packages INSIDE runtime\site-packages ...'
    & $Python $pip install --disable-pip-version-check --only-binary=:all: --upgrade --target (Join-Path $Runtime 'site-packages') --requirement $req
    if ($LASTEXITCODE -ne 0) { throw 'Local dependency installation failed.' }
    Set-Content -Path $stamp -Value $hash -Encoding ascii
}
Write-Host "UV3D is ready in $Root"
if (-not $NoLaunch) { & (Join-Path $Root 'Start-UV3D.cmd') }
