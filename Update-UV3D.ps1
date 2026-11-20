param([int]$WaitForPid = 0)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Git = Join-Path $Root 'runtime\git\cmd\git.exe'
$LogDir = Join-Path $Root 'workspace\logs'
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$LogFile = Join-Path $LogDir 'update.log'
function Log([string]$Message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
    Write-Host $line
}
try {
    if ($WaitForPid -gt 0) {
        Log "Waiting for UV3D PID $WaitForPid to exit"
        try { Wait-Process -Id $WaitForPid -Timeout 120 -ErrorAction Stop } catch {
            if (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue) { throw 'UV3D did not exit. Update cancelled.' }
        }
    }
    if (-not (Test-Path $Git)) { throw 'Portable Git missing. Run Install-UV3D.ps1.' }
    if (-not (Test-Path (Join-Path $Root '.git'))) { throw 'This directory is not a Git checkout.' }
    $branch = (& $Git -C $Root branch --show-current | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $branch -ne 'main') { throw "Expected main branch, found '$branch'. Refusing to switch branches." }
    $dirty = (& $Git -C $Root status --porcelain --untracked-files=no | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $dirty) { throw 'Tracked files have local changes. Update aborted to protect your work.' }
    Log 'Fetching origin/main...'
    & $Git -C $Root fetch --no-tags origin main
    if ($LASTEXITCODE -ne 0) { throw 'Git fetch failed.' }
    & $Git -C $Root merge --ff-only origin/main
    if ($LASTEXITCODE -ne 0) { throw 'Fast-forward failed. Local branch was not modified.' }
    Log 'Updating folder-local dependencies if requirements changed...'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root 'Install-UV3D.ps1') -NoLaunch
    if ($LASTEXITCODE -ne 0) { throw 'Dependency setup failed.' }
    Log 'Update complete. Starting UV3D.'
    Start-Process -FilePath (Join-Path $Root 'Start-UV3D.cmd') -WorkingDirectory $Root
} catch {
    Log "ERROR: $_"
    Write-Host "Update failed. See $LogFile" -ForegroundColor Red
    exit 1
}
