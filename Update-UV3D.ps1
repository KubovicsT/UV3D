# UV3D in-folder branch updater.
# Interactive progress is printed to the terminal and saved in workspace\logs\update.log.
param(
    [int]$WaitForPid = 0,
    [switch]$NoRestart
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Git = Join-Path $Root 'runtime\git\cmd\git.exe'
$LogDir = Join-Path $Root 'workspace\logs'
$LogFile = Join-Path $LogDir 'update.log'
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

function Log([string]$Message) {
    $entry = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $Message"
    Add-Content -LiteralPath $LogFile -Value $entry -Encoding UTF8
    Write-Host $entry
}
function Step([int]$Number, [string]$Description) {
    $percent = [Math]::Min(100, [Math]::Round($Number * 100 / 7))
    Write-Progress -Activity 'UV3D: updating from GitHub main' -Status $Description -PercentComplete $percent
    Log "[$Number/7] $Description"
}
function Assert-GitSuccess([string]$Operation) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Operation failed (Git exit code $LASTEXITCODE)."
    }
}

try {
    Log "UV3D updater started in $Root"
    Step 1 'Waiting for UV3D to close'
    if ($WaitForPid -gt 0) {
        if (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue) {
            Write-Host "Waiting for UV3D (PID $WaitForPid) to close..."
            try {
                Wait-Process -Id $WaitForPid -Timeout 120 -ErrorAction Stop
            } catch {
                if (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue) {
                    throw 'UV3D remained open for 120 seconds. Update cancelled without modifying files.'
                }
            }
        }
    }
    if (-not (Test-Path -LiteralPath $Git -PathType Leaf)) {
        throw "Portable Git is missing: $Git. Run Install-UV3D.ps1 from this folder."
    }
    if (-not (Test-Path -LiteralPath (Join-Path $Root '.git'))) {
        throw 'This is not a Git checkout. Run Install-UV3D.ps1 on a fresh extracted UV3D folder.'
    }

    Step 2 'Checking branch and protecting local changes'
    $branch = (& $Git -C $Root branch --show-current | Out-String).Trim()
    Assert-GitSuccess 'Checking the current branch'
    if ($branch -ne 'main') {
        throw "Expected the 'main' branch, but found '$branch'. Refusing to switch branches automatically."
    }
    $dirty = (& $Git -C $Root status --porcelain --untracked-files=no | Out-String).Trim()
    Assert-GitSuccess 'Checking for local modifications'
    if ($dirty) {
        throw "Tracked source files have local modifications. Update stopped to protect them: $dirty"
    }
    $before = (& $Git -C $Root rev-parse --short HEAD | Out-String).Trim()
    Assert-GitSuccess 'Reading the installed commit'
    Log "Installed commit: $before"

    Step 3 'Downloading latest main branch (Git progress shown below)'
    & $Git -C $Root fetch --progress --no-tags origin main
    Assert-GitSuccess 'Fetching origin/main'

    Step 4 'Fast-forwarding safely (no reset, merge commit or release)'
    & $Git -C $Root merge --ff-only origin/main
    Assert-GitSuccess 'Fast-forwarding to origin/main'
    $after = (& $Git -C $Root rev-parse --short HEAD | Out-String).Trim()
    Assert-GitSuccess 'Reading the updated commit'
    if ($after -eq $before) {
        Log "Already up to date ($after)."
    } else {
        Log "Updated: $before -> $after"
    }

    Step 5 'Checking folder-local Python and dependencies'
    $installer = Join-Path $Root 'Install-UV3D.ps1'
    if (-not (Test-Path -LiteralPath $installer -PathType Leaf)) {
        throw "Missing installer: $installer"
    }
    # Installs only under this application's runtime\ directory.
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installer -NoLaunch
    if ($LASTEXITCODE -ne 0) {
        throw "Local dependency setup failed (exit code $LASTEXITCODE)."
    }

    Step 6 'Finishing the update and writing the log'
    Log "Source branch: main; installed commit: $after"
    Log "User projects and folder-local runtime were preserved."

    Step 7 'Update complete'
    Write-Progress -Activity 'UV3D: updating from GitHub main' -Completed
    Write-Host ""
    Write-Host "UV3D is up to date ($after)." -ForegroundColor Green
    Write-Host "Log: $LogFile" -ForegroundColor Gray
    if (-not $NoRestart) {
        Log 'Launching UV3D.'
        Start-Process -FilePath (Join-Path $Root 'Start-UV3D.cmd') -WorkingDirectory $Root
    } else {
        Log 'NoRestart selected. UV3D was not launched.'
    }
} catch {
    Write-Progress -Activity 'UV3D: updating from GitHub main' -Completed
    Log "ERROR: $($_.Exception.Message)"
    Write-Host ""
    Write-Host 'UV3D update FAILED. Files were not forcibly reset.' -ForegroundColor Red
    Write-Host "See $LogFile for details." -ForegroundColor Yellow
    throw
}
