# Deploy persian-tts-api from Windows PowerShell to the GPU host.
# Usage (from repo root OR scripts folder):
#   powershell -ExecutionPolicy Bypass -File scripts\deploy_remote.ps1
#
# Optional env overrides:
#   $env:REMOTE = "mahyar@10.1.20.25"
#   $env:REMOTE_DIR = "~/persian-tts-api"

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Remote = if ($env:REMOTE) { $env:REMOTE } else { "mahyar@10.1.20.25" }
$RemoteDir = if ($env:REMOTE_DIR) { $env:REMOTE_DIR } else { "~/persian-tts-api" }
$Archive = Join-Path $env:TEMP "persian-tts-api-upload.tar.gz"
$RemoteTmp = "/tmp/persian-tts-api-upload.tar.gz"

foreach ($cmd in @("ssh", "scp", "tar")) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        throw "$cmd not found. Install OpenSSH Client (Windows Optional Features) and ensure tar is on PATH."
    }
}

Write-Host "Packing $Root ..."
if (Test-Path $Archive) { Remove-Item -Force $Archive }

Push-Location $Root
try {
    & tar -czf $Archive `
        --exclude=.venv `
        --exclude=.manatts-packages `
        --exclude=model_cache `
        --exclude=outputs `
        --exclude=__pycache__ `
        --exclude=.pytest_cache `
        --exclude=.git `
        --exclude=third_party `
        .
    if ($LASTEXITCODE -ne 0) { throw "tar failed ($LASTEXITCODE)" }
} finally {
    Pop-Location
}

Write-Host "Uploading to ${Remote}:${RemoteDir} ..."
& scp $Archive "${Remote}:${RemoteTmp}"
if ($LASTEXITCODE -ne 0) { throw "scp failed ($LASTEXITCODE)" }

& ssh $Remote "mkdir -p $RemoteDir && tar -xzf $RemoteTmp -C $RemoteDir && rm -f $RemoteTmp && chmod +x $RemoteDir/scripts/*.sh"
if ($LASTEXITCODE -ne 0) { throw "remote extract failed ($LASTEXITCODE)" }

Remove-Item -Force $Archive -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Synced to ${Remote}:${RemoteDir}"
Write-Host "Next on the server:"
Write-Host "  ssh $Remote"
Write-Host "  cd $RemoteDir"
Write-Host "  bash scripts/install_server.sh"
Write-Host "  # set HF_TOKEN in .env, then:"
Write-Host "  bash scripts/run_api.sh"
