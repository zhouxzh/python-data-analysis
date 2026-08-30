[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Message,

    [switch]$SkipBuild,

    [switch]$SkipPush
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

if (-not $SkipBuild) {
    Write-Host "==> npm run docs:build"
    npm run docs:build
    if ($LASTEXITCODE -ne 0) {
        throw "VuePress build failed."
    }
}
else {
    Write-Host "==> skip build"
}

Write-Host "==> git diff --check"
git diff --check
if ($LASTEXITCODE -ne 0) {
    throw "git diff --check failed."
}

$status = git status --porcelain
if ($status) {
    Write-Host "==> git add -A"
    git add -A
    if ($LASTEXITCODE -ne 0) {
        throw "git add failed."
    }

    Write-Host "==> git commit -m '$Message'"
    git commit -m $Message
    if ($LASTEXITCODE -ne 0) {
        throw "git commit failed."
    }
}
else {
    Write-Host "==> no changes to commit"
}

if ($SkipPush) {
    Write-Host "==> skip push by request"
    exit 0
}

Write-Host "==> git push origin main"
git push origin main
if ($LASTEXITCODE -ne 0) {
    throw "git push failed."
}

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Host "gh CLI not found; skip workflow watch. Please open GitHub Actions to confirm Pages deployment."
    exit 0
}

$run = gh run list --branch main --limit 1 --json databaseId,status,conclusion | ConvertFrom-Json
if (-not $run -or -not $run[0].databaseId) {
    throw "Could not find the latest GitHub Actions run."
}

$runId = $run[0].databaseId
Write-Host "==> gh run watch $runId"
gh run watch $runId --exit-status --interval 10
if ($LASTEXITCODE -ne 0) {
    throw "GitHub Actions workflow failed."
}
