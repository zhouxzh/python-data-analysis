[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resultsRoot = Join-Path $repoRoot "results"
Set-Location $repoRoot

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$scriptDirs = Get-ChildItem (Join-Path $repoRoot "scripts") -Directory | Sort-Object Name

foreach ($dir in $scriptDirs) {
    $targetDir = Join-Path $resultsRoot $dir.Name
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    Get-ChildItem $targetDir -Filter "*-result.txt" -ErrorAction SilentlyContinue | Remove-Item -Force

    $pyFiles = Get-ChildItem $dir.FullName -Filter "*.py" | Sort-Object Name
    foreach ($py in $pyFiles) {
        $resultPath = Join-Path $targetDir "$($py.BaseName)-result.txt"
        python $py.FullName *> $resultPath
        if ($LASTEXITCODE -ne 0) {
            throw "$($dir.Name)/$($py.Name) failed with exit code $LASTEXITCODE"
        }
        Write-Host "$($dir.Name)/$($py.Name) ok"
    }
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
Get-ChildItem $resultsRoot -Recurse -Filter "*-result.txt" | ForEach-Object {
    $text = [System.IO.File]::ReadAllText($_.FullName, [System.Text.Encoding]::Unicode)
    [System.IO.File]::WriteAllText($_.FullName, $text, $utf8)
}

$demoRoot = Join-Path $repoRoot "projects\demo"
if (Test-Path $demoRoot) {
    Get-ChildItem $demoRoot -Directory | ForEach-Object {
        $target = Join-Path $resultsRoot $_.Name
        New-Item -ItemType Directory -Force -Path $target | Out-Null
        Copy-Item (Join-Path $_.FullName "*.png") $target -Force
    }
    Remove-Item -Recurse -Force $demoRoot
}

Write-Host "results updated: $resultsRoot"
