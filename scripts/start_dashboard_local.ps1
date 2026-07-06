param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Repo virtualenv Python not found: $Python"
}

Set-Location $RepoRoot

Write-Host "Starting HAR dashboard on http://127.0.0.1:$Port"
Write-Host "Keep this window open while friends are viewing the dashboard."

& $Python -m streamlit run scripts\phase1_streamlit_dashboard.py `
    --server.address 127.0.0.1 `
    --server.port $Port `
    --server.headless true
