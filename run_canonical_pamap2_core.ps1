$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$logDir = Join-Path $PSScriptRoot "results\canonical\run_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$out = Join-Path $logDir "pamap2_core_$stamp.out.log"
$err = Join-Path $logDir "pamap2_core_$stamp.err.log"
$status = Join-Path $logDir "pamap2_core_$stamp.status.json"

$payload = @{
    started_at = (Get-Date).ToString("o")
    status = "running"
    stdout = $out
    stderr = $err
    command = ".\.venv\Scripts\python.exe -u scripts\canonical_experiment_launcher.py --skip-existing --include-xgb"
}
$payload | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $status

try {
    & ".\.venv\Scripts\python.exe" -u "scripts\canonical_experiment_launcher.py" --skip-existing --include-xgb 1> $out 2> $err
    $exitCode = $LASTEXITCODE
    $payload.status = if ($exitCode -eq 0) { "complete" } else { "failed" }
    $payload.exit_code = $exitCode
} catch {
    $payload.status = "failed"
    $payload.error = $_.Exception.Message
    $exitCode = 1
}

$payload.finished_at = (Get-Date).ToString("o")
$payload | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $status
exit $exitCode
