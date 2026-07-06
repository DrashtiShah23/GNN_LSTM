param(
    [int]$Epochs = 30,
    [int]$Patience = 8,
    [int]$BatchSize = 32,
    [int]$ParallelJobs = 2,
    [int]$NumWorkers = 0,
    [string[]]$FeatureSets = @("acc16_hr", "acc16_gyro", "acc16_gyro_hr"),
    [string[]]$Protocols = @("loso", "random_holdout")
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Repo virtualenv Python not found: $Python"
}

Set-Location $RepoRoot

$Models = "improved_gnn_lstm_res,improved_gnn_lstm_attn_adj_resbn"
$Variant = "v3_residual_arch"

foreach ($FeatureSet in $FeatureSets) {
    foreach ($Protocol in $Protocols) {
        $ProcessedDir = "data/processed/canonical_protocol_only/pamap2/$FeatureSet/overlapping"
        $RunRoot = "results/canonical_protocol_only_v3/core_comparison/pamap2/$FeatureSet/overlapping/$Protocol/deep"

        if (-not (Test-Path -LiteralPath $ProcessedDir)) {
            throw "Processed dataset not found: $ProcessedDir"
        }

        Write-Host ""
        Write-Host "=== v3 residual protocol12: $FeatureSet / $Protocol ==="
        Write-Host "Processed: $ProcessedDir"
        Write-Host "Results:   $RunRoot"

        & $Python scripts/phase2_repo_deep_parallel_v2.py `
            --datasets pamap2 `
            --models $Models `
            --eval-modes sequence `
            --processed-dir $ProcessedDir `
            --run-root $RunRoot `
            --variant-name $Variant `
            --eval-protocol $Protocol `
            --parallel-jobs $ParallelJobs `
            --epochs $Epochs `
            --patience $Patience `
            --batch-size $BatchSize `
            --device cuda `
            --num-workers $NumWorkers `
            --sequence-length 10 `
            --sequence-stride 1 `
            --sequence-target-policy last `
            --early-stop-metric val_macro_f1 `
            --early-stop-mode auto `
            --lr 0.0003 `
            --optimizer adam `
            --weight-decay 0.0 `
            --disable-cudnn-for-sequence-models `
            --skip-existing

        if ($LASTEXITCODE -ne 0) {
            throw "v3 residual run failed for feature set/protocol: $FeatureSet / $Protocol"
        }
    }
}
