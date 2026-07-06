param(
    [int]$Epochs = 30,
    [int]$Patience = 8,
    [int]$BatchSize = 64,
    [int]$ParallelJobs = 1,
    [int]$NumWorkers = 0
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Repo virtualenv Python not found: $Python"
}

Set-Location $RepoRoot

$ProcessedDir = "data/processed/canonical_protocol_only/pamap2/acc16_gyro_hr/overlapping"
$RunRoot = "results/canonical_protocol_only_v2/core_comparison/pamap2/acc16_gyro_hr/overlapping/loso/deep"

& $Python scripts/phase2_repo_deep_parallel_v2.py `
    --datasets pamap2 `
    --models improved_gnn_lstm,improved_gnn_lstm_attn_adj `
    --eval-modes sequence `
    --processed-dir $ProcessedDir `
    --run-root $RunRoot `
    --variant-name v2_norm_balanced_ls005 `
    --eval-protocol loso `
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
    --lr 0.0005 `
    --optimizer adamw `
    --weight-decay 0.0003 `
    --standardize-input `
    --class-balanced-loss `
    --label-smoothing 0.05 `
    --disable-cudnn-for-sequence-models `
    --skip-existing
