param(
    [int]$V3ParallelJobs = 2,
    [int]$V3Epochs = 30,
    [int]$V3Patience = 8,
    [int]$V3BatchSize = 32,
    [string]$RealExpFeatureSets = "acc16_hr,acc16_gyro,acc16_gyro_hr",
    [string]$RealExpFamilies = "baseline,v3",
    [string]$RealExpBaselineModels = "dummy_most_frequent,gaussian_nb,knn_k5,linear_svm,rbf_svm,decision_tree_entropy,bagged_tree_entropy,random_forest,adaboost_tree,xgboost_hist",
    [int]$RealExpBaselineParallelJobs = 3,
    [int]$RealExpBaselineEstimatorJobs = 4,
    [int]$RealExp6Epochs = 5,
    [switch]$RealExp6VerboseEpochs,
    [switch]$SkipCoreComparison,
    [switch]$SkipV3Training,
    [switch]$SkipRealExp3Exp6,
    [switch]$AllowCleanReferenceExp3Exp6
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Repo virtualenv Python not found: $Python"
}

Set-Location $RepoRoot

if (-not $SkipCoreComparison) {
    Write-Host ""
    Write-Host "=== Step 1/3: canonical v1 baselines + original deep models ==="
    & $Python scripts/canonical_experiment_launcher.py `
        --processed-root data/processed/canonical_protocol_only `
        --results-root results/canonical_protocol_only/core_comparison `
        --include-xgb `
        --skip-existing

    if ($LASTEXITCODE -ne 0) {
        throw "Canonical core comparison failed."
    }
}

if (-not $SkipV3Training) {
    Write-Host ""
    Write-Host "=== Step 2/3: canonical v3 residual models ==="
    & .\scripts\run_improved_residual_v3_protocol12.ps1 `
        -ParallelJobs $V3ParallelJobs `
        -Epochs $V3Epochs `
        -Patience $V3Patience `
        -BatchSize $V3BatchSize

    if ($LASTEXITCODE -ne 0) {
        throw "Canonical v3 residual run failed."
    }
}

Write-Host ""
if (-not $SkipRealExp3Exp6) {
    Write-Host "=== Step 3/4: real canonical Exp3 robustness + Exp6 few-shot calibration ==="
    $RealExpArgs = @(
        "scripts/run_canonical_protocol12_real_exp3_exp6.py",
        "--processed-root", "data/processed/canonical_protocol_only",
        "--v3-root", "results/canonical_protocol_only_v3",
        "--out-root", "results/canonical_protocol12_seven_experiments",
        "--feature-sets", $RealExpFeatureSets,
        "--families", $RealExpFamilies,
        "--baseline-models", $RealExpBaselineModels,
        "--include-xgb",
        "--baseline-parallel-jobs", $RealExpBaselineParallelJobs,
        "--baseline-estimator-jobs", $RealExpBaselineEstimatorJobs,
        "--device", "cuda",
        "--batch-size", $V3BatchSize,
        "--exp6-epochs", $RealExp6Epochs
    )
    if ($RealExp6VerboseEpochs) {
        $RealExpArgs += "--exp6-verbose-epochs"
    }

    & $Python @RealExpArgs

    if ($LASTEXITCODE -ne 0) {
        throw "Real canonical Exp3/Exp6 run failed."
    }
}

Write-Host ""
Write-Host "=== Step 4/4: canonical DOCX seven-experiment tables ==="

$ExpArgs = @(
    "scripts/run_canonical_protocol12_seven_experiments.py",
    "--include-baselines",
    "--include-v3",
    "--require-v3-complete",
    "--out-root",
    "results/canonical_protocol12_seven_experiments"
)

if ($AllowCleanReferenceExp3Exp6) {
    $ExpArgs += "--allow-clean-reference-exp3-exp6"
}

& $Python @ExpArgs

if ($LASTEXITCODE -ne 0) {
    throw "Canonical seven-experiment table generation stopped. If this is strict Exp 3/6 blocking, see results/canonical_protocol12_seven_experiments/BLOCKED_EXP3_EXP6.md."
}
