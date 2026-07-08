param(
    [string]$FeatureSets = "acc16_hr,acc16_gyro,acc16_gyro_hr",
    [string]$BaselineModels = "random_forest,knn_k5",
    [string]$V3Models = "improved_gnn_lstm_res,improved_gnn_lstm_attn_adj_resbn",
    [int]$V3ParallelJobs = 2,
    [int]$V3Epochs = 30,
    [int]$V3Patience = 8,
    [int]$V3BatchSize = 32,
    [int]$RealExpBaselineParallelJobs = 3,
    [int]$RealExpBaselineEstimatorJobs = 4,
    [switch]$SkipNonOverlappingCore,
    [switch]$SkipV3NonOverlapping,
    [switch]$SkipExtraRobustness,
    [switch]$SkipReportRebuild,
    [switch]$OnlyBaselineProbabilityRefresh
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Repo virtualenv Python not found: $Python"
}

Set-Location $RepoRoot

$ProcessedRoot = "data/processed/canonical_protocol_only"
$V3Root = "results/canonical_protocol_only_v3"
$OutRoot = "results/canonical_protocol12_seven_experiments_top4"

if ($OnlyBaselineProbabilityRefresh) {
    $SkipV3NonOverlapping = $true
    $SkipExtraRobustness = $true
}

$FeatureSetList = $FeatureSets.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
$BaselineModelList = $BaselineModels.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
$V3ModelList = $V3Models.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
$ProtocolList = @("loso", "random_holdout")
$NonOverlapCoreCells = $FeatureSetList.Count * $ProtocolList.Count * ($BaselineModelList.Count + $V3ModelList.Count)
$NonOverlapBaselineCells = $FeatureSetList.Count * $ProtocolList.Count * $BaselineModelList.Count
$NonOverlapV3Cells = $FeatureSetList.Count * $ProtocolList.Count * $V3ModelList.Count
$BaselineProbabilityRefreshCells = $FeatureSetList.Count * $ProtocolList.Count * $BaselineModelList.Count * 2
$ExtraRobustnessConditions = $FeatureSetList.Count * 2 * 3 * ($BaselineModelList.Count + $V3ModelList.Count)
$V3ParallelJobsEffective = [Math]::Min([Math]::Max(1, $V3ParallelJobs), [Math]::Max(1, $V3ModelList.Count))

Write-Host ""
Write-Host "=== PAMAP2 top4 fanout plan ==="
Write-Host "Output root: $OutRoot"
Write-Host "Feature sets: $($FeatureSetList -join ', ')"
Write-Host "Baseline models: $($BaselineModelList -join ', ')"
Write-Host "V3 deep models: $($V3ModelList -join ', ')"
Write-Host "Non-overlapping core cells: $NonOverlapCoreCells total ($NonOverlapBaselineCells baseline + $NonOverlapV3Cells v3)"
Write-Host "Baseline probability refresh check: $BaselineProbabilityRefreshCells cells across overlapping + non_overlapping"
Write-Host "Extra Exp3 robustness conditions: $ExtraRobustnessConditions"
Write-Host "Safe fanout: baselines run as $($BaselineModelList.Count) concurrent model jobs; v3 runs $V3ParallelJobsEffective models concurrently per feature/protocol group."
Write-Host "GPU feature/protocol groups remain sequential to avoid VRAM contention."
if ($OnlyBaselineProbabilityRefresh) {
    Write-Host "Mode: baseline probability refresh only; v3 non-overlap and extra Exp3 stages are skipped."
}

if (-not $SkipNonOverlappingCore) {
    Write-Host ""
    Write-Host "=== PAMAP2 DOCX remaining: prepare non-overlapping protocol12 data ==="
    & $Python scripts/canonical_prepare_datasets.py `
        --dataset pamap2 `
        --out-root $ProcessedRoot `
        --pamap2-feature-sets $FeatureSets `
        --pamap2-task protocol12 `
        --pamap2-sessions protocol `
        --window-types non_overlapping

    if ($LASTEXITCODE -ne 0) {
        throw "Preparing non-overlapping PAMAP2 protocol12 data failed."
    }

    Write-Host ""
    Write-Host "=== PAMAP2 DOCX remaining: baseline probability/core refresh ==="
    $BaselineWindowTypes = "overlapping,non_overlapping"
    $BaselineJobs = @()
    foreach ($ModelName in $BaselineModelList) {
        $ModelArgs = @(
            "scripts/canonical_baseline_runner.py",
            "--dataset", "pamap2",
            "--processed-root", $ProcessedRoot,
            "--out-root", "results/canonical_protocol_only/core_comparison",
            "--feature-sets", $FeatureSets,
            "--window-types", $BaselineWindowTypes,
            "--protocols", "random_holdout,loso",
            "--models", $ModelName,
            "--skip-existing",
            "--require-probabilities"
        )
        $BaselineJobs += Start-Job -Name "pamap2_baseline_$ModelName" -ScriptBlock {
            param($RepoRootArg, $PythonArg, $ArgsForPython)
            Set-Location $RepoRootArg
            & $PythonArg @ArgsForPython
            if ($LASTEXITCODE -ne 0) {
                throw "Baseline job failed with exit code $LASTEXITCODE"
            }
        } -ArgumentList $RepoRoot, $Python, $ModelArgs
    }

    if ($BaselineJobs.Count -gt 0) {
        Wait-Job -Job $BaselineJobs | Out-Null
        foreach ($Job in $BaselineJobs) {
            if ($Job.State -ne "Completed") {
                Receive-Job -Job $Job -Keep -ErrorAction Continue
                throw "Non-overlapping PAMAP2 baseline job failed: $($Job.Name)"
            }
            Receive-Job -Job $Job -ErrorAction SilentlyContinue
            Remove-Job -Job $Job
        }
    }

    Write-Host ""
    Write-Host "=== PAMAP2 DOCX remaining: repair combined baseline summaries ==="
    $RepairArgs = @(
        "scripts/canonical_baseline_runner.py",
        "--dataset", "pamap2",
        "--processed-root", $ProcessedRoot,
        "--out-root", "results/canonical_protocol_only/core_comparison",
        "--feature-sets", $FeatureSets,
        "--window-types", $BaselineWindowTypes,
        "--protocols", "random_holdout,loso",
        "--models", $BaselineModels,
        "--skip-existing",
        "--require-probabilities"
    )
    & $Python @RepairArgs

    if ($LASTEXITCODE -ne 0) {
        throw "Repairing combined PAMAP2 baseline summaries failed."
    }

    if (-not $SkipV3NonOverlapping) {
        Write-Host ""
        Write-Host "=== PAMAP2 DOCX remaining: non-overlapping v3 models ==="
        foreach ($FeatureSet in $FeatureSetList) {
            foreach ($Protocol in @("loso", "random_holdout")) {
                $ProcessedDir = "$ProcessedRoot/pamap2/$FeatureSet/non_overlapping"
                $RunRoot = "$V3Root/core_comparison/pamap2/$FeatureSet/non_overlapping/$Protocol/deep"
                & $Python scripts/phase2_repo_deep_parallel_v2.py `
                    --datasets pamap2 `
                    --models $V3Models `
                    --eval-modes sequence `
                    --processed-dir $ProcessedDir `
                    --run-root $RunRoot `
                    --eval-protocol $Protocol `
                    --parallel-jobs $V3ParallelJobsEffective `
                    --epochs $V3Epochs `
                    --patience $V3Patience `
                    --batch-size $V3BatchSize `
                    --device cuda `
                    --num-workers 0 `
                    --cpu-threads-per-job 4 `
                    --sequence-length 10 `
                    --sequence-stride 1 `
                    --sequence-target-policy last `
                    --early-stop-metric val_macro_f1 `
                    --early-stop-mode auto `
                    --optimizer adamw `
                    --lr 0.0003 `
                    --weight-decay 0.0003 `
                    --label-smoothing 0.05 `
                    --class-balanced-loss `
                    --standardize-input `
                    --variant-name v3_residual_arch_nonoverlap `
                    --disable-cudnn-for-sequence-models `
                    --skip-existing

                if ($LASTEXITCODE -ne 0) {
                    throw "Non-overlapping PAMAP2 v3 run failed for feature_set=$FeatureSet protocol=$Protocol."
                }
            }
        }
    } else {
        Write-Host ""
        Write-Host "=== PAMAP2 DOCX remaining: non-overlapping v3 models skipped ==="
    }
}

if (-not $SkipExtraRobustness) {
    Write-Host ""
    Write-Host "=== PAMAP2 DOCX remaining: append extra real Exp3 robustness perturbations ==="
    & $Python scripts/run_canonical_protocol12_real_exp3_exp6.py `
        --processed-root $ProcessedRoot `
        --v3-root $V3Root `
        --out-root $OutRoot `
        --feature-sets $FeatureSets `
        --experiments exp3 `
        --families baseline,v3 `
        --baseline-models $BaselineModels `
        --v3-models $V3Models `
        --perturbations sensor_node_zero,random_window_dropout `
        --severities low,medium,high `
        --baseline-parallel-jobs $RealExpBaselineParallelJobs `
        --baseline-estimator-jobs $RealExpBaselineEstimatorJobs `
        --v3-parallel-jobs $V3ParallelJobsEffective `
        --device cuda `
        --batch-size $V3BatchSize `
        --append-existing

    if ($LASTEXITCODE -ne 0) {
        throw "Extra PAMAP2 Exp3 robustness run failed."
    }
}

if (-not $SkipReportRebuild) {
    Write-Host ""
    Write-Host "=== PAMAP2 DOCX remaining: rebuild tables and report ==="
    & $Python scripts/run_canonical_protocol12_seven_experiments.py `
        --include-baselines `
        --include-v3 `
        --require-v3-complete `
        --out-root $OutRoot `
        --baseline-models $BaselineModels `
        --v3-models $V3Models

    if ($LASTEXITCODE -ne 0) {
        throw "Canonical seven-experiment table rebuild failed."
    }

    & $Python scripts/build_pamap2_docx_standard_report.py `
        --out-root $OutRoot `
        --baseline-models $BaselineModels `
        --v3-models $V3Models

    if ($LASTEXITCODE -ne 0) {
        throw "PAMAP2 DOCX-standard report rebuild failed."
    }
}

Write-Host ""
Write-Host "[OK] PAMAP2 DOCX remaining workflow finished."
