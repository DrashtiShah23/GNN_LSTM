# Codex task: fix Phase 2 deep runner for nested LOSO, overfitting diagnostics, fair GNN vs GNN-LSTM evaluation, and full uncapped HHAR

Repository: `C:\Users\Dhruv\HAR\GNN_LSTM`

Primary files to update:

- `scripts\phase2_repo_deep_parallel_v2.py`
- `run_phase2_repo_parallel_v2.ps1`

Do **not** modify the raw datasets. Do **not** silently cap HHAR. Add a new versioned backup of any changed file before editing, for example:

```powershell
Copy-Item .\scripts\phase2_repo_deep_parallel_v2.py .\scripts\phase2_repo_deep_parallel_v2.before_codex_fix.py -Force
Copy-Item .\run_phase2_repo_parallel_v2.ps1 .\run_phase2_repo_parallel_v2.before_codex_fix.ps1 -Force
```

## 1. Current problem to fix

The current Phase 2 run can complete and save outputs, but the results are not fully reliable for the final paper because:

1. `gnn` and `gnn_lstm` are not being evaluated on the same unit.
   - Example from the existing run:
     - PAMAP2 `gnn`: `n_samples=18939`
     - PAMAP2 `gnn_lstm`: `n_samples=1889`
     - HHAR `gnn`: `n_samples=45000`
     - HHAR `gnn_lstm`: `n_samples=4500`
   - This looks like a hidden 10-window sequence reduction for `gnn_lstm`.
   - Therefore the current `gnn` vs `gnn_lstm` table is **not apples-to-apples**.

2. HHAR must be run completely uncapped when the user passes `-NoHharCap`.
   - No `5000` cap should apply anywhere when `-NoHharCap` is passed.
   - No model-specific hidden cap is allowed.
   - No sequence-model-only truncation is allowed.

3. The final protocol must use nested subject-disjoint validation:
   - Outer test subject: one held-out subject/user.
   - Inner validation subject: one different subject/user from the training pool.
   - Training subjects: all remaining subjects except test and validation.
   - The LOSO test subject must never be used for early stopping, hyperparameter selection, scaling, class weights, calibration, or thresholding.

4. Known sklearn LOSO metric warnings must not crash the launcher.
   - Warnings like `y_pred contains classes not in y_true` are expected when a held-out subject lacks some labels.
   - These warnings must be handled by metric code and by the PowerShell launcher.

5. Overfitting must be measured per fold and summarized across folds.

## 2. Required behavior

### 2.1 Strict nested LOSO split

For each outer fold:

```text
test_subject = held-out LOSO subject
validation_subject = one subject from all_subjects excluding test_subject
train_subjects = all_subjects excluding test_subject and validation_subject
```

Add or verify these CLI args in `phase2_repo_deep_parallel_v2.py`:

```text
--val-strategy inner_subject
--val-subject-policy round_robin|min_count|max_count|random
--early-stop-metric val_macro_f1|val_loss|val_acc
--early-stop-mode auto|min|max
```

Default values should be:

```text
--val-strategy inner_subject
--val-subject-policy round_robin
--early-stop-metric val_macro_f1
--early-stop-mode auto
```

For each fold, write:

```text
fold_split_subjects.csv
```

Required columns:

```text
dataset
model
eval_unit
fold
test_subject
validation_subject
train_subjects
n_train_windows
n_val_windows
n_test_windows
n_train_eval_samples
n_val_eval_samples
n_test_eval_samples
```

Assertions that must be enforced in code:

```python
assert test_subject != validation_subject
assert test_subject not in train_subjects
assert validation_subject not in train_subjects
```

If there are fewer than three subjects, fail with a clear error.

### 2.2 Full uncapped HHAR

When the PowerShell user passes:

```powershell
-NoHharCap
```

then Python must receive and honor:

```text
--no-hhar-cap
```

and the effective cap must be `None`, not `5000`.

In `phase2_repo_deep_parallel_v2.py`, ensure this logic is explicit and printed:

```python
if args.no_hhar_cap:
    effective_hhar_cap = None
else:
    effective_hhar_cap = args.max_windows_per_subject
```

For PAMAP2, the cap must always be `None` unless a future explicit PAMAP2 cap option is added.

In each job manifest, write:

```json
{
  "dataset": "hhar",
  "no_hhar_cap": true,
  "max_windows_per_subject_arg": 5000,
  "effective_max_windows_per_subject": null,
  "n_source_windows_before_cap": <int>,
  "n_source_windows_after_cap": <same int for uncapped>
}
```

Acceptance check for uncapped HHAR:

```powershell
python -c "import numpy as np; X=np.load('data/processed/hhar_X.npy', mmap_mode='r'); print(X.shape)"
```

Then after running Phase 2 with `-NoHharCap`, each HHAR job manifest must show:

```text
n_source_windows_after_cap == data/processed/hhar_X.npy.shape[0]
```

No deep job should show `45000` unless the processed HHAR file itself has exactly 45000 windows.

### 2.3 Fair GNN vs GNN-LSTM evaluation

Do not compare window-level `gnn` directly against sequence-level `gnn_lstm` as if they are the same protocol.

Implement **two explicit eval modes**:

```text
window
sequence
```

Add CLI args:

```text
--eval-modes window,sequence
--sequence-length 10
--sequence-stride 1
--sequence-target-policy last|majority
```

Default:

```text
--eval-modes window,sequence
--sequence-length 10
--sequence-stride 1
--sequence-target-policy last
```

Required model behavior:

- `gnn` must run in `window` mode.
- `gnn_lstm` must run in `sequence` mode.
- Additionally, for a fair comparison, `gnn` must also produce a `sequence_aligned` evaluation on the **same target indices** used by `gnn_lstm` sequence samples.

The final outputs should separate these protocols clearly:

```text
results/phase2_repo_parallel/<timestamp>/<dataset>/<model>/<eval_unit>/...
```

or, if changing directory layout is too disruptive, include `eval_unit` in every CSV row and filename.

Required `eval_unit` values:

```text
window
sequence
sequence_aligned
```

For final comparison, the table should include:

```text
dataset  model      eval_unit          n_eval_samples  macro_f1  subject_macro_f1_mean
pamap2   gnn        window             18939           ...       ...
pamap2   gnn        sequence_aligned   1889            ...       ...
pamap2   gnn_lstm   sequence           1889            ...       ...
hhar     gnn        window             full_count      ...       ...
hhar     gnn        sequence_aligned   full_seq_count  ...       ...
hhar     gnn_lstm   sequence           full_seq_count  ...       ...
```

Acceptance criteria:

- For each dataset, `gnn` `sequence_aligned` and `gnn_lstm` `sequence` must have exactly the same:
  - fold list
  - test subjects
  - `n_eval_samples`
  - target labels
  - target source window indices
- Write the alignment mapping to:

```text
sequence_alignment_manifest.csv
```

Required columns:

```text
dataset
fold
test_subject
eval_sample_id
sequence_start_source_index
sequence_end_source_index
target_source_index
y_true
```

### 2.4 Avoid hidden sequence truncation

Find any logic in `HARSequenceDataset`, the runner, or data loader that reduces the dataset to fixed chunks by integer division, for example `len(X)//10`, and verify it does not silently drop too much data.

The sequence dataset may naturally reduce `N` to approximately `N - sequence_length + 1` per subject/activity-contiguous segment if using stride 1, or approximately `N/sequence_length` if using non-overlapping stride 10. But this must be explicit in:

```text
sequence_length
sequence_stride
sequence_target_policy
n_source_windows
n_eval_samples
```

Use `sequence_stride=1` by default for full sequence evaluation unless memory is impossible. This avoids unnecessary 10x reduction. If stride 10 is used, it must be explicit in the manifest and command.

### 2.5 Warning handling

In Python metric code, set labels explicitly and avoid undefined metric warnings:

```python
labels = list(range(n_classes))
classification_report(..., labels=labels, zero_division=0)
precision_recall_fscore_support(..., labels=labels, zero_division=0)
precision_score(..., labels=labels, zero_division=0)
recall_score(..., labels=labels, zero_division=0)
f1_score(..., labels=labels, zero_division=0)
```

For confusion matrix:

```python
confusion_matrix(y_true, y_pred, labels=labels)
```

Do not treat the following as fatal:

```text
y_pred contains classes not in y_true
Precision is ill-defined
Recall is ill-defined
F-score is ill-defined
```

In `run_phase2_repo_parallel_v2.ps1`, unless `-StrictExitCode` is passed:

- Suppress known sklearn LOSO metric warnings via `PYTHONWARNINGS` filters.
- Clear any existing `PYTHONWARNINGS=error` unless `-PreservePythonWarnings` is passed.
- If Python exits nonzero after writing all expected `DONE.json`, `metrics_summary.csv`, and `predictions.csv` files, treat the run as soft-success and aggregate artifacts.

### 2.6 Overfitting diagnostics

For each fold, record training, validation, and test metrics from the best checkpoint selected by validation metric.

Write:

```text
overfitting_by_fold.csv
overfitting_summary.csv
```

Required fold-level columns:

```text
dataset
model
eval_unit
fold
test_subject
validation_subject
best_epoch
last_epoch
early_stop_metric
early_stop_mode
train_loss_at_best
val_loss_at_best
train_acc_at_best
val_acc_at_best
train_macro_f1_at_best
val_macro_f1_at_best
test_acc
test_macro_f1
test_balanced_accuracy
train_val_loss_gap_at_best
train_val_acc_gap_at_best
train_val_macro_f1_gap_at_best
val_test_macro_f1_gap
val_macro_f1_drop_best_to_last
overfit_risk_score
overfit_flag_f1_gap_gt_0_15
overfit_flag_val_drop_gt_0_10
overfit_flag_val_test_gap_gt_0_15
```

Suggested formulas:

```python
train_val_loss_gap_at_best = val_loss_at_best - train_loss_at_best
train_val_acc_gap_at_best = train_acc_at_best - val_acc_at_best
train_val_macro_f1_gap_at_best = train_macro_f1_at_best - val_macro_f1_at_best
val_test_macro_f1_gap = val_macro_f1_at_best - test_macro_f1
val_macro_f1_drop_best_to_last = best_val_macro_f1 - last_val_macro_f1

overfit_risk_score = (
    max(0.0, train_val_macro_f1_gap_at_best) +
    max(0.0, val_test_macro_f1_gap) +
    max(0.0, val_macro_f1_drop_best_to_last)
)
```

The score is diagnostic only, not a formal statistical test.

### 2.7 Checkpoint and output requirements

For each dataset/model/eval unit, save:

```text
DONE.json
run_manifest.json
dataset_manifest.json
fold_split_subjects.csv
metrics_summary.csv
metrics_by_fold.csv
predictions.csv
classification_report.csv
classification_report.json
confusion_matrix.csv
confusion_matrix.png
overfitting_by_fold.csv
overfitting_summary.csv
checkpoints/fold_<k>_best.pt
checkpoints/fold_<k>_last.pt
fold_predictions/fold_<k>_predictions.csv
plots/fold_<k>_training_curves.csv
plots/fold_<k>_training_curve.png
```

`DONE.json` must be written only after all required files for that job are complete.

At the run root, aggregate:

```text
metrics_ranked_all_jobs.csv
metrics_by_fold_all_jobs.csv
experiment2_statistical_reliability.csv
overfitting_by_fold_all_jobs.csv
overfitting_summary_all_jobs.csv
launcher_manifest.json
```

## 3. PowerShell launcher requirements

Update `run_phase2_repo_parallel_v2.ps1` parameters:

```powershell
[string]$EvalModes = "window,sequence",
[int]$SequenceLength = 10,
[int]$SequenceStride = 1,
[ValidateSet("last", "majority")]
[string]$SequenceTargetPolicy = "last",
[switch]$NoHharCap,
[switch]$StrictExitCode,
[switch]$PreservePythonWarnings,
[switch]$ShowSklearnMetricWarnings
```

When building the Python arg list:

```powershell
$ArgsList += "--eval-modes", $EvalModes
$ArgsList += "--sequence-length", "$SequenceLength"
$ArgsList += "--sequence-stride", "$SequenceStride"
$ArgsList += "--sequence-target-policy", $SequenceTargetPolicy

if ($NoHharCap) {
    $ArgsList += "--no-hhar-cap"
}
else {
    $ArgsList += "--max-windows-per-subject", "$MaxWindowsPerSubject"
}
```

Important: when `-NoHharCap` is passed, do **not** also pass an effective cap. Passing the default `5000` next to `--no-hhar-cap` is confusing and should be avoided.

Print a clear run plan before launching:

```text
HHAR cap: NONE / FULL DATA
Eval modes: window,sequence
Sequence length: 10
Sequence stride: 1
Nested validation: inner_subject / round_robin
Early stopping: val_macro_f1 max
```

## 4. Commands that must work after the fix

### 4.1 Verify processed HHAR size

```powershell
python -c "import numpy as np; X=np.load('data/processed/hhar_X.npy', mmap_mode='r'); y=np.load('data/processed/hhar_y.npy', mmap_mode='r'); s=np.load('data/processed/hhar_subjects.npy', mmap_mode='r'); print('HHAR X', X.shape, 'y', y.shape, 'subjects', len(set(s.tolist())))"
```

### 4.2 Full uncapped HHAR Phase 2 run

Run this from repo root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\run_phase2_repo_parallel_v2.ps1 `
  -Datasets hhar `
  -Models "gnn_lstm,gnn" `
  -ParallelJobs 1 `
  -Device cuda `
  -Epochs 30 `
  -Patience 8 `
  -BatchSize 128 `
  -NoHharCap `
  -ValStrategy inner_subject `
  -ValSubjectPolicy round_robin `
  -EarlyStopMetric val_macro_f1 `
  -EvalModes "window,sequence" `
  -SequenceLength 10 `
  -SequenceStride 1 `
  -SequenceTargetPolicy last `
  -DisableCudnnForSequenceModels
```

If CUDA LSTM still crashes on Windows, this command must also work on CPU:

```powershell
.\run_phase2_repo_parallel_v2.ps1 `
  -Datasets hhar `
  -Models "gnn_lstm,gnn" `
  -ParallelJobs 1 `
  -Device cpu `
  -Epochs 30 `
  -Patience 8 `
  -BatchSize 128 `
  -NoHharCap `
  -ValStrategy inner_subject `
  -ValSubjectPolicy round_robin `
  -EarlyStopMetric val_macro_f1 `
  -EvalModes "window,sequence" `
  -SequenceLength 10 `
  -SequenceStride 1 `
  -SequenceTargetPolicy last
```

### 4.3 Full PAMAP2 + full HHAR run

```powershell
.\run_phase2_repo_parallel_v2.ps1 `
  -Datasets pamap2,hhar `
  -Models "gnn_lstm,gnn" `
  -ParallelJobs 1 `
  -Device cuda `
  -Epochs 30 `
  -Patience 8 `
  -BatchSize 128 `
  -MaxWindowsPerSubject 0 `
  -NoHharCap `
  -ValStrategy inner_subject `
  -ValSubjectPolicy round_robin `
  -EarlyStopMetric val_macro_f1 `
  -EvalModes "window,sequence" `
  -SequenceLength 10 `
  -SequenceStride 1 `
  -SequenceTargetPolicy last `
  -DisableCudnnForSequenceModels
```

## 5. Acceptance tests

After a run, the following must pass.

### 5.1 Artifact existence

```powershell
$RunRoot = Get-ChildItem .\results\phase2_repo_parallel -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-ChildItem $RunRoot.FullName -Recurse -Filter DONE.json | Select-Object FullName
Get-ChildItem $RunRoot.FullName -Recurse -Filter metrics_summary.csv | Select-Object FullName
Get-ChildItem $RunRoot.FullName -Recurse -Filter predictions.csv | Select-Object FullName
Get-ChildItem $RunRoot.FullName -Recurse -Filter overfitting_by_fold.csv | Select-Object FullName
Get-ChildItem $RunRoot.FullName -Recurse -Filter fold_split_subjects.csv | Select-Object FullName
```

### 5.2 No HHAR cap

```powershell
$Code = @"
from pathlib import Path
import json
import numpy as np
import pandas as pd

run_root = sorted(Path('results/phase2_repo_parallel').glob('*'))[-1]
full_n = np.load('data/processed/hhar_X.npy', mmap_mode='r').shape[0]
print('processed HHAR windows:', full_n)

for p in run_root.glob('hhar/**/dataset_manifest.json'):
    m = json.loads(p.read_text())
    after = m.get('n_source_windows_after_cap')
    print(p, 'after_cap=', after, 'effective_cap=', m.get('effective_max_windows_per_subject'))
    assert after == full_n, f'HHAR was capped in {p}: {after} != {full_n}'
print('No HHAR cap verified')
"@
Set-Content .\tmp_check_no_hhar_cap.py $Code -Encoding UTF8
python .\tmp_check_no_hhar_cap.py
Remove-Item .\tmp_check_no_hhar_cap.py
```

### 5.3 Nested subject split verified

```powershell
$Code = @"
from pathlib import Path
import pandas as pd

run_root = sorted(Path('results/phase2_repo_parallel').glob('*'))[-1]
for p in run_root.glob('**/fold_split_subjects.csv'):
    df = pd.read_csv(p)
    for _, r in df.iterrows():
        test = str(r['test_subject'])
        val = str(r['validation_subject'])
        train = {x.strip() for x in str(r['train_subjects']).split(',') if x.strip()}
        assert test != val, (p, r['fold'], test, val)
        assert test not in train, (p, r['fold'], test, train)
        assert val not in train, (p, r['fold'], val, train)
    print('OK nested split:', p)
"@
Set-Content .\tmp_check_nested_split.py $Code -Encoding UTF8
python .\tmp_check_nested_split.py
Remove-Item .\tmp_check_nested_split.py
```

### 5.4 Fair sequence comparison verified

```powershell
$Code = @"
from pathlib import Path
import pandas as pd

run_root = sorted(Path('results/phase2_repo_parallel').glob('*'))[-1]
summary = pd.concat([pd.read_csv(p) for p in run_root.glob('**/metrics_summary.csv')], ignore_index=True)
print(summary[['dataset','model','eval_unit','n_samples','macro_f1','subject_macro_f1_mean']].to_string(index=False))

for dataset in summary['dataset'].unique():
    sub = summary[summary['dataset'] == dataset]
    gnn_seq = sub[(sub['model'] == 'gnn') & (sub['eval_unit'] == 'sequence_aligned')]
    lstm_seq = sub[(sub['model'] == 'gnn_lstm') & (sub['eval_unit'] == 'sequence')]
    if len(gnn_seq) and len(lstm_seq):
        a = int(gnn_seq.iloc[0]['n_samples'])
        b = int(lstm_seq.iloc[0]['n_samples'])
        assert a == b, f'{dataset}: sequence_aligned GNN n={a}, GNN-LSTM n={b}'
        print('OK sequence-aligned comparison:', dataset, a)
"@
Set-Content .\tmp_check_fair_seq.py $Code -Encoding UTF8
python .\tmp_check_fair_seq.py
Remove-Item .\tmp_check_fair_seq.py
```

## 6. Final deliverables from Codex

After editing, provide:

1. A concise list of changed files.
2. Exact commands used for a smoke test.
3. Exact commands for full uncapped HHAR.
4. A sample output table from `metrics_ranked_all_jobs.csv`.
5. Confirmation that:
   - HHAR was uncapped.
   - Validation subject and test subject are always distinct.
   - GNN sequence-aligned and GNN-LSTM sequence evaluations have matching sample counts.
   - Warnings did not terminate the run.

