# Publication Experiments Pipeline

This document describes the **seven publication-grade experiments** added to the HAR GNN+LSTM project for journal submission.

## Repository Audit (confirmed from code)

| Item | Value |
|------|--------|
| **Framework** | PyTorch (Python 3.11) |
| **Datasets** | PAMAP2, HHAR (UCI) |
| **Processed data** | `data/processed/{pamap2,hhar}_{X,y,subjects}.npy` |
| **Subject IDs** | HHAR: `user` column → `a`–`i` in `hhar_subjects.npy`; PAMAP2: `101`–`109` in `pamap2_subjects.npy` |
| **Window size** | 128 samples @ 50 Hz (~2.56 s) |
| **Overlapping stride** | 64 (50% overlap) — `OVERLAP=0.5` in `src/config.py` |
| **Non-overlapping stride** | 128 (every 2nd window per subject from processed arrays) |
| **Models** | `CNN1DModel`, `GNNFlattenLSTMModel` (Flatten_LSTM), `ImprovedGNNLSTMModel` (Improved_GNN_LSTM) |
| **Best GNN+LSTM variant** | `ImprovedGNNLSTMAttnAdj` (separate script; publication pipeline uses `Improved_GNN_LSTM` as specified) |
| **LOSO** | `src/train.py::loso_splits()` |
| **Existing training scripts** | `scripts/run_full_pipeline.py`, `run_improved_gnnlstm.py`, `run_attention_adj.py` |

## Directory Structure

```text
configs/
  publication_experiments.yaml
  activity_group_mapping.yaml
src/publication/
  config.py, seeds.py, windowing.py, splits.py, metrics.py
  calibration.py, statistics.py, perturbations.py, validation.py
  outputs.py, data.py, models_registry.py, train_eval.py, plots.py, activity_groups.py
scripts/
  run_experiment_1_leakage_control.py
  run_experiment_2_statistical_reliability.py
  ...
  run_all_publication_experiments.py
results/
  experiment_1_leakage_control/
  experiment_2_statistical_reliability/
  ...
  manuscript_tables/
  manuscript_figures/
  logs/
```

## Commands

### Smoke test (fast, ~minutes)

```bash
HAR_FORCE_DEVICE=cpu .venv/bin/python scripts/run_all_publication_experiments.py --smoke
```

### Individual experiments

```bash
HAR_FORCE_DEVICE=cpu .venv/bin/python scripts/run_experiment_1_leakage_control.py
HAR_FORCE_DEVICE=cpu .venv/bin/python scripts/run_experiment_2_statistical_reliability.py
HAR_FORCE_DEVICE=cpu .venv/bin/python scripts/run_experiment_3_robustness.py
HAR_FORCE_DEVICE=cpu .venv/bin/python scripts/run_experiment_4_calibration_uncertainty.py
HAR_FORCE_DEVICE=cpu .venv/bin/python scripts/run_experiment_5_subject_failure_analysis.py
HAR_FORCE_DEVICE=cpu .venv/bin/python scripts/run_experiment_6_few_shot_calibration.py
HAR_FORCE_DEVICE=cpu .venv/bin/python scripts/run_experiment_7_health_group_analysis.py
```

### Full pipeline

```bash
HAR_FORCE_DEVICE=cpu .venv/bin/python scripts/run_all_publication_experiments.py
```

### PAMAP2 only (faster)

```bash
HAR_FORCE_DEVICE=cpu .venv/bin/python scripts/run_all_publication_experiments.py --datasets pamap2
```

## Experiments Summary

| # | Folder | Purpose |
|---|--------|---------|
| 1 | `experiment_1_leakage_control/` | Overlapping vs non-overlapping × random holdout vs LOSO |
| 2 | `experiment_2_statistical_reliability/` | Fold-level stats, Wilcoxon, bootstrap CIs, rank stability |
| 3 | `experiment_3_robustness/` | Test-time sensor/noise/window perturbations |
| 4 | `experiment_4_calibration_uncertainty/` | ECE, Brier, NLL, selective prediction |
| 5 | `experiment_5_subject_failure_analysis/` | Per-subject failures and heatmaps |
| 6 | `experiment_6_few_shot_calibration/` | 1/5/10% subject calibration, head vs full FT |
| 7 | `experiment_7_health_group_analysis/` | Clinical activity groups vs fine-grained labels |

## Leakage Controls

- `assert_loso_no_leakage()` — test subject never in training indices
- `assert_calibration_no_leakage()` — calibration/test window disjointness (Exp 6)
- Classical baselines fit `StandardScaler` on train fold only (existing `src/baselines.py`)
- Random holdout **intentionally** mixes subjects (documented inflation baseline)
- Segment z-score in preprocessing uses only within-segment timesteps (no cross-subject fit)

## Unresolved Assumptions

1. **Non-overlapping windows** are derived by subsampling every 2nd overlapping window per subject (equivalent to stride 128). Ideal path: re-run `src/preprocessing.py` with `OVERLAP=0`.
2. **Heart rate perturbations** marked `N/A` — processed features use 18 IMU channels only (no HR).
3. **HHAR full runs** cap at 5000 windows/subject by default (matches `run_improved_gnnlstm.py`).
4. **Validation split** uses last 15% of training indices (matches improved GNN scripts), not subject-based val from `run_full_pipeline.py`.
5. **PAMAP2 class count** after remap may be 12–18 depending on which activities appear in processed data; group mapping uses activity names from `configs/activity_group_mapping.yaml`.

## Unmapped Activity Labels

Check after running Exp 7:

```text
results/experiment_7_health_group_analysis/unmapped_activities.json
```

## Dependencies

Install/update:

```bash
.venv/bin/pip install pyyaml pandas
```

## Validation Checks Implemented

See `src/publication/validation.py`:

- Probability sums to 1
- Confusion matrix dimensions
- Required CSV columns
- Subject array length matches predictions
- LOSO / calibration leakage assertions in split utilities

## Known Limitations

- Full 7-experiment run on both datasets with all models is **compute-intensive** (LOSO × 9 folds × 3 models).
- Exp 3 re-trains models per perturbation setting (correct but slow); cache clean LOSO checkpoints for production runs.
- Sequence models (Flatten_LSTM, Improved_GNN_LSTM) evaluate on `HARSequenceDataset` (seq_len=10); CNN1D uses window-level data.
- Exp 1 random holdout uses a single 80/20 window split (seed=42), not repeated runs.

## Manuscript Outputs

Tables and figures are copied to:

- `results/manuscript_tables/table_exp*.csv`
- `results/manuscript_figures/fig_exp*.png` (+ PDF where generated)
