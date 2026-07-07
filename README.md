# Continuous HAR Enhancement Using GNN + LSTM

**DATA 245 — Machine Learning Technologies**  
*Dhruv Patel · Drashti Shah · Viraat Chaudhary*

---

## Overview

This project implements a hybrid **GNN + LSTM** architecture for continuous Human Activity Recognition (HAR) that explicitly models inter-sensor dependencies (GNN) while capturing temporal dynamics across windows (LSTM).  

We evaluate on two public datasets:
| Dataset | Sensors | Activities |
|---------|---------|------------|
| **HHAR** | Phone & watch accel/gyro; multiple device models | 6 activities; cross-device + cross-user shifts |
| **PAMAP2** | 3 IMUs (wrist/chest/ankle) + heart rate | 18 activities; multi-position fusion, missing values |

---

## Project Structure

```
ML-Project-LSTM/
├── data/
│   ├── raw/
│   │   ├── hhar/           ← downloaded HHAR dataset
│   │   └── pamap2/         ← downloaded PAMAP2 dataset
│   └── processed/          ← preprocessed .npy files
├── src/
│   ├── config.py           ← all hyperparameters & paths
│   ├── data_download.py    ← download HHAR & PAMAP2
│   ├── preprocessing.py    ← resample, normalise, sliding window
│   ├── graph_construction.py ← adjacency matrices, node features
│   ├── dataset.py          ← PyTorch Dataset classes
│   ├── models.py           ← GNN+LSTM, LSTM-only, GNN-only
│   ├── train.py            ← training loop, LOSO splits, early stopping
│   ├── evaluation.py       ← metrics, confusion matrix, plots
│   └── baselines.py        ← SVM, RF, XGBoost baselines
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_pipeline.ipynb
│   ├── 03_baselines.ipynb
│   ├── 04_gnn_lstm_training.ipynb
│   └── 05_evaluation_interpretability.ipynb
├── results/
│   ├── models/             ← saved .pt checkpoints
│   ├── plots/              ← confusion matrices, loss curves
│   └── metrics/            ← JSON metric files, .npy predictions
├── tests/
│   └── test_models.py
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create & activate virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Install PyTorch Geometric (CPU)
```powershell
.\.venv\Scripts\python.exe -m pip install torch-geometric
```
For GPU / specific CUDA versions see: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

### 4. Download datasets
```powershell
.\.venv\Scripts\python.exe -m src.data_download
```

---

## Current Canonical Protocol12 Workflow

Use these commands for the cleaned PAMAP2 protocol-only experiments and dashboard. Run all commands from the repo root:

```powershell
Set-Location C:\Users\Dhruv\HAR\GNN_LSTM
```

### Prepare PAMAP2 Protocol12 Data

The protocol-only processed datasets are expected under:

```text
data/processed/canonical_protocol_only/pamap2/<feature_set>/overlapping
```

Feature sets:

```text
acc16_hr
acc16_gyro
acc16_gyro_hr
```

If these folders already exist, do not regenerate them before analysis unless you intentionally want to replace the processed data.

### Run Canonical Baselines + Original Deep Models

This is the main v1-style comparison across baselines and deep models:

```powershell
.\.venv\Scripts\python.exe scripts\canonical_experiment_launcher.py `
  --processed-root data/processed/canonical_protocol_only `
  --results-root results/canonical_protocol_only/core_comparison `
  --include-xgb `
  --skip-existing
```

Expected result layout:

```text
results/canonical_protocol_only/core_comparison/pamap2/<feature_set>/overlapping/<protocol>/
```

Protocols:

```text
loso
random_holdout
```

### Run v2 Improved-Model Trial

This runs the earlier v2 normalization/class-balancing trial only for `acc16_gyro_hr` LOSO:

```powershell
.\scripts\run_improved_v2_protocol12.ps1
```

Output:

```text
results/canonical_protocol_only_v2/core_comparison/pamap2/acc16_gyro_hr/overlapping/loso/deep
```

Use this as an ablation result, not as the current best model replacement.

### Run v3 Residual Improved Models

This runs the widened residual v3 models on all three feature sets and both protocols. It trains the two v3 models in parallel by default.

```powershell
.\scripts\run_improved_residual_v3_protocol12.ps1
```

Clean rerun:

```powershell
$v3Root = "results\canonical_protocol_only_v3"

if (Test-Path -LiteralPath $v3Root) {
    Remove-Item -LiteralPath $v3Root -Recurse -Force
}

.\scripts\run_improved_residual_v3_protocol12.ps1
```

Conservative rerun if CUDA memory is tight:

```powershell
.\scripts\run_improved_residual_v3_protocol12.ps1 -ParallelJobs 1
```

Expected v3 result layout:

```text
results/canonical_protocol_only_v3/core_comparison/pamap2/<feature_set>/overlapping/<loso|random_holdout>/deep
```

Expected v3 jobs:

```text
3 feature sets x 2 protocols x 2 models = 12 model runs
```

### Profile Model Parameter Counts

```powershell
.\.venv\Scripts\python.exe scripts\canonical_model_profile.py `
  --processed-root data/processed/canonical_protocol_only `
  --out-dir results/canonical_protocol_only/model_profiles
```

### Verify Finished Results

Check for failures:

```powershell
Get-ChildItem results -Recurse -Filter FAILED.json
```

Check completed v3 summaries:

```powershell
Get-ChildItem results\canonical_protocol_only_v3 -Recurse -Filter metrics_summary.csv
```

### Run the Real Seven-Experiment Suite

First run real Exp3/Exp6. This uses canonical v3 checkpoints only, and refits baselines from canonical protocol-only data because baseline estimator checkpoints were not saved:

```powershell
.\.venv\Scripts\python.exe scripts\run_canonical_protocol12_real_exp3_exp6.py `
  --processed-root data/processed/canonical_protocol_only `
  --v3-root results/canonical_protocol_only_v3 `
  --out-root results/canonical_protocol12_seven_experiments `
  --feature-sets acc16_hr,acc16_gyro,acc16_gyro_hr `
  --families baseline,v3 `
  --include-xgb `
  --baseline-parallel-jobs 3 `
  --baseline-estimator-jobs 4 `
  --device cuda `
  --batch-size 32 `
  --exp6-epochs 5
```

Then rebuild all DOCX-aligned tables:

```powershell
.\.venv\Scripts\python.exe scripts\run_canonical_protocol12_seven_experiments.py `
  --include-baselines `
  --include-v3 `
  --require-v3-complete `
  --out-root results\canonical_protocol12_seven_experiments
```

Expected real Exp3/Exp6 tables:

```text
results/canonical_protocol12_seven_experiments/manuscript_tables/table_exp3_robustness.csv
results/canonical_protocol12_seven_experiments/manuscript_tables/table_exp6_few_shot_calibration.csv
```

One-command suite, assuming v1/v3 are already present:

```powershell
.\scripts\run_canonical_protocol12_suite.ps1 -SkipCoreComparison -SkipV3Training
```

For stronger CPU use on a 24-thread machine, run 4 baseline model workers with 3 estimator threads each:

```powershell
.\scripts\run_canonical_protocol12_suite.ps1 `
  -SkipCoreComparison `
  -SkipV3Training `
  -RealExpBaselineParallelJobs 4 `
  -RealExpBaselineEstimatorJobs 3
```

To print every v3 few-shot fine-tuning epoch as well:

```powershell
.\scripts\run_canonical_protocol12_suite.ps1 `
  -SkipCoreComparison `
  -SkipV3Training `
  -RealExpBaselineParallelJobs 4 `
  -RealExpBaselineEstimatorJobs 3 `
  -RealExp6VerboseEpochs
```

Track progress even if PowerShell clips the terminal buffer:

```powershell
.\.venv\Scripts\python.exe scripts\summarize_real_exp_progress.py
```

Persistent progress events are written to:

```text
results/canonical_protocol12_seven_experiments/real_exp3_exp6/progress_events.jsonl
```

If you only want to rebuild tables after Exp3/Exp6 already exist:

```powershell
.\scripts\run_canonical_protocol12_suite.ps1 -SkipCoreComparison -SkipV3Training -SkipRealExp3Exp6
```

---

## Streamlit Dashboard

Start the local dashboard:

```powershell
.\scripts\start_dashboard_local.ps1 -Port 8501
```

Then open:

```text
http://localhost:8501
```

The dashboard reads canonical result sets such as:

```text
results/canonical_protocol_only
results/canonical_protocol_only_v2
results/canonical_protocol_only_v3
```

Use the sidebar filters for result set, variant, feature set, protocol, model family, and model.

### Share Dashboard From This Laptop

Keep the Streamlit PowerShell window open. Open a second PowerShell window for the tunnel.

Option A, Cloudflare Quick Tunnel:

```powershell
winget install Cloudflare.cloudflared
cloudflared tunnel --url http://localhost:8501
```

Copy the generated `https://*.trycloudflare.com` URL and send it to your friends.

Option B, ngrok:

```powershell
winget install Ngrok.Ngrok
ngrok http 8501
```

Copy the generated `https://*.ngrok-free.app` URL and send it to your friends.

Safety notes:

- Anyone with the tunnel URL can view result artifacts exposed by the dashboard.
- Stop sharing by closing the tunnel PowerShell window.
- Stop the dashboard by closing the Streamlit PowerShell window.
- Keep the laptop awake and online while others are viewing it.
- Do not router port-forward this dashboard.

More details: `docs/DASHBOARD_REMOTE_DEPLOY.md`.

---

## Reproduce Experiments

### Preprocessing
```bash
python -m src.preprocessing
```

### Classical baselines (SVM, RF, XGBoost)
Open `notebooks/03_baselines.ipynb` or run:
```bash
python -c "
from src.preprocessing import preprocess_pamap2
from src.baselines import run_baselines_loso
X, y, subj = preprocess_pamap2()
run_baselines_loso(X, y, subj, 'pamap2')
"
```

### Full pipeline (deep models + plots)
```bash
# Full run: both datasets, all 3 models, final plots
python scripts/rerun_gnnlstm.py

# Or step-by-step (PAMAP2 only, all 3 models):
python scripts/run_full_pipeline.py
```

### GNN + LSTM training (notebook)
Open `notebooks/04_gnn_lstm_training.ipynb`

---

## Model Architecture

```
Input windows  →  [GCN Layer 1]  →  [GCN Layer 2]  →  Mean Pool
                                                             ↓
                                                       [LSTM (2 layers)]
                                                             ↓
                                                     [MLP Classifier]
                                                             ↓
                                                     Activity Probabilities
```

---

## Evaluation Protocol

- **LOSO** (leave-one-subject-out) for both datasets
- Metrics: accuracy, balanced accuracy, macro-F1
- Interpretability: SHAP (tree models), LIME (neural models)

---

## Results Summary

### PAMAP2 — LOSO Evaluation (9-fold, subjects 101–109)

| Model | Accuracy | Macro F1 | Balanced Acc |
|---|---|---|---|
| XGBoost (baseline) | **80.76%** | 73.14% | — |
| SVM (baseline) | 79.18% | 72.44% | — |
| Random Forest (baseline) | 77.49% | 71.21% | — |
| LSTM-only | 59.39% | 59.51% | 58.96% |
| GNN-only | 72.06% | 71.51% | 70.80% |
| **GNN + LSTM (proposed)** | **64.18%** | **58.74%** | **62.76%** |

### HHAR — LOSO Evaluation (9-fold, subjects a–i; 5,000 windows/subject cap)

| Model | Accuracy | Macro F1 | Balanced Acc |
|---|---|---|---|
| XGBoost (baseline) | **59.00%** | 57.79% | — |
| SVM (baseline) | 58.12% | 56.68% | — |
| Random Forest (baseline) | 56.30% | 54.81% | — |
| LSTM-only | 48.16% | 48.59% | 48.06% |
| **GNN-only** | **60.14%** | **60.11%** | **59.78%** |
| GNN + LSTM (proposed) | 56.29% | 52.22% | 57.63% |

### Model Profiling (PAMAP2, single-sample inference)

| Model | Parameters | Latency (ms/sample) |
|---|---|---|
| LSTM-only | 1,387,340 | 0.186 |
| GNN-only | 11,724 | 0.254 |
| GNN+LSTM | 247,244 | 0.486 |

### Generated Plots (`results/plots/`)
- Confusion matrices: `cm_{model}_{dataset}.png` (6 total — LOSO aggregated)
- Model comparison: `model_comparison_pamap2.png`, `model_comparison_hhar.png`
- Cross-dataset: `cross_dataset_comparison.png`
- SHAP: `shap_rf_pamap2.png`
- Profiling: `model_profiling.png`

### Node Feature Engineering
- **Before**: mean-pooled features (1 value/channel/node — loses all temporal structure)  
- **After**: 6 statistical descriptors × channels/node = **36 features/node (PAMAP2)**, **18 features/node (HHAR)**  
  (mean, std, min, max, RMS, IQR)

### Architecture
- GNN encoder: 2-layer GCN (64→64) with symmetric normalised adjacency + self-loops  
- LSTM: 2-layer, 128 hidden units, batch-first  
- MLP head: 64-unit hidden layer, dropout=0.3  
- Training: Adam, lr=1e-3, weight decay=1e-4, 100 epochs, patience=15 early stopping  
- Evaluation: LOSO (leave-one-subject-out), 9 folds each dataset

---

