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
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install PyTorch Geometric (CPU)
```bash
pip install torch-geometric
```
For GPU / specific CUDA versions see: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

### 4. Download datasets
```bash
python -m src.data_download
```

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

## Progress Checklist

- [x] Project structure scaffolded
- [x] Config & hyperparameters defined
- [x] Preprocessing pipeline (resample, normalise, sliding window)
- [x] Graph construction (fixed & learnable adjacency, rich node features)
- [x] Dataset classes (flat, graph, sequence with subject-boundary safety)
- [x] GNN + LSTM model + ablations (GNN-only, LSTM-only)
- [x] Training loop with early stopping + LOSO splits (100 epochs, patience=15)
- [x] Classical baselines (SVM, RF, XGBoost) — PAMAP2 LOSO ✅
- [x] PAMAP2 data downloaded & preprocessed (15,049 windows × 128 × 18)
- [x] HHAR data downloaded & preprocessed (454,577 windows → 45,000 capped × 128 × 3)
- [x] Deep model LOSO evaluation — PAMAP2 ✅ (LSTM 59.4%, GNN 72.1%, GNN+LSTM 64.2%)
- [x] Deep model LOSO evaluation — HHAR ✅ (LSTM 48.2%, GNN 60.1%, GNN+LSTM 56.3%)
- [x] SHAP feature importance for RF on PAMAP2 ✅ → `results/plots/shap_rf_pamap2.png`
- [x] Model profiling — parameter count & latency ✅ → `results/plots/model_profiling.png`
- [x] Confusion matrices & comparison plots ✅ → `results/plots/` (15 plots total)
- [x] Unit tests: 15/15 passing
- [ ] Final report writing
