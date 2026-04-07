# Copilot Instructions — GNN + LSTM Human Activity Recognition

## Project Overview
This project implements a hybrid GNN + LSTM architecture for continuous Human Activity Recognition (HAR) on two public datasets: HHAR and PAMAP2.

## Tech Stack
- Python 3.11
- PyTorch + PyTorch Geometric (GNN)
- NumPy, Pandas, SciPy
- Scikit-learn (baselines: SVM, RF, XGBoost)
- SHAP / LIME (interpretability)
- Matplotlib / Seaborn (visualization)
- Jupyter Notebooks

## Project Structure
- `data/` — raw and processed datasets (HHAR, PAMAP2)
- `src/` — source modules (preprocessing, graph construction, models, training, evaluation)
- `notebooks/` — exploratory and experiment notebooks
- `results/` — saved models, metrics, plots
- `tests/` — unit tests

## Checklist
- [x] Project scaffolded
- [x] copilot-instructions.md created
- [x] README.md created
- [x] Dependencies installed (requirements.txt, all packages verified)
- [x] Data download scripts ready (src/data_download.py)
- [x] Preprocessing pipeline complete (src/preprocessing.py)
- [x] Baseline models complete (src/baselines.py — SVM, RF, XGBoost)
- [x] GNN + LSTM model complete (src/models.py — GNNLSTMModel, ablations)
- [x] Evaluation & interpretability complete (src/evaluation.py)
- [x] 5 Jupyter notebooks created (exploration → training → evaluation)
- [x] Unit tests: 15/15 passing (tests/test_models.py)
