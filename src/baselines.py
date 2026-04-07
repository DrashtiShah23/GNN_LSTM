"""
Classical baseline models: SVM, Random Forest, XGBoost.

For each model we:
  1. Extract hand-crafted window-level features (mean, std, FFT energy)
  2. Train with sklearn API
  3. Evaluate (accuracy, macro-F1, balanced accuracy)
  4. Compute SHAP feature importances (for tree models)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from src.config import METRICS_DIR, SEED


# ── Feature engineering ───────────────────────────────────────────────────────

def extract_features(X: np.ndarray) -> np.ndarray:
    """
    Extract window-level statistical features.

    Input  : X of shape (N, WINDOW_SIZE, n_channels)
    Output : feature matrix of shape (N, n_features)
    """
    feats = []
    for i in range(len(X)):
        win = X[i]  # (WINDOW_SIZE, n_channels)

        mean = win.mean(axis=0)
        std = win.std(axis=0)
        min_ = win.min(axis=0)
        max_ = win.max(axis=0)
        rms = np.sqrt((win ** 2).mean(axis=0))

        # FFT energy (first half of spectrum)
        fft = np.fft.rfft(win, axis=0)
        fft_energy = (np.abs(fft) ** 2).mean(axis=0)

        feats.append(np.concatenate([mean, std, min_, max_, rms, fft_energy]))

    return np.vstack(feats)


# ── Train / evaluate helpers ──────────────────────────────────────────────────

def train_evaluate_baseline(
    clf,
    X_train_feat: np.ndarray,
    y_train: np.ndarray,
    X_test_feat: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    clf.fit(X_train_feat, y_train)
    y_pred = clf.predict(X_test_feat)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "y_pred": y_pred,
    }


# ── LOSO baseline evaluation ──────────────────────────────────────────────────

def run_baselines_loso(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    dataset_name: str = "dataset",
) -> Dict:
    """
    Run SVM, RF (and XGBoost if available) with LOSO evaluation.
    Returns per-model aggregated metrics.
    """
    from sklearn.preprocessing import LabelEncoder
    from src.train import loso_splits

    print(f"\n══ Classical Baselines — {dataset_name} ══")
    X_feat = extract_features(X)

    # Re-encode labels to be contiguous 0..N-1 (required by XGBoost)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    models = {
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, random_state=SEED)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)),
        ]),
    }
    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200, learning_rate=0.1,
                eval_metric="mlogloss",
                random_state=SEED, n_jobs=-1,
            )),
        ])

    results = {}
    for model_name, clf in models.items():
        print(f"\n── {model_name} ──")
        fold_accs, fold_f1s = [], []
        all_true, all_pred = [], []

        for train_idx, test_idx, test_subj in loso_splits(subjects):
            clf_copy = clone_pipeline(clf)
            res = train_evaluate_baseline(
                clf_copy,
                X_feat[train_idx], y_enc[train_idx],
                X_feat[test_idx], y_enc[test_idx],
            )
            fold_accs.append(res["accuracy"])
            fold_f1s.append(res["macro_f1"])
            all_true.extend(y_enc[test_idx])
            all_pred.extend(res["y_pred"])
            print(f"  Subject {test_subj}: acc={res['accuracy']:.4f}, macro-F1={res['macro_f1']:.4f}")

        mean_acc = float(np.mean(fold_accs))
        mean_f1 = float(np.mean(fold_f1s))
        print(f"  Mean Acc: {mean_acc:.4f} ± {np.std(fold_accs):.4f}")
        print(f"  Mean F1 : {mean_f1:.4f} ± {np.std(fold_f1s):.4f}")

        results[model_name] = {
            "mean_accuracy": mean_acc,
            "std_accuracy": float(np.std(fold_accs)),
            "mean_macro_f1": mean_f1,
            "std_macro_f1": float(np.std(fold_f1s)),
            "per_fold_accuracy": fold_accs,
        }

    # Save
    out_dir = Path(METRICS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{dataset_name}_baselines.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved baseline results to {METRICS_DIR}/{dataset_name}_baselines.json")

    return results


def clone_pipeline(pipeline):
    """Deep copy a sklearn pipeline."""
    from sklearn.base import clone
    return clone(pipeline)
