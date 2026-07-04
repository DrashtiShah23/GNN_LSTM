"""
Run Logistic Regression and Naive Bayes baselines with LOSO evaluation
on PAMAP2 and HHAR, using the same feature extraction and train split
as the existing SVM / RF / XGBoost baselines.

Outputs:
  results/metrics/extra_baselines_pamap2.json
  results/metrics/extra_baselines_hhar.json
  results/plots/cm_lr_pamap2.png
  results/plots/cm_nb_pamap2.png
  results/plots/cm_lr_hhar.png
  results/plots/cm_nb_hhar.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import extract_features
from src.train import loso_splits

DATA   = ROOT / "data" / "processed"
PLOTS  = ROOT / "results" / "plots"
METS   = ROOT / "results" / "metrics"
PLOTS.mkdir(parents=True, exist_ok=True)
METS.mkdir(parents=True, exist_ok=True)

PAMAP2_LABELS = [
    "lying", "sitting", "standing", "walking", "running",
    "cycling", "nordic_walk", "watch_tv", "computer",
    "ascending", "descending", "vacuum",
]
HHAR_LABELS = ["biking", "sitting", "standing", "walking", "stairsup", "stairsdown"]


def run_loso(X_feat, y_enc, subjects, models, dataset_name):
    results = {}
    for model_name, clf_template in models.items():
        print(f"\n── {model_name} ({dataset_name}) ──")
        from sklearn.base import clone

        fold_accs, fold_f1s, fold_baccs = [], [], []
        all_true, all_pred = [], []

        for train_idx, test_idx, test_subj in loso_splits(subjects):
            clf = clone(clf_template)
            clf.fit(X_feat[train_idx], y_enc[train_idx])
            y_pred = clf.predict(X_feat[test_idx])

            acc  = accuracy_score(y_enc[test_idx], y_pred)
            f1   = f1_score(y_enc[test_idx], y_pred, average="macro", zero_division=0)
            bacc = balanced_accuracy_score(y_enc[test_idx], y_pred)

            fold_accs.append(acc)
            fold_f1s.append(f1)
            fold_baccs.append(bacc)
            all_true.extend(y_enc[test_idx].tolist())
            all_pred.extend(y_pred.tolist())

            print(f"  Subj {test_subj}: acc={acc:.4f}  macro-F1={f1:.4f}  bal-acc={bacc:.4f}")

        mean_acc  = float(np.mean(fold_accs))
        mean_f1   = float(np.mean(fold_f1s))
        mean_bacc = float(np.mean(fold_baccs))
        std_acc   = float(np.std(fold_accs,  ddof=1))
        std_f1    = float(np.std(fold_f1s,   ddof=1))
        std_bacc  = float(np.std(fold_baccs, ddof=1))

        print(f"  ── Summary ──")
        print(f"  Accuracy  : {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  Macro-F1  : {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"  Bal-Acc   : {mean_bacc:.4f} ± {std_bacc:.4f}")

        results[model_name] = {
            "mean_accuracy":    mean_acc,
            "std_accuracy":     std_acc,
            "mean_macro_f1":    mean_f1,
            "std_macro_f1":     std_f1,
            "mean_balanced_acc": mean_bacc,
            "std_balanced_acc": std_bacc,
            "per_fold_accuracy": fold_accs,
            "per_fold_macro_f1": fold_f1s,
            "per_fold_balanced_acc": fold_baccs,
            "y_true": all_true,
            "y_pred": all_pred,
        }
    return results


def plot_cm(y_true, y_pred, labels, title, out_path):
    n_classes = max(max(y_true), max(y_pred)) + 1
    labs = labels[:n_classes]
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(max(7, n_classes), max(6, n_classes - 1)))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labs, yticklabels=labs,
        linewidths=0.4, linecolor="lightgrey",
        vmin=0, vmax=1, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def run_dataset(name, labels):
    print(f"\n{'═'*60}")
    print(f"  Dataset: {name.upper()}")
    print(f"{'═'*60}")

    X       = np.load(DATA / f"{name}_X.npy")
    y_raw   = np.load(DATA / f"{name}_y.npy")
    subjects = np.load(DATA / f"{name}_subjects.npy")

    print(f"  Windows: {X.shape[0]}  Channels: {X.shape[2]}  Classes: {len(np.unique(y_raw))}")

    X_feat = extract_features(X)
    le     = LabelEncoder()
    y_enc  = le.fit_transform(y_raw)

    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                max_iter=1000, C=1.0, solver="lbfgs",
                random_state=42, n_jobs=-1,
            )),
        ]),
        "NaiveBayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GaussianNB()),
        ]),
    }

    results = run_loso(X_feat, y_enc, subjects, models, name.upper())

    # Save json
    out_json = METS / f"extra_baselines_{name}.json"
    # Strip y_true/y_pred from saved JSON (keep it clean)
    save_results = {}
    for k, v in results.items():
        save_results[k] = {kk: vv for kk, vv in v.items() if kk not in ("y_true", "y_pred")}
    out_json.write_text(json.dumps(save_results, indent=2))
    print(f"\n  Saved → {out_json}")

    # Confusion matrices
    tag_map = {"LogisticRegression": "lr", "NaiveBayes": "nb"}
    disp_map = {"LogisticRegression": "Logistic Regression", "NaiveBayes": "Naive Bayes"}
    for model_name, r in results.items():
        plot_cm(
            r["y_true"], r["y_pred"], labels,
            f"{disp_map[model_name]} — {name.upper()} (LOSO, normalised)",
            PLOTS / f"cm_{tag_map[model_name]}_{name}.png",
        )

    return results


def print_summary(pamap2_res, hhar_res):
    print(f"\n{'═'*70}")
    print("  FINAL SUMMARY — Extra Baselines")
    print(f"{'═'*70}")
    header = f"  {'Model':<22} {'Dataset':<8} {'Accuracy':>12} {'Macro-F1':>12} {'Bal-Acc':>12}"
    print(header)
    print(f"  {'─'*22} {'─'*8} {'─'*12} {'─'*12} {'─'*12}")
    for ds_name, res in [("PAMAP2", pamap2_res), ("HHAR", hhar_res)]:
        for model_name, r in res.items():
            disp = model_name.replace("LogisticRegression", "Log. Regression")
            print(f"  {disp:<22} {ds_name:<8} "
                  f"{r['mean_accuracy']:>6.4f}±{r['std_accuracy']:.4f}  "
                  f"{r['mean_macro_f1']:>6.4f}±{r['std_macro_f1']:.4f}  "
                  f"{r['mean_balanced_acc']:>6.4f}±{r['std_balanced_acc']:.4f}")
    print(f"{'═'*70}")


if __name__ == "__main__":
    p2_res   = run_dataset("pamap2", PAMAP2_LABELS)
    hhar_res = run_dataset("hhar",   HHAR_LABELS)
    print_summary(p2_res, hhar_res)
    print("\nDone.")
