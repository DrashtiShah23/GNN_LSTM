"""
Standalone XGBoost worker — runs WITHOUT importing PyTorch.

Called by train_holdout.py via subprocess to avoid the PyTorch MPS ↔ XGBoost
OpenMP segfault on Apple Silicon (macOS arm64).

Usage:
    python scripts/_xgboost_worker.py \
        --dataset pamap2 --test-size 0.2 \
        --max-windows 5000 --seed 42 \
        --tag-suffix full --save-model 1

Outputs:
    results/metrics/xgboost_{dataset}_{tag_suffix}_y_{true,pred}.npy
    results/models/xgboost_{dataset}_{tag_suffix}.pkl   (if --save-model 1)
    Prints JSON result dict on the last line (stdout)
"""

from __future__ import annotations
import argparse, json, pickle, sys, time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baselines import extract_features


PROCESSED_DIR = ROOT / "data" / "processed"
METRICS_DIR   = ROOT / "results" / "metrics"
MODELS_DIR    = ROOT / "results" / "models"


def remap_labels(y: np.ndarray):
    classes = np.unique(y)
    mapping = {int(c): i for i, c in enumerate(classes)}
    return np.array([mapping[int(c)] for c in y]), mapping


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     required=True)
    p.add_argument("--test-size",   type=float, default=0.2)
    p.add_argument("--max-windows", type=int,   default=None)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--tag-suffix",  default="full")
    p.add_argument("--save-model",  type=int,   default=1)
    args = p.parse_args()

    base = PROCESSED_DIR
    X        = np.load(base / f"{args.dataset}_X.npy")
    y_raw    = np.load(base / f"{args.dataset}_y.npy")
    subjects = np.load(base / f"{args.dataset}_subjects.npy")
    y, mapping = remap_labels(y_raw)

    # Optional per-subject cap
    if args.max_windows is not None:
        rng = np.random.default_rng(args.seed)
        keep = []
        for s in np.unique(subjects):
            idx = np.where(subjects == s)[0]
            if len(idx) > args.max_windows:
                idx = rng.choice(idx, args.max_windows, replace=False)
                idx = np.sort(idx)
            keep.append(idx)
        keep = np.concatenate(keep)
        X, y = X[keep], y[keep]

    print(f"  [xgboost/{args.dataset}] X={X.shape}  classes={len(np.unique(y))}",
          flush=True)

    idx_tr, idx_te = train_test_split(
        np.arange(len(X)), test_size=args.test_size,
        random_state=args.seed, stratify=y,
    )
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_te, y_te = X[idx_te], y[idx_te]
    print(f"  Split  →  train={len(X_tr)}  test={len(X_te)}", flush=True)

    print("  Extracting features …", flush=True)
    t0 = time.time()
    X_tr_feat = extract_features(X_tr)
    X_te_feat = extract_features(X_te)
    print(f"  Feature shape: {X_tr_feat.shape[1]} features per window", flush=True)

    # Scale manually (Pipeline + XGBoost segfaults after torch is imported on arm64)
    scaler = StandardScaler()
    X_tr_feat = scaler.fit_transform(X_tr_feat)
    X_te_feat = scaler.transform(X_te_feat)

    print("  Training XGBOOST …", flush=True)
    clf = XGBClassifier(
        n_estimators=200, learning_rate=0.1,
        eval_metric="mlogloss", random_state=args.seed, n_jobs=-1,
    )
    clf.fit(X_tr_feat, y_tr)
    elapsed = time.time() - t0

    y_pred = clf.predict(X_te_feat)
    acc = float(accuracy_score(y_te, y_pred))
    f1  = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
    bal = float(balanced_accuracy_score(y_te, y_pred))

    print(f"\n  ── Results ──────────────────────────────────────")
    print(f"  Accuracy     : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro F1     : {f1:.4f}  ({f1*100:.2f}%)")
    print(f"  Balanced Acc : {bal:.4f}")
    print(f"  Train time   : {elapsed:.1f}s")
    print(classification_report(y_te, y_pred, zero_division=0))

    tag = f"xgboost_{args.dataset}_{args.tag_suffix}"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(METRICS_DIR / f"{tag}_y_true.npy", np.array(y_te))
    np.save(METRICS_DIR / f"{tag}_y_pred.npy", np.array(y_pred))
    print(f"  Saved predictions → results/metrics/{tag}_y_{{true,pred}}.npy")

    if args.save_model:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        mp = MODELS_DIR / f"{tag}.pkl"
        with open(mp, "wb") as f:
            pickle.dump({"scaler": scaler, "clf": clf}, f)
        print(f"  Saved model      → {mp}")

    result = {
        "dataset":      args.dataset,
        "model":        "xgboost",
        "split":        f"{int((1-args.test_size)*100)}/{int(args.test_size*100)}",
        "n_train":      len(X_tr),
        "n_test":       len(X_te),
        "accuracy":     acc,
        "macro_f1":     f1,
        "balanced_acc": bal,
        "train_time_s": round(elapsed, 1),
        "params":       None,
    }
    # Print result as JSON on last line — parent process will parse it
    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()
