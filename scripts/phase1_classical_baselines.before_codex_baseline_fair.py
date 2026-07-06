#!/usr/bin/env python
"""Standalone Phase 1 HAR classical baselines for PAMAP2 and HHAR.

Outputs per-fold metrics, aggregate summaries, per-sample predictions,
classification reports, confusion matrices, and comparison charts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import iqr, skew, kurtosis
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

SEED = 42
np.random.seed(SEED)
warnings.filterwarnings("ignore", category=RuntimeWarning)

PAMAP2_ACTIVITY_MAP = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    9: "watching_tv",
    10: "computer_work",
    11: "car_driving",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping",
}
PAMAP2_PROTOCOL12 = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
HHAR_ACTIVITIES = ["bike", "sit", "stand", "walk", "stairsup", "stairsdown"]


def build_pamap2_columns() -> List[str]:
    cols = ["timestamp", "activity_id", "heart_rate"]
    for pos in ["hand", "chest", "ankle"]:
        cols += [
            f"{pos}_temp",
            f"{pos}_acc16_x", f"{pos}_acc16_y", f"{pos}_acc16_z",
            f"{pos}_acc6_x", f"{pos}_acc6_y", f"{pos}_acc6_z",
            f"{pos}_gyro_x", f"{pos}_gyro_y", f"{pos}_gyro_z",
            f"{pos}_mag_x", f"{pos}_mag_y", f"{pos}_mag_z",
            f"{pos}_ori_1", f"{pos}_ori_2", f"{pos}_ori_3", f"{pos}_ori_4",
        ]
    return cols

PAMAP2_COLS = build_pamap2_columns()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def finite_fill_df(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    if not cols:
        return df
    df = df.copy()
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    df[cols] = df[cols].interpolate(method="linear", limit_direction="both")
    df[cols] = df[cols].ffill().bfill()
    df[cols] = df[cols].fillna(0.0)
    return df


def window_array(arr: np.ndarray, window: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(arr) < window:
        return np.empty((0, window, arr.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    starts = np.arange(0, len(arr) - window + 1, step, dtype=np.int64)
    wins = np.stack([arr[s:s + window] for s in starts]).astype(np.float32)
    return wins, starts


def safe_entropy(x: np.ndarray, bins: int = 16) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    hist, _ = np.histogram(x, bins=bins)
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p[p > 0] / s
    return float(-(p * np.log2(p)).sum())


def extract_window_features(windows: np.ndarray, prefix_cols: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """Extract time/frequency features from windows (N, T, C)."""
    if windows.ndim != 3:
        raise ValueError(f"Expected windows with shape (N,T,C), got {windows.shape}")
    n, t, c = windows.shape
    feats: List[np.ndarray] = []
    names: List[str] = []

    def add_feature(block: np.ndarray, suffix: str) -> None:
        feats.append(block)
        for col in prefix_cols:
            names.append(f"{col}_{suffix}")

    x = np.asarray(windows, dtype=np.float64)
    add_feature(np.nanmean(x, axis=1), "mean")
    add_feature(np.nanstd(x, axis=1), "std")
    add_feature(np.nanmin(x, axis=1), "min")
    add_feature(np.nanmax(x, axis=1), "max")
    add_feature(np.nanmedian(x, axis=1), "median")
    add_feature(np.nanmean(np.abs(x - np.nanmean(x, axis=1, keepdims=True)), axis=1), "mad")
    add_feature(np.sqrt(np.nanmean(x * x, axis=1)), "rms")
    add_feature(np.nanmean(x * x, axis=1), "energy")
    add_feature(np.apply_along_axis(lambda v: iqr(v, nan_policy="omit"), 1, x), "iqr")
    add_feature(np.apply_along_axis(lambda v: skew(v, nan_policy="omit"), 1, x), "skew")
    add_feature(np.apply_along_axis(lambda v: kurtosis(v, nan_policy="omit"), 1, x), "kurtosis")

    # Signal slope from first to last sample in the window.
    slope = (x[:, -1, :] - x[:, 0, :]) / max(t - 1, 1)
    add_feature(slope, "slope")

    # Frequency-domain features.
    fft = np.fft.rfft(np.nan_to_num(x, nan=0.0), axis=1)
    mag = np.abs(fft)
    add_feature(np.mean(mag * mag, axis=1), "fft_energy")
    add_feature(np.argmax(mag, axis=1).astype(np.float64), "fft_argmax")

    entropy_values = np.zeros((n, c), dtype=np.float64)
    for j in range(c):
        for i in range(n):
            entropy_values[i, j] = safe_entropy(x[i, :, j])
    add_feature(entropy_values, "entropy")

    Xf = np.concatenate(feats, axis=1)
    Xf = np.nan_to_num(Xf, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return Xf, names


def pamap2_feature_cols(feature_set: str) -> List[str]:
    cols = ["heart_rate"]
    for pos in ["hand", "chest", "ankle"]:
        cols += [f"{pos}_acc16_x", f"{pos}_acc16_y", f"{pos}_acc16_z"]
        if feature_set in {"acc16_gyro_hr", "allimu_hr"}:
            cols += [f"{pos}_gyro_x", f"{pos}_gyro_y", f"{pos}_gyro_z"]
        if feature_set == "allimu_hr":
            cols += [f"{pos}_mag_x", f"{pos}_mag_y", f"{pos}_mag_z"]
    return cols


def find_pamap2_protocol_dir(raw_root: Path) -> Path:
    candidates = [
        raw_root / "pamap2" / "PAMAP2_Dataset" / "Protocol",
        raw_root / "pamap2" / "Protocol",
        raw_root / "PAMAP2_Dataset" / "Protocol",
        raw_root / "Protocol",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "PAMAP2 Protocol folder not found. Expected one of:\n" +
        "\n".join(str(c) for c in candidates)
    )


def load_pamap2_dataset(raw_root: Path, task: str, feature_set: str, window: int, step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, List[str], List[str]]:
    protocol_dir = find_pamap2_protocol_dir(raw_root)
    files = sorted(protocol_dir.glob("subject*.dat"))
    if not files:
        raise FileNotFoundError(f"No subject*.dat files found in {protocol_dir}")

    keep_ids = PAMAP2_PROTOCOL12 if task == "protocol12" else sorted(PAMAP2_ACTIVITY_MAP.keys())
    feature_cols = pamap2_feature_cols(feature_set)
    windows_all: List[np.ndarray] = []
    labels_all: List[int] = []
    subjects_all: List[int] = []
    meta_rows: List[Dict] = []
    sample_id = 0

    for f in files:
        m = re.search(r"(\d+)", f.stem)
        subject = int(m.group(1)) if m else len(subjects_all)
        df = pd.read_csv(f, sep=r"\s+", header=None, engine="python")
        df = df.iloc[:, :len(PAMAP2_COLS)]
        df.columns = PAMAP2_COLS[:df.shape[1]]
        if "activity_id" not in df.columns:
            raise ValueError(f"Missing activity_id column in {f}")
        df = df[df["activity_id"].isin(keep_ids)].copy()
        if df.empty:
            continue
        available = [c for c in feature_cols if c in df.columns]
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            print(f"[WARN] {f.name}: missing columns {missing}; continuing with {len(available)} columns")
        df = finite_fill_df(df, available)

        # Avoid windows crossing activity boundaries or timestamp jumps.
        act_change = df["activity_id"].ne(df["activity_id"].shift()).fillna(True)
        ts_gap = pd.Series(False, index=df.index)
        if "timestamp" in df.columns:
            ts_gap = df["timestamp"].diff().fillna(0).abs() > 2.0
        df["segment_id"] = (act_change | ts_gap).cumsum()

        for (seg_id, act_id), grp in df.groupby(["segment_id", "activity_id"], sort=False):
            arr = grp[available].to_numpy(dtype=np.float32)
            wins, starts = window_array(arr, window, step)
            if len(wins) == 0:
                continue
            windows_all.append(wins)
            labels_all.extend([int(act_id)] * len(wins))
            subjects_all.extend([subject] * len(wins))
            timestamps = grp["timestamp"].to_numpy() if "timestamp" in grp.columns else np.arange(len(grp))
            for local_idx, s in enumerate(starts):
                meta_rows.append({
                    "sample_id": sample_id,
                    "dataset": "pamap2",
                    "source": f.name,
                    "subject": subject,
                    "activity_id": int(act_id),
                    "activity_name": PAMAP2_ACTIVITY_MAP.get(int(act_id), str(act_id)),
                    "segment_id": int(seg_id),
                    "window_start_row": int(s),
                    "window_end_row": int(s + window - 1),
                    "window_start_time": float(timestamps[s]) if s < len(timestamps) else math.nan,
                    "window_end_time": float(timestamps[min(s + window - 1, len(timestamps) - 1)]) if len(timestamps) else math.nan,
                })
                sample_id += 1

    if not windows_all:
        raise RuntimeError("No PAMAP2 windows generated. Check data layout/window size/task.")
    windows = np.concatenate(windows_all, axis=0)
    labels = np.asarray(labels_all)
    subjects = np.asarray(subjects_all)
    meta = pd.DataFrame(meta_rows)
    X_feat, feature_names = extract_window_features(windows, feature_cols)
    class_names = [PAMAP2_ACTIVITY_MAP[i] for i in sorted(np.unique(labels))]
    return X_feat, labels, subjects, meta, feature_names, class_names


def find_hhar_dir(raw_root: Path) -> Path:
    candidates = [
        raw_root / "hhar" / "Activity recognition exp",
        raw_root / "hhar",
        raw_root / "Activity recognition exp",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("HHAR folder not found. Expected data/raw/hhar/Activity recognition exp or data/raw/hhar")


def load_hhar_dataset(raw_root: Path, window: int, step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, List[str], List[str]]:
    hhar_dir = find_hhar_dir(raw_root)
    files = [
        "Phones_accelerometer.csv",
        "Phones_gyroscope.csv",
        "Watch_accelerometer.csv",
        "Watch_gyroscope.csv",
    ]
    windows_all: List[np.ndarray] = []
    labels_all: List[str] = []
    subjects_all: List[str] = []
    meta_rows: List[Dict] = []
    sample_id = 0
    feature_cols = ["x", "y", "z"]

    for fname in files:
        path = hhar_dir / fname
        if not path.exists():
            print(f"[WARN] HHAR file missing: {path}")
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        if not set(feature_cols).issubset(df.columns):
            print(f"[WARN] {fname}: missing x/y/z columns; skipping")
            continue
        activity_col = "gt" if "gt" in df.columns else None
        if activity_col is None:
            for c in df.columns:
                if "activity" in c or c in {"label", "class"}:
                    activity_col = c
                    break
        user_col = "user" if "user" in df.columns else None
        if activity_col is None or user_col is None:
            print(f"[WARN] {fname}: could not find gt and user columns; skipping")
            continue
        df[activity_col] = df[activity_col].astype(str).str.lower().str.strip()
        df = df[df[activity_col].isin(HHAR_ACTIVITIES)].copy()
        if df.empty:
            continue
        sort_cols = [c for c in ["creation_time", "arrival_time", "timestamp"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols[0])
        df = finite_fill_df(df, feature_cols)
        source = fname.replace(".csv", "")
        group_cols = [user_col, activity_col]
        if "model" in df.columns:
            group_cols.append("model")
        if "device" in df.columns:
            group_cols.append("device")
        for keys, grp in df.groupby(group_cols, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            user = keys[0]
            act = keys[1]
            arr = grp[feature_cols].to_numpy(dtype=np.float32)
            wins, starts = window_array(arr, window, step)
            if len(wins) == 0:
                continue
            windows_all.append(wins)
            labels_all.extend([act] * len(wins))
            subjects_all.extend([str(user)] * len(wins))
            for s in starts:
                meta_rows.append({
                    "sample_id": sample_id,
                    "dataset": "hhar",
                    "source": source,
                    "subject": str(user),
                    "activity_name": act,
                    "window_start_row": int(s),
                    "window_end_row": int(s + window - 1),
                })
                sample_id += 1
    if not windows_all:
        raise RuntimeError("No HHAR windows generated. Check that HHAR CSV files are in data/raw/hhar.")
    windows = np.concatenate(windows_all, axis=0)
    labels = np.asarray(labels_all)
    subjects = np.asarray(subjects_all)
    meta = pd.DataFrame(meta_rows)
    X_feat, feature_names = extract_window_features(windows, feature_cols)
    class_names = sorted(np.unique(labels).tolist())
    return X_feat, labels, subjects, meta, feature_names, class_names


def make_models(include_xgb: bool, use_cuda: bool, fast: bool = False) -> Dict[str, object]:
    models: Dict[str, object] = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "knn_k5": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1))]),
        "gaussian_nb": Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())]),
        "decision_tree_entropy": DecisionTreeClassifier(criterion="entropy", random_state=SEED, min_samples_leaf=3),
        "bagged_tree_entropy": BaggingClassifier(
            estimator=DecisionTreeClassifier(criterion="entropy", random_state=SEED, min_samples_leaf=3),
            n_estimators=75,
            random_state=SEED,
            n_jobs=-1,
        ),
        "adaboost_tree": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, criterion="entropy", random_state=SEED),
            n_estimators=100,
            learning_rate=0.5,
            random_state=SEED,
        ),
        "random_forest": RandomForestClassifier(n_estimators=300, criterion="entropy", random_state=SEED, n_jobs=-1, class_weight="balanced_subsample"),
        "linear_svm": Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=1.0, random_state=SEED, class_weight="balanced", dual="auto", max_iter=10000))]),
    }
    if not fast:
        models["rbf_svm"] = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced", probability=False, random_state=SEED))])
    if include_xgb and HAS_XGB:
        xgb_params = dict(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=SEED,
            n_jobs=-1,
            tree_method="hist",
        )
        if use_cuda:
            xgb_params["device"] = "cuda"
        models["xgboost_hist" + ("_cuda" if use_cuda else "")] = XGBClassifier(**xgb_params)
    elif include_xgb and not HAS_XGB:
        print("[WARN] xgboost requested but not importable; skipping XGBoost")
    return models


def loso_indices(subjects: np.ndarray) -> Iterable[Tuple[np.ndarray, np.ndarray, str]]:
    for subj in np.unique(subjects):
        test_idx = np.where(subjects == subj)[0]
        train_idx = np.where(subjects != subj)[0]
        if len(test_idx) and len(train_idx):
            yield train_idx, test_idx, str(subj)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def plot_confusion(cm: np.ndarray, labels: Sequence[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55), max(6, len(labels) * 0.5)))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_model_comparison(summary: pd.DataFrame, out_path: Path) -> None:
    if summary.empty:
        return
    plot_df = summary.sort_values("macro_f1", ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.45)))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["macro_f1"].values)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["model"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("Macro F1")
    ax.set_title("Phase 1 LOSO Baseline Comparison")
    for yi, val in zip(y, plot_df["macro_f1"].values):
        ax.text(val + 0.005, yi, f"{val:.3f}", va="center")
    ax.set_xlim(0, min(1.0, max(0.2, plot_df["macro_f1"].max() + 0.1)))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def extract_feature_importance(model: object, feature_names: Sequence[str]) -> Optional[pd.DataFrame]:
    clf = model
    if isinstance(model, Pipeline):
        clf = model.steps[-1][1]
    imp = getattr(clf, "feature_importances_", None)
    if imp is None:
        return None
    return pd.DataFrame({"feature": list(feature_names), "importance": imp}).sort_values("importance", ascending=False)


def run_loso_experiment(X: np.ndarray, y_raw: np.ndarray, subjects: np.ndarray, meta: pd.DataFrame, class_names_hint: Sequence[str], feature_names: Sequence[str], out_dir: Path, include_xgb: bool, use_cuda: bool, fast: bool) -> None:
    ensure_dir(out_dir)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    labels = list(le.classes_)
    labels_display = [str(x) for x in labels]

    with open(out_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({int(i): str(lbl) for i, lbl in enumerate(labels)}, f, indent=2)
    with open(out_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(list(feature_names), f, indent=2)

    models = make_models(include_xgb=include_xgb, use_cuda=use_cuda, fast=fast)
    fold_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    for model_name, base_model in models.items():
        print(f"\n=== {model_name} ===")
        pred_rows: List[pd.DataFrame] = []
        y_true_all: List[int] = []
        y_pred_all: List[int] = []
        start_model = time.time()

        for train_idx, test_idx, fold_subject in loso_indices(subjects):
            model = clone(base_model)
            t0 = time.time()
            try:
                model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[test_idx])
            except Exception as exc:
                print(f"[ERROR] {model_name} fold subject={fold_subject} failed: {exc}")
                fold_rows.append({"model": model_name, "fold_subject": fold_subject, "error": str(exc)})
                continue
            elapsed = time.time() - t0
            m = metrics_dict(y[test_idx], pred)
            fold_rows.append({"model": model_name, "fold_subject": fold_subject, "n_train": len(train_idx), "n_test": len(test_idx), "fit_predict_sec": elapsed, **m})
            print(f"fold subject={fold_subject}: acc={m['accuracy']:.4f}, macro_f1={m['macro_f1']:.4f}, bacc={m['balanced_accuracy']:.4f}, sec={elapsed:.1f}")
            y_true_all.extend(y[test_idx].tolist())
            y_pred_all.extend(pred.tolist())
            fold_meta = meta.iloc[test_idx].copy().reset_index(drop=True)
            fold_meta["fold_subject"] = fold_subject
            fold_meta["model"] = model_name
            fold_meta["y_true_id"] = y[test_idx]
            fold_meta["y_pred_id"] = pred
            fold_meta["y_true"] = le.inverse_transform(y[test_idx])
            fold_meta["y_pred"] = le.inverse_transform(pred.astype(int))
            pred_rows.append(fold_meta)

        if not y_true_all:
            continue
        y_true_arr = np.asarray(y_true_all)
        y_pred_arr = np.asarray(y_pred_all)
        agg = metrics_dict(y_true_arr, y_pred_arr)
        agg["model"] = model_name
        agg["n_samples"] = int(len(y_true_arr))
        agg["total_sec"] = float(time.time() - start_model)
        summary_rows.append(agg)

        pred_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
        pred_df.to_csv(out_dir / f"predictions_{model_name}.csv", index=False)

        report = classification_report(y_true_arr, y_pred_arr, target_names=labels_display, output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(out_dir / f"classification_report_{model_name}.csv")
        with open(out_dir / f"classification_report_{model_name}.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=np.arange(len(labels)))
        pd.DataFrame(cm, index=labels_display, columns=labels_display).to_csv(out_dir / f"confusion_matrix_{model_name}.csv")
        plot_confusion(cm, labels_display, f"{model_name} confusion matrix", out_dir / f"confusion_matrix_{model_name}.png")

        fi = extract_feature_importance(base_model if False else model, feature_names)
        if fi is not None:
            fi.to_csv(out_dir / f"feature_importance_{model_name}.csv", index=False)

    folds = pd.DataFrame(fold_rows)
    summary = pd.DataFrame(summary_rows)
    folds.to_csv(out_dir / "metrics_by_fold.csv", index=False)
    if not summary.empty:
        summary = summary.sort_values("macro_f1", ascending=False)
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    plot_model_comparison(summary, out_dir / "model_comparison_macro_f1.png")
    print(f"\nSaved results to: {out_dir}")
    if not summary.empty:
        print(summary[["model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]].to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["pamap2", "hhar", "both"], default="both")
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--out-dir", default="results/phase1")
    p.add_argument("--pamap2-window", type=int, default=512)
    p.add_argument("--pamap2-step", type=int, default=100)
    p.add_argument("--hhar-window", type=int, default=128)
    p.add_argument("--hhar-step", type=int, default=64)
    p.add_argument("--pamap2-task", choices=["protocol12", "all18"], default="protocol12")
    p.add_argument("--pamap2-feature-set", choices=["acc16_hr", "acc16_gyro_hr", "allimu_hr"], default="acc16_hr")
    p.add_argument("--include-xgb", action="store_true")
    p.add_argument("--use-cuda", action="store_true")
    p.add_argument("--fast", action="store_true", help="Skip RBF SVM for quicker smoke tests")
    args = p.parse_args()

    raw_root = Path(args.data_root)
    out_base = Path(args.out_dir) / now_stamp()
    ensure_dir(out_base)
    with open(out_base / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    datasets = ["pamap2", "hhar"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        print(f"\n######## Loading {ds} ########")
        if ds == "pamap2":
            X, y, subjects, meta, feature_names, class_names = load_pamap2_dataset(
                raw_root=raw_root,
                task=args.pamap2_task,
                feature_set=args.pamap2_feature_set,
                window=args.pamap2_window,
                step=args.pamap2_step,
            )
            ds_out = out_base / f"pamap2_{args.pamap2_task}_{args.pamap2_feature_set}_w{args.pamap2_window}_s{args.pamap2_step}"
        else:
            X, y, subjects, meta, feature_names, class_names = load_hhar_dataset(raw_root=raw_root, window=args.hhar_window, step=args.hhar_step)
            ds_out = out_base / f"hhar_w{args.hhar_window}_s{args.hhar_step}"
        print(f"Generated feature matrix: X={X.shape}, labels={len(np.unique(y))}, subjects={len(np.unique(subjects))}")
        ensure_dir(ds_out)
        meta.to_csv(ds_out / "window_manifest.csv", index=False)
        dataset_info = {
            "dataset": ds,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_subjects": int(len(np.unique(subjects))),
            "labels": [str(x) for x in sorted(np.unique(y).tolist())],
            "subjects": [str(x) for x in sorted(np.unique(subjects).tolist())],
        }
        with open(ds_out / "dataset_manifest.json", "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2)
        run_loso_experiment(X, y, subjects, meta, class_names, feature_names, ds_out, include_xgb=args.include_xgb, use_cuda=args.use_cuda, fast=args.fast)


if __name__ == "__main__":
    main()
