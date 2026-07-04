"""Dataset loading for publication experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.config import PROCESSED_DIR, HHAR_ACTIVITIES, PAMAP2_ACTIVITIES
from src.publication.metrics import remap_labels
from src.publication.windowing import cap_windows_per_subject, subsample_non_overlapping_windows


def load_processed_dataset(
    name: str,
    *,
    window_type: str = "overlapping",
    max_windows_per_subject: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    p = Path(PROCESSED_DIR)
    X = np.load(p / f"{name}_X.npy")
    y_raw = np.load(p / f"{name}_y.npy")
    subj = np.load(p / f"{name}_subjects.npy")
    y, label_map = remap_labels(y_raw)

    meta = {
        "dataset": name,
        "window_type": window_type,
        "original_n_windows": len(X),
        "subject_column": "subjects array (HHAR: user a-i; PAMAP2: 101-109)",
    }

    if window_type == "non_overlapping":
        X, y, subj = subsample_non_overlapping_windows(X, y, subj)
        meta["stride"] = 128
        meta["overlap"] = 0.0
    else:
        meta["stride"] = 64
        meta["overlap"] = 0.5

    if max_windows_per_subject:
        X, y, subj = cap_windows_per_subject(X, y, subj, max_windows_per_subject, seed)

    if name == "hhar" and max_windows_per_subject is None and window_type == "overlapping":
        # Default HHAR cap for tractability unless full data requested
        pass

    label_names = get_label_names(name, y)
    return {
        "X": X,
        "y": y,
        "subjects": subj,
        "n_classes": int(y.max()) + 1,
        "label_names": label_names,
        "label_map": label_map,
        "meta": meta,
    }


def get_label_names(dataset: str, y: np.ndarray) -> list[str]:
    classes = sorted(np.unique(y).astype(int))
    if dataset == "pamap2":
        inv = {v: k for k, v in remap_labels(np.load(Path(PROCESSED_DIR) / "pamap2_y.npy"))[1].items()}
        # After remap, class index i corresponds to sorted unique original labels
        orig_classes = sorted(np.unique(np.load(Path(PROCESSED_DIR) / "pamap2_y.npy")).astype(int))
        return [PAMAP2_ACTIVITIES.get(int(orig_classes[i]), str(i)) for i in classes]
    return [HHAR_ACTIVITIES[i] if i < len(HHAR_ACTIVITIES) else str(i) for i in classes]


def subject_feature_summary(X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> dict[str, dict]:
    """Per-subject summaries for failure analysis."""
    out = {}
    for subj in np.unique(subjects):
        mask = subjects == subj
        Xs, ys = X[mask], y[mask]
        counts = np.bincount(ys, minlength=int(y.max()) + 1)
        probs = counts / counts.sum()
        imbalance = float(counts.max() / (counts.sum() + 1e-8))
        out[str(subj)] = {
            "n_windows": int(mask.sum()),
            "missingness": float(np.isnan(Xs).mean()),
            "activity_distribution": counts.tolist(),
            "activity_imbalance": imbalance,
            "sensor_mean": float(Xs.mean()),
            "sensor_variance": float(Xs.var()),
            "hr_missingness": None,  # N/A — HR not in processed 18-ch IMU features
        }
    return out
