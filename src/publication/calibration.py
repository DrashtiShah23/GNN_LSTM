"""Calibration and selective prediction metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, log_loss


def max_confidence(probs: np.ndarray) -> np.ndarray:
    return probs.max(axis=1)


def expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> float:
    conf = max_confidence(probs)
    preds = probs.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf <= hi)
        if mask.sum() == 0:
            continue
        acc = (preds[mask] == y_true[mask]).mean()
        ece += mask.mean() * abs(acc - conf[mask].mean())
    return float(ece)


def brier_score_multiclass(y_true: np.ndarray, probs: np.ndarray) -> float:
    n_classes = probs.shape[1]
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def negative_log_likelihood(y_true: np.ndarray, probs: np.ndarray) -> float:
    eps = 1e-12
    p = probs[np.arange(len(y_true)), y_true]
    return float(-np.mean(np.log(np.clip(p, eps, 1.0))))


def selective_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    coverage: float,
) -> dict[str, float]:
    """Keep highest-confidence coverage fraction; reject rest."""
    conf = max_confidence(probs)
    n_keep = max(1, int(len(y_true) * coverage))
    order = np.argsort(-conf)
    keep = order[:n_keep]
    yt, yp = y_true[keep], y_pred[keep]
    return {
        "coverage": coverage,
        "rejection_rate": 1.0 - coverage,
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "n_kept": int(n_keep),
    }


def full_calibration_report(y_true: np.ndarray, probs: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    out = {
        "ece": expected_calibration_error(y_true, probs),
        "brier_score": brier_score_multiclass(y_true, probs),
        "nll": negative_log_likelihood(y_true, probs),
    }
    for cov, label in [(0.9, "90"), (0.8, "80"), (0.7, "70")]:
        sm = selective_prediction_metrics(y_true, y_pred, probs, cov)
        out[f"accuracy_at_{label}_coverage"] = sm["accuracy"]
        out[f"macro_f1_at_{label}_coverage"] = sm["macro_f1"]
        out[f"balanced_acc_at_{label}_coverage"] = sm["balanced_accuracy"]
    return out
