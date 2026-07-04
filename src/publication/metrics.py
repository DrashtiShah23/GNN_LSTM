"""Comprehensive metrics for publication experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int | None = None,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    labels = list(range(n_classes)) if n_classes else None
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )

    # One-vs-rest sensitivity/specificity per class
    sensitivities, specificities = [], []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sensitivities.append(tp / (tp + fn + 1e-8))
        specificities.append(tn / (tn + fp + 1e-8))

    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_sensitivity": float(np.mean(sensitivities)),
        "macro_specificity": float(np.mean(specificities)),
        "per_class_precision": prec.tolist(),
        "per_class_recall": rec.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
        "per_class_sensitivity": sensitivities,
        "per_class_specificity": specificities,
        "confusion_matrix": cm.tolist(),
    }
    if label_names:
        out["classification_report"] = classification_report(
            y_true, y_pred, target_names=label_names, zero_division=0,
        )
    return out


def aggregate_subject_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subjects: np.ndarray,
    n_classes: int | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for subj in np.unique(subjects):
        mask = subjects == subj
        yt, yp = y_true[mask], y_pred[mask]
        m = compute_full_metrics(yt, yp, n_classes=n_classes)
        rows.append({
            "subject": str(subj),
            "n_windows": int(mask.sum()),
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "balanced_accuracy": m["balanced_accuracy"],
            "confusion_matrix": m["confusion_matrix"],
            "per_class_recall": m["per_class_recall"],
            "per_class_f1": m["per_class_f1"],
        })
    return rows


def most_affected_class_name(
    y_true: np.ndarray,
    y_pred_clean: np.ndarray,
    y_pred_pert: np.ndarray,
    label_names: list[str],
) -> str:
    """Return fine-grained class with largest per-class F1 drop (clean vs perturbed)."""
    n = min(len(y_true), len(y_pred_clean), len(y_pred_pert))
    if n == 0:
        return ""
    yt = y_true[:n]
    yc = y_pred_clean[:n]
    yp = y_pred_pert[:n]
    classes = sorted(set(yt.tolist()) | set(yc.tolist()) | set(yp.tolist()))
    _, _, f1_clean, _ = precision_recall_fscore_support(yt, yc, labels=classes, zero_division=0)
    _, _, f1_pert, _ = precision_recall_fscore_support(yt, yp, labels=classes, zero_division=0)
    drops = f1_clean - f1_pert
    worst_i = int(np.argmax(drops))
    cls_id = int(classes[worst_i])
    if 0 <= cls_id < len(label_names):
        return label_names[cls_id]
    return str(cls_id)


def remap_labels(y: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    classes = np.unique(y)
    mapping = {int(old): int(new) for new, old in enumerate(classes)}
    y_new = np.vectorize(mapping.__getitem__)(y)
    return y_new, mapping
