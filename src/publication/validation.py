"""Automated validation checks."""

from __future__ import annotations

import numpy as np


def validate_probabilities(probs: np.ndarray, tol: float = 1e-4) -> None:
    if probs.ndim != 2:
        raise ValueError(f"Expected 2D probs, got {probs.shape}")
    sums = probs.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=tol):
        bad = np.where(np.abs(sums - 1.0) > tol)[0][:5]
        raise ValueError(f"Probabilities do not sum to 1 (examples idx {bad.tolist()})")


def validate_confusion_matrix(cm: np.ndarray, n_classes: int) -> None:
    if cm.shape != (n_classes, n_classes):
        raise ValueError(f"CM shape {cm.shape} != ({n_classes}, {n_classes})")


def validate_required_columns(rows: list[dict], required: list[str], table_name: str) -> None:
    if not rows:
        raise ValueError(f"Empty table: {table_name}")
    missing = set(required) - set(rows[0].keys())
    if missing:
        raise ValueError(f"{table_name} missing columns: {missing}")


def validate_subjects_present(subjects: np.ndarray, y_true: np.ndarray) -> None:
    if len(subjects) != len(y_true):
        raise ValueError(f"Subject length {len(subjects)} != predictions {len(y_true)}")
