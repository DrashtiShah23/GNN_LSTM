"""Train/validation/test split utilities with leakage checks."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from src.train import loso_splits


def random_holdout_split(
    subjects: np.ndarray,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Random window holdout — subjects may appear in both train and test (inflation).
    Documented for leakage comparison only.
    """
    rng = np.random.default_rng(seed)
    n = len(subjects)
    perm = rng.permutation(n)
    n_test = max(1, int(n * test_fraction))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return train_idx, test_idx


def loso_fold_splits(
    subjects: np.ndarray,
    max_folds: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, object, int]]:
    for fi, (train_idx, test_idx, test_subj) in enumerate(loso_splits(subjects)):
        if max_folds is not None and fi >= max_folds:
            break
        yield train_idx, test_idx, test_subj, fi


def subject_val_split(
    train_idx: np.ndarray,
    subjects: np.ndarray,
    val_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out last val_fraction of training indices (matches improved GNN scripts)."""
    n_val = max(1, int(len(train_idx) * val_fraction))
    val_idx = train_idx[-n_val:]
    tr_idx = train_idx[:-n_val]
    return tr_idx, val_idx


def calibration_test_split(
    subject_indices: np.ndarray,
    calibration_fraction: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split held-out subject windows into calibration vs test (Exp 6)."""
    rng = np.random.default_rng(seed)
    idx = subject_indices.copy()
    rng.shuffle(idx)
    n_cal = max(1, int(len(idx) * calibration_fraction)) if calibration_fraction > 0 else 0
    cal_idx = idx[:n_cal]
    test_idx = idx[n_cal:]
    return cal_idx, test_idx


def assert_loso_no_leakage(
    subjects: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    test_subject: object,
) -> None:
    train_subjs = set(np.unique(subjects[train_idx]))
    test_subjs = set(np.unique(subjects[test_idx]))
    if test_subject in train_subjs:
        raise RuntimeError(
            f"LOSO leakage: test subject {test_subject!r} found in training subjects {train_subjs}"
        )
    if test_subjs != {test_subject}:
        raise RuntimeError(f"LOSO test set has unexpected subjects: {test_subjs}")


def assert_calibration_no_leakage(cal_idx: np.ndarray, test_idx: np.ndarray) -> None:
    overlap = set(cal_idx.tolist()) & set(test_idx.tolist())
    if overlap:
        raise RuntimeError(f"Calibration/test overlap: {len(overlap)} shared indices")
