"""Windowing utilities: overlapping vs non-overlapping."""

from __future__ import annotations

import numpy as np

from src.config import OVERLAP, WINDOW_SIZE


def overlap_stride(window_size: int = WINDOW_SIZE, overlap: float = OVERLAP) -> int:
    return int(window_size * (1 - overlap))


def subsample_non_overlapping_windows(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    *,
    window_size: int = WINDOW_SIZE,
    overlap_stride_val: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Derive non-overlapping windows from preprocessed overlapping windows.

    Processed arrays use stride = overlap_stride (64). Taking every 2nd window
    per subject yields stride 128 (non-overlapping), documented in README.
    """
    step = 2 if overlap_stride_val is None else max(1, window_size // overlap_stride_val)
    keep = []
    for subj in np.unique(subjects):
        idx = np.where(subjects == subj)[0]
        keep.append(idx[::step])
    keep = np.concatenate(keep)
    return X[keep], y[keep], subjects[keep]


def cap_windows_per_subject(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    max_per_subject: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keep = []
    for s in np.unique(subjects):
        idx = np.where(subjects == s)[0]
        if len(idx) > max_per_subject:
            idx = np.sort(rng.choice(idx, max_per_subject, replace=False))
        keep.append(idx)
    keep = np.concatenate(keep)
    return X[keep], y[keep], subjects[keep]
