"""Test-time perturbations for robustness experiments."""

from __future__ import annotations

from enum import Enum

import numpy as np


class PerturbationType(str, Enum):
    REMOVE_ONE_SENSOR_NODE = "remove_one_sensor_node"
    REMOVE_HEART_RATE = "remove_heart_rate_channel"
    GAUSSIAN_NOISE = "gaussian_noise"
    MASK_RANDOM_CHANNELS = "mask_random_channels"
    REMOVE_RANDOM_WINDOWS = "remove_random_windows"
    MISSING_HEART_RATE = "missing_heart_rate_signals"


SEVERITY_LEVELS = ("low", "medium", "high")

NOISE_STD = {"low": 0.05, "medium": 0.15, "high": 0.30}
MASK_FRAC = {"low": 0.10, "medium": 0.30, "high": 0.50}
WINDOW_DROP_FRAC = {"low": 0.10, "medium": 0.30, "high": 0.50}


def _pamap2_node_channels(n_channels: int = 18) -> list[tuple[int, int]]:
    """Return (start, end) channel indices per node (wrist, chest, ankle)."""
    per = n_channels // 3
    return [(0, per), (per, 2 * per), (2 * per, n_channels)]


def apply_perturbation_x(
    X: np.ndarray,
    perturbation: PerturbationType,
    severity: str,
    dataset: str,
    seed: int = 42,
) -> np.ndarray:
    """Apply perturbation to window tensor X (N, T, C). Returns modified copy."""
    rng = np.random.default_rng(seed)
    Xp = X.copy().astype(np.float64)

    if perturbation == PerturbationType.REMOVE_HEART_RATE:
        return Xp  # N/A — no HR in processed features; caller marks N/A

    if perturbation == PerturbationType.MISSING_HEART_RATE:
        return Xp  # N/A

    if perturbation == PerturbationType.REMOVE_ONE_SENSOR_NODE:
        if dataset != "pamap2":
            # HHAR: zero one axis node (1 channel block)
            node = 0
            Xp[:, :, node] = 0.0
        else:
            nodes = _pamap2_node_channels(X.shape[2])
            lo, hi = nodes[0]  # remove wrist
            Xp[:, :, lo:hi] = 0.0
        return Xp

    if perturbation == PerturbationType.GAUSSIAN_NOISE:
        std = NOISE_STD[severity] * np.std(X, axis=(0, 1), keepdims=True)
        Xp += rng.normal(0, std + 1e-8, size=Xp.shape)
        return Xp

    if perturbation == PerturbationType.MASK_RANDOM_CHANNELS:
        frac = MASK_FRAC[severity]
        n_ch = X.shape[2]
        n_mask = max(1, int(n_ch * frac))
        channels = rng.choice(n_ch, n_mask, replace=False)
        Xp[:, :, channels] = 0.0
        return Xp

    raise ValueError(f"Unsupported window perturbation: {perturbation}")


def drop_random_windows(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    severity: str,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    frac = WINDOW_DROP_FRAC[severity]
    keep = []
    for i in range(len(X)):
        if rng.random() > frac:
            keep.append(i)
    if not keep:
        keep = [0]
    keep = np.array(keep)
    return X[keep], y[keep], subjects[keep]
