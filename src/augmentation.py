"""
Data augmentation utilities for HAR sensor windows.

All transforms operate on numpy arrays of shape (N, WINDOW_SIZE, n_channels).

Available:
  gaussian_noise   — add random Gaussian noise to signals
  time_warp        — stretch/compress time axis with random piecewise linear warp
  amplitude_scale  — randomly scale signal amplitude per channel
  augment_dataset  — apply one or more transforms to a dataset
"""

from __future__ import annotations

import numpy as np


def gaussian_noise(X: np.ndarray, sigma: float = 0.05, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Add Gaussian noise N(0, sigma) to every sample.

    Parameters
    ----------
    X     : (N, T, C)
    sigma : noise standard deviation (relative to signal std by default)

    Returns
    -------
    noisy X, same shape
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=X.shape).astype(np.float32)
    return (X + noise).astype(np.float32)


def amplitude_scale(X: np.ndarray, scale_range: tuple = (0.8, 1.2),
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Randomly scale each window's amplitude by a factor in scale_range.
    The same scale factor is applied to all channels of one window.

    Parameters
    ----------
    X           : (N, T, C)
    scale_range : (min_scale, max_scale)
    """
    if rng is None:
        rng = np.random.default_rng()
    scales = rng.uniform(scale_range[0], scale_range[1], size=(len(X), 1, 1)).astype(np.float32)
    return (X * scales).astype(np.float32)


def time_warp(X: np.ndarray, sigma: float = 0.1, knots: int = 4,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Time-warping augmentation using piecewise linear interpolation.
    Randomly stretches/compresses the time axis.

    Parameters
    ----------
    X      : (N, T, C)
    sigma  : std of random knot displacements
    knots  : number of interior warp knots
    """
    if rng is None:
        rng = np.random.default_rng()
    N, T, C = X.shape
    result = np.empty_like(X)
    orig_steps = np.linspace(0, T - 1, T)

    for i in range(N):
        # Random knot positions
        knot_steps = np.linspace(0, T - 1, knots + 2)
        displace   = rng.normal(0.0, sigma * T, size=knots + 2)
        displace[0] = 0.0; displace[-1] = 0.0  # anchor endpoints
        warped_knots = knot_steps + displace
        warped_knots = np.clip(warped_knots, 0, T - 1)
        warped_knots = np.sort(warped_knots)

        # Map original steps through the warp
        new_steps = np.interp(orig_steps, warped_knots, knot_steps)
        new_steps = np.clip(new_steps, 0, T - 1)

        for c in range(C):
            result[i, :, c] = np.interp(orig_steps, new_steps, X[i, :, c])

    return result.astype(np.float32)


def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "gaussian",
    sigma: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment a dataset by applying a transform and appending to the original.
    Returns (X_aug, y_aug) with 2× the original samples.

    Parameters
    ----------
    X      : (N, T, C)
    y      : (N,)
    method : one of 'gaussian', 'scale', 'timewarp'
    sigma  : noise/warp strength parameter
    seed   : random seed

    Returns
    -------
    X_aug : (2N, T, C)
    y_aug : (2N,)
    """
    rng = np.random.default_rng(seed)
    if method == "gaussian":
        X_new = gaussian_noise(X, sigma=sigma, rng=rng)
    elif method == "scale":
        X_new = amplitude_scale(X, rng=rng)
    elif method == "timewarp":
        X_new = time_warp(X, sigma=sigma, rng=rng)
    else:
        raise ValueError(f"Unknown augmentation method: {method}. "
                         f"Choose from 'gaussian', 'scale', 'timewarp'.")

    X_out = np.concatenate([X, X_new], axis=0).astype(np.float32)
    y_out = np.concatenate([y, y],     axis=0)
    return X_out, y_out
