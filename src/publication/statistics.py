"""Statistical testing for model comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def mean_ci95(values: np.ndarray) -> tuple[float, float, float]:
    m = float(np.mean(values))
    s = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    if len(values) <= 1:
        return m, m, m
    se = s / np.sqrt(len(values))
    h = 1.96 * se
    return m, m - h, m + h


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    if np.allclose(diff, 0):
        return 1.0
    try:
        _, p = stats.wilcoxon(diff, alternative="two-sided")
        return float(p)
    except ValueError:
        return 1.0


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    if len(diff) < 2:
        return 0.0
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-8))


def cohens_d_one_sample(values: np.ndarray, mu0: float) -> float:
    if len(values) < 2:
        return 0.0
    return float((np.mean(values) - mu0) / (np.std(values, ddof=1) + 1e-8))


def wilcoxon_one_sample(values: np.ndarray, mu0: float) -> float:
    """One-sample Wilcoxon signed-rank test: H0 median(values) == mu0."""
    diff = values - mu0
    if np.allclose(diff, 0):
        return 1.0
    try:
        _, p = stats.wilcoxon(diff, alternative="two-sided")
        return float(p)
    except ValueError:
        return 1.0


def bootstrap_mean_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diffs = a - b
    observed = float(np.mean(diffs))
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(diffs), len(diffs))
        boots.append(float(np.mean(diffs[idx])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return observed, float(lo), float(hi)


def rank_stability(fold_scores: dict[str, list[float]]) -> dict[str, dict[int, int]]:
    """How often each model ranks 1st, 2nd, ... across folds."""
    models = list(fold_scores.keys())
    n_folds = len(next(iter(fold_scores.values())))
    counts = {m: {} for m in models}
    for fi in range(n_folds):
        scores = {m: fold_scores[m][fi] for m in models}
        ranked = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        for rank, m in enumerate(ranked, start=1):
            counts[m][rank] = counts[m].get(rank, 0) + 1
    return counts


def format_rank_stability(counts: dict[int, int], n_folds: int) -> str:
    parts = [f"rank{r}:{c}/{n_folds}" for r, c in sorted(counts.items())]
    return "; ".join(parts)
