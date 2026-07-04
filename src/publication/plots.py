"""Plotting utilities for publication experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def grouped_bar_chart(
    rows: list[dict],
    x_col: str,
    y_col: str,
    hue_col: str,
    facet_col: str | None,
    title: str,
    out_path: Path,
    ylabel: str = "Score",
) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)
    if facet_col and facet_col in df.columns:
        facets = df[facet_col].unique()
        fig, axes = plt.subplots(1, len(facets), figsize=(5 * len(facets), 5), squeeze=False)
        for i, facet in enumerate(facets):
            sub = df[df[facet_col] == facet]
            ax = axes[0, i]
            x_vals = sub[x_col].unique()
            width = 0.35
            hues = sub[hue_col].unique()
            for j, h in enumerate(hues):
                vals = [sub[(sub[x_col] == x) & (sub[hue_col] == h)][y_col].mean() for x in x_vals]
                xs = np.arange(len(x_vals)) + j * width
                ax.bar(xs, vals, width, label=str(h))
            ax.set_xticks(np.arange(len(x_vals)) + width * (len(hues) - 1) / 2)
            ax.set_xticklabels(x_vals, rotation=45, ha="right")
            ax.set_title(str(facet))
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        x_vals = df[x_col].unique()
        width = 0.35
        hues = df[hue_col].unique()
        for j, h in enumerate(hues):
            vals = [df[(df[x_col] == x) & (df[hue_col] == h)][y_col].mean() for x in x_vals]
            xs = np.arange(len(x_vals)) + j * width
            ax.bar(xs, vals, width, label=str(h))
        ax.set_xticks(np.arange(len(x_vals)) + width * (len(hues) - 1) / 2)
        ax.set_xticklabels(x_vals, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.legend()
    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def reliability_diagram(y_true: np.ndarray, probs: np.ndarray, title: str, out_path: Path, n_bins: int = 15) -> None:
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    accs, confs = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf <= hi)
        if mask.sum() == 0:
            continue
        accs.append((preds[mask] == y_true[mask]).mean())
        confs.append(conf[mask].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.plot(confs, accs, "o-", label="Model")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def degradation_curve(severities: list, drops: list, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(severities, drops, "o-")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Accuracy Drop")
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def subject_activity_heatmap(data: np.ndarray, row_labels: list, col_labels: list, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(col_labels)), max(6, len(row_labels) * 0.4)))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=col_labels, yticklabels=row_labels, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, labels, title, out_path):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
