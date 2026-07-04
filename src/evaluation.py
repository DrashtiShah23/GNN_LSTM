"""
Evaluation utilities: metrics, confusion matrix, LOSO evaluation loop,
SHAP and LIME interpretability stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.config import METRICS_DIR, PLOTS_DIR, BATCH_SIZE
from src.train import evaluate, loso_splits, get_device


# ── Core metrics ──────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str] | None = None) -> Dict:
    """Return a dict of key metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "report": classification_report(y_true, y_pred, target_names=label_names, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


# ── Get all predictions ───────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_adj: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y_true, y_pred) arrays."""
    model.eval()
    all_true, all_pred = [], []
    for batch in loader:
        if use_adj:
            x, adj, y = batch
            x, adj = x.to(device), adj.to(device)
            logits = model(x, adj)
        else:
            x, y = batch
            logits = model(x.to(device))
        all_true.extend(y.numpy())
        all_pred.extend(logits.argmax(dim=1).cpu().numpy())
    return np.array(all_true), np.array(all_pred)


# ── LOSO evaluation loop ──────────────────────────────────────────────────────

def loso_evaluate(
    model_cls,
    model_kwargs: dict,
    dataset,
    subjects: np.ndarray,
    use_adj: bool = True,
    run_name: str = "model",
) -> Dict:
    """
    Run leave-one-subject-out evaluation.
    Returns aggregated metrics across all folds.
    """
    from torch.utils.data import DataLoader
    from src.train import train_model, set_seed

    device = get_device()
    criterion = nn.CrossEntropyLoss()

    all_true, all_pred = [], []
    fold_accs = []

    for fold, (train_idx, test_idx, test_subj) in enumerate(loso_splits(subjects)):
        set_seed()
        print(f"\n── Fold {fold + 1}: test subject = {test_subj} ──")

        # Create subsets
        train_set = Subset(dataset, train_idx)
        test_set = Subset(dataset, test_idx)

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

        # Reinitialise model for each fold
        model = model_cls(**model_kwargs).to(device)

        train_model(
            model, train_loader, test_loader,
            use_adj=use_adj,
            run_name=f"{run_name}_fold{fold + 1}",
        )

        y_true, y_pred = get_predictions(model, test_loader, device, use_adj)
        all_true.extend(y_true)
        all_pred.extend(y_pred)
        fold_acc = accuracy_score(y_true, y_pred)
        fold_accs.append(fold_acc)
        print(f"  Fold accuracy: {fold_acc:.4f}")

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    metrics = compute_metrics(all_true, all_pred)
    metrics["per_fold_accuracy"] = fold_accs
    metrics["mean_fold_accuracy"] = float(np.mean(fold_accs))
    metrics["std_fold_accuracy"] = float(np.std(fold_accs))

    # Save metrics
    out_dir = Path(METRICS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{run_name}_y_true.npy", all_true)
    np.save(out_dir / f"{run_name}_y_pred.npy", all_pred)

    print(f"\n── LOSO Results for {run_name} ──")
    print(f"  Mean accuracy : {metrics['mean_fold_accuracy']:.4f} ± {metrics['std_fold_accuracy']:.4f}")
    print(f"  Macro F1      : {metrics['macro_f1']:.4f}")
    print(metrics["report"])

    return metrics


# ── Plotting utilities ────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: List[str],
    title: str = "Confusion Matrix",
    save_name: str | None = None,
) -> None:
    """Plot and optionally save confusion matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    if save_name:
        out_dir = Path(PLOTS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{save_name}.png", dpi=150)
        print(f"  Saved plot to {PLOTS_DIR}/{save_name}.png")
    plt.show()


def plot_training_history(
    history: dict,
    run_name: str = "model",
    *,
    best_epoch: int | None = None,
    stop_epoch: int | None = None,
    early_stopped: bool | None = None,
    reg_summary: str | None = None,
    suptitle: str | None = None,
    output_path: Path | None = None,
) -> Path:
    """Plot train/val loss and val accuracy curves; save PNG under PLOTS_DIR."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n_epochs = len(history["train_loss"])
    epochs = list(range(1, n_epochs + 1))
    if best_epoch is None:
        best_epoch = int(np.argmin(history["val_loss"])) + 1
    if stop_epoch is None:
        stop_epoch = n_epochs

    has_footer = reg_summary is not None
    fig = plt.figure(figsize=(12, 7.2 if has_footer else 4.0))
    gs = GridSpec(2 if has_footer else 1, 2, figure=fig, height_ratios=[4, 1.55] if has_footer else [1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_note = fig.add_subplot(gs[1, :]) if has_footer else None

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.axvline(best_epoch, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.85,
                label=f"Best epoch ({best_epoch})")
    show_stop = early_stopped or (stop_epoch != best_epoch)
    if show_stop:
        ax1.axvline(stop_epoch, color="#d62728", linestyle=":", linewidth=1.2, alpha=0.85,
                    label=f"Stop epoch ({stop_epoch})")
    ax1.set_title("Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right", fontsize=8)

    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="green")
    ax2.axvline(best_epoch, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.85,
                label="Best epoch")
    if show_stop:
        ax2.axvline(stop_epoch, color="#d62728", linestyle=":", linewidth=1.2, alpha=0.85,
                    label="Stop epoch")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="lower right", fontsize=8)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight="bold")

    if ax_note is not None and reg_summary:
        ax_note.axis("off")
        ax_note.text(
            0.5, 0.5, reg_summary,
            transform=ax_note.transAxes,
            ha="center", va="center",
            fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f7f7f7", edgecolor="#cccccc"),
        )

    fig.tight_layout()
    out_path = Path(output_path) if output_path else Path(PLOTS_DIR) / f"{run_name}_loss_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved training history to {out_path}")
    return out_path


def plot_loso_folds_grid(
    histories: list[dict],
    titles: list[str],
    out_path: Path,
    *,
    suptitle: str | None = None,
) -> Path:
    """3×N grid of train/val loss curves (one panel per LOSO fold)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(histories)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.8 * nrows), squeeze=False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    for i, (hist, title) in enumerate(zip(histories, titles)):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        epochs = list(range(1, len(hist["train_loss"]) + 1))
        best_epoch = hist.get("best_epoch") or int(np.argmin(hist["val_loss"])) + 1
        stop_epoch = hist.get("stop_epoch") or len(epochs)
        ax.plot(epochs, hist["train_loss"], label="Train", linewidth=1.2)
        ax.plot(epochs, hist["val_loss"], label="Val", linewidth=1.2)
        ax.axvline(best_epoch, color="#2ca02c", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.axvline(stop_epoch, color="#d62728", linestyle=":", linewidth=0.9, alpha=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96] if suptitle else None)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fold grid to {out_path}")
    return out_path
