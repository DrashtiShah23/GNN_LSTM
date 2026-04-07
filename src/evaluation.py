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


def plot_training_history(history: dict, run_name: str = "model") -> None:
    """Plot train/val loss and val accuracy curves."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_title("Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history["val_acc"], label="Val Accuracy", color="green")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    out_dir = Path(PLOTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{run_name}_history.png", dpi=150)
    print(f"  Saved training history to {PLOTS_DIR}/{run_name}_history.png")
    plt.show()
