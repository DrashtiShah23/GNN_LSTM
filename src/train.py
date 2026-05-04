"""
Training loop utilities: train_epoch, evaluate, early stopping, LOSO split.
"""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.config import (
    LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE,
    BATCH_SIZE, SEED, MODELS_DIR,
)


def set_seed(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Use env `HAR_FORCE_DEVICE=cpu` to avoid MPS OOM / driver crashes on long LOSO runs."""
    forced = os.environ.get("HAR_FORCE_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda", "mps"):
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")   # Apple Silicon
    return torch.device("cpu")


# ── Single epoch helpers ──────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_adj: bool = True,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        if use_adj:
            x, adj, y = batch
            x, adj, y = x.to(device), adj.to(device), y.to(device)
            logits = model(x, adj)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_adj: bool = True,
) -> Tuple[float, float]:
    """Returns (loss, accuracy)."""
    model.eval()
    total_loss, correct = 0.0, 0
    for batch in loader:
        if use_adj:
            x, adj, y = batch
            x, adj, y = x.to(device), adj.to(device), y.to(device)
            logits = model(x, adj)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


# ── Full training run with early stopping ────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    use_adj: bool = True,
    run_name: str = "model",
) -> Dict:
    device = get_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_val_acc  = 0.0          # tracked for logging only
    best_state = None
    patience_count = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, use_adj)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_adj)
        scheduler.step(val_loss)   # LR scheduler tracks loss

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Early stopping on val_loss — unbiased on imbalanced classes
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model
    out_dir = Path(MODELS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / f"{run_name}.pt")
    print(f"  Best val loss: {best_val_loss:.4f}  val acc: {best_val_acc:.4f} — saved to {MODELS_DIR}/{run_name}.pt")

    return {"history": history, "best_val_acc": best_val_acc}


# ── LOSO split utility ────────────────────────────────────────────────────────

def loso_splits(subjects: np.ndarray):
    """
    Yields (train_indices, test_indices) for each unique subject.
    Used for leave-one-subject-out evaluation.
    """
    unique_subjects = np.unique(subjects)
    for test_subj in unique_subjects:
        test_idx = np.where(subjects == test_subj)[0]
        train_idx = np.where(subjects != test_subj)[0]
        yield train_idx, test_idx, test_subj
