"""
run_hhar_sequential.py
----------------------
Runs two HHAR experiments back-to-back on full data (454 K windows)
with val_loss-based early stopping:

  1. GNN-only LOSO          → gnn_hhar_fold{1-9}.pt
  2. CNN1D LOSO             → cnn1d_hhar_fold{1-9}.pt

Saves after EACH fold so a crash doesn't lose everything.
Logs fold time and running total so you know exactly how long is left.

Monitor:  grep "" /tmp/hhar_sequential.log
"""

from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (
    PROCESSED_DIR, MODELS_DIR, PLOTS_DIR, METRICS_DIR,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE, SEED,
    HHAR_NODE_FEAT_DIM, HHAR_ACTIVITIES,
)
from src.models import GNNOnlyModel, CNN1DModel
from src.dataset import HARGraphDataset, HARWindowDataset2D
from src.train import get_device, set_seed, loso_splits
from src.evaluation import get_predictions

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

METS   = Path(METRICS_DIR)
PLOTS  = Path(PLOTS_DIR)
MODELS = Path(MODELS_DIR)
for d in [METS, PLOTS, MODELS]:
    d.mkdir(parents=True, exist_ok=True)


def remap(y):
    classes = np.unique(y)
    mp = {int(old): int(new) for new, old in enumerate(classes)}
    return np.vectorize(mp.__getitem__)(y), mp


def loso_fold(model, tr_loader, val_loader, te_loader, device, tag_fold, use_adj):
    """
    Train one LOSO fold with val_loss early stopping.
    Returns (y_true, y_pred, best_val_loss).
    """
    opt   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit  = nn.CrossEntropyLoss()
    best_val_loss, best_state, pat = float("inf"), None, 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for batch in tr_loader:
            opt.zero_grad()
            if use_adj:
                x, adj, yb = batch
                loss = crit(model(x.to(device), adj.to(device)), yb.to(device))
            else:
                x, yb = batch
                loss = crit(model(x.to(device)), yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_loss_sum = correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                if use_adj:
                    x, adj, yb = batch
                    logits = model(x.to(device), adj.to(device))
                else:
                    x, yb = batch
                    logits = model(x.to(device))
                yb_d = yb.to(device)
                val_loss_sum += crit(logits, yb_d).item() * len(yb)
                correct      += (logits.argmax(1) == yb_d).sum().item()
                total        += len(yb)
        val_loss = val_loss_sum / total if total else float("inf")
        sched.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break

    model.load_state_dict(best_state)
    torch.save(best_state, MODELS / f"{tag_fold}.pt")
    yt, yp = get_predictions(model, te_loader, device, use_adj=use_adj)
    return np.array(yt), np.array(yp), best_val_loss


def run_experiment(name, dataset, subjects, model_factory, tag_prefix, n_cls, use_adj):
    print(f"\n{'='*70}")
    print(f"{name}  —  full HHAR  454,577 windows  val_loss early stopping")
    print(f"{'='*70}")
    device = get_device()
    all_true, all_pred = [], []
    fold_times = []
    t_exp = time.perf_counter()

    for fi, (tr_idx, te_idx, te_subj) in enumerate(loso_splits(subjects)):
        set_seed(SEED)
        n_val     = max(1, int(len(tr_idx) * 0.15))
        val_idx   = tr_idx[-n_val:]
        train_idx = tr_idx[:-n_val]

        tr_loader  = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
        val_loader = DataLoader(Subset(dataset, val_idx),   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        te_loader  = DataLoader(Subset(dataset, te_idx),    batch_size=BATCH_SIZE, shuffle=False)

        model = model_factory().to(device)
        t_fold = time.perf_counter()
        yt, yp, bvl = loso_fold(model, tr_loader, val_loader, te_loader, device,
                                  f"{tag_prefix}_fold{fi+1}", use_adj)
        fold_min = (time.perf_counter() - t_fold) / 60
        fold_times.append(fold_min)

        acc = accuracy_score(yt, yp)
        all_true.extend(yt); all_pred.extend(yp)

        folds_left = 9 - (fi + 1)
        avg_min    = sum(fold_times) / len(fold_times)
        eta_min    = avg_min * folds_left
        print(f"  Fold {fi+1}/9  test={te_subj}  acc={acc:.4f}  val_loss={bvl:.4f}  "
              f"[{fold_min:.1f} min]  ETA remaining: {eta_min:.0f} min", flush=True)

    yt_all = np.array(all_true);  yp_all = np.array(all_pred)
    acc  = float(accuracy_score(yt_all, yp_all))
    f1   = float(f1_score(yt_all, yp_all, average="macro", zero_division=0))
    ba   = float(balanced_accuracy_score(yt_all, yp_all))
    total_min = (time.perf_counter() - t_exp) / 60

    print(f"\n  [{tag_prefix}] Acc={acc:.4f}  F1={f1:.4f}  BalAcc={ba:.4f}  "
          f"Total={total_min:.1f} min")

    # Save arrays
    np.save(METS / f"{tag_prefix}_y_true.npy", yt_all)
    np.save(METS / f"{tag_prefix}_y_pred.npy", yp_all)

    # Confusion matrix
    classes = sorted(np.unique(yt_all).tolist())
    labels  = [HHAR_ACTIVITIES[k] if k < len(HHAR_ACTIVITIES) else str(k) for k in classes]
    cm = confusion_matrix(yt_all, yp_all, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax, annot_kws={"size": 9})
    ax.set_title(f"{name} HHAR (full) — LOSO  Acc={acc:.3f}", fontsize=11)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(PLOTS / f"cm_{tag_prefix}.png", dpi=150); plt.close(fig)
    print(f"  Saved cm_{tag_prefix}.png")

    return {"accuracy": acc, "macro_f1": f1, "balanced_acc": ba,
            "note": "full HHAR 454K windows, val_loss early stopping"}


def main():
    t_wall = time.time()
    print("=" * 70)
    print("HHAR Sequential Retraining  (val_loss early stopping, full 454K)")
    print("=" * 70)
    print("Order:  1. GNN-only   2. CNN1D")
    print("Estimated total wall time: 8–14 hours")
    print("Monitor:  grep \"\" /tmp/hhar_sequential.log")

    # Load data once
    p = Path(PROCESSED_DIR)
    X     = np.load(p / "hhar_X.npy")
    y_raw = np.load(p / "hhar_y.npy")
    subj  = np.load(p / "hhar_subjects.npy")
    y, _  = remap(y_raw)
    n_cls = len(np.unique(y))
    T, C  = X.shape[1], X.shape[2]
    print(f"\nLoaded: {len(X):,} windows  {n_cls} classes  device={get_device()}\n")

    results = {}

    # ── 1. GNN-only ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    graph_ds = HARGraphDataset(X, y, dataset="hhar", cache=True)
    print(f"HARGraphDataset init: {(time.perf_counter()-t0)*1000:.1f} ms  (lazy)")

    res = run_experiment(
        name          = "GNN-only",
        dataset       = graph_ds,
        subjects      = subj,
        model_factory = lambda: GNNOnlyModel(HHAR_NODE_FEAT_DIM, 2, n_cls),
        tag_prefix    = "gnn_hhar",
        n_cls         = n_cls,
        use_adj       = True,
    )
    results["gnn"] = res

    # Update hhar_deep_models.json after GNN finishes
    hd_path = METS / "hhar_deep_models.json"
    hd = json.load(open(hd_path)) if hd_path.exists() else {}
    hd["gnn"] = res
    json.dump(hd, open(hd_path, "w"), indent=2)
    print("  Updated hhar_deep_models.json  [gnn]")

    # ── 2. CNN1D ───────────────────────────────────────────────────────────
    win_ds = HARWindowDataset2D(X, y)

    res = run_experiment(
        name          = "CNN1D",
        dataset       = win_ds,
        subjects      = subj,
        model_factory = lambda: CNN1DModel(n_timesteps=T, n_channels=C, n_classes=n_cls),
        tag_prefix    = "cnn1d_hhar",
        n_cls         = n_cls,
        use_adj       = False,
    )
    results["cnn1d"] = res

    # Update cnn1d_results.json
    cnn_path = METS / "cnn1d_results.json"
    cnn = json.load(open(cnn_path)) if cnn_path.exists() else {}
    cnn["hhar"] = res
    json.dump(cnn, open(cnn_path, "w"), indent=2)
    print("  Updated cnn1d_results.json  [hhar]")

    # Also update hhar_deep_models.json with cnn1d
    hd = json.load(open(hd_path))
    hd["cnn1d"] = res
    json.dump(hd, open(hd_path, "w"), indent=2)

    # ── Final summary ──────────────────────────────────────────────────────
    total_hr = (time.time() - t_wall) / 3600
    print(f"\n{'='*70}")
    print(f"All done in {total_hr:.2f} hours")
    print(f"{'='*70}")
    for k, v in results.items():
        print(f"  {k:10s}  Acc={v['accuracy']:.4f}  F1={v['macro_f1']:.4f}  BalAcc={v['balanced_acc']:.4f}")


if __name__ == "__main__":
    main()
