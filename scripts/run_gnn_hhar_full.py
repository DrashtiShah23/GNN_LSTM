"""
run_gnn_hhar_full.py
--------------------
Runs GNN-only LOSO on the full HHAR dataset (454 K windows, no cap).
Saves:
  results/metrics/gnn_hhar_y_true.npy          ← overwrites old capped result
  results/metrics/gnn_hhar_y_pred.npy
  results/metrics/hhar_deep_models.json         ← updates gnn entry only
  results/plots/cm_gnn_hhar_full.png
  results/plots/complete_comparison_hhar.png    ← regenerated
"""

from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (
    PROCESSED_DIR, MODELS_DIR, PLOTS_DIR, METRICS_DIR,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE, SEED,
    HHAR_NODE_FEAT_DIM, HHAR_ACTIVITIES,
)
from src.models import GNNOnlyModel
from src.dataset import HARGraphDataset
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


# ── Helpers ────────────────────────────────────────────────────────────────

def remap(y):
    classes = np.unique(y)
    mp = {int(old): int(new) for new, old in enumerate(classes)}
    return np.vectorize(mp.__getitem__)(y), mp


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("GNN-only LOSO — full HHAR (454 K windows, no cap)")
    print("=" * 70)

    # Load full dataset
    p = Path(PROCESSED_DIR)
    X     = np.load(p / "hhar_X.npy")
    y_raw = np.load(p / "hhar_y.npy")
    subj  = np.load(p / "hhar_subjects.npy")
    y, _  = remap(y_raw)
    n_cls = len(np.unique(y))

    print(f"Loaded HHAR: {len(X):,} windows × {X.shape[1]} timesteps × {X.shape[2]} ch  |  {n_cls} classes")
    for s in np.unique(subj):
        print(f"  subj {s}: {(subj==s).sum():,} windows")

    # Lazy dataset — O(1) init
    t0 = time.perf_counter()
    dataset = HARGraphDataset(X, y, dataset="hhar", cache=True)
    print(f"\nDataset init: {(time.perf_counter()-t0)*1000:.1f} ms  (lazy)")

    device = get_device()
    print(f"Device: {device}\n")

    all_true, all_pred = [], []
    fold_accs = []

    for fi, (tr_idx, te_idx, te_subj) in enumerate(loso_splits(subj)):
        set_seed(SEED)

        n_val     = max(1, int(len(tr_idx) * 0.15))
        val_idx   = tr_idx[-n_val:]
        train_idx = tr_idx[:-n_val]

        tr_loader  = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
        val_loader = DataLoader(Subset(dataset, val_idx),   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        te_loader  = DataLoader(Subset(dataset, te_idx),    batch_size=BATCH_SIZE, shuffle=False)

        model = GNNOnlyModel(HHAR_NODE_FEAT_DIM, 2, n_cls).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        crit  = nn.CrossEntropyLoss()
        best_val_loss, best_state, pat = float("inf"), None, 0

        t_fold = time.perf_counter()
        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            for x, adj, yb in tr_loader:
                opt.zero_grad()
                loss = crit(model(x.to(device), adj.to(device)), yb.to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            # Validation: compute both loss and accuracy
            model.eval()
            val_loss_sum = correct = total = 0
            with torch.no_grad():
                for x, adj, yb in val_loader:
                    logits = model(x.to(device), adj.to(device))
                    yb_d   = yb.to(device)
                    val_loss_sum += crit(logits, yb_d).item() * len(yb)
                    correct      += (logits.argmax(1) == yb_d).sum().item()
                    total        += len(yb)
            val_loss = val_loss_sum / total if total else float("inf")
            val_acc  = correct / total if total else 0.0
            sched.step(val_loss)   # LR scheduler tracks loss

            # Early stopping tracks loss — unbiased on imbalanced classes
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in model.state_dict().items()}
                pat = 0
            else:
                pat += 1
                if pat >= PATIENCE:
                    break

        model.load_state_dict(best_state)
        torch.save(best_state, MODELS / f"gnn_hhar_fold{fi+1}.pt")

        yt, yp = get_predictions(model, te_loader, device, use_adj=True)
        acc = accuracy_score(yt, yp)
        fold_accs.append(acc)
        all_true.extend(yt)
        all_pred.extend(yp)
        elapsed = time.perf_counter() - t_fold
        print(f"  Fold {fi+1}/9  test={te_subj}  acc={acc:.4f}  best_val_loss={best_val_loss:.4f}  "
              f"(train={len(train_idx):,} val={len(val_idx):,} test={len(te_idx):,})  "
              f"[{elapsed/60:.1f} min]", flush=True)

    yt_all = np.array(all_true)
    yp_all = np.array(all_pred)

    acc  = float(accuracy_score(yt_all, yp_all))
    f1   = float(f1_score(yt_all, yp_all, average="macro", zero_division=0))
    ba   = float(balanced_accuracy_score(yt_all, yp_all))

    print(f"\n{'='*70}")
    print(f"GNN HHAR (full) — Acc={acc:.4f}  F1={f1:.4f}  BalAcc={ba:.4f}")
    print(f"{'='*70}\n")

    # ── Save arrays ─────────────────────────────────────────────────────
    np.save(METS / "gnn_hhar_y_true.npy", yt_all)
    np.save(METS / "gnn_hhar_y_pred.npy", yp_all)
    print("Saved gnn_hhar_y_true/pred.npy")

    # ── Update hhar_deep_models.json (gnn entry only) ───────────────────
    hd_path = METS / "hhar_deep_models.json"
    hd = json.load(open(hd_path)) if hd_path.exists() else {}
    hd["gnn"] = {"accuracy": acc, "macro_f1": f1, "balanced_acc": ba,
                 "note": "full HHAR 454K windows, no cap"}
    with open(hd_path, "w") as fh:
        json.dump(hd, fh, indent=2)
    print("Updated hhar_deep_models.json  [gnn entry]")

    # ── Confusion matrix plot ────────────────────────────────────────────
    classes = sorted(np.unique(yt_all).tolist())
    labels  = [HHAR_ACTIVITIES[k] if k < len(HHAR_ACTIVITIES) else str(k) for k in classes]
    cm = confusion_matrix(yt_all, yp_all, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax, annot_kws={"size": 9})
    ax.set_title(f"GNN-only HHAR (full data) — LOSO  Acc={acc:.3f}", fontsize=11)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(PLOTS / "cm_gnn_hhar_full.png", dpi=150)
    plt.close(fig)
    print("Saved cm_gnn_hhar_full.png")

    # ── Updated HHAR comparison bar chart ───────────────────────────────
    _make_hhar_comparison(acc, f1)

    print("\nAll outputs saved. Done.")


def _make_hhar_comparison(gnn_full_acc: float, gnn_full_f1: float):
    """Regenerate the HHAR model comparison chart with the new full-data GNN result."""
    METS_  = Path(METRICS_DIR)
    PLOTS_ = Path(PLOTS_DIR)

    hd   = json.load(open(METS_ / "hhar_deep_models.json"))   if (METS_ / "hhar_deep_models.json").exists()  else {}
    hhbl = json.load(open(METS_ / "HHAR_baselines.json"))     if (METS_ / "HHAR_baselines.json").exists()    else {}
    cnn  = json.load(open(METS_ / "cnn1d_results.json"))      if (METS_ / "cnn1d_results.json").exists()     else {}

    def g(d, k, sub="accuracy"):
        return d.get(k, {}).get(sub, d.get(k, {}).get("mean_accuracy", 0)) if d else 0

    rows = {
        "SVM":           g(hhbl, "SVM",          "mean_accuracy"),
        "Random Forest": g(hhbl, "RandomForest",  "mean_accuracy"),
        "XGBoost":       g(hhbl, "XGBoost",       "mean_accuracy"),
        "CNN1D":         g(cnn,  "hhar"),
        "LSTM-only":     g(hd,   "lstm"),
        "GNN-only\n(capped 5K)": 0.6014,           # old result for comparison
        "GNN-only\n(full 454K)": gnn_full_acc,      # new result
        "GNN+LSTM":      g(hd,   "gnn_lstm"),
    }

    names  = list(rows.keys())
    vals   = [v * 100 for v in rows.values()]
    colors = (["#4472C4"] * 3 +          # classical
              ["#ED7D31"] * 2 +          # CNN1D + LSTM
              ["#A9A9A9"]     +          # old GNN (grey = outdated)
              ["#70AD47"]     +          # new GNN (green = new)
              ["#ED7D31"])               # GNN+LSTM

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(names, vals, color=colors, alpha=0.88)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_ylabel("LOSO Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_title("HHAR Model Comparison (full-data GNN result highlighted)", fontweight="bold")

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#4472C4", label="Classical ML"),
        Patch(facecolor="#ED7D31", label="Deep Learning"),
        Patch(facecolor="#A9A9A9", label="GNN capped (old)"),
        Patch(facecolor="#70AD47", label="GNN full-data (new)"),
    ]
    ax.legend(handles=legend_els, loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(PLOTS_ / "complete_comparison_hhar.png", dpi=150)
    plt.close(fig)
    print("Saved complete_comparison_hhar.png")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total wall time: {(time.time()-t0)/60:.1f} min")
