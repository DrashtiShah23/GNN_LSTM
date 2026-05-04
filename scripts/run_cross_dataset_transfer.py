"""
Zero-shot cross-dataset transfer: ImprovedGNNLSTMModel.

Protocol (matches midway-report Table IV):
  - Train on ALL subjects of source dataset (no LOSO).
  - Evaluate on ALL subjects of target dataset without fine-tuning.
  - Shared activities only, PAMAP2 reduced to wrist accelerometer (ch 0-2)
    so both datasets share the same 3-channel, 3-node graph structure.

Shared activities (5):
  biking/cycling | sitting | standing | stairsup/ascending | walking
  Remapped to 0-4 in unified label space.

Channels used:
  HHAR   : ch 0-2  (acc x, y, z  — merged phone/watch accel)
  PAMAP2 : ch 0-2  (wrist_acc1_x, wrist_acc1_y, wrist_acc1_z)

Node structure (same for both after channel selection):
  n_nodes=3 (x=0, y=1, z=2), node_feat_dim=6 (6 stats × 1 ch per node)

Outputs:
  results/metrics/cross_dataset_transfer_results.json
  results/plots/cm_transfer_hhar2pamap2.png
  results/plots/cm_transfer_pamap2hhar.png
  results/plots/comparison_table_std_transfer.png  (printed + saved)
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    confusion_matrix, classification_report,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.config import (
    PROCESSED_DIR, METRICS_DIR, PLOTS_DIR,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE, SEED,
)
from src.dataset import HARSequenceDataset
from src.graph_construction import build_hhar_adj           # 3-node x/y/z adj (reused)
from src.models import ImprovedGNNLSTMModel
from src.train import get_device, set_seed

for d in [METRICS_DIR, PLOTS_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

# ── Shared activity definitions ───────────────────────────────────────────────
# Raw stored labels before remap() — these are le_map / LabelEncoder indices.

# HHAR: LabelEncoder sorts HHAR_ACTIVITIES alphabetically:
#   bike=0, sit=1, stand=2, stairsdown=3, stairsup=4, walk=5
HHAR_SHARED_RAW = {0: "cycling", 1: "sitting", 2: "standing", 4: "stairsup", 5: "walking"}

# PAMAP2: le_map[activity_id] for shared activity_ids:
#   cycling(6)=6, sitting(2)=2, standing(3)=3, ascending_stairs(12)=11, walking(4)=4
PAMAP2_SHARED_RAW = {6: "cycling", 2: "sitting", 3: "standing", 11: "stairsup", 4: "walking"}

# Unified label space (alphabetical for reproducibility)
SHARED_ACTIVITIES = ["cycling", "sitting", "standing", "stairsup", "walking"]
N_SHARED = len(SHARED_ACTIVITIES)          # 5
ACT2IDX  = {a: i for i, a in enumerate(SHARED_ACTIVITIES)}

SHARED_LABELS = ["cycling", "sitting", "standing", "stairsup↑", "walking"]


# ── Data loading helpers ───────────────────────────────────────────────────────

def load_hhar_shared(max_per_subject: int | None = 5000):
    X_all    = np.load(Path(PROCESSED_DIR) / "hhar_X.npy")      # (N,128,3)
    y_raw    = np.load(Path(PROCESSED_DIR) / "hhar_y.npy")
    subj_all = np.load(Path(PROCESSED_DIR) / "hhar_subjects.npy")

    mask = np.isin(y_raw, list(HHAR_SHARED_RAW.keys()))
    X, y_raw, subj = X_all[mask], y_raw[mask], subj_all[mask]

    # Remap to unified 0-4
    y = np.array([ACT2IDX[HHAR_SHARED_RAW[int(v)]] for v in y_raw], dtype=np.int64)

    if max_per_subject:
        rng = np.random.default_rng(SEED)
        keep = []
        for s in np.unique(subj):
            idx = np.where(subj == s)[0]
            if len(idx) > max_per_subject:
                idx = np.sort(rng.choice(idx, max_per_subject, replace=False))
            keep.append(idx)
        idx = np.concatenate(keep)
        X, y, subj = X[idx], y[idx], subj[idx]

    return X, y, subj


def load_pamap2_shared():
    X_all    = np.load(Path(PROCESSED_DIR) / "pamap2_X.npy")     # (N,128,18)
    y_raw    = np.load(Path(PROCESSED_DIR) / "pamap2_y.npy")
    subj_all = np.load(Path(PROCESSED_DIR) / "pamap2_subjects.npy")

    mask = np.isin(y_raw, list(PAMAP2_SHARED_RAW.keys()))
    X_full, y_raw, subj = X_all[mask], y_raw[mask], subj_all[mask]

    # Use only wrist accelerometer channels (0-2: acc1_x, acc1_y, acc1_z)
    X = X_full[:, :, :3]

    # Remap to unified 0-4
    y = np.array([ACT2IDX[PAMAP2_SHARED_RAW[int(v)]] for v in y_raw], dtype=np.int64)

    return X, y, subj


# ── Training helper ───────────────────────────────────────────────────────────

def train_full(model, dataset, device, val_frac: float = 0.15):
    """Train on all data with a held-out validation slice for early stopping."""
    set_seed(SEED)
    n      = len(dataset)
    n_val  = max(1, int(n * val_frac))
    tr_idx = list(range(n - n_val))
    va_idx = list(range(n - n_val, n))

    from torch.utils.data import Subset
    tr_loader = DataLoader(Subset(dataset, tr_idx), batch_size=BATCH_SIZE,
                           shuffle=True, drop_last=False)
    va_loader = DataLoader(Subset(dataset, va_idx), batch_size=BATCH_SIZE,
                           shuffle=False)

    opt   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit  = nn.CrossEntropyLoss()
    best_loss, best_state, pat = float("inf"), None, 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for x, adj, y in tr_loader:
            x, adj, y = x.to(device), adj.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x, adj), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, adj, y in va_loader:
                x, adj, y = x.to(device), adj.to(device), y.to(device)
                val_loss += crit(model(x, adj), y).item() * len(y)
        val_loss /= len(va_loader.dataset)
        sched.step(val_loss)

        if val_loss < best_loss:
            best_loss, best_state, pat = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_all(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    yt, yp = [], []
    for x, adj, y in loader:
        x, adj = x.to(device), adj.to(device)
        yp.extend(model(x, adj).argmax(1).cpu().numpy())
        yt.extend(y.numpy())
    return np.array(yt), np.array(yp)


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_cm(y_true, y_pred, labels, title, out_path):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.4, linecolor="lightgrey", vmin=0, vmax=1, ax=ax)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_table(rows, title, out_path):
    col_labels = ["Direction", "Accuracy", "Macro-F1", "Balanced Acc"]
    table_data = [[r["direction"], f"{r['acc']:.4f}", f"{r['f1']:.4f}", f"{r['bacc']:.4f}"]
                  for r in rows]
    colors = [["#d4edda"] * 4 for _ in rows]

    fig, ax = plt.subplots(figsize=(10, 0.6 + 0.5 * len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   cellLoc="center", loc="center", cellColours=colors)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#343a40")
            cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#dee2e6")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Main transfer experiment ──────────────────────────────────────────────────

def run_transfer(direction: str, seq_len: int = 10) -> dict:
    """direction: 'hhar2pamap2' or 'pamap2hhar'"""
    device = get_device()
    adj    = build_hhar_adj()          # 3-node fully connected adj (x/y/z)
    node_feat_dim = 6                  # 1 ch/node × 6 stats (same for both)
    n_nodes       = 3

    if direction == "hhar2pamap2":
        src_label = "HHAR → PAMAP2"
        X_src, y_src, subj_src = load_hhar_shared(max_per_subject=5000)
        X_tgt, y_tgt, _        = load_pamap2_shared()
    else:
        src_label = "PAMAP2 → HHAR"
        X_src, y_src, _ = load_pamap2_shared()
        X_tgt, y_tgt, _ = load_hhar_shared(max_per_subject=5000)

    print(f"\n  [{src_label}]  src={len(X_src)} windows  tgt={len(X_tgt)} windows")
    print(f"   source classes: {dict(zip(*np.unique(y_src, return_counts=True)))}")
    print(f"   target classes: {dict(zip(*np.unique(y_tgt, return_counts=True)))}")

    # Build datasets (HARSequenceDataset for both using the fixed HHAR-style adj)
    src_subj = subj_src if direction == "hhar2pamap2" else np.zeros(len(X_src), dtype=int)
    ds_src = HARSequenceDataset(X_src, y_src, subjects=src_subj,
                                dataset="hhar", seq_len=seq_len, cache=True)
    ds_tgt = HARSequenceDataset(X_tgt, y_tgt,
                                subjects=np.zeros(len(X_tgt), dtype=int),
                                dataset="hhar", seq_len=seq_len, cache=True)

    model = ImprovedGNNLSTMModel(
        node_feat_dim=node_feat_dim, n_nodes=n_nodes, n_classes=N_SHARED
    ).to(device)

    t0 = time.time()
    model = train_full(model, ds_src, device)
    y_true, y_pred = predict_all(model, ds_tgt, device)
    elapsed = time.time() - t0

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bacc = balanced_accuracy_score(y_true, y_pred)

    print(f"  {src_label}: acc={acc:.4f}  macro-F1={f1:.4f}  bal-acc={bacc:.4f}  ({elapsed:.0f}s)")
    print(classification_report(y_true, y_pred, target_names=SHARED_LABELS, zero_division=0))

    # Save predictions for confusion matrix
    tag = f"transfer_{direction}"
    np.save(Path(METRICS_DIR) / f"{tag}_y_true.npy", y_true)
    np.save(Path(METRICS_DIR) / f"{tag}_y_pred.npy", y_pred)

    return {"direction": src_label, "acc": acc, "f1": f1, "bacc": bacc,
            "n_src": int(len(y_true)), "n_tgt": int(len(y_pred))}


def main():
    print("\n" + "="*68)
    print("  Cross-Dataset Transfer — ImprovedGNNLSTM (zero-shot)")
    print("  Shared activities: cycling, sitting, standing, stairsup, walking")
    print("  Feature space: wrist accel x/y/z → 3 nodes × 6 stats")
    print("="*68)

    results = []
    for direction in ["hhar2pamap2", "pamap2hhar"]:
        r = run_transfer(direction)
        results.append(r)

    # Confusion matrices
    print("\n── Confusion matrices ──────────────────────────────────────────")
    dir_map = {"hhar2pamap2": results[0], "pamap2hhar": results[1]}
    for direction in ["hhar2pamap2", "pamap2hhar"]:
        yt = np.load(Path(METRICS_DIR) / f"transfer_{direction}_y_true.npy")
        yp = np.load(Path(METRICS_DIR) / f"transfer_{direction}_y_pred.npy")
        r  = dir_map[direction]
        title = f"{r['direction']} transfer (zero-shot, {N_SHARED} shared activities)"
        fname = f"cm_transfer_{direction}.png"
        plot_cm(yt, yp, SHARED_LABELS, title, Path(PLOTS_DIR) / fname)

    # Summary table
    print("\n── Transfer results ────────────────────────────────────────────")
    print(f"  {'Direction':<25} {'Accuracy':>10} {'Macro-F1':>10} {'Bal-Acc':>10}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10}")
    for r in results:
        print(f"  {r['direction']:<25} {r['acc']:>10.4f} {r['f1']:>10.4f} {r['bacc']:>10.4f}")

    plot_table(results,
               "Cross-Dataset Transfer — ImprovedGNNLSTM (zero-shot, 5 shared activities)",
               Path(PLOTS_DIR) / "comparison_table_transfer.png")

    # Save JSON
    out = Path(METRICS_DIR) / "cross_dataset_transfer_improved.json"
    out.write_text(json.dumps({"shared_activities": SHARED_ACTIVITIES,
                               "n_shared_classes": N_SHARED,
                               "channel_info": "HHAR: acc x/y/z (ch0-2); PAMAP2: wrist acc1_x/y/z (ch0-2)",
                               "results": results}, indent=2))
    print(f"\n  Results saved → {out}")


if __name__ == "__main__":
    main()
