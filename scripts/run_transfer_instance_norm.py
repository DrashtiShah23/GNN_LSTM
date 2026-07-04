"""
Cross-dataset transfer with instance normalisation.

Same protocol as run_cross_dataset_transfer.py (zero-shot, 5 shared
activities, wrist acc ch0-2, 3-node graph) but each window is z-scored
per channel before graph feature extraction.  This removes sensor-level
bias caused by different device calibration / placement between PAMAP2
and HHAR.

Outputs:
  results/metrics/transfer_instance_norm_results.json
  results/plots/cm_transfer_norm_hhar2pamap2.png
  results/plots/cm_transfer_norm_pamap2hhar.png
  results/plots/comparison_table_transfer_norm.png
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
from src.graph_construction import build_hhar_adj
from src.models import ImprovedGNNLSTMModel
from src.train import get_device, set_seed

for d in [METRICS_DIR, PLOTS_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

# ── Shared activity definitions ───────────────────────────────────────────────
HHAR_SHARED_RAW   = {0: "cycling", 1: "sitting", 2: "standing", 4: "stairsup", 5: "walking"}
PAMAP2_SHARED_RAW = {6: "cycling", 2: "sitting", 3: "standing", 11: "stairsup", 4: "walking"}
SHARED_ACTIVITIES = ["cycling", "sitting", "standing", "stairsup", "walking"]
N_SHARED = len(SHARED_ACTIVITIES)
ACT2IDX  = {a: i for i, a in enumerate(SHARED_ACTIVITIES)}
SHARED_LABELS = ["cycling", "sitting", "standing", "stairsup↑", "walking"]


# ── Instance normalisation ────────────────────────────────────────────────────

def instance_normalize(X: np.ndarray) -> np.ndarray:
    """Z-score each window per channel independently.

    X: (N, T, C) — normalises along the T axis so each (T,) slice has
    zero mean and unit variance.  Prevents sensor-level offset / scale
    differences between PAMAP2 and HHAR from dominating the features.
    """
    mean = X.mean(axis=1, keepdims=True)          # (N, 1, C)
    std  = X.std(axis=1, keepdims=True) + 1e-8    # (N, 1, C)
    return (X - mean) / std


# ── Data loading ──────────────────────────────────────────────────────────────

def load_hhar_shared(max_per_subject: int | None = 5000):
    X_all    = np.load(Path(PROCESSED_DIR) / "hhar_X.npy")
    y_raw    = np.load(Path(PROCESSED_DIR) / "hhar_y.npy")
    subj_all = np.load(Path(PROCESSED_DIR) / "hhar_subjects.npy")

    mask = np.isin(y_raw, list(HHAR_SHARED_RAW.keys()))
    X, y_raw, subj = X_all[mask], y_raw[mask], subj_all[mask]
    y = np.array([ACT2IDX[HHAR_SHARED_RAW[int(v)]] for v in y_raw], dtype=np.int64)

    if max_per_subject:
        rng  = np.random.default_rng(SEED)
        keep = []
        for s in np.unique(subj):
            idx = np.where(subj == s)[0]
            if len(idx) > max_per_subject:
                idx = np.sort(rng.choice(idx, max_per_subject, replace=False))
            keep.append(idx)
        idx = np.concatenate(keep)
        X, y, subj = X[idx], y[idx], subj[idx]

    return instance_normalize(X), y, subj


def load_pamap2_shared():
    X_all    = np.load(Path(PROCESSED_DIR) / "pamap2_X.npy")
    y_raw    = np.load(Path(PROCESSED_DIR) / "pamap2_y.npy")
    subj_all = np.load(Path(PROCESSED_DIR) / "pamap2_subjects.npy")

    mask = np.isin(y_raw, list(PAMAP2_SHARED_RAW.keys()))
    X_full, y_raw, subj = X_all[mask], y_raw[mask], subj_all[mask]
    X = X_full[:, :, :3]   # wrist acc x/y/z only
    y = np.array([ACT2IDX[PAMAP2_SHARED_RAW[int(v)]] for v in y_raw], dtype=np.int64)

    return instance_normalize(X), y, subj


# ── Training / inference ──────────────────────────────────────────────────────

def train_full(model, dataset, device, val_frac: float = 0.15):
    set_seed(SEED)
    n     = len(dataset)
    n_val = max(1, int(n * val_frac))
    from torch.utils.data import Subset
    tr_loader = DataLoader(Subset(dataset, list(range(n - n_val))),
                           batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    va_loader = DataLoader(Subset(dataset, list(range(n - n_val, n))),
                           batch_size=BATCH_SIZE, shuffle=False)

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


def plot_comparison_table(rows, out_path):
    """Side-by-side before/after table."""
    col_labels = ["Direction", "Zero-shot Acc", "Norm Acc", "Δ Acc",
                                "Zero-shot F1",  "Norm F1",  "Δ F1"]
    table_data, row_colors = [], []
    for r in rows:
        d_acc = r["norm_acc"] - r["zero_acc"]
        d_f1  = r["norm_f1"]  - r["zero_f1"]
        table_data.append([
            r["direction"],
            f"{r['zero_acc']:.4f}", f"{r['norm_acc']:.4f}", f"{d_acc:+.4f}",
            f"{r['zero_f1']:.4f}",  f"{r['norm_f1']:.4f}",  f"{d_f1:+.4f}",
        ])
        color = "#d4edda" if d_acc > 0 else "#fce8e8"
        row_colors.append([color] * 7)

    fig, ax = plt.subplots(figsize=(14, 0.6 + 0.7 * len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   cellLoc="center", loc="center", cellColours=row_colors)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#343a40")
            cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#dee2e6")
    ax.set_title("Cross-Dataset Transfer: Zero-shot vs Instance Normalisation",
                 fontsize=12, fontweight="bold", pad=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_transfer_norm(direction: str, seq_len: int = 10) -> dict:
    device        = get_device()
    adj           = build_hhar_adj()
    node_feat_dim = 6
    n_nodes       = 3

    if direction == "hhar2pamap2":
        src_label = "HHAR → PAMAP2"
        X_src, y_src, subj_src = load_hhar_shared(max_per_subject=5000)
        X_tgt, y_tgt, _        = load_pamap2_shared()
    else:
        src_label = "PAMAP2 → HHAR"
        X_src, y_src, subj_src = load_pamap2_shared()
        X_tgt, y_tgt, _        = load_hhar_shared(max_per_subject=5000)

    print(f"\n  [{src_label}]  src={len(X_src)}  tgt={len(X_tgt)}")

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
    model    = train_full(model, ds_src, device)
    y_true, y_pred = predict_all(model, ds_tgt, device)
    elapsed  = time.time() - t0

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bacc = balanced_accuracy_score(y_true, y_pred)

    print(f"  {src_label}: acc={acc:.4f}  F1={f1:.4f}  bal-acc={bacc:.4f}  ({elapsed:.0f}s)")
    print(classification_report(y_true, y_pred, target_names=SHARED_LABELS, zero_division=0))

    np.save(Path(METRICS_DIR) / f"transfer_norm_{direction}_y_true.npy", y_true)
    np.save(Path(METRICS_DIR) / f"transfer_norm_{direction}_y_pred.npy", y_pred)

    return {"direction": src_label, "acc": acc, "f1": f1, "bacc": bacc}


def main():
    print("\n" + "="*68)
    print("  Cross-Dataset Transfer — Instance Normalisation")
    print("  Per-window z-score per channel before graph feature extraction")
    print("="*68)

    # Load baseline zero-shot numbers
    baseline_path = Path(METRICS_DIR) / "cross_dataset_transfer_improved.json"
    baseline = {}
    if baseline_path.exists():
        data = json.loads(baseline_path.read_text())
        for r in data.get("results", []):
            baseline[r["direction"]] = r
        print("\n  Baseline (zero-shot) results loaded from JSON.")
    else:
        print("\n  WARNING: baseline JSON not found — run run_cross_dataset_transfer.py first.")

    norm_results = []
    for direction in ["hhar2pamap2", "pamap2hhar"]:
        r = run_transfer_norm(direction)
        norm_results.append(r)

    # Confusion matrices
    print("\n── Confusion matrices ──────────────────────────────────────────")
    for direction in ["hhar2pamap2", "pamap2hhar"]:
        yt = np.load(Path(METRICS_DIR) / f"transfer_norm_{direction}_y_true.npy")
        yp = np.load(Path(METRICS_DIR) / f"transfer_norm_{direction}_y_pred.npy")
        r  = next(x for x in norm_results if direction.replace("hhar2pamap2", "HHAR → PAMAP2")
                                                        .replace("pamap2hhar",  "PAMAP2 → HHAR") in x["direction"])
        title = f"{r['direction']} — instance normalised (acc={r['acc']:.4f})"
        plot_cm(yt, yp, SHARED_LABELS, title,
                Path(PLOTS_DIR) / f"cm_transfer_norm_{direction}.png")

    # Before/after comparison table
    print("\n── Before vs After ─────────────────────────────────────────────")
    comparison_rows = []
    dir_map = {"HHAR → PAMAP2": "hhar2pamap2", "PAMAP2 → HHAR": "pamap2hhar"}
    for r_norm in norm_results:
        key = r_norm["direction"]
        b   = baseline.get(key, {})
        row = {
            "direction": key,
            "zero_acc":  b.get("acc", float("nan")),
            "zero_f1":   b.get("f1",  float("nan")),
            "norm_acc":  r_norm["acc"],
            "norm_f1":   r_norm["f1"],
        }
        comparison_rows.append(row)
        d_acc = row["norm_acc"] - row["zero_acc"]
        d_f1  = row["norm_f1"]  - row["zero_f1"]
        print(f"  {key:<20}  acc: {row['zero_acc']:.4f} → {row['norm_acc']:.4f} ({d_acc:+.4f})"
              f"  F1: {row['zero_f1']:.4f} → {row['norm_f1']:.4f} ({d_f1:+.4f})")

    plot_comparison_table(comparison_rows,
                          Path(PLOTS_DIR) / "comparison_table_transfer_norm.png")

    # Save JSON
    out = {
        "method": "instance_normalisation",
        "description": "Per-window z-score per channel (axis=1) applied to both source and target before graph feature extraction.",
        "results": norm_results,
        "comparison": comparison_rows,
    }
    out_path = Path(METRICS_DIR) / "transfer_instance_norm_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
