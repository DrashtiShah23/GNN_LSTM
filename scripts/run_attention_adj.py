"""
LOSO evaluation: ImprovedGNNLSTMAttnAdj vs ImprovedGNNLSTM (fixed adj).

ImprovedGNNLSTMAttnAdj replaces the fixed anatomical adjacency with a
learnable attention gate: A_eff = row_norm(A_fixed ⊙ sigmoid(G+Gᵀ)/2)
All other architecture choices are identical (concat pool, LayerNorm,
skip connection, bidir LSTM, temporal attention).

Outputs (per dataset):
  results/metrics/attn_adj_results.json        — all metrics + per-fold
  results/plots/cm_attn_adj_{dataset}.png      — confusion matrix
  results/plots/comparison_table_attn_adj.png  — ± std table vs fixed
"""

from __future__ import annotations

import argparse, json, sys, time, warnings
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
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.config import (
    PROCESSED_DIR, METRICS_DIR, PLOTS_DIR,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE, SEED,
    PAMAP2_NODE_FEAT_DIM, HHAR_NODE_FEAT_DIM,
)
from src.dataset import HARSequenceDataset
from src.graph_construction import build_pamap2_adj, build_hhar_adj
from src.models import ImprovedGNNLSTMAttnAdj
from src.train import get_device, loso_splits, set_seed

for d in [METRICS_DIR, PLOTS_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def remap(y):
    classes = np.unique(y)
    mp = {int(old): int(new) for new, old in enumerate(classes)}
    return np.vectorize(mp.__getitem__)(y), mp


def load(name, max_per_subject=None):
    p = Path(PROCESSED_DIR)
    X    = np.load(p / f"{name}_X.npy")
    y, _ = remap(np.load(p / f"{name}_y.npy"))
    subj = np.load(p / f"{name}_subjects.npy")
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
    return X, y, subj


def train_fold(model, tr_loader, va_loader, device):
    opt   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit  = nn.CrossEntropyLoss()
    best_loss, best_state, pat = float("inf"), None, 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for x, adj, y in tr_loader:
            x, adj, y = x.to(device), adj.to(device), y.to(device)
            opt.zero_grad()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            crit(model(x, adj), y).backward()
            opt.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for x, adj, y in va_loader:
                x, adj, y = x.to(device), adj.to(device), y.to(device)
                vl += crit(model(x, adj), y).item() * len(y)
        vl /= len(va_loader.dataset)
        sched.step(vl)

        if vl < best_loss:
            best_loss, best_state, pat = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    yt, yp = [], []
    for x, adj, y in loader:
        x, adj = x.to(device), adj.to(device)
        yp.extend(model(x, adj).argmax(1).cpu().numpy())
        yt.extend(y.numpy())
    return np.array(yt), np.array(yp)


# ── LOSO runner ───────────────────────────────────────────────────────────────

def run_loso(name, seq_len=10, max_per_subject=None):
    cap_note = f", capped {max_per_subject}/subj" if max_per_subject else ""
    print(f"\n{'='*68}")
    print(f"  ImprovedGNNLSTM [AttnAdj] — {name.upper()} LOSO (seq_len={seq_len}{cap_note})")
    print(f"{'='*68}")

    X, y, subj = load(name, max_per_subject)
    n_classes  = int(y.max()) + 1
    node_feat  = PAMAP2_NODE_FEAT_DIM if name == "pamap2" else HHAR_NODE_FEAT_DIM
    n_nodes    = 3
    init_adj   = build_pamap2_adj() if name == "pamap2" else build_hhar_adj()
    device     = get_device()

    dataset = HARSequenceDataset(X, y, subjects=subj, dataset=name,
                                 seq_len=seq_len, cache=True)
    seq_subjects = []
    for s in np.unique(subj):
        mask   = np.where(subj == s)[0]
        n_seqs = len(range(0, len(mask) - seq_len + 1, seq_len))
        seq_subjects.extend([s] * n_seqs)
    seq_subjects = np.array(seq_subjects)

    all_true, all_pred = [], []
    fold_accs, fold_f1s, fold_baccs = [], [], []

    print(f"\n  {'Fold':<6} {'Subject':<10} {'Accuracy':>10} {'Macro-F1':>10} {'Bal-Acc':>10} {'Time':>8}")
    print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")

    for fi, (tr_idx, te_idx, te_subj) in enumerate(loso_splits(seq_subjects)):
        set_seed(SEED)
        n_val   = max(1, int(len(tr_idx) * 0.15))
        val_idx = tr_idx[-n_val:]; tr_idx = tr_idx[:-n_val]

        tr_loader = DataLoader(Subset(dataset, tr_idx),  batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
        te_loader = DataLoader(Subset(dataset, te_idx),  batch_size=BATCH_SIZE, shuffle=False)

        model = ImprovedGNNLSTMAttnAdj(
            node_feat_dim=node_feat, n_nodes=n_nodes, n_classes=n_classes,
            init_adj=init_adj,
        ).to(device)

        t0 = time.time()
        model  = train_fold(model, tr_loader, va_loader, device)
        yt, yp = predict(model, te_loader, device)

        fa  = accuracy_score(yt, yp)
        ff  = f1_score(yt, yp, average="macro", zero_division=0)
        fb  = balanced_accuracy_score(yt, yp)
        fold_accs.append(fa); fold_f1s.append(ff); fold_baccs.append(fb)
        all_true.extend(yt); all_pred.extend(yp)
        print(f"  {fi+1:<6} {str(te_subj):<10} {fa:>10.4f} {ff:>10.4f} {fb:>10.4f} {time.time()-t0:>7.0f}s")

    all_true = np.array(all_true); all_pred = np.array(all_pred)
    acc  = accuracy_score(all_true, all_pred)
    f1   = f1_score(all_true, all_pred, average="macro", zero_division=0)
    bacc = balanced_accuracy_score(all_true, all_pred)
    acc_std  = float(np.std(fold_accs))
    f1_std   = float(np.std(fold_f1s))
    bacc_std = float(np.std(fold_baccs))

    print(f"\n  {'─'*68}")
    print(f"  {'MEAN':<16} {acc:>10.4f} {f1:>10.4f} {bacc:>10.4f}")
    print(f"  {'STD':<16} {acc_std:>10.4f} {f1_std:>10.4f} {bacc_std:>10.4f}")
    print(f"\n  Per-class report:")
    print(classification_report(all_true, all_pred, zero_division=0))

    tag = f"attn_adj_{name}"
    np.save(Path(METRICS_DIR) / f"{tag}_y_true.npy", all_true)
    np.save(Path(METRICS_DIR) / f"{tag}_y_pred.npy", all_pred)

    return {"accuracy": acc, "accuracy_std": acc_std,
            "macro_f1": f1, "macro_f1_std": f1_std,
            "balanced_acc": bacc, "balanced_acc_std": bacc_std,
            "per_fold": {"accuracy": [round(v,4) for v in fold_accs],
                         "macro_f1": [round(v,4) for v in fold_f1s],
                         "balanced_acc": [round(v,4) for v in fold_baccs]}}


# ── Plots and table ───────────────────────────────────────────────────────────

PAMAP2_LABELS = ["lying","sitting","standing","walking","running",
                 "cycling","car_drv","comp_wk","asc_strs","ironing","fold_lndry","soccer"]
HHAR_LABELS   = ["biking","sitting","standing","stairsdown","stairsup","walking"]


def plot_cm(y_true, y_pred, labels, title, out_path):
    n = max(y_true.max(), y_pred.max()) + 1
    labels = labels[:n]
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n-1)))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.4, linecolor="lightgrey", vmin=0, vmax=1, ax=ax)
    ax.set_xlabel("Predicted", fontsize=10); ax.set_ylabel("True", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.xticks(rotation=35, ha="right", fontsize=8); plt.yticks(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def fmt(v, s=None):
    if v is None: return "—"
    return f"{v:.4f}" + (f" ±{s:.4f}" if s else "")


def plot_comparison_table(attn_res, fixed_res, out_path):
    rows = []
    for ds in ["pamap2", "hhar"]:
        a = attn_res.get(ds, {}); f = fixed_res.get(ds, {})
        rows.append([f"PAMAP2" if ds=="pamap2" else "HHAR",
                     fmt(f.get("accuracy"), f.get("accuracy_std")),
                     fmt(f.get("macro_f1"), f.get("macro_f1_std")),
                     fmt(a.get("accuracy"), a.get("accuracy_std")),
                     fmt(a.get("macro_f1"), a.get("macro_f1_std"))])

    col_labels = ["Dataset", "Fixed Adj Acc ±std", "Fixed Adj F1 ±std",
                  "Attn Adj Acc ±std", "Attn Adj F1 ±std"]
    row_colors = [["#f0f4ff"] * 5 for _ in rows]
    fig, ax = plt.subplots(figsize=(13, 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc="center", loc="center", cellColours=row_colors)
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#343a40")
            cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#dee2e6")
    ax.set_title("Fixed Adjacency vs Attention Adjacency — ImprovedGNNLSTM (LOSO ±std)",
                 fontsize=10, fontweight="bold", pad=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["pamap2","hhar"])
    parser.add_argument("--seq-len", type=int, default=10)
    args = parser.parse_args()

    attn_results = {}
    out_path = Path(METRICS_DIR) / "attn_adj_results.json"
    if out_path.exists():
        attn_results = json.loads(out_path.read_text())

    for ds in args.datasets:
        mps = 5000 if ds == "hhar" else None
        attn_results[ds] = run_loso(ds, seq_len=args.seq_len, max_per_subject=mps)
        out_path.write_text(json.dumps(attn_results, indent=2))

    # Confusion matrices
    print("\n── Confusion matrices ──────────────────────────────────────────")
    for ds in args.datasets:
        yt = np.load(Path(METRICS_DIR) / f"attn_adj_{ds}_y_true.npy")
        yp = np.load(Path(METRICS_DIR) / f"attn_adj_{ds}_y_pred.npy")
        labels = PAMAP2_LABELS if ds == "pamap2" else HHAR_LABELS
        plot_cm(yt, yp, labels,
                f"ImprovedGNNLSTM [AttnAdj] — {ds.upper()} (LOSO, normalised)",
                Path(PLOTS_DIR) / f"cm_attn_adj_{ds}.png")

    # Comparison table: attn adj vs fixed adj
    fixed_path = Path(METRICS_DIR) / "improved_gnnlstm_results.json"
    fixed_res  = json.loads(fixed_path.read_text()) if fixed_path.exists() else {}

    print("\n── Comparison: Fixed vs Attention Adjacency ────────────────────")
    for ds in ["pamap2", "hhar"]:
        a = attn_results.get(ds, {}); f = fixed_res.get(ds, {})
        if not a or not f: continue
        delta_acc = a["accuracy"] - f["accuracy"]
        delta_f1  = a["macro_f1"] - f["macro_f1"]
        print(f"  {ds.upper():8s}  Fixed: acc={f['accuracy']:.4f} f1={f['macro_f1']:.4f} "
              f"| AttnAdj: acc={a['accuracy']:.4f} f1={a['macro_f1']:.4f} "
              f"| Δacc={delta_acc:+.4f}  Δf1={delta_f1:+.4f}")

    plot_comparison_table(attn_results, fixed_res,
                          Path(PLOTS_DIR) / "comparison_table_attn_adj.png")
    print(f"\n  All results saved → {out_path}")


if __name__ == "__main__":
    main()
