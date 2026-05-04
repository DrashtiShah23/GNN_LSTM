"""
Run LOSO evaluation for ImprovedGNNLSTMModel on PAMAP2 and HHAR.

Architecture fixes vs original GNNLSTMModel:
  1. Concat pooling  — preserves per-node sensor identity (was: mean pool)
  2. LayerNorm       — stable for small LOSO folds (was: BatchNorm)
  3. Temporal attn   — weighted sum over all LSTM steps (was: last step only)
  4. Skip connection — raw features concatenated to GCN output
  5. Bidir LSTM      — 2× capacity, forward + backward context

HHAR graph fix:
  - Each accelerometer axis (x, y, z) is now its own graph node with 6 stats.
  - Previously both nodes received identical features — GCN was a no-op.

Reported metrics per dataset (mean ± std across 9 LOSO folds):
  - Accuracy, Macro F1, Balanced Accuracy

Usage:
    python scripts/run_improved_gnnlstm.py
    python scripts/run_improved_gnnlstm.py --datasets pamap2
    python scripts/run_improved_gnnlstm.py --datasets hhar --max-per-subject 5000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    classification_report,
)
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.config import (
    PROCESSED_DIR, METRICS_DIR, MODELS_DIR,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE, SEED,
    PAMAP2_NODE_FEAT_DIM, HHAR_NODE_FEAT_DIM,
)
from src.dataset import HARSequenceDataset
from src.models import ImprovedGNNLSTMModel
from src.train import get_device, loso_splits, set_seed

for d in [METRICS_DIR, MODELS_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def remap(y: np.ndarray):
    classes = np.unique(y)
    mp = {int(old): int(new) for new, old in enumerate(classes)}
    return np.vectorize(mp.__getitem__)(y), mp


def load(name: str, max_per_subject: int | None = None):
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
        X, y, subj = X[np.concatenate(keep)], y[np.concatenate(keep)], subj[np.concatenate(keep)]
    return X, y, subj


def train_fold(model, tr_loader, val_loader, device):
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
            for x, adj, y in val_loader:
                x, adj, y = x.to(device), adj.to(device), y.to(device)
                val_loss += crit(model(x, adj), y).item() * len(y)
        val_loss /= len(val_loader.dataset)
        sched.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    for x, adj, y in loader:
        x, adj = x.to(device), adj.to(device)
        preds = model(x, adj).argmax(dim=1).cpu().numpy()
        all_pred.extend(preds)
        all_true.extend(y.numpy())
    return np.array(all_true), np.array(all_pred)


# ── LOSO runner ───────────────────────────────────────────────────────────────

def run_loso(name: str, seq_len: int = 10, max_per_subject: int | None = None) -> dict:
    cap_note = f", capped {max_per_subject}/subj" if max_per_subject else ""
    print(f"\n{'='*68}")
    print(f"  ImprovedGNNLSTM — {name.upper()} LOSO (seq_len={seq_len}{cap_note})")
    print(f"{'='*68}")

    X, y, subj = load(name, max_per_subject=max_per_subject)
    n_classes  = int(y.max()) + 1
    # Both datasets now use 3 nodes; feat dims differ
    node_feat  = PAMAP2_NODE_FEAT_DIM if name == "pamap2" else HHAR_NODE_FEAT_DIM
    n_nodes    = 3  # PAMAP2: wrist/chest/ankle; HHAR: x/y/z axes
    device     = get_device()

    dataset = HARSequenceDataset(X, y, subjects=subj, dataset=name,
                                 seq_len=seq_len, cache=True)

    # Build per-sequence subject labels that exactly match HARSequenceDataset indexing
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
        val_idx = tr_idx[-n_val:]
        tr_idx  = tr_idx[:-n_val]

        tr_loader  = DataLoader(Subset(dataset, tr_idx),  batch_size=BATCH_SIZE,
                                shuffle=True, drop_last=False)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE,
                                shuffle=False)
        te_loader  = DataLoader(Subset(dataset, te_idx),  batch_size=BATCH_SIZE,
                                shuffle=False)

        model = ImprovedGNNLSTMModel(
            node_feat_dim=node_feat,
            n_nodes=n_nodes,
            n_classes=n_classes,
        ).to(device)

        t0 = time.time()
        model    = train_fold(model, tr_loader, val_loader, device)
        y_true, y_pred = predict(model, te_loader, device)
        elapsed  = time.time() - t0

        fold_acc  = accuracy_score(y_true, y_pred)
        fold_f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
        fold_bacc = balanced_accuracy_score(y_true, y_pred)

        fold_accs.append(fold_acc)
        fold_f1s.append(fold_f1)
        fold_baccs.append(fold_bacc)
        all_true.extend(y_true)
        all_pred.extend(y_pred)

        print(f"  {fi+1:<6} {str(te_subj):<10} {fold_acc:>10.4f} {fold_f1:>10.4f} "
              f"{fold_bacc:>10.4f} {elapsed:>7.0f}s")

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    acc  = accuracy_score(all_true, all_pred)
    f1   = f1_score(all_true, all_pred, average="macro", zero_division=0)
    bacc = balanced_accuracy_score(all_true, all_pred)

    acc_std  = float(np.std(fold_accs))
    f1_std   = float(np.std(fold_f1s))
    bacc_std = float(np.std(fold_baccs))

    print(f"\n  {'─'*68}")
    print(f"  {'MEAN':<16} {acc:>10.4f} {f1:>10.4f} {bacc:>10.4f}")
    print(f"  {'STD':<16} {acc_std:>10.4f} {f1_std:>10.4f} {bacc_std:>10.4f}")
    print(f"  {'─'*68}")

    print(f"\n  Per-class report (aggregated across all folds):")
    print(classification_report(all_true, all_pred, zero_division=0))

    # Save predictions
    tag = f"improved_gnnlstm_{name}"
    np.save(Path(METRICS_DIR) / f"{tag}_y_true.npy", all_true)
    np.save(Path(METRICS_DIR) / f"{tag}_y_pred.npy", all_pred)

    return {
        "accuracy":      acc,
        "accuracy_std":  acc_std,
        "macro_f1":      f1,
        "macro_f1_std":  f1_std,
        "balanced_acc":  bacc,
        "balanced_acc_std": bacc_std,
        "per_fold": {
            "accuracy":      [round(v, 4) for v in fold_accs],
            "macro_f1":      [round(v, 4) for v in fold_f1s],
            "balanced_acc":  [round(v, 4) for v in fold_baccs],
        },
    }


# ── comparison table ──────────────────────────────────────────────────────────

KNOWN_RESULTS = {
    "pamap2": {
        "SVM":           {"accuracy": 0.7918, "macro_f1": 0.7244, "balanced_acc": 0.7918},
        "RF":            {"accuracy": 0.7749, "macro_f1": 0.7121, "balanced_acc": 0.7749},
        "XGBoost":       {"accuracy": 0.8076, "macro_f1": 0.7314, "balanced_acc": 0.8076},
        "LSTM-only":     {"accuracy": 0.5939, "macro_f1": 0.5951, "balanced_acc": 0.5896},
        "GNN-only":      {"accuracy": 0.7206, "macro_f1": 0.7151, "balanced_acc": 0.7080},
        "GNN+LSTM (old)":{"accuracy": 0.6418, "macro_f1": 0.5874, "balanced_acc": 0.6276},
        "Flatten+LSTM":  {"accuracy": 0.8453, "macro_f1": 0.7935, "balanced_acc": 0.7886},
        "CNN1D":         {"accuracy": 0.7860, "macro_f1": 0.7938, "balanced_acc": 0.7895},
    },
    "hhar": {
        "SVM":           {"accuracy": 0.5812, "macro_f1": 0.5668, "balanced_acc": 0.5812},
        "RF":            {"accuracy": 0.5630, "macro_f1": 0.5481, "balanced_acc": 0.5630},
        "XGBoost":       {"accuracy": 0.5900, "macro_f1": 0.5779, "balanced_acc": 0.5900},
        "GNN-only":      {"accuracy": 0.5895, "macro_f1": 0.5864, "balanced_acc": 0.5877},
        "CNN1D":         {"accuracy": 0.6627, "macro_f1": 0.6585, "balanced_acc": 0.6602},
    },
}


def print_comparison(results: dict[str, dict]):
    for ds, new_res in results.items():
        known = KNOWN_RESULTS.get(ds, {})
        print(f"\n{'═'*72}")
        print(f"  {ds.upper()} — Full LOSO comparison")
        print(f"{'═'*72}")
        print(f"  {'Model':<22} {'Accuracy':>10} {'Macro-F1':>10} {'Bal-Acc':>10}")
        print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*10}")
        rows = list(known.items()) + [("ImprovedGNNLSTM ★", new_res)]
        rows.sort(key=lambda r: r[1]["accuracy"])
        for mname, m in rows:
            mark = " ★" if mname == "ImprovedGNNLSTM ★" else ""
            print(f"  {mname:<22} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} "
                  f"{m['balanced_acc']:>10.4f}{mark}")
        print(f"{'═'*72}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["pamap2", "hhar"],
                        choices=["pamap2", "hhar"])
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--max-per-subject", type=int, default=None,
                        help="Cap windows per subject (use 5000 for HHAR to keep runtime sane)")
    args = parser.parse_args()

    all_results = {}

    # Load any previously saved results so we can merge
    out_path = Path(METRICS_DIR) / "improved_gnnlstm_results.json"
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)

    for ds in args.datasets:
        mps = args.max_per_subject
        # Default cap for HHAR to keep runtime manageable
        if ds == "hhar" and mps is None:
            mps = 5000
        all_results[ds] = run_loso(ds, seq_len=args.seq_len, max_per_subject=mps)
        # Save after each dataset so results aren't lost if the second run fails
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print_comparison(all_results)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
