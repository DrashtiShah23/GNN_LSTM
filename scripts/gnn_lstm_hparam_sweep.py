"""
Hyperparameter sweep for GNN+LSTM (PAMAP2 LOSO).

Grid (default): gcn_hidden ∈ {64,128,256}, num_gcn_layers ∈ {1,2,3},
lstm_layers ∈ {1,2}, dropout ∈ {0.1,0.3,0.5} — use --quick for a tiny subset.

Each configuration runs full LOSO via the same training loop as rerun_gnnlstm.

Usage:
  python scripts/gnn_lstm_hparam_sweep.py --quick
  python scripts/gnn_lstm_hparam_sweep.py --dry-run
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.config import (
    PROCESSED_DIR,
    METRICS_DIR,
    MODELS_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    PATIENCE,
    SEED,
    PAMAP2_NODE_FEAT_DIM,
)
from src.models import GNNLSTMModel
from src.dataset import HARSequenceDataset
from src.graph_construction import build_pamap2_adj
from src.train import get_device, set_seed
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

for d in [METRICS_DIR, MODELS_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)


def remap_labels(y):
    classes = np.unique(y)
    mapping = {int(old): int(new) for new, old in enumerate(classes)}
    return np.vectorize(mapping.__getitem__)(y), mapping


def train_one_fold(model, tr_loader, val_loader, device):
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit = nn.CrossEntropyLoss()
    best_acc, best_state, patience_cnt = 0.0, None, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for batch in tr_loader:
            opt.zero_grad()
            x, adj, y = batch
            x, adj, y = x.to(device), adj.to(device), y.to(device)
            crit(model(x, adj), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        val_loss = correct = total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, adj, yb = batch
                x, adj, yb = x.to(device), adj.to(device), yb.to(device)
                logits = model(x, adj)
                val_loss += crit(logits, yb).item() * len(yb)
                correct += (logits.argmax(1) == yb).sum().item()
                total += len(yb)
        if total == 0:
            continue
        val_acc = correct / total
        sched.step(val_loss / total)
        if val_acc > best_acc:
            best_acc, best_state, patience_cnt = val_acc, copy.deepcopy(model.state_dict()), 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def loso_gnn_lstm_config(X, y, subjects, cfg: dict) -> dict:
    device = get_device()
    set_seed(SEED)
    n_cls = len(np.unique(y))
    all_true, all_pred = [], []
    fold_rows = []
    for fold_i, test_subj in enumerate(np.unique(subjects), 1):
        train_mask = subjects != test_subj
        test_mask = subjects == test_subj
        X_tr, y_tr, s_tr = X[train_mask], y[train_mask], subjects[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]

        val_idx_list, tr_idx_list = [], []
        for subj in np.unique(s_tr):
            idx = np.where(s_tr == subj)[0]
            if len(idx) < 20:
                tr_idx_list.append(idx)
                continue
            split = max(10, int(0.8 * len(idx)))
            tr_idx_list.append(idx[:split])
            val_idx_list.append(idx[split:])
        tr_all = np.concatenate(tr_idx_list)
        val_all = np.concatenate(val_idx_list) if val_idx_list else tr_all
        X_tr2, y_tr2, s_tr2 = X_tr[tr_all], y_tr[tr_all], s_tr[tr_all]
        X_val, y_val, s_val = X_tr[val_all], y_tr[val_all], s_tr[val_all]

        ds_tr = HARSequenceDataset(X_tr2, y_tr2, subjects=s_tr2, dataset="pamap2")
        ds_val = HARSequenceDataset(X_val, y_val, subjects=s_val, dataset="pamap2")
        ds_te = HARSequenceDataset(X_te, y_te, dataset="pamap2")
        if len(ds_tr) == 0:
            ds_tr = HARSequenceDataset(X_tr, y_tr, subjects=s_tr, dataset="pamap2")
        if len(ds_val) == 0:
            ds_val = ds_tr

        tr_l = DataLoader(ds_tr, BATCH_SIZE, shuffle=True, num_workers=0)
        va_l = DataLoader(ds_val, BATCH_SIZE, shuffle=False, num_workers=0)
        te_l = DataLoader(ds_te, BATCH_SIZE, shuffle=False, num_workers=0)

        model = GNNLSTMModel(
            PAMAP2_NODE_FEAT_DIM,
            3,
            n_cls,
            gcn_hidden=cfg["gcn_hidden"],
            gcn_output=cfg["gcn_hidden"],
            num_gcn_layers=cfg["num_gcn_layers"],
            lstm_hidden=cfg["lstm_hidden"],
            lstm_layers=cfg["lstm_layers"],
            dropout=cfg["dropout"],
        ).to(device)
        model = train_one_fold(model, tr_l, va_l, device)

        ft, fp = [], []
        with torch.no_grad():
            for batch in te_l:
                x, adj, yb = batch
                x, adj, yb = x.to(device), adj.to(device), yb.to(device)
                ft.extend(yb.cpu().tolist())
                fp.extend(model(x, adj).argmax(1).cpu().tolist())
        acc = accuracy_score(ft, fp)
        fold_rows.append(
            {
                "fold": fold_i,
                "test_subject": int(test_subj) if np.issubdtype(type(test_subj), np.integer) else str(test_subj),
                "accuracy": float(acc),
            }
        )
        all_true.extend(ft)
        all_pred.extend(fp)

    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    bal = balanced_accuracy_score(all_true, all_pred)
    facc = [r["accuracy"] for r in fold_rows]
    return {
        "config": cfg,
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "balanced_acc": float(bal),
        "accuracy_std": float(np.std(facc)) if facc else 0.0,
        "folds": fold_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Run 2 configs × subset of data")
    ap.add_argument("--dry-run", action="store_true", help="Print grid only")
    args = ap.parse_args()

    base = Path(PROCESSED_DIR)
    X = np.load(base / "pamap2_X.npy")
    y, _ = remap_labels(np.load(base / "pamap2_y.npy"))
    subjects = np.load(base / "pamap2_subjects.npy")

    if args.quick:
        X = X[:8000]
        y = y[:8000]
        subjects = subjects[:8000]

    grid = list(
        product(
            [64, 128, 256],
            [1, 2, 3],
            [1, 2],
            [0.1, 0.3, 0.5],
        )
    )
    if args.quick:
        grid = [(64, 2, 1, 0.3), (128, 3, 2, 0.1)]

    if args.dry_run:
        print(f"Would run {len(grid)} configurations (quick={args.quick})")
        return

    results = []
    for gh, nl, ll, dr in grid:
        cfg = {
            "gcn_hidden": gh,
            "num_gcn_layers": nl,
            "lstm_layers": ll,
            "lstm_hidden": max(64, gh),
            "dropout": dr,
        }
        print(f"\n=== Sweep {cfg} ===", flush=True)
        r = loso_gnn_lstm_config(X, y, subjects, cfg)
        results.append(r)
        print(f"  → acc={r['accuracy']:.4f} ± {r['accuracy_std']:.4f}  macro_f1={r['macro_f1']:.4f}", flush=True)

    best = max(results, key=lambda r: r["accuracy"])
    out = {"results": results, "best_by_accuracy": best}
    out_path = Path(METRICS_DIR) / "gnn_lstm_hparam_sweep.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
