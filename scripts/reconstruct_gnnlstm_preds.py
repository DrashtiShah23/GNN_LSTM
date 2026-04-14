"""
Reconstruct full GNN+LSTM PAMAP2 predictions from the 9 saved fold models.
No retraining — inference only. Takes ~2 minutes.
"""
from __future__ import annotations
import sys, json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (
    PROCESSED_DIR, MODELS_DIR, METRICS_DIR, BATCH_SIZE, SEED,
    PAMAP2_NODE_FEAT_DIM,
)
from src.models import GNNLSTMModel
from src.dataset import HARSequenceDataset
from src.train import get_device, set_seed

set_seed(SEED)
device = get_device()
print(f"Device: {device}")

# ── Load data ──────────────────────────────────────────────────────────────
X = np.load(Path(PROCESSED_DIR) / "pamap2_X.npy")
y = np.load(Path(PROCESSED_DIR) / "pamap2_y.npy")
s = np.load(Path(PROCESSED_DIR) / "pamap2_subjects.npy")

# Remap labels to 0-indexed (same as training)
classes = np.unique(y)
mapping = {int(old): int(new) for new, old in enumerate(classes)}
y = np.vectorize(mapping.__getitem__)(y)
n_classes = len(classes)
n_nodes   = 3
print(f"PAMAP2: {len(X)} windows, {n_classes} classes, subjects={sorted(set(s))}")

# ── Inference per fold ─────────────────────────────────────────────────────
unique_subjs = np.unique(s)
all_true, all_pred = [], []

for fold_i, test_subj in enumerate(unique_subjs, 1):
    model_path = Path(MODELS_DIR) / f"gnnlstm_pamap2_fold{fold_i}.pt"
    if not model_path.exists():
        print(f"  [SKIP] fold {fold_i}: model file not found")
        continue

    test_mask = s == test_subj
    X_te, y_te = X[test_mask], y[test_mask]
    print(f"\nFold {fold_i}/9: test_subj={test_subj}  ({len(X_te)} windows)", flush=True)

    ds_te = HARSequenceDataset(X_te, y_te, dataset="pamap2")
    if len(ds_te) == 0:
        print(f"  [SKIP] no sequences for subject {test_subj}")
        continue
    te_loader = DataLoader(ds_te, BATCH_SIZE, shuffle=False, num_workers=0)

    m = GNNLSTMModel(PAMAP2_NODE_FEAT_DIM, n_nodes, n_classes).to(device)
    m.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    m.eval()

    ft, fp = [], []
    with torch.no_grad():
        for batch in te_loader:
            x, adj, yb = batch
            x, adj, yb = x.to(device), adj.to(device), yb.to(device)
            preds = m(x, adj).argmax(1)
            fp.extend(preds.cpu().tolist())
            ft.extend(yb.cpu().tolist())

    fold_acc = accuracy_score(ft, fp)
    fold_f1  = f1_score(ft, fp, average="macro", zero_division=0)
    print(f"  Fold acc={fold_acc:.4f}  f1={fold_f1:.4f}  n={len(ft)}", flush=True)
    all_true.extend(ft)
    all_pred.extend(fp)

# ── Aggregate ──────────────────────────────────────────────────────────────
all_true = np.array(all_true)
all_pred = np.array(all_pred)

acc = accuracy_score(all_true, all_pred)
f1  = f1_score(all_true, all_pred, average="macro", zero_division=0)
bal = balanced_accuracy_score(all_true, all_pred)

print(f"\n{'='*55}")
print(f"GNN+LSTM PAMAP2 FULL LOSO ({len(all_true)} predictions)")
print(f"  Accuracy     : {acc:.4f}")
print(f"  Macro F1     : {f1:.4f}")
print(f"  Balanced Acc : {bal:.4f}")
print(f"{'='*55}")

# ── Save ───────────────────────────────────────────────────────────────────
np.save(Path(METRICS_DIR) / "gnnlstm_pamap2_y_true.npy", all_true)
np.save(Path(METRICS_DIR) / "gnnlstm_pamap2_y_pred.npy", all_pred)

# Update pamap2_deep_models.json
deep_path = Path(METRICS_DIR) / "pamap2_deep_models.json"
deep = json.load(open(deep_path))
deep["gnn_lstm"] = {"accuracy": acc, "macro_f1": f1, "balanced_acc": bal}
json.dump(deep, open(deep_path, "w"), indent=2)
print("Updated pamap2_deep_models.json")

# Update master_comparison.json
master_path = Path(METRICS_DIR) / "master_comparison.json"
master = json.load(open(master_path))
master.setdefault("PAMAP2", {})["GNN+LSTM"] = {"accuracy": acc, "macro_f1": f1, "balanced_acc": bal}
json.dump(master, open(master_path, "w"), indent=2)
print("Updated master_comparison.json")

print("\nDone — full predictions saved.")
