"""
Patch accuracy_std / macro_f1_std / balanced_acc_std for models that were
saved without per-fold breakdown.

Strategy: the LOSO loop saves concatenated predictions in fold order
(sorted subjects). Re-split using the subjects npy to get per-fold
metrics, then inject std back into master_comparison_with_std.json and
re-run generate_improved_plots_table to regenerate the PNG.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

METS = ROOT / "results" / "metrics"
PROC = ROOT / "data" / "processed"


def fold_std(y_true: np.ndarray, y_pred: np.ndarray, subjects: np.ndarray):
    """Compute per-fold accuracy, macro-F1, balanced-acc given concatenated
    LOSO predictions (fold order = sorted unique subjects)."""
    fold_acc, fold_f1, fold_bacc = [], [], []
    offset = 0
    for subj in np.unique(subjects):
        n = int((subjects == subj).sum())
        yt = y_true[offset: offset + n]
        yp = y_pred[offset: offset + n]
        offset += n
        fold_acc.append(accuracy_score(yt, yp))
        fold_f1.append(f1_score(yt, yp, average="macro", zero_division=0))
        fold_bacc.append(balanced_accuracy_score(yt, yp))
    return (
        float(np.std(fold_acc,  ddof=1)),
        float(np.std(fold_f1,   ddof=1)),
        float(np.std(fold_bacc, ddof=1)),
    )


# ── Load subject arrays ───────────────────────────────────────────────────────
pamap2_subj = np.load(PROC / "pamap2_subjects.npy")
hhar_subj   = np.load(PROC / "hhar_subjects.npy")

# ablation_flatten_lstm uses sequence-level subjects (seq_len=10, stride=10)
# matching the HARSequenceDataset + seq_subj logic in experiments.py
seq_subj = []
for s in np.unique(pamap2_subj):
    mask = np.where(pamap2_subj == s)[0]
    n_seqs = max(0, (len(mask) - 10 + 1) // 10)
    seq_subj.extend([s] * n_seqs)
seq_subj = np.array(seq_subj)

# ── Models to patch ──────────────────────────────────────────────────────────
patches = [
    # (tag, dataset, subjects_array)
    ("cnn1d_pamap2",          "pamap2", pamap2_subj),
    ("cnn1d_hhar",            "hhar",   hhar_subj),
    ("ablation_fixed_adj",    "pamap2", pamap2_subj),
    ("ablation_learnable_adj","pamap2", pamap2_subj),
    ("ablation_flatten_lstm", "pamap2", seq_subj),
]

std_map = {}
for tag, dataset, subj in patches:
    yt = np.load(METS / f"{tag}_y_true.npy")
    yp = np.load(METS / f"{tag}_y_pred.npy")
    acc_std, f1_std, bacc_std = fold_std(yt, yp, subj)
    std_map[tag] = (acc_std, f1_std, bacc_std)
    print(f"  {tag}: acc_std={acc_std:.4f}  f1_std={f1_std:.4f}  bacc_std={bacc_std:.4f}")

# ── Patch master_comparison_with_std.json ────────────────────────────────────
mc_path = METS / "master_comparison_with_std.json"
mc = json.loads(mc_path.read_text())

tag_to_model = {
    "cnn1d_pamap2":           ("pamap2", "CNN1D"),
    "cnn1d_hhar":             ("hhar",   "CNN1D"),
    "ablation_fixed_adj":     ("pamap2", "GNN (fixed adj)"),
    "ablation_learnable_adj": ("pamap2", "GNN (learnable adj)"),
    "ablation_flatten_lstm":  ("pamap2", "Flatten+LSTM"),
}

for tag, (acc_std, f1_std, bacc_std) in std_map.items():
    dataset, model_name = tag_to_model[tag]
    if model_name in mc.get(dataset, {}):
        mc[dataset][model_name]["accuracy_std"]    = round(acc_std,  6)
        mc[dataset][model_name]["macro_f1_std"]    = round(f1_std,   6)
        mc[dataset][model_name]["balanced_acc_std"] = round(bacc_std, 6)
        print(f"  Patched {dataset}/{model_name}")
    else:
        print(f"  WARNING: {dataset}/{model_name} not found in master JSON")

mc_path.write_text(json.dumps(mc, indent=2))
print(f"\nUpdated {mc_path}")

# ── Also patch the source JSONs so re-runs don't lose this ───────────────────
# cnn1d_results.json
cnn_path = METS / "cnn1d_results.json"
cnn = json.loads(cnn_path.read_text())
for ds, tag in [("pamap2", "cnn1d_pamap2"), ("hhar", "cnn1d_hhar")]:
    if ds in cnn:
        acc_std, f1_std, bacc_std = std_map[tag]
        cnn[ds]["accuracy_std"]    = round(acc_std,  6)
        cnn[ds]["macro_f1_std"]    = round(f1_std,   6)
        cnn[ds]["balanced_acc_std"] = round(bacc_std, 6)
cnn_path.write_text(json.dumps(cnn, indent=2))
print(f"Updated {cnn_path}")

# graph_ablation_results.json
abl_path = METS / "graph_ablation_results.json"
abl = json.loads(abl_path.read_text())
for key, tag in [("ablation_fixed_adj", "ablation_fixed_adj"),
                  ("ablation_learnable_adj", "ablation_learnable_adj"),
                  ("ablation_flatten_lstm", "ablation_flatten_lstm")]:
    if key in abl:
        acc_std, f1_std, bacc_std = std_map[tag]
        abl[key]["accuracy_std"]    = round(acc_std,  6)
        abl[key]["macro_f1_std"]    = round(f1_std,   6)
        abl[key]["balanced_acc_std"] = round(bacc_std, 6)
abl_path.write_text(json.dumps(abl, indent=2))
print(f"Updated {abl_path}")

print("\nDone. Re-run generate_improved_plots_table.py to regenerate PNGs.")
