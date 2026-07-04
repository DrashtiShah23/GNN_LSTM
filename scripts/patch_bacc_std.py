"""
Patch missing balanced_acc_std for all models in both datasets.

Sources:
  - Deep models (LSTM, GNN, GNN+LSTM): compute per-fold balanced_acc
    from saved y_true/y_pred npy files + subjects array.
  - Classical models (SVM, RF, XGBoost): bacc = accuracy in the stored
    data, so bacc_std = acc_std directly.
  - Ablation / CNN1D: already patched by patch_missing_std.py.

Updates:
  pamap2_deep_models.json, hhar_deep_models.json,
  pamap2_baselines.json, HHAR_baselines.json,
  master_comparison_with_std.json
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
METS = ROOT / "results" / "metrics"
PROC = ROOT / "data" / "processed"

pamap2_subj = np.load(PROC / "pamap2_subjects.npy")
hhar_subj   = np.load(PROC / "hhar_subjects.npy")

# ── Custom subject arrays for capped / sequence-level runs ───────────────────
# HHAR deep models were capped at 5000 windows/subject → 45000 total
hhar_subj_cap = np.repeat(np.unique(hhar_subj), 5000)          # 9 × 5000 = 45000

# Sequence-level PAMAP2 (seq_len=10 stride=10) — same formula as experiments.py
seq_subj_pamap2 = []
for s in np.unique(pamap2_subj):
    n = int((pamap2_subj == s).sum())
    n_seqs = max(0, (n - 10 + 1) // 10)
    seq_subj_pamap2.extend([s] * n_seqs)
seq_subj_pamap2 = np.array(seq_subj_pamap2)                    # ~1502

# Sequence-level HHAR capped (5000/subj → 499 seqs/subj ≈ 4500 total)
seq_subj_hhar = []
for s in np.unique(hhar_subj):
    n_seqs = max(0, (5000 - 10 + 1) // 10)                     # 499
    seq_subj_hhar.extend([s] * n_seqs)
seq_subj_hhar = np.array(seq_subj_hhar)                        # 9 × 499 = 4491

def fold_bacc_std(y_true, y_pred, subjects):
    """Per-fold balanced accuracy std (ddof=1)."""
    vals = []
    offset = 0
    for s in np.unique(subjects):
        n = int((subjects == s).sum())
        yt, yp = y_true[offset:offset+n], y_pred[offset:offset+n]
        offset += n
        if len(yt) > 0:
            vals.append(balanced_accuracy_score(yt, yp))
    if len(vals) < 2:
        return float("nan")
    return float(np.std(vals, ddof=1))

# ── 1. Deep models: compute bacc_std from npy files ──────────────────────────
deep_npy = [
    # (tag,                     dataset,  subjects,         json_path,              key)
    ("lstm_pamap2",             "pamap2", pamap2_subj,      "pamap2_deep_models.json", "lstm"),
    ("lstm_hhar",               "hhar",   hhar_subj_cap,    "hhar_deep_models.json",   "lstm"),
    ("gnn_pamap2",              "pamap2", pamap2_subj,      "pamap2_deep_models.json", "gnn"),
    ("gnn_hhar",                "hhar",   hhar_subj_cap,    "hhar_deep_models.json",   "gnn"),
    ("gnnlstm_pamap2",          "pamap2", seq_subj_pamap2,  "pamap2_deep_models.json", "gnn_lstm"),
    ("improved_gnnlstm_pamap2", "pamap2", seq_subj_pamap2,  "improved_gnnlstm_results.json", None),
    ("improved_gnnlstm_hhar",   "hhar",   seq_subj_hhar,    "improved_gnnlstm_results.json", None),
    ("attn_adj_pamap2",         "pamap2", seq_subj_pamap2,  "attn_adj_results.json",   None),
    ("attn_adj_hhar",           "hhar",   seq_subj_hhar,    "attn_adj_results.json",   None),
]

json_cache = {}

for tag, dataset, subj, json_fname, key in deep_npy:
    yt_path = METS / f"{tag}_y_true.npy"
    yp_path = METS / f"{tag}_y_pred.npy"
    if not yt_path.exists():
        print(f"  SKIP {tag} (no npy)")
        continue
    yt = np.load(yt_path)
    yp = np.load(yp_path)
    std = fold_bacc_std(yt, yp, subj)
    print(f"  {tag}: bacc_std={std:.4f}")

    # load json
    jpath = METS / json_fname
    if json_fname not in json_cache:
        json_cache[json_fname] = json.loads(jpath.read_text())
    d = json_cache[json_fname]

    if key:                        # pamap2_deep_models / hhar_deep_models
        d[key]["balanced_acc_std"] = round(std, 6)
    else:                          # improved_gnnlstm / attn_adj — keyed by dataset
        if dataset in d:
            d[dataset]["balanced_acc_std"] = round(std, 6)

# write updated deep model jsons
for fname, data in json_cache.items():
    (METS / fname).write_text(json.dumps(data, indent=2))
    print(f"  Updated {fname}")

# ── 2. Classical models: bacc_std = acc_std ──────────────────────────────────
for fname, keys in [
    ("pamap2_baselines.json", ["SVM","RandomForest","XGBoost"]),
    ("HHAR_baselines.json",   ["SVM","RandomForest","XGBoost"]),
]:
    path = METS / fname
    if not path.exists():
        print(f"  SKIP {fname}")
        continue
    d = json.loads(path.read_text())
    for k in keys:
        if k in d and "std_accuracy" in d[k]:
            d[k]["balanced_acc_std"] = round(d[k]["std_accuracy"], 6)
            print(f"  {fname}/{k}: bacc_std = acc_std = {d[k]['balanced_acc_std']:.4f}")
    path.write_text(json.dumps(d, indent=2))
    print(f"  Updated {fname}")

print("\nDone. Re-run generate_improved_plots_table.py.")
