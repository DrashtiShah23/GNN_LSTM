"""
1) Paired tests across LOSO folds (requires fold-wise metrics in *_deep_models.json).
2) Confusion rates for activity pairs (sitting/lying, walking vs Nordic walking).

Reads:
  results/metrics/pamap2_deep_models.json (and optionally hhar_deep_models.json)
  results/metrics/cnn1d_results.json (CNN1D has no folds saved historically — skip)

Writes:
  results/metrics/loso_significance.json
  results/metrics/confusion_pairs_pamap2.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import METRICS_DIR, PROCESSED_DIR, PAMAP2_ACTIVITIES


def _pamap_inv_label_names() -> dict[int, str]:
    """Remapped class index → activity name (same ordering as preprocessing)."""
    sorted_keys = sorted(PAMAP2_ACTIVITIES.keys())
    return {i: PAMAP2_ACTIVITIES[k] for i, k in enumerate(sorted_keys)}


def paired_ttest_on_folds(folds_a: list[dict], folds_b: list[dict], key: str = "accuracy"):
    """Match folds by test_subject and run paired t-test on fold metric."""
    by_sub_a = {f["test_subject"]: f[key] for f in folds_a}
    by_sub_b = {f["test_subject"]: f[key] for f in folds_b}
    common = sorted(set(by_sub_a) & set(by_sub_b), key=lambda x: (str(type(x)), str(x)))
    if len(common) < 2:
        return None
    va = np.array([by_sub_a[s] for s in common])
    vb = np.array([by_sub_b[s] for s in common])
    t_stat, p_value = stats.ttest_rel(va, vb)
    return {
        "n_pairs": len(common),
        "mean_a": float(va.mean()),
        "mean_b": float(vb.mean()),
        "std_a": float(va.std(ddof=1)) if len(va) > 1 else 0.0,
        "std_b": float(vb.std(ddof=1)) if len(vb) > 1 else 0.0,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
    }


def confusion_pairs_from_preds(y_true: np.ndarray, y_pred: np.ndarray, names: dict[int, str], model: str):
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    name_list = [names.get(i, str(i)) for i in labels]

    def rate(true_name_sub: str, pred_name_sub: str) -> float | None:
        try:
            ti = next(i for i, n in enumerate(name_list) if true_name_sub.lower() in n.lower())
            pi = next(i for i, n in enumerate(name_list) if pred_name_sub.lower() in n.lower())
            return float(cm[ti, pi])
        except StopIteration:
            return None

    return {
        "model": model,
        "sitting_as_lying": rate("sitting", "lying"),
        "lying_as_sitting": rate("lying", "sitting"),
        "walking_as_nordic": rate("walking", "nordic"),
        "nordic_as_walking": rate("nordic", "walking"),
        "label_order": name_list,
    }


def main():
    met = Path(METRICS_DIR)
    out_sig = {}
    p2_deep_path = met / "pamap2_deep_models.json"
    if p2_deep_path.exists():
        deep = json.loads(p2_deep_path.read_text())
        pairs_models = []
        for k1, k2 in [("gnn", "gnn_lstm"), ("gnn", "lstm"), ("lstm", "gnn_lstm")]:
            a, b = deep.get(k1), deep.get(k2)
            if not a or not b:
                continue
            fa, fb = a.get("folds"), b.get("folds")
            if fa and fb:
                out_sig[f"{k1}_vs_{k2}_accuracy"] = paired_ttest_on_folds(fa, fb, "accuracy")
                out_sig[f"{k1}_vs_{k2}_macro_f1"] = paired_ttest_on_folds(fa, fb, "macro_f1")
            pairs_models.append((k1, k2))
        if out_sig:
            (met / "loso_significance.json").write_text(json.dumps(out_sig, indent=2))
            print(f"Wrote {met / 'loso_significance.json'}")
        else:
            print(
                "No fold-wise entries in pamap2_deep_models.json — re-run "
                "scripts/run_full_pipeline.py to populate 'folds' arrays."
            )

    names = _pamap_inv_label_names()
    pair_summary = {}
    for tag in ["gnn_pamap2", "cnn1d_pamap2", "gnnlstm_pamap2"]:
        yt_path = met / f"{tag}_y_true.npy"
        yp_path = met / f"{tag}_y_pred.npy"
        if not yt_path.exists():
            continue
        yt = np.load(yt_path)
        yp = np.load(yp_path)
        pair_summary[tag] = confusion_pairs_from_preds(yt, yp, names, tag)
    if pair_summary:
        (met / "confusion_pairs_pamap2.json").write_text(json.dumps(pair_summary, indent=2))
        print(f"Wrote {met / 'confusion_pairs_pamap2.json'}")


if __name__ == "__main__":
    main()
