"""
Generate final report plots and statistics:
  2. Per-class F1 plots — updated to include ImprovedGNNLSTM + AttnAdj
  3. Paired t-tests — ImprovedGNNLSTM vs old GNN+LSTM, AttnAdj vs fixed
  4. ROC / AUC curves — multi-class one-vs-rest for key models

Outputs:
  results/plots/per_class_f1_pamap2.png          (updated)
  results/plots/per_class_f1_hhar.png            (updated)
  results/plots/roc_auc_pamap2.png
  results/plots/roc_auc_hhar.png
  results/metrics/significance_extended.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

METRICS = ROOT / "results/metrics"
PLOTS   = ROOT / "results/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

PAMAP2_LABELS = ["lying","sitting","standing","walking","running",
                 "cycling","nordic_walk","watch_tv","computer",
                 "ascending","descending","vacuum"]
HHAR_LABELS   = ["biking","sitting","standing","stairsdown","stairsup","walking"]


# ── helpers ───────────────────────────────────────────────────────────────────

def load(tag):
    yt = np.load(METRICS / f"{tag}_y_true.npy")
    yp = np.load(METRICS / f"{tag}_y_pred.npy")
    return yt, yp


def n_classes_from(yt, yp):
    return int(max(yt.max(), yp.max())) + 1


# ── 2. Per-class F1 plots ─────────────────────────────────────────────────────

def per_class_f1_data(ds):
    if ds == "pamap2":
        tags = [
            ("GNN+LSTM (old)",          "gnnlstm_pamap2"),
            ("GNN-only",                "gnn_pamap2"),
            ("CNN1D",                   "cnn1d_pamap2"),
            ("ImprovedGNNLSTM",         "improved_gnnlstm_pamap2"),
            ("ImprovedGNNLSTM+AttnAdj", "attn_adj_pamap2"),
        ]
        labels = PAMAP2_LABELS
    else:
        tags = [
            ("GNN+LSTM (old)",          "gnnlstm_hhar"),
            ("GNN-only",                "gnn_hhar"),
            ("CNN1D",                   "cnn1d_hhar"),
            ("ImprovedGNNLSTM",         "improved_gnnlstm_hhar"),
            ("ImprovedGNNLSTM+AttnAdj", "attn_adj_hhar"),
        ]
        labels = HHAR_LABELS

    results = {}
    for label, tag in tags:
        p = METRICS / f"{tag}_y_true.npy"
        if not p.exists():
            continue
        yt, yp = load(tag)
        nc = n_classes_from(yt, yp)
        f1 = f1_score(yt, yp, average=None, zero_division=0, labels=list(range(nc)))
        results[label] = f1
    return results, labels


def plot_per_class_f1(ds):
    data, labels = per_class_f1_data(ds)
    if not data:
        print(f"  No data for {ds}"); return

    nc = max(len(v) for v in data.values())
    labels = labels[:nc]

    x = np.arange(nc)
    n_models = len(data)
    width = 0.75 / n_models
    offsets = np.linspace(-(n_models-1)/2, (n_models-1)/2, n_models) * width

    palette = sns.color_palette("tab10", n_models)
    fig, ax = plt.subplots(figsize=(max(10, nc * 1.3), 5))

    for i, (model, f1s) in enumerate(data.items()):
        f1s_padded = np.zeros(nc)
        f1s_padded[:len(f1s)] = f1s
        ax.bar(x + offsets[i], f1s_padded, width, label=model,
               color=palette[i], alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score (per class)", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.axhline(0.8, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.set_title(f"Per-Class F1 — {ds.upper()} (LOSO, all models)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS / f"per_class_f1_{ds}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ── 3. Paired t-tests ─────────────────────────────────────────────────────────

def paired_ttest(a, b, label_a, label_b):
    a, b = np.array(a), np.array(b)
    t, p = stats.ttest_rel(a, b)
    return {
        "comparison": f"{label_a} vs {label_b}",
        "n_folds": len(a),
        f"mean_{label_a.replace(' ', '_').replace('+', '')}": float(a.mean()),
        f"mean_{label_b.replace(' ', '_').replace('+', '')}": float(b.mean()),
        "delta_mean": float(a.mean() - b.mean()),
        "t_statistic": float(t),
        "p_value": float(p),
        "significant_p05": bool(p < 0.05),
    }


def run_significance():
    imp  = json.loads((METRICS / "improved_gnnlstm_results.json").read_text())
    p2d  = json.loads((METRICS / "pamap2_deep_models.json").read_text())
    attn = json.loads((METRICS / "attn_adj_results.json").read_text())
    old_sig = json.loads((METRICS / "loso_significance.json").read_text())

    results = {}

    # PAMAP2: ImprovedGNNLSTM vs old GNN+LSTM
    imp_p2 = imp["pamap2"]["per_fold"]["accuracy"]
    old_p2 = [f["accuracy"] for f in p2d["gnn_lstm"]["folds"]]
    results["pamap2_ImprovedGNNLSTM_vs_GNNLSTMold"] = paired_ttest(
        imp_p2, old_p2, "ImprovedGNNLSTM", "GNN+LSTM(old)")

    # PAMAP2: AttnAdj vs ImprovedGNNLSTM
    attn_p2 = attn["pamap2"]["per_fold"]["accuracy"]
    results["pamap2_AttnAdj_vs_ImprovedGNNLSTM"] = paired_ttest(
        attn_p2, imp_p2, "AttnAdj", "ImprovedGNNLSTM")

    # HHAR: AttnAdj vs ImprovedGNNLSTM
    imp_h  = imp["hhar"]["per_fold"]["accuracy"]
    attn_h = attn["hhar"]["per_fold"]["accuracy"]
    results["hhar_AttnAdj_vs_ImprovedGNNLSTM"] = paired_ttest(
        attn_h, imp_h, "AttnAdj", "ImprovedGNNLSTM")

    # Carry over legacy significance results
    for k, v in old_sig.items():
        results[f"legacy_{k}"] = v

    out = METRICS / "significance_extended.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"  Saved → {out}")

    print(f"\n  {'Comparison':<50} {'Δmean':>8} {'p-value':>10} {'sig?':>6}")
    print(f"  {'─'*50} {'─'*8} {'─'*10} {'─'*6}")
    for k, v in results.items():
        if "legacy" in k:
            continue
        delta = v["delta_mean"]
        p     = v["p_value"]
        sig   = "YES *" if v["significant_p05"] else "no"
        print(f"  {v['comparison']:<50} {delta:>+8.4f} {p:>10.4f} {sig:>6}")


# ── 4. ROC / AUC curves ────────────────────────────────────────────────────────

def plot_roc(ds):
    if ds == "pamap2":
        model_tags = [
            ("GNN-only",                "gnn_pamap2"),
            ("GNN+LSTM (old)",          "gnnlstm_pamap2"),
            ("CNN1D",                   "cnn1d_pamap2"),
            ("ImprovedGNNLSTM",         "improved_gnnlstm_pamap2"),
            ("ImprovedGNNLSTM+AttnAdj", "attn_adj_pamap2"),
        ]
    else:
        model_tags = [
            ("GNN-only",                "gnn_hhar"),
            ("GNN+LSTM (old)",          "gnnlstm_hhar"),
            ("CNN1D",                   "cnn1d_hhar"),
            ("ImprovedGNNLSTM",         "improved_gnnlstm_hhar"),
            ("ImprovedGNNLSTM+AttnAdj", "attn_adj_hhar"),
        ]

    palette = sns.color_palette("tab10", len(model_tags))
    fig, ax = plt.subplots(figsize=(8, 7))

    roc_aucs = {}
    for (label, tag), color in zip(model_tags, palette):
        p = METRICS / f"{tag}_y_true.npy"
        if not p.exists():
            continue
        yt, yp = load(tag)
        nc = n_classes_from(yt, yp)
        classes = list(range(nc))

        yt_bin = label_binarize(yt, classes=classes)
        yp_bin = label_binarize(yp, classes=classes)

        # Per-class AUC, then macro average
        fpr_list, tpr_list, aucs = [], [], []
        for c in range(nc):
            if yt_bin[:, c].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(yt_bin[:, c], yp_bin[:, c])
            aucs.append(auc(fpr, tpr))
            fpr_list.append(fpr)
            tpr_list.append(tpr)

        mean_fpr = np.linspace(0, 1, 300)
        mean_tpr = np.mean([np.interp(mean_fpr, f, t) for f, t in zip(fpr_list, tpr_list)], axis=0)
        macro_auc = float(np.mean(aucs))
        roc_aucs[label] = macro_auc

        ax.plot(mean_fpr, mean_tpr, lw=2.2, color=color,
                label=f"{label}  (AUC = {macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random  (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"Macro-Average ROC (one-vs-rest) — {ds.upper()} LOSO",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = PLOTS / f"roc_auc_{ds}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")

    for m, a_val in sorted(roc_aucs.items(), key=lambda x: x[1]):
        print(f"    {m:<35} AUC = {a_val:.4f}")

    return roc_aucs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n── 2. Per-class F1 plots ────────────────────────────────────────")
    for ds in ["pamap2", "hhar"]:
        plot_per_class_f1(ds)

    print("\n── 3. Paired t-tests ────────────────────────────────────────────")
    run_significance()

    print("\n── 4. ROC / AUC curves ──────────────────────────────────────────")
    for ds in ["pamap2", "hhar"]:
        print(f"  {ds.upper()}")
        plot_roc(ds)

    print("\nDone.")


if __name__ == "__main__":
    main()
