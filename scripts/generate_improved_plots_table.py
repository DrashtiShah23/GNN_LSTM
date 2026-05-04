"""
Generate confusion matrices and the full ± std comparison table for
ImprovedGNNLSTMModel results, then update master_comparison.json.

Outputs:
  results/plots/cm_improved_gnnlstm_pamap2.png
  results/plots/cm_improved_gnnlstm_hhar.png
  results/plots/comparison_table_with_std.png
  results/metrics/master_comparison_with_std.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

METRICS = Path("results/metrics")
PLOTS   = Path("results/plots")
PLOTS.mkdir(parents=True, exist_ok=True)

# ── Activity label maps ────────────────────────────────────────────────────────

PAMAP2_LABELS = [
    "lying", "sitting", "standing", "walking",
    "running", "cycling", "nordic_walk",
    "watch_tv", "computer", "ascending",
    "descending", "vacuum",
]

HHAR_LABELS = ["biking", "sitting", "standing", "walking", "stairsup", "stairsdown"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_preds(tag: str):
    yt = np.load(METRICS / f"{tag}_y_true.npy")
    yp = np.load(METRICS / f"{tag}_y_pred.npy")
    return yt, yp


def plot_cm(y_true, y_pred, labels, title, out_path, figsize=(10, 8)):
    n = len(labels)
    # Trim labels to actual number of classes present
    n_classes = max(y_true.max(), y_pred.max()) + 1
    labels = labels[:n_classes]

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.4, linecolor="lightgrey",
        vmin=0, vmax=1, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Build the full ± std table ─────────────────────────────────────────────────

def build_std_table():
    """
    Returns two dicts (pamap2_rows, hhar_rows), each a list of:
      {"model": str, "acc": float, "acc_std": float|None,
       "f1": float, "f1_std": float|None,
       "bacc": float, "bacc_std": float|None}
    """
    pb  = json.loads((METRICS / "pamap2_baselines.json").read_text())
    hb  = json.loads((METRICS / "HHAR_baselines.json").read_text())
    pd_ = json.loads((METRICS / "pamap2_deep_models.json").read_text())
    hd  = json.loads((METRICS / "hhar_deep_models.json").read_text())
    abl = json.loads((METRICS / "graph_ablation_results.json").read_text())
    cn  = json.loads((METRICS / "cnn1d_results.json").read_text())
    imp  = json.loads((METRICS / "improved_gnnlstm_results.json").read_text())
    attn = json.loads((METRICS / "attn_adj_results.json").read_text()) if (METRICS / "attn_adj_results.json").exists() else {}

    def row(model, acc, f1, bacc, acc_std=None, f1_std=None, bacc_std=None):
        return dict(model=model, acc=acc, acc_std=acc_std,
                    f1=f1, f1_std=f1_std, bacc=bacc, bacc_std=bacc_std)

    # ── PAMAP2 ────────────────────────────────────────────────────────────────
    p2 = [
        row("SVM",
            pb["SVM"]["mean_accuracy"], pb["SVM"]["mean_macro_f1"],
            pb["SVM"]["mean_accuracy"],                           # no bacc sep
            pb["SVM"]["std_accuracy"],  pb["SVM"]["std_macro_f1"]),
        row("Random Forest",
            pb["RandomForest"]["mean_accuracy"], pb["RandomForest"]["mean_macro_f1"],
            pb["RandomForest"]["mean_accuracy"],
            pb["RandomForest"]["std_accuracy"],  pb["RandomForest"]["std_macro_f1"]),
        row("XGBoost",
            pb["XGBoost"]["mean_accuracy"], pb["XGBoost"]["mean_macro_f1"],
            pb["XGBoost"]["mean_accuracy"],
            pb["XGBoost"]["std_accuracy"],  pb["XGBoost"]["std_macro_f1"]),
        row("LSTM-only",
            0.5939, 0.5951, 0.5896,
            pd_["lstm"]["accuracy_std"], pd_["lstm"]["macro_f1_std"]),
        # Canonical value from final_table.csv (latest LOSO run)
        row("GNN-only",
            0.7206, 0.7151, 0.7080,
            pd_["gnn"]["accuracy_std"], pd_["gnn"]["macro_f1_std"]),
        # Canonical midway-report value (final_table.csv); std from per-fold data
        row("GNN+LSTM (old)",
            0.6418, 0.5874, 0.6276,
            pd_["gnn_lstm"]["accuracy_std"], pd_["gnn_lstm"]["macro_f1_std"]),
        row("GNN (fixed adj)",
            abl["ablation_fixed_adj"]["accuracy"],
            abl["ablation_fixed_adj"]["macro_f1"],
            abl["ablation_fixed_adj"]["balanced_acc"]),
        row("GNN (learnable adj)",
            abl["ablation_learnable_adj"]["accuracy"],
            abl["ablation_learnable_adj"]["macro_f1"],
            abl["ablation_learnable_adj"]["balanced_acc"]),
        row("Flatten+LSTM",
            abl["ablation_flatten_lstm"]["accuracy"],
            abl["ablation_flatten_lstm"]["macro_f1"],
            abl["ablation_flatten_lstm"]["balanced_acc"]),
        row("CNN1D",
            cn["pamap2"]["accuracy"], cn["pamap2"]["macro_f1"], cn["pamap2"]["balanced_acc"]),
        row("ImprovedGNNLSTM ★",
            imp["pamap2"]["accuracy"],      imp["pamap2"]["macro_f1"],      imp["pamap2"]["balanced_acc"],
            imp["pamap2"]["accuracy_std"],  imp["pamap2"]["macro_f1_std"],  imp["pamap2"]["balanced_acc_std"]),
    ]
    if "pamap2" in attn:
        p2.append(row("ImprovedGNNLSTM+AttnAdj ★",
            attn["pamap2"]["accuracy"],     attn["pamap2"]["macro_f1"],     attn["pamap2"]["balanced_acc"],
            attn["pamap2"]["accuracy_std"], attn["pamap2"]["macro_f1_std"], attn["pamap2"]["balanced_acc_std"]))

    # ── HHAR ──────────────────────────────────────────────────────────────────
    h2 = [
        row("SVM",
            hb["SVM"]["mean_accuracy"], hb["SVM"]["mean_macro_f1"],
            hb["SVM"]["mean_accuracy"],
            hb["SVM"]["std_accuracy"],  hb["SVM"]["std_macro_f1"]),
        row("Random Forest",
            hb["RandomForest"]["mean_accuracy"], hb["RandomForest"]["mean_macro_f1"],
            hb["RandomForest"]["mean_accuracy"],
            hb["RandomForest"]["std_accuracy"],  hb["RandomForest"]["std_macro_f1"]),
        row("XGBoost",
            hb["XGBoost"]["mean_accuracy"], hb["XGBoost"]["mean_macro_f1"],
            hb["XGBoost"]["mean_accuracy"],
            hb["XGBoost"]["std_accuracy"],  hb["XGBoost"]["std_macro_f1"]),
        row("LSTM-only",
            hd["lstm"]["accuracy"], hd["lstm"]["macro_f1"], hd["lstm"]["balanced_acc"],
            hd["lstm"]["accuracy_std"], hd["lstm"]["macro_f1_std"]),
        row("GNN-only",
            hd["gnn"]["accuracy"], hd["gnn"]["macro_f1"], hd["gnn"]["balanced_acc"],
            hd["gnn"]["accuracy_std"], hd["gnn"]["macro_f1_std"]),
        row("CNN1D",
            cn["hhar"]["accuracy"], cn["hhar"]["macro_f1"], cn["hhar"]["balanced_acc"]),
        row("ImprovedGNNLSTM ★",
            imp["hhar"]["accuracy"],      imp["hhar"]["macro_f1"],      imp["hhar"]["balanced_acc"],
            imp["hhar"]["accuracy_std"],  imp["hhar"]["macro_f1_std"],  imp["hhar"]["balanced_acc_std"]),
    ]
    if "hhar" in attn:
        h2.append(row("ImprovedGNNLSTM+AttnAdj",
            attn["hhar"]["accuracy"],     attn["hhar"]["macro_f1"],     attn["hhar"]["balanced_acc"],
            attn["hhar"]["accuracy_std"], attn["hhar"]["macro_f1_std"], attn["hhar"]["balanced_acc_std"]))

    return p2, h2


def fmt(val, std=None):
    if val is None:
        return "—"
    s = f"{val:.4f}"
    if std is not None:
        s += f" ±{std:.4f}"
    return s


def print_table(rows, title):
    print(f"\n{'═'*85}")
    print(f"  {title}")
    print(f"{'═'*85}")
    print(f"  {'Model':<24} {'Accuracy (±std)':>22} {'Macro-F1 (±std)':>22} {'Bal-Acc (±std)':>18}")
    print(f"  {'─'*24} {'─'*22} {'─'*22} {'─'*18}")
    rows_sorted = sorted(rows, key=lambda r: r["acc"])
    for r in rows_sorted:
        mark = " ★" if "★" in r["model"] else ""
        print(f"  {r['model']:<24} {fmt(r['acc'], r['acc_std']):>22} "
              f"{fmt(r['f1'], r['f1_std']):>22} {fmt(r['bacc'], r['bacc_std']):>18}{mark}")
    print(f"{'═'*85}")


def plot_table(rows, title, out_path):
    rows_sorted = sorted(rows, key=lambda r: r["acc"])
    col_labels  = ["Model", "Accuracy ± Std", "Macro-F1 ± Std", "Bal-Acc ± Std"]
    table_data  = []
    row_colors  = []
    for r in rows_sorted:
        is_new = "★" in r["model"]
        table_data.append([
            r["model"],
            fmt(r["acc"], r["acc_std"]),
            fmt(r["f1"],  r["f1_std"]),
            fmt(r["bacc"],r["bacc_std"]),
        ])
        row_colors.append(["#d4edda"] * 4 if is_new else ["#f8f9fa"] * 4)

    fig, ax = plt.subplots(figsize=(13, 0.5 + 0.45 * len(rows_sorted)))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#343a40")
            cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#dee2e6")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def update_master_json(p2_rows, h2_rows):
    out = {}
    for ds_label, rows in [("pamap2", p2_rows), ("hhar", h2_rows)]:
        out[ds_label] = {
            r["model"]: {
                "accuracy":     round(r["acc"], 6),
                "accuracy_std": round(r["acc_std"], 6) if r["acc_std"] else None,
                "macro_f1":     round(r["f1"],  6),
                "macro_f1_std": round(r["f1_std"],  6) if r["f1_std"] else None,
                "balanced_acc": round(r["bacc"], 6),
                "balanced_acc_std": round(r["bacc_std"], 6) if r["bacc_std"] else None,
            }
            for r in rows
        }
    path = METRICS / "master_comparison_with_std.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"  Saved → {path}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n── Confusion matrices ───────────────────────────────────────────")

    yt_p, yp_p = load_preds("improved_gnnlstm_pamap2")
    plot_cm(yt_p, yp_p, PAMAP2_LABELS,
            "ImprovedGNNLSTM — PAMAP2 (LOSO, normalised)",
            PLOTS / "cm_improved_gnnlstm_pamap2.png",
            figsize=(10, 8))

    yt_h, yp_h = load_preds("improved_gnnlstm_hhar")
    plot_cm(yt_h, yp_h, HHAR_LABELS,
            "ImprovedGNNLSTM — HHAR (LOSO, normalised)",
            PLOTS / "cm_improved_gnnlstm_hhar.png",
            figsize=(7, 6))

    print("\n── ± std comparison tables ──────────────────────────────────────")
    p2_rows, h2_rows = build_std_table()

    print_table(p2_rows, "PAMAP2 — LOSO results with ± std (9 folds)")
    print_table(h2_rows, "HHAR   — LOSO results with ± std (9 folds)")

    plot_table(p2_rows,
               "PAMAP2 LOSO — All Models (Accuracy / Macro-F1 / Balanced Acc ± Std, 9 folds)",
               PLOTS / "comparison_table_std_pamap2.png")
    plot_table(h2_rows,
               "HHAR LOSO — All Models (Accuracy / Macro-F1 / Balanced Acc ± Std, 9 folds)",
               PLOTS / "comparison_table_std_hhar.png")

    print("\n── Updating master comparison JSON ──────────────────────────────")
    update_master_json(p2_rows, h2_rows)

    print("\nDone.")


if __name__ == "__main__":
    main()
