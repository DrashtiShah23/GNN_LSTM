#!/usr/bin/env python3
"""
Single high-resolution figure: all models on PAMAP2 & HHAR with
LOSO accuracy & macro F1 (± fold std where available).

Also writes presentation_summary_table.csv for copy-paste into slides.

Usage:
    .venv/bin/python scripts/generate_presentation_summary_figure.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MET = ROOT / "results" / "metrics"
PLO = ROOT / "results" / "plots"
MET.mkdir(parents=True, exist_ok=True)
PLO.mkdir(parents=True, exist_ok=True)


def load(name: str) -> dict:
    p = MET / name
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def baseline_rows(b: dict, prefix: str) -> list[dict]:
    out = []
    mapping = [
        ("SVM", "SVM"),
        ("Random Forest", "RandomForest"),
        ("XGBoost", "XGBoost"),
    ]
    for label, key in mapping:
        if key not in b:
            continue
        v = b[key]
        out.append(
            {
                "model": label,
                "acc": float(v.get("mean_accuracy", 0)),
                "acc_std": float(v.get("std_accuracy", 0)),
                "f1": float(v.get("mean_macro_f1", 0)),
                "f1_std": float(v.get("std_macro_f1", 0)),
                "bal": float(v["balanced_acc"]) if "balanced_acc" in v else None,
                "group": "classical",
            }
        )
    return out


def deep_rows(d: dict, prefix: str) -> list[dict]:
    out = []
    order = [
        ("LSTM-only", "lstm"),
        ("GNN-only", "gnn"),
        ("GNN+LSTM (proposed)", "gnn_lstm"),
    ]
    for label, key in order:
        if key not in d:
            continue
        v = d[key]
        out.append(
            {
                "model": label,
                "acc": float(v.get("accuracy", 0)),
                "acc_std": float(v.get("accuracy_std", 0)),
                "f1": float(v.get("macro_f1", 0)),
                "f1_std": float(v.get("macro_f1_std", 0)),
                "bal": float(v.get("balanced_acc", 0)) if v.get("balanced_acc") is not None else None,
                "group": "proposed" if "proposed" in label else "deep",
            }
        )
    return out


def pamap2_extra_ablation() -> list[dict]:
    ab = load("graph_ablation_results.json")
    if not ab:
        return []
    rows = []
    if "ablation_learnable_adj" in ab:
        v = ab["ablation_learnable_adj"]
        rows.append(
            {
                "model": "GNN (learnable adj.)",
                "acc": float(v["accuracy"]),
                "acc_std": 0.0,
                "f1": float(v["macro_f1"]),
                "f1_std": 0.0,
                "bal": float(v.get("balanced_acc", 0)),
                "group": "ablation",
            }
        )
    if "ablation_flatten_lstm" in ab:
        v = ab["ablation_flatten_lstm"]
        rows.append(
            {
                "model": "Flatten + LSTM (no GCN)",
                "acc": float(v["accuracy"]),
                "acc_std": 0.0,
                "f1": float(v["macro_f1"]),
                "f1_std": 0.0,
                "bal": float(v.get("balanced_acc", 0)),
                "group": "ablation",
            }
        )
    return rows


def cnn1d_row(cnn: dict, ds: str) -> dict | None:
    if ds not in cnn:
        return None
    v = cnn[ds]
    return {
        "model": "CNN1D",
        "acc": float(v.get("accuracy", 0)),
        "acc_std": 0.0,
        "f1": float(v.get("macro_f1", 0)),
        "f1_std": 0.0,
        "bal": float(v.get("balanced_acc", 0)) if v.get("balanced_acc") is not None else None,
        "group": "deep",
    }


def build_dataset_rows(dataset: str) -> list[dict]:
    if dataset == "pamap2":
        rows: list[dict] = []
        rows.extend(baseline_rows(load("pamap2_baselines.json"), "p2"))
        c = load("cnn1d_results.json")
        cr = cnn1d_row(c, "pamap2")
        if cr:
            rows.append(cr)
        rows.extend(deep_rows(load("pamap2_deep_models.json"), "p2"))
        rows.extend(pamap2_extra_ablation())
        return rows

    # HHAR
    p = MET / "HHAR_baselines.json"
    if not p.exists():
        p = MET / "hhar_baselines.json"
    baselines = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    rows = baseline_rows(baselines, "hh")
    c = load("cnn1d_results.json")
    cr = cnn1d_row(c, "hhar")
    if cr:
        rows.append(cr)
    rows.extend(deep_rows(load("hhar_deep_models.json"), "hh"))
    return rows


def color_for_row(r: dict) -> str:
    if r["group"] == "classical":
        return "#4C72B0"
    if r["group"] == "proposed":
        return "#2ca02c"
    if r["group"] == "ablation":
        return "#9467bd"
    return "#DD8452"


def is_proposed(r: dict) -> bool:
    return r["group"] == "proposed"


def plot_panel(ax, rows: list[dict], metric: str, title: str, n_classes: int = 6) -> None:
    names = [r["model"] for r in rows]
    if metric == "acc":
        vals = np.array([r["acc"] for r in rows])
        errs = np.array([r["acc_std"] for r in rows])
    else:
        vals = np.array([r["f1"] for r in rows])
        errs = np.array([r["f1_std"] for r in rows])

    y = np.arange(len(names))
    colors = [color_for_row(r) for r in rows]
    bars = ax.barh(y, vals, xerr=errs, color=colors, edgecolor="white", height=0.72, capsize=3, alpha=0.92)
    for i, r in enumerate(rows):
        if is_proposed(r):
            bars[i].set_edgecolor("#1f4f1f")
            bars[i].set_linewidth(2.2)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Score (LOSO mean ± fold std)", fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.set_title(title, fontsize=12, fontweight="bold")
    p_chance = 1.0 / max(n_classes, 2)
    ax.axvline(p_chance, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(p_chance + 0.01, len(names) - 0.3, f"≈ chance ({n_classes}-class)", fontsize=7, color="gray")


def write_csv(all_rows: list[dict]) -> None:
    path = MET / "presentation_summary_table.csv"
    fields = ["dataset", "model", "accuracy", "accuracy_std", "macro_f1", "macro_f1_std", "balanced_acc"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow(
                {
                    "dataset": r["dataset"],
                    "model": r["model"],
                    "accuracy": f"{r['acc']:.4f}",
                    "accuracy_std": f"{r['acc_std']:.4f}",
                    "macro_f1": f"{r['f1']:.4f}",
                    "macro_f1_std": f"{r['f1_std']:.4f}",
                    "balanced_acc": f"{r['bal']:.4f}" if r.get("bal") is not None else "",
                }
            )
    print(f"Wrote {path}")


def main() -> None:
    p2 = build_dataset_rows("pamap2")
    hh = build_dataset_rows("hhar")
    for r in p2:
        r["dataset"] = "PAMAP2"
    for r in hh:
        r["dataset"] = "HHAR"

    write_csv([{**r} for r in p2] + [{**r} for r in hh])

    plt.rcParams.update({"font.family": "sans-serif", "figure.dpi": 120})
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)

    plot_panel(axes[0, 0], p2, "acc", "PAMAP2 — LOSO accuracy (± std across folds)", n_classes=12)
    plot_panel(axes[0, 1], p2, "f1", "PAMAP2 — Macro F1 (± std across folds)", n_classes=12)
    plot_panel(axes[1, 0], hh, "acc", "HHAR — LOSO accuracy (± std across folds)", n_classes=6)
    plot_panel(axes[1, 1], hh, "f1", "HHAR — Macro F1 (± std across folds)", n_classes=6)

    legend_elems = [
        mpatches.Patch(facecolor="#4C72B0", edgecolor="white", label="Classical (SVM / RF / XGB)"),
        mpatches.Patch(facecolor="#DD8452", edgecolor="white", label="Deep (CNN1D, LSTM, GNN)"),
        mpatches.Patch(facecolor="#2ca02c", edgecolor="#1f4f1f", linewidth=2, label="GNN+LSTM (proposed hybrid)"),
        mpatches.Patch(facecolor="#9467bd", edgecolor="white", label="PAMAP2 ablations"),
    ]
    fig.legend(handles=legend_elems, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), fontsize=10)
    fig.suptitle(
        "Human activity recognition — full model comparison (LOSO)\n"
        "Error bars = standard deviation of per-fold metrics (9 subjects). "
        "Dotted line ≈ uniform random guess (12 classes on PAMAP2, 6 on HHAR).",
        fontsize=11,
        y=1.06,
    )

    out_png = PLO / "presentation_all_models_summary.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
