"""
Export a merged midway comparison table across experiment groups.

Outputs:
  - results/metrics/midway_comparison.csv
  - results/metrics/midway_comparison.md
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
OUT_CSV = METRICS / "midway_comparison.csv"
OUT_MD = METRICS / "midway_comparison.md"


def load_json(name: str) -> dict:
    p = METRICS / name
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def add_row(
    rows: list[dict],
    dataset: str,
    model: str,
    exp_type: str,
    split: str,
    acc,
    f1,
    bal,
    acc_std=None,
    f1_std=None,
) -> None:
    rows.append(
        {
            "dataset": dataset,
            "model": model,
            "experiment_type": exp_type,
            "split_protocol": split,
            "accuracy": acc,
            "macro_f1": f1,
            "balanced_acc": bal,
            "accuracy_std": acc_std,
            "macro_f1_std": f1_std,
        }
    )


def fmt(v) -> str:
    if isinstance(v, (int, float)):
        return f"{v:.4f}"
    return ""


def main() -> None:
    rows: list[dict] = []

    # LOSO baselines
    pam_base = load_json("pamap2_baselines.json")
    hhar_base = load_json("HHAR_baselines.json")
    for model, m in pam_base.items():
        add_row(
            rows,
            "PAMAP2",
            model,
            "baseline",
            "LOSO",
            m.get("mean_accuracy"),
            m.get("mean_macro_f1"),
            None,
        )
    for model, m in hhar_base.items():
        add_row(
            rows,
            "HHAR",
            model,
            "baseline",
            "LOSO",
            m.get("mean_accuracy"),
            m.get("mean_macro_f1"),
            None,
        )

    # LOSO deep
    pam_deep = load_json("pamap2_deep_models.json")
    hhar_deep = load_json("hhar_deep_models.json")
    deep_name = {"lstm": "LSTM", "gnn": "GNN", "gnn_lstm": "GNN+LSTM", "cnn1d": "CNN1D"}
    for model, m in pam_deep.items():
        add_row(
            rows,
            "PAMAP2",
            deep_name.get(model, model),
            "deep",
            "LOSO",
            m.get("accuracy"),
            m.get("macro_f1"),
            m.get("balanced_acc"),
            m.get("accuracy_std"),
            m.get("macro_f1_std"),
        )
    for model, m in hhar_deep.items():
        add_row(
            rows,
            "HHAR",
            deep_name.get(model, model),
            "deep",
            "LOSO",
            m.get("accuracy"),
            m.get("macro_f1"),
            m.get("balanced_acc"),
            m.get("accuracy_std"),
            m.get("macro_f1_std"),
        )

    # LOSO graph ablations (PAMAP2)
    ab = load_json("graph_ablation_results.json")
    ab_name = {
        "ablation_fixed_adj": "GNN (fixed adj)",
        "ablation_learnable_adj": "GNN (learnable adj)",
        "ablation_flatten_lstm": "Flatten+LSTM",
    }
    for k, m in ab.items():
        add_row(
            rows,
            "PAMAP2",
            ab_name.get(k, k),
            "ablation",
            "LOSO",
            m.get("accuracy"),
            m.get("macro_f1"),
            m.get("balanced_acc"),
        )

    # Holdout (80/20)
    hold = load_json("holdout_results_full.json")
    for dataset, models in hold.items():
        ds = dataset.upper()
        for model, m in models.items():
            add_row(
                rows,
                ds,
                deep_name.get(model, model.upper()),
                "holdout",
                m.get("split", "80/20"),
                m.get("accuracy"),
                m.get("macro_f1"),
                m.get("balanced_acc"),
            )

    if not rows:
        raise ValueError("No rows found from available metrics files.")

    # Sort for readability
    exp_order = {"baseline": 0, "deep": 1, "ablation": 2, "holdout": 3}
    rows.sort(key=lambda r: (r["dataset"], exp_order.get(r["experiment_type"], 9), r["model"]))

    METRICS.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "model",
        "experiment_type",
        "split_protocol",
        "accuracy",
        "accuracy_std",
        "macro_f1",
        "macro_f1_std",
        "balanced_acc",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Markdown table for easy PPT/doc pasting
    with OUT_MD.open("w", encoding="utf-8") as f:
        f.write(
            "| Dataset | Model | Experiment Type | Split | Acc | Acc σ | Macro-F1 | F1 σ | Balanced Acc |\n"
        )
        f.write("|---|---|---|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            asd = fmt(r["accuracy_std"]) if r.get("accuracy_std") is not None else ""
            fsd = fmt(r["macro_f1_std"]) if r.get("macro_f1_std") is not None else ""
            f.write(
                f"| {r['dataset']} | {r['model']} | {r['experiment_type']} | {r['split_protocol']} | "
                f"{fmt(r['accuracy'])} | {asd} | {fmt(r['macro_f1'])} | {fsd} | {fmt(r['balanced_acc'])} |\n"
            )

    print(f"Exported {len(rows)} rows")
    print(f"CSV: {OUT_CSV}")
    print(f"MD : {OUT_MD}")


if __name__ == "__main__":
    main()
