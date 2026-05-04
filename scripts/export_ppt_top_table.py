"""
Create a compact PPT-friendly top table from midway_comparison.csv.

Selection logic:
  - For each dataset and split group:
      * keep best baseline by accuracy
      * keep best deep by accuracy
      * keep best ablation by accuracy (LOSO only, if available)
      * keep best holdout by accuracy

Outputs:
  - results/metrics/ppt_top_table.csv
  - results/metrics/ppt_top_table.md
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
IN_CSV = METRICS / "midway_comparison.csv"
OUT_CSV = METRICS / "ppt_top_table.csv"
OUT_MD = METRICS / "ppt_top_table.md"


def parse_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("-inf")


def choose_best(rows: list[dict], exp_type: str, dataset: str, split_contains: str | None = None) -> dict | None:
    cand = [r for r in rows if r["dataset"] == dataset and r["experiment_type"] == exp_type]
    if split_contains is not None:
        cand = [r for r in cand if split_contains in r["split_protocol"]]
    if not cand:
        return None
    return max(cand, key=lambda r: parse_float(r["accuracy"]))


def fmt(v: str) -> str:
    x = parse_float(v)
    if x == float("-inf"):
        return ""
    return f"{x:.4f}"


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing input: {IN_CSV}. Run export_midway_comparison.py first.")

    with IN_CSV.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out_rows: list[dict] = []
    datasets = sorted(set(r["dataset"] for r in rows))

    for ds in datasets:
        # LOSO highlights
        for exp in ("baseline", "deep", "ablation"):
            best = choose_best(rows, exp_type=exp, dataset=ds, split_contains="LOSO")
            if best:
                out_rows.append(best)
        # Holdout highlights
        best_hold = choose_best(rows, exp_type="holdout", dataset=ds, split_contains="80/20")
        if best_hold:
            out_rows.append(best_hold)

    # sort for readability
    exp_order = {"baseline": 0, "deep": 1, "ablation": 2, "holdout": 3}
    out_rows.sort(key=lambda r: (r["dataset"], exp_order.get(r["experiment_type"], 9)))

    fieldnames = [
        "dataset",
        "model",
        "experiment_type",
        "split_protocol",
        "accuracy",
        "macro_f1",
        "balanced_acc",
    ]

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    with OUT_MD.open("w", encoding="utf-8") as f:
        f.write("| Dataset | Model | Experiment Type | Split | Accuracy | Macro-F1 | Balanced Acc |\n")
        f.write("|---|---|---|---|---:|---:|---:|\n")
        for r in out_rows:
            f.write(
                f"| {r['dataset']} | {r['model']} | {r['experiment_type']} | {r['split_protocol']} | "
                f"{fmt(r['accuracy'])} | {fmt(r['macro_f1'])} | {fmt(r['balanced_acc'])} |\n"
            )

    print(f"Exported {len(out_rows)} rows")
    print(f"CSV: {OUT_CSV}")
    print(f"MD : {OUT_MD}")


if __name__ == "__main__":
    main()
