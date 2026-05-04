"""
Export a normalized final metrics table from master_comparison.json.

Output:
    results/metrics/final_table.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MASTER_JSON = ROOT / "results" / "metrics" / "master_comparison.json"
OUT_CSV = ROOT / "results" / "metrics" / "final_table.csv"


def main() -> None:
    if not MASTER_JSON.exists():
        raise FileNotFoundError(f"Missing input file: {MASTER_JSON}")

    with MASTER_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    # Preferred structured sections with nested metric dicts
    for ds_key in ("PAMAP2", "HHAR"):
        section = data.get(ds_key)
        if isinstance(section, dict):
            for model, metrics in section.items():
                if isinstance(metrics, dict):
                    rows.append(
                        {
                            "dataset": ds_key,
                            "model": model,
                            "accuracy": metrics.get("accuracy"),
                            "macro_f1": metrics.get("macro_f1"),
                            "balanced_acc": metrics.get("balanced_acc"),
                            "source_section": ds_key,
                        }
                    )

    # Fallback flat sections (accuracy-only summary)
    for ds_key in ("pamap2", "hhar"):
        section = data.get(ds_key)
        if isinstance(section, dict):
            for model, acc in section.items():
                rows.append(
                    {
                        "dataset": ds_key.upper(),
                        "model": model,
                        "accuracy": acc if isinstance(acc, (int, float)) else None,
                        "macro_f1": None,
                        "balanced_acc": None,
                        "source_section": ds_key,
                    }
                )

    if not rows:
        raise ValueError("No rows parsed from master_comparison.json")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "model",
        "accuracy",
        "macro_f1",
        "balanced_acc",
        "source_section",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} rows to: {OUT_CSV}")


if __name__ == "__main__":
    main()
