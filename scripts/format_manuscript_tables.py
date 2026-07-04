#!/usr/bin/env python3
"""Format combined manuscript CSVs for publication display.

Writes to results/manuscript_tables/formatted/ without modifying combined/.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COMBINED = ROOT / "results" / "manuscript_tables" / "combined"
FORMATTED = ROOT / "results" / "manuscript_tables" / "formatted"

# Proportion metrics → percentage XX.XX
PCT_EXACT = {
    "Accuracy", "Macro_F1", "Balanced_Accuracy", "Sensitivity", "Specificity",
    "Clean_Accuracy", "Perturbed_Accuracy", "Accuracy_Drop",
    "Clean_Macro_F1", "Perturbed_Macro_F1", "Macro_F1_Drop",
    "Accuracy_Mean", "Accuracy_SD", "Accuracy_CI95_Lower", "Accuracy_CI95_Upper",
    "Macro_F1_Mean", "Macro_F1_SD", "Macro_F1_CI95_Lower", "Macro_F1_CI95_Upper",
    "Fine_Grained_Accuracy", "Group_Level_Accuracy",
    "Fine_Grained_Macro_F1", "Group_Level_Macro_F1",
    "Uncalibrated_Accuracy", "Calibrated_Accuracy", "Accuracy_Improvement",
    "Uncalibrated_Macro_F1", "Calibrated_Macro_F1", "Macro_F1_Improvement",
    "Accuracy_At_90_Coverage", "Accuracy_At_80_Coverage", "Accuracy_At_70_Coverage",
    "Macro_F1_At_90_Coverage", "Macro_F1_At_80_Coverage", "Macro_F1_At_70_Coverage",
    "ECE", "Brier_Score", "Missingness", "Activity_Imbalance",
    "Calibration_Percentage", "Mean_Difference",
    "Bootstrap_CI95_Lower", "Bootstrap_CI95_Upper",
}

PCT_SUFFIXES = (
    "_Accuracy", "_Macro_F1", "_Sensitivity", "_Specificity",
    "_Balanced_Accuracy",
)

MEAN_SD_PAIRS = [
    ("Accuracy_Mean", "Accuracy_SD", "Accuracy_Mean_SD"),
    ("Macro_F1_Mean", "Macro_F1_SD", "Macro_F1_Mean_SD"),
]

PVALUE_COLS = {"Wilcoxon_P_Value"}
EFFECT_COLS = {"Effect_Size"}
# Leave as-is (not percentages): NLL, Sensor_Variability, ranks, labels
SKIP_PCT = {"NLL", "Sensor_Variability", "Rank_Stability"}


def is_missing(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return True
    s = str(val).strip().lower()
    return s in {"", "nan", "none", "not_applicable", "n/a", "na"}


def fmt_na(val):
    return "N/A" if is_missing(val) else val


def to_float(val):
    if is_missing(val):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def fmt_pct(val) -> str:
    """Proportion in [0,1] (or already-ish) → XX.XX percentage string."""
    x = to_float(val)
    if x is None:
        return "N/A"
    # Values already look like percentages (>1.5 and not tiny drops) stay as-is scale
    # All our metrics are proportions in [0,1] or small signed drops.
    return f"{round(x * 100, 2):.2f}"


def fmt_pvalue(val) -> str:
    x = to_float(val)
    if x is None:
        return "N/A"
    if x < 0.001:
        return "p < 0.001"
    return f"p = {x:.4f}"


def fmt_effect(val) -> str:
    x = to_float(val)
    if x is None:
        return "N/A"
    return f"d = {x:.2f}"


def is_pct_column(name: str) -> bool:
    if name in SKIP_PCT or name in PVALUE_COLS or name in EFFECT_COLS:
        return False
    if name in PCT_EXACT:
        return True
    return any(name.endswith(suf) for suf in PCT_SUFFIXES)


def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Mean ± SD display columns (percentages)
    for mean_col, sd_col, display_col in MEAN_SD_PAIRS:
        if mean_col in out.columns and sd_col in out.columns:
            display = []
            for m, s in zip(out[mean_col], out[sd_col]):
                if is_missing(m) or is_missing(s):
                    display.append("N/A")
                else:
                    display.append(f"{fmt_pct(m)} ± {fmt_pct(s)}")
            out[display_col] = display

    for col in list(out.columns):
        if col in PVALUE_COLS:
            out[col] = [fmt_pvalue(v) for v in out[col]]
        elif col in EFFECT_COLS:
            out[col] = [fmt_effect(v) for v in out[col]]
        elif is_pct_column(col):
            out[col] = [fmt_pct(v) for v in out[col]]
        else:
            # Categorical / IDs / free text: only normalize missing markers
            out[col] = [fmt_na(v) if is_missing(v) else v for v in out[col]]

    return out


def main() -> None:
    FORMATTED.mkdir(parents=True, exist_ok=True)
    paths = sorted(COMBINED.glob("*.csv"))
    if not paths:
        raise SystemExit(f"No combined CSVs found in {COMBINED}")

    for path in paths:
        df = pd.read_csv(path)
        formatted = format_dataframe(df)
        out_path = FORMATTED / path.name
        formatted.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(formatted)} rows)")

    print(f"Done. Formatted tables in {FORMATTED}")


if __name__ == "__main__":
    main()
