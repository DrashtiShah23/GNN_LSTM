#!/usr/bin/env bash
# Archive exp3/exp7 snapshots after reruns, merge tables, and regenerate MANUSCRIPT_RESULTS.md.
# Never overwrites a non-empty per-dataset snapshot with an empty filter result.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=".venv/bin/python"

archive() {
  "$PY" <<'PY'
import sys
from pathlib import Path
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))
from scripts.prepare_manuscript import copy_dataset_snapshot, SNAPSHOTS, HHAR_SNAPSHOTS

for ds, out in [("pamap2", SNAPSHOTS), ("hhar", HHAR_SNAPSHOTS)]:
    prefix = f"{ds}_"
    for table in (
        "table_exp3_robustness.csv",
        "table_exp7_health_groups.csv",
        "table_exp7_fine_vs_group.csv",
    ):
        copy_dataset_snapshot(table, f"{prefix}{table}", ds, out)
PY
}

archive
"$PY" scripts/prepare_manuscript.py --skip-rerun
"$PY" scripts/write_manuscript_results.py

"$PY" - <<'PY'
import pandas as pd
from pathlib import Path

def count_ds(path, dataset):
    df = pd.read_csv(path)
    return int((df["Dataset"].astype(str).str.lower() == dataset).sum())

checks = [
    ("Effect_Size", "Wilcoxon_P_Value"),
    ("Most_Affected_Class",),
    ("Main_Confusion",),
]
paths = [
    "results/manuscript_tables/combined/table_exp2_statistical_reliability.csv",
    "results/manuscript_tables/combined/table_exp3_robustness.csv",
    "results/manuscript_tables/combined/table_exp7_health_groups.csv",
]
for p, cols in zip(paths, checks):
    df = pd.read_csv(p)
    if "Most_Affected_Class" in cols:
        hr = df["Perturbation"].astype(str).str.contains("heart", case=False)
        bad = {c: int(df.loc[~hr, c].isna().sum()) for c in cols}
    else:
        bad = {c: int(df[c].isna().sum()) for c in cols}
    print(Path(p).name, bad)
    if any(bad.values()):
        raise SystemExit(f"NaN remain in {p}: {bad}")

for table, need_pamap2 in [
    ("results/manuscript_tables/combined/table_exp3_robustness.csv", 54),
    ("results/manuscript_tables/combined/table_exp7_health_groups.csv", 18),
    ("results/manuscript_tables/combined/table_exp7_fine_vs_group.csv", 3),
]:
    n = count_ds(table, "pamap2")
    print(f"{Path(table).name}: pamap2={n} (need>={need_pamap2})")
    if n < need_pamap2:
        raise SystemExit(f"PAMAP2 rows missing in {table}: got {n}")

print("All target columns populated; PAMAP2 rows present for exp3/exp7.")
PY
