#!/usr/bin/env python3
"""Run all seven publication experiments sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = [
    "run_experiment_1_leakage_control.py",
    "run_experiment_2_statistical_reliability.py",
    "run_experiment_3_robustness.py",
    "run_experiment_4_calibration_uncertainty.py",
    "run_experiment_5_subject_failure_analysis.py",
    "run_experiment_6_few_shot_calibration.py",
    "run_experiment_7_health_group_analysis.py",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--only", nargs="+", type=int, choices=range(1, 8), help="Run subset e.g. --only 1 2")
    args = p.parse_args()

    scripts = EXPERIMENTS
    if args.only:
        scripts = [EXPERIMENTS[i - 1] for i in sorted(args.only)]

    py = sys.executable
    for script in scripts:
        cmd = [py, str(ROOT / script)]
        if args.smoke:
            cmd.append("--smoke")
        if args.datasets:
            cmd.extend(["--datasets", *args.datasets])
        print("\n" + "=" * 70)
        print("Running:", " ".join(cmd))
        print("=" * 70)
        subprocess.run(cmd, check=True)

    # Combined summary stub
    summary = ROOT.parent / "results" / "manuscript_tables" / "overall_final_summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    if not summary.exists():
        summary.write_text("note,Run individual experiment CSVs in manuscript_tables/ for full summary\n")
    print("\nAll requested publication experiments finished.")


if __name__ == "__main__":
    main()
