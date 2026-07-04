#!/usr/bin/env python3
"""Validate outputs, merge PAMAP2+HHAR tables, export figures, build combined CSVs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.publication.statistics import (
    bootstrap_mean_diff_ci,
    cohens_d,
    cohens_d_one_sample,
    mean_ci95,
    rank_stability,
    format_rank_stability,
    wilcoxon_signed_rank,
    wilcoxon_one_sample,
)

TABLES = ROOT / "results" / "manuscript_tables"
COMBINED = TABLES / "combined"
FIGURES = ROOT / "results" / "manuscript_figures"
SNAPSHOTS = TABLES / "pamap2_snapshots"
HHAR_SNAPSHOTS = TABLES / "hhar_snapshots"
EXP = ROOT / "results"

TABLE_MAP = {
    "table_exp1_leakage_control.csv": ("experiment_1_leakage_control", "leakage_control_summary.csv"),
    "table_exp2_statistical_reliability.csv": (
        "experiment_2_statistical_reliability",
        "statistical_reliability_summary.csv",
    ),
    "table_exp2_pairwise_comparisons.csv": (
        "experiment_2_statistical_reliability",
        "pairwise_comparisons.csv",
    ),
    "table_exp3_robustness.csv": ("experiment_3_robustness", "robustness_summary.csv"),
    "table_exp4_calibration.csv": ("experiment_4_calibration_uncertainty", "calibration_summary.csv"),
    "table_exp5_subject_failure.csv": (
        "experiment_5_subject_failure_analysis",
        "subject_failure_summary.csv",
    ),
    "table_exp6_few_shot_calibration.csv": (
        "experiment_6_few_shot_calibration",
        "few_shot_calibration_summary.csv",
    ),
    "table_exp7_fine_vs_group.csv": (
        "experiment_7_health_group_analysis",
        "fine_vs_group_comparison.csv",
    ),
    "table_exp7_health_groups.csv": (
        "experiment_7_health_group_analysis",
        "health_group_metrics.csv",
    ),
}

EXP_SCRIPTS = {
    3: "scripts/run_experiment_3_robustness.py",
    5: "scripts/run_experiment_5_subject_failure_analysis.py",
    6: "scripts/run_experiment_6_few_shot_calibration.py",
    7: "scripts/run_experiment_7_health_group_analysis.py",
}


def validate_csvs() -> list[str]:
    issues = []
    for f in sorted(ROOT.glob("results/**/*.csv")):
        if "combined" in f.parts or "pamap2_snapshots" in f.parts or "hhar_snapshots" in f.parts:
            continue
        try:
            df = pd.read_csv(f)
            if len(df) == 0:
                issues.append(f"EMPTY: {f.relative_to(ROOT)}")
            elif len(df) == 1:
                issues.append(f"HEADER ONLY: {f.relative_to(ROOT)}")
        except Exception as e:
            issues.append(f"ERROR: {f.relative_to(ROOT)} -> {e}")
    return issues


def reconstruct_exp1_pamap2() -> pd.DataFrame:
    pred_dir = EXP / "experiment_1_leakage_control" / "predictions"
    rows = []
    cache: dict[tuple, dict] = {}
    for meta_path in sorted(pred_dir.glob("pamap2_*_metadata.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        wt = meta["window_type"]
        protocol = meta["protocol"]
        model = meta["model"]
        agg = meta["aggregate"]
        key = ("pamap2", wt, model, protocol)
        cache[key] = agg
        rows.append({
            "Dataset": "pamap2",
            "Window_Type": wt,
            "Evaluation_Protocol": protocol,
            "Model": model,
            "Accuracy": agg["accuracy"],
            "Macro_F1": agg["macro_f1"],
            "Balanced_Accuracy": agg["balanced_accuracy"],
            "Leakage_Gap": None,
        })
    for r in rows:
        if r["Evaluation_Protocol"] == "random_holdout":
            lo = cache.get(("pamap2", r["Window_Type"], r["Model"], "loso"))
            if lo:
                r["Leakage_Gap"] = r["Accuracy"] - lo["accuracy"]
    return pd.DataFrame(rows)


def reconstruct_exp2_pamap2() -> tuple[pd.DataFrame, pd.DataFrame]:
    out = EXP / "experiment_2_statistical_reliability"
    summary_rows, pair_rows = [], []
    fold_scores: dict[str, list[float]] = {}
    for path in sorted(out.glob("fold_metrics_pamap2_*.json")):
        model = path.stem.replace("fold_metrics_pamap2_", "")
        with open(path) as f:
            folds = json.load(f)
        accs = [x["accuracy"] for x in folds]
        f1s = [x["macro_f1"] for x in folds]
        fold_scores[model] = accs
        acc_arr = np.array(accs)
        chance_acc = 1.0 / 12  # PAMAP2 fine-grained classes
        am, alo, ahi = mean_ci95(acc_arr)
        fm, flo, fhi = mean_ci95(np.array(f1s))
        rs = rank_stability(fold_scores)
        summary_rows.append({
            "Dataset": "pamap2",
            "Model": model,
            "Accuracy_Mean": am,
            "Accuracy_SD": float(np.std(accs, ddof=1) if len(accs) > 1 else 0),
            "Accuracy_CI95_Lower": alo,
            "Accuracy_CI95_Upper": ahi,
            "Macro_F1_Mean": fm,
            "Macro_F1_SD": float(np.std(f1s, ddof=1) if len(f1s) > 1 else 0),
            "Macro_F1_CI95_Lower": flo,
            "Macro_F1_CI95_Upper": fhi,
            "Effect_Size": cohens_d_one_sample(acc_arr, chance_acc),
            "Wilcoxon_P_Value": wilcoxon_one_sample(acc_arr, chance_acc),
            "Rank_Stability": format_rank_stability(rs[model], len(accs)),
        })
    models = list(fold_scores.keys())
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            ma, mb = models[i], models[j]
            a = np.array(fold_scores[ma])
            b = np.array(fold_scores[mb])
            md, blo, bhi = bootstrap_mean_diff_ci(a, b, seed=42)
            pair_rows.append({
                "Dataset": "pamap2",
                "Metric": "accuracy",
                "Model_A": ma,
                "Model_B": mb,
                "Mean_Difference": md,
                "Bootstrap_CI95_Lower": blo,
                "Bootstrap_CI95_Upper": bhi,
                "Effect_Size": cohens_d(a, b),
                "Wilcoxon_P_Value": wilcoxon_signed_rank(a, b),
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)


def reconstruct_exp4_pamap2() -> pd.DataFrame:
    out = EXP / "experiment_4_calibration_uncertainty"
    rows = []
    for path in sorted(out.glob("calibration_pamap2_*.json")):
        model = path.stem.replace("calibration_pamap2_", "")
        with open(path) as f:
            cal = json.load(f)
        rows.append({
            "Dataset": "pamap2",
            "Model": model,
            "ECE": cal["ece"],
            "Brier_Score": cal["brier_score"],
            "NLL": cal["nll"],
            "Accuracy_At_90_Coverage": cal["accuracy_at_90_coverage"],
            "Accuracy_At_80_Coverage": cal["accuracy_at_80_coverage"],
            "Accuracy_At_70_Coverage": cal["accuracy_at_70_coverage"],
            "Macro_F1_At_90_Coverage": cal["macro_f1_at_90_coverage"],
            "Macro_F1_At_80_Coverage": cal["macro_f1_at_80_coverage"],
            "Macro_F1_At_70_Coverage": cal["macro_f1_at_70_coverage"],
        })
    return pd.DataFrame(rows)


def run_dataset_experiment(exp_num: int, dataset: str) -> None:
    script = ROOT / EXP_SCRIPTS[exp_num]
    print(f"Running {dataset} experiment {exp_num}")
    subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(script), "--datasets", dataset],
        cwd=ROOT,
        check=True,
        env={**dict(__import__("os").environ), "HAR_FORCE_DEVICE": "cpu"},
    )


def copy_dataset_snapshot(table_name: str, snapshot_name: str, dataset: str, out_dir: Path) -> pd.DataFrame:
    exp_dir_name, csv_name = TABLE_MAP[table_name]
    src = EXP / exp_dir_name / csv_name
    df = pd.read_csv(src)
    if "Dataset" in df.columns:
        df = df[df["Dataset"].astype(str).str.lower() == dataset]
    path = out_dir / snapshot_name
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite a non-empty snapshot with an empty filter result
    # (single-dataset runs leave only the other dataset in experiment CSVs).
    if len(df) == 0 and path.exists():
        existing = pd.read_csv(path)
        if len(existing) > 0:
            print(
                f"SKIP empty snapshot for {snapshot_name}: experiment CSV has no "
                f"{dataset} rows; keeping existing {len(existing)}-row snapshot"
            )
            return existing
    df.to_csv(path, index=False)
    print(f"Saved snapshot {snapshot_name} ({len(df)} rows)")
    return df


def try_archive_from_experiment(table_name: str, snapshot_name: str, dataset: str, out_dir: Path) -> bool:
    exp_dir_name, csv_name = TABLE_MAP[table_name]
    src = EXP / exp_dir_name / csv_name
    if not src.exists():
        return False
    df = pd.read_csv(src)
    if "Dataset" not in df.columns:
        return False
    sub = df[df["Dataset"].astype(str).str.lower() == dataset]
    if len(sub) == 0:
        return False
    path = out_dir / snapshot_name
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    print(f"Archived {snapshot_name} from experiment output ({len(sub)} rows)")
    return True


def ensure_hhar_snapshots(rerun_missing: bool = True) -> None:
    HHAR_SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    # Exp1 HHAR from prediction metadata
    def recon_hhar_exp1():
        pred_dir = EXP / "experiment_1_leakage_control" / "predictions"
        rows = []
        cache = {}
        for meta_path in sorted(pred_dir.glob("hhar_*_metadata.json")):
            with open(meta_path) as f:
                meta = json.load(f)
            wt, protocol, model = meta["window_type"], meta["protocol"], meta["model"]
            agg = meta["aggregate"]
            cache[("hhar", wt, model, protocol)] = agg
            rows.append({
                "Dataset": "hhar", "Window_Type": wt, "Evaluation_Protocol": protocol,
                "Model": model, "Accuracy": agg["accuracy"], "Macro_F1": agg["macro_f1"],
                "Balanced_Accuracy": agg["balanced_accuracy"], "Leakage_Gap": None,
            })
        for r in rows:
            if r["Evaluation_Protocol"] == "random_holdout":
                lo = cache.get(("hhar", r["Window_Type"], r["Model"], "loso"))
                if lo:
                    r["Leakage_Gap"] = r["Accuracy"] - lo["accuracy"]
        return pd.DataFrame(rows)

    def recon_hhar_exp2():
        out = EXP / "experiment_2_statistical_reliability"
        summary_rows, pair_rows = [], []
        fold_scores = {}
        for path in sorted(out.glob("fold_metrics_hhar_*.json")):
            model = path.stem.replace("fold_metrics_hhar_", "")
            with open(path) as f:
                folds = json.load(f)
            accs = [x["accuracy"] for x in folds]
            f1s = [x["macro_f1"] for x in folds]
            fold_scores[model] = accs
            acc_arr = np.array(accs)
            chance_acc = 1.0 / 6  # HHAR fine-grained classes
            am, alo, ahi = mean_ci95(acc_arr)
            fm, flo, fhi = mean_ci95(np.array(f1s))
            rs = rank_stability(fold_scores)
            summary_rows.append({
                "Dataset": "hhar", "Model": model,
                "Accuracy_Mean": am, "Accuracy_SD": float(np.std(accs, ddof=1) if len(accs) > 1 else 0),
                "Accuracy_CI95_Lower": alo, "Accuracy_CI95_Upper": ahi,
                "Macro_F1_Mean": fm, "Macro_F1_SD": float(np.std(f1s, ddof=1) if len(f1s) > 1 else 0),
                "Macro_F1_CI95_Lower": flo, "Macro_F1_CI95_Upper": fhi,
                "Effect_Size": cohens_d_one_sample(acc_arr, chance_acc),
                "Wilcoxon_P_Value": wilcoxon_one_sample(acc_arr, chance_acc),
                "Rank_Stability": format_rank_stability(rs[model], len(accs)),
            })
        models = list(fold_scores.keys())
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                ma, mb = models[i], models[j]
                a, b = np.array(fold_scores[ma]), np.array(fold_scores[mb])
                md, blo, bhi = bootstrap_mean_diff_ci(a, b, seed=42)
                pair_rows.append({
                    "Dataset": "hhar", "Metric": "accuracy", "Model_A": ma, "Model_B": mb,
                    "Mean_Difference": md, "Bootstrap_CI95_Lower": blo, "Bootstrap_CI95_Upper": bhi,
                    "Effect_Size": cohens_d(a, b), "Wilcoxon_P_Value": wilcoxon_signed_rank(a, b),
                })
        return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)

    hhar_recon = {
        "hhar_table_exp1_leakage_control.csv": recon_hhar_exp1,
    }
    s2, p2 = recon_hhar_exp2()
    hhar_recon["hhar_table_exp2_statistical_reliability.csv"] = lambda: s2
    hhar_recon["hhar_table_exp2_pairwise_comparisons.csv"] = lambda: p2

    def recon_hhar_exp4():
        out = EXP / "experiment_4_calibration_uncertainty"
        rows = []
        for path in sorted(out.glob("calibration_hhar_*.json")):
            model = path.stem.replace("calibration_hhar_", "")
            with open(path) as f:
                cal = json.load(f)
            rows.append({
                "Dataset": "hhar", "Model": model,
                "ECE": cal["ece"], "Brier_Score": cal["brier_score"], "NLL": cal["nll"],
                "Accuracy_At_90_Coverage": cal["accuracy_at_90_coverage"],
                "Accuracy_At_80_Coverage": cal["accuracy_at_80_coverage"],
                "Accuracy_At_70_Coverage": cal["accuracy_at_70_coverage"],
                "Macro_F1_At_90_Coverage": cal["macro_f1_at_90_coverage"],
                "Macro_F1_At_80_Coverage": cal["macro_f1_at_80_coverage"],
                "Macro_F1_At_70_Coverage": cal["macro_f1_at_70_coverage"],
            })
        return pd.DataFrame(rows)

    hhar_recon["hhar_table_exp4_calibration.csv"] = recon_hhar_exp4

    for name, fn in hhar_recon.items():
        path = HHAR_SNAPSHOTS / name
        if not path.exists() or len(pd.read_csv(path)) <= 1:
            fn().to_csv(path, index=False)
            print(f"Reconstructed {name}")

    hhar_need_run = {
        3: [("table_exp3_robustness.csv", "hhar_table_exp3_robustness.csv")],
        5: [("table_exp5_subject_failure.csv", "hhar_table_exp5_subject_failure.csv")],
        6: [("table_exp6_few_shot_calibration.csv", "hhar_table_exp6_few_shot_calibration.csv")],
        7: [
            ("table_exp7_fine_vs_group.csv", "hhar_table_exp7_fine_vs_group.csv"),
            ("table_exp7_health_groups.csv", "hhar_table_exp7_health_groups.csv"),
        ],
    }
    for exp_num, pairs in hhar_need_run.items():
        for table_name, snap_name in pairs:
            path = HHAR_SNAPSHOTS / snap_name
            if path.exists() and len(pd.read_csv(path)) > 1:
                continue
            try_archive_from_experiment(table_name, snap_name, "hhar", HHAR_SNAPSHOTS)
        missing = [s for _, s in pairs if not (HHAR_SNAPSHOTS / s).exists() or len(pd.read_csv(HHAR_SNAPSHOTS / s)) <= 1]
        if missing and rerun_missing and exp_num in EXP_SCRIPTS:
            run_dataset_experiment(exp_num, "hhar")
            for table_name, snap_name in pairs:
                copy_dataset_snapshot(table_name, snap_name, "hhar", HHAR_SNAPSHOTS)


def ensure_pamap2_snapshots(rerun_missing: bool = True) -> None:
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    recon = {
        "pamap2_table_exp1_leakage_control.csv": reconstruct_exp1_pamap2,
    }
    s, p = reconstruct_exp2_pamap2()
    recon["pamap2_table_exp2_statistical_reliability.csv"] = lambda: s
    recon["pamap2_table_exp2_pairwise_comparisons.csv"] = lambda: p
    recon["pamap2_table_exp4_calibration.csv"] = reconstruct_exp4_pamap2

    for name, fn in recon.items():
        path = SNAPSHOTS / name
        if not path.exists() or len(pd.read_csv(path)) == 0:
            df = fn()
            df.to_csv(path, index=False)
            print(f"Reconstructed {path.name} ({len(df)} rows)")

    need_run = {
        3: [
            ("table_exp3_robustness.csv", "pamap2_table_exp3_robustness.csv"),
        ],
        5: [
            ("table_exp5_subject_failure.csv", "pamap2_table_exp5_subject_failure.csv"),
        ],
        6: [
            ("table_exp6_few_shot_calibration.csv", "pamap2_table_exp6_few_shot_calibration.csv"),
        ],
        7: [
            ("table_exp7_fine_vs_group.csv", "pamap2_table_exp7_fine_vs_group.csv"),
            ("table_exp7_health_groups.csv", "pamap2_table_exp7_health_groups.csv"),
        ],
    }
    for exp_num, pairs in need_run.items():
        for table_name, snap_name in pairs:
            path = SNAPSHOTS / snap_name
            if path.exists() and len(pd.read_csv(path)) > 1:
                continue
            try_archive_from_experiment(table_name, snap_name, "pamap2", SNAPSHOTS)

    for exp_num, pairs in need_run.items():
        missing = [s for _, s in pairs if not (SNAPSHOTS / s).exists() or len(pd.read_csv(SNAPSHOTS / s)) <= 1]
        if not missing:
            continue
        if not rerun_missing:
            print(f"MISSING PAMAP2 snapshots for experiment {exp_num}: {missing}")
            continue
        run_dataset_experiment(exp_num, "pamap2")
        for table_name, snap_name in pairs:
            copy_dataset_snapshot(table_name, snap_name, "pamap2", SNAPSHOTS)


def load_snapshot_table(table_name: str, dataset: str) -> pd.DataFrame | None:
    prefix = f"{dataset}_"
    snap_name = prefix + table_name
    snap_dir = SNAPSHOTS if dataset == "pamap2" else HHAR_SNAPSHOTS
    path = snap_dir / snap_name
    if path.exists() and len(pd.read_csv(path)) > 0:
        return pd.read_csv(path)
    return None


def merge_tables() -> None:
    COMBINED.mkdir(parents=True, exist_ok=True)
    table_names = [
        "table_exp1_leakage_control.csv",
        "table_exp2_statistical_reliability.csv",
        "table_exp2_pairwise_comparisons.csv",
        "table_exp3_robustness.csv",
        "table_exp4_calibration.csv",
        "table_exp5_subject_failure.csv",
        "table_exp6_few_shot_calibration.csv",
        "table_exp7_fine_vs_group.csv",
        "table_exp7_health_groups.csv",
    ]
    for name in table_names:
        pam = load_snapshot_table(name, "pamap2")
        hhar = load_snapshot_table(name, "hhar")
        parts = [d for d in (pam, hhar) if d is not None and len(d)]
        if not parts:
            print(f"SKIP merge (no snapshots): {name}")
            continue
        combined = pd.concat(parts, ignore_index=True)
        out = COMBINED / name
        combined.to_csv(out, index=False)
        print(
            f"Merged {out.name}: pamap2={0 if pam is None else len(pam)} "
            f"hhar={0 if hhar is None else len(hhar)} total={len(combined)}"
        )


def export_figures() -> None:
    """Ensure PNG+PDF pairs exist in manuscript_figures; convert lone PNGs to PDF."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.image import imread
    except ImportError:
        print("matplotlib not available for figure export")
        return

    FIGURES.mkdir(parents=True, exist_ok=True)
    for png in sorted(FIGURES.glob("*.png")):
        pdf = png.with_suffix(".pdf")
        if not pdf.exists():
            img = imread(png)
            fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
            ax.imshow(img)
            ax.axis("off")
            fig.savefig(pdf, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            print(f"Created PDF: {pdf.name}")

    # Copy experiment figures missing from manuscript_figures
    exp5 = EXP / "experiment_5_subject_failure_analysis"
    for cm in exp5.glob("cm_*.png"):
        dest = FIGURES / f"fig_exp5_cm_{cm.stem.replace('cm_', '')}.png"
        if not dest.exists():
            dest.write_bytes(cm.read_bytes())


def fix_overall_summary() -> None:
    rows = []
    for f in sorted(COMBINED.glob("table_exp*.csv")):
        df = pd.read_csv(f)
        rows.append({"Table": f.name, "Rows": len(df), "Datasets": ",".join(sorted(df["Dataset"].unique())) if "Dataset" in df.columns else "n/a"})
    pd.DataFrame(rows).to_csv(TABLES / "overall_final_summary.csv", index=False)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--skip-rerun", action="store_true", help="Do not rerun missing PAMAP2 experiments")
    args = p.parse_args()

    issues = validate_csvs()
    for i in issues:
        print(i)
    empty = [x for x in issues if x.startswith("EMPTY:")]
    if empty and "overall_final_summary" not in empty[0]:
        print("WARNING: unexpected empty CSVs")

    ensure_hhar_snapshots(rerun_missing=not args.skip_rerun)
    ensure_pamap2_snapshots(rerun_missing=not args.skip_rerun)
    merge_tables()
    export_figures()
    fix_overall_summary()
    print("Manuscript preparation complete.")


if __name__ == "__main__":
    main()
