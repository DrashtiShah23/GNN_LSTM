#!/usr/bin/env python
"""Build the seven protocol12 experiment tables from canonical result artifacts.

This runner is intentionally post-hoc: it consumes canonical v1/v3 result files
instead of retraining the older publication pipeline. It is meant for comparing
canonical baselines against the v3 improved GNN models.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


LABEL_GROUPS = {
    "lying": "posture",
    "sitting": "posture",
    "standing": "posture",
    "walking": "locomotion",
    "running": "locomotion",
    "cycling": "locomotion",
    "nordic_walking": "locomotion",
    "ascending_stairs": "stairs",
    "descending_stairs": "stairs",
    "vacuum_cleaning": "household",
    "ironing": "household",
    "rope_jumping": "jump",
}


BASELINE_MODELS = {
    "dummy_most_frequent",
    "gaussian_nb",
    "knn_k5",
    "linear_svm",
    "rbf_svm",
    "decision_tree_entropy",
    "bagged_tree_entropy",
    "random_forest",
    "adaboost_tree",
    "xgboost_hist",
}

V3_MODELS = {"improved_gnn_lstm_res", "improved_gnn_lstm_attn_adj_resbn"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical protocol12 seven-experiment tables.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--out-root", default="results/canonical_protocol12_seven_experiments")
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--include-v3", action="store_true")
    parser.add_argument("--feature-sets", default="acc16_hr,acc16_gyro,acc16_gyro_hr")
    parser.add_argument("--baseline-models", default=",".join(sorted(BASELINE_MODELS)), help="Comma-separated baseline models to include, or all.")
    parser.add_argument("--v3-models", default=",".join(sorted(V3_MODELS)), help="Comma-separated v3 deep models to include, or all.")
    parser.add_argument("--require-v3-complete", action="store_true")
    parser.add_argument(
        "--allow-clean-reference-exp3-exp6",
        action="store_true",
        help="Write clean-reference placeholder tables for Exp 3 and Exp 6. Without this, the runner refuses to call the suite complete.",
    )
    return parser.parse_args()


def parse_csv(value: str, default: set[str]) -> set[str]:
    items = {x.strip() for x in str(value).replace(";", ",").split(",") if x.strip()}
    if not items or items == {"all"}:
        return set(default)
    return items


def read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists() and path.stat().st_size:
            return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def path_context(path: Path, results_root: Path) -> dict[str, str]:
    parts = path.resolve().relative_to(results_root.resolve()).parts
    ctx = {
        "result_set": parts[0] if parts else "",
        "dataset": "",
        "feature_set": "",
        "window_type": "",
        "protocol": "",
        "family": "",
        "model": "",
        "eval_unit": "",
    }
    if "core_comparison" in parts:
        i = parts.index("core_comparison")
        if len(parts) > i + 5:
            ctx.update({
                "dataset": parts[i + 1],
                "feature_set": parts[i + 2],
                "window_type": parts[i + 3],
                "protocol": parts[i + 4],
                "family": parts[i + 5],
            })
    if "deep" in parts:
        j = parts.index("deep")
        if len(parts) > j + 3:
            ctx["model"] = parts[j + 2]
            ctx["eval_unit"] = parts[j + 3]
    elif "baselines" in parts:
        ctx["family"] = "baseline"
    return ctx


def normalize_summary(path: Path, results_root: Path) -> pd.DataFrame:
    df = read_csv(path)
    if df.empty:
        return df
    ctx = path_context(path, results_root)
    for key, value in ctx.items():
        if key not in df.columns:
            df[key] = value
        else:
            df[key] = df[key].fillna("").astype(str)
            df.loc[df[key] == "", key] = value
    if "eval_protocol" in df.columns:
        df["protocol"] = df["protocol"].fillna("").astype(str)
        df.loc[df["protocol"] == "", "protocol"] = df["eval_protocol"].astype(str)
    if "model_family" in df.columns:
        df["family"] = df["model_family"].replace({"baselines": "baseline", "deep": "deep"})
    if "model" not in df.columns or df["model"].fillna("").eq("").all():
        df["model"] = ctx["model"]
    df.loc[df["model"].isin(BASELINE_MODELS), "family"] = "baseline"
    df.loc[df["model"].isin(V3_MODELS), "family"] = "deep"
    df["artifact_dir"] = str(path.parent)
    return df


def normalize_folds(path: Path, results_root: Path) -> pd.DataFrame:
    df = read_csv(path)
    if df.empty:
        return df
    ctx = path_context(path, results_root)
    for key, value in ctx.items():
        if key not in df.columns:
            df[key] = value
        else:
            df[key] = df[key].fillna("").astype(str)
            df.loc[df[key] == "", key] = value
    if "eval_protocol" in df.columns:
        df["protocol"] = df["protocol"].fillna("").astype(str)
        df.loc[df["protocol"] == "", "protocol"] = df["eval_protocol"].astype(str)
    if "test_subject" not in df.columns and "fold_subject" in df.columns:
        df["test_subject"] = df["fold_subject"]
    return df


def load_tables(
    results_root: Path,
    feature_sets: set[str],
    include_baselines: bool,
    include_v3: bool,
    baseline_models: set[str],
    v3_models: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_sets = []
    if include_baselines:
        result_sets.append("canonical_protocol_only")
    if include_v3:
        result_sets.append("canonical_protocol_only_v3")
    summaries, folds = [], []
    for rs in result_sets:
        base = results_root / rs
        for path in base.rglob("metrics_summary.csv"):
            part = normalize_summary(path, results_root)
            if not part.empty:
                summaries.append(part)
        for path in base.rglob("metrics_by_fold.csv"):
            part = normalize_folds(path, results_root)
            if not part.empty:
                folds.append(part)
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    fold = pd.concat(folds, ignore_index=True) if folds else pd.DataFrame()
    if not summary.empty:
        summary = summary[summary["feature_set"].astype(str).isin(feature_sets)]
        keep_model = summary["model"].isin(baseline_models | v3_models)
        summary = summary[keep_model].copy()
    if not fold.empty:
        fold = fold[fold["feature_set"].astype(str).isin(feature_sets)]
        fold = fold[fold["model"].isin(baseline_models | v3_models)].copy()
    return summary, fold


def load_predictions(summary: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, row in summary.iterrows():
        artifact = Path(str(row["artifact_dir"]))
        if str(row.get("family", "")) == "baseline":
            pred = artifact / f"predictions_{row['model']}.csv"
        else:
            pred = artifact / "predictions.csv"
        df = read_csv(pred)
        if df.empty:
            continue
        for key in ["result_set", "feature_set", "window_type", "protocol", "family", "model", "eval_unit"]:
            df[key] = row.get(key, "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def has_real_table(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        df = pd.read_csv(path, nrows=5)
    except Exception:
        return False
    return not df.empty and "note" not in df.columns


def exp1(summary: pd.DataFrame) -> pd.DataFrame:
    cols = ["result_set", "feature_set", "window_type", "protocol", "family", "model", "accuracy", "balanced_accuracy", "macro_f1"]
    rows = summary[[c for c in cols if c in summary.columns]].copy()
    pivot = summary.pivot_table(index=["result_set", "feature_set", "window_type", "family", "model"], columns="protocol", values="accuracy", aggfunc="first")
    gaps = []
    for idx, vals in pivot.iterrows():
        gap = vals.get("random_holdout", np.nan) - vals.get("loso", np.nan)
        gaps.append((*idx, gap))
    gap_df = pd.DataFrame(gaps, columns=["result_set", "feature_set", "window_type", "family", "model", "holdout_minus_loso_accuracy"])
    return rows.merge(gap_df, on=["result_set", "feature_set", "window_type", "family", "model"], how="left")


def exp2(folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    loso = folds[folds["protocol"].astype(str).eq("loso")].copy()
    group = ["result_set", "feature_set", "window_type", "family", "model"]
    summary = loso.groupby(group, dropna=False).agg(
        folds=("fold", "nunique"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
    ).reset_index()
    pairs = []
    for (rs, fs, wt), part in loso.groupby(["result_set", "feature_set", "window_type"]):
        models = sorted(part["model"].unique())
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                aa = part[part["model"].eq(a)].sort_values("test_subject")["macro_f1"].to_numpy()
                bb = part[part["model"].eq(b)].sort_values("test_subject")["macro_f1"].to_numpy()
                n = min(len(aa), len(bb))
                if n:
                    pairs.append({
                        "result_set": rs,
                        "feature_set": fs,
                        "window_type": wt,
                        "model_a": a,
                        "model_b": b,
                        "macro_f1_mean_diff_a_minus_b": float(np.mean(aa[:n] - bb[:n])),
                        "paired_folds": int(n),
                    })
    return summary, pd.DataFrame(pairs)


def exp3_placeholder(summary: pd.DataFrame) -> pd.DataFrame:
    # True sensor-perturbation robustness requires rerunning checkpoints.
    rows = summary[summary["protocol"].astype(str).eq("loso")].copy()
    return rows[["result_set", "feature_set", "window_type", "family", "model", "accuracy", "macro_f1", "artifact_dir"]].assign(
        perturbation="clean_reference",
        note="Clean canonical reference only; sensor perturbation rerun requires checkpoint-based v3/baseline robustness runner.",
    )


def exp4(preds: pd.DataFrame) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame()
    proba_cols = [c for c in preds.columns if c.startswith("proba_")]
    if not proba_cols:
        return pd.DataFrame()
    rows = []
    for keys, part in preds.groupby(["result_set", "feature_set", "window_type", "protocol", "family", "model"], dropna=False):
        probs = part[proba_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(probs).all(axis=1)
        if not valid.any():
            rows.append({
                "result_set": keys[0], "feature_set": keys[1], "window_type": keys[2], "protocol": keys[3], "family": keys[4], "model": keys[5],
                "ece": math.nan,
                "mean_confidence": math.nan,
                "accuracy": math.nan,
                "macro_f1": math.nan,
                "status": "probabilities_not_available",
            })
            continue
        part = part.loc[valid].copy()
        probs = probs[valid]
        y_true = part["y_true_id"].to_numpy() if "y_true_id" in part.columns else part["y_true"].to_numpy()
        y_pred = part["y_pred_id"].to_numpy() if "y_pred_id" in part.columns else part["y_pred"].to_numpy()
        conf = probs.max(axis=1)
        correct = (y_true == y_pred).astype(float)
        bins = np.linspace(0, 1, 11)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
            if mask.any():
                ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
        rows.append({
            "result_set": keys[0], "feature_set": keys[1], "window_type": keys[2], "protocol": keys[3], "family": keys[4], "model": keys[5],
            "ece": float(ece),
            "mean_confidence": float(conf.mean()),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "status": "ok",
        })
    return pd.DataFrame(rows)


def exp5(folds: pd.DataFrame) -> pd.DataFrame:
    loso = folds[folds["protocol"].astype(str).eq("loso")].copy()
    cols = ["result_set", "feature_set", "window_type", "family", "model", "test_subject", "n_test_classes", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    return loso[[c for c in cols if c in loso.columns]].sort_values(["feature_set", "window_type", "model", "test_subject"])


def exp6_placeholder(summary: pd.DataFrame) -> pd.DataFrame:
    rows = summary[summary["protocol"].astype(str).eq("loso")].copy()
    return rows[["result_set", "feature_set", "window_type", "family", "model", "accuracy", "macro_f1", "artifact_dir"]].assign(
        calibration_fraction="0_clean_reference",
        note="Few-shot subject calibration requires fold-checkpoint fine-tuning; not derivable from aggregate CSV alone.",
    )


def exp7(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if preds.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows, group_rows = [], []
    for keys, part in preds.groupby(["result_set", "feature_set", "window_type", "protocol", "family", "model"], dropna=False):
        yt_label = part["y_true_label"].astype(str).map(LABEL_GROUPS)
        yp_label = part["y_pred_label"].astype(str).map(LABEL_GROUPS)
        mask = yt_label.notna() & yp_label.notna()
        if not mask.any():
            continue
        yt = yt_label[mask].to_numpy()
        yp = yp_label[mask].to_numpy()
        rows.append({
            "result_set": keys[0], "feature_set": keys[1], "window_type": keys[2], "protocol": keys[3], "family": keys[4], "model": keys[5],
            "fine_accuracy": float(part["correct"].astype(str).isin(["True", "true", "1"]).mean()) if "correct" in part else math.nan,
            "group_accuracy": float(accuracy_score(yt, yp)),
            "group_macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        })
        for group in sorted(set(yt)):
            gm = yt == group
            group_rows.append({
                "result_set": keys[0], "feature_set": keys[1], "window_type": keys[2], "protocol": keys[3], "family": keys[4], "model": keys[5],
                "activity_group": group,
                "support": int(gm.sum()),
                "recall": float((yp[gm] == group).mean()) if gm.any() else math.nan,
            })
    return pd.DataFrame(rows), pd.DataFrame(group_rows)


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    out_root = Path(args.out_root)
    feature_sets = {x.strip() for x in args.feature_sets.split(",") if x.strip()}
    baseline_models = parse_csv(args.baseline_models, BASELINE_MODELS)
    v3_models = parse_csv(args.v3_models, V3_MODELS)
    unknown_baselines = sorted(baseline_models - BASELINE_MODELS)
    unknown_v3 = sorted(v3_models - V3_MODELS)
    if unknown_baselines:
        raise SystemExit("Unknown baseline model(s): " + ", ".join(unknown_baselines))
    if unknown_v3:
        raise SystemExit("Unknown v3 model(s): " + ", ".join(unknown_v3))

    summary, folds = load_tables(results_root, feature_sets, args.include_baselines, args.include_v3, baseline_models, v3_models)
    if summary.empty:
        raise SystemExit("No canonical summaries found for requested selection.")
    if args.require_v3_complete and args.include_v3:
        expected = {(fs, p, m) for fs in feature_sets for p in ["loso", "random_holdout"] for m in v3_models}
        actual = set(
            tuple(x)
            for x in summary[summary["model"].isin(v3_models)][["feature_set", "protocol", "model"]].drop_duplicates().to_numpy()
        )
        missing = sorted(expected - actual)
        if missing:
            raise SystemExit("Missing v3 result groups: " + json.dumps(missing, indent=2))

    preds = load_predictions(summary)
    tables = out_root / "manuscript_tables"
    tables.mkdir(parents=True, exist_ok=True)
    exp3_real_path = tables / "table_exp3_robustness.csv"
    exp6_real_path = tables / "table_exp6_few_shot_calibration.csv"
    exp3_real_exists = has_real_table(exp3_real_path)
    exp6_real_exists = has_real_table(exp6_real_path)

    write_table(exp1(summary), tables / "table_exp1_leakage_control.csv")
    exp2_summary, exp2_pairs = exp2(folds)
    write_table(exp2_summary, tables / "table_exp2_statistical_reliability.csv")
    write_table(exp2_pairs, tables / "table_exp2_pairwise_comparisons.csv")
    if args.allow_clean_reference_exp3_exp6 and not exp3_real_exists:
        write_table(exp3_placeholder(summary), tables / "table_exp3_robustness_clean_reference.csv")
    write_table(exp4(preds), tables / "table_exp4_calibration.csv")
    write_table(exp5(folds), tables / "table_exp5_subject_failure.csv")
    if args.allow_clean_reference_exp3_exp6 and not exp6_real_exists:
        write_table(exp6_placeholder(summary), tables / "table_exp6_few_shot_calibration_clean_reference.csv")
    exp7_compare, exp7_groups = exp7(preds)
    write_table(exp7_compare, tables / "table_exp7_fine_vs_group.csv")
    write_table(exp7_groups, tables / "table_exp7_health_groups.csv")

    manifest = {
        "results_root": str(results_root),
        "out_root": str(out_root),
        "feature_sets": sorted(feature_sets),
        "include_baselines": args.include_baselines,
        "include_v3": args.include_v3,
        "baseline_models": sorted(baseline_models),
        "v3_models": sorted(v3_models),
        "summary_rows": int(len(summary)),
        "fold_rows": int(len(folds)),
        "prediction_rows": int(len(preds)),
        "exp3_status": "real_table_present" if exp3_real_exists else ("clean_reference_written" if args.allow_clean_reference_exp3_exp6 else "blocked_requires_checkpoint_robustness_runner"),
        "exp6_status": "real_table_present" if exp6_real_exists else ("clean_reference_written" if args.allow_clean_reference_exp3_exp6 else "blocked_requires_checkpoint_few_shot_runner"),
    }
    (out_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not args.allow_clean_reference_exp3_exp6 and not (exp3_real_exists and exp6_real_exists):
        block = out_root / "BLOCKED_EXP3_EXP6.md"
        block.write_text(
            "# Canonical Seven-Experiment Suite Blocked\n\n"
            "Experiments 1, 2, 4, 5, and 7 were generated from canonical artifacts.\n\n"
            "Experiments 3 and 6 were not generated because clean-reference tables are not true robustness or few-shot calibration experiments.\n\n"
            "Required next implementation:\n"
            "- Exp 3: checkpoint-based sensor perturbation/noise/missing-data evaluation for v3 models plus baseline perturbation evaluation.\n"
            "- Exp 6: checkpoint-based held-out-subject calibration/fine-tuning for v3 models plus baseline calibration protocol.\n\n",
            encoding="utf-8",
        )
    elif exp3_real_exists and exp6_real_exists:
        block = out_root / "BLOCKED_EXP3_EXP6.md"
        if block.exists():
            block.unlink()
    print("Wrote canonical seven-experiment tables to", tables)
    print(json.dumps(manifest, indent=2))
    if not args.allow_clean_reference_exp3_exp6 and not (exp3_real_exists and exp6_real_exists):
        raise SystemExit(
            "Strict mode: Exp 3 and Exp 6 require true checkpoint-based reruns. "
            f"Generated available tables and wrote {out_root / 'BLOCKED_EXP3_EXP6.md'}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
