#!/usr/bin/env python3
"""Experiment 2: Statistical reliability and model ranking stability."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.publication_common import base_parser, init_experiment
from src.publication.data import load_processed_dataset
from src.publication.models_registry import get_model_factory
from src.publication.train_eval import run_evaluation
from src.publication.statistics import (
    mean_ci95, wilcoxon_signed_rank, cohens_d, cohens_d_one_sample, wilcoxon_one_sample,
    bootstrap_mean_diff_ci, rank_stability, format_rank_stability,
)
from src.publication.outputs import save_csv, save_json, save_markdown_table, copy_to_manuscript
from src.publication.validation import validate_required_columns


EXP = "experiment_2_statistical_reliability"
SUMMARY_COLS = [
    "Dataset", "Model", "Accuracy_Mean", "Accuracy_SD",
    "Accuracy_CI95_Lower", "Accuracy_CI95_Upper",
    "Macro_F1_Mean", "Macro_F1_SD", "Macro_F1_CI95_Lower", "Macro_F1_CI95_Upper",
    "Effect_Size", "Wilcoxon_P_Value", "Rank_Stability",
]
PAIR_COLS = [
    "Dataset", "Metric", "Model_A", "Model_B", "Mean_Difference",
    "Bootstrap_CI95_Lower", "Bootstrap_CI95_Upper", "Effect_Size", "Wilcoxon_P_Value",
]


def main():
    args = base_parser("Experiment 2: Statistical reliability").parse_args()
    cfg, out_dir, log = init_experiment(EXP, args)
    tables_dir, _ = Path(cfg["_root"]) / cfg["results_root"] / "manuscript_tables", None
    tables_dir = Path(cfg["_root"]) / cfg["results_root"] / "manuscript_tables"

    summary_rows, pair_rows = [], []
    fold_scores_by_ds: dict[str, dict[str, list[float]]] = {}

    for ds in cfg["datasets"]:
        max_w = cfg.get("_max_windows_per_subject") or (cfg.get("hhar_max_windows_per_subject") if ds == "hhar" and not cfg["_smoke"] else None)
        data = load_processed_dataset(ds, window_type="overlapping", max_windows_per_subject=max_w, seed=cfg["seed"])
        X, y, subj = data["X"], data["y"], data["subjects"]
        fold_scores_by_ds[ds] = {}
        chance_acc = 1.0 / data["n_classes"]

        for model_name in cfg["models"]:
            factory, use_adj, mtype = get_model_factory(model_name, ds, data["n_classes"], X.shape[1:])
            res = run_evaluation(
                model_factory=factory, use_adj=use_adj, model_type=mtype,
                X=X, y=y, subjects=subj, dataset=ds, protocol="loso",
                cfg=cfg, seed=cfg["seed"], max_folds=cfg.get("_max_folds"),
            )
            accs = [f["accuracy"] for f in res["fold_metrics"]]
            f1s = [f["macro_f1"] for f in res["fold_metrics"]]
            fold_scores_by_ds[ds][model_name] = accs

            acc_arr = np.array(accs)
            am, alo, ahi = mean_ci95(acc_arr)
            fm, flo, fhi = mean_ci95(np.array(f1s))
            rs = rank_stability({m: fold_scores_by_ds[ds][m] for m in fold_scores_by_ds[ds]})
            summary_rows.append({
                "Dataset": ds, "Model": model_name,
                "Accuracy_Mean": am, "Accuracy_SD": float(np.std(accs, ddof=1) if len(accs) > 1 else 0),
                "Accuracy_CI95_Lower": alo, "Accuracy_CI95_Upper": ahi,
                "Macro_F1_Mean": fm, "Macro_F1_SD": float(np.std(f1s, ddof=1) if len(f1s) > 1 else 0),
                "Macro_F1_CI95_Lower": flo, "Macro_F1_CI95_Upper": fhi,
                "Effect_Size": cohens_d_one_sample(acc_arr, chance_acc),
                "Wilcoxon_P_Value": wilcoxon_one_sample(acc_arr, chance_acc),
                "Rank_Stability": format_rank_stability(rs[model_name], len(accs)),
            })
            save_json(out_dir / f"fold_metrics_{ds}_{model_name}.json", res["fold_metrics"])

        models = list(fold_scores_by_ds[ds].keys())
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                ma, mb = models[i], models[j]
                a = np.array(fold_scores_by_ds[ds][ma])
                b = np.array(fold_scores_by_ds[ds][mb])
                md, blo, bhi = bootstrap_mean_diff_ci(a, b, seed=cfg["seed"])
                pair_rows.append({
                    "Dataset": ds, "Metric": "accuracy", "Model_A": ma, "Model_B": mb,
                    "Mean_Difference": md, "Bootstrap_CI95_Lower": blo, "Bootstrap_CI95_Upper": bhi,
                    "Effect_Size": cohens_d(a, b), "Wilcoxon_P_Value": wilcoxon_signed_rank(a, b),
                })

    validate_required_columns(summary_rows, SUMMARY_COLS[:13], "statistical_summary")
    save_csv(out_dir / "statistical_reliability_summary.csv", summary_rows, SUMMARY_COLS)
    save_csv(out_dir / "pairwise_comparisons.csv", pair_rows, PAIR_COLS)
    if summary_rows:
        save_markdown_table(out_dir / "statistical_reliability_summary.md", summary_rows, SUMMARY_COLS)
    if (out_dir / "statistical_reliability_summary.csv").exists():
        copy_to_manuscript(out_dir / "statistical_reliability_summary.csv", tables_dir, "table_exp2_statistical_reliability.csv")
    if (out_dir / "pairwise_comparisons.csv").exists():
        copy_to_manuscript(out_dir / "pairwise_comparisons.csv", tables_dir, "table_exp2_pairwise_comparisons.csv")
    log.info("Experiment 2 complete → %s", out_dir)


if __name__ == "__main__":
    main()
