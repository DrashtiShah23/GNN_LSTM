#!/usr/bin/env python3
"""Experiment 5: Subject-level generalization failure analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.publication_common import base_parser, init_experiment
from src.publication.data import load_processed_dataset, subject_feature_summary
from src.publication.models_registry import get_model_factory
from src.publication.train_eval import run_evaluation
from src.publication.metrics import aggregate_subject_metrics
from src.publication.outputs import save_csv, save_json, copy_to_manuscript
from src.publication.plots import subject_activity_heatmap, save_confusion_matrix
from src.publication.validation import validate_required_columns


EXP = "experiment_5_subject_failure_analysis"
COLS = [
    "Subject", "Dataset", "Model", "Accuracy", "Macro_F1", "Balanced_Accuracy",
    "Worst_Activity", "Dominant_Confusion", "Missingness", "Activity_Imbalance", "Sensor_Variability",
]


def main():
    args = base_parser("Experiment 5: Subject failure analysis").parse_args()
    cfg, out_dir, log = init_experiment(EXP, args)
    fig_dir = Path(cfg["_root"]) / cfg["results_root"] / "manuscript_figures"
    tables_dir = Path(cfg["_root"]) / cfg["results_root"] / "manuscript_tables"

    rows = []
    for ds in cfg["datasets"]:
        max_w = cfg.get("_max_windows_per_subject") or (cfg.get("hhar_max_windows_per_subject") if ds == "hhar" and not cfg["_smoke"] else None)
        data = load_processed_dataset(ds, window_type="overlapping", max_windows_per_subject=max_w, seed=cfg["seed"])
        X, y, subj = data["X"], data["y"], data["subjects"]
        subj_feats = subject_feature_summary(X, y, subj)
        label_names = data["label_names"]

        for model_name in cfg["models"]:
            factory, use_adj, mtype = get_model_factory(model_name, ds, data["n_classes"], X.shape[1:])
            res = run_evaluation(
                model_factory=factory, use_adj=use_adj, model_type=mtype,
                X=X, y=y, subjects=subj, dataset=ds, protocol="loso",
                cfg=cfg, seed=cfg["seed"], max_folds=cfg.get("_max_folds"),
            )
            n_classes = len(label_names)
            subj_metrics = aggregate_subject_metrics(
                res["y_true"], res["y_pred"], res["subjects"].astype(str), n_classes=n_classes,
            )

            heat_rows, heat_cols = [], label_names
            heat_data = []

            for sm in subj_metrics:
                s = sm["subject"]
                sf = subj_feats.get(s, {})
                recalls = sm["per_class_recall"]
                worst_i = int(np.argmin(recalls)) if recalls else 0
                cm = np.array(sm["confusion_matrix"])
                dom = ""
                if cm.size:
                    flat = cm.copy()
                    np.fill_diagonal(flat, 0)
                    if flat.max() > 0:
                        ti, pi = np.unravel_index(flat.argmax(), flat.shape)
                        dom = f"{label_names[ti]}→{label_names[pi]}"

                rows.append({
                    "Subject": s, "Dataset": ds, "Model": model_name,
                    "Accuracy": sm["accuracy"], "Macro_F1": sm["macro_f1"],
                    "Balanced_Accuracy": sm["balanced_accuracy"],
                    "Worst_Activity": label_names[worst_i] if worst_i < len(label_names) else str(worst_i),
                    "Dominant_Confusion": dom,
                    "Missingness": sf.get("missingness"),
                    "Activity_Imbalance": sf.get("activity_imbalance"),
                    "Sensor_Variability": sf.get("sensor_variance"),
                })
                heat_data.append(recalls)

            if heat_data:
                subject_activity_heatmap(
                    np.array(heat_data), [sm["subject"] for sm in subj_metrics], heat_cols,
                    f"{ds} {model_name} per-class recall", fig_dir / f"fig_exp5_heatmap_{ds}_{model_name}.png",
                )
            save_confusion_matrix(
                res["y_true"], res["y_pred"], label_names,
                f"{ds} {model_name} aggregated CM", out_dir / f"cm_{ds}_{model_name}.png",
            )

    validate_required_columns(rows, COLS, "subject_failure")
    save_csv(out_dir / "subject_failure_summary.csv", rows, COLS)
    copy_to_manuscript(out_dir / "subject_failure_summary.csv", tables_dir, "table_exp5_subject_failure.csv")
    log.info("Experiment 5 complete → %s", out_dir)


if __name__ == "__main__":
    main()
