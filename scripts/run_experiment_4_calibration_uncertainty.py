#!/usr/bin/env python3
"""Experiment 4: Calibration, uncertainty, and selective prediction."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.publication_common import base_parser, init_experiment
from src.publication.data import load_processed_dataset
from src.publication.models_registry import get_model_factory
from src.publication.train_eval import run_evaluation
from src.publication.calibration import full_calibration_report
from src.publication.outputs import save_csv, save_json, copy_to_manuscript
from src.publication.plots import reliability_diagram
from src.publication.validation import validate_probabilities, validate_required_columns


EXP = "experiment_4_calibration_uncertainty"
COLS = [
    "Dataset", "Model", "ECE", "Brier_Score", "NLL",
    "Accuracy_At_90_Coverage", "Accuracy_At_80_Coverage", "Accuracy_At_70_Coverage",
    "Macro_F1_At_90_Coverage", "Macro_F1_At_80_Coverage", "Macro_F1_At_70_Coverage",
]


def main():
    args = base_parser("Experiment 4: Calibration").parse_args()
    cfg, out_dir, log = init_experiment(EXP, args)
    fig_dir = Path(cfg["_root"]) / cfg["results_root"] / "manuscript_figures"
    tables_dir = Path(cfg["_root"]) / cfg["results_root"] / "manuscript_tables"

    rows = []
    for ds in cfg["datasets"]:
        max_w = cfg.get("_max_windows_per_subject") or (cfg.get("hhar_max_windows_per_subject") if ds == "hhar" and not cfg["_smoke"] else None)
        data = load_processed_dataset(ds, window_type="overlapping", max_windows_per_subject=max_w, seed=cfg["seed"])
        X, y, subj = data["X"], data["y"], data["subjects"]

        for model_name in cfg["models"]:
            factory, use_adj, mtype = get_model_factory(model_name, ds, data["n_classes"], X.shape[1:])
            res = run_evaluation(
                model_factory=factory, use_adj=use_adj, model_type=mtype,
                X=X, y=y, subjects=subj, dataset=ds, protocol="loso",
                cfg=cfg, seed=cfg["seed"], max_folds=cfg.get("_max_folds"),
            )
            yt, yp, pr = res["y_true"], res["y_pred"], res["probs"]
            validate_probabilities(pr)
            cal = full_calibration_report(yt, pr, yp)
            rows.append({
                "Dataset": ds, "Model": model_name,
                "ECE": cal["ece"], "Brier_Score": cal["brier_score"], "NLL": cal["nll"],
                "Accuracy_At_90_Coverage": cal["accuracy_at_90_coverage"],
                "Accuracy_At_80_Coverage": cal["accuracy_at_80_coverage"],
                "Accuracy_At_70_Coverage": cal["accuracy_at_70_coverage"],
                "Macro_F1_At_90_Coverage": cal["macro_f1_at_90_coverage"],
                "Macro_F1_At_80_Coverage": cal["macro_f1_at_80_coverage"],
                "Macro_F1_At_70_Coverage": cal["macro_f1_at_70_coverage"],
            })
            save_json(out_dir / f"calibration_{ds}_{model_name}.json", cal)
            reliability_diagram(yt, pr, f"{ds} {model_name}", fig_dir / f"fig_exp4_reliability_{ds}_{model_name}.png")

    validate_required_columns(rows, COLS, "calibration")
    save_csv(out_dir / "calibration_summary.csv", rows, COLS)
    copy_to_manuscript(out_dir / "calibration_summary.csv", tables_dir, "table_exp4_calibration.csv")
    log.info("Experiment 4 complete → %s", out_dir)


if __name__ == "__main__":
    main()
