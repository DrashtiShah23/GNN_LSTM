#!/usr/bin/env python3
"""Experiment 6: Few-shot subject calibration."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.publication_common import base_parser, init_experiment
from src.publication.data import load_processed_dataset
from src.publication.models_registry import get_model_factory
from src.publication.train_eval import (
    _make_dataset, train_model, predict, finetune_model, build_sequence_subjects,
)
from src.publication.splits import loso_fold_splits, subject_val_split, assert_loso_no_leakage, calibration_test_split, assert_calibration_no_leakage
from src.publication.metrics import compute_full_metrics
from src.publication.outputs import save_csv, copy_to_manuscript
from src.publication.plots import degradation_curve
from src.publication.validation import validate_required_columns
from src.train import get_device, set_seed
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score


EXP = "experiment_6_few_shot_calibration"
COLS = [
    "Dataset", "Subject", "Calibration_Percentage", "Fine_Tuning_Strategy", "Model",
    "Uncalibrated_Accuracy", "Calibrated_Accuracy", "Accuracy_Improvement",
    "Uncalibrated_Macro_F1", "Calibrated_Macro_F1", "Macro_F1_Improvement",
]
CAL_PCTS = [0.0, 0.01, 0.05, 0.10]
STRATEGIES = ["none", "classifier_head_only", "full_model"]


def main():
    args = base_parser("Experiment 6: Few-shot calibration").parse_args()
    cfg, out_dir, log = init_experiment(EXP, args)
    fig_dir = Path(cfg["_root"]) / cfg["results_root"] / "manuscript_figures"
    tables_dir = Path(cfg["_root"]) / cfg["results_root"] / "manuscript_tables"

    if cfg["_smoke"]:
        CAL_PCTS_RUN = [0.0, 0.01]
        strategies_run = ["classifier_head_only"]
    else:
        CAL_PCTS_RUN = [0.0, 0.01, 0.05, 0.10]
        strategies_run = ["classifier_head_only", "full_model"]

    rows = []
    for ds in cfg["datasets"]:
        max_w = cfg.get("_max_windows_per_subject") or (cfg.get("hhar_max_windows_per_subject") if ds == "hhar" and not cfg["_smoke"] else None)
        data = load_processed_dataset(ds, window_type="overlapping", max_windows_per_subject=max_w, seed=cfg["seed"])
        X, y, subj = data["X"], data["y"], data["subjects"]

        for model_name in cfg["models"]:
            factory, use_adj, mtype = get_model_factory(model_name, ds, data["n_classes"], X.shape[1:])
            device = get_device()
            torch_ds, eval_subj = _make_dataset(mtype, X, y, subj, ds, cfg["sequence"]["seq_len"])
            batch_size = cfg["training"]["batch_size"]

            for tr_idx, te_idx, test_subj, fi in loso_fold_splits(eval_subj, cfg.get("_max_folds")):
                assert_loso_no_leakage(eval_subj, tr_idx, te_idx, test_subj)
                set_seed(cfg["seed"] + fi)
                tr, val_idx = subject_val_split(tr_idx, eval_subj, cfg["training"]["val_fraction"])
                train_loader = DataLoader(Subset(torch_ds, tr), batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(Subset(torch_ds, val_idx), batch_size=batch_size, shuffle=False)

                model = factory().to(device)
                model = train_model(model, train_loader, val_loader, use_adj=use_adj, cfg_train=cfg["training"], device=device)

                # Uncalibrated test on all held-out subject sequences
                te_loader = DataLoader(Subset(torch_ds, te_idx), batch_size=batch_size, shuffle=False)
                yt0, yp0, _ = predict(model, te_loader, device, use_adj)
                base_acc = accuracy_score(yt0, yp0)
                base_f1 = f1_score(yt0, yp0, average="macro", zero_division=0)

                rows.append({
                    "Dataset": ds, "Subject": str(test_subj), "Calibration_Percentage": 0.0,
                    "Fine_Tuning_Strategy": "none", "Model": model_name,
                    "Uncalibrated_Accuracy": base_acc, "Calibrated_Accuracy": base_acc,
                    "Accuracy_Improvement": 0.0,
                    "Uncalibrated_Macro_F1": base_f1, "Calibrated_Macro_F1": base_f1,
                    "Macro_F1_Improvement": 0.0,
                })

                for cal_pct in [p for p in CAL_PCTS_RUN if p > 0]:
                    cal_idx, test_idx = calibration_test_split(te_idx, cal_pct, seed=cfg["seed"])
                    assert_calibration_no_leakage(cal_idx, test_idx)

                    for strategy in strategies_run:
                        m = copy.deepcopy(model)
                        cal_loader = DataLoader(Subset(torch_ds, cal_idx), batch_size=batch_size, shuffle=True)
                        if strategy == "classifier_head_only":
                            m = finetune_model(m, cal_loader, use_adj=use_adj, strategy="classifier_head_only", cfg_train=cfg["training"], device=device)
                        else:
                            m = finetune_model(m, cal_loader, use_adj=use_adj, strategy="full_model", cfg_train=cfg["training"], device=device)
                        test_loader = DataLoader(Subset(torch_ds, test_idx), batch_size=batch_size, shuffle=False)
                        yt, yp, _ = predict(m, test_loader, device, use_adj)
                        cal_acc = accuracy_score(yt, yp)
                        cal_f1 = f1_score(yt, yp, average="macro", zero_division=0)
                        rows.append({
                            "Dataset": ds, "Subject": str(test_subj), "Calibration_Percentage": cal_pct,
                            "Fine_Tuning_Strategy": strategy, "Model": model_name,
                            "Uncalibrated_Accuracy": base_acc, "Calibrated_Accuracy": cal_acc,
                            "Accuracy_Improvement": cal_acc - base_acc,
                            "Uncalibrated_Macro_F1": base_f1, "Calibrated_Macro_F1": cal_f1,
                            "Macro_F1_Improvement": cal_f1 - base_f1,
                        })

            # Efficiency curve (mean improvement vs cal %)
            for strategy in strategies_run:
                imps = []
                for pct in [p for p in CAL_PCTS_RUN if p > 0]:
                    sub = [r for r in rows if r["Dataset"] == ds and r["Model"] == model_name
                           and r["Fine_Tuning_Strategy"] == strategy and r["Calibration_Percentage"] == pct]
                    if sub:
                        imps.append(float(np.mean([r["Accuracy_Improvement"] for r in sub])))
                if imps:
                    labels = [f"{int(p*100)}%" for p in [pct for pct in CAL_PCTS_RUN if pct > 0][:len(imps)]]
                    degradation_curve(labels, imps,
                                      f"{ds} {model_name} {strategy} calibration",
                                      fig_dir / f"fig_exp6_cal_{ds}_{model_name}_{strategy}.png")

    validate_required_columns(rows, COLS, "few_shot_calibration")
    save_csv(out_dir / "few_shot_calibration_summary.csv", rows, COLS)
    copy_to_manuscript(out_dir / "few_shot_calibration_summary.csv", tables_dir, "table_exp6_few_shot_calibration.csv")
    log.info("Experiment 6 complete → %s", out_dir)


if __name__ == "__main__":
    main()
