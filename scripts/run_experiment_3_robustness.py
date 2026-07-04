#!/usr/bin/env python3
"""Experiment 3: Robustness to sensor failure, noise, and missing data.

Train once per LOSO fold on clean data, then evaluate the same models under
each test-time perturbation (no retraining).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.publication_common import base_parser, init_experiment
from src.publication.data import load_processed_dataset
from src.publication.models_registry import get_model_factory
from src.publication.train_eval import _make_dataset, train_model, predict
from src.publication.splits import loso_fold_splits, subject_val_split, assert_loso_no_leakage
from src.publication.metrics import most_affected_class_name
from src.publication.perturbations import (
    PerturbationType, SEVERITY_LEVELS, apply_perturbation_x, drop_random_windows,
)
from src.publication.outputs import save_csv, copy_to_manuscript
from src.publication.plots import degradation_curve
from src.train import get_device, set_seed
from torch.utils.data import DataLoader, Subset


EXP = "experiment_3_robustness"
COLS = [
    "Dataset", "Perturbation", "Severity", "Model",
    "Clean_Accuracy", "Perturbed_Accuracy", "Accuracy_Drop",
    "Clean_Macro_F1", "Perturbed_Macro_F1", "Macro_F1_Drop", "Most_Affected_Class",
]

HR_PERTS = {PerturbationType.REMOVE_HEART_RATE, PerturbationType.MISSING_HEART_RATE}


def train_loso_once(X, y, subj, ds, cfg, factory, use_adj, mtype):
    """Train one model per LOSO fold on clean data; return states and clean preds."""
    device = get_device()
    torch_ds, eval_subj = _make_dataset(mtype, X, y, subj, ds, cfg["sequence"]["seq_len"])
    batch_size = cfg["training"]["batch_size"]
    fold_states, fold_test_subj = [], []
    all_true, all_pred = [], []

    for tr_idx, te_idx, test_subj, fi in loso_fold_splits(eval_subj, cfg.get("_max_folds")):
        assert_loso_no_leakage(eval_subj, tr_idx, te_idx, test_subj)
        set_seed(cfg["seed"] + fi)
        tr, val_idx = subject_val_split(tr_idx, eval_subj, cfg["training"]["val_fraction"])
        loaders = (
            DataLoader(Subset(torch_ds, tr), batch_size=batch_size, shuffle=True),
            DataLoader(Subset(torch_ds, val_idx), batch_size=batch_size, shuffle=False),
            DataLoader(Subset(torch_ds, te_idx), batch_size=batch_size, shuffle=False),
        )
        model = factory().to(device)
        model = train_model(model, loaders[0], loaders[1], use_adj=use_adj, cfg_train=cfg["training"], device=device)
        yt, yp, _ = predict(model, loaders[2], device, use_adj)
        fold_states.append(copy.deepcopy(model.state_dict()))
        fold_test_subj.append(test_subj)
        all_true.extend(yt)
        all_pred.extend(yp)

    return fold_states, fold_test_subj, np.array(all_true), np.array(all_pred)


def predict_loso_with_states(X, y, subj, ds, cfg, factory, use_adj, mtype, fold_states, fold_test_subj):
    """Evaluate saved LOSO models on (possibly perturbed) data for the same test subjects."""
    device = get_device()
    torch_ds, eval_subj = _make_dataset(mtype, X, y, subj, ds, cfg["sequence"]["seq_len"])
    batch_size = cfg["training"]["batch_size"]
    all_true, all_pred = [], []

    for state, test_subj in zip(fold_states, fold_test_subj):
        te_idx = np.where(eval_subj == test_subj)[0]
        if len(te_idx) == 0:
            continue
        loader = DataLoader(Subset(torch_ds, te_idx), batch_size=batch_size, shuffle=False)
        model = factory().to(device)
        model.load_state_dict(state)
        yt, yp, _ = predict(model, loader, device, use_adj)
        all_true.extend(yt)
        all_pred.extend(yp)

    return np.array(all_true), np.array(all_pred)


def merge_dataset_rows(out_csv: Path, new_rows: list[dict], datasets: list[str]) -> list[dict]:
    """Keep other-dataset rows when a single-dataset run overwrites the experiment CSV."""
    if not out_csv.exists():
        return new_rows
    prev = pd.read_csv(out_csv)
    if "Dataset" not in prev.columns:
        return new_rows
    keep = {d.lower() for d in datasets}
    other = prev[~prev["Dataset"].astype(str).str.lower().isin(keep)]
    if len(other) == 0:
        return new_rows
    return pd.concat([other, pd.DataFrame(new_rows)], ignore_index=True).to_dict(orient="records")


def main():
    args = base_parser("Experiment 3: Robustness").parse_args()
    cfg, out_dir, log = init_experiment(EXP, args)
    fig_dir = Path(cfg["_root"]) / cfg["results_root"] / "manuscript_figures"
    tables_dir = Path(cfg["_root"]) / cfg["results_root"] / "manuscript_tables"

    rows = []
    perturbations = list(PerturbationType)
    severities = SEVERITY_LEVELS
    if cfg["_smoke"]:
        perturbations = [PerturbationType.GAUSSIAN_NOISE]
        severities = ("low",)

    for ds in cfg["datasets"]:
        max_w = cfg.get("_max_windows_per_subject") or (cfg.get("hhar_max_windows_per_subject") if ds == "hhar" and not cfg["_smoke"] else None)
        data = load_processed_dataset(ds, window_type="overlapping", max_windows_per_subject=max_w, seed=cfg["seed"])
        X, y, subj = data["X"], data["y"], data["subjects"]
        label_names = data["label_names"]

        for model_name in cfg["models"]:
            factory, use_adj, mtype = get_model_factory(model_name, ds, data["n_classes"], X.shape[1:])
            fold_states, fold_test_subj, yt_clean, yp_clean = train_loso_once(
                X, y, subj, ds, cfg, factory, use_adj, mtype,
            )
            clean_acc = float((yt_clean == yp_clean).mean())
            clean_f1 = float(f1_score(yt_clean, yp_clean, average="macro", zero_division=0))

            for pert in perturbations:
                for sev in severities:
                    if pert in HR_PERTS:
                        rows.append({
                            "Dataset": ds, "Perturbation": pert.value, "Severity": sev, "Model": model_name,
                            # Use not_applicable (not "N/A") so pandas does not coerce to NaN on reload.
                            "Clean_Accuracy": clean_acc, "Perturbed_Accuracy": "not_applicable", "Accuracy_Drop": "not_applicable",
                            "Clean_Macro_F1": clean_f1, "Perturbed_Macro_F1": "not_applicable", "Macro_F1_Drop": "not_applicable",
                            "Most_Affected_Class": "not_applicable",
                        })
                        continue

                    if pert == PerturbationType.REMOVE_RANDOM_WINDOWS:
                        Xp, yp, subp = drop_random_windows(X, y, subj, sev, seed=cfg["seed"])
                    else:
                        Xp = apply_perturbation_x(X, pert, sev, ds, seed=cfg["seed"])
                        yp, subp = y, subj

                    yt_pred, yp_pred = predict_loso_with_states(
                        Xp, yp, subp, ds, cfg, factory, use_adj, mtype, fold_states, fold_test_subj,
                    )
                    n = min(len(yt_pred), len(yp_pred))
                    yt_n, yp_n = yt_pred[:n], yp_pred[:n]
                    pert_acc = float((yt_n == yp_n).mean()) if n else 0.0
                    pert_f1 = float(f1_score(yt_n, yp_n, average="macro", zero_division=0)) if n else 0.0

                    n_aff = min(len(yt_clean), len(yp_clean), len(yp_pred))
                    if len(yt_clean) == len(yp_pred) and n_aff > 0:
                        affected = most_affected_class_name(yt_clean, yp_clean, yp_pred, label_names)
                    elif n > 0:
                        # Window-drop changes length; score F1 drop on perturbed labels.
                        yc_trunc = yp_clean[:n]
                        affected = most_affected_class_name(yt_n, yc_trunc, yp_n, label_names)
                    else:
                        affected = ""

                    rows.append({
                        "Dataset": ds, "Perturbation": pert.value, "Severity": sev, "Model": model_name,
                        "Clean_Accuracy": clean_acc, "Perturbed_Accuracy": pert_acc,
                        "Accuracy_Drop": clean_acc - pert_acc,
                        "Clean_Macro_F1": clean_f1, "Perturbed_Macro_F1": pert_f1,
                        "Macro_F1_Drop": clean_f1 - pert_f1,
                        "Most_Affected_Class": affected,
                    })

            drops = [r["Accuracy_Drop"] for r in rows if r["Model"] == model_name and r["Dataset"] == ds
                     and r["Perturbation"] == PerturbationType.GAUSSIAN_NOISE.value
                     and r["Accuracy_Drop"] != "not_applicable"]
            if drops:
                degradation_curve(list(severities), drops,
                                  f"{ds} {model_name} noise robustness",
                                  fig_dir / f"fig_exp3_robustness_{ds}_{model_name}.png")

    out_csv = out_dir / "robustness_summary.csv"
    rows = merge_dataset_rows(out_csv, rows, cfg["datasets"])
    save_csv(out_csv, rows, COLS)
    copy_to_manuscript(out_csv, tables_dir, "table_exp3_robustness.csv")
    log.info("Experiment 3 complete → %s", out_dir)


if __name__ == "__main__":
    main()
