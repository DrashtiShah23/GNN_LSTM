#!/usr/bin/env python3
"""Experiment 7: Health-relevant activity group analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.publication_common import base_parser, init_experiment
from src.publication.data import load_processed_dataset
from src.publication.models_registry import get_model_factory
from src.publication.train_eval import run_evaluation
from src.publication.activity_groups import load_activity_groups, build_group_mapping, collapse_to_groups, dominant_group_confusion
from src.publication.metrics import compute_full_metrics
from src.publication.outputs import save_csv, save_json, save_markdown_table, copy_to_manuscript
from src.publication.plots import save_confusion_matrix
from src.publication.validation import validate_required_columns


def merge_dataset_rows(path: Path, new_rows: list[dict], datasets: list[str]) -> list[dict]:
    """Keep other-dataset rows when a single-dataset run would otherwise wipe them."""
    if not path.exists() or not new_rows:
        return new_rows
    prev = pd.read_csv(path)
    if "Dataset" not in prev.columns:
        return new_rows
    keep = {d.lower() for d in datasets}
    other = prev[~prev["Dataset"].astype(str).str.lower().isin(keep)]
    if len(other) == 0:
        return new_rows
    return pd.concat([other, pd.DataFrame(new_rows)], ignore_index=True).to_dict(orient="records")


EXP = "experiment_7_health_group_analysis"
GROUP_COLS = [
    "Dataset", "Clinical_Activity_Group", "Included_Activities",
    "Sensitivity", "Specificity", "Macro_F1", "Main_Confusion",
]
COMPARE_COLS = [
    "Dataset", "Model", "Fine_Grained_Accuracy", "Group_Level_Accuracy",
    "Fine_Grained_Macro_F1", "Group_Level_Macro_F1",
]


def main():
    args = base_parser("Experiment 7: Health group analysis").parse_args()
    cfg, out_dir, log = init_experiment(EXP, args)
    tables_dir = Path(cfg["_root"]) / cfg["results_root"] / "manuscript_tables"
    fig_dir = Path(cfg["_root"]) / cfg["results_root"] / "manuscript_figures"

    ag_cfg = load_activity_groups(Path(cfg["_root"]) / cfg["activity_group_config"])
    group_rows, compare_rows = [], []
    unmapped_report = {}

    for ds in cfg["datasets"]:
        max_w = cfg.get("_max_windows_per_subject") or (cfg.get("hhar_max_windows_per_subject") if ds == "hhar" and not cfg["_smoke"] else None)
        data = load_processed_dataset(ds, window_type="overlapping", max_windows_per_subject=max_w, seed=cfg["seed"])
        X, y, subj = data["X"], data["y"], data["subjects"]
        label_names = data["label_names"]
        class_to_group, unmapped = build_group_mapping(ds, label_names, ag_cfg)
        unmapped_report[ds] = unmapped

        for model_name in cfg["models"]:
            factory, use_adj, mtype = get_model_factory(model_name, ds, data["n_classes"], X.shape[1:])
            res = run_evaluation(
                model_factory=factory, use_adj=use_adj, model_type=mtype,
                X=X, y=y, subjects=subj, dataset=ds, protocol="loso",
                cfg=cfg, seed=cfg["seed"], max_folds=cfg.get("_max_folds"),
            )
            fine = res["aggregate"]
            groups = sorted(set(class_to_group.values()))
            yg_true, yg_pred = collapse_to_groups(res["y_true"], res["y_pred"], class_to_group, groups)
            group_m = compute_full_metrics(yg_true, yg_pred, n_classes=len(groups))

            compare_rows.append({
                "Dataset": ds, "Model": model_name,
                "Fine_Grained_Accuracy": fine["accuracy"],
                "Group_Level_Accuracy": group_m["accuracy"],
                "Fine_Grained_Macro_F1": fine["macro_f1"],
                "Group_Level_Macro_F1": group_m["macro_f1"],
            })

            group_labels = groups
            save_confusion_matrix(
                yg_true, yg_pred, group_labels,
                f"{ds} {model_name} health groups",
                fig_dir / f"fig_exp7_health_cm_{ds}_{model_name}.png",
            )

            for gi, gname in enumerate(groups):
                acts = [label_names[i] for i, g in class_to_group.items() if g == gname]
                group_rows.append({
                    "Dataset": ds, "Clinical_Activity_Group": gname,
                    "Included_Activities": "; ".join(acts),
                    "Sensitivity": group_m["per_class_sensitivity"][gi] if gi < len(group_m["per_class_sensitivity"]) else None,
                    "Specificity": group_m["per_class_specificity"][gi] if gi < len(group_m["per_class_specificity"]) else None,
                    "Macro_F1": group_m["per_class_f1"][gi] if gi < len(group_m["per_class_f1"]) else None,
                    "Main_Confusion": dominant_group_confusion(group_m["confusion_matrix"], gi, group_labels),
                })

    group_path = out_dir / "health_group_metrics.csv"
    compare_path = out_dir / "fine_vs_group_comparison.csv"
    group_rows = merge_dataset_rows(group_path, group_rows, cfg["datasets"])
    compare_rows = merge_dataset_rows(compare_path, compare_rows, cfg["datasets"])

    validate_required_columns(compare_rows, COMPARE_COLS, "health_compare")
    save_csv(group_path, group_rows, GROUP_COLS)
    save_csv(compare_path, compare_rows, COMPARE_COLS)
    save_markdown_table(out_dir / "fine_vs_group_comparison.md", compare_rows, COMPARE_COLS)

    unmapped_path = out_dir / "unmapped_activities.json"
    if unmapped_path.exists():
        try:
            prev_unmapped = json.loads(unmapped_path.read_text())
            if isinstance(prev_unmapped, dict):
                for k, v in prev_unmapped.items():
                    if k not in unmapped_report:
                        unmapped_report[k] = v
        except json.JSONDecodeError:
            pass
    save_json(unmapped_path, unmapped_report)
    copy_to_manuscript(compare_path, tables_dir, "table_exp7_fine_vs_group.csv")
    copy_to_manuscript(group_path, tables_dir, "table_exp7_health_groups.csv")
    log.info("Experiment 7 complete → %s", out_dir)


if __name__ == "__main__":
    main()
