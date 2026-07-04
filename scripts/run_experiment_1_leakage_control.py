#!/usr/bin/env python3
"""Experiment 1: Leakage control — overlapping vs non-overlapping windows × holdout vs LOSO."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.publication_common import base_parser, init_experiment
from src.publication.config import results_dir
from src.publication.data import load_processed_dataset
from src.publication.models_registry import get_model_factory
from src.publication.train_eval import run_evaluation
from src.publication.outputs import save_csv, save_json, save_markdown_table, copy_to_manuscript
from src.publication.plots import grouped_bar_chart
from src.publication.validation import validate_required_columns


EXP = "experiment_1_leakage_control"
COLS = [
    "Dataset", "Window_Type", "Evaluation_Protocol", "Model",
    "Accuracy", "Macro_F1", "Balanced_Accuracy", "Leakage_Gap",
]


def main():
    args = base_parser("Experiment 1: Leakage control").parse_args()
    cfg, out_dir, log = init_experiment(EXP, args)
    tables_dir, fig_dir = Path(cfg["_root"]) / cfg["results_root"] / "manuscript_tables", Path(cfg["_root"]) / cfg["results_root"] / "manuscript_figures"

    rows = []
    loso_cache = {}

    for ds in cfg["datasets"]:
        max_w = cfg.get("_max_windows_per_subject")
        if ds == "hhar" and not cfg["_smoke"]:
            max_w = max_w or cfg.get("hhar_max_windows_per_subject")

        for window_type in cfg["window_types"]:
            data = load_processed_dataset(ds, window_type=window_type, max_windows_per_subject=max_w, seed=cfg["seed"])
            X, y, subj = data["X"], data["y"], data["subjects"]

            for model_name in cfg["models"]:
                factory, use_adj, mtype = get_model_factory(model_name, ds, data["n_classes"], X.shape[1:])
                log.info("%s | %s | %s | %s", ds, window_type, model_name, mtype)

                for protocol in cfg["evaluation_protocols"]:
                    tag = f"{ds}_{window_type}_{protocol}_{model_name}".lower()
                    meta_path = out_dir / "predictions" / f"{tag}_metadata.json"
                    if getattr(args, "resume", False) and meta_path.exists():
                        import json
                        with open(meta_path) as f:
                            meta = json.load(f)
                        res = {"aggregate": meta["aggregate"]}
                        log.info("Resume: skipping %s", tag)
                    else:
                        res = run_evaluation(
                            model_factory=factory,
                            use_adj=use_adj,
                            model_type=mtype,
                            X=X, y=y, subjects=subj,
                            dataset=ds,
                            protocol=protocol,
                            cfg=cfg,
                            seed=cfg["seed"],
                            max_folds=cfg.get("_max_folds"),
                        )
                        save_json(meta_path, {
                            **data["meta"],
                            "protocol": protocol,
                            "model": model_name,
                            "seed": cfg["seed"],
                            "aggregate": res["aggregate"],
                        })
                    key = (ds, window_type, model_name, protocol)
                    loso_cache[key] = res

                    rows.append({
                        "Dataset": ds,
                        "Window_Type": window_type,
                        "Evaluation_Protocol": protocol,
                        "Model": model_name,
                        "Accuracy": res["aggregate"]["accuracy"],
                        "Macro_F1": res["aggregate"]["macro_f1"],
                        "Balanced_Accuracy": res["aggregate"]["balanced_accuracy"],
                        "Leakage_Gap": None,
                    })

    # Leakage gaps: RH - LOSO per (dataset, window, model)
    for ds in cfg["datasets"]:
        for window_type in cfg["window_types"]:
            for model_name in cfg["models"]:
                rh = loso_cache.get((ds, window_type, model_name, "random_holdout"))
                lo = loso_cache.get((ds, window_type, model_name, "loso"))
                if rh and lo:
                    gap = rh["aggregate"]["accuracy"] - lo["aggregate"]["accuracy"]
                    for r in rows:
                        if (r["Dataset"] == ds and r["Window_Type"] == window_type
                                and r["Model"] == model_name and r["Evaluation_Protocol"] == "random_holdout"):
                            r["Leakage_Gap"] = gap

    validate_required_columns(rows, COLS, "leakage_control")
    save_csv(out_dir / "leakage_control_summary.csv", rows, COLS)
    save_markdown_table(out_dir / "leakage_control_summary.md", rows, COLS)
    copy_to_manuscript(out_dir / "leakage_control_summary.csv", tables_dir, "table_exp1_leakage_control.csv")

    chart_rows = [r for r in rows if r["Evaluation_Protocol"] in ("random_holdout", "loso") and r["Window_Type"] == "overlapping"]
    grouped_bar_chart(
        chart_rows, x_col="Model", y_col="Accuracy", hue_col="Evaluation_Protocol",
        facet_col="Dataset", title="Leakage Control: Accuracy by Protocol",
        out_path=fig_dir / "fig_exp1_leakage_grouped_bar.png",
    )
    copy_to_manuscript(fig_dir / "fig_exp1_leakage_grouped_bar.png", out_dir, "leakage_grouped_bar.png")
    log.info("Experiment 1 complete → %s", out_dir)


if __name__ == "__main__":
    main()
