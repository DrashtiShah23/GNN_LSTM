#!/usr/bin/env python
"""Compute parameter counts for canonical deep models without training."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase2_repo_deep_parallel_v2 import (
    build_model,
    get_dataset_meta,
    load_processed_dataset,
    model_parameter_profile,
    parse_csv_arg,
    resolve_models,
)


DEFAULT_FEATURE_SETS = ["acc16_hr", "acc16_gyro", "acc16_gyro_hr"]
DEFAULT_MODELS = [
    "lstm",
    "gnn",
    "gnn_lstm",
    "improved_gnn_lstm",
    "improved_gnn_lstm_res",
    "gnn_flatten_lstm",
    "gnn_learnable_adj",
    "gnn_attention_adj",
    "improved_gnn_lstm_attn_adj",
    "improved_gnn_lstm_attn_adj_resbn",
    "cnn",
]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile canonical deep model parameter counts")
    parser.add_argument("--dataset", choices=["pamap2"], default="pamap2")
    parser.add_argument("--processed-root", default="data/processed/canonical")
    parser.add_argument("--out-dir", default="results/canonical/model_profiles")
    parser.add_argument("--feature-sets", default=",".join(DEFAULT_FEATURE_SETS))
    parser.add_argument("--window-type", default="overlapping")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--eval-unit", default="auto", choices=["auto", "window", "sequence", "sequence_aligned"])
    return parser.parse_args(argv)


def eval_unit_for_model(model_name: str, requested: str) -> str:
    if requested != "auto":
        return requested
    if model_name in {
        "gnn_lstm",
        "improved_gnn_lstm",
        "improved_gnn_lstm_res",
        "improved_gnn_lstm_attn_adj",
        "improved_gnn_lstm_attn_adj_resbn",
        "gnn_flatten_lstm",
    }:
        return "sequence"
    return "window"


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    device = torch.device("cpu")
    models = resolve_models(args.models, top_k=len(DEFAULT_MODELS), rank_source=None, datasets=[args.dataset], metric="macro_f1")

    for feature_set in parse_csv_arg(args.feature_sets):
        processed_dir = Path(args.processed_root) / args.dataset / feature_set / args.window_type
        X, y, subjects, source_indices, label_mapping, inv_label_mapping, data_manifest = load_processed_dataset(
            args.dataset,
            max_windows_per_subject=None,
            seed=42,
            processed_dir=str(processed_dir),
        )
        n_classes = int(len(np.unique(y)))
        n_nodes, node_feat_dim, adj_builder = get_dataset_meta(args.dataset, X_sample=X[0])
        adj_fixed = adj_builder().to(device)

        for model_name in models:
            eval_unit = eval_unit_for_model(model_name, args.eval_unit)
            model = build_model(model_name, args.dataset, X[:1], n_classes, device, adj_fixed)
            profile = model_parameter_profile(
                model,
                dataset=args.dataset,
                model_name=model_name,
                eval_unit=eval_unit,
                raw_window_shape=X.shape[1:],
                n_nodes=n_nodes,
                node_feat_dim=node_feat_dim,
                n_classes=n_classes,
            )
            profile["feature_set"] = feature_set
            profile["window_type"] = args.window_type
            profile["processed_dir"] = str(processed_dir)
            rows.append(profile)
            print(
                f"{feature_set:15s} {model_name:28s} "
                f"params={profile['total_params']:,} node_feat_dim={node_feat_dim}"
            )

    csv_path = out_dir / "deep_model_parameter_counts.csv"
    json_path = out_dir / "deep_model_parameter_counts.json"
    fields = [
        "dataset",
        "feature_set",
        "window_type",
        "model",
        "eval_unit",
        "raw_window_shape",
        "n_nodes",
        "node_feat_dim",
        "n_classes",
        "total_params",
        "trainable_params",
        "non_trainable_params",
        "buffer_values",
        "parameter_size_mb_float32",
        "profile_method",
        "processed_dir",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(rows, fp, indent=2)
    print(f"[OK] wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
