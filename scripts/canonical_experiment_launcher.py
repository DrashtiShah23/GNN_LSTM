#!/usr/bin/env python
"""Launch canonical core comparison experiments in controlled stages.

The launcher intentionally separates:
  1. classical baselines
  2. deep window models
  3. deep sequence models

This avoids saturating one GPU with every model at once while still using
parallelism inside each stage.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"

DEFAULT_FEATURE_SETS = ["acc16_hr", "acc16_gyro", "acc16_gyro_hr"]
BASELINE_MODELS = [
    "dummy_most_frequent",
    "gaussian_nb",
    "knn_k5",
    "linear_svm",
    "rbf_svm",
    "decision_tree_entropy",
    "bagged_tree_entropy",
    "random_forest",
    "adaboost_tree",
]
DEEP_WINDOW_MODELS = [
    "cnn",
    "lstm",
    "gnn",
    "gnn_learnable_adj",
    "gnn_attention_adj",
]
DEEP_SEQUENCE_MODELS = [
    "gnn_lstm",
    "gnn_flatten_lstm",
    "improved_gnn_lstm",
    "improved_gnn_lstm_attn_adj",
]


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).replace(";", ",").split(",") if x.strip()]


def command_text(cmd: list[str]) -> str:
    return " ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd)


def run_command(cmd: list[str], *, cwd: Path, dry_run: bool) -> int:
    print("\n" + command_text(cmd), flush=True)
    if dry_run:
        return 0
    started = time.time()
    result = subprocess.run(cmd, cwd=str(cwd))
    elapsed = time.time() - started
    print(f"[COMMAND DONE] rc={result.returncode} elapsed_sec={elapsed:.1f}", flush=True)
    return int(result.returncode)


def baseline_command(args: argparse.Namespace, feature_sets: list[str]) -> list[str]:
    models = parse_csv(args.baseline_models)
    cmd = [
        str(PY),
        "scripts/canonical_baseline_runner.py",
        "--dataset",
        "pamap2",
        "--processed-root",
        args.processed_root,
        "--out-root",
        args.results_root,
        "--feature-sets",
        ",".join(feature_sets),
        "--window-types",
        args.window_type,
        "--protocols",
        args.baseline_protocols,
        "--models",
        ",".join(models),
        "--seed",
        str(args.seed),
        "--test-fraction",
        str(args.test_fraction),
    ]
    if args.include_xgb:
        cmd.append("--include-xgb")
    if args.xgb_cuda:
        cmd.append("--use-cuda")
    if args.fast_baselines:
        cmd.append("--fast")
    if args.baseline_max_windows_per_subject > 0:
        cmd += ["--max-windows-per-subject", str(args.baseline_max_windows_per_subject)]
    if args.skip_existing:
        cmd.append("--skip-existing")
    return cmd


def deep_command(
    args: argparse.Namespace,
    *,
    feature_set: str,
    protocol: str,
    models: list[str],
    eval_modes: str,
    batch_size: int,
    parallel_jobs: int,
    stage_name: str,
) -> list[str]:
    processed_dir = Path(args.processed_root) / "pamap2" / feature_set / args.window_type
    run_root = (
        Path(args.results_root)
        / "pamap2"
        / feature_set
        / args.window_type
        / protocol
        / "deep"
    )
    cmd = [
        str(PY),
        "scripts/phase2_repo_deep_parallel_v2.py",
        "--datasets",
        "pamap2",
        "--models",
        ",".join(models),
        "--eval-modes",
        eval_modes,
        "--processed-dir",
        str(processed_dir),
        "--run-root",
        str(run_root),
        "--eval-protocol",
        protocol,
        "--parallel-jobs",
        str(parallel_jobs),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(batch_size),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--cpu-threads-per-job",
        str(args.cpu_threads_per_job),
        "--sequence-length",
        str(args.sequence_length),
        "--sequence-stride",
        str(args.sequence_stride),
        "--sequence-target-policy",
        args.sequence_target_policy,
        "--early-stop-metric",
        args.early_stop_metric,
        "--early-stop-mode",
        args.early_stop_mode,
    ]
    if args.deep_max_windows_per_subject > 0:
        cmd += [
            "--max-windows-per-subject",
            str(args.deep_max_windows_per_subject),
            "--apply-window-cap-to-all-datasets",
        ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.disable_cudnn_for_sequence_models or stage_name == "deep_sequence":
        cmd.append("--disable-cudnn-for-sequence-models")
    return cmd


def write_launcher_manifest(args: argparse.Namespace, commands: list[dict]) -> None:
    manifest_dir = Path(args.results_root) / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / "canonical_launcher_manifest.json"
    with open(path, "w", encoding="utf-8") as fp:
        json.dump({
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "cwd": str(ROOT),
            "python": str(PY),
            "args": vars(args),
            "commands": commands,
        }, fp, indent=2)
    print(f"[OK] launcher manifest: {path}", flush=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch canonical HAR core comparison experiments")
    p.add_argument("--stages", default="baselines,deep_window,deep_sequence", help="Comma-separated stages or all")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--processed-root", default="data/processed/canonical")
    p.add_argument("--results-root", default="results/canonical/core_comparison")
    p.add_argument("--feature-sets", default=",".join(DEFAULT_FEATURE_SETS))
    p.add_argument("--window-type", default="overlapping")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-fraction", type=float, default=0.2)

    p.add_argument("--baseline-protocols", default="random_holdout,loso")
    p.add_argument("--baseline-models", default=",".join(BASELINE_MODELS))
    p.add_argument("--include-xgb", action="store_true")
    p.add_argument("--xgb-cuda", action="store_true", help="Use CUDA for XGBoost if available; normally keep off while deep GPU jobs run.")
    p.add_argument("--fast-baselines", action="store_true")
    p.add_argument("--baseline-max-windows-per-subject", type=int, default=0, help="Smoke/debug cap for baseline runs; 0 means uncapped.")

    p.add_argument("--deep-window-models", default=",".join(DEEP_WINDOW_MODELS))
    p.add_argument("--deep-sequence-models", default=",".join(DEEP_SEQUENCE_MODELS))
    p.add_argument("--deep-protocols", default="loso,random_holdout")
    p.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--window-batch-size", type=int, default=128)
    p.add_argument("--sequence-batch-size", type=int, default=64)
    p.add_argument("--deep-max-windows-per-subject", type=int, default=0, help="Smoke/debug cap for deep runs; 0 means uncapped.")
    p.add_argument("--window-parallel-jobs", type=int, default=2)
    p.add_argument("--sequence-parallel-jobs", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu-threads-per-job", type=int, default=4)
    p.add_argument("--sequence-length", type=int, default=10)
    p.add_argument("--sequence-stride", type=int, default=1)
    p.add_argument("--sequence-target-policy", choices=["last", "majority"], default="last")
    p.add_argument("--early-stop-metric", choices=["val_macro_f1", "val_loss", "val_acc"], default="val_macro_f1")
    p.add_argument("--early-stop-mode", choices=["auto", "min", "max"], default="auto")
    p.add_argument("--disable-cudnn-for-sequence-models", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if not PY.exists():
        raise SystemExit(f"Expected repo venv Python not found: {PY}")

    stages = parse_csv(args.stages)
    if stages == ["all"]:
        stages = ["baselines", "deep_window", "deep_sequence"]
    allowed = {"baselines", "deep_window", "deep_sequence"}
    invalid = sorted(set(stages) - allowed)
    if invalid:
        raise SystemExit("Invalid stage(s): " + ", ".join(invalid))

    feature_sets = parse_csv(args.feature_sets)
    commands_for_manifest: list[dict] = []
    failures = 0

    if "baselines" in stages:
        cmd = baseline_command(args, feature_sets)
        commands_for_manifest.append({"stage": "baselines", "feature_sets": feature_sets, "cmd": cmd})
        failures += run_command(cmd, cwd=ROOT, dry_run=args.dry_run) != 0

    if "deep_window" in stages:
        for feature_set in feature_sets:
            for protocol in parse_csv(args.deep_protocols):
                cmd = deep_command(
                    args,
                    feature_set=feature_set,
                    protocol=protocol,
                    models=parse_csv(args.deep_window_models),
                    eval_modes="window",
                    batch_size=args.window_batch_size,
                    parallel_jobs=args.window_parallel_jobs,
                    stage_name="deep_window",
                )
                commands_for_manifest.append({"stage": "deep_window", "feature_set": feature_set, "protocol": protocol, "cmd": cmd})
                failures += run_command(cmd, cwd=ROOT, dry_run=args.dry_run) != 0

    if "deep_sequence" in stages:
        for feature_set in feature_sets:
            for protocol in parse_csv(args.deep_protocols):
                cmd = deep_command(
                    args,
                    feature_set=feature_set,
                    protocol=protocol,
                    models=parse_csv(args.deep_sequence_models),
                    eval_modes="sequence",
                    batch_size=args.sequence_batch_size,
                    parallel_jobs=args.sequence_parallel_jobs,
                    stage_name="deep_sequence",
                )
                commands_for_manifest.append({"stage": "deep_sequence", "feature_set": feature_set, "protocol": protocol, "cmd": cmd})
                failures += run_command(cmd, cwd=ROOT, dry_run=args.dry_run) != 0

    write_launcher_manifest(args, commands_for_manifest)
    if failures:
        print(f"[FAIL] {failures} stage command(s) failed", flush=True)
        return 1
    print("[OK] selected stages complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
