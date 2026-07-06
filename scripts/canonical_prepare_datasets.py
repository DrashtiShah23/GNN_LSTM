#!/usr/bin/env python
"""Prepare deterministic canonical processed datasets.

This script is a thin wrapper around the existing raw-window preparation code.
It writes feature-set/windowing-specific directories so experiments no longer
depend on a mutable generic ``data/processed/pamap2_X.npy`` file.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_phase2_processed_datasets import prepare_hhar, prepare_pamap2


DEFAULT_PAMAP2_FEATURE_SETS = ["acc16_hr", "acc16_gyro", "acc16_gyro_hr"]
DEFAULT_WINDOW_TYPES = ["overlapping"]


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).replace(";", ",").split(",") if x.strip()]


def window_step(window_type: str, window_size: int, overlap_step: int | None) -> int:
    if window_type == "overlapping":
        return int(overlap_step if overlap_step is not None else window_size // 2)
    if window_type == "non_overlapping":
        return int(window_size)
    raise ValueError(f"Unknown window type: {window_type}")


def discover_existing_rows(out_root: Path) -> list[dict]:
    rows: list[dict] = []
    for manifest_path in sorted(out_root.glob("*/*/*/*_processed_manifest.json")):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fp:
                manifest = json.load(fp)
        except Exception:
            continue
        processed_dir = manifest_path.parent
        try:
            dataset, feature_set, window_type = processed_dir.relative_to(out_root).parts[:3]
        except ValueError:
            continue
        rows.append({
            "dataset": dataset,
            "feature_set": manifest.get("feature_set", feature_set),
            "window_type": window_type,
            "window_size": int(manifest.get("window", 0)),
            "stride": int(manifest.get("step", 0)),
            "processed_dir": str(processed_dir),
            "manifest_path": str(manifest_path),
        })
    return rows


def write_index(rows: list[dict], out_root: Path) -> None:
    manifest_dir = out_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "feature_set",
        "window_type",
        "window_size",
        "stride",
        "processed_dir",
        "manifest_path",
    ]
    merged: dict[tuple[str, str, str], dict] = {}
    for row in discover_existing_rows(out_root) + rows:
        key = (str(row["dataset"]), str(row["feature_set"]), str(row["window_type"]))
        merged[key] = row
    indexed_rows = [merged[key] for key in sorted(merged)]
    with open(manifest_dir / "canonical_data_manifest.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(indexed_rows)
    with open(manifest_dir / "canonical_data_manifest.json", "w", encoding="utf-8") as fp:
        json.dump(indexed_rows, fp, indent=2)


def prepare_pamap2_matrix(args: argparse.Namespace, rows: list[dict]) -> None:
    for feature_set in parse_csv(args.pamap2_feature_sets):
        for window_type in parse_csv(args.window_types):
            step = window_step(window_type, args.window_size, args.overlap_step)
            out_dir = (
                Path(args.out_root)
                / "pamap2"
                / feature_set
                / window_type
            )
            prepare_pamap2(
                data_root=Path(args.data_root),
                out_dir=out_dir,
                task=args.pamap2_task,
                sessions=args.pamap2_sessions,
                feature_set=feature_set,
                window=args.window_size,
                step=step,
                overwrite=args.overwrite,
            )
            rows.append({
                "dataset": "pamap2",
                "feature_set": feature_set,
                "window_type": window_type,
                "window_size": int(args.window_size),
                "stride": int(step),
                "processed_dir": str(out_dir),
                "manifest_path": str(out_dir / "pamap2_processed_manifest.json"),
            })


def prepare_hhar_matrix(args: argparse.Namespace, rows: list[dict]) -> None:
    for window_type in parse_csv(args.window_types):
        step = window_step(window_type, args.window_size, args.overlap_step)
        out_dir = Path(args.out_root) / "hhar" / "accel_gyro" / window_type
        prepare_hhar(
            data_root=Path(args.data_root),
            out_dir=out_dir,
            window=args.window_size,
            step=step,
            overwrite=args.overwrite,
        )
        rows.append({
            "dataset": "hhar",
            "feature_set": "accel_gyro",
            "window_type": window_type,
            "window_size": int(args.window_size),
            "stride": int(step),
            "processed_dir": str(out_dir),
            "manifest_path": str(out_dir / "hhar_processed_manifest.json"),
        })


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare canonical HAR datasets")
    parser.add_argument("--dataset", choices=["pamap2", "hhar", "both"], default="pamap2")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--out-root", default="data/processed/canonical")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pamap2-feature-sets", default=",".join(DEFAULT_PAMAP2_FEATURE_SETS))
    parser.add_argument("--pamap2-task", choices=["protocol12", "all18"], default="all18")
    parser.add_argument("--pamap2-sessions", choices=["protocol", "optional", "all"], default="all")
    parser.add_argument("--window-types", default=",".join(DEFAULT_WINDOW_TYPES))
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--overlap-step", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if args.dataset in {"pamap2", "both"}:
        prepare_pamap2_matrix(args, rows)
    if args.dataset in {"hhar", "both"}:
        prepare_hhar_matrix(args, rows)

    write_index(rows, out_root)
    print(f"[OK] Wrote canonical data index to {out_root / 'manifests'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
