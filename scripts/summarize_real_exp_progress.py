#!/usr/bin/env python
"""Summarize persistent progress for canonical real Exp3/Exp6."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize real Exp3/Exp6 progress log.")
    parser.add_argument(
        "--progress-log",
        type=Path,
        default=Path("results/canonical_protocol12_seven_experiments/real_exp3_exp6/progress_events.jsonl"),
    )
    parser.add_argument("--tail", type=int, default=12)
    return parser.parse_args()


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Progress log not found: {path}")
    events = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).replace(";", ",").split(",") if x.strip()]


def read_manifest(progress_log: Path) -> dict:
    manifest_path = progress_log.parent / "run_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return {}


def exp3_baseline_denominators(progress_log: Path) -> tuple[int, dict[str, int]]:
    manifest = read_manifest(progress_log)
    args = manifest.get("args", {}) if isinstance(manifest, dict) else {}
    models = parse_csv(args.get("baseline_models", ""))
    feature_sets = parse_csv(args.get("feature_sets", "acc16_hr,acc16_gyro,acc16_gyro_hr"))
    perturbations = parse_csv(args.get("perturbations", "gaussian_noise,random_channel_dropout,heart_rate_zero"))
    severities = parse_csv(args.get("severities", "low,medium,high"))
    if not models:
        models = ["unknown"]
    per_feature: dict[str, int] = {}
    for feature in feature_sets:
        valid_conditions = len(perturbations) * len(severities)
        if "heart_rate_zero" in perturbations and feature == "acc16_gyro":
            valid_conditions -= len(severities)
        per_feature[feature] = len(models) * valid_conditions
    return sum(per_feature.values()), per_feature


def field(message: str, name: str) -> str:
    match = re.search(rf"{re.escape(name)}=([^\s]+)", message)
    return match.group(1) if match else ""


def main() -> int:
    args = parse_args()
    events = read_events(args.progress_log)
    if not events:
        print(f"No events yet in {args.progress_log}")
        return 0

    messages = [str(e.get("message", "")) for e in events]
    exp3_agg = [m for m in messages if "EXP3 baseline perturb aggregate done" in m]
    exp3_done_models = [m for m in messages if "EXP3 baseline done feature_set=" in m and " rows=" in m]
    exp3_v3_done = [m for m in messages if "EXP3 v3 done" in m]
    exp6_table = [m for m in messages if "EXP6 table written" in m]
    exp3_table = [m for m in messages if "EXP3 table written" in m]
    exp3_total, exp3_per_feature = exp3_baseline_denominators(args.progress_log)

    by_feature = Counter(field(m, "feature_set") for m in exp3_agg)
    by_model = Counter(field(m, "model") for m in exp3_agg)
    last = events[-1]

    print(f"Progress log: {args.progress_log}")
    print(f"Events: {len(events)}")
    print(f"Last event: [{last.get('time')}] pid={last.get('pid')} {last.get('message')}")
    print()
    print("Exp3 baseline aggregate conditions:")
    print(f"  completed: {len(exp3_agg)} / {exp3_total} ({len(exp3_agg) / max(exp3_total, 1) * 100:.1f}%)")
    for feature, total in exp3_per_feature.items():
        count = by_feature.get(feature, 0)
        print(f"  {feature}: {count} / {total} ({count / max(total, 1) * 100:.1f}%)")
    print()
    print(f"Exp3 baseline completed model blocks: {len(exp3_done_models)}")
    print(f"Exp3 v3 completed model blocks: {len(exp3_v3_done)}")
    print(f"Exp3 table written: {'yes' if exp3_table else 'no'}")
    print(f"Exp6 table written: {'yes' if exp6_table else 'no'}")
    print()
    print("Top completed Exp3 baseline models by condition count:")
    for model, count in by_model.most_common(10):
        print(f"  {model}: {count}")
    print()
    print(f"Last {min(args.tail, len(events))} events:")
    for event in events[-args.tail:]:
        print(f"  [{event.get('time')}] pid={event.get('pid')} {event.get('message')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
