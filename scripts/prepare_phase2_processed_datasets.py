#!/usr/bin/env python
"""Prepare raw-window .npy files for Phase 2 repo deep-model runs.

Creates:
  data/processed/pamap2_X.npy, pamap2_y.npy, pamap2_subjects.npy
  data/processed/hhar_X.npy, hhar_y.npy, hhar_subjects.npy

This does not modify any existing repo script. It builds raw windows (N,T,C), not
handcrafted feature matrices, because the repo torch datasets expect raw windows.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

PAMAP2_ACTIVITY_MAP = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    9: "watching_tv",
    10: "computer_work",
    11: "car_driving",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping",
}
PAMAP2_PROTOCOL12 = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
PAMAP2_ALL18 = sorted(PAMAP2_ACTIVITY_MAP.keys())
PAMAP2_TIMESTAMP_AUDIT_COLUMNS = [
    "dataset",
    "scope",
    "source",
    "session",
    "subject",
    "timestamp",
    "n_rows",
    "activity_ids",
    "action",
]
HHAR_ACTIVITIES = ["bike", "sit", "stand", "walk", "stairsup", "stairsdown"]


def build_pamap2_columns() -> List[str]:
    cols = ["timestamp", "activity_id", "heart_rate"]
    for pos in ["hand", "chest", "ankle"]:
        cols += [
            f"{pos}_temp",
            f"{pos}_acc16_x", f"{pos}_acc16_y", f"{pos}_acc16_z",
            f"{pos}_acc6_x", f"{pos}_acc6_y", f"{pos}_acc6_z",
            f"{pos}_gyro_x", f"{pos}_gyro_y", f"{pos}_gyro_z",
            f"{pos}_mag_x", f"{pos}_mag_y", f"{pos}_mag_z",
            f"{pos}_ori_1", f"{pos}_ori_2", f"{pos}_ori_3", f"{pos}_ori_4",
        ]
    return cols


PAMAP2_COLS = build_pamap2_columns()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def window_array(arr: np.ndarray, window: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(arr) < window:
        return np.empty((0, window, arr.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    starts = np.arange(0, len(arr) - window + 1, step, dtype=np.int64)
    wins = np.stack([arr[s:s + window] for s in starts]).astype(np.float32)
    return wins, starts


def fill_numeric_segment(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    if not cols:
        return out
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan)
    out[cols] = out[cols].interpolate(method="linear", limit_direction="both")
    out[cols] = out[cols].ffill().bfill().fillna(0.0)
    return out


def clean_pamap2_timestamp_conflicts(
    df: pd.DataFrame,
    source: str,
    session: str,
    subject: int,
    audit_rows: List[Dict[str, object]],
) -> pd.DataFrame:
    if "timestamp" not in df.columns or "activity_id" not in df.columns:
        return df
    work = df.copy()
    dup = work[work.duplicated(["timestamp"], keep=False)]
    if dup.empty:
        return work

    for ts, grp in dup.groupby("timestamp", sort=False):
        acts = sorted(int(x) for x in grp["activity_id"].dropna().unique().tolist())
        audit_rows.append({
            "dataset": "pamap2",
            "scope": "within_recording",
            "source": source,
            "session": session,
            "subject": int(subject),
            "timestamp": float(ts),
            "n_rows": int(len(grp)),
            "activity_ids": ",".join(str(x) for x in acts),
            "action": "drop_all_conflicting_rows" if len(acts) > 1 else "dedupe_keep_first",
        })

    conflict_ts = dup.groupby("timestamp")["activity_id"].nunique()
    conflict_values = set(conflict_ts[conflict_ts > 1].index.tolist())
    if conflict_values:
        work = work[~work["timestamp"].isin(conflict_values)].copy()
    work = work.drop_duplicates(["timestamp", "activity_id"], keep="first").copy()
    return work


def find_pamap2_root(data_root: Path) -> Path:
    candidates = [
        data_root / "pamap2" / "PAMAP2_Dataset",
        data_root / "PAMAP2_Dataset",
        data_root / "pamap2",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find PAMAP2 root under data/raw. Expected data/raw/pamap2/PAMAP2_Dataset")


def pamap2_session_dirs(root: Path, sessions: str) -> List[Path]:
    wanted = []
    if sessions in {"protocol", "all"}:
        wanted.append(root / "Protocol")
    if sessions in {"optional", "all"}:
        wanted.append(root / "Optional")
    found = [p for p in wanted if p.exists()]
    if not found:
        raise FileNotFoundError("No PAMAP2 session folders found. Checked: " + ", ".join(str(p) for p in wanted))
    return found


def pamap2_feature_cols(feature_set: str) -> List[str]:
    # Order is important for repo graph mapping:
    # [hand/wrist channels..., chest channels..., ankle channels...]
    cols: List[str] = []
    for pos in ["hand", "chest", "ankle"]:
        if feature_set in {"acc16", "acc16_hr", "acc16_gyro", "acc16_gyro_hr"}:
            cols += [f"{pos}_acc16_x", f"{pos}_acc16_y", f"{pos}_acc16_z"]
        if feature_set in {"acc16_gyro", "acc16_gyro_hr"}:
            cols += [f"{pos}_gyro_x", f"{pos}_gyro_y", f"{pos}_gyro_z"]
        if feature_set in {"allimu_no_orientation", "allimu_hr"}:
            cols += [
                f"{pos}_acc16_x", f"{pos}_acc16_y", f"{pos}_acc16_z",
                f"{pos}_acc6_x", f"{pos}_acc6_y", f"{pos}_acc6_z",
                f"{pos}_gyro_x", f"{pos}_gyro_y", f"{pos}_gyro_z",
                f"{pos}_mag_x", f"{pos}_mag_y", f"{pos}_mag_z",
            ]
    if feature_set in {"acc16_hr", "acc16_gyro_hr", "allimu_hr"}:
        cols.append("heart_rate")
    return cols


def prepare_pamap2(
    data_root: Path,
    out_dir: Path,
    task: str,
    sessions: str,
    feature_set: str,
    window: int,
    step: int,
    overwrite: bool,
) -> None:
    x_path = out_dir / "pamap2_X.npy"
    y_path = out_dir / "pamap2_y.npy"
    s_path = out_dir / "pamap2_subjects.npy"
    if not overwrite and x_path.exists() and y_path.exists() and s_path.exists():
        print(f"[SKIP] PAMAP2 processed files already exist in {out_dir}. Use --overwrite to rebuild.")
        return

    root = find_pamap2_root(data_root)
    dirs = pamap2_session_dirs(root, sessions)
    keep_ids = PAMAP2_PROTOCOL12 if task == "protocol12" else PAMAP2_ALL18
    feature_cols = pamap2_feature_cols(feature_set)
    if len(feature_cols) % 3 not in {0, 1}:
        raise ValueError(
            "PAMAP2 channel count must be divisible by 3, or divisible by 3 with one trailing global channel such as heart rate; "
            f"got {len(feature_cols)}"
        )

    windows_all: List[np.ndarray] = []
    labels_all: List[int] = []
    subjects_all: List[int] = []
    meta_rows: List[Dict[str, object]] = []
    timestamp_audit_rows: List[Dict[str, object]] = []
    sample_id = 0

    files: List[Path] = []
    for d in dirs:
        files.extend(sorted(d.glob("subject*.dat")))
    if not files:
        raise FileNotFoundError("No subject*.dat files found in: " + ", ".join(str(d) for d in dirs))

    for f in files:
        m = re.search(r"(\d+)", f.stem)
        subject = int(m.group(1)) if m else -1
        session = f.parent.name.lower()
        print(f"[PAMAP2] reading {f}")
        df = pd.read_csv(f, sep=r"\s+", header=None, engine="python")
        df = df.iloc[:, :len(PAMAP2_COLS)]
        df.columns = PAMAP2_COLS[:df.shape[1]]
        df = clean_pamap2_timestamp_conflicts(df, f.name, session, subject, timestamp_audit_rows)
        df = df[df["activity_id"].isin(keep_ids)].copy()
        if df.empty:
            continue
        available = [c for c in feature_cols if c in df.columns]
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{f.name} missing required columns: {missing}")

        # Build segment boundaries before filling so interpolation never crosses labels or timestamp gaps.
        act_change = df["activity_id"].ne(df["activity_id"].shift()).fillna(True)
        ts_gap = df["timestamp"].diff().fillna(0).abs() > 2.0 if "timestamp" in df.columns else False
        df["segment_id"] = (act_change | ts_gap).cumsum()

        for (seg_id, act_id), grp0 in df.groupby(["segment_id", "activity_id"], sort=False):
            grp = fill_numeric_segment(grp0, available)
            arr = grp[available].to_numpy(dtype=np.float32)
            wins, starts = window_array(arr, window, step)
            if len(wins) == 0:
                continue
            windows_all.append(wins)
            labels_all.extend([int(act_id)] * len(wins))
            subjects_all.extend([subject] * len(wins))
            timestamps = grp["timestamp"].to_numpy() if "timestamp" in grp.columns else np.arange(len(grp))
            for s in starts:
                meta_rows.append({
                    "sample_id": sample_id,
                    "dataset": "pamap2",
                    "source": f.name,
                    "session": session,
                    "recording_id": f"{session}:{f.name}",
                    "subject": subject,
                    "activity_id": int(act_id),
                    "activity_name": PAMAP2_ACTIVITY_MAP.get(int(act_id), str(act_id)),
                    "segment_id": int(seg_id),
                    "window_start_row": int(s),
                    "window_end_row": int(s + window - 1),
                    "window_start_time": float(timestamps[s]) if s < len(timestamps) else math.nan,
                    "window_end_time": float(timestamps[min(s + window - 1, len(timestamps) - 1)]) if len(timestamps) else math.nan,
                })
                sample_id += 1

    if not windows_all:
        raise RuntimeError("No PAMAP2 windows generated. Check task/session/window settings.")
    X = np.concatenate(windows_all, axis=0).astype(np.float32)
    y = np.asarray(labels_all, dtype=np.int64)
    subjects = np.asarray(subjects_all, dtype=np.int64)
    present_ids = set(int(x) for x in np.unique(y).tolist())
    if 0 in present_ids:
        raise RuntimeError("PAMAP2 activity_id=0 was included; transient activity must be discarded.")
    if task == "all18":
        missing = sorted(set(PAMAP2_ALL18) - present_ids)
        if missing:
            raise RuntimeError(
                "PAMAP2 all18 requested but these activity IDs were not generated: "
                + ", ".join(str(x) for x in missing)
                + ". Use --pamap2-sessions all so Optional activities are included."
            )
    ensure_dir(out_dir)
    np.save(x_path, X, allow_pickle=False)
    np.save(y_path, y, allow_pickle=False)
    np.save(s_path, subjects, allow_pickle=False)
    pd.DataFrame(meta_rows).to_csv(out_dir / "pamap2_window_manifest.csv", index=False)
    pd.DataFrame(timestamp_audit_rows, columns=PAMAP2_TIMESTAMP_AUDIT_COLUMNS).to_csv(
        out_dir / "pamap2_timestamp_audit.csv",
        index=False,
    )
    with open(out_dir / "pamap2_processed_manifest.json", "w", encoding="utf-8") as fp:
        json.dump({
            "dataset": "pamap2",
            "task": task,
            "sessions": sessions,
            "feature_set": feature_set,
            "feature_columns": feature_cols,
            "window": window,
            "step": step,
            "shape": list(X.shape),
            "n_subjects": int(len(np.unique(subjects))),
            "subjects": [int(x) for x in np.unique(subjects).tolist()],
            "labels": [int(x) for x in np.unique(y).tolist()],
            "label_names": [PAMAP2_ACTIVITY_MAP[int(x)] for x in sorted(np.unique(y).tolist())],
            "discarded_activity_ids": [0],
            "timestamp_conflict_audit": "pamap2_timestamp_audit.csv",
            "timestamp_conflict_policy": "Within one source recording, duplicate timestamps with different activity IDs are dropped before windowing; duplicate timestamps with the same activity ID keep the first row. Protocol and Optional files are separate recordings and windows never cross source files.",
            "n_timestamp_audit_rows": int(len(timestamp_audit_rows)),
            "activity_mapping": PAMAP2_ACTIVITY_MAP,
        }, fp, indent=2)
    print(f"[OK] PAMAP2 saved X={X.shape}, labels={len(np.unique(y))}, subjects={len(np.unique(subjects))} to {out_dir}")


def find_hhar_dir(data_root: Path) -> Path:
    candidates = [
        data_root / "hhar" / "Activity recognition exp",
        data_root / "hhar",
        data_root / "Activity recognition exp",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find HHAR folder under data/raw/hhar or data/raw/hhar/Activity recognition exp")


def find_activity_col(cols: Sequence[str]) -> str | None:
    if "gt" in cols:
        return "gt"
    for c in cols:
        if c in {"activity", "label", "class"} or "activity" in c:
            return c
    return None


def prepare_hhar(data_root: Path, out_dir: Path, window: int, step: int, overwrite: bool) -> None:
    x_path = out_dir / "hhar_X.npy"
    y_path = out_dir / "hhar_y.npy"
    s_path = out_dir / "hhar_subjects.npy"
    if not overwrite and x_path.exists() and y_path.exists() and s_path.exists():
        print(f"[SKIP] HHAR processed files already exist in {out_dir}. Use --overwrite to rebuild.")
        return

    hhar_dir = find_hhar_dir(data_root)
    files = [
        "Phones_accelerometer.csv",
        "Phones_gyroscope.csv",
        "Watch_accelerometer.csv",
        "Watch_gyroscope.csv",
    ]
    windows_all: List[np.ndarray] = []
    labels_all: List[str] = []
    subjects_all: List[str] = []
    meta_rows: List[Dict[str, object]] = []
    sample_id = 0
    channels = ["x", "y", "z"]

    for fname in files:
        path = hhar_dir / fname
        if not path.exists():
            print(f"[WARN] HHAR file missing: {path}")
            continue
        print(f"[HHAR] reading {path}")
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        if not set(channels).issubset(df.columns):
            print(f"[WARN] {fname}: missing x/y/z; skipping")
            continue
        activity_col = find_activity_col(df.columns)
        user_col = "user" if "user" in df.columns else None
        if activity_col is None or user_col is None:
            print(f"[WARN] {fname}: could not find activity/user columns; skipping")
            continue
        df[activity_col] = df[activity_col].astype(str).str.lower().str.strip()
        df = df[df[activity_col].isin(HHAR_ACTIVITIES)].copy()
        if df.empty:
            continue
        sort_col = next((c for c in ["creation_time", "arrival_time", "timestamp"] if c in df.columns), None)
        if sort_col:
            df = df.sort_values(sort_col)
        source = fname.replace(".csv", "")
        group_cols = [user_col, activity_col]
        for extra in ["model", "device"]:
            if extra in df.columns:
                group_cols.append(extra)
        for keys, grp0 in df.groupby(group_cols, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            user = str(keys[0])
            act = str(keys[1])
            grp = fill_numeric_segment(grp0, channels)
            arr = grp[channels].to_numpy(dtype=np.float32)
            wins, starts = window_array(arr, window, step)
            if len(wins) == 0:
                continue
            windows_all.append(wins)
            labels_all.extend([act] * len(wins))
            subjects_all.extend([user] * len(wins))
            for s in starts:
                meta_rows.append({
                    "sample_id": sample_id,
                    "dataset": "hhar",
                    "source": source,
                    "subject": user,
                    "activity_name": act,
                    "window_start_row": int(s),
                    "window_end_row": int(s + window - 1),
                })
                sample_id += 1

    if not windows_all:
        raise RuntimeError("No HHAR windows generated. Check raw files and folder layout.")
    X = np.concatenate(windows_all, axis=0).astype(np.float32)
    y = np.asarray(labels_all, dtype="U32")
    subjects = np.asarray(subjects_all, dtype="U32")
    ensure_dir(out_dir)
    np.save(x_path, X, allow_pickle=False)
    np.save(y_path, y, allow_pickle=False)
    np.save(s_path, subjects, allow_pickle=False)
    pd.DataFrame(meta_rows).to_csv(out_dir / "hhar_window_manifest.csv", index=False)
    with open(out_dir / "hhar_processed_manifest.json", "w", encoding="utf-8") as fp:
        json.dump({
            "dataset": "hhar",
            "window": window,
            "step": step,
            "shape": list(X.shape),
            "channels": channels,
            "n_subjects": int(len(np.unique(subjects))),
            "subjects": [str(x) for x in np.unique(subjects).tolist()],
            "labels": [str(x) for x in np.unique(y).tolist()],
            "source_files": files,
        }, fp, indent=2)
    print(f"[OK] HHAR saved X={X.shape}, labels={len(np.unique(y))}, subjects={len(np.unique(subjects))} to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["pamap2", "hhar", "both"], default="both")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pamap2-task", choices=["protocol12", "all18"], default="all18")
    parser.add_argument("--pamap2-sessions", choices=["protocol", "optional", "all"], default="all")
    parser.add_argument(
        "--pamap2-feature-set",
        choices=["acc16", "acc16_hr", "acc16_gyro", "acc16_gyro_hr", "allimu_no_orientation", "allimu_hr"],
        default="acc16_gyro",
    )
    parser.add_argument("--pamap2-window", type=int, default=512)
    parser.add_argument("--pamap2-step", type=int, default=256)
    parser.add_argument("--hhar-window", type=int, default=128)
    parser.add_argument("--hhar-step", type=int, default=64)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if args.dataset in {"pamap2", "both"}:
        prepare_pamap2(
            data_root=data_root,
            out_dir=out_dir,
            task=args.pamap2_task,
            sessions=args.pamap2_sessions,
            feature_set=args.pamap2_feature_set,
            window=args.pamap2_window,
            step=args.pamap2_step,
            overwrite=args.overwrite,
        )
    if args.dataset in {"hhar", "both"}:
        prepare_hhar(
            data_root=data_root,
            out_dir=out_dir,
            window=args.hhar_window,
            step=args.hhar_step,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
