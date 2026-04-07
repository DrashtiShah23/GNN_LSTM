"""
Preprocessing pipeline for HHAR and PAMAP2 datasets.

Steps (per proposal):
  1. Synchronise & resample streams to TARGET_SAMPLING_RATE
  2. Per-channel normalisation (z-score)
  3. Segment with overlapping sliding window
  4. Handle missing values (masking / forward-fill)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.preprocessing import LabelEncoder

from src.config import (
    HHAR_RAW_DIR, PAMAP2_RAW_DIR, PROCESSED_DIR,
    WINDOW_SIZE, OVERLAP, TARGET_SAMPLING_RATE,
    HHAR_ACTIVITIES, PAMAP2_ACTIVITIES, PAMAP2_POSITIONS,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _sliding_windows(
    data: np.ndarray,
    window_size: int,
    overlap: float,
) -> np.ndarray:
    """Return array of shape (n_windows, window_size, n_channels)."""
    step = int(window_size * (1 - overlap))
    starts = range(0, len(data) - window_size + 1, step)
    return np.stack([data[s: s + window_size] for s in starts])


def _normalise(data: np.ndarray) -> np.ndarray:
    """Z-score normalise per channel (axis 0 = time)."""
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True) + 1e-8
    return (data - mean) / std


def _resample_to_target(data: np.ndarray, orig_rate: int) -> np.ndarray:
    """Resample time-series rows to TARGET_SAMPLING_RATE."""
    if orig_rate == TARGET_SAMPLING_RATE:
        return data
    n_target = int(len(data) * TARGET_SAMPLING_RATE / orig_rate)
    return resample(data, n_target, axis=0)


# ── HHAR ─────────────────────────────────────────────────────────────────────

def load_hhar_raw() -> pd.DataFrame:
    """
    Load HHAR CSV files. Expected layout after extraction:
      data/raw/hhar/Activity recognition exp/
        Phones_accelerometer.csv
        Phones_gyroscope.csv
        Watch_accelerometer.csv
        Watch_gyroscope.csv
    """
    raw_dir = Path(HHAR_RAW_DIR)
    # Try both possible sub-folder names
    candidates = [
        raw_dir / "Activity recognition exp",
        raw_dir,
    ]
    data_dir = next((c for c in candidates if c.exists()), raw_dir)

    dfs = []
    for fname in [
        "Phones_accelerometer.csv",
        "Phones_gyroscope.csv",
        "Watch_accelerometer.csv",
        "Watch_gyroscope.csv",
    ]:
        fpath = data_dir / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            df["source"] = fname.split(".")[0]
            dfs.append(df)
        else:
            print(f"  [WARNING] {fpath} not found — skipping.")

    if not dfs:
        raise FileNotFoundError(
            f"No HHAR CSV files found in {data_dir}. "
            "Run src/data_download.py first."
        )
    return pd.concat(dfs, ignore_index=True)


def preprocess_hhar() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X : (N, WINDOW_SIZE, n_features)
    y : (N,)  integer activity labels
    subjects : (N,)  subject IDs (for LOSO splits)
    """
    print("Preprocessing HHAR …")
    df = load_hhar_raw()

    # Standardise column names to lower-case
    df.columns = [c.lower() for c in df.columns]

    # Keep only known activities
    activity_col = "gt"  # ground-truth column name in HHAR
    if activity_col not in df.columns:
        # fall back
        activity_col = [c for c in df.columns if "gt" in c or "activity" in c][0]

    df = df[df[activity_col].isin(HHAR_ACTIVITIES)].copy()
    df[activity_col] = df[activity_col].str.lower().str.strip()

    le = LabelEncoder()
    df["label"] = le.fit_transform(df[activity_col])

    # Sensor columns: x, y, z
    sensor_cols = ["x", "y", "z"]
    subject_col = "user"

    windows, labels, subjects = [], [], []

    for (user, activity), grp in df.groupby([subject_col, activity_col]):
        data = grp[sensor_cols].values.astype(np.float32)
        data = _resample_to_target(data, orig_rate=TARGET_SAMPLING_RATE)
        data = _normalise(data)
        wins = _sliding_windows(data, WINDOW_SIZE, OVERLAP)
        if len(wins) == 0:
            continue
        label_id = grp["label"].iloc[0]
        windows.append(wins)
        labels.extend([label_id] * len(wins))
        subjects.extend([user] * len(wins))

    X = np.concatenate(windows, axis=0)
    y = np.array(labels, dtype=np.int64)
    subj = np.array(subjects)
    print(f"  HHAR windows: {X.shape}, labels: {y.shape}")

    out = Path(PROCESSED_DIR)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "hhar_X.npy", X)
    np.save(out / "hhar_y.npy", y)
    np.save(out / "hhar_subjects.npy", subj)
    print("  Saved to data/processed/")
    return X, y, subj


# ── PAMAP2 ────────────────────────────────────────────────────────────────────

def _pamap2_col_names() -> list:
    cols = ["timestamp", "activity_id", "heart_rate"]
    sensors = [
        "temp",
        "acc1_x", "acc1_y", "acc1_z",
        "acc2_x", "acc2_y", "acc2_z",
        "gyro_x", "gyro_y", "gyro_z",
        "mag_x", "mag_y", "mag_z",
        "orient_1", "orient_2", "orient_3", "orient_4",
    ]
    for pos in PAMAP2_POSITIONS:
        for sensor in sensors:
            cols.append(f"{pos}_{sensor}")
    return cols

_PAMAP2_COLS = _pamap2_col_names()


def load_pamap2_subject(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=" ", header=None)
    # Assign as many column names as there are columns
    cols = _PAMAP2_COLS[: len(df.columns)]
    df.columns = cols
    return df


def preprocess_pamap2() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X : (N, WINDOW_SIZE, n_features)  — per-position sensor channels
    y : (N,)
    subjects : (N,)
    """
    print("Preprocessing PAMAP2 …")
    raw_dir = Path(PAMAP2_RAW_DIR)
    protocol_dir = raw_dir / "PAMAP2_Dataset" / "Protocol"
    if not protocol_dir.exists():
        protocol_dir = raw_dir / "Protocol"
    if not protocol_dir.exists():
        raise FileNotFoundError(
            f"PAMAP2 Protocol folder not found under {raw_dir}. "
            "Run src/data_download.py first."
        )

    # Sensor columns to use: 3-axis accel + gyro for each IMU position
    feature_cols = [
        col for col in _PAMAP2_COLS
        if any(s in col for s in ["acc1_x", "acc1_y", "acc1_z",
                                   "gyro_x", "gyro_y", "gyro_z"])
    ]

    windows, labels, subjects = [], [], []

    for fpath in sorted(protocol_dir.glob("subject*.dat")):
        subject_id = int(fpath.stem.replace("subject", ""))
        df = load_pamap2_subject(fpath)

        # Keep only labelled, known activities (drop activity 0 = transient)
        df = df[df["activity_id"].isin(PAMAP2_ACTIVITIES.keys())].copy()

        # Forward-fill missing values
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        le_map = {v: i for i, v in enumerate(sorted(PAMAP2_ACTIVITIES.keys()))}

        for activity_id, grp in df.groupby("activity_id"):
            available_cols = [c for c in feature_cols if c in grp.columns]
            data = grp[available_cols].values.astype(np.float32)
            if len(data) < WINDOW_SIZE:
                continue  # skip segments shorter than one window
            data = _resample_to_target(data, orig_rate=100)  # PAMAP2 is at 100 Hz
            if len(data) < WINDOW_SIZE:
                continue
            data = _normalise(data)
            wins = _sliding_windows(data, WINDOW_SIZE, OVERLAP)
            if len(wins) == 0:
                continue
            label_id = le_map[activity_id]
            windows.append(wins)
            labels.extend([label_id] * len(wins))
            subjects.extend([subject_id] * len(wins))

    X = np.concatenate(windows, axis=0)
    y = np.array(labels, dtype=np.int64)
    subj = np.array(subjects)
    print(f"  PAMAP2 windows: {X.shape}, labels: {y.shape}")

    out = Path(PROCESSED_DIR)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pamap2_X.npy", X)
    np.save(out / "pamap2_y.npy", y)
    np.save(out / "pamap2_subjects.npy", subj)
    print("  Saved to data/processed/")
    return X, y, subj


if __name__ == "__main__":
    preprocess_hhar()
    preprocess_pamap2()
