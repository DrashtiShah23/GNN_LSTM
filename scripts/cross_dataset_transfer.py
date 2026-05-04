"""
Cross-dataset transfer: train on all subjects of source dataset, test on target.

Aligned input: 3-axis accelerometer, 128 timesteps, z-score normalised per window
(as in processed npy). PAMAP2 uses wrist acc1_x/y/z (channels 0–2 of processed X).

Labels are mapped to the 6 HHAR activity indices where a semantic match exists;
target samples with no mapping are excluded from evaluation (reported counts).

Outputs:
  results/metrics/cross_dataset_transfer.json
  results/plots/cross_dataset_transfer.png

Usage:
  python scripts/cross_dataset_transfer.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

HHAR_NUM_CLASSES = 6  # fixed label space for transfer / metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    PROCESSED_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    PAMAP2_ACTIVITIES,
    HHAR_ACTIVITIES,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    PATIENCE,
    SEED,
)
from src.models import LSTMOnlyModel
from src.train import get_device, set_seed

Path(METRICS_DIR).mkdir(parents=True, exist_ok=True)
Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)

# PAMAP2 protocol activity_id -> HHAR coarse label index (None = skip sample)
_RAW_PAMAP_TO_HHAR_IDX = {
    0: None,
    1: 1,   # lying ~ sedentary / static → sit bucket (documented as coarse)
    2: 1,   # sitting
    3: 2,   # standing
    4: 3,   # walking
    5: None,
    6: 0,   # cycling → bike
    7: 3,   # nordic walking → walk (confusion pair of interest)
    9: 1,
    10: 1,
    11: None,
    12: 4,
    13: 5,
    14: None,
    15: 1,
    16: 1,
    17: None,
    18: None,
    24: None,
}


def _pamap_sorted_raw_ids() -> list[int]:
    return sorted(PAMAP2_ACTIVITIES.keys())


def pamap2_remapped_to_hhar_y(y_remapped: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y_hhar_space, valid_mask). Invalid samples get y=-1 and mask False."""
    sid = _pamap_sorted_raw_ids()
    out = np.full(len(y_remapped), -1, dtype=np.int64)
    mask = np.zeros(len(y_remapped), dtype=bool)
    for i, yi in enumerate(y_remapped):
        if yi < 0 or yi >= len(sid):
            continue
        raw = sid[int(yi)]
        h = _RAW_PAMAP_TO_HHAR_IDX.get(raw)
        if h is not None:
            out[i] = h
            mask[i] = True
    return out, mask


def load_xy_subj(name: str):
    base = Path(PROCESSED_DIR)
    X = np.load(base / f"{name}_X.npy")
    y = np.load(base / f"{name}_y.npy")
    s = np.load(base / f"{name}_subjects.npy")
    return X, y, s


def lstm_transfer_train_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_train: int | None = 5000,
    n_classes: int = HHAR_NUM_CLASSES,
) -> dict:
    """LSTM on (batch, time, channels). Optional max_train caps epochs cost."""
    device = get_device()
    set_seed(SEED)
    if max_train is not None and len(X_train) > max_train:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X_train), max_train, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    labels_all = np.arange(n_classes)
    # LSTMOnlyModel expects (batch, seq, feat); use per-timestep 3-dim channels as features
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    Xv, Xva, yv, yva = train_test_split(
        Xt, yt, test_size=0.15, random_state=SEED, stratify=yt
    )
    tr_loader = DataLoader(TensorDataset(Xv, yv), batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(TensorDataset(Xva, yva), batch_size=BATCH_SIZE, shuffle=False)
    te_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = LSTMOnlyModel(input_dim=X_train.shape[2], n_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    best_loss, best_state, pat = float("inf"), None, 0

    max_epochs = min(12, NUM_EPOCHS)
    pat_max = min(4, PATIENCE)
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += crit(model(xb), yb).item() * len(yb)
                n += len(yb)
        val_loss /= max(n, 1)
        if val_loss < best_loss:
            best_loss, best_state, pat = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= pat_max:
                break
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            logits = model(xb.to(device))
            preds.extend(logits.argmax(1).cpu().numpy().tolist())
            trues.extend(yb.numpy().tolist())
    yt = np.array(trues)
    yp = np.array(preds)
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(
            f1_score(yt, yp, average="macro", labels=labels_all, zero_division=0)
        ),
        "balanced_acc": float(balanced_accuracy_score(yt, yp)),
    }


def rf_transfer(
    X_train, y_train, X_test, y_test, n_classes: int = HHAR_NUM_CLASSES
) -> dict:
    Xtr = X_train.reshape(len(X_train), -1)
    Xte = X_test.reshape(len(X_test), -1)
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=20,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    clf.fit(Xtr, y_train)
    yp = clf.predict(Xte)
    lab = np.arange(n_classes)
    return {
        "accuracy": float(accuracy_score(y_test, yp)),
        "macro_f1": float(f1_score(y_test, yp, average="macro", labels=lab, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_test, yp)),
    }


def main(include_lstm: bool = False):
    Xh, yh, _ = load_xy_subj("hhar")
    Xp_full, yp_raw, _ = load_xy_subj("pamap2")
    hhar_orig_n = len(Xh)

    # Subsample HHAR for tractable RF / LSTM transfer (full HHAR is hundreds of k windows)
    max_hhar = 12000
    if len(Xh) > max_hhar:
        rng = np.random.default_rng(SEED)
        ix = rng.choice(len(Xh), max_hhar, replace=False)
        Xh, yh = Xh[ix], yh[ix]
        print(f"[HHAR] subsampled to {max_hhar} windows for transfer")

    # Remap PAMAP2 labels to contiguous (same convention as pipeline)
    u = np.unique(yp_raw)
    mp = {int(o): i for i, o in enumerate(u)}
    yp = np.vectorize(mp.__getitem__)(yp_raw)

    Xp = Xp_full[:, :, 0:3]  # wrist accelerometer
    Xh = Xh[:, :, :3]

    yp_mapped, m = pamap2_remapped_to_hhar_y(yp)
    Xp_f = Xp[m]
    yp_f = yp_mapped[m]

    print(f"[HHAR] X={Xh.shape}  classes={len(np.unique(yh))}")
    print(f"[PAMAP2→coarse] kept {len(Xp_f)}/{len(Xp)} windows with HHAR-aligned labels")

    results = {
        "description": "Train all subjects on source; test on target. Input: (128,3) wrist-phone accel.",
        "hhar_windows_used": len(Xh),
        "hhar_windows_total_before_subsample": hhar_orig_n,
        "hhar_classes": HHAR_ACTIVITIES,
        "pamap2_eval_windows": int(len(Xp_f)),
        "lstm_train_cap": 5000,
        "note": "Default RF-only for speed; use --lstm for neural transfer (slow on CPU).",
        "include_lstm": include_lstm,
        "directions": {},
    }

    # HHAR -> PAMAP2 (mapped)
    d1 = {"rf": rf_transfer(Xh, yh, Xp_f, yp_f)}
    if include_lstm:
        d1["lstm"] = lstm_transfer_train_eval(Xh, yh, Xp_f, yp_f, max_train=5000)
    results["directions"]["train_hhar_test_pamap2"] = d1

    # PAMAP2 (mapped) -> HHAR — train only on mappable PAMAP2 subset
    d2 = {"rf": rf_transfer(Xp_f, yp_f, Xh, yh)}
    if include_lstm:
        d2["lstm"] = lstm_transfer_train_eval(Xp_f, yp_f, Xh, yh, max_train=5000)
    results["directions"]["train_pamap2_test_hhar"] = d2

    out_json = Path(METRICS_DIR) / "cross_dataset_transfer.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_json}")

    # Bar plot
    dirs = list(results["directions"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["accuracy", "macro_f1"]):
        x = np.arange(len(dirs))
        w = 0.35
        rf_vals = [results["directions"][d]["rf"][metric] for d in dirs]
        ax.bar(x - (w / 2 if include_lstm else 0), rf_vals, w if include_lstm else 0.5, label="RF (flat window)")
        if include_lstm:
            ls_vals = [results["directions"][d]["lstm"][metric] for d in dirs]
            ax.bar(x + w / 2, ls_vals, w, label="LSTM (3-ch)")
        ax.set_xticks(x)
        ax.set_xticklabels(["HHAR→PAMAP2", "PAMAP2→HHAR"], rotation=15, ha="right")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.set_title(f"Cross-dataset transfer — {metric}")
    fig.suptitle("Train on source (all subjects), test on target (aligned 3-ch accel)", fontsize=11)
    plt.tight_layout()
    ppath = Path(PLOTS_DIR) / "cross_dataset_transfer.png"
    fig.savefig(ppath, dpi=150)
    plt.close()
    print(f"Wrote {ppath}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lstm",
        action="store_true",
        help="Also train/eval LSTM transfer (slow on CPU; RF-only is default)",
    )
    args = ap.parse_args()
    main(include_lstm=args.lstm)
