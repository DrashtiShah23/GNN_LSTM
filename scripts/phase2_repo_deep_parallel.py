#!/usr/bin/env python3
"""
Phase 2 repo-deep-model parallel runner for DrashtiShah23/GNN_LSTM.

This file is intentionally standalone and does NOT modify existing repo scripts.
It imports the repo's src/ modules and runs selected/proposed deep algorithms with
isolated artifact folders, fold checkpoints, predictions, reports, and aggregate
experiment metadata.

Typical use from repo root:
  python scripts/phase2_repo_deep_parallel.py --datasets pamap2 hhar --models auto --top-k 2 --parallel-jobs 2

Worker mode is internal:
  python scripts/phase2_repo_deep_parallel.py --worker --dataset pamap2 --model improved_gnn_lstm --run-root results/...
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for Phase 2 deep-model experiments. Install torch first.") from exc

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise SystemExit("pandas is required. Run: python -m pip install pandas") from exc

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

SEED = 42

# Experiments from the uploaded research-plan document. These are attached to
# every run as metadata so the resulting artifacts can be mapped to the planned
# manuscript experiments.
EXPERIMENT_CATALOG: Dict[str, Dict[str, Any]] = {
    "exp1_leakage_nonoverlap": {
        "title": "Leakage-Control Evaluation with Non-Overlapping Windows",
        "goal": "Compare overlapping vs non-overlapping windows under random holdout and LOSO.",
        "status_in_this_runner": "partial: this runner produces LOSO artifacts; non-overlap requires preprocessing with stride=window.",
        "deliverables": ["fold metrics", "predictions", "confusion matrices", "model comparison inputs"],
    },
    "exp2_statistical_reliability": {
        "title": "Statistical Reliability and Model Ranking Stability",
        "goal": "Compute fold-level mean/std/CI and ranking stability across LOSO folds.",
        "status_in_this_runner": "implemented: fold metrics and aggregate ranking CSV are written.",
        "deliverables": ["metrics_by_fold.csv", "metrics_summary.csv", "metrics_ranked_all_jobs.csv"],
    },
    "exp3_robustness_sensor_failure": {
        "title": "Robustness to Sensor Failure, Noise, and Missing Data",
        "goal": "Evaluate test-time degradation under missing/noisy sensors and windows.",
        "status_in_this_runner": "metadata only: use saved checkpoints/predictions as base artifacts for robustness reruns.",
        "deliverables": ["checkpointed fold models"],
    },
    "exp4_calibration_uncertainty": {
        "title": "Calibration, Uncertainty, and Selective Prediction",
        "goal": "Store probabilities/logits for ECE, Brier, NLL, reliability diagrams and selective prediction.",
        "status_in_this_runner": "implemented: per-sample predicted probabilities are saved when logits are available.",
        "deliverables": ["predictions.csv with proba_* columns", "classification_report.csv"],
    },
    "exp5_subject_failure": {
        "title": "Subject-Level Generalization Failure Analysis",
        "goal": "Find hardest/easiest LOSO subjects and dominant confusions.",
        "status_in_this_runner": "implemented: subject_failure_analysis.csv is written per job.",
        "deliverables": ["subject_failure_analysis.csv", "confusion_matrix.csv/png"],
    },
    "exp6_few_shot_calibration": {
        "title": "Few-Shot Subject Calibration",
        "goal": "Fine-tune with 1/5/10 percent held-out-subject calibration data.",
        "status_in_this_runner": "metadata only: saved fold checkpoints enable future fine-tuning experiments.",
        "deliverables": ["fold checkpoints"],
    },
    "exp7_health_group_analysis": {
        "title": "Health-Relevant Activity Group Analysis",
        "goal": "Map labels to clinically meaningful activity groups and report group-level metrics.",
        "status_in_this_runner": "metadata only: predictions.csv can be remapped later for group metrics.",
        "deliverables": ["predictions.csv"],
    },
}

DEFAULT_PROPOSED_CANDIDATES = [
    "improved_gnn_lstm",
    "improved_gnn_lstm_attn_adj",
    "gnn_lstm",
    "gnn_flatten_lstm",
    "gnn_learnable_adj",
    "gnn_attention_adj",
    "gnn",
    "lstm",
]

# Models which consume graph sequence datasets.
SEQUENCE_MODELS = {"gnn_lstm", "improved_gnn_lstm", "improved_gnn_lstm_attn_adj", "gnn_flatten_lstm"}
GRAPH_MODELS = {"gnn", "gnn_learnable_adj", "gnn_attention_adj"}
WINDOW_MODELS = {"lstm"}


@dataclass
class JobSpec:
    dataset: str
    model: str
    run_root: str
    epochs: int
    patience: int
    batch_size: int
    max_windows_per_subject: Optional[int]
    device: str
    num_workers: int
    seed: int
    skip_existing: bool
    experiments: List[str]


def repo_root() -> Path:
    return Path.cwd().resolve()


def add_repo_to_path() -> None:
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str) -> torch.device:
    d = (device_arg or "auto").lower().strip()
    if d == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


def safe_json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def remap_labels(y: np.ndarray) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    classes = np.unique(y)
    forward: Dict[str, int] = {str(old): int(new) for new, old in enumerate(classes)}
    inv: Dict[int, str] = {int(new): str(old) for new, old in enumerate(classes)}
    y_new = np.array([forward[str(v)] for v in y], dtype=np.int64)
    return y_new, forward, inv


def load_processed_dataset(name: str, max_windows_per_subject: Optional[int], seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
    add_repo_to_path()
    from src.config import PROCESSED_DIR

    base = Path(PROCESSED_DIR)
    x_path = base / f"{name}_X.npy"
    y_path = base / f"{name}_y.npy"
    s_path = base / f"{name}_subjects.npy"
    missing = [str(p) for p in [x_path, y_path, s_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Processed dataset files are missing: " + ", ".join(missing) + ". Run repo preprocessing first.")

    X = np.load(x_path, allow_pickle=False)
    y_raw = np.load(y_path, allow_pickle=False)
    subjects = np.load(s_path, allow_pickle=False)
    y, mapping, inv = remap_labels(y_raw)

    if max_windows_per_subject is not None and max_windows_per_subject > 0:
        rng = np.random.default_rng(seed)
        keep_parts: List[np.ndarray] = []
        for subj in np.unique(subjects):
            idx = np.where(subjects == subj)[0]
            if len(idx) > max_windows_per_subject:
                idx = rng.choice(idx, max_windows_per_subject, replace=False)
            keep_parts.append(np.sort(idx))
        keep = np.concatenate(keep_parts)
        keep.sort()
        X, y, subjects = X[keep], y[keep], subjects[keep]

    return X, y, subjects, mapping, inv


def get_dataset_meta(dataset: str) -> Tuple[int, int, Any]:
    add_repo_to_path()
    from src.config import PAMAP2_NODE_FEAT_DIM, HHAR_NODE_FEAT_DIM
    from src.graph_construction import build_pamap2_adj, build_hhar_adj
    if dataset == "pamap2":
        return 3, int(PAMAP2_NODE_FEAT_DIM), build_pamap2_adj
    if dataset == "hhar":
        return 2, int(HHAR_NODE_FEAT_DIM), build_hhar_adj
    raise ValueError(dataset)


def import_model_classes() -> Dict[str, Any]:
    add_repo_to_path()
    models_mod = importlib.import_module("src.models")
    result: Dict[str, Any] = {}
    names = {
        "lstm": "LSTMOnlyModel",
        "gnn": "GNNOnlyModel",
        "gnn_lstm": "GNNLSTMModel",
        "improved_gnn_lstm": "ImprovedGNNLSTMModel",
        "improved_gnn_lstm_attn_adj": "ImprovedGNNLSTMAttnAdj",
        "gnn_flatten_lstm": "GNNFlattenLSTMModel",
        "gnn_learnable_adj": "GNNLearnableAdjModel",
        "gnn_attention_adj": "GNNAttentionAdjModel",
    }
    for key, cls_name in names.items():
        if hasattr(models_mod, cls_name):
            result[key] = getattr(models_mod, cls_name)
    return result


def make_datasets(model_name: str, X_tr: np.ndarray, y_tr: np.ndarray, s_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, s_val: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, dataset: str):
    add_repo_to_path()
    from src.dataset import HARWindowDataset, HARGraphDataset, HARSequenceDataset

    if model_name in WINDOW_MODELS:
        return (
            HARWindowDataset(X_tr, y_tr),
            HARWindowDataset(X_val, y_val),
            HARWindowDataset(X_te, y_te),
            False,
        )
    if model_name in GRAPH_MODELS:
        return (
            HARGraphDataset(X_tr, y_tr, dataset=dataset),
            HARGraphDataset(X_val, y_val, dataset=dataset),
            HARGraphDataset(X_te, y_te, dataset=dataset),
            True,
        )
    if model_name in SEQUENCE_MODELS:
        ds_tr = HARSequenceDataset(X_tr, y_tr, subjects=s_tr, dataset=dataset)
        ds_val = HARSequenceDataset(X_val, y_val, subjects=s_val, dataset=dataset)
        ds_te = HARSequenceDataset(X_te, y_te, dataset=dataset)
        # Some small folds may produce no sequences after subject-aware sequencing.
        # Fall back to available data so the fold can still run and be audited.
        if len(ds_tr) == 0:
            ds_tr = HARSequenceDataset(X_tr, y_tr, dataset=dataset)
        if len(ds_val) == 0:
            ds_val = ds_tr
        if len(ds_te) == 0:
            ds_te = HARSequenceDataset(X_te, y_te, subjects=None, dataset=dataset)
        return ds_tr, ds_val, ds_te, True
    raise ValueError(f"Unknown model {model_name}")


def infer_flat_input_dim(X: np.ndarray) -> int:
    if X.ndim == 1:
        return 1
    return int(np.prod(X.shape[1:]))


def build_model(model_name: str, dataset: str, X_sample: np.ndarray, n_classes: int, device: torch.device, adj_fixed: Optional[torch.Tensor]):
    model_classes = import_model_classes()
    if model_name not in model_classes:
        available = ", ".join(sorted(model_classes.keys()))
        raise ValueError(f"Model {model_name!r} not available in src.models. Available: {available}")
    cls = model_classes[model_name]
    n_nodes, node_feat_dim, _ = get_dataset_meta(dataset)

    if model_name == "lstm":
        model = cls(input_dim=infer_flat_input_dim(X_sample), n_classes=n_classes)
    elif model_name in {"gnn", "gnn_lstm", "improved_gnn_lstm", "gnn_flatten_lstm"}:
        model = cls(node_feat_dim=node_feat_dim, n_nodes=n_nodes, n_classes=n_classes)
    elif model_name in {"gnn_learnable_adj", "gnn_attention_adj"}:
        if adj_fixed is None:
            raise ValueError("init_adj required")
        model = cls(node_feat_dim=node_feat_dim, n_nodes=n_nodes, n_classes=n_classes, init_adj=adj_fixed.detach().cpu())
    elif model_name == "improved_gnn_lstm_attn_adj":
        if adj_fixed is None:
            raise ValueError("init_adj required")
        model = cls(node_feat_dim=node_feat_dim, n_nodes=n_nodes, n_classes=n_classes, init_adj=adj_fixed.detach().cpu())
    else:
        raise ValueError(model_name)
    return model.to(device)


def train_one_fold(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, use_adj: bool, device: torch.device, epochs: int, patience: int) -> Tuple[nn.Module, Dict[str, List[float]]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    except TypeError:
        scheduler = None

    best_state = None
    best_val_loss = float("inf")
    stale = 0
    hist: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        t0 = time.time()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if use_adj:
                x, adj, y = batch
                x, adj, y = x.to(device), adj.to(device), y.to(device)
                logits = model(x, adj)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(len(y))
            total_n += int(len(y))
        train_loss = total_loss / max(total_n, 1)

        val_loss, val_acc = evaluate_loss_acc(model, val_loader, criterion, use_adj, device)
        if scheduler is not None:
            scheduler.step(val_loss)
        hist["train_loss"].append(float(train_loss))
        hist["val_loss"].append(float(val_loss))
        hist["val_acc"].append(float(val_acc))
        print(f"epoch={epoch:03d}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} sec={time.time()-t0:.1f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                print(f"early_stop epoch={epoch} best_val_loss={best_val_loss:.4f}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def evaluate_loss_acc(model: nn.Module, loader: DataLoader, criterion: nn.Module, use_adj: bool, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total_n = 0
    for batch in loader:
        if use_adj:
            x, adj, y = batch
            x, adj, y = x.to(device), adj.to(device), y.to(device)
            logits = model(x, adj)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.detach().cpu()) * int(len(y))
        pred = logits.argmax(1)
        correct += int((pred == y).sum().detach().cpu())
        total_n += int(len(y))
    return total_loss / max(total_n, 1), correct / max(total_n, 1)


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, use_adj: bool, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    probs: List[np.ndarray] = []
    for batch in loader:
        if use_adj:
            x, adj, y = batch
            x, adj, y = x.to(device), adj.to(device), y.to(device)
            logits = model(x, adj)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
        p = torch.softmax(logits, dim=1)
        y_true.extend(y.detach().cpu().numpy().astype(int).tolist())
        y_pred.extend(logits.argmax(1).detach().cpu().numpy().astype(int).tolist())
        probs.append(p.detach().cpu().numpy())
    if probs:
        proba = np.concatenate(probs, axis=0)
    else:
        proba = np.empty((0, 0), dtype=np.float32)
    return np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64), proba


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str], title: str, path: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.7), max(6, len(labels) * 0.6)))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    thresh = float(cm.max()) / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_training_curve(history: Dict[str, List[float]], path: Path, title: str) -> None:
    if plt is None:
        return
    if not history.get("train_loss"):
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="train_loss")
    ax.plot(epochs, history["val_loss"], label="val_loss")
    ax2 = ax.twinx()
    ax2.plot(epochs, history["val_acc"], label="val_acc", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax2.set_ylabel("Validation accuracy")
    ax.set_title(title)
    lines, labs = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labs + labs2, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def split_train_val_by_subject(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, test_subj: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_mask = subjects != test_subj
    test_mask = subjects == test_subj
    X_train_all = X[train_mask]
    y_train_all = y[train_mask]
    s_train_all = subjects[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    test_indices = np.where(test_mask)[0]

    train_subjects = np.unique(s_train_all)
    if len(train_subjects) > 1:
        counts = {s: int(np.sum(s_train_all == s)) for s in train_subjects}
        val_subj = min(counts, key=counts.get)
        val_mask = s_train_all == val_subj
        tr_mask = ~val_mask
        X_tr, y_tr, s_tr = X_train_all[tr_mask], y_train_all[tr_mask], s_train_all[tr_mask]
        X_val, y_val, s_val = X_train_all[val_mask], y_train_all[val_mask], s_train_all[val_mask]
    else:
        # Fallback random validation split inside train data.
        n = len(X_train_all)
        idx = np.arange(n)
        np.random.default_rng(SEED).shuffle(idx)
        cut = max(1, int(0.85 * n))
        tr_idx, val_idx = idx[:cut], idx[cut:]
        if len(val_idx) == 0:
            val_idx = tr_idx
        X_tr, y_tr, s_tr = X_train_all[tr_idx], y_train_all[tr_idx], s_train_all[tr_idx]
        X_val, y_val, s_val = X_train_all[val_idx], y_train_all[val_idx], s_train_all[val_idx]
    return X_tr, y_tr, s_tr, X_val, y_val, s_val, X_test, y_test, test_indices


def worker_run(spec: JobSpec) -> int:
    t_start = time.time()
    set_seed(spec.seed)
    add_repo_to_path()
    device = select_device(spec.device)
    dataset = spec.dataset
    model_name = spec.model
    out_dir = Path(spec.run_root) / dataset / model_name
    if spec.skip_existing and (out_dir / "DONE.json").exists():
        print(f"[SKIP] {dataset}/{model_name} already has DONE.json", flush=True)
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    (out_dir / "fold_predictions").mkdir(exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    safe_json_dump({
        "job": asdict(spec),
        "experiment_catalog": {k: EXPERIMENT_CATALOG[k] for k in spec.experiments if k in EXPERIMENT_CATALOG},
        "all_experiments_reference": EXPERIMENT_CATALOG,
        "created_at": now_stamp(),
        "repo_root": str(repo_root()),
    }, out_dir / "run_manifest.json")

    print(f"[START] dataset={dataset} model={model_name} device={device} out={out_dir}", flush=True)
    X, y, subjects, label_mapping, inv_label_mapping = load_processed_dataset(
        dataset,
        max_windows_per_subject=spec.max_windows_per_subject if dataset == "hhar" else None,
        seed=spec.seed,
    )
    n_classes = int(len(np.unique(y)))
    n_nodes, node_feat_dim, adj_builder = get_dataset_meta(dataset)
    adj_fixed = adj_builder().to(device)
    safe_json_dump({
        "dataset": dataset,
        "model": model_name,
        "shape_X": list(X.shape),
        "n_classes": n_classes,
        "n_subjects": int(len(np.unique(subjects))),
        "subjects": [str(s) for s in np.unique(subjects).tolist()],
        "label_mapping_raw_to_encoded": label_mapping,
        "encoded_to_raw_label": inv_label_mapping,
        "n_nodes": n_nodes,
        "node_feat_dim": node_feat_dim,
        "device": str(device),
    }, out_dir / "dataset_manifest.json")

    all_true: List[int] = []
    all_pred: List[int] = []
    all_pred_rows: List[pd.DataFrame] = []
    fold_rows: List[Dict[str, Any]] = []

    unique_subjects = np.unique(subjects)
    for fold_i, test_subj in enumerate(unique_subjects, 1):
        fold_name = f"fold_{fold_i:02d}_subject_{str(test_subj).replace(os.sep, '_')}"
        ckpt_path = out_dir / "checkpoints" / f"{fold_name}.pt"
        pred_path = out_dir / "fold_predictions" / f"{fold_name}_predictions.csv"
        hist_path = out_dir / "fold_predictions" / f"{fold_name}_history.json"
        print(f"\n[{dataset}/{model_name}] {fold_name} ({fold_i}/{len(unique_subjects)})", flush=True)

        X_tr, y_tr, s_tr, X_val, y_val, s_val, X_te, y_te, test_indices = split_train_val_by_subject(X, y, subjects, test_subj)
        tr_ds, val_ds, te_ds, use_adj = make_datasets(model_name, X_tr, y_tr, s_tr, X_val, y_val, s_val, X_te, y_te, dataset)
        tr_loader = DataLoader(tr_ds, batch_size=spec.batch_size, shuffle=True, num_workers=spec.num_workers, pin_memory=(device.type == "cuda"))
        val_loader = DataLoader(val_ds, batch_size=spec.batch_size, shuffle=False, num_workers=spec.num_workers, pin_memory=(device.type == "cuda"))
        te_loader = DataLoader(te_ds, batch_size=spec.batch_size, shuffle=False, num_workers=spec.num_workers, pin_memory=(device.type == "cuda"))

        model = build_model(model_name, dataset, X_tr, n_classes, device, adj_fixed)
        model, history = train_one_fold(model, tr_loader, val_loader, use_adj, device, spec.epochs, spec.patience)
        torch.save({
            "model_state_dict": model.state_dict(),
            "dataset": dataset,
            "model": model_name,
            "fold_subject": str(test_subj),
            "n_classes": n_classes,
            "n_nodes": n_nodes,
            "node_feat_dim": node_feat_dim,
            "label_mapping": label_mapping,
        }, ckpt_path)
        safe_json_dump(history, hist_path)
        write_training_curve(history, out_dir / "plots" / f"{fold_name}_training_curve.png", f"{dataset} {model_name} {fold_name}")

        y_true, y_pred, proba = predict_loader(model, te_loader, use_adj, device)
        m = metrics(y_true, y_pred)
        print(f"[{dataset}/{model_name}] {fold_name} acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} bacc={m['balanced_accuracy']:.4f}", flush=True)

        # Dataset classes may change the number/order of test samples for sequence models.
        # Use sequential ids for rows and keep original test indices when lengths match.
        n_pred = len(y_true)
        original_idx = test_indices[:n_pred] if len(test_indices) >= n_pred else np.arange(n_pred)
        row_df = pd.DataFrame({
            "dataset": dataset,
            "model": model_name,
            "fold": fold_i,
            "fold_subject": str(test_subj),
            "row_in_fold": np.arange(n_pred),
            "original_window_index": original_idx,
            "y_true_id": y_true,
            "y_pred_id": y_pred,
            "y_true_raw": [inv_label_mapping.get(int(v), str(v)) for v in y_true],
            "y_pred_raw": [inv_label_mapping.get(int(v), str(v)) for v in y_pred],
            "correct": y_true == y_pred,
        })
        if proba.size and proba.shape[0] == n_pred:
            for c in range(proba.shape[1]):
                row_df[f"proba_{c}"] = proba[:, c]
            row_df["confidence"] = proba.max(axis=1)
        row_df.to_csv(pred_path, index=False)
        all_pred_rows.append(row_df)
        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())
        fold_rows.append({
            "dataset": dataset,
            "model": model_name,
            "fold": fold_i,
            "fold_subject": str(test_subj),
            "n_train_windows": int(len(X_tr)),
            "n_val_windows": int(len(X_val)),
            "n_test_windows_raw": int(len(X_te)),
            "n_test_predictions": int(n_pred),
            "n_test_classes": int(len(np.unique(y_true))) if n_pred else 0,
            **m,
        })

    y_true_all = np.asarray(all_true, dtype=np.int64)
    y_pred_all = np.asarray(all_pred, dtype=np.int64)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "metrics_by_fold.csv", index=False)
    pred_df = pd.concat(all_pred_rows, ignore_index=True) if all_pred_rows else pd.DataFrame()
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    agg = metrics(y_true_all, y_pred_all) if len(y_true_all) else {}
    summary = {
        "dataset": dataset,
        "model": model_name,
        "n_samples": int(len(y_true_all)),
        "n_folds": int(len(fold_df)),
        "runtime_sec": float(time.time() - t_start),
        **agg,
        "subject_macro_accuracy_mean": float(fold_df["accuracy"].mean()) if not fold_df.empty else math.nan,
        "subject_macro_accuracy_std": float(fold_df["accuracy"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        "subject_macro_f1_mean": float(fold_df["macro_f1"].mean()) if not fold_df.empty else math.nan,
        "subject_macro_f1_std": float(fold_df["macro_f1"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
    }
    pd.DataFrame([summary]).to_csv(out_dir / "metrics_summary.csv", index=False)
    safe_json_dump(summary, out_dir / "metrics_summary.json")

    labels = list(range(n_classes))
    label_names = [inv_label_mapping.get(int(i), str(i)) for i in labels]
    if len(y_true_all):
        rep = classification_report(y_true_all, y_pred_all, labels=labels, target_names=label_names, output_dict=True, zero_division=0)
        pd.DataFrame(rep).T.to_csv(out_dir / "classification_report.csv")
        safe_json_dump(rep, out_dir / "classification_report.json")
        cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
        pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(out_dir / "confusion_matrix.csv")
        plot_confusion_matrix(cm, label_names, f"{dataset} {model_name} confusion matrix", out_dir / "confusion_matrix.png")

    # Experiment 5: subject-level failure analysis.
    if not pred_df.empty:
        subj_rows: List[Dict[str, Any]] = []
        for subj, g in pred_df.groupby("fold_subject"):
            yt = g["y_true_id"].to_numpy(dtype=int)
            yp = g["y_pred_id"].to_numpy(dtype=int)
            wrong = g[g["correct"] == False]
            dominant_confusion = ""
            if not wrong.empty:
                pairs = wrong.groupby(["y_true_raw", "y_pred_raw"]).size().sort_values(ascending=False)
                if len(pairs):
                    (true_lbl, pred_lbl), count = pairs.index[0], int(pairs.iloc[0])
                    dominant_confusion = f"{true_lbl}->{pred_lbl} ({count})"
            subj_rows.append({
                "dataset": dataset,
                "model": model_name,
                "subject": str(subj),
                "n_predictions": int(len(g)),
                "n_true_classes": int(g["y_true_id"].nunique()),
                "accuracy": float(accuracy_score(yt, yp)),
                "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
                "dominant_confusion": dominant_confusion,
            })
        pd.DataFrame(subj_rows).sort_values("macro_f1").to_csv(out_dir / "subject_failure_analysis.csv", index=False)

    safe_json_dump({"status": "done", "completed_at": now_stamp(), "summary": summary}, out_dir / "DONE.json")
    print(f"[DONE] dataset={dataset} model={model_name} summary={summary}", flush=True)
    return 0


def load_ranked_models_from_existing(rank_source: Optional[str], datasets: Sequence[str], metric: str) -> List[str]:
    rows: List[Tuple[str, float]] = []
    candidates = set(DEFAULT_PROPOSED_CANDIDATES)

    paths: List[Path] = []
    if rank_source:
        p = Path(rank_source)
        if p.exists():
            if p.is_dir():
                paths += list(p.rglob("metrics_summary.csv"))
                paths += list(p.rglob("*_deep_models.json"))
            else:
                paths.append(p)
    # Default places.
    paths += list(Path("results").rglob("metrics_summary.csv")) if Path("results").exists() else []
    metrics_dir = Path("results") / "metrics"
    if metrics_dir.exists():
        paths += list(metrics_dir.glob("*_deep_models.json"))

    seen_paths = set()
    for p in paths:
        if p in seen_paths or not p.exists():
            continue
        seen_paths.add(p)
        try:
            if p.suffix.lower() == ".csv":
                df = pd.read_csv(p)
                if "model" in df.columns and metric in df.columns:
                    for _, r in df.iterrows():
                        m = str(r["model"])
                        if m in candidates:
                            rows.append((m, float(r[metric])))
            elif p.suffix.lower() == ".json":
                data = read_json(p)
                for model, vals in data.items():
                    if model in candidates and isinstance(vals, dict) and metric in vals:
                        rows.append((model, float(vals[metric])))
        except Exception:
            continue

    if not rows:
        return []
    best: Dict[str, float] = {}
    for m, v in rows:
        best[m] = max(v, best.get(m, -1e9))
    return [m for m, _ in sorted(best.items(), key=lambda kv: kv[1], reverse=True)]


def resolve_models(models_arg: str, top_k: int, rank_source: Optional[str], datasets: Sequence[str], metric: str) -> List[str]:
    if models_arg and models_arg.lower() != "auto":
        models = [m.strip() for m in models_arg.replace(";", ",").split(",") if m.strip()]
    else:
        ranked = load_ranked_models_from_existing(rank_source, datasets, metric)
        if ranked:
            models = ranked[:top_k]
        else:
            models = DEFAULT_PROPOSED_CANDIDATES[:top_k]
    available = import_model_classes()
    filtered = []
    for m in models:
        if m in available:
            filtered.append(m)
        else:
            print(f"[WARN] selected model {m} is not importable from src.models; skipping", flush=True)
    if not filtered:
        raise ValueError("No runnable models selected. Available: " + ", ".join(sorted(available.keys())))
    return filtered


def write_job_specs(run_root: Path, specs: List[JobSpec]) -> None:
    safe_json_dump({
        "run_root": str(run_root),
        "jobs": [asdict(s) for s in specs],
        "experiment_catalog": EXPERIMENT_CATALOG,
    }, run_root / "launcher_manifest.json")


def launch_jobs(specs: List[JobSpec], parallel_jobs: int, extra_env: Dict[str, str]) -> int:
    script_path = Path(__file__).resolve()
    queue = list(specs)
    running: List[Tuple[subprocess.Popen, JobSpec, Path, Path]] = []
    failures = 0

    def start_one(spec: JobSpec) -> None:
        nonlocal running
        log_dir = Path(spec.run_root) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{spec.dataset}_{spec.model}"
        out_log = log_dir / f"{tag}_stdout.log"
        err_log = log_dir / f"{tag}_stderr.log"
        cmd = [
            sys.executable,
            str(script_path),
            "--worker",
            "--dataset", spec.dataset,
            "--model", spec.model,
            "--run-root", spec.run_root,
            "--epochs", str(spec.epochs),
            "--patience", str(spec.patience),
            "--batch-size", str(spec.batch_size),
            "--device", spec.device,
            "--num-workers", str(spec.num_workers),
            "--seed", str(spec.seed),
            "--experiments", ",".join(spec.experiments),
        ]
        if spec.max_windows_per_subject is not None:
            cmd += ["--max-windows-per-subject", str(spec.max_windows_per_subject)]
        if spec.skip_existing:
            cmd += ["--skip-existing"]
        env = os.environ.copy()
        env.update(extra_env)
        f_out = open(out_log, "w", encoding="utf-8")
        f_err = open(err_log, "w", encoding="utf-8")
        print(f"[LAUNCH] {tag} logs={out_log}", flush=True)
        p = subprocess.Popen(cmd, stdout=f_out, stderr=f_err, cwd=str(repo_root()), env=env)
        running.append((p, spec, out_log, err_log))

    while queue or running:
        while queue and len(running) < max(1, parallel_jobs):
            start_one(queue.pop(0))
        time.sleep(5)
        still: List[Tuple[subprocess.Popen, JobSpec, Path, Path]] = []
        for p, spec, out_log, err_log in running:
            rc = p.poll()
            if rc is None:
                still.append((p, spec, out_log, err_log))
            else:
                tag = f"{spec.dataset}/{spec.model}"
                if rc != 0:
                    failures += 1
                    print(f"[FAIL] {tag} rc={rc} stdout={out_log} stderr={err_log}", flush=True)
                else:
                    print(f"[OK] {tag} stdout={out_log}", flush=True)
        running = still
    return failures


def aggregate_run(run_root: Path) -> None:
    rows = []
    folds = []
    for p in run_root.glob("*/*/metrics_summary.csv"):
        try:
            df = pd.read_csv(p)
            df["artifact_dir"] = str(p.parent)
            rows.append(df)
        except Exception:
            pass
    for p in run_root.glob("*/*/metrics_by_fold.csv"):
        try:
            df = pd.read_csv(p)
            df["artifact_dir"] = str(p.parent)
            folds.append(df)
        except Exception:
            pass
    if rows:
        summary = pd.concat(rows, ignore_index=True)
        sort_col = "macro_f1" if "macro_f1" in summary.columns else summary.columns[0]
        summary = summary.sort_values(["dataset", sort_col], ascending=[True, False])
        summary.to_csv(run_root / "metrics_ranked_all_jobs.csv", index=False)
    if folds:
        fold_all = pd.concat(folds, ignore_index=True)
        fold_all.to_csv(run_root / "metrics_by_fold_all_jobs.csv", index=False)
        # Experiment 2: per model fold mean/std/95pct CI.
        stat_rows = []
        for (dataset, model), g in fold_all.groupby(["dataset", "model"]):
            for metric in ["accuracy", "macro_f1", "balanced_accuracy"]:
                vals = g[metric].dropna().astype(float).to_numpy()
                if len(vals) == 0:
                    continue
                mean = float(vals.mean())
                std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                ci95 = float(1.96 * std / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
                stat_rows.append({"dataset": dataset, "model": model, "metric": metric, "folds": int(len(vals)), "mean": mean, "std": std, "ci95_half_width": ci95, "ci95_low": mean - ci95, "ci95_high": mean + ci95})
        pd.DataFrame(stat_rows).to_csv(run_root / "experiment2_statistical_reliability.csv", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel runner for repo proposed HAR deep algorithms.")
    p.add_argument("--worker", action="store_true", help="Internal worker mode for one dataset/model job.")
    p.add_argument("--datasets", nargs="+", choices=["pamap2", "hhar"], default=["pamap2", "hhar"])
    p.add_argument("--dataset", choices=["pamap2", "hhar"], default=None, help="Worker dataset.")
    p.add_argument("--models", default="auto", help="Comma-separated models or 'auto'.")
    p.add_argument("--model", default=None, help="Worker model.")
    p.add_argument("--top-k", type=int, default=2, help="When --models auto, choose top K proposed models from existing metrics or fallback priority.")
    p.add_argument("--rank-source", default=None, help="Optional metrics file/dir to rank models from.")
    p.add_argument("--rank-metric", default="macro_f1")
    p.add_argument("--run-root", default=None, help="Output root. Launcher creates timestamped one if omitted.")
    p.add_argument("--parallel-jobs", type=int, default=2, help="Max concurrent dataset/model jobs.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-windows-per-subject", type=int, default=5000, help="HHAR cap per subject; ignored for PAMAP2.")
    p.add_argument("--no-hhar-cap", action="store_true", help="Use full HHAR processed data.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--experiments", default="exp2_statistical_reliability,exp4_calibration_uncertainty,exp5_subject_failure", help="Comma-separated experiment IDs to attach/produce.")
    p.add_argument("--cpu-threads-per-job", type=int, default=4, help="Sets OMP/MKL/OPENBLAS thread env per launched job.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    add_repo_to_path()
    set_seed(args.seed)

    experiments = [x.strip() for x in str(args.experiments).split(",") if x.strip()]
    if args.worker:
        if not args.dataset or not args.model or not args.run_root:
            raise SystemExit("Worker mode requires --dataset, --model and --run-root")
        spec = JobSpec(
            dataset=args.dataset,
            model=args.model,
            run_root=str(args.run_root),
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            max_windows_per_subject=None if args.no_hhar_cap else args.max_windows_per_subject,
            device=args.device,
            num_workers=args.num_workers,
            seed=args.seed,
            skip_existing=args.skip_existing,
            experiments=experiments,
        )
        try:
            return worker_run(spec)
        except Exception:
            traceback.print_exc()
            out_dir = Path(args.run_root) / args.dataset / args.model
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_json_dump({"status": "failed", "traceback": traceback.format_exc(), "time": now_stamp()}, out_dir / "FAILED.json")
            return 1

    models = resolve_models(args.models, args.top_k, args.rank_source, args.datasets, args.rank_metric)
    run_root = Path(args.run_root) if args.run_root else Path("results") / "phase2_repo_parallel" / now_stamp()
    run_root.mkdir(parents=True, exist_ok=True)

    specs: List[JobSpec] = []
    for dataset in args.datasets:
        for model in models:
            specs.append(JobSpec(
                dataset=dataset,
                model=model,
                run_root=str(run_root),
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                max_windows_per_subject=None if args.no_hhar_cap else args.max_windows_per_subject,
                device=args.device,
                num_workers=args.num_workers,
                seed=args.seed,
                skip_existing=args.skip_existing,
                experiments=experiments,
            ))
    write_job_specs(run_root, specs)
    print("Selected models:", ", ".join(models), flush=True)
    print("Datasets:", ", ".join(args.datasets), flush=True)
    print("Run root:", run_root, flush=True)
    print("Jobs:", len(specs), "parallel_jobs:", args.parallel_jobs, flush=True)
    extra_env = {
        "OMP_NUM_THREADS": str(args.cpu_threads_per_job),
        "MKL_NUM_THREADS": str(args.cpu_threads_per_job),
        "OPENBLAS_NUM_THREADS": str(args.cpu_threads_per_job),
        "NUMEXPR_NUM_THREADS": str(args.cpu_threads_per_job),
        "PYTHONUNBUFFERED": "1",
    }
    failures = launch_jobs(specs, args.parallel_jobs, extra_env)
    aggregate_run(run_root)
    if failures:
        print(f"Completed with {failures} failed job(s). See logs in {run_root / 'logs'}", flush=True)
        return 1
    print(f"All jobs finished. Aggregated results: {run_root / 'metrics_ranked_all_jobs.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
