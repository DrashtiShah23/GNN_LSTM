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
import warnings
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
    "cnn",
    "lstm",
]

# Models which consume graph sequence datasets.
SEQUENCE_MODELS = {"gnn_lstm", "improved_gnn_lstm", "improved_gnn_lstm_attn_adj", "gnn_flatten_lstm"}
GRAPH_MODELS = {"gnn", "gnn_learnable_adj", "gnn_attention_adj"}
WINDOW_MODELS = {"cnn", "lstm"}
DEFAULT_EXPERIMENTS = (
    "exp1_leakage_nonoverlap,"
    "exp2_statistical_reliability,"
    "exp3_robustness_sensor_failure,"
    "exp4_calibration_uncertainty,"
    "exp5_subject_failure,"
    "exp6_few_shot_calibration,"
    "exp7_health_group_analysis"
)

PAMAP2_ACTIVITY_NAMES = {
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


@dataclass
class JobSpec:
    dataset: str
    model: str
    run_root: str
    processed_dir: Optional[str]
    eval_protocol: str
    epochs: int
    patience: int
    batch_size: int
    max_windows_per_subject: Optional[int]
    no_hhar_cap: bool
    max_windows_per_subject_arg: int
    apply_window_cap_to_all_datasets: bool
    eval_unit: str
    val_strategy: str
    val_subject_policy: str
    early_stop_metric: str
    early_stop_mode: str
    sequence_length: int
    sequence_stride: int
    sequence_target_policy: str
    disable_cudnn_for_sequence_models: bool
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


def display_label_name(dataset: str, encoded_label: int, inv_label_mapping: Dict[int, str]) -> str:
    raw = inv_label_mapping.get(int(encoded_label), str(encoded_label))
    if dataset == "pamap2":
        try:
            return PAMAP2_ACTIVITY_NAMES.get(int(raw), str(raw))
        except Exception:
            return str(raw)
    return str(raw)


def display_label_names(dataset: str, encoded_labels: Sequence[int], inv_label_mapping: Dict[int, str]) -> List[str]:
    return [display_label_name(dataset, int(i), inv_label_mapping) for i in encoded_labels]


def load_processed_dataset(
    name: str,
    max_windows_per_subject: Optional[int],
    seed: int,
    processed_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[int, str], Dict[str, Any]]:
    add_repo_to_path()
    from src.config import PROCESSED_DIR

    base = Path(processed_dir) if processed_dir else Path(PROCESSED_DIR)
    x_path = base / f"{name}_X.npy"
    y_path = base / f"{name}_y.npy"
    s_path = base / f"{name}_subjects.npy"
    source_manifest_path = base / f"{name}_processed_manifest.json"
    missing = [str(p) for p in [x_path, y_path, s_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Processed dataset files are missing: " + ", ".join(missing) + ". Run repo preprocessing first.")

    X = np.load(x_path, allow_pickle=False)
    y_raw = np.load(y_path, allow_pickle=False)
    subjects = np.load(s_path, allow_pickle=False)
    n_source_windows_before_cap = int(len(X))
    source_indices = np.arange(len(X), dtype=np.int64)
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
        source_indices = source_indices[keep]

    manifest = {
        "processed_dataset_dir": str(base),
        "processed_dataset_files": {
            "X": str(x_path),
            "y": str(y_path),
            "subjects": str(s_path),
        },
        "source_processed_manifest_path": str(source_manifest_path) if source_manifest_path.exists() else None,
        "source_processed_manifest": read_json(source_manifest_path) if source_manifest_path.exists() else None,
        "n_source_windows_before_cap": n_source_windows_before_cap,
        "n_source_windows_after_cap": int(len(X)),
        "effective_max_windows_per_subject": max_windows_per_subject if max_windows_per_subject and max_windows_per_subject > 0 else None,
    }
    return X, y, subjects, source_indices, mapping, inv, manifest


def processed_context(data_manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Extract stable dataset identity from the processed manifest/path."""
    source_manifest = data_manifest.get("source_processed_manifest") or {}
    processed_dir = Path(str(data_manifest.get("processed_dataset_dir") or ""))
    return {
        "feature_set": source_manifest.get("feature_set") or (processed_dir.parent.name if processed_dir.parent.name else None),
        "window_type": source_manifest.get("window_type") or (processed_dir.name if processed_dir.name else None),
        "task": source_manifest.get("task"),
        "sessions": source_manifest.get("sessions"),
        "processed_dataset_dir": data_manifest.get("processed_dataset_dir"),
    }


def get_dataset_meta(dataset: str, X_sample: Optional[np.ndarray] = None) -> Tuple[int, int, Any]:
    add_repo_to_path()
    from src.config import HHAR_NODE_FEAT_DIM
    from src.graph_construction import build_pamap2_adj, build_hhar_adj, window_to_node_features_pamap2
    if dataset == "pamap2":
        if X_sample is None:
            raise ValueError("PAMAP2 node feature dimension requires a sample window")
        n_channels = int(X_sample.shape[1])
        node_features = window_to_node_features_pamap2(X_sample)
        return int(node_features.shape[0]), int(node_features.shape[-1]), lambda: build_pamap2_adj(n_channels)
    if dataset == "hhar":
        return 3, int(HHAR_NODE_FEAT_DIM), build_hhar_adj
    raise ValueError(dataset)


def import_model_classes() -> Dict[str, Any]:
    add_repo_to_path()
    models_mod = importlib.import_module("src.models")
    result: Dict[str, Any] = {}
    names = {
        "cnn": "CNN1DModel",
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


def _sequence_alignment_frame(dataset: str, fold_i: int, test_subj: Any, seq_ds: Any) -> pd.DataFrame:
    return pd.DataFrame({
        "dataset": dataset,
        "fold": int(fold_i),
        "test_subject": str(test_subj),
        "eval_sample_id": np.arange(len(seq_ds), dtype=int),
        "sequence_start_source_index": np.asarray(seq_ds.sequence_start_source_indices),
        "sequence_end_source_index": np.asarray(seq_ds.sequence_end_source_indices),
        "target_source_index": np.asarray(seq_ds.target_source_indices),
        "y_true": seq_ds.labels.detach().cpu().numpy().astype(int) if len(seq_ds) else np.asarray([], dtype=int),
    })


def _target_local_indices_from_sequence_dataset(seq_ds: Any) -> np.ndarray:
    if len(seq_ds) == 0:
        return np.asarray([], dtype=np.int64)
    return np.asarray([int(seq[-1]) for seq in seq_ds._seq_indices], dtype=np.int64)


def make_datasets(
    model_name: str,
    eval_unit: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    s_tr: np.ndarray,
    src_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    s_val: np.ndarray,
    src_val: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    s_te: np.ndarray,
    src_te: np.ndarray,
    dataset: str,
    sequence_length: int,
    sequence_stride: int,
    sequence_target_policy: str,
    fold_i: int,
    test_subj: Any,
):
    add_repo_to_path()
    from src.dataset import HARWindowDataset, HARWindowDataset2D, HARGraphDataset, HARSequenceDataset

    if model_name in WINDOW_MODELS:
        if model_name in {"cnn", "lstm"}:
            return (
                HARWindowDataset2D(X_tr, y_tr),
                HARWindowDataset2D(X_val, y_val),
                HARWindowDataset2D(X_te, y_te),
                False,
                {},
            )
        return (
            HARWindowDataset(X_tr, y_tr),
            HARWindowDataset(X_val, y_val),
            HARWindowDataset(X_te, y_te),
            False,
            {},
        )
    if model_name in GRAPH_MODELS:
        if eval_unit == "sequence_aligned":
            seq_val = HARSequenceDataset(
                X_val, y_val, subjects=s_val, dataset=dataset, seq_len=sequence_length,
                seq_stride=sequence_stride, target_policy=sequence_target_policy, source_indices=src_val,
                cache=False,
            )
            seq_te = HARSequenceDataset(
                X_te, y_te, subjects=s_te, dataset=dataset, seq_len=sequence_length,
                seq_stride=sequence_stride, target_policy=sequence_target_policy, source_indices=src_te,
                cache=False,
            )
            val_target_local = _target_local_indices_from_sequence_dataset(seq_val)
            te_target_local = _target_local_indices_from_sequence_dataset(seq_te)
            return (
                HARGraphDataset(X_tr, y_tr, dataset=dataset),
                HARGraphDataset(X_val[val_target_local], seq_val.labels.detach().cpu().numpy(), dataset=dataset),
                HARGraphDataset(X_te[te_target_local], seq_te.labels.detach().cpu().numpy(), dataset=dataset),
                True,
                {
                    "val_target_source_indices": np.asarray(seq_val.target_source_indices),
                    "test_target_source_indices": np.asarray(seq_te.target_source_indices),
                    "test_alignment": _sequence_alignment_frame(dataset, fold_i, test_subj, seq_te),
                },
            )
        return (
            HARGraphDataset(X_tr, y_tr, dataset=dataset),
            HARGraphDataset(X_val, y_val, dataset=dataset),
            HARGraphDataset(X_te, y_te, dataset=dataset),
            True,
            {},
        )
    if model_name in SEQUENCE_MODELS:
        ds_tr = HARSequenceDataset(
            X_tr, y_tr, subjects=s_tr, dataset=dataset, seq_len=sequence_length,
            seq_stride=sequence_stride, target_policy=sequence_target_policy, source_indices=src_tr,
        )
        ds_val = HARSequenceDataset(
            X_val, y_val, subjects=s_val, dataset=dataset, seq_len=sequence_length,
            seq_stride=sequence_stride, target_policy=sequence_target_policy, source_indices=src_val,
        )
        ds_te = HARSequenceDataset(
            X_te, y_te, subjects=s_te, dataset=dataset, seq_len=sequence_length,
            seq_stride=sequence_stride, target_policy=sequence_target_policy, source_indices=src_te,
        )
        if len(ds_tr) == 0 or len(ds_val) == 0 or len(ds_te) == 0:
            raise ValueError(
                f"Sequence dataset produced an empty split with sequence_length={sequence_length}, "
                f"sequence_stride={sequence_stride}; train={len(ds_tr)} val={len(ds_val)} test={len(ds_te)}"
            )
        return ds_tr, ds_val, ds_te, True, {
            "train_target_source_indices": np.asarray(ds_tr.target_source_indices),
            "val_target_source_indices": np.asarray(ds_val.target_source_indices),
            "test_target_source_indices": np.asarray(ds_te.target_source_indices),
            "test_alignment": _sequence_alignment_frame(dataset, fold_i, test_subj, ds_te),
        }
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
    sample_window = X_sample[0] if isinstance(X_sample, np.ndarray) and X_sample.ndim == 3 else X_sample
    n_nodes, node_feat_dim, _ = get_dataset_meta(dataset, X_sample=sample_window)

    if model_name == "lstm":
        if sample_window.ndim != 2:
            raise ValueError(f"LSTMOnlyModel expects a single raw window sample with shape (timesteps, channels); got {sample_window.shape}")
        model = cls(input_dim=int(sample_window.shape[1]), n_classes=n_classes)
    elif model_name == "cnn":
        if sample_window.ndim != 2:
            raise ValueError(f"CNN1DModel expects a single raw window sample with shape (timesteps, channels); got {sample_window.shape}")
        model = cls(n_timesteps=int(sample_window.shape[0]), n_channels=int(sample_window.shape[1]), n_classes=n_classes)
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


def model_parameter_profile(
    model: nn.Module,
    *,
    dataset: str,
    model_name: str,
    eval_unit: str,
    raw_window_shape: Sequence[int],
    n_nodes: int,
    node_feat_dim: int,
    n_classes: int,
) -> Dict[str, Any]:
    total_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    non_trainable_params = int(total_params - trainable_params)
    buffers = int(sum(b.numel() for b in model.buffers()))
    return {
        "dataset": dataset,
        "model": model_name,
        "eval_unit": eval_unit,
        "raw_window_shape": [int(x) for x in raw_window_shape],
        "n_nodes": int(n_nodes),
        "node_feat_dim": int(node_feat_dim),
        "n_classes": int(n_classes),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "buffer_values": buffers,
        "parameter_size_mb_float32": float(total_params * 4 / (1024 ** 2)),
        "profile_method": "computed from instantiated torch module using sum(p.numel())",
    }


def resolve_early_stop_mode(metric_name: str, mode: str) -> str:
    mode = str(mode).lower()
    if mode in {"min", "max"}:
        return mode
    metric_name = str(metric_name).lower()
    return "min" if metric_name.endswith("loss") or metric_name == "val_loss" else "max"


def _is_better(value: float, best: float, mode: str) -> bool:
    return value < best if mode == "min" else value > best


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    use_adj: bool,
    device: torch.device,
    epochs: int,
    patience: int,
    n_classes: int,
    early_stop_metric: str,
    early_stop_mode: str,
) -> Tuple[nn.Module, Dict[str, List[float]], Dict[str, Any], Dict[str, torch.Tensor]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    except TypeError:
        scheduler = None

    resolved_mode = resolve_early_stop_mode(early_stop_metric, early_stop_mode)
    best_state = None
    last_state: Dict[str, torch.Tensor] = {}
    best_metric = float("inf") if resolved_mode == "min" else -float("inf")
    best_epoch = 0
    stale = 0
    hist: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
    }

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        train_true: List[int] = []
        train_pred: List[int] = []
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
            train_true.extend(y.detach().cpu().numpy().astype(int).tolist())
            train_pred.extend(logits.argmax(1).detach().cpu().numpy().astype(int).tolist())
        train_loss = total_loss / max(total_n, 1)
        train_m = metrics(np.asarray(train_true, dtype=np.int64), np.asarray(train_pred, dtype=np.int64), n_classes)

        val_loss, val_acc, val_macro_f1 = evaluate_loss_acc_f1(model, val_loader, criterion, use_adj, device, n_classes)
        if scheduler is not None:
            scheduler.step(val_loss)
        hist["train_loss"].append(float(train_loss))
        hist["val_loss"].append(float(val_loss))
        hist["train_acc"].append(float(train_m["accuracy"]))
        hist["val_acc"].append(float(val_acc))
        hist["train_macro_f1"].append(float(train_m["macro_f1"]))
        hist["val_macro_f1"].append(float(val_macro_f1))
        last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        metric_value = {
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_macro_f1": float(val_macro_f1),
        }.get(str(early_stop_metric), float(val_macro_f1))
        print(
            f"epoch={epoch:03d}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_f1={train_m['macro_f1']:.4f} val_f1={val_macro_f1:.4f} "
            f"val_acc={val_acc:.4f} sec={time.time()-t0:.1f}",
            flush=True,
        )

        if _is_better(metric_value, best_metric, resolved_mode):
            best_metric = metric_value
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                print(f"early_stop epoch={epoch} best_{early_stop_metric}={best_metric:.4f}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    meta = {
        "best_epoch": int(best_epoch or len(hist["train_loss"])),
        "last_epoch": int(len(hist["train_loss"])),
        "early_stop_metric": early_stop_metric,
        "early_stop_mode": resolved_mode,
        "best_metric_value": float(best_metric),
    }
    return model, hist, meta, last_state


@torch.no_grad()
def evaluate_loss_acc_f1(model: nn.Module, loader: DataLoader, criterion: nn.Module, use_adj: bool, device: torch.device, n_classes: int) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total_n = 0
    all_true: List[int] = []
    all_pred: List[int] = []
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
        all_true.extend(y.detach().cpu().numpy().astype(int).tolist())
        all_pred.extend(pred.detach().cpu().numpy().astype(int).tolist())
    f1 = metrics(np.asarray(all_true, dtype=np.int64), np.asarray(all_pred, dtype=np.int64), n_classes)["macro_f1"] if total_n else 0.0
    return total_loss / max(total_n, 1), correct / max(total_n, 1), float(f1)


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


def metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: Optional[int] = None) -> Dict[str, float]:
    labels = list(range(int(n_classes))) if n_classes is not None else sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
        }
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1)
    recalls = np.divide(np.diag(cm), row_sums, out=np.zeros(len(labels), dtype=float), where=row_sums != 0)
    present = row_sums != 0
    balanced = float(recalls[present].mean()) if present.any() else 0.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*y_pred contains classes not in y_true.*")
        warnings.filterwarnings("ignore", message=".*ill-defined.*")
        macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        macro_precision = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        macro_recall = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": balanced,
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
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


def choose_validation_subject(train_subjects: np.ndarray, subjects: np.ndarray, fold_index: int, policy: str, seed: int) -> Any:
    policy = str(policy).lower()
    ordered = list(train_subjects)
    if policy == "round_robin":
        return ordered[(fold_index - 1) % len(ordered)]
    counts = {s: int(np.sum(subjects == s)) for s in ordered}
    if policy == "min_count":
        return min(ordered, key=lambda s: (counts[s], str(s)))
    if policy == "max_count":
        return max(ordered, key=lambda s: (counts[s], str(s)))
    if policy == "random":
        rng = np.random.default_rng(seed + fold_index)
        return ordered[int(rng.integers(0, len(ordered)))]
    raise ValueError(f"Unknown validation subject policy: {policy}")


def split_train_val_by_subject(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    source_indices: np.ndarray,
    test_subj: Any,
    fold_index: int,
    val_strategy: str,
    val_subject_policy: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, List[Any]]:
    if str(val_strategy).lower() != "inner_subject":
        raise ValueError("Only --val-strategy inner_subject is supported for strict nested LOSO.")
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) < 3:
        raise ValueError("Strict nested LOSO requires at least three subjects.")
    train_mask = subjects != test_subj
    test_mask = subjects == test_subj
    X_train_all = X[train_mask]
    y_train_all = y[train_mask]
    s_train_all = subjects[train_mask]
    src_train_all = source_indices[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    s_test = subjects[test_mask]
    src_test = source_indices[test_mask]
    test_indices = np.where(test_mask)[0]

    train_subjects = np.unique(s_train_all)
    if len(train_subjects) < 2:
        raise ValueError("Strict nested LOSO requires at least two non-test subjects.")
    val_subj = choose_validation_subject(train_subjects, s_train_all, fold_index, val_subject_policy, seed)
    val_mask = s_train_all == val_subj
    tr_mask = ~val_mask
    X_tr, y_tr, s_tr, src_tr = X_train_all[tr_mask], y_train_all[tr_mask], s_train_all[tr_mask], src_train_all[tr_mask]
    X_val, y_val, s_val, src_val = X_train_all[val_mask], y_train_all[val_mask], s_train_all[val_mask], src_train_all[val_mask]
    nested_train_subjects = [s for s in np.unique(s_tr).tolist()]
    assert test_subj != val_subj
    assert test_subj not in nested_train_subjects
    assert val_subj not in nested_train_subjects
    return X_tr, y_tr, s_tr, src_tr, X_val, y_val, s_val, src_val, X_test, y_test, s_test, src_test, test_indices, val_subj, nested_train_subjects


def random_train_val_test_indices(
    n_items: int,
    *,
    seed: int,
    test_fraction: float = 0.2,
    val_fraction: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_items < 3:
        raise ValueError(f"Need at least 3 samples for random holdout, got {n_items}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_items)
    n_test = max(1, int(round(n_items * test_fraction)))
    remaining = perm[n_test:]
    n_val = max(1, int(round(len(remaining) * val_fraction)))
    test_idx = np.sort(perm[:n_test])
    val_idx = np.sort(remaining[:n_val])
    train_idx = np.sort(remaining[n_val:])
    if len(train_idx) == 0:
        raise ValueError("Random holdout produced empty train split")
    return train_idx, val_idx, test_idx


def write_training_curve_csv(history: Dict[str, List[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    n_epochs = len(history.get("train_loss", []))
    for i in range(n_epochs):
        row = {"epoch": i + 1}
        for key, values in history.items():
            row[key] = values[i] if i < len(values) else math.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def overfitting_row(
    dataset: str,
    model_name: str,
    eval_unit: str,
    fold_i: int,
    test_subj: Any,
    val_subj: Any,
    train_meta: Dict[str, Any],
    history: Dict[str, List[float]],
    test_metrics: Dict[str, float],
) -> Dict[str, Any]:
    best_epoch = int(train_meta.get("best_epoch", 1))
    last_epoch = int(train_meta.get("last_epoch", len(history.get("train_loss", []))))
    best_idx = max(0, min(best_epoch - 1, len(history.get("train_loss", [0])) - 1))
    last_idx = max(0, min(last_epoch - 1, len(history.get("train_loss", [0])) - 1))

    def h(key: str, idx: int) -> float:
        vals = history.get(key, [])
        return float(vals[idx]) if vals and idx < len(vals) else math.nan

    train_loss_at_best = h("train_loss", best_idx)
    val_loss_at_best = h("val_loss", best_idx)
    train_acc_at_best = h("train_acc", best_idx)
    val_acc_at_best = h("val_acc", best_idx)
    train_macro_f1_at_best = h("train_macro_f1", best_idx)
    val_macro_f1_at_best = h("val_macro_f1", best_idx)
    last_val_macro_f1 = h("val_macro_f1", last_idx)
    train_val_loss_gap = val_loss_at_best - train_loss_at_best
    train_val_acc_gap = train_acc_at_best - val_acc_at_best
    train_val_macro_f1_gap = train_macro_f1_at_best - val_macro_f1_at_best
    val_test_macro_f1_gap = val_macro_f1_at_best - float(test_metrics.get("macro_f1", 0.0))
    val_drop = val_macro_f1_at_best - last_val_macro_f1
    risk = max(0.0, train_val_macro_f1_gap) + max(0.0, val_test_macro_f1_gap) + max(0.0, val_drop)
    return {
        "dataset": dataset,
        "model": model_name,
        "eval_unit": eval_unit,
        "fold": fold_i,
        "test_subject": str(test_subj),
        "validation_subject": str(val_subj),
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "early_stop_metric": train_meta.get("early_stop_metric"),
        "early_stop_mode": train_meta.get("early_stop_mode"),
        "train_loss_at_best": train_loss_at_best,
        "val_loss_at_best": val_loss_at_best,
        "train_acc_at_best": train_acc_at_best,
        "val_acc_at_best": val_acc_at_best,
        "train_macro_f1_at_best": train_macro_f1_at_best,
        "val_macro_f1_at_best": val_macro_f1_at_best,
        "test_acc": float(test_metrics.get("accuracy", 0.0)),
        "test_macro_f1": float(test_metrics.get("macro_f1", 0.0)),
        "test_balanced_accuracy": float(test_metrics.get("balanced_accuracy", 0.0)),
        "train_val_loss_gap_at_best": train_val_loss_gap,
        "train_val_acc_gap_at_best": train_val_acc_gap,
        "train_val_macro_f1_gap_at_best": train_val_macro_f1_gap,
        "val_test_macro_f1_gap": val_test_macro_f1_gap,
        "val_macro_f1_drop_best_to_last": val_drop,
        "overfit_risk_score": risk,
        "overfit_flag_f1_gap_gt_0_15": bool(train_val_macro_f1_gap > 0.15),
        "overfit_flag_val_drop_gt_0_10": bool(val_drop > 0.10),
        "overfit_flag_val_test_gap_gt_0_15": bool(val_test_macro_f1_gap > 0.15),
    }


def write_overfitting_summary(overfit_df: pd.DataFrame, path: Path) -> None:
    if overfit_df.empty:
        pd.DataFrame().to_csv(path, index=False)
        return
    summary = overfit_df.groupby(["dataset", "model", "eval_unit"], dropna=False).agg(
        folds=("fold", "count"),
        overfit_risk_score_mean=("overfit_risk_score", "mean"),
        overfit_risk_score_max=("overfit_risk_score", "max"),
        train_val_macro_f1_gap_mean=("train_val_macro_f1_gap_at_best", "mean"),
        val_test_macro_f1_gap_mean=("val_test_macro_f1_gap", "mean"),
        val_macro_f1_drop_mean=("val_macro_f1_drop_best_to_last", "mean"),
        overfit_flag_f1_gap_gt_0_15_rate=("overfit_flag_f1_gap_gt_0_15", "mean"),
        overfit_flag_val_drop_gt_0_10_rate=("overfit_flag_val_drop_gt_0_10", "mean"),
        overfit_flag_val_test_gap_gt_0_15_rate=("overfit_flag_val_test_gap_gt_0_15", "mean"),
    ).reset_index()
    summary.to_csv(path, index=False)


def worker_run(spec: JobSpec) -> int:
    t_start = time.time()
    set_seed(spec.seed)
    add_repo_to_path()
    device = select_device(spec.device)
    dataset = spec.dataset
    model_name = spec.model
    eval_unit = spec.eval_unit
    eval_protocol = str(spec.eval_protocol).lower()
    if eval_protocol not in {"loso", "random_holdout"}:
        raise ValueError(f"Unsupported eval_protocol: {spec.eval_protocol}")
    if spec.disable_cudnn_for_sequence_models and model_name in SEQUENCE_MODELS and device.type == "cuda":
        torch.backends.cudnn.enabled = False
        print("[INFO] cuDNN disabled for sequence model worker", flush=True)
    out_dir = Path(spec.run_root) / dataset / model_name / eval_unit
    if spec.skip_existing and (out_dir / "DONE.json").exists():
        print(f"[SKIP] {dataset}/{model_name}/{eval_unit} already has DONE.json", flush=True)
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

    if spec.apply_window_cap_to_all_datasets:
        effective_hhar_cap = spec.max_windows_per_subject
    elif dataset == "hhar":
        effective_hhar_cap = None if spec.no_hhar_cap else spec.max_windows_per_subject
    else:
        effective_hhar_cap = None
    print(
        f"[START] dataset={dataset} model={model_name} eval_unit={eval_unit} device={device} "
        f"hhar_effective_cap={effective_hhar_cap} out={out_dir}",
        flush=True,
    )
    X, y, subjects, source_indices, label_mapping, inv_label_mapping, data_manifest = load_processed_dataset(
        dataset,
        max_windows_per_subject=effective_hhar_cap,
        seed=spec.seed,
        processed_dir=spec.processed_dir,
    )
    proc_ctx = processed_context(data_manifest)
    n_classes = int(len(np.unique(y)))
    n_nodes, node_feat_dim, adj_builder = get_dataset_meta(dataset, X_sample=X[0])
    adj_fixed = adj_builder().to(device)
    profile_model = build_model(model_name, dataset, X[:1], n_classes, device, adj_fixed)
    model_profile = model_parameter_profile(
        profile_model,
        dataset=dataset,
        model_name=model_name,
        eval_unit=eval_unit,
        raw_window_shape=X.shape[1:],
        n_nodes=n_nodes,
        node_feat_dim=node_feat_dim,
        n_classes=n_classes,
    )
    del profile_model
    safe_json_dump(model_profile, out_dir / "model_profile.json")
    encoded_labels = list(range(n_classes))
    label_names = display_label_names(dataset, encoded_labels, inv_label_mapping)
    safe_json_dump({
        "dataset": dataset,
        **proc_ctx,
        "model": model_name,
        "eval_unit": eval_unit,
        "eval_protocol": eval_protocol,
        "source_processed_manifest_path": data_manifest.get("source_processed_manifest_path"),
        "source_processed_manifest": data_manifest.get("source_processed_manifest"),
        "processed_dataset_dir": data_manifest.get("processed_dataset_dir"),
        "processed_dataset_files": data_manifest.get("processed_dataset_files"),
        "no_hhar_cap": bool(spec.no_hhar_cap),
        "max_windows_per_subject_arg": int(spec.max_windows_per_subject_arg),
        "effective_max_windows_per_subject": data_manifest["effective_max_windows_per_subject"],
        "n_source_windows_before_cap": data_manifest["n_source_windows_before_cap"],
        "n_source_windows_after_cap": data_manifest["n_source_windows_after_cap"],
        "sequence_length": int(spec.sequence_length),
        "sequence_stride": int(spec.sequence_stride),
        "sequence_target_policy": spec.sequence_target_policy,
        "val_strategy": spec.val_strategy,
        "val_subject_policy": spec.val_subject_policy,
        "early_stop_metric": spec.early_stop_metric,
        "early_stop_mode": resolve_early_stop_mode(spec.early_stop_metric, spec.early_stop_mode),
        "shape_X": list(X.shape),
        "n_classes": n_classes,
        "n_subjects": int(len(np.unique(subjects))),
        "subjects": [str(s) for s in np.unique(subjects).tolist()],
        "label_mapping_raw_to_encoded": label_mapping,
        "encoded_to_raw_label": inv_label_mapping,
        "encoded_to_label_name": {int(i): label_names[int(i)] for i in encoded_labels},
        "discarded_activity_ids": [0] if dataset == "pamap2" else [],
        "n_nodes": n_nodes,
        "node_feat_dim": node_feat_dim,
        "model_profile": model_profile,
        "device": str(device),
    }, out_dir / "dataset_manifest.json")

    all_true: List[int] = []
    all_pred: List[int] = []
    all_pred_rows: List[pd.DataFrame] = []
    fold_rows: List[Dict[str, Any]] = []
    fold_split_rows: List[Dict[str, Any]] = []
    overfit_rows: List[Dict[str, Any]] = []
    alignment_rows: List[pd.DataFrame] = []

    unique_subjects = np.unique(subjects)
    if eval_protocol == "loso" and len(unique_subjects) < 3:
        raise ValueError(f"Dataset {dataset} has {len(unique_subjects)} subject(s); strict nested LOSO requires at least 3.")
    if eval_protocol == "loso":
        fold_items = [(int(i), subj) for i, subj in enumerate(unique_subjects, 1)]
    else:
        fold_items = [(1, "mixed_subjects")]

    for fold_i, test_subj in fold_items:
        fold_name = f"fold_{fold_i:02d}_subject_{str(test_subj).replace(os.sep, '_')}"
        ckpt_best_path = out_dir / "checkpoints" / f"fold_{fold_i:02d}_best.pt"
        ckpt_last_path = out_dir / "checkpoints" / f"fold_{fold_i:02d}_last.pt"
        pred_path = out_dir / "fold_predictions" / f"fold_{fold_i:02d}_predictions.csv"
        hist_path = out_dir / "fold_predictions" / f"{fold_name}_history.json"
        curve_csv_path = out_dir / "plots" / f"fold_{fold_i:02d}_training_curves.csv"
        print(f"\n[{dataset}/{model_name}/{eval_unit}] {fold_name} ({fold_i}/{len(unique_subjects)})", flush=True)

        if eval_protocol == "loso":
            (
                X_tr, y_tr, s_tr, src_tr,
                X_val, y_val, s_val, src_val,
                X_te, y_te, s_te, src_te,
                test_indices, val_subj, train_subjects,
            ) = split_train_val_by_subject(
                X, y, subjects, source_indices, test_subj, fold_i,
                spec.val_strategy, spec.val_subject_policy, spec.seed,
            )
            tr_ds, val_ds, te_ds, use_adj, ds_meta = make_datasets(
                model_name, eval_unit,
                X_tr, y_tr, s_tr, src_tr,
                X_val, y_val, s_val, src_val,
                X_te, y_te, s_te, src_te,
                dataset,
                spec.sequence_length,
                spec.sequence_stride,
                spec.sequence_target_policy,
                fold_i,
                test_subj,
            )
        else:
            val_subj = "random_validation"
            train_subjects = [str(s) for s in np.unique(subjects).tolist()]
            if model_name in SEQUENCE_MODELS:
                add_repo_to_path()
                from torch.utils.data import Subset
                from src.dataset import HARSequenceDataset

                full_ds = HARSequenceDataset(
                    X, y, subjects=subjects, dataset=dataset, seq_len=spec.sequence_length,
                    seq_stride=spec.sequence_stride, target_policy=spec.sequence_target_policy,
                    source_indices=source_indices,
                )
                seq_train_idx, seq_val_idx, seq_test_idx = random_train_val_test_indices(
                    len(full_ds), seed=spec.seed + fold_i
                )
                tr_ds = Subset(full_ds, seq_train_idx.tolist())
                val_ds = Subset(full_ds, seq_val_idx.tolist())
                te_ds = Subset(full_ds, seq_test_idx.tolist())
                use_adj = True
                target_source = np.asarray(full_ds.target_source_indices)[seq_test_idx]
                test_alignment = pd.DataFrame({
                    "dataset": dataset,
                    "fold": int(fold_i),
                    "test_subject": str(test_subj),
                    "eval_sample_id": np.arange(len(seq_test_idx), dtype=int),
                    "target_source_index": target_source,
                    "y_true": full_ds.labels.detach().cpu().numpy()[seq_test_idx].astype(int),
                })
                ds_meta = {
                    "test_target_source_indices": target_source,
                    "test_alignment": test_alignment,
                }
                target_local = np.asarray([int(full_ds._seq_indices[int(i)][-1]) for i in seq_test_idx], dtype=np.int64)
                X_tr = X[seq_train_idx[: min(len(seq_train_idx), len(X))]] if len(seq_train_idx) else X[:0]
                y_tr = y[seq_train_idx[: min(len(seq_train_idx), len(y))]] if len(seq_train_idx) else y[:0]
                X_val = X[seq_val_idx[: min(len(seq_val_idx), len(X))]] if len(seq_val_idx) else X[:0]
                y_val = y[seq_val_idx[: min(len(seq_val_idx), len(y))]] if len(seq_val_idx) else y[:0]
                X_te = X[target_local]
                y_te = y[target_local]
                src_te = source_indices[target_local]
                s_tr = subjects[:0]
                s_val = subjects[:0]
                s_te = subjects[target_local]
                src_tr = source_indices[:0]
                src_val = source_indices[:0]
            else:
                train_idx, val_idx, test_indices = random_train_val_test_indices(
                    len(X), seed=spec.seed + fold_i
                )
                X_tr, y_tr, s_tr, src_tr = X[train_idx], y[train_idx], subjects[train_idx], source_indices[train_idx]
                X_val, y_val, s_val, src_val = X[val_idx], y[val_idx], subjects[val_idx], source_indices[val_idx]
                X_te, y_te, s_te, src_te = X[test_indices], y[test_indices], subjects[test_indices], source_indices[test_indices]
                tr_ds, val_ds, te_ds, use_adj, ds_meta = make_datasets(
                    model_name, eval_unit,
                    X_tr, y_tr, s_tr, src_tr,
                    X_val, y_val, s_val, src_val,
                    X_te, y_te, s_te, src_te,
                    dataset,
                    spec.sequence_length,
                    spec.sequence_stride,
                    spec.sequence_target_policy,
                    fold_i,
                    test_subj,
                )
        if "test_alignment" in ds_meta:
            alignment_rows.append(ds_meta["test_alignment"])
        tr_loader = DataLoader(tr_ds, batch_size=spec.batch_size, shuffle=True, num_workers=spec.num_workers, pin_memory=(device.type == "cuda"))
        val_loader = DataLoader(val_ds, batch_size=spec.batch_size, shuffle=False, num_workers=spec.num_workers, pin_memory=(device.type == "cuda"))
        te_loader = DataLoader(te_ds, batch_size=spec.batch_size, shuffle=False, num_workers=spec.num_workers, pin_memory=(device.type == "cuda"))

        model = build_model(model_name, dataset, X_tr, n_classes, device, adj_fixed)
        model, history, train_meta, last_state = train_one_fold(
            model, tr_loader, val_loader, use_adj, device, spec.epochs, spec.patience,
            n_classes, spec.early_stop_metric, spec.early_stop_mode,
        )
        best_payload = {
            "model_state_dict": model.state_dict(),
            "dataset": dataset,
            "model": model_name,
            "eval_unit": eval_unit,
            "fold_subject": str(test_subj),
            "validation_subject": str(val_subj),
            "n_classes": n_classes,
            "n_nodes": n_nodes,
            "node_feat_dim": node_feat_dim,
            "label_mapping": label_mapping,
            "train_meta": train_meta,
        }
        torch.save(best_payload, ckpt_best_path)
        last_payload = dict(best_payload)
        last_payload["model_state_dict"] = last_state if last_state else model.state_dict()
        torch.save(last_payload, ckpt_last_path)
        safe_json_dump(history, hist_path)
        write_training_curve_csv(history, curve_csv_path)
        write_training_curve(history, out_dir / "plots" / f"fold_{fold_i:02d}_training_curve.png", f"{dataset} {model_name} {eval_unit} {fold_name}")

        y_true, y_pred, proba = predict_loader(model, te_loader, use_adj, device)
        m = metrics(y_true, y_pred, n_classes)
        print(f"[{dataset}/{model_name}/{eval_unit}] {fold_name} acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} bacc={m['balanced_accuracy']:.4f}", flush=True)

        n_pred = len(y_true)
        if "test_target_source_indices" in ds_meta:
            source_window_idx = np.asarray(ds_meta["test_target_source_indices"])[:n_pred]
        else:
            source_window_idx = src_te[:n_pred] if len(src_te) >= n_pred else np.arange(n_pred)
        row_df = pd.DataFrame({
            "dataset": dataset,
            "feature_set": proc_ctx["feature_set"],
            "window_type": proc_ctx["window_type"],
            "task": proc_ctx["task"],
            "sessions": proc_ctx["sessions"],
            "model": model_name,
            "eval_unit": eval_unit,
            "fold": fold_i,
            "fold_subject": str(test_subj),
            "test_subject": str(test_subj),
            "validation_subject": str(val_subj),
            "row_in_fold": np.arange(n_pred),
            "source_window_index": source_window_idx,
            "original_window_index": source_window_idx,
            "y_true_id": y_true,
            "y_pred_id": y_pred,
            "y_true_raw": [inv_label_mapping.get(int(v), str(v)) for v in y_true],
            "y_pred_raw": [inv_label_mapping.get(int(v), str(v)) for v in y_pred],
            "y_true_label": [display_label_name(dataset, int(v), inv_label_mapping) for v in y_true],
            "y_pred_label": [display_label_name(dataset, int(v), inv_label_mapping) for v in y_pred],
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
            "feature_set": proc_ctx["feature_set"],
            "window_type": proc_ctx["window_type"],
            "task": proc_ctx["task"],
            "sessions": proc_ctx["sessions"],
            "eval_protocol": eval_protocol,
            "model": model_name,
            "eval_unit": eval_unit,
            "fold": fold_i,
            "fold_subject": str(test_subj),
            "test_subject": str(test_subj),
            "validation_subject": str(val_subj),
            "n_train_windows": int(len(X_tr)),
            "n_val_windows": int(len(X_val)),
            "n_test_windows": int(len(X_te)),
            "n_test_windows_raw": int(len(X_te)),
            "n_train_eval_samples": int(len(tr_ds)),
            "n_val_eval_samples": int(len(val_ds)),
            "n_test_eval_samples": int(n_pred),
            "n_test_predictions": int(n_pred),
            "n_test_classes": int(len(np.unique(y_true))) if n_pred else 0,
            "total_params": model_profile["total_params"],
            "trainable_params": model_profile["trainable_params"],
            "non_trainable_params": model_profile["non_trainable_params"],
            "parameter_size_mb_float32": model_profile["parameter_size_mb_float32"],
            **m,
        })
        fold_split_rows.append({
            "dataset": dataset,
            "feature_set": proc_ctx["feature_set"],
            "window_type": proc_ctx["window_type"],
            "task": proc_ctx["task"],
            "sessions": proc_ctx["sessions"],
            "eval_protocol": eval_protocol,
            "model": model_name,
            "eval_unit": eval_unit,
            "fold": fold_i,
            "test_subject": str(test_subj),
            "validation_subject": str(val_subj),
            "train_subjects": ",".join(str(s) for s in train_subjects),
            "n_train_windows": int(len(X_tr)),
            "n_val_windows": int(len(X_val)),
            "n_test_windows": int(len(X_te)),
            "n_train_eval_samples": int(len(tr_ds)),
            "n_val_eval_samples": int(len(val_ds)),
            "n_test_eval_samples": int(n_pred),
        })
        overfit_rows.append(overfitting_row(dataset, model_name, eval_unit, fold_i, test_subj, val_subj, train_meta, history, m))

    y_true_all = np.asarray(all_true, dtype=np.int64)
    y_pred_all = np.asarray(all_pred, dtype=np.int64)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "metrics_by_fold.csv", index=False)
    pd.DataFrame(fold_split_rows).to_csv(out_dir / "fold_split_subjects.csv", index=False)
    align_df = pd.concat(alignment_rows, ignore_index=True) if alignment_rows else pd.DataFrame(columns=[
        "dataset", "fold", "test_subject", "eval_sample_id", "sequence_start_source_index",
        "sequence_end_source_index", "target_source_index", "y_true",
    ])
    align_df.to_csv(out_dir / "sequence_alignment_manifest.csv", index=False)
    overfit_df = pd.DataFrame(overfit_rows)
    overfit_df.to_csv(out_dir / "overfitting_by_fold.csv", index=False)
    write_overfitting_summary(overfit_df, out_dir / "overfitting_summary.csv")
    pred_df = pd.concat(all_pred_rows, ignore_index=True) if all_pred_rows else pd.DataFrame()
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    agg = metrics(y_true_all, y_pred_all, n_classes) if len(y_true_all) else {}
    summary = {
        "dataset": dataset,
        "feature_set": proc_ctx["feature_set"],
        "window_type": proc_ctx["window_type"],
        "task": proc_ctx["task"],
        "sessions": proc_ctx["sessions"],
        "processed_dataset_dir": proc_ctx["processed_dataset_dir"],
        "eval_protocol": eval_protocol,
        "model": model_name,
        "eval_unit": eval_unit,
        "n_samples": int(len(y_true_all)),
        "n_eval_samples": int(len(y_true_all)),
        "n_folds": int(len(fold_df)),
        "runtime_sec": float(time.time() - t_start),
        "total_params": model_profile["total_params"],
        "trainable_params": model_profile["trainable_params"],
        "non_trainable_params": model_profile["non_trainable_params"],
        "parameter_size_mb_float32": model_profile["parameter_size_mb_float32"],
        "n_nodes": model_profile["n_nodes"],
        "node_feat_dim": model_profile["node_feat_dim"],
        **agg,
        "subject_macro_accuracy_mean": float(fold_df["accuracy"].mean()) if not fold_df.empty else math.nan,
        "subject_macro_accuracy_std": float(fold_df["accuracy"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        "subject_macro_f1_mean": float(fold_df["macro_f1"].mean()) if not fold_df.empty else math.nan,
        "subject_macro_f1_std": float(fold_df["macro_f1"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
    }
    pd.DataFrame([summary]).to_csv(out_dir / "metrics_summary.csv", index=False)
    safe_json_dump(summary, out_dir / "metrics_summary.json")

    labels = list(range(n_classes))
    if len(y_true_all):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*y_pred contains classes not in y_true.*")
            warnings.filterwarnings("ignore", message=".*ill-defined.*")
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
                "eval_protocol": eval_protocol,
                "model": model_name,
                "subject": str(subj),
                "n_predictions": int(len(g)),
                "n_true_classes": int(g["y_true_id"].nunique()),
                **metrics(yt, yp, n_classes),
                "dominant_confusion": dominant_confusion,
            })
        pd.DataFrame(subj_rows).sort_values("macro_f1").to_csv(out_dir / "subject_failure_analysis.csv", index=False)

    required = [
        "run_manifest.json",
        "dataset_manifest.json",
        "model_profile.json",
        "fold_split_subjects.csv",
        "metrics_summary.csv",
        "metrics_by_fold.csv",
        "predictions.csv",
        "classification_report.csv",
        "classification_report.json",
        "confusion_matrix.csv",
        "overfitting_by_fold.csv",
        "overfitting_summary.csv",
        "sequence_alignment_manifest.csv",
    ]
    missing_required = [name for name in required if not (out_dir / name).exists()]
    if missing_required:
        raise RuntimeError("Required artifacts missing before DONE.json: " + ", ".join(missing_required))
    safe_json_dump({"status": "done", "completed_at": now_stamp(), "summary": summary}, out_dir / "DONE.json")
    print(f"[DONE] dataset={dataset} model={model_name} eval_unit={eval_unit} summary={summary}", flush=True)
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


def parse_csv_arg(value: str) -> List[str]:
    return [x.strip() for x in str(value).replace(";", ",").split(",") if x.strip()]


def eval_units_for_model(model_name: str, eval_modes: Sequence[str]) -> List[str]:
    modes = {m.lower() for m in eval_modes}
    units: List[str] = []
    if model_name in SEQUENCE_MODELS:
        if "sequence" in modes:
            units.append("sequence")
    elif model_name in GRAPH_MODELS:
        if "window" in modes:
            units.append("window")
        if model_name == "gnn" and "sequence" in modes:
            units.append("sequence_aligned")
    elif model_name in WINDOW_MODELS:
        if "window" in modes:
            units.append("window")
    return units


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

    def worker_completed(spec: JobSpec) -> bool:
        out_dir = Path(spec.run_root) / spec.dataset / spec.model / spec.eval_unit
        done_path = out_dir / "DONE.json"
        if not done_path.exists():
            return False
        required = [
            out_dir / "metrics_summary.csv",
            out_dir / "predictions.csv",
        ]
        return all(p.exists() for p in required)

    def start_one(spec: JobSpec) -> None:
        nonlocal running
        log_dir = Path(spec.run_root) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{spec.dataset}_{spec.model}_{spec.eval_unit}"
        out_log = log_dir / f"{tag}_stdout.log"
        err_log = log_dir / f"{tag}_stderr.log"
        cmd = [
            sys.executable,
            str(script_path),
            "--worker",
            "--dataset", spec.dataset,
            "--model", spec.model,
            "--eval-unit", spec.eval_unit,
            "--run-root", spec.run_root,
            "--eval-protocol", spec.eval_protocol,
            *(
                ["--processed-dir", spec.processed_dir]
                if spec.processed_dir
                else []
            ),
            "--epochs", str(spec.epochs),
            "--patience", str(spec.patience),
            "--batch-size", str(spec.batch_size),
            "--device", spec.device,
            "--num-workers", str(spec.num_workers),
            "--seed", str(spec.seed),
            "--experiments", ",".join(spec.experiments),
            "--val-strategy", spec.val_strategy,
            "--val-subject-policy", spec.val_subject_policy,
            "--early-stop-metric", spec.early_stop_metric,
            "--early-stop-mode", spec.early_stop_mode,
            "--sequence-length", str(spec.sequence_length),
            "--sequence-stride", str(spec.sequence_stride),
            "--sequence-target-policy", spec.sequence_target_policy,
        ]
        if spec.no_hhar_cap:
            cmd += ["--no-hhar-cap"]
        if spec.disable_cudnn_for_sequence_models:
            cmd += ["--disable-cudnn-for-sequence-models"]
        if spec.max_windows_per_subject is not None and not spec.no_hhar_cap:
            cmd += ["--max-windows-per-subject", str(spec.max_windows_per_subject)]
        if spec.apply_window_cap_to_all_datasets:
            cmd += ["--apply-window-cap-to-all-datasets"]
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
                tag = f"{spec.dataset}/{spec.model}/{spec.eval_unit}"
                if rc != 0:
                    if worker_completed(spec):
                        print(f"[WARN] {tag} rc={rc} but DONE.json and required artifacts exist; treating as success. stdout={out_log}", flush=True)
                    else:
                        failures += 1
                        print(f"[FAIL] {tag} rc={rc} stdout={out_log} stderr={err_log}", flush=True)
                else:
                    print(f"[OK] {tag} stdout={out_log}", flush=True)
        running = still
    return failures


def aggregate_run(run_root: Path) -> None:
    rows = []
    folds = []
    overfit_folds = []
    overfit_summaries = []
    for p in run_root.glob("*/*/*/metrics_summary.csv"):
        try:
            df = pd.read_csv(p)
            df["artifact_dir"] = str(p.parent)
            rows.append(df)
        except Exception:
            pass
    for p in run_root.glob("*/*/*/metrics_by_fold.csv"):
        try:
            df = pd.read_csv(p)
            df["artifact_dir"] = str(p.parent)
            folds.append(df)
        except Exception:
            pass
    for p in run_root.glob("*/*/*/overfitting_by_fold.csv"):
        try:
            df = pd.read_csv(p)
            df["artifact_dir"] = str(p.parent)
            overfit_folds.append(df)
        except Exception:
            pass
    for p in run_root.glob("*/*/*/overfitting_summary.csv"):
        try:
            df = pd.read_csv(p)
            df["artifact_dir"] = str(p.parent)
            overfit_summaries.append(df)
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
        group_cols = ["dataset", "model", "eval_unit"] if "eval_unit" in fold_all.columns else ["dataset", "model"]
        for group_key, g in fold_all.groupby(group_cols):
            if len(group_cols) == 3:
                dataset, model, eval_unit = group_key
            else:
                dataset, model = group_key
                eval_unit = ""
            for metric in ["accuracy", "macro_f1", "balanced_accuracy"]:
                vals = g[metric].dropna().astype(float).to_numpy()
                if len(vals) == 0:
                    continue
                mean = float(vals.mean())
                std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                ci95 = float(1.96 * std / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
                stat_rows.append({"dataset": dataset, "model": model, "eval_unit": eval_unit, "metric": metric, "folds": int(len(vals)), "mean": mean, "std": std, "ci95_half_width": ci95, "ci95_low": mean - ci95, "ci95_high": mean + ci95})
        pd.DataFrame(stat_rows).to_csv(run_root / "experiment2_statistical_reliability.csv", index=False)
    if overfit_folds:
        pd.concat(overfit_folds, ignore_index=True).to_csv(run_root / "overfitting_by_fold_all_jobs.csv", index=False)
    if overfit_summaries:
        pd.concat(overfit_summaries, ignore_index=True).to_csv(run_root / "overfitting_summary_all_jobs.csv", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel runner for repo proposed HAR deep algorithms.")
    p.add_argument("--worker", action="store_true", help="Internal worker mode for one dataset/model job.")
    p.add_argument("--datasets", nargs="+", choices=["pamap2", "hhar"], default=["pamap2", "hhar"])
    p.add_argument("--dataset", choices=["pamap2", "hhar"], default=None, help="Worker dataset.")
    p.add_argument("--models", default="auto", help="Comma-separated models or 'auto'.")
    p.add_argument("--model", default=None, help="Worker model.")
    p.add_argument("--eval-modes", default="window,sequence", help="Comma-separated eval modes: window,sequence.")
    p.add_argument("--eval-unit", choices=["window", "sequence", "sequence_aligned"], default=None, help="Worker eval unit.")
    p.add_argument("--eval-protocol", choices=["loso", "random_holdout"], default="loso")
    p.add_argument("--top-k", type=int, default=2, help="When --models auto, choose top K proposed models from existing metrics or fallback priority.")
    p.add_argument("--rank-source", default=None, help="Optional metrics file/dir to rank models from.")
    p.add_argument("--rank-metric", default="macro_f1")
    p.add_argument("--run-root", default=None, help="Output root. Launcher creates timestamped one if omitted.")
    p.add_argument("--processed-dir", default=None, help="Directory containing {dataset}_X/y/subjects.npy files.")
    p.add_argument("--parallel-jobs", type=int, default=2, help="Max concurrent dataset/model jobs.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-windows-per-subject", type=int, default=5000, help="HHAR cap per subject; ignored for PAMAP2.")
    p.add_argument("--apply-window-cap-to-all-datasets", action="store_true", help="Apply --max-windows-per-subject to PAMAP2 too; intended for smoke/debug runs only.")
    p.add_argument("--no-hhar-cap", action="store_true", help="Use full HHAR processed data.")
    p.add_argument("--val-strategy", choices=["inner_subject"], default="inner_subject")
    p.add_argument("--val-subject-policy", choices=["round_robin", "min_count", "max_count", "random"], default="round_robin")
    p.add_argument("--early-stop-metric", choices=["val_macro_f1", "val_loss", "val_acc"], default="val_macro_f1")
    p.add_argument("--early-stop-mode", choices=["auto", "min", "max"], default="auto")
    p.add_argument("--sequence-length", type=int, default=10)
    p.add_argument("--sequence-stride", type=int, default=1)
    p.add_argument("--sequence-target-policy", choices=["last", "majority"], default="last")
    p.add_argument("--disable-cudnn-for-sequence-models", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--experiments", default=DEFAULT_EXPERIMENTS, help="Comma-separated experiment IDs to attach/produce.")
    p.add_argument("--cpu-threads-per-job", type=int, default=4, help="Sets OMP/MKL/OPENBLAS thread env per launched job.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    add_repo_to_path()
    set_seed(args.seed)

    experiments = [x.strip() for x in str(args.experiments).split(",") if x.strip()]
    if args.worker:
        if not args.dataset or not args.model or not args.run_root or not args.eval_unit:
            raise SystemExit("Worker mode requires --dataset, --model, --eval-unit and --run-root")
        effective_hhar_cap = None if args.no_hhar_cap else args.max_windows_per_subject
        spec = JobSpec(
            dataset=args.dataset,
            model=args.model,
            run_root=str(args.run_root),
            processed_dir=args.processed_dir,
            eval_protocol=args.eval_protocol,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            max_windows_per_subject=effective_hhar_cap,
            no_hhar_cap=bool(args.no_hhar_cap),
            max_windows_per_subject_arg=int(args.max_windows_per_subject),
            apply_window_cap_to_all_datasets=bool(args.apply_window_cap_to_all_datasets),
            eval_unit=args.eval_unit,
            val_strategy=args.val_strategy,
            val_subject_policy=args.val_subject_policy,
            early_stop_metric=args.early_stop_metric,
            early_stop_mode=args.early_stop_mode,
            sequence_length=args.sequence_length,
            sequence_stride=args.sequence_stride,
            sequence_target_policy=args.sequence_target_policy,
            disable_cudnn_for_sequence_models=bool(args.disable_cudnn_for_sequence_models),
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
            out_dir = Path(args.run_root) / args.dataset / args.model / args.eval_unit
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_json_dump({"status": "failed", "traceback": traceback.format_exc(), "time": now_stamp()}, out_dir / "FAILED.json")
            return 1

    models = resolve_models(args.models, args.top_k, args.rank_source, args.datasets, args.rank_metric)
    eval_modes = parse_csv_arg(args.eval_modes)
    invalid_modes = sorted(set(eval_modes) - {"window", "sequence"})
    if invalid_modes:
        raise ValueError("--eval-modes may contain only window and sequence; invalid: " + ", ".join(invalid_modes))
    run_root = Path(args.run_root) if args.run_root else Path("results") / "phase2_repo_parallel" / now_stamp()
    run_root.mkdir(parents=True, exist_ok=True)

    specs: List[JobSpec] = []
    for dataset in args.datasets:
        for model in models:
            for eval_unit in eval_units_for_model(model, eval_modes):
                specs.append(JobSpec(
                    dataset=dataset,
                    model=model,
                    run_root=str(run_root),
                    processed_dir=args.processed_dir,
                    eval_protocol=args.eval_protocol,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    max_windows_per_subject=None if args.no_hhar_cap else args.max_windows_per_subject,
                    no_hhar_cap=bool(args.no_hhar_cap),
                    max_windows_per_subject_arg=int(args.max_windows_per_subject),
                    apply_window_cap_to_all_datasets=bool(args.apply_window_cap_to_all_datasets),
                    eval_unit=eval_unit,
                    val_strategy=args.val_strategy,
                    val_subject_policy=args.val_subject_policy,
                    early_stop_metric=args.early_stop_metric,
                    early_stop_mode=args.early_stop_mode,
                    sequence_length=args.sequence_length,
                    sequence_stride=args.sequence_stride,
                    sequence_target_policy=args.sequence_target_policy,
                    disable_cudnn_for_sequence_models=bool(args.disable_cudnn_for_sequence_models),
                    device=args.device,
                    num_workers=args.num_workers,
                    seed=args.seed,
                    skip_existing=args.skip_existing,
                    experiments=experiments,
                ))
    if not specs:
        raise ValueError("No jobs selected after applying --eval-modes to the selected models.")
    write_job_specs(run_root, specs)
    print("Selected models:", ", ".join(models), flush=True)
    print("Datasets:", ", ".join(args.datasets), flush=True)
    print("Eval modes:", ",".join(eval_modes), flush=True)
    print("Eval protocol:", args.eval_protocol, flush=True)
    print("Sequence length:", args.sequence_length, "stride:", args.sequence_stride, "target_policy:", args.sequence_target_policy, flush=True)
    print("Nested validation:", args.val_strategy, "/", args.val_subject_policy, flush=True)
    print("Early stopping:", args.early_stop_metric, resolve_early_stop_mode(args.early_stop_metric, args.early_stop_mode), flush=True)
    print("HHAR cap:", "NONE / FULL DATA" if args.no_hhar_cap else args.max_windows_per_subject, flush=True)
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
