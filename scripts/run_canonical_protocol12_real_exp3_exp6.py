#!/usr/bin/env python
"""Run real canonical protocol12 Exp3 and Exp6.

Exp3 evaluates robustness by reusing canonical v3 fold checkpoints and by
refitting classical baselines on the canonical LOSO folds, then perturbing only
the held-out test data.

Exp6 evaluates few-shot held-subject calibration by reusing canonical v3 fold
checkpoints and fine-tuning on a small calibration subset from the held-out
subject. Classical baselines are recalibrated by refitting on train+shots,
because the canonical baseline artifacts do not save fitted estimators.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.canonical_baseline_runner import load_canonical_dataset, split_indices
from scripts.phase1_classical_baselines import (
    display_class_names,
    extract_window_features,
    fit_predict_fold_model,
    make_models,
)
from scripts.phase2_repo_deep_parallel_v2 import (
    apply_channel_standardizer,
    build_model,
    class_weights_from_labels,
    display_label_names,
    fit_channel_standardizer,
    get_dataset_meta,
    load_processed_dataset,
    make_datasets,
    metrics,
    predict_loader,
    safe_json_dump,
    select_device,
    set_seed,
    split_train_val_by_subject,
)


DEFAULT_FEATURE_SETS = ["acc16_hr", "acc16_gyro", "acc16_gyro_hr"]
DEFAULT_BASELINE_MODELS = [
    "dummy_most_frequent",
    "gaussian_nb",
    "knn_k5",
    "linear_svm",
    "rbf_svm",
    "decision_tree_entropy",
    "bagged_tree_entropy",
    "random_forest",
    "adaboost_tree",
    "xgboost_hist",
]
DEFAULT_V3_MODELS = ["improved_gnn_lstm_res", "improved_gnn_lstm_attn_adj_resbn"]
SEVERITY_SCALE = {"low": 0.05, "medium": 0.10, "high": 0.20}
CALIBRATION_PCTS = [0.0, 0.01, 0.05, 0.10]

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")


def log_progress(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp} pid={os.getpid()}] {message}", flush=True)
    log_path = os.environ.get("HAR_REAL_EXP_PROGRESS_LOG")
    if log_path:
        event = {
            "time": stamp,
            "pid": os.getpid(),
            "message": message,
        }
        try:
            with open(log_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(event, default=str) + "\n")
        except Exception:
            pass


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).replace(";", ",").split(",") if x.strip()]


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def raw_feature_columns(manifest: dict[str, Any], feature_set: str, n_channels: int) -> list[str]:
    cols = manifest.get("feature_columns")
    if isinstance(cols, list) and len(cols) == n_channels:
        return [str(c) for c in cols]
    if feature_set.endswith("_hr") and n_channels:
        return ["heart_rate"] + [f"channel_{i}" for i in range(1, n_channels)]
    return [f"channel_{i}" for i in range(n_channels)]


def hr_channel_index(feature_columns: Sequence[str]) -> int | None:
    for i, name in enumerate(feature_columns):
        if str(name).lower() == "heart_rate":
            return i
    return None


def perturb_windows(
    X: np.ndarray,
    *,
    perturbation: str,
    severity: str,
    feature_columns: Sequence[str],
    seed: int,
    reference: np.ndarray | None = None,
) -> tuple[np.ndarray | None, str]:
    Xp = np.asarray(X, dtype=np.float32).copy()
    rng = np.random.default_rng(seed)
    scale = float(SEVERITY_SCALE.get(severity, 0.10))
    ref = np.asarray(reference if reference is not None and len(reference) else X, dtype=np.float32)

    if perturbation == "gaussian_noise":
        sigma = np.nanstd(ref.reshape(-1, ref.shape[-1]), axis=0).astype(np.float32)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        noise = rng.normal(0.0, scale, size=Xp.shape).astype(np.float32) * sigma.reshape(1, 1, -1)
        return Xp + noise, ""

    if perturbation == "random_channel_dropout":
        mask = rng.random(size=Xp.shape) < scale
        Xp[mask] = 0.0
        return Xp, ""

    if perturbation == "heart_rate_zero":
        idx = hr_channel_index(feature_columns)
        if idx is None:
            return None, "not_applicable_no_heart_rate_channel"
        Xp[:, :, idx] = 0.0
        return Xp, ""

    raise ValueError(f"Unknown perturbation: {perturbation}")


def most_affected_class(y_true: np.ndarray, y_clean: np.ndarray, y_pert: np.ndarray, label_names: Sequence[str]) -> str:
    labels = sorted(set(np.asarray(y_true, dtype=int).tolist()))
    worst_name, worst_drop = "", -math.inf
    for label in labels:
        mask = np.asarray(y_true) == label
        if not mask.any():
            continue
        clean_recall = float((np.asarray(y_clean)[mask] == label).mean())
        pert_recall = float((np.asarray(y_pert)[mask] == label).mean())
        drop = clean_recall - pert_recall
        if drop > worst_drop:
            worst_drop = drop
            worst_name = str(label_names[int(label)]) if int(label) < len(label_names) else str(label)
    return worst_name


def canonical_v3_artifact(v3_root: Path, feature_set: str, protocol: str, model: str) -> Path:
    return (
        v3_root
        / "core_comparison"
        / "pamap2"
        / feature_set
        / "overlapping"
        / protocol
        / "deep"
        / "pamap2"
        / model
        / "sequence"
    )


def load_v3_model(
    artifact: Path,
    model_name: str,
    fold: int,
    X_sample: np.ndarray,
    n_classes: int,
    device: torch.device,
):
    sample_window = X_sample[0] if X_sample.ndim == 3 else X_sample
    _n_nodes, _, adj_builder = get_dataset_meta("pamap2", X_sample=sample_window)
    adj_fixed = adj_builder().to(device)
    model = build_model(model_name, "pamap2", X_sample[:1], n_classes, device, adj_fixed)
    ckpt_path = artifact / "checkpoints" / f"fold_{fold:02d}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing canonical v3 checkpoint: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, adj_fixed, ckpt_path


def make_loso_deep_test_dataset(
    *,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    source_indices: np.ndarray,
    fold: int,
    test_subject: Any,
    model_name: str,
    sequence_length: int,
    sequence_stride: int,
    sequence_target_policy: str,
    standardize_input: bool,
    perturbation: str | None = None,
    severity: str = "clean",
    feature_columns: Sequence[str] = (),
    seed: int = 42,
) -> tuple[Any, np.ndarray, np.ndarray, str]:
    (
        X_tr, y_tr, s_tr, src_tr,
        X_val, y_val, s_val, src_val,
        X_te, y_te, s_te, src_te,
        _test_indices, _val_subj, _train_subjects,
    ) = split_train_val_by_subject(
        X,
        y,
        subjects,
        source_indices,
        test_subject,
        fold,
        "inner_subject",
        "round_robin",
        seed,
    )
    if perturbation:
        Xp, note = perturb_windows(
            X_te,
            perturbation=perturbation,
            severity=severity,
            feature_columns=feature_columns,
            seed=seed + fold,
            reference=X_tr,
        )
        if Xp is None:
            return None, np.asarray([], dtype=int), np.asarray([], dtype=int), note
        X_te = Xp
    if standardize_input:
        scaler_mean, scaler_std = fit_channel_standardizer(X_tr)
        X_tr = apply_channel_standardizer(X_tr, scaler_mean, scaler_std)
        X_val = apply_channel_standardizer(X_val, scaler_mean, scaler_std)
        X_te = apply_channel_standardizer(X_te, scaler_mean, scaler_std)
    _tr_ds, _val_ds, te_ds, use_adj, _meta = make_datasets(
        model_name,
        "sequence",
        X_tr,
        y_tr,
        s_tr,
        src_tr,
        X_val,
        y_val,
        s_val,
        src_val,
        X_te,
        y_te,
        s_te,
        src_te,
        "pamap2",
        sequence_length,
        sequence_stride,
        sequence_target_policy,
        fold,
        test_subject,
    )
    return te_ds, y_te, s_te, ""


def predict_deep_dataset(model, dataset, *, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    return predict_loader(model, loader, True, device)


def run_exp3_v3(args: argparse.Namespace, feature_set: str, model_name: str) -> pd.DataFrame:
    log_progress(f"EXP3 v3 start feature_set={feature_set} model={model_name}")
    processed_dir = args.processed_root / "pamap2" / feature_set / "overlapping"
    X, y, subjects, source_indices, _mapping, inv, data_manifest = load_processed_dataset(
        "pamap2",
        max_windows_per_subject=None,
        seed=args.seed,
        processed_dir=str(processed_dir),
    )
    source_manifest = data_manifest.get("source_processed_manifest") or {}
    feature_columns = raw_feature_columns(source_manifest, feature_set, X.shape[-1])
    label_names = display_label_names("pamap2", list(range(len(np.unique(y)))), inv)
    device = select_device(args.device)
    artifact = canonical_v3_artifact(args.v3_root, feature_set, "loso", model_name)
    manifest = read_json(artifact / "dataset_manifest.json")
    folds = pd.read_csv(artifact / "fold_split_subjects.csv")
    rows = []

    for _, fold_row in folds.iterrows():
        fold = int(fold_row["fold"])
        test_subject = fold_row["test_subject"]
        fold_start = time.time()
        log_progress(f"EXP3 v3 load checkpoint feature_set={feature_set} model={model_name} fold={fold} subject={test_subject}")
        model, _adj, ckpt_path = load_v3_model(artifact, model_name, fold, X, len(np.unique(y)), device)
        clean_ds, _yt_raw, _subj_raw, note = make_loso_deep_test_dataset(
            X=X,
            y=y,
            subjects=subjects,
            source_indices=source_indices,
            fold=fold,
            test_subject=test_subject,
            model_name=model_name,
            sequence_length=int(manifest.get("sequence_length", args.sequence_length)),
            sequence_stride=int(manifest.get("sequence_stride", args.sequence_stride)),
            sequence_target_policy=str(manifest.get("sequence_target_policy", "last")),
            standardize_input=bool(manifest.get("standardize_input", False)),
            seed=args.seed,
        )
        if clean_ds is None:
            raise RuntimeError(note)
        y_true, y_clean, _ = predict_deep_dataset(model, clean_ds, batch_size=args.batch_size, device=device)
        clean_metrics = metrics(y_true, y_clean, len(np.unique(y)))
        log_progress(
            f"EXP3 v3 clean done feature_set={feature_set} model={model_name} fold={fold} "
            f"subject={test_subject} acc={clean_metrics['accuracy']:.4f} macro_f1={clean_metrics['macro_f1']:.4f}"
        )
        for perturbation in parse_csv(args.perturbations):
            for severity in parse_csv(args.severities):
                pert_start = time.time()
                log_progress(
                    f"EXP3 v3 perturb start feature_set={feature_set} model={model_name} fold={fold} "
                    f"subject={test_subject} perturbation={perturbation} severity={severity}"
                )
                pert_ds, _yt, _subj, note = make_loso_deep_test_dataset(
                    X=X,
                    y=y,
                    subjects=subjects,
                    source_indices=source_indices,
                    fold=fold,
                    test_subject=test_subject,
                    model_name=model_name,
                    sequence_length=int(manifest.get("sequence_length", args.sequence_length)),
                    sequence_stride=int(manifest.get("sequence_stride", args.sequence_stride)),
                    sequence_target_policy=str(manifest.get("sequence_target_policy", "last")),
                    standardize_input=bool(manifest.get("standardize_input", False)),
                    perturbation=perturbation,
                    severity=severity,
                    feature_columns=feature_columns,
                    seed=args.seed,
                )
                if pert_ds is None:
                    rows.append({
                        "dataset": "pamap2",
                        "feature_set": feature_set,
                        "protocol": "loso",
                        "family": "deep",
                        "model": model_name,
                        "fold": fold,
                        "test_subject": str(test_subject),
                        "perturbation": perturbation,
                        "severity": severity,
                        "status": note,
                        "checkpoint_path": str(ckpt_path),
                    })
                    continue
                yt, yp, _ = predict_deep_dataset(model, pert_ds, batch_size=args.batch_size, device=device)
                pert_metrics = metrics(yt, yp, len(np.unique(y)))
                n = min(len(y_true), len(y_clean), len(yp))
                log_progress(
                    f"EXP3 v3 perturb done feature_set={feature_set} model={model_name} fold={fold} "
                    f"perturbation={perturbation} severity={severity} macro_f1_drop={clean_metrics['macro_f1'] - pert_metrics['macro_f1']:.4f} "
                    f"sec={time.time() - pert_start:.1f}"
                )
                rows.append({
                    "dataset": "pamap2",
                    "feature_set": feature_set,
                    "protocol": "loso",
                    "family": "deep",
                    "model": model_name,
                    "fold": fold,
                    "test_subject": str(test_subject),
                    "perturbation": perturbation,
                    "severity": severity,
                    "status": "ok",
                    "clean_accuracy": clean_metrics["accuracy"],
                    "perturbed_accuracy": pert_metrics["accuracy"],
                    "accuracy_drop": clean_metrics["accuracy"] - pert_metrics["accuracy"],
                    "clean_macro_f1": clean_metrics["macro_f1"],
                    "perturbed_macro_f1": pert_metrics["macro_f1"],
                    "macro_f1_drop": clean_metrics["macro_f1"] - pert_metrics["macro_f1"],
                    "most_affected_class": most_affected_class(y_true[:n], y_clean[:n], yp[:n], label_names),
                    "n_eval_samples": int(len(yt)),
                    "checkpoint_path": str(ckpt_path),
                })
        log_progress(
            f"EXP3 v3 fold done feature_set={feature_set} model={model_name} fold={fold} "
            f"subject={test_subject} sec={time.time() - fold_start:.1f}"
        )
    log_progress(f"EXP3 v3 done feature_set={feature_set} model={model_name} rows={len(rows)}")
    return pd.DataFrame(rows)


def make_baseline_model_map(args: argparse.Namespace) -> dict[str, object]:
    models = make_models(include_xgb=args.include_xgb, use_cuda=args.xgb_cuda, fast=args.fast_baselines)
    requested = parse_csv(args.baseline_models)
    if requested == ["all"]:
        requested = list(models)
    requested = [m for m in requested if m in models]
    return {name: cap_estimator_jobs(models[name], int(args.baseline_estimator_jobs)) for name in requested}


def cap_estimator_jobs(estimator: object, jobs: int) -> object:
    if jobs <= 0 or not hasattr(estimator, "get_params") or not hasattr(estimator, "set_params"):
        return estimator
    params = estimator.get_params(deep=True)
    updates = {}
    for key in params:
        if key == "n_jobs" or key.endswith("__n_jobs"):
            updates[key] = int(jobs)
    if updates:
        try:
            estimator.set_params(**updates)
        except Exception:
            pass
    return estimator


def baseline_fit_model(base_model: object, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> tuple[object, np.ndarray | None]:
    model = clone(base_model)
    if model_name.startswith("xgboost"):
        train_classes = np.unique(y_train)
        local_map = {int(cls): i for i, cls in enumerate(train_classes)}
        y_local = np.asarray([local_map[int(v)] for v in y_train], dtype=int)
        model.fit(X_train, y_local)
        return model, train_classes.astype(int)
    model.fit(X_train, y_train)
    return model, None


def baseline_predict_model(model: object, local_classes: np.ndarray | None, X_test: np.ndarray) -> np.ndarray:
    pred = model.predict(X_test).astype(int)
    if local_classes is not None:
        pred = local_classes[pred]
    return pred.astype(int)


def baseline_fit_predict(base_model: object, model_name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    model, local_classes = baseline_fit_model(base_model, model_name, X_train, y_train)
    return baseline_predict_model(model, local_classes, X_test)


def run_exp3_baselines(args: argparse.Namespace, feature_set: str, model_map: dict[str, object]) -> pd.DataFrame:
    log_progress(f"EXP3 baseline block start feature_set={feature_set} models={','.join(model_map.keys())}")
    X, y_raw, subjects, _meta, manifest = load_canonical_dataset(args.processed_root, "pamap2", feature_set, "overlapping")
    feature_columns = raw_feature_columns(manifest, feature_set, X.shape[-1])
    X_feat_clean, _feature_names = extract_window_features(X, feature_columns)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    label_names = display_class_names("pamap2", le.classes_)
    rows = []
    for model_name, base_model in model_map.items():
        log_progress(f"EXP3 baseline clean fitting start feature_set={feature_set} model={model_name}")
        clean_fold_cache: list[dict[str, Any]] = []
        for fold, (train_idx, test_idx, test_subject, _n_test) in enumerate(
            split_indices("loso", subjects, y, args.seed, args.test_fraction),
            start=1,
        ):
            t0 = time.time()
            fitted_model, local_classes = baseline_fit_model(base_model, model_name, X_feat_clean[train_idx], y[train_idx])
            pred_clean = baseline_predict_model(fitted_model, local_classes, X_feat_clean[test_idx])
            clean_fold_cache.append({
                "fold": fold,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "test_subject": test_subject,
                "model": fitted_model,
                "local_classes": local_classes,
                "pred_clean": pred_clean,
            })
            log_progress(
                f"EXP3 baseline clean fold done feature_set={feature_set} model={model_name} "
                f"fold={fold} subject={test_subject} sec={time.time() - t0:.1f}"
            )
        for perturbation in parse_csv(args.perturbations):
            for severity in parse_csv(args.severities):
                pert_start = time.time()
                log_progress(
                    f"EXP3 baseline perturb start feature_set={feature_set} model={model_name} "
                    f"perturbation={perturbation} severity={severity}"
                )
                y_true_all, y_clean_all, y_pert_all = [], [], []
                status = "ok"
                for cached in clean_fold_cache:
                    fold = int(cached["fold"])
                    train_idx = cached["train_idx"]
                    test_idx = cached["test_idx"]
                    test_subject = cached["test_subject"]
                    pred_clean = cached["pred_clean"]
                    Xp, note = perturb_windows(
                        X[test_idx],
                        perturbation=perturbation,
                        severity=severity,
                        feature_columns=feature_columns,
                        seed=args.seed + fold,
                        reference=X[train_idx],
                    )
                    if Xp is None:
                        status = note
                        break
                    Xp_feat, _ = extract_window_features(Xp, feature_columns)
                    pred_pert = baseline_predict_model(cached["model"], cached["local_classes"], Xp_feat)
                    y_true_all.extend(y[test_idx].tolist())
                    y_clean_all.extend(pred_clean.astype(int).tolist())
                    y_pert_all.extend(pred_pert.astype(int).tolist())
                    rows.append({
                        "dataset": "pamap2",
                        "feature_set": feature_set,
                        "protocol": "loso",
                        "family": "baseline",
                        "model": model_name,
                        "fold": fold,
                        "test_subject": str(test_subject),
                        "perturbation": perturbation,
                        "severity": severity,
                        "status": "ok",
                        "clean_accuracy": float(accuracy_score(y[test_idx], pred_clean)),
                        "perturbed_accuracy": float(accuracy_score(y[test_idx], pred_pert)),
                        "accuracy_drop": float(accuracy_score(y[test_idx], pred_clean) - accuracy_score(y[test_idx], pred_pert)),
                        "clean_macro_f1": float(f1_score(y[test_idx], pred_clean, average="macro", zero_division=0)),
                        "perturbed_macro_f1": float(f1_score(y[test_idx], pred_pert, average="macro", zero_division=0)),
                        "macro_f1_drop": float(
                            f1_score(y[test_idx], pred_clean, average="macro", zero_division=0)
                            - f1_score(y[test_idx], pred_pert, average="macro", zero_division=0)
                        ),
                        "most_affected_class": most_affected_class(y[test_idx], pred_clean, pred_pert, label_names),
                        "n_eval_samples": int(len(test_idx)),
                        "checkpoint_path": "",
                    })
                    log_progress(
                        f"EXP3 baseline perturb fold done feature_set={feature_set} model={model_name} "
                        f"fold={fold} subject={test_subject} perturbation={perturbation} severity={severity}"
                    )
                if status != "ok":
                    rows.append({
                        "dataset": "pamap2",
                        "feature_set": feature_set,
                        "protocol": "loso",
                        "family": "baseline",
                        "model": model_name,
                        "fold": "all",
                        "test_subject": "all",
                        "perturbation": perturbation,
                        "severity": severity,
                        "status": status,
                    })
                    continue
                if y_true_all:
                    clean_acc = accuracy_score(y_true_all, y_clean_all)
                    pert_acc = accuracy_score(y_true_all, y_pert_all)
                    clean_f1 = f1_score(y_true_all, y_clean_all, average="macro", zero_division=0)
                    pert_f1 = f1_score(y_true_all, y_pert_all, average="macro", zero_division=0)
                    rows.append({
                        "dataset": "pamap2",
                        "feature_set": feature_set,
                        "protocol": "loso",
                        "family": "baseline",
                        "model": model_name,
                        "fold": "aggregate",
                        "test_subject": "aggregate",
                        "perturbation": perturbation,
                        "severity": severity,
                        "status": "ok",
                        "clean_accuracy": float(clean_acc),
                        "perturbed_accuracy": float(pert_acc),
                        "accuracy_drop": float(clean_acc - pert_acc),
                        "clean_macro_f1": float(clean_f1),
                        "perturbed_macro_f1": float(pert_f1),
                        "macro_f1_drop": float(clean_f1 - pert_f1),
                        "most_affected_class": most_affected_class(np.asarray(y_true_all), np.asarray(y_clean_all), np.asarray(y_pert_all), label_names),
                        "n_eval_samples": int(len(y_true_all)),
                        "checkpoint_path": "",
                    })
                    log_progress(
                        f"EXP3 baseline perturb aggregate done feature_set={feature_set} model={model_name} "
                        f"perturbation={perturbation} severity={severity} macro_f1_drop={clean_f1 - pert_f1:.4f} "
                        f"sec={time.time() - pert_start:.1f}"
                    )
        log_progress(f"EXP3 baseline model done feature_set={feature_set} model={model_name} rows={len(rows)}")
    return pd.DataFrame(rows)


def calibration_split(n_items: int, pct: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n_items < 2:
        return np.asarray([], dtype=int), np.arange(n_items, dtype=int)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_items)
    n_cal = int(round(n_items * float(pct)))
    n_cal = max(1, n_cal) if pct > 0 else 0
    n_cal = min(n_cal, n_items - 1)
    cal = np.sort(perm[:n_cal])
    test = np.sort(perm[n_cal:])
    return cal, test


def freeze_for_strategy(model: nn.Module, strategy: str) -> None:
    for param in model.parameters():
        param.requires_grad = strategy == "full_model"
    if strategy == "classifier_head_only":
        for name, param in model.named_parameters():
            if name.startswith("classifier."):
                param.requires_grad = True


def finetune_deep(model: nn.Module, dataset, *, strategy: str, args: argparse.Namespace, device: torch.device, n_classes: int) -> nn.Module:
    model = copy.deepcopy(model)
    freeze_for_strategy(model, strategy)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError(f"No trainable parameters for strategy {strategy}")
    labels = []
    for i in range(len(dataset)):
        item = dataset[i]
        labels.append(int(item[-1]))
    weights = class_weights_from_labels(np.asarray(labels, dtype=np.int64), n_classes).to(device) if labels else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(params, lr=float(args.exp6_lr), weight_decay=float(args.exp6_weight_decay))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    model.train()
    for epoch in range(1, int(args.exp6_epochs) + 1):
        total_loss = 0.0
        total_n = 0
        for x, adj, y in loader:
            x, adj, y = x.to(device), adj.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, adj), y)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(len(y))
            total_n += int(len(y))
        if args.exp6_verbose_epochs:
            log_progress(
                f"EXP6 v3 finetune epoch={epoch}/{args.exp6_epochs} strategy={strategy} "
                f"samples={len(dataset)} loss={total_loss / max(total_n, 1):.4f}"
            )
    model.eval()
    return model


def run_exp6_v3(args: argparse.Namespace, feature_set: str, model_name: str) -> pd.DataFrame:
    log_progress(f"EXP6 v3 start feature_set={feature_set} model={model_name}")
    processed_dir = args.processed_root / "pamap2" / feature_set / "overlapping"
    X, y, subjects, source_indices, _mapping, _inv, _data_manifest = load_processed_dataset(
        "pamap2",
        max_windows_per_subject=None,
        seed=args.seed,
        processed_dir=str(processed_dir),
    )
    device = select_device(args.device)
    n_classes = len(np.unique(y))
    artifact = canonical_v3_artifact(args.v3_root, feature_set, "loso", model_name)
    manifest = read_json(artifact / "dataset_manifest.json")
    folds = pd.read_csv(artifact / "fold_split_subjects.csv")
    rows = []
    for _, fold_row in folds.iterrows():
        fold = int(fold_row["fold"])
        test_subject = fold_row["test_subject"]
        fold_start = time.time()
        log_progress(f"EXP6 v3 load checkpoint feature_set={feature_set} model={model_name} fold={fold} subject={test_subject}")
        base_model, _adj, ckpt_path = load_v3_model(artifact, model_name, fold, X, n_classes, device)
        te_ds, _yt_raw, _subj_raw, note = make_loso_deep_test_dataset(
            X=X,
            y=y,
            subjects=subjects,
            source_indices=source_indices,
            fold=fold,
            test_subject=test_subject,
            model_name=model_name,
            sequence_length=int(manifest.get("sequence_length", args.sequence_length)),
            sequence_stride=int(manifest.get("sequence_stride", args.sequence_stride)),
            sequence_target_policy=str(manifest.get("sequence_target_policy", "last")),
            standardize_input=bool(manifest.get("standardize_input", False)),
            seed=args.seed,
        )
        if te_ds is None:
            raise RuntimeError(note)
        for pct in [float(x) for x in parse_csv(args.calibration_percentages)]:
            cal_idx, test_idx = calibration_split(len(te_ds), pct, args.seed + fold)
            eval_ds = Subset(te_ds, test_idx.tolist())
            log_progress(
                f"EXP6 v3 eval base feature_set={feature_set} model={model_name} fold={fold} "
                f"subject={test_subject} pct={pct} cal_samples={len(cal_idx)} eval_samples={len(test_idx)}"
            )
            y_true_base, y_pred_base, _ = predict_deep_dataset(base_model, eval_ds, batch_size=args.batch_size, device=device)
            base_m = metrics(y_true_base, y_pred_base, n_classes)
            if pct == 0:
                rows.append({
                    "dataset": "pamap2",
                    "feature_set": feature_set,
                    "protocol": "loso",
                    "family": "deep",
                    "model": model_name,
                    "fold": fold,
                    "test_subject": str(test_subject),
                    "calibration_percentage": pct,
                    "fine_tuning_strategy": "none",
                    "uncalibrated_accuracy": base_m["accuracy"],
                    "calibrated_accuracy": base_m["accuracy"],
                    "accuracy_improvement": 0.0,
                    "uncalibrated_macro_f1": base_m["macro_f1"],
                    "calibrated_macro_f1": base_m["macro_f1"],
                    "macro_f1_improvement": 0.0,
                    "n_calibration_samples": 0,
                    "n_eval_samples": int(len(test_idx)),
                    "checkpoint_path": str(ckpt_path),
                    "status": "ok",
                })
                continue
            cal_ds = Subset(te_ds, cal_idx.tolist())
            for strategy in parse_csv(args.exp6_strategies):
                tune_start = time.time()
                log_progress(
                    f"EXP6 v3 finetune start feature_set={feature_set} model={model_name} fold={fold} "
                    f"subject={test_subject} pct={pct} strategy={strategy} cal_samples={len(cal_idx)}"
                )
                tuned = finetune_deep(base_model, cal_ds, strategy=strategy, args=args, device=device, n_classes=n_classes)
                yt, yp, _ = predict_deep_dataset(tuned, eval_ds, batch_size=args.batch_size, device=device)
                cal_m = metrics(yt, yp, n_classes)
                log_progress(
                    f"EXP6 v3 finetune done feature_set={feature_set} model={model_name} fold={fold} "
                    f"pct={pct} strategy={strategy} macro_f1_improvement={cal_m['macro_f1'] - base_m['macro_f1']:.4f} "
                    f"sec={time.time() - tune_start:.1f}"
                )
                rows.append({
                    "dataset": "pamap2",
                    "feature_set": feature_set,
                    "protocol": "loso",
                    "family": "deep",
                    "model": model_name,
                    "fold": fold,
                    "test_subject": str(test_subject),
                    "calibration_percentage": pct,
                    "fine_tuning_strategy": strategy,
                    "uncalibrated_accuracy": base_m["accuracy"],
                    "calibrated_accuracy": cal_m["accuracy"],
                    "accuracy_improvement": cal_m["accuracy"] - base_m["accuracy"],
                    "uncalibrated_macro_f1": base_m["macro_f1"],
                    "calibrated_macro_f1": cal_m["macro_f1"],
                    "macro_f1_improvement": cal_m["macro_f1"] - base_m["macro_f1"],
                    "n_calibration_samples": int(len(cal_idx)),
                    "n_eval_samples": int(len(test_idx)),
                    "checkpoint_path": str(ckpt_path),
                    "status": "ok",
                })
        log_progress(
            f"EXP6 v3 fold done feature_set={feature_set} model={model_name} fold={fold} "
            f"subject={test_subject} sec={time.time() - fold_start:.1f}"
        )
    log_progress(f"EXP6 v3 done feature_set={feature_set} model={model_name} rows={len(rows)}")
    return pd.DataFrame(rows)


def run_exp6_baselines(args: argparse.Namespace, feature_set: str, model_map: dict[str, object]) -> pd.DataFrame:
    log_progress(f"EXP6 baseline block start feature_set={feature_set} models={','.join(model_map.keys())}")
    X, y_raw, subjects, _meta, manifest = load_canonical_dataset(args.processed_root, "pamap2", feature_set, "overlapping")
    feature_columns = raw_feature_columns(manifest, feature_set, X.shape[-1])
    X_feat, _feature_names = extract_window_features(X, feature_columns)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    rows = []
    for model_name, base_model in model_map.items():
        log_progress(f"EXP6 baseline model start feature_set={feature_set} model={model_name}")
        for fold, (train_idx, held_idx, test_subject, _n_test) in enumerate(
            split_indices("loso", subjects, y, args.seed, args.test_fraction),
            start=1,
        ):
            t0 = time.time()
            base_fitted, base_local_classes = baseline_fit_model(base_model, model_name, X_feat[train_idx], y[train_idx])
            log_progress(
                f"EXP6 baseline base fit done feature_set={feature_set} model={model_name} "
                f"fold={fold} subject={test_subject} sec={time.time() - t0:.1f}"
            )
            for pct in [float(x) for x in parse_csv(args.calibration_percentages)]:
                pct_start = time.time()
                local_cal, local_test = calibration_split(len(held_idx), pct, args.seed + fold)
                cal_idx = held_idx[local_cal]
                eval_idx = held_idx[local_test]
                log_progress(
                    f"EXP6 baseline pct start feature_set={feature_set} model={model_name} fold={fold} "
                    f"subject={test_subject} pct={pct} cal_samples={len(cal_idx)} eval_samples={len(eval_idx)}"
                )
                base_pred = baseline_predict_model(base_fitted, base_local_classes, X_feat[eval_idx])
                base_acc = float(accuracy_score(y[eval_idx], base_pred))
                base_f1 = float(f1_score(y[eval_idx], base_pred, average="macro", zero_division=0))
                if pct == 0:
                    rows.append({
                        "dataset": "pamap2",
                        "feature_set": feature_set,
                        "protocol": "loso",
                        "family": "baseline",
                        "model": model_name,
                        "fold": fold,
                        "test_subject": str(test_subject),
                        "calibration_percentage": pct,
                        "fine_tuning_strategy": "none",
                        "uncalibrated_accuracy": base_acc,
                        "calibrated_accuracy": base_acc,
                        "accuracy_improvement": 0.0,
                        "uncalibrated_macro_f1": base_f1,
                        "calibrated_macro_f1": base_f1,
                        "macro_f1_improvement": 0.0,
                        "n_calibration_samples": 0,
                        "n_eval_samples": int(len(eval_idx)),
                        "checkpoint_path": "",
                        "status": "ok",
                    })
                    continue
                combined = np.concatenate([train_idx, cal_idx])
                pred = baseline_fit_predict(base_model, model_name, X_feat[combined], y[combined], X_feat[eval_idx])
                cal_acc = float(accuracy_score(y[eval_idx], pred))
                cal_f1 = float(f1_score(y[eval_idx], pred, average="macro", zero_division=0))
                log_progress(
                    f"EXP6 baseline pct done feature_set={feature_set} model={model_name} fold={fold} "
                    f"pct={pct} macro_f1_improvement={cal_f1 - base_f1:.4f} sec={time.time() - pct_start:.1f}"
                )
                rows.append({
                    "dataset": "pamap2",
                    "feature_set": feature_set,
                    "protocol": "loso",
                    "family": "baseline",
                    "model": model_name,
                    "fold": fold,
                    "test_subject": str(test_subject),
                    "calibration_percentage": pct,
                    "fine_tuning_strategy": "refit_train_plus_subject_shots",
                    "uncalibrated_accuracy": base_acc,
                    "calibrated_accuracy": cal_acc,
                    "accuracy_improvement": cal_acc - base_acc,
                    "uncalibrated_macro_f1": base_f1,
                    "calibrated_macro_f1": cal_f1,
                    "macro_f1_improvement": cal_f1 - base_f1,
                    "n_calibration_samples": int(len(cal_idx)),
                    "n_eval_samples": int(len(eval_idx)),
                    "checkpoint_path": "",
                    "status": "ok",
                })
        log_progress(f"EXP6 baseline model done feature_set={feature_set} model={model_name} rows={len(rows)}")
    return pd.DataFrame(rows)


def args_for_worker(args: argparse.Namespace) -> dict[str, Any]:
    data = vars(args).copy()
    for key in ["processed_root", "v3_root", "out_root"]:
        data[key] = str(data[key])
    return data


def args_from_worker(data: dict[str, Any]) -> argparse.Namespace:
    restored = data.copy()
    for key in ["processed_root", "v3_root", "out_root"]:
        restored[key] = Path(restored[key])
    return argparse.Namespace(**restored)


def baseline_worker(args_data: dict[str, Any], experiment: str, feature_set: str, model_name: str) -> tuple[str, pd.DataFrame]:
    args = args_from_worker(args_data)
    os.environ["OMP_NUM_THREADS"] = str(max(1, int(args.baseline_estimator_jobs)))
    os.environ["MKL_NUM_THREADS"] = str(max(1, int(args.baseline_estimator_jobs)))
    os.environ["OPENBLAS_NUM_THREADS"] = str(max(1, int(args.baseline_estimator_jobs)))
    os.environ["NUMEXPR_NUM_THREADS"] = str(max(1, int(args.baseline_estimator_jobs)))
    args.baseline_models = model_name
    model_map = make_baseline_model_map(args)
    if not model_map:
        return model_name, pd.DataFrame()
    if experiment == "exp3":
        return model_name, run_exp3_baselines(args, feature_set, model_map)
    if experiment == "exp6":
        return model_name, run_exp6_baselines(args, feature_set, model_map)
    raise ValueError(experiment)


def run_baseline_models(args: argparse.Namespace, experiment: str, feature_set: str, model_names: Sequence[str]) -> list[pd.DataFrame]:
    names = [name for name in model_names if name]
    if not names:
        return []
    parallel_jobs = max(1, int(args.baseline_parallel_jobs))
    if parallel_jobs == 1 or len(names) == 1:
        frames = []
        for name in names:
            started = time.time()
            log_progress(f"{experiment.upper()} baseline start feature_set={feature_set} model={name}")
            _name, df = baseline_worker(args_for_worker(args), experiment, feature_set, name)
            log_progress(f"{experiment.upper()} baseline done feature_set={feature_set} model={name} rows={len(df)} sec={time.time() - started:.1f}")
            frames.append(df)
        return frames

    frames: list[pd.DataFrame] = []
    log_progress(
        f"{experiment.upper()} baseline parallel feature_set={feature_set} "
        f"models={len(names)} parallel_jobs={parallel_jobs} estimator_jobs={args.baseline_estimator_jobs}"
    )
    with ProcessPoolExecutor(max_workers=parallel_jobs) as executor:
        start_times = {}
        future_map = {
            executor.submit(baseline_worker, args_for_worker(args), experiment, feature_set, name): name
            for name in names
        }
        start_times = {future: time.time() for future in future_map}
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                model_name, df = future.result()
            except Exception as exc:
                log_progress(f"{experiment.upper()} baseline failed feature_set={feature_set} model={name} error={exc}")
                raise
            log_progress(
                f"{experiment.upper()} baseline done feature_set={feature_set} model={model_name} "
                f"rows={len(df)} sec={time.time() - start_times[future]:.1f}"
            )
            frames.append(df)
    return frames


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real canonical protocol12 Exp3/Exp6.")
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/canonical_protocol_only"))
    parser.add_argument("--v3-root", type=Path, default=Path("results/canonical_protocol_only_v3"))
    parser.add_argument("--out-root", type=Path, default=Path("results/canonical_protocol12_seven_experiments"))
    parser.add_argument("--feature-sets", default=",".join(DEFAULT_FEATURE_SETS))
    parser.add_argument("--experiments", default="exp3,exp6")
    parser.add_argument("--families", default="baseline,v3")
    parser.add_argument("--baseline-models", default=",".join(DEFAULT_BASELINE_MODELS))
    parser.add_argument("--v3-models", default=",".join(DEFAULT_V3_MODELS))
    parser.add_argument("--include-xgb", action="store_true")
    parser.add_argument("--xgb-cuda", action="store_true")
    parser.add_argument("--fast-baselines", action="store_true")
    parser.add_argument("--baseline-parallel-jobs", type=int, default=3, help="Number of baseline models to evaluate concurrently.")
    parser.add_argument("--baseline-estimator-jobs", type=int, default=4, help="n_jobs cap passed into sklearn/XGBoost estimators inside each baseline worker.")
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--sequence-stride", type=int, default=1)
    parser.add_argument("--perturbations", default="gaussian_noise,random_channel_dropout,heart_rate_zero")
    parser.add_argument("--severities", default="low,medium,high")
    parser.add_argument("--calibration-percentages", default=",".join(str(x) for x in CALIBRATION_PCTS))
    parser.add_argument("--exp6-strategies", default="classifier_head_only,full_model")
    parser.add_argument("--exp6-epochs", type=int, default=5)
    parser.add_argument("--exp6-lr", type=float, default=1e-4)
    parser.add_argument("--exp6-weight-decay", type=float, default=1e-4)
    parser.add_argument("--exp6-verbose-epochs", action="store_true", help="Print each fine-tuning epoch loss for Exp6 v3.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    feature_sets = parse_csv(args.feature_sets)
    experiments = set(parse_csv(args.experiments))
    families = set(parse_csv(args.families))
    tables = args.out_root / "manuscript_tables"
    detailed = args.out_root / "real_exp3_exp6"
    tables.mkdir(parents=True, exist_ok=True)
    detailed.mkdir(parents=True, exist_ok=True)
    progress_log = detailed / "progress_events.jsonl"
    progress_log.write_text("", encoding="utf-8")
    os.environ["HAR_REAL_EXP_PROGRESS_LOG"] = str(progress_log)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "processed_root": str(args.processed_root),
        "v3_root": str(args.v3_root),
        "out_root": str(args.out_root),
        "progress_log": str(progress_log),
        "args": vars(args) | {
            "processed_root": str(args.processed_root),
            "v3_root": str(args.v3_root),
            "out_root": str(args.out_root),
        },
        "note": "Real Exp3/Exp6: v3 uses canonical checkpoints only; baselines are refit from canonical data because baseline estimators were not checkpointed.",
    }
    safe_json_dump(manifest, detailed / "run_manifest.json")

    log_progress(
        "real Exp3/Exp6 start "
        f"experiments={','.join(sorted(experiments))} families={','.join(sorted(families))} "
        f"feature_sets={','.join(feature_sets)} baseline_parallel_jobs={args.baseline_parallel_jobs} "
        f"baseline_estimator_jobs={args.baseline_estimator_jobs} device={args.device} batch_size={args.batch_size}"
    )

    baseline_model_names = parse_csv(args.baseline_models) if "baseline" in families else []
    if baseline_model_names == ["all"]:
        baseline_model_names = list(make_baseline_model_map(args))
    exp3_frames, exp6_frames = [], []

    if "exp3" in experiments:
        target = tables / "table_exp3_robustness.csv"
        if args.skip_existing and target.exists():
            print(f"[SKIP] {target}")
        else:
            for feature_set in feature_sets:
                if "baseline" in families:
                    log_progress(f"EXP3 baseline feature_set start feature_set={feature_set}")
                    exp3_frames.extend(run_baseline_models(args, "exp3", feature_set, baseline_model_names))
                if "v3" in families:
                    for model_name in parse_csv(args.v3_models):
                        log_progress(f"EXP3 v3 feature/model start feature_set={feature_set} model={model_name}")
                        exp3_frames.append(run_exp3_v3(args, feature_set, model_name))
            exp3 = pd.concat(exp3_frames, ignore_index=True) if exp3_frames else pd.DataFrame()
            write_csv(exp3, target)
            write_csv(exp3, detailed / "exp3_robustness_detailed.csv")
            log_progress(f"EXP3 table written path={target} rows={len(exp3)}")

    if "exp6" in experiments:
        target = tables / "table_exp6_few_shot_calibration.csv"
        if args.skip_existing and target.exists():
            print(f"[SKIP] {target}")
        else:
            for feature_set in feature_sets:
                if "baseline" in families:
                    log_progress(f"EXP6 baseline feature_set start feature_set={feature_set}")
                    exp6_frames.extend(run_baseline_models(args, "exp6", feature_set, baseline_model_names))
                if "v3" in families:
                    for model_name in parse_csv(args.v3_models):
                        log_progress(f"EXP6 v3 feature/model start feature_set={feature_set} model={model_name}")
                        exp6_frames.append(run_exp6_v3(args, feature_set, model_name))
            exp6 = pd.concat(exp6_frames, ignore_index=True) if exp6_frames else pd.DataFrame()
            write_csv(exp6, target)
            write_csv(exp6, detailed / "exp6_few_shot_calibration_detailed.csv")
            log_progress(f"EXP6 table written path={target} rows={len(exp6)}")

    log_progress(f"real Exp3/Exp6 done out_root={args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
