#!/usr/bin/env python
"""Run selected Phase 1 HAR baselines without modifying the original script.

This file intentionally imports the existing scripts/phase1_classical_baselines.py
for dataset loading, feature extraction, and baseline definitions, then adds:
  - selected model execution
  - fold-level prediction checkpoints
  - optional fold-level model checkpoints
  - skip/resume behavior
  - safe model-specific artifact names
  - summary combining across parallel processes

Typical use from repo root:
  python scripts/phase1_selected_model_runner.py --dataset hhar --models xgboost_hist --include-xgb --run-dir results/phase1_parallel/20260704_150000
  python scripts/phase1_selected_model_runner.py --dataset hhar --models rbf_svm --run-dir results/phase1_parallel/20260704_150000 --rbf-max-train-samples 50000
  python scripts/phase1_selected_model_runner.py --dataset hhar --run-dir results/phase1_parallel/20260704_150000 --combine-only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from joblib import dump
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

SEED = 42
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


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def repo_root() -> Path:
    return Path.cwd().resolve()


def import_base_module(repo: Path):
    base_path = repo / "scripts" / "phase1_classical_baselines.py"
    if not base_path.exists():
        raise FileNotFoundError(f"Expected existing baseline script at {base_path}")
    spec = importlib.util.spec_from_file_location("phase1_classical_baselines_base", str(base_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {base_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def set_estimator_threads(model, n_threads: Optional[int]):
    if not n_threads or n_threads <= 0:
        return model
    # sklearn Pipelines need nested parameter names.
    if isinstance(model, Pipeline):
        params = {}
        for step_name, step in model.steps:
            if hasattr(step, "n_jobs"):
                params[f"{step_name}__n_jobs"] = n_threads
        if params:
            model.set_params(**params)
        return model
    if hasattr(model, "n_jobs"):
        try:
            model.set_params(n_jobs=n_threads)
        except Exception:
            try:
                model.n_jobs = n_threads
            except Exception:
                pass
    return model


def load_dataset(base, args):
    raw_root = Path(args.data_root)
    if args.dataset == "pamap2":
        X, y, subjects, meta, feature_names, class_names = base.load_pamap2_dataset(
            raw_root=raw_root,
            task=args.pamap2_task,
            feature_set=args.pamap2_feature_set,
            window=args.pamap2_window,
            step=args.pamap2_step,
            sessions=args.pamap2_sessions,
        )
        exp_name = f"pamap2_{args.pamap2_task}_{args.pamap2_sessions}_{args.pamap2_feature_set}_w{args.pamap2_window}_s{args.pamap2_step}"
    elif args.dataset == "hhar":
        X, y, subjects, meta, feature_names, class_names = base.load_hhar_dataset(
            raw_root=raw_root,
            window=args.hhar_window,
            step=args.hhar_step,
        )
        exp_name = f"hhar_w{args.hhar_window}_s{args.hhar_step}"
    else:
        raise ValueError(args.dataset)
    return X, y, subjects, meta, feature_names, class_names, exp_name


def get_experiment_dir(args, exp_name: str) -> Path:
    if args.experiment_dir:
        return Path(args.experiment_dir)
    if args.run_dir:
        return Path(args.run_dir) / exp_name
    return Path(args.out_dir) / now_stamp() / exp_name


def write_dataset_artifacts(ds_out: Path, args, X, y, subjects, meta, feature_names) -> None:
    ensure_dir(ds_out)
    config_path = ds_out / "selected_runner_config.json"
    if not config_path.exists():
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)
    manifest_path = ds_out / "dataset_manifest.json"
    if not manifest_path.exists():
        dataset_info = {
            "dataset": args.dataset,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_subjects": int(len(np.unique(subjects))),
            "labels": [str(x) for x in sorted(np.unique(y).tolist())],
            "subjects": [str(x) for x in sorted(np.unique(subjects).tolist())],
            "runner": "phase1_selected_model_runner.py",
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2)
    feature_path = ds_out / "feature_columns.json"
    if not feature_path.exists():
        with open(feature_path, "w", encoding="utf-8") as f:
            json.dump(list(feature_names), f, indent=2)
    window_manifest = ds_out / "window_manifest.csv"
    if not window_manifest.exists():
        meta.to_csv(window_manifest, index=False)
    if args.dataset == "pamap2":
        audit_path = ds_out / "pamap2_timestamp_audit.csv"
        if not audit_path.exists():
            pd.DataFrame(
                meta.attrs.get("timestamp_audit_rows", []),
                columns=PAMAP2_TIMESTAMP_AUDIT_COLUMNS,
            ).to_csv(audit_path, index=False)


def selected_model_names(args) -> List[str]:
    names: List[str] = []
    if args.models:
        names = [m.strip() for m in args.models.split(",") if m.strip()]
    elif args.model:
        names = [args.model.strip()]
    if not names:
        raise ValueError("Provide --model or --models")
    return names


def artifact_model_name(model_name: str, args) -> str:
    if model_name == "rbf_svm" and args.rbf_max_train_samples and args.rbf_max_train_samples > 0:
        return f"rbf_svm_traincap{args.rbf_max_train_samples}"
    return model_name


def make_available_models(base, args) -> Dict[str, object]:
    requested = selected_model_names(args)
    include_xgb = args.include_xgb or any(m.startswith("xgboost") for m in requested)
    # Keep fast=False so rbf_svm exists when explicitly requested.
    models = base.make_models(include_xgb=include_xgb, use_cuda=args.use_cuda, fast=False)
    return models


def loso_folds(subjects: np.ndarray, selected_folds: Optional[Sequence[str]] = None) -> Iterable[Tuple[np.ndarray, np.ndarray, str]]:
    selected = None if not selected_folds else {str(x) for x in selected_folds}
    for subj in np.unique(subjects):
        subj_s = str(subj)
        if selected is not None and subj_s not in selected:
            continue
        test_idx = np.where(subjects == subj)[0]
        train_idx = np.where(subjects != subj)[0]
        if len(test_idx) and len(train_idx):
            yield train_idx, test_idx, subj_s


def maybe_subsample_train(train_idx: np.ndarray, y: np.ndarray, max_train: Optional[int], seed: int) -> Tuple[np.ndarray, str]:
    if not max_train or max_train <= 0 or len(train_idx) <= max_train:
        return train_idx, ""
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=max_train, random_state=seed)
    rel = np.arange(len(train_idx))
    # If stratified sampling fails for rare classes, fall back to random sampling.
    try:
        sub_rel, _ = next(splitter.split(rel, y[train_idx]))
        new_idx = train_idx[sub_rel]
    except Exception:
        rng = np.random.default_rng(seed)
        new_idx = rng.choice(train_idx, size=max_train, replace=False)
    note = f"train set stratified/random capped from {len(train_idx)} to {len(new_idx)} samples"
    return np.asarray(new_idx), note


def fold_pred_path(ds_out: Path, art_name: str, fold_subject: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(fold_subject))
    return ds_out / "fold_predictions" / art_name / f"predictions_fold_subject_{safe}.csv"


def fold_model_path(ds_out: Path, art_name: str, fold_subject: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(fold_subject))
    return ds_out / "models" / art_name / f"fold_subject_{safe}.joblib"


def load_existing_fold_prediction(path: Path) -> Optional[pd.DataFrame]:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return None


def fit_predict_fold_model(model, model_name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, str]:
    if model_name.startswith("xgboost"):
        train_classes = np.unique(y_train).astype(int)
        local_y = np.searchsorted(train_classes, y_train.astype(int))
        model.fit(X_train, local_y)
        local_pred = model.predict(X_test).astype(int)
        pred = train_classes[local_pred]
        note = ""
        if not np.array_equal(train_classes, np.arange(train_classes.size)):
            note = "xgboost trained with fold-local contiguous label ids and predictions mapped back to global ids"
        return pred.astype(int), note
    model.fit(X_train, y_train)
    return model.predict(X_test).astype(int), ""


def run_one_model(base, args, model_name: str, X, y_raw, subjects, meta, feature_names, ds_out: Path) -> Dict[str, object]:
    models = make_available_models(base, args)
    if model_name not in models:
        raise ValueError(f"Unknown model {model_name}. Available models: {sorted(models.keys())}")
    art_name = artifact_model_name(model_name, args)
    model_dir = ds_out / "models" / art_name
    fold_pred_dir = ds_out / "fold_predictions" / art_name
    ensure_dir(model_dir)
    ensure_dir(fold_pred_dir)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    labels = list(le.classes_)
    if hasattr(base, "display_class_names"):
        labels_display = base.display_class_names(args.dataset, labels)
    else:
        labels_display = [str(x) for x in labels]
    label_mapping_path = ds_out / "label_mapping.json"
    if not label_mapping_path.exists():
        with open(label_mapping_path, "w", encoding="utf-8") as f:
            json.dump({int(i): str(lbl) for i, lbl in enumerate(labels)}, f, indent=2)
    label_names_path = ds_out / "label_names.json"
    if not label_names_path.exists():
        with open(label_names_path, "w", encoding="utf-8") as f:
            json.dump({int(i): labels_display[int(i)] for i in range(len(labels_display))}, f, indent=2)

    base_model = set_estimator_threads(models[model_name], args.estimator_threads)
    selected_folds = [f.strip() for f in args.folds.split(",") if f.strip()] if args.folds else None

    pred_rows: List[pd.DataFrame] = []
    fold_rows: List[Dict[str, object]] = []
    start_model = time.time()

    print(f"\n=== Running {model_name} as artifact {art_name} ===", flush=True)
    if model_name == "rbf_svm" and not args.rbf_max_train_samples:
        print("[WARN] Full RBF SVM on HHAR can take many hours. Use --rbf-max-train-samples for a practical capped run.", flush=True)

    for fold_n, (train_idx, test_idx, fold_subject) in enumerate(loso_folds(subjects, selected_folds=selected_folds), start=1):
        if args.max_folds and fold_n > args.max_folds:
            break
        fp = fold_pred_path(ds_out, art_name, fold_subject)
        mp = fold_model_path(ds_out, art_name, fold_subject)

        if args.skip_existing and fp.exists():
            existing = load_existing_fold_prediction(fp)
            if existing is not None:
                pred_rows.append(existing)
                if {"y_true_id", "y_pred_id"}.issubset(existing.columns):
                    m = base.metrics_dict(existing["y_true_id"].to_numpy(), existing["y_pred_id"].to_numpy())
                    fold_rows.append({
                        "model": art_name,
                        "base_model": model_name,
                        "fold_subject": fold_subject,
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        "fit_predict_sec": 0.0,
                        "skipped_existing": True,
                        "train_cap_note": "",
                        **m,
                    })
                print(f"fold subject={fold_subject}: skipped existing {fp}", flush=True)
                continue

        model = clone(base_model)
        effective_train_idx = train_idx
        train_cap_note = ""
        if model_name == "rbf_svm" and args.rbf_max_train_samples and args.rbf_max_train_samples > 0:
            effective_train_idx, train_cap_note = maybe_subsample_train(
                train_idx, y, int(args.rbf_max_train_samples), seed=SEED + fold_n
            )

        t0 = time.time()
        try:
            pred, label_encoding_note = fit_predict_fold_model(
                model,
                model_name,
                X[effective_train_idx],
                y[effective_train_idx],
                X[test_idx],
            )
        except Exception as exc:
            err = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            print(f"[ERROR] {model_name} fold subject={fold_subject} failed: {err}", flush=True)
            fold_rows.append({
                "model": art_name,
                "base_model": model_name,
                "fold_subject": fold_subject,
                "n_train": int(len(effective_train_idx)),
                "n_test": int(len(test_idx)),
                "fit_predict_sec": float(time.time() - t0),
                "error": err,
                "train_cap_note": train_cap_note,
                "label_encoding_note": "",
            })
            pd.DataFrame(fold_rows).to_csv(ds_out / f"metrics_by_fold_{art_name}.csv", index=False)
            continue

        elapsed = time.time() - t0
        if args.save_models:
            try:
                dump(model, mp)
            except Exception as exc:
                print(f"[WARN] Could not save model {mp}: {exc}", flush=True)

        m = base.metrics_dict(y[test_idx], pred)
        fold_row = {
            "model": art_name,
            "base_model": model_name,
            "fold_subject": fold_subject,
            "n_train": int(len(effective_train_idx)),
            "n_train_original": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "fit_predict_sec": float(elapsed),
            "skipped_existing": False,
            "train_cap_note": train_cap_note,
            "label_encoding_note": label_encoding_note,
            **m,
        }
        fold_rows.append(fold_row)
        print(
            f"fold subject={fold_subject}: acc={m['accuracy']:.4f}, macro_f1={m['macro_f1']:.4f}, bacc={m['balanced_accuracy']:.4f}, sec={elapsed:.1f}",
            flush=True,
        )

        fold_meta = meta.iloc[test_idx].copy().reset_index(drop=True)
        fold_meta["fold_subject"] = fold_subject
        fold_meta["model"] = art_name
        fold_meta["base_model"] = model_name
        fold_meta["y_true_id"] = y[test_idx]
        fold_meta["y_pred_id"] = pred.astype(int)
        fold_meta["y_true"] = le.inverse_transform(y[test_idx])
        fold_meta["y_pred"] = le.inverse_transform(pred.astype(int))
        fold_meta["y_true_label"] = [labels_display[int(i)] for i in y[test_idx]]
        fold_meta["y_pred_label"] = [labels_display[int(i)] for i in pred.astype(int)]
        fold_meta["train_cap_note"] = train_cap_note
        fold_meta["label_encoding_note"] = label_encoding_note
        fold_meta.to_csv(fp, index=False)
        pred_rows.append(fold_meta)

        # Save metrics after every fold so Ctrl+C does not lose progress.
        pd.DataFrame(fold_rows).to_csv(ds_out / f"metrics_by_fold_{art_name}.csv", index=False)

    if not pred_rows:
        print(f"[WARN] No predictions generated for {art_name}", flush=True)
        return {"model": art_name, "status": "no_predictions"}

    pred_df = pd.concat(pred_rows, ignore_index=True)
    pred_df.to_csv(ds_out / f"predictions_{art_name}.csv", index=False)

    y_true_arr = pred_df["y_true_id"].to_numpy(dtype=int)
    y_pred_arr = pred_df["y_pred_id"].to_numpy(dtype=int)
    agg = base.metrics_dict(y_true_arr, y_pred_arr)
    agg.update({
        "model": art_name,
        "base_model": model_name,
        "n_samples": int(len(y_true_arr)),
        "total_sec": float(time.time() - start_model),
        "save_models": bool(args.save_models),
        "rbf_max_train_samples": int(args.rbf_max_train_samples or 0),
    })
    pd.DataFrame([agg]).to_csv(ds_out / f"metrics_summary_{art_name}.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(ds_out / f"metrics_by_fold_{art_name}.csv", index=False)

    report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=np.arange(len(labels)),
        target_names=labels_display,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(ds_out / f"classification_report_{art_name}.csv")
    with open(ds_out / f"classification_report_{art_name}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=np.arange(len(labels)))
    pd.DataFrame(cm, index=labels_display, columns=labels_display).to_csv(ds_out / f"confusion_matrix_{art_name}.csv")
    base.plot_confusion(cm, labels_display, f"{art_name} confusion matrix", ds_out / f"confusion_matrix_{art_name}.png")

    try:
        fi = base.extract_feature_importance(model, feature_names)
        if fi is not None:
            fi.to_csv(ds_out / f"feature_importance_{art_name}.csv", index=False)
    except Exception:
        pass

    print(f"\nSaved {art_name} results to: {ds_out}", flush=True)
    print(pd.DataFrame([agg])[["model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "n_samples", "total_sec"]].to_string(index=False), flush=True)
    return agg


def combine_summaries(ds_out: Path) -> None:
    summary_files = sorted(ds_out.glob("metrics_summary_*.csv"))
    fold_files = sorted(ds_out.glob("metrics_by_fold_*.csv"))
    summaries = []
    for p in summary_files:
        try:
            df = pd.read_csv(p)
            if not df.empty:
                summaries.append(df)
        except Exception as exc:
            print(f"[WARN] Could not read {p}: {exc}", flush=True)
    if summaries:
        summary = pd.concat(summaries, ignore_index=True)
        if "macro_f1" in summary.columns:
            summary = summary.sort_values("macro_f1", ascending=False)
        summary.to_csv(ds_out / "metrics_summary.csv", index=False)
        print("\nCombined metrics_summary.csv", flush=True)
        cols = [c for c in ["model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "n_samples", "total_sec"] if c in summary.columns]
        print(summary[cols].to_string(index=False), flush=True)
    else:
        print(f"[WARN] No metrics_summary_*.csv files found in {ds_out}", flush=True)

    folds = []
    for p in fold_files:
        try:
            df = pd.read_csv(p)
            if not df.empty:
                folds.append(df)
        except Exception as exc:
            print(f"[WARN] Could not read {p}: {exc}", flush=True)
    if folds:
        fold_df = pd.concat(folds, ignore_index=True)
        fold_df.to_csv(ds_out / "metrics_by_fold.csv", index=False)
        # subject-macro summary
        if {"model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"}.issubset(fold_df.columns):
            clean = fold_df.dropna(subset=["macro_f1"]).copy()
            if not clean.empty:
                subj_macro = clean.groupby("model").agg(
                    folds=("fold_subject", "count"),
                    accuracy_mean=("accuracy", "mean"),
                    accuracy_std=("accuracy", "std"),
                    balanced_accuracy_mean=("balanced_accuracy", "mean"),
                    balanced_accuracy_std=("balanced_accuracy", "std"),
                    macro_f1_mean=("macro_f1", "mean"),
                    macro_f1_std=("macro_f1", "std"),
                    weighted_f1_mean=("weighted_f1", "mean"),
                    weighted_f1_std=("weighted_f1", "std"),
                    n_test_total=("n_test", "sum"),
                ).reset_index().sort_values("macro_f1_mean", ascending=False)
                subj_macro.to_csv(ds_out / "metrics_subject_macro_summary.csv", index=False)
                print("Combined metrics_by_fold.csv and metrics_subject_macro_summary.csv", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["pamap2", "hhar"], required=True)
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--out-dir", default="results/phase1_parallel")
    p.add_argument("--run-dir", default=None, help="Timestamp/run root. Dataset experiment subfolder will be created inside this.")
    p.add_argument("--experiment-dir", default=None, help="Exact dataset experiment directory. Overrides --run-dir.")

    p.add_argument("--model", default=None, help="Single model name to run.")
    p.add_argument("--models", default=None, help="Comma-separated model names to run in this process.")
    p.add_argument("--include-xgb", action="store_true")
    p.add_argument("--use-cuda", action="store_true")
    p.add_argument("--estimator-threads", type=int, default=1)

    p.add_argument("--pamap2-window", type=int, default=512)
    p.add_argument("--pamap2-step", type=int, default=256)
    p.add_argument("--pamap2-sessions", choices=["protocol", "optional", "all"], default="all")
    p.add_argument("--pamap2-task", choices=["protocol12", "all18"], default="all18")
    p.add_argument("--pamap2-feature-set", choices=["acc16", "acc16_hr", "acc16_gyro", "acc16_gyro_hr", "allimu_hr"], default="acc16_hr")
    p.add_argument("--hhar-window", type=int, default=128)
    p.add_argument("--hhar-step", type=int, default=64)

    p.add_argument("--rbf-max-train-samples", type=int, default=0, help="Cap RBF SVM train samples per LOSO fold. 0 means full exact RBF.")
    p.add_argument("--folds", default=None, help="Comma-separated held-out subject/fold values to run only those folds.")
    p.add_argument("--max-folds", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true", default=False)
    p.add_argument("--force", action="store_true", help="Disable skip-existing behavior.")
    p.add_argument("--save-models", action="store_true", default=True)
    p.add_argument("--no-save-models", dest="save_models", action="store_false")
    p.add_argument("--combine-only", action="store_true", help="Do not train; combine metrics_summary_*.csv and metrics_by_fold_*.csv in the experiment dir.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.force:
        args.skip_existing = False

    # Keep BLAS/OpenMP from over-threading when PowerShell launches parallel processes.
    if args.estimator_threads and args.estimator_threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.estimator_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.estimator_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.estimator_threads))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(args.estimator_threads))

    repo = repo_root()
    base = import_base_module(repo)

    if args.combine_only:
        # Need experiment name to find ds_out unless exact experiment dir was supplied.
        if args.experiment_dir:
            ds_out = Path(args.experiment_dir)
        else:
            if args.dataset == "pamap2":
                exp_name = f"pamap2_{args.pamap2_task}_{args.pamap2_sessions}_{args.pamap2_feature_set}_w{args.pamap2_window}_s{args.pamap2_step}"
            else:
                exp_name = f"hhar_w{args.hhar_window}_s{args.hhar_step}"
            if not args.run_dir:
                raise ValueError("--combine-only needs --run-dir or --experiment-dir")
            ds_out = Path(args.run_dir) / exp_name
        combine_summaries(ds_out)
        return

    X, y_raw, subjects, meta, feature_names, class_names, exp_name = load_dataset(base, args)
    ds_out = get_experiment_dir(args, exp_name)
    write_dataset_artifacts(ds_out, args, X, y_raw, subjects, meta, feature_names)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Experiment dir: {ds_out}", flush=True)
    print(f"Generated feature matrix: X={X.shape}, labels={len(np.unique(y_raw))}, subjects={len(np.unique(subjects))}", flush=True)

    results = []
    for model_name in selected_model_names(args):
        results.append(run_one_model(base, args, model_name, X, y_raw, subjects, meta, feature_names, ds_out))

    combine_summaries(ds_out)


if __name__ == "__main__":
    main()
