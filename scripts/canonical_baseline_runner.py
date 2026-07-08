#!/usr/bin/env python
"""Run canonical classical baseline comparisons.

Outputs are written to stable, interpretable paths:

results/canonical/core_comparison/<dataset>/<feature_set>/<window_type>/<protocol>/baselines/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase1_classical_baselines import (
    display_class_names,
    ensure_dir,
    extract_feature_importance,
    extract_window_features,
    fit_predict_fold_model,
    make_models,
    metrics_dict,
    plot_confusion,
    plot_model_comparison,
)


DEFAULT_FEATURE_SETS = ["acc16_hr", "acc16_gyro", "acc16_gyro_hr"]
DEFAULT_WINDOW_TYPES = ["overlapping"]
DEFAULT_PROTOCOLS = ["loso", "random_holdout"]


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value).replace(";", ",").split(",") if x.strip()]


def load_canonical_dataset(
    processed_root: Path,
    dataset: str,
    feature_set: str,
    window_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict]:
    base = processed_root / dataset / feature_set / window_type
    prefix = dataset
    paths = {
        "X": base / f"{prefix}_X.npy",
        "y": base / f"{prefix}_y.npy",
        "subjects": base / f"{prefix}_subjects.npy",
        "window_manifest": base / f"{prefix}_window_manifest.csv",
        "processed_manifest": base / f"{prefix}_processed_manifest.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Canonical processed files are missing. Run scripts/canonical_prepare_datasets.py first. Missing: "
            + ", ".join(missing)
        )

    X = np.load(paths["X"], allow_pickle=False)
    y = np.load(paths["y"], allow_pickle=False)
    subjects = np.load(paths["subjects"], allow_pickle=False)
    meta = pd.read_csv(paths["window_manifest"])
    with open(paths["processed_manifest"], "r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    return X, y, subjects, meta, manifest


def cap_windows_per_subject(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    meta: pd.DataFrame,
    max_windows_per_subject: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    if not max_windows_per_subject or max_windows_per_subject <= 0:
        return X, y, subjects, meta
    rng = np.random.default_rng(seed)
    keep_parts: list[np.ndarray] = []
    for subject in np.unique(subjects):
        idx = np.where(subjects == subject)[0]
        if len(idx) > max_windows_per_subject:
            idx = rng.choice(idx, size=max_windows_per_subject, replace=False)
        keep_parts.append(np.sort(idx))
    keep = np.concatenate(keep_parts)
    keep.sort()
    return X[keep], y[keep], subjects[keep], meta.iloc[keep].reset_index(drop=True)


def split_indices(protocol: str, subjects: np.ndarray, y: np.ndarray, seed: int, test_fraction: float):
    if protocol == "loso":
        for subject in np.unique(subjects):
            test_idx = np.where(subjects == subject)[0]
            train_idx = np.where(subjects != subject)[0]
            if len(train_idx) and len(test_idx):
                yield train_idx, test_idx, str(subject), int(len(test_idx))
        return

    if protocol == "random_holdout":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(y))
        n_test = max(1, int(round(len(y) * test_fraction)))
        test_idx = np.sort(perm[:n_test])
        train_idx = np.sort(perm[n_test:])
        yield train_idx, test_idx, "mixed_subjects", int(len(test_idx))
        return

    raise ValueError(f"Unsupported protocol: {protocol}")


def selected_models(include_xgb: bool, use_cuda: bool, fast: bool, models_arg: str) -> dict[str, object]:
    models = make_models(include_xgb=include_xgb, use_cuda=use_cuda, fast=fast)
    requested = parse_csv(models_arg)
    if not requested or requested == ["all"]:
        return models
    missing = sorted(set(requested) - set(models))
    if missing:
        raise ValueError("Unknown baseline model(s): " + ", ".join(missing) + ". Available: " + ", ".join(sorted(models)))
    return {name: models[name] for name in requested}


def predict_proba_global(model: object, X_test: np.ndarray, n_classes: int) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    try:
        local_proba = model.predict_proba(X_test)
    except Exception:
        return None
    if local_proba is None:
        return None
    local_proba = np.asarray(local_proba, dtype=float)
    if local_proba.ndim != 2 or local_proba.shape[0] != len(X_test):
        return None
    proba = np.zeros((local_proba.shape[0], n_classes), dtype=float)
    classes = getattr(model, "classes_", None)
    if classes is None:
        if local_proba.shape[1] == n_classes:
            return local_proba
        return None
    for local_idx, cls in enumerate(np.asarray(classes).astype(int)):
        if 0 <= int(cls) < n_classes and local_idx < local_proba.shape[1]:
            proba[:, int(cls)] = local_proba[:, local_idx]
    row_sum = proba.sum(axis=1, keepdims=True)
    valid = row_sum[:, 0] > 0
    proba[valid] = proba[valid] / row_sum[valid]
    return proba


def write_common_manifests(
    out_dir: Path,
    *,
    dataset: str,
    feature_set: str,
    window_type: str,
    protocol: str,
    seed: int,
    max_windows_per_subject: int | None,
    manifest: dict,
    feature_names: Sequence[str],
    labels_display: Sequence[str],
) -> None:
    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as fp:
        json.dump({
            "dataset": dataset,
            "feature_set": feature_set,
            "window_type": window_type,
            "protocol": protocol,
            "model_family": "baselines",
            "seed": int(seed),
            "max_windows_per_subject": max_windows_per_subject,
            "is_capped_smoke": bool(max_windows_per_subject and max_windows_per_subject > 0),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source_processed_manifest": manifest,
        }, fp, indent=2)
    with open(out_dir / "feature_columns.json", "w", encoding="utf-8") as fp:
        json.dump(list(feature_names), fp, indent=2)
    with open(out_dir / "label_names.json", "w", encoding="utf-8") as fp:
        json.dump({int(i): str(name) for i, name in enumerate(labels_display)}, fp, indent=2)


def run_protocol(
    *,
    X_feat: np.ndarray,
    y_raw: np.ndarray,
    subjects: np.ndarray,
    meta: pd.DataFrame,
    feature_names: Sequence[str],
    dataset: str,
    feature_set: str,
    window_type: str,
    protocol: str,
    out_dir: Path,
    include_xgb: bool,
    use_cuda: bool,
    fast: bool,
    models_arg: str,
    seed: int,
    test_fraction: float,
    max_windows_per_subject: int | None,
    skip_existing: bool,
    require_probabilities: bool,
    manifest: dict,
) -> None:
    ensure_dir(out_dir)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    labels = list(le.classes_)
    labels_display = display_class_names(dataset, labels)
    write_common_manifests(
        out_dir,
        dataset=dataset,
        feature_set=feature_set,
        window_type=window_type,
        protocol=protocol,
        seed=seed,
        max_windows_per_subject=max_windows_per_subject,
        manifest=manifest,
        feature_names=feature_names,
        labels_display=labels_display,
    )

    models = selected_models(include_xgb, use_cuda, fast, models_arg)
    fold_rows: list[dict] = []
    summary_rows: list[dict] = []

    for model_name, base_model in models.items():
        pred_file = out_dir / f"predictions_{model_name}.csv"
        if skip_existing and pred_file.exists():
            try:
                existing_pred = pd.read_csv(pred_file)
                if {"y_true_id", "y_pred_id"}.issubset(existing_pred.columns):
                    has_proba = any(str(c).startswith("proba_") for c in existing_pred.columns)
                    if require_probabilities and not has_proba:
                        print(f"[REFIT] {model_name}: existing predictions lack probability columns: {pred_file}")
                    else:
                        y_true_arr = existing_pred["y_true_id"].to_numpy(dtype=int)
                        y_pred_arr = existing_pred["y_pred_id"].to_numpy(dtype=int)
                        if {"fold", "test_subject"}.issubset(existing_pred.columns):
                            for (fold_id, fold_subject), fold_pred in existing_pred.groupby(["fold", "test_subject"], dropna=False):
                                fold_true = fold_pred["y_true_id"].to_numpy(dtype=int)
                                fold_y_pred = fold_pred["y_pred_id"].to_numpy(dtype=int)
                                fold_metric = metrics_dict(fold_true, fold_y_pred)
                                fold_rows.append({
                                    "dataset": dataset,
                                    "feature_set": feature_set,
                                    "window_type": window_type,
                                    "window_size": int(manifest.get("window", 0)),
                                    "stride": int(manifest.get("step", 0)),
                                    "protocol": protocol,
                                    "model_family": "baselines",
                                    "model": model_name,
                                    "fold": fold_id,
                                    "test_subject": fold_subject,
                                    "seed": int(seed),
                                    "n_train": None,
                                    "n_test": int(len(fold_pred)),
                                    "fit_predict_sec": None,
                                    "label_encoding_note": "reconstructed_from_existing_predictions",
                                    **fold_metric,
                                })
                        agg = metrics_dict(y_true_arr, y_pred_arr)
                        summary_rows.append({
                            "dataset": dataset,
                            "feature_set": feature_set,
                            "window_type": window_type,
                            "window_size": int(manifest.get("window", 0)),
                            "stride": int(manifest.get("step", 0)),
                            "protocol": protocol,
                            "model_family": "baselines",
                            "model": model_name,
                            "fold": "aggregate",
                            "test_subject": "aggregate",
                            "seed": int(seed),
                            "n_samples": int(len(y_true_arr)),
                            "total_sec": None,
                            **agg,
                        })
                        print(f"[SKIP] {model_name}: using existing {pred_file}")
                        continue
            except Exception as exc:
                print(f"[WARN] Could not reuse existing predictions for {model_name}: {exc}")

        print(f"\n[{dataset}/{feature_set}/{window_type}/{protocol}] {model_name}")
        pred_rows: list[pd.DataFrame] = []
        y_true_all: list[int] = []
        y_pred_all: list[int] = []
        start_model = time.time()
        last_model = None

        for fold_number, (train_idx, test_idx, fold_subject, n_test) in enumerate(
            split_indices(protocol, subjects, y, seed, test_fraction),
            start=1,
        ):
            model = clone(base_model)
            t0 = time.time()
            label_encoding_note = ""
            try:
                pred, label_encoding_note = fit_predict_fold_model(
                    model,
                    model_name,
                    X_feat[train_idx],
                    y[train_idx],
                    X_feat[test_idx],
                )
                proba = predict_proba_global(model, X_feat[test_idx], len(labels))
            except Exception as exc:
                print(f"[ERROR] {model_name} fold={fold_number} subject={fold_subject} failed: {exc}")
                fold_rows.append({
                    "dataset": dataset,
                    "feature_set": feature_set,
                    "window_type": window_type,
                    "protocol": protocol,
                    "model_family": "baselines",
                    "model": model_name,
                    "fold": fold_number,
                    "test_subject": fold_subject,
                    "seed": int(seed),
                    "error": str(exc),
                })
                continue

            last_model = model
            elapsed = time.time() - t0
            m = metrics_dict(y[test_idx], pred)
            row = {
                "dataset": dataset,
                "feature_set": feature_set,
                "window_type": window_type,
                "window_size": int(manifest.get("window", 0)),
                "stride": int(manifest.get("step", 0)),
                "protocol": protocol,
                "model_family": "baselines",
                "model": model_name,
                "fold": fold_number,
                "test_subject": fold_subject,
                "seed": int(seed),
                "n_train": int(len(train_idx)),
                "n_test": int(n_test),
                "fit_predict_sec": float(elapsed),
                "label_encoding_note": label_encoding_note,
                **m,
            }
            fold_rows.append(row)
            print(
                f"fold={fold_number} subject={fold_subject}: "
                f"acc={m['accuracy']:.4f}, macro_f1={m['macro_f1']:.4f}, "
                f"bacc={m['balanced_accuracy']:.4f}, sec={elapsed:.1f}"
            )

            y_true_all.extend(y[test_idx].tolist())
            y_pred_all.extend(pred.tolist())
            fold_meta = meta.iloc[test_idx].copy().reset_index(drop=True)
            fold_meta["dataset"] = dataset
            fold_meta["feature_set"] = feature_set
            fold_meta["window_type"] = window_type
            fold_meta["protocol"] = protocol
            fold_meta["model_family"] = "baselines"
            fold_meta["model"] = model_name
            fold_meta["fold"] = fold_number
            fold_meta["test_subject"] = fold_subject
            fold_meta["seed"] = int(seed)
            fold_meta["y_true_id"] = y[test_idx]
            fold_meta["y_pred_id"] = pred
            fold_meta["y_true"] = le.inverse_transform(y[test_idx])
            fold_meta["y_pred"] = le.inverse_transform(pred.astype(int))
            fold_meta["y_true_label"] = [labels_display[int(i)] for i in y[test_idx]]
            fold_meta["y_pred_label"] = [labels_display[int(i)] for i in pred.astype(int)]
            fold_meta["correct"] = fold_meta["y_true_id"] == fold_meta["y_pred_id"]
            if proba is not None:
                for class_idx in range(proba.shape[1]):
                    fold_meta[f"proba_{class_idx}"] = proba[:, class_idx]
                fold_meta["confidence"] = proba.max(axis=1)
            pred_rows.append(fold_meta)

        if not y_true_all:
            continue

        y_true_arr = np.asarray(y_true_all)
        y_pred_arr = np.asarray(y_pred_all)
        agg = metrics_dict(y_true_arr, y_pred_arr)
        summary_rows.append({
            "dataset": dataset,
            "feature_set": feature_set,
            "window_type": window_type,
            "window_size": int(manifest.get("window", 0)),
            "stride": int(manifest.get("step", 0)),
            "protocol": protocol,
            "model_family": "baselines",
            "model": model_name,
            "fold": "aggregate",
            "test_subject": "aggregate",
            "seed": int(seed),
            "n_samples": int(len(y_true_arr)),
            "total_sec": float(time.time() - start_model),
            **agg,
        })

        pred_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
        pred_df.to_csv(out_dir / f"predictions_{model_name}.csv", index=False)

        report = classification_report(
            y_true_arr,
            y_pred_arr,
            labels=np.arange(len(labels)),
            target_names=labels_display,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).T.to_csv(out_dir / f"classification_report_{model_name}.csv")
        with open(out_dir / f"classification_report_{model_name}.json", "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2)

        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=np.arange(len(labels)))
        pd.DataFrame(cm, index=labels_display, columns=labels_display).to_csv(out_dir / f"confusion_matrix_{model_name}.csv")
        plot_confusion(cm, labels_display, f"{model_name} confusion matrix", out_dir / f"confusion_matrix_{model_name}.png")

        if last_model is not None:
            fi = extract_feature_importance(last_model, feature_names)
            if fi is not None:
                fi.to_csv(out_dir / f"feature_importance_{model_name}.csv", index=False)

        folds = pd.DataFrame(fold_rows)
        summary = pd.DataFrame(summary_rows)
        if not summary.empty:
            summary = summary.sort_values("macro_f1", ascending=False)
        folds.to_csv(out_dir / "metrics_by_fold.csv", index=False)
        summary.to_csv(out_dir / "metrics_summary.csv", index=False)
        plot_model_comparison(summary.rename(columns={"model": "model"}), out_dir / "model_comparison_macro_f1.png")

    folds = pd.DataFrame(fold_rows)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values("macro_f1", ascending=False)
    folds.to_csv(out_dir / "metrics_by_fold.csv", index=False)
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    plot_model_comparison(summary.rename(columns={"model": "model"}), out_dir / "model_comparison_macro_f1.png")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run canonical classical baselines")
    parser.add_argument("--dataset", choices=["pamap2"], default="pamap2")
    parser.add_argument("--processed-root", default="data/processed/canonical")
    parser.add_argument("--out-root", default="results/canonical/core_comparison")
    parser.add_argument("--feature-sets", default=",".join(DEFAULT_FEATURE_SETS))
    parser.add_argument("--window-types", default=",".join(DEFAULT_WINDOW_TYPES))
    parser.add_argument("--protocols", default=",".join(DEFAULT_PROTOCOLS))
    parser.add_argument("--models", default="all", help="Comma-separated baseline models or all")
    parser.add_argument("--include-xgb", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Skip slower baseline models where supported")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--max-windows-per-subject", type=int, default=0, help="Smoke/debug cap; 0 means no cap")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing per-model prediction files when present.")
    parser.add_argument("--require-probabilities", action="store_true", help="With --skip-existing, refit models whose existing prediction files lack proba_* columns.")
    parser.add_argument(
        "--allow-capped-canonical",
        action="store_true",
        help="Allow capped runs to write to --out-root instead of routing to results/canonical_smoke.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    out_root = Path(args.out_root)
    if args.max_windows_per_subject > 0 and not args.allow_capped_canonical:
        default_root = Path("results/canonical/core_comparison")
        if out_root == default_root:
            out_root = Path("results/canonical_smoke/core_comparison")
            print(
                "[INFO] max-windows-per-subject is set; routing capped smoke outputs to "
                f"{out_root}. Use --allow-capped-canonical to override."
            )
    for feature_set in parse_csv(args.feature_sets):
        for window_type in parse_csv(args.window_types):
            X_raw, y_raw, subjects, meta, manifest = load_canonical_dataset(
                Path(args.processed_root),
                args.dataset,
                feature_set,
                window_type,
            )
            X_raw, y_raw, subjects, meta = cap_windows_per_subject(
                X_raw,
                y_raw,
                subjects,
                meta,
                args.max_windows_per_subject,
                args.seed,
            )
            feature_columns = manifest.get("feature_columns")
            if not feature_columns:
                raise ValueError(f"Processed manifest for {feature_set}/{window_type} lacks feature_columns")
            X_feat, feature_names = extract_window_features(X_raw, feature_columns)

            for protocol in parse_csv(args.protocols):
                out_dir = (
                    out_root
                    / args.dataset
                    / feature_set
                    / window_type
                    / protocol
                    / "baselines"
                )
                run_protocol(
                    X_feat=X_feat,
                    y_raw=y_raw,
                    subjects=subjects,
                    meta=meta,
                    feature_names=feature_names,
                    dataset=args.dataset,
                    feature_set=feature_set,
                    window_type=window_type,
                    protocol=protocol,
                    out_dir=out_dir,
                    include_xgb=args.include_xgb,
                    use_cuda=args.use_cuda,
                    fast=args.fast,
                    models_arg=args.models,
                    seed=args.seed,
                    test_fraction=args.test_fraction,
                    max_windows_per_subject=args.max_windows_per_subject if args.max_windows_per_subject > 0 else None,
                    skip_existing=args.skip_existing,
                    require_probabilities=args.require_probabilities,
                    manifest=manifest,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
