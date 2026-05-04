#!/usr/bin/env python3
"""
Regenerate all result plots from saved metrics (no model retraining).

Fixes legacy naming (lstm_pamap2 vs lstm_only_pamap2) and baseline path
(HHAR_baselines.json). Run from repo root:

    .venv/bin/python scripts/regenerate_all_plots.py
    .venv/bin/python scripts/regenerate_all_plots.py --minimal   # fewer PNGs

To move non-essential PNGs out of results/plots/ after a full run:

    .venv/bin/python scripts/organize_result_plots.py --preset talk --move
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    METRICS_DIR,
    PLOTS_DIR,
    PAMAP2_ACTIVITIES,
    PROCESSED_DIR,
    SEED,
    WINDOW_SIZE,
    PAMAP2_NODE_FEAT_DIM,
)
from src.models import GNNLSTMModel, GNNOnlyModel, LSTMOnlyModel  # noqa: E402
from src.train import get_device  # noqa: E402

MET = Path(METRICS_DIR)
PLO = Path(PLOTS_DIR)
MET.mkdir(parents=True, exist_ok=True)
PLO.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.05)


def _load_json(name: str) -> dict:
    p = MET / name
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _pamap2_class_names() -> list[str]:
    y_raw = np.load(Path(PROCESSED_DIR) / "pamap2_y.npy")
    le = LabelEncoder()
    le.fit(y_raw)
    return [PAMAP2_ACTIVITIES[int(k)] for k in sorted(le.classes_)]


def _hhar_class_names() -> list[str]:
    y_raw = np.load(Path(PROCESSED_DIR) / "hhar_y.npy")
    le = LabelEncoder()
    le.fit(y_raw)
    return [str(c) for c in le.classes_]


def _first_existing(*candidates: Path) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def plot_confusion(tag: str, class_names: list[str], title_suffix: str = "") -> bool:
    yt = _first_existing(MET / f"{tag}_y_true.npy", MET / f"{tag}_full_y_true.npy")
    yp = _first_existing(MET / f"{tag}_y_pred.npy", MET / f"{tag}_full_y_pred.npy")
    if yt is None or yp is None:
        print(f"  [skip CM] missing preds: {tag}")
        return False
    y_true = np.load(yt)
    y_pred = np.load(yp)
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    name_list = [class_names[i] if i < len(class_names) else str(i) for i in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    sz = max(8, len(labels))
    fig, ax = plt.subplots(figsize=(sz, sz - 1))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=name_list,
        yticklabels=name_list,
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title(f"Confusion — {tag}{title_suffix}", fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = PLO / f"cm_{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")
    return True


def plot_model_comparison_pamap2() -> None:
    baselines = _load_json("pamap2_baselines.json")
    deep = _load_json("pamap2_deep_models.json")
    cnn = _load_json("cnn1d_results.json")

    rows: list[tuple[str, float, float, float, float]] = []
    # (display, acc, acc_std, f1, f1_std)
    for key, disp in [
        ("SVM", "SVM"),
        ("RandomForest", "Random Forest"),
        ("XGBoost", "XGBoost"),
    ]:
        if key in baselines:
            b = baselines[key]
            rows.append(
                (
                    disp,
                    float(b.get("mean_accuracy", 0)),
                    float(b.get("std_accuracy", 0)),
                    float(b.get("mean_macro_f1", 0)),
                    float(b.get("std_macro_f1", 0)),
                )
            )
    key_map = [("lstm", "LSTM-only"), ("gnn", "GNN-only"), ("gnn_lstm", "GNN+LSTM")]
    for key, disp in key_map:
        if key in deep:
            d = deep[key]
            rows.append(
                (
                    disp,
                    float(d.get("accuracy", 0)),
                    float(d.get("accuracy_std", 0)),
                    float(d.get("macro_f1", 0)),
                    float(d.get("macro_f1_std", 0)),
                )
            )
    if cnn.get("pamap2"):
        p = cnn["pamap2"]
        rows.append(
            (
                "CNN1D",
                float(p.get("accuracy", 0)),
                0.0,
                float(p.get("macro_f1", 0)),
                0.0,
            )
        )

    if not rows:
        print("  [skip] model_comparison_pamap2 — no data")
        return

    names, accs, acc_e, f1s, f1_e = zip(*rows)
    x = np.arange(len(names))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, vals, errs, ylab, title in zip(
        axes,
        [accs, f1s],
        [acc_e, f1_e],
        ["LOSO accuracy", "Macro F1"],
        ["PAMAP2 — accuracy", "PAMAP2 — macro F1"],
    ):
        ax.bar(x, vals, yerr=errs, capsize=4, color="steelblue", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.suptitle("Model comparison — PAMAP2 (LOSO)", fontweight="bold")
    plt.tight_layout()
    out = PLO / "model_comparison_pamap2.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_model_comparison_hhar() -> None:
    p_hh = MET / "HHAR_baselines.json"
    if not p_hh.exists():
        p_hh = MET / "hhar_baselines.json"
    baselines = json.loads(p_hh.read_text(encoding="utf-8")) if p_hh.exists() else {}
    deep = _load_json("hhar_deep_models.json")
    cnn = _load_json("cnn1d_results.json")

    rows: list[tuple[str, float, float, float, float]] = []
    for key, disp in [
        ("SVM", "SVM"),
        ("RandomForest", "Random Forest"),
        ("XGBoost", "XGBoost"),
    ]:
        if key in baselines:
            b = baselines[key]
            rows.append(
                (
                    disp,
                    float(b.get("mean_accuracy", 0)),
                    float(b.get("std_accuracy", 0)),
                    float(b.get("mean_macro_f1", 0)),
                    float(b.get("std_macro_f1", 0)),
                )
            )
    for key, disp in [("lstm", "LSTM-only"), ("gnn", "GNN-only"), ("gnn_lstm", "GNN+LSTM")]:
        if key in deep:
            d = deep[key]
            rows.append(
                (
                    disp,
                    float(d.get("accuracy", 0)),
                    float(d.get("accuracy_std", 0)),
                    float(d.get("macro_f1", 0)),
                    float(d.get("macro_f1_std", 0)),
                )
            )
    if cnn.get("hhar"):
        p = cnn["hhar"]
        rows.append(
            (
                "CNN1D",
                float(p.get("accuracy", 0)),
                0.0,
                float(p.get("macro_f1", 0)),
                0.0,
            )
        )

    if not rows:
        print("  [skip] model_comparison_hhar — no data")
        return

    names, accs, acc_e, f1s, f1_e = zip(*rows)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, vals, errs, ylab, title in zip(
        axes,
        [accs, f1s],
        [acc_e, f1_e],
        ["LOSO accuracy", "Macro F1"],
        ["HHAR — accuracy", "HHAR — macro F1"],
    ):
        ax.bar(x, vals, yerr=errs, capsize=4, color="darkorange", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
    fig.suptitle("Model comparison — HHAR (LOSO)", fontweight="bold")
    plt.tight_layout()
    out = PLO / "model_comparison_hhar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_cross_dataset_loso() -> None:
    """Side-by-side PAMAP2 vs HHAR LOSO (same as midway report)."""
    baselines_p = _load_json("pamap2_baselines.json")
    p_hh = MET / "HHAR_baselines.json"
    if not p_hh.exists():
        p_hh = MET / "hhar_baselines.json"
    baselines_h = json.loads(p_hh.read_text(encoding="utf-8")) if p_hh.exists() else {}
    deep_p = _load_json("pamap2_deep_models.json")
    deep_h = _load_json("hhar_deep_models.json")
    if not baselines_p or not baselines_h or not deep_p or not deep_h:
        print("  [skip] cross_dataset_comparison — incomplete JSON")
        return

    labels = ["SVM", "Random Forest", "XGBoost", "LSTM-only", "GNN-only", "GNN+LSTM"]
    keys_b = ["SVM", "RandomForest", "XGBoost"]
    keys_d = ["lstm", "gnn", "gnn_lstm"]

    def acc_b(b, k):
        return float(b.get(k, {}).get("mean_accuracy", 0))

    def acc_d(d, k):
        return float(d.get(k, {}).get("accuracy", 0))

    p_accs = [acc_b(baselines_p, k) for k in keys_b] + [acc_d(deep_p, k) for k in keys_d]
    h_accs = [acc_b(baselines_h, k) for k in keys_b] + [acc_d(deep_h, k) for k in keys_d]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, p_accs, w, label="PAMAP2", color="steelblue", alpha=0.9)
    ax.bar(x + w / 2, h_accs, w, label="HHAR", color="darkorange", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("LOSO accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cross-dataset comparison (LOSO accuracy)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    out = PLO / "cross_dataset_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_graph_ablation() -> None:
    data = _load_json("graph_ablation_results.json")
    if not data:
        print("  [skip] graph_ablation.png — no graph_ablation_results.json")
        return
    names = ["Fixed adj\n(GNN)", "Learnable adj", "Flatten+LSTM"]
    keys = ["ablation_fixed_adj", "ablation_learnable_adj", "ablation_flatten_lstm"]
    accs = [data[k]["accuracy"] for k in keys if k in data]
    f1s = [data[k]["macro_f1"] for k in keys if k in data]
    if len(accs) != 3:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, vals, t in zip(axes, [accs, f1s], ["Accuracy", "Macro F1"]):
        ax.bar(names, vals, color=["steelblue", "darkorange", "crimson"], alpha=0.85)
        ax.set_ylim(0, 1.05)
        ax.set_title(t)
    fig.suptitle("Graph ablation — PAMAP2 LOSO", fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLO / "graph_ablation.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {PLO / 'graph_ablation.png'}")


def plot_per_class_f1_from_error_analysis() -> None:
    err = _load_json("error_analysis.json")
    if not err:
        print("  [skip] per_class_f1_* — no error_analysis.json")
        return
    for ds in ("pamap2", "hhar"):
        ds_e = err.get(ds, {})
        if not ds_e or not all(m in ds_e for m in ("lstm", "gnn", "gnnlstm")):
            continue
        # align class names from gnn keys
        classes = list(ds_e["gnn"]["per_class_f1"].keys())
        lstm_f1 = [ds_e["lstm"]["per_class_f1"].get(c, 0) for c in classes]
        gnn_f1 = [ds_e["gnn"]["per_class_f1"].get(c, 0) for c in classes]
        gl_f1 = [ds_e["gnnlstm"]["per_class_f1"].get(c, 0) for c in classes]
        x = np.arange(len(classes))
        w = 0.25
        fig, ax = plt.subplots(figsize=(max(10, len(classes)), 5))
        ax.bar(x - w, lstm_f1, w, label="LSTM-only", color="steelblue")
        ax.bar(x, gnn_f1, w, label="GNN-only", color="darkorange")
        ax.bar(x + w, gl_f1, w, label="GNN+LSTM", color="green")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Per-class F1")
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.set_title(f"Per-class F1 — {ds.upper()}")
        plt.tight_layout()
        out = PLO / f"per_class_f1_{ds}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved {out}")


def plot_shap_rf() -> None:
    try:
        import shap
        from sklearn.ensemble import RandomForestClassifier

        from src.baselines import extract_features
    except ImportError as e:
        print(f"  [skip] shap_rf_pamap2.png — {e}")
        return
    X_p = np.load(Path(PROCESSED_DIR) / "pamap2_X.npy")
    y_raw = np.load(Path(PROCESSED_DIR) / "pamap2_y.npy")
    le = LabelEncoder()
    y_p = le.fit_transform(y_raw)
    X_feat = extract_features(X_p)
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(X_feat, y_p)
    n_ch = X_p.shape[2]
    feat_stats = ["mean", "std", "min", "max", "rms", "fft_energy"]
    feat_names = [f"{stat}_ch{ch}" for stat in feat_stats for ch in range(n_ch)]
    explainer = shap.TreeExplainer(rf)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_feat), size=min(500, len(X_feat)), replace=False)
    sv = explainer.shap_values(X_feat[idx])
    if isinstance(sv, list):
        mean_shap = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
    else:
        mean_shap = np.abs(sv).mean(axis=0)
    mean_shap = np.asarray(mean_shap).ravel()
    n_feat = min(len(feat_names), len(mean_shap))
    mean_shap = mean_shap[:n_feat]
    feat_names = feat_names[:n_feat]
    top_k = min(20, n_feat)
    top_idx = np.argsort(mean_shap)[::-1][:top_k]
    order = [int(i) for i in top_idx[::-1]]
    fig, ax = plt.subplots(figsize=(10, 6))
    labels_ = [feat_names[i] for i in order]
    vals_ = mean_shap[order]
    ax.barh(labels_, vals_, color="steelblue")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title("SHAP — Random Forest, PAMAP2 (top-20)")
    plt.tight_layout()
    fig.savefig(PLO / "shap_rf_pamap2.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {PLO / 'shap_rf_pamap2.png'}")


def plot_model_profiling() -> None:
    import torch

    device = get_device()
    act_p = _pamap2_class_names()
    n_cls = len(act_p)
    configs = [
        ("LSTM-only", LSTMOnlyModel(WINDOW_SIZE * 18, n_cls), False),
        ("GNN-only", GNNOnlyModel(PAMAP2_NODE_FEAT_DIM, 3, n_cls), True),
        ("GNN+LSTM", GNNLSTMModel(PAMAP2_NODE_FEAT_DIM, 3, n_cls), True),
    ]
    rows = []
    n_reps = 80
    for name, model, use_adj in configs:
        model = model.to(device).eval()
        n_params = sum(p.numel() for p in model.parameters())
        if use_adj:
            if name == "GNN+LSTM":
                dx = torch.randn(1, 10, 3, PAMAP2_NODE_FEAT_DIM).to(device)
            else:
                dx = torch.randn(1, 3, PAMAP2_NODE_FEAT_DIM).to(device)
            adj = torch.eye(3).to(device)
            with torch.no_grad():
                for _ in range(10):
                    model(dx, adj)
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_reps):
                    model(dx, adj)
        else:
            dx = torch.randn(1, WINDOW_SIZE * 18).to(device)
            with torch.no_grad():
                for _ in range(10):
                    model(dx)
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_reps):
                    model(dx)
        ms = (time.perf_counter() - t0) / n_reps * 1000
        rows.append({"Model": name, "Params": n_params, "Latency_ms": ms})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names_ = [r["Model"] for r in rows]
    axes[0].bar(names_, [r["Params"] / 1e6 for r in rows], color=["steelblue", "darkorange", "green"])
    axes[0].set_ylabel("Parameters (M)")
    axes[0].set_title("Model size")
    axes[1].bar(names_, [r["Latency_ms"] for r in rows], color=["steelblue", "darkorange", "green"])
    axes[1].set_ylabel("Latency (ms / forward)")
    axes[1].set_title("Inference (device=%s)" % device)
    plt.tight_layout()
    fig.savefig(PLO / "model_profiling.png", dpi=150)
    plt.close(fig)
    (MET / "model_profiling.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"  Saved {PLO / 'model_profiling.png'}")


def _augment_metric_keys(data: dict) -> list[str]:
    return [k for k in data if isinstance(data.get(k), dict) and "accuracy" in data[k]]


def _plot_augmentation_from_json(json_name: str, png_name: str, suptitle: str) -> None:
    data = _load_json(json_name)
    if not data:
        print(f"  [skip] {png_name}")
        return
    labels = _augment_metric_keys(data)
    if not labels:
        print(f"  [skip] {png_name} — no metric rows")
        return
    accs = [data[k]["accuracy"] for k in labels]
    f1s = [data[k]["macro_f1"] for k in labels]
    disp = ["No Aug", "Gaussian", "Scale", "Time warp"][: len(labels)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, vals, t in zip(axes, [accs, f1s], ["Accuracy", "Macro F1"]):
        ax.bar(disp[: len(vals)], vals, color="steelblue", alpha=0.85)
        ax.set_ylim(0, 1.05)
        ax.set_title(t)
    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    fig.savefig(PLO / png_name, dpi=150)
    plt.close(fig)
    print(f"  Saved {PLO / png_name}")


def plot_augmentation() -> None:
    _plot_augmentation_from_json(
        "augmentation_results.json",
        "augmentation_comparison.png",
        "Augmentation — HHAR GNN-only LOSO",
    )
    _plot_augmentation_from_json(
        "augmentation_results_demo.json",
        "augmentation_comparison_demo.png",
        "Augmentation — HHAR GNN-only LOSO (demo cap)",
    )


def plot_master_comparison() -> None:
    master = _load_json("master_comparison.json")
    if not master:
        print("  [skip] complete_comparison_*.png — no master_comparison.json")
        return

    def acc_pct(v) -> float:
        if isinstance(v, dict):
            return float(v.get("accuracy", 0)) * 100
        return float(v) * 100

    pairs = [("PAMAP2", "pamap2"), ("HHAR", "hhar")]
    for upper, lower in pairs:
        section = master.get(upper) or master.get(lower)
        if not isinstance(section, dict):
            continue
        names = list(section.keys())
        vals = [acc_pct(section[k]) for k in names]
        fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.45), 5))
        ax.bar(names, vals, color="steelblue", alpha=0.85)
        ax.set_ylabel("LOSO accuracy (%)")
        ax.set_ylim(0, 105)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"Master comparison — {upper}")
        plt.tight_layout()
        fig.savefig(PLO / f"complete_comparison_{lower}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {PLO / f'complete_comparison_{lower}.png'}")


def plot_optional_json_plots() -> None:
    """Recreate plots that experiments.py normally writes, from saved JSON only."""
    # XGBoost feature importance
    p = MET / "xgb_feature_importance.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        names = d.get("top_features", [])
        vals = np.array(d.get("top_importances", []), dtype=float)
        if len(names) and len(vals):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(names[::-1], vals[::-1], color="crimson", alpha=0.85)
            ax.set_xlabel("XGBoost importance")
            ax.set_title("XGBoost feature importance — PAMAP2 (top-20)")
            plt.tight_layout()
            fig.savefig(PLO / "xgb_feature_importance_pamap2.png", dpi=150)
            plt.close(fig)
            print(f"  Saved {PLO / 'xgb_feature_importance_pamap2.png'}")

    # Mobile optimisation
    p = MET / "mobile_optimisation_results.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        std = d.get("standard_gnn", {})
        tiny = d.get("tiny_gnn", {})
        if std and tiny:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            labels = ["Standard GNN", "Tiny GNN"]
            axes[0].bar(labels, [std.get("accuracy", 0) * 100, tiny.get("accuracy", 0) * 100])
            axes[0].set_title("Accuracy %")
            axes[1].bar(labels, [std.get("params", 0) / 1e3, tiny.get("params", 0) / 1e3])
            axes[1].set_title("Params (K)")
            axes[2].bar(
                labels,
                [std.get("latency_ms", 0), tiny.get("latency_f32_ms", tiny.get("latency_ms", 0))],
            )
            axes[2].set_title("Latency (ms)")
            fig.suptitle("Mobile optimisation — PAMAP2")
            plt.tight_layout()
            fig.savefig(PLO / "mobile_optimisation.png", dpi=150)
            plt.close(fig)
            print(f"  Saved {PLO / 'mobile_optimisation.png'}")

    # Integrated gradients (from neural_interpretability.json)
    p = MET / "neural_interpretability.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        names = d.get("top_features", [])
        vals = d.get("top_attr_values", [])
        if names and vals:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(names[::-1], vals[::-1], color="darkorange", alpha=0.85)
            ax.set_xlabel("Mean |Integrated Gradient|")
            ax.set_title("Integrated gradients — GNN PAMAP2 (top-20)")
            plt.tight_layout()
            fig.savefig(PLO / "ig_gnn_pamap2.png", dpi=150)
            plt.close(fig)
            print(f"  Saved {PLO / 'ig_gnn_pamap2.png'}")

    # Cross-device HHAR
    p = MET / "cross_device_results.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        loso = d.get("loso_reference", {}).get("accuracy") or 0
        a = d.get("train_AB_test_CD", {}).get("accuracy", 0)
        b = d.get("train_CD_test_AB", {}).get("accuracy", 0)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(["LOSO ref", "A→B", "B→A"], [loso, a, b], color=["steelblue", "darkorange", "crimson"])
        ax.set_ylim(0, 1.05)
        ax.set_title("HHAR GNN-only: LOSO vs cross-device")
        plt.tight_layout()
        fig.savefig(PLO / "cross_device_comparison.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {PLO / 'cross_device_comparison.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate result plots from metrics JSON/npy.")
    ap.add_argument(
        "--minimal",
        action="store_true",
        help="Skip ablations, per-class F1, augmentation, SHAP, profiling, master charts, "
        "optional XGB/mobile, cross-dataset RF transfer, and LSTM/GNN confusion matrices.",
    )
    args = ap.parse_args()
    minimal = args.minimal

    print("Regenerating plots from", MET, flush=True)
    if minimal:
        print("(minimal mode: fewer figures)", flush=True)
    pam_names = _pamap2_class_names()
    hhar_names = _hhar_class_names()

    print("\n── Confusion matrices (LOSO preds) ──")
    if minimal:
        plot_confusion("gnnlstm_pamap2", pam_names)
        if not plot_confusion("cnn1d_pamap2", pam_names):
            plot_confusion("cnn1d_pamap2_full", pam_names, " (full eval)")
        plot_confusion("gnnlstm_hhar", hhar_names)
        if not plot_confusion("cnn1d_hhar", hhar_names):
            plot_confusion("cnn1d_hhar_full", hhar_names, " (full eval)")
    else:
        for tag in ["lstm_pamap2", "gnn_pamap2", "gnnlstm_pamap2"]:
            plot_confusion(tag, pam_names)
        if not plot_confusion("cnn1d_pamap2", pam_names):
            plot_confusion("cnn1d_pamap2_full", pam_names, " (full eval)")
        for tag in ["lstm_hhar", "gnn_hhar", "gnnlstm_hhar"]:
            plot_confusion(tag, hhar_names)
        if not plot_confusion("cnn1d_hhar", hhar_names):
            plot_confusion("cnn1d_hhar_full", hhar_names, " (full eval)")

    print("\n── Model comparison & cross-dataset LOSO ──")
    plot_model_comparison_pamap2()
    plot_model_comparison_hhar()
    plot_cross_dataset_loso()

    if not minimal:
        print("\n── Ablations & error analysis plots ──")
        plot_graph_ablation()
        plot_per_class_f1_from_error_analysis()
        plot_augmentation()

        print("\n── SHAP & profiling ──")
        plot_shap_rf()
        plot_model_profiling()

        print("\n── Master comparison charts ──")
        plot_master_comparison()

        print("\n── Optional plots from experiment JSON ──")
        plot_optional_json_plots()

        print("\n── Cross-dataset transfer (RF) ──", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "cross_dataset_transfer.py")],
            cwd=str(ROOT),
            check=False,
        )

    lime_json = MET / "lime_lstm_pamap2.json"
    if lime_json.exists():
        print("\n── LIME (existing metrics; re-run run_lime_pamap2_lstm.py to refresh) ──")
    else:
        print("\n── LIME — run: .venv/bin/python scripts/run_lime_pamap2_lstm.py ──")

    print("\n── Presentation summary (all models, ± std) ──", flush=True)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_presentation_summary_figure.py")],
        cwd=str(ROOT),
        check=False,
    )

    print("\n✅ Plot regeneration finished →", PLO)


if __name__ == "__main__":
    main()
