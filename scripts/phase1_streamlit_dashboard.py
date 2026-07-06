"""Streamlit dashboard for canonical HAR experiment results.

Run from repo root:
  streamlit run scripts/phase1_streamlit_dashboard.py

The dashboard reads canonical outputs such as:
  results/canonical_protocol_only/core_comparison/<dataset>/<feature_set>/<window>/<protocol>/baselines/
  results/canonical_protocol_only/core_comparison/<dataset>/<feature_set>/<window>/<protocol>/deep/<dataset>/<model>/<eval_unit>/

It also tolerates future canonical/all-session folders under results/canonical.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="HAR Results Dashboard",
    page_icon="HAR",
    layout="wide",
    initial_sidebar_state="expanded",
)


METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_precision",
    "macro_recall",
    "subject_macro_accuracy_mean",
    "subject_macro_f1_mean",
]

BASELINE_MODELS = {
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
}

DEEP_MODELS = {
    "cnn",
    "lstm",
    "gnn",
    "gnn_learnable_adj",
    "gnn_attention_adj",
    "gnn_lstm",
    "gnn_flatten_lstm",
    "improved_gnn_lstm",
    "improved_gnn_lstm_res",
    "improved_gnn_lstm_attn_adj",
    "improved_gnn_lstm_attn_adj_resbn",
}


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    result_set: str
    dataset: str
    feature_set: str
    window_type: str
    protocol: str
    family: str
    model: str
    eval_unit: str


def read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists() and path.stat().st_size:
            return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {path}: {exc}")
    return pd.DataFrame()


def read_json(path: Path) -> dict[str, Any]:
    try:
        if path.exists() and path.stat().st_size:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def first_present(row: pd.Series, names: Iterable[str], default: str = "") -> str:
    for name in names:
        if name in row and pd.notna(row[name]) and str(row[name]) != "":
            return str(row[name])
    return default


def path_context(metrics_file: Path, results_root: Path) -> dict[str, str]:
    """Infer canonical context from a metrics file path."""
    parts = list(metrics_file.resolve().parts)
    context = {
        "result_set": "",
        "dataset": "",
        "feature_set": "",
        "window_type": "",
        "protocol": "",
        "family": "",
        "model": "",
        "eval_unit": "",
    }

    if "results" in parts:
        idx = parts.index("results")
        if idx + 1 < len(parts):
            context["result_set"] = parts[idx + 1]

    # canonical: .../<dataset>/<feature>/<window>/<protocol>/baselines/metrics_summary.csv
    # canonical: .../<dataset>/<feature>/<window>/<protocol>/deep/<dataset>/<model>/<eval>/metrics_summary.csv
    try:
        core_idx = parts.index("core_comparison")
        context["dataset"] = parts[core_idx + 1]
        context["feature_set"] = parts[core_idx + 2]
        context["window_type"] = parts[core_idx + 3]
        context["protocol"] = parts[core_idx + 4]
        marker = parts[core_idx + 5]
        if marker == "baselines":
            context["family"] = "baseline"
        elif marker == "deep":
            context["family"] = "deep"
            context["model"] = parts[core_idx + 7] if core_idx + 7 < len(parts) else ""
            context["eval_unit"] = parts[core_idx + 8] if core_idx + 8 < len(parts) else ""
    except Exception:
        pass

    if not context["result_set"]:
        try:
            context["result_set"] = metrics_file.resolve().relative_to(results_root.resolve()).parts[0]
        except Exception:
            context["result_set"] = results_root.name
    return context


def normalize_summary_file(path: Path, results_root: Path) -> pd.DataFrame:
    df = read_csv(path)
    if df.empty:
        return df

    ctx = path_context(path, results_root)
    manifest = read_json(path.parent / "dataset_manifest.json")
    run_manifest = read_json(path.parent / "run_manifest.json")
    job = run_manifest.get("job") if isinstance(run_manifest.get("job"), dict) else {}

    for key, value in ctx.items():
        if key not in df.columns:
            df[key] = value
        else:
            df[key] = df[key].fillna("").astype(str)
            df.loc[df[key] == "", key] = value

    if "protocol" not in df.columns and "eval_protocol" in df.columns:
        df["protocol"] = df["eval_protocol"]
    if "eval_protocol" not in df.columns:
        df["eval_protocol"] = df.get("protocol", ctx["protocol"])

    if "model_family" in df.columns:
        df["family"] = df["model_family"].replace({"baselines": "baseline", "deep": "deep"})
    if "family" not in df.columns or df["family"].eq("").all():
        df["family"] = ctx["family"]

    if "model" not in df.columns:
        df["model"] = ctx["model"]
    df["model"] = df["model"].fillna("").astype(str)
    df.loc[df["model"] == "", "model"] = ctx["model"]
    df.loc[df["model"].isin(BASELINE_MODELS), "family"] = "baseline"
    df.loc[df["model"].isin(DEEP_MODELS), "family"] = "deep"

    for key in ["dataset", "feature_set", "window_type", "task", "sessions", "variant_name"]:
        value = manifest.get(key) or job.get(key) or ctx.get(key, "")
        if key not in df.columns:
            df[key] = value
        else:
            df[key] = df[key].fillna("").astype(str)
            df.loc[df[key] == "", key] = value

    if "eval_unit" not in df.columns:
        df["eval_unit"] = ctx["eval_unit"] or ("window" if ctx["family"] == "baseline" else "")
    df["artifact_dir"] = str(path.parent)
    df["summary_file"] = str(path)
    for metric in METRIC_COLUMNS + ["total_params", "n_samples", "n_eval_samples", "n_folds"]:
        if metric in df.columns:
            df[metric] = as_float(df[metric])
    return df


def normalize_fold_file(path: Path, results_root: Path) -> pd.DataFrame:
    df = read_csv(path)
    if df.empty:
        return df

    ctx = path_context(path, results_root)
    manifest = read_json(path.parent / "dataset_manifest.json")
    for key, value in ctx.items():
        if key not in df.columns:
            df[key] = value
        else:
            df[key] = df[key].fillna("").astype(str)
            df.loc[df[key] == "", key] = value

    if "protocol" not in df.columns and "eval_protocol" in df.columns:
        df["protocol"] = df["eval_protocol"]
    if "eval_protocol" not in df.columns:
        df["eval_protocol"] = df.get("protocol", ctx["protocol"])

    if "model_family" in df.columns:
        df["family"] = df["model_family"].replace({"baselines": "baseline", "deep": "deep"})
    if "family" not in df.columns or df["family"].eq("").all():
        df["family"] = ctx["family"]
    if "model" not in df.columns:
        df["model"] = ctx["model"]
    df["model"] = df["model"].fillna("").astype(str)
    df.loc[df["model"] == "", "model"] = ctx["model"]
    df.loc[df["model"].isin(BASELINE_MODELS), "family"] = "baseline"
    df.loc[df["model"].isin(DEEP_MODELS), "family"] = "deep"

    for key in ["dataset", "feature_set", "window_type", "task", "sessions", "variant_name"]:
        value = manifest.get(key) or ctx.get(key, "")
        if key not in df.columns:
            df[key] = value
        else:
            df[key] = df[key].fillna("").astype(str)
            df.loc[df[key] == "", key] = value

    if "test_subject" not in df.columns and "fold_subject" in df.columns:
        df["test_subject"] = df["fold_subject"]
    if "fold_subject" not in df.columns and "test_subject" in df.columns:
        df["fold_subject"] = df["test_subject"]

    df["artifact_dir"] = str(path.parent)
    df["fold_file"] = str(path)
    for metric in METRIC_COLUMNS + ["fold", "total_params", "n_train", "n_test", "n_train_windows", "n_test_windows"]:
        if metric in df.columns:
            df[metric] = as_float(df[metric])
    return df


@st.cache_data(show_spinner=True)
def load_results(results_root_str: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_root = Path(results_root_str)
    summary_parts: list[pd.DataFrame] = []
    fold_parts: list[pd.DataFrame] = []
    if not results_root.exists():
        return pd.DataFrame(), pd.DataFrame()

    for path in sorted(results_root.rglob("metrics_summary.csv")):
        part = normalize_summary_file(path, results_root)
        if not part.empty:
            summary_parts.append(part)

    for path in sorted(results_root.rglob("metrics_by_fold.csv")):
        part = normalize_fold_file(path, results_root)
        if not part.empty:
            fold_parts.append(part)

    summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    folds = pd.concat(fold_parts, ignore_index=True) if fold_parts else pd.DataFrame()
    return summary, folds


def filter_df(df: pd.DataFrame, filters: dict[str, list[str]]) -> pd.DataFrame:
    out = df.copy()
    for col, values in filters.items():
        if values and col in out.columns:
            out = out[out[col].astype(str).isin(values)]
    return out


def normalize_variant_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "variant_name" not in out.columns:
        out["variant_name"] = ""
    if "result_set" not in out.columns:
        out["result_set"] = ""
    variant = out["variant_name"].fillna("").astype(str)
    result_set = out["result_set"].fillna("").astype(str)
    missing = variant.str.strip().isin(["", "nan", "None"])
    out.loc[missing & result_set.eq("canonical_protocol_only"), "variant_name"] = "v1_original"
    out.loc[missing & result_set.eq("canonical_protocol_only_v2"), "variant_name"] = "v2_norm_balanced_ls005"
    out.loc[missing & result_set.eq("canonical_protocol_only_v3"), "variant_name"] = "v3_residual_arch"
    out["variant_name"] = out["variant_name"].fillna("").astype(str)
    return out


def sorted_unique(df: pd.DataFrame, col: str) -> list[str]:
    if col not in df.columns or df.empty:
        return []
    vals = [str(v) for v in df[col].dropna().unique().tolist() if str(v) != ""]
    return sorted(vals)


def metric_options(df: pd.DataFrame) -> list[str]:
    return [c for c in METRIC_COLUMNS if c in df.columns] or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def format_metrics(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    fmt: dict[str, str] = {}
    for col in df.columns:
        low = col.lower()
        if any(k in low for k in ["accuracy", "f1", "precision", "recall"]):
            fmt[col] = "{:.4f}"
        elif any(k in low for k in ["sec", "mb"]):
            fmt[col] = "{:.2f}"
        elif col in {"total_params", "trainable_params", "n_samples", "n_eval_samples"}:
            fmt[col] = "{:,.0f}"
    return df.style.format(fmt, na_rep="")


def bar_chart(df: pd.DataFrame, x: str, y: str, color: str | None = None, title: str = "") -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        st.info("No data available for this chart.")
        return
    plot_df = df.dropna(subset=[y]).copy()
    if plot_df.empty:
        st.info("No numeric metric values available.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(plot_df[x].unique()) * 0.8), 4.8))
    if color and color in plot_df.columns:
        groups = list(plot_df.groupby(color, sort=False))
        labels = list(dict.fromkeys(plot_df[x].astype(str).tolist()))
        width = 0.8 / max(len(groups), 1)
        base = np.arange(len(labels))
        for i, (name, part) in enumerate(groups):
            values = part.set_index(part[x].astype(str))[y].reindex(labels)
            ax.bar(base + (i - (len(groups) - 1) / 2) * width, values, width=width, label=str(name))
        ax.set_xticks(base)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend(loc="best", fontsize=8)
    else:
        ax.bar(plot_df[x].astype(str), plot_df[y].astype(float))
        ax.tick_params(axis="x", rotation=45)
    ax.set_title(title or y)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(axis="y", alpha=0.25)
    st.pyplot(fig, clear_figure=True)


def line_chart(df: pd.DataFrame, x: str, y: str, group: str | None = None, title: str = "") -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        st.info("No data available for this chart.")
        return
    plot_df = df.dropna(subset=[x, y]).copy()
    if plot_df.empty:
        st.info("No numeric metric values available.")
        return

    fig, ax = plt.subplots(figsize=(10, 4.8))
    if group and group in plot_df.columns:
        for name, part in plot_df.groupby(group):
            part = part.sort_values(x)
            ax.plot(part[x].astype(str), part[y].astype(float), marker="o", label=str(name))
        ax.legend(loc="best", fontsize=8)
    else:
        plot_df = plot_df.sort_values(x)
        ax.plot(plot_df[x].astype(str), plot_df[y].astype(float), marker="o")
    ax.tick_params(axis="x", rotation=45)
    ax.set_title(title or y)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(alpha=0.25)
    st.pyplot(fig, clear_figure=True)


def render_matrix_csv(path: Path, normalize: bool) -> None:
    df = read_csv(path)
    if df.empty:
        st.info("No confusion matrix CSV found.")
        return
    if df.columns[0].lower().startswith("unnamed") or df.columns[0].lower() in {"label", "class"}:
        df = df.set_index(df.columns[0])
    values = df.apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)
    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)

    size = max(6, min(16, 0.55 * max(values.shape)))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(values, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    max_value = float(np.nanmax(values)) if values.size else 0.0
    threshold = max_value * 0.5
    text_size = 8 if max(values.shape) <= 12 else 6
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = float(values[i, j])
            label = f"{value:.2f}" if normalize else f"{int(round(value))}"
            color = "white" if value > threshold else "black"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=text_size)
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels([str(c) for c in df.columns], rotation=90, fontsize=8)
    ax.set_yticklabels([str(i) for i in df.index], fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig, clear_figure=True)


def artifact_for(row: pd.Series, kind: str) -> Path | None:
    artifact_dir = Path(str(row.get("artifact_dir", "")))
    model = str(row.get("model", ""))
    family = str(row.get("family", ""))
    if not artifact_dir.exists():
        return None
    if kind == "confusion_csv":
        candidates = [artifact_dir / "confusion_matrix.csv"]
        if family == "baseline":
            candidates.insert(0, artifact_dir / f"confusion_matrix_{model}.csv")
    elif kind == "confusion_png":
        candidates = [artifact_dir / "confusion_matrix.png"]
        if family == "baseline":
            candidates.insert(0, artifact_dir / f"confusion_matrix_{model}.png")
    elif kind == "predictions":
        candidates = [artifact_dir / "predictions.csv"]
        if family == "baseline":
            candidates.insert(0, artifact_dir / f"predictions_{model}.csv")
    elif kind == "classification_report":
        candidates = [artifact_dir / "classification_report.csv"]
        if family == "baseline":
            candidates.insert(0, artifact_dir / f"classification_report_{model}.csv")
    else:
        candidates = []
    for path in candidates:
        if path.exists():
            return path
    return None


def best_by_group(df: pd.DataFrame, metric: str, group_cols: list[str]) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    ranked = df.dropna(subset=[metric]).sort_values(metric, ascending=False)
    return ranked.groupby(group_cols, as_index=False, dropna=False).head(1)


st.title("HAR Results Dashboard")
st.caption("Canonical comparison of baselines, deep models, LOSO folds, and random-holdout leakage gaps.")

repo_root = Path.cwd()
default_results = repo_root / "results"
results_root_text = st.sidebar.text_input("Results root", value=str(default_results))
results_root = Path(results_root_text).expanduser().resolve()

if st.sidebar.button("Refresh"):
    st.cache_data.clear()
    st.rerun()

summary_all, folds_all = load_results(str(results_root))
summary_all = normalize_variant_labels(summary_all)
folds_all = normalize_variant_labels(folds_all)

if summary_all.empty:
    st.error(f"No metrics_summary.csv files found under {results_root}")
    st.stop()

canonical_sets = [
    s for s in ["canonical_protocol_only", "canonical_protocol_only_v2", "canonical_protocol_only_v3"]
    if s in set(summary_all["result_set"].astype(str))
]
if st.sidebar.button("Show all canonical v1/v2/v3"):
    for key in [
        "filter_result_set",
        "filter_dataset",
        "filter_task",
        "filter_sessions",
        "filter_variant_name",
        "filter_feature_set",
        "filter_window_type",
        "filter_protocol",
        "filter_family",
        "selected_models",
    ]:
        st.session_state.pop(key, None)
    st.rerun()

filters: dict[str, list[str]] = {}
with st.sidebar:
    st.subheader("Filters")
    for col, label in [
        ("result_set", "Result set"),
        ("dataset", "Dataset"),
        ("task", "Task"),
        ("sessions", "Sessions"),
        ("variant_name", "Variant"),
        ("feature_set", "Feature set"),
        ("window_type", "Window"),
        ("protocol", "Protocol"),
        ("family", "Family"),
    ]:
        options = sorted_unique(summary_all, col)
        if col == "result_set" and any(x.startswith("canonical_protocol_only") for x in options):
            default = canonical_sets or [x for x in options if x.startswith("canonical_protocol_only")]
        elif col == "result_set":
            default = [x for x in options if "smoke" not in x.lower()] or options
        else:
            default = options
        filters[col] = st.multiselect(label, options=options, default=default, key=f"filter_{col}")

summary = filter_df(summary_all, filters)
folds = filter_df(folds_all, filters)

metrics = metric_options(summary)
metric = st.sidebar.selectbox("Primary metric", options=metrics, index=metrics.index("macro_f1") if "macro_f1" in metrics else 0)

model_options = sorted_unique(summary, "model")
selected_models = st.sidebar.multiselect("Models", options=model_options, default=model_options, key="selected_models")
if selected_models:
    summary = summary[summary["model"].astype(str).isin(selected_models)]
    if not folds.empty and "model" in folds.columns:
        folds = folds[folds["model"].astype(str).isin(selected_models)]

expected_canonical = ["canonical_protocol_only", "canonical_protocol_only_v2", "canonical_protocol_only_v3"]
loaded_canonical = [s for s in expected_canonical if s in set(summary_all["result_set"].astype(str))]
visible_canonical = [s for s in expected_canonical if s in set(summary["result_set"].astype(str))]
if loaded_canonical:
    st.sidebar.caption("Loaded canonical: " + ", ".join(loaded_canonical))
missing_visible = [s for s in loaded_canonical if s not in visible_canonical]
if missing_visible:
    st.warning(
        "Some loaded canonical result sets are hidden by the current filters: "
        + ", ".join(missing_visible)
        + ". Use the sidebar button 'Show all canonical v1/v2/v3' to reset."
    )

tabs = st.tabs([
    "Overview",
    "Baseline vs Deep",
    "LOSO Drilldown",
    "Holdout Drilldown",
    "Model Detail",
    "Artifacts",
])

with tabs[0]:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Summary rows", f"{len(summary):,}")
    c2.metric("Fold rows", f"{len(folds):,}")
    c3.metric("Feature sets", summary["feature_set"].nunique() if "feature_set" in summary else 0)
    c4.metric("Models", summary["model"].nunique() if "model" in summary else 0)
    c5.metric("Protocols", summary["protocol"].nunique() if "protocol" in summary else 0)

    st.subheader("Coverage")
    coverage_cols = ["result_set", "dataset", "task", "sessions", "variant_name", "feature_set", "protocol", "family"]
    coverage = summary.groupby([c for c in coverage_cols if c in summary.columns], dropna=False).agg(
        models=("model", "nunique"),
        rows=("model", "size"),
    ).reset_index()
    st.dataframe(coverage, width="stretch")

    st.subheader("Top results")
    show_cols = [
        "result_set", "dataset", "task", "sessions", "variant_name", "feature_set", "protocol",
        "family", "model", "eval_unit", "total_params", "accuracy",
        "balanced_accuracy", "macro_f1", "artifact_dir",
    ]
    show_cols = [c for c in show_cols if c in summary.columns]
    st.dataframe(format_metrics(summary.sort_values(metric, ascending=False)[show_cols]), width="stretch")

    best = best_by_group(summary, metric, ["feature_set", "protocol", "family"])
    st.subheader(f"Best model per feature/protocol/family by {metric}")
    st.dataframe(format_metrics(best[[c for c in show_cols if c != "artifact_dir"]]), width="stretch")

with tabs[1]:
    st.subheader("Baseline vs deep models")
    compare_protocol = st.radio("Protocol", options=sorted_unique(summary, "protocol"), horizontal=True)
    comp = summary[summary["protocol"].astype(str) == compare_protocol].copy()
    if comp.empty:
        st.info("No rows for selected protocol.")
    else:
        best_family = best_by_group(comp, metric, ["feature_set", "family"])
        st.write(f"Best baseline and best deep model per feature set by `{metric}`.")
        bar_chart(best_family, "feature_set", metric, color="family", title=f"{compare_protocol}: best baseline vs best deep")
        st.dataframe(format_metrics(best_family.sort_values(["feature_set", "family"])), width="stretch")

        st.subheader("All selected models")
        compact_cols = ["feature_set", "protocol", "family", "model", "eval_unit", "total_params", "accuracy", "balanced_accuracy", "macro_f1"]
        compact_cols = [c for c in compact_cols if c in comp.columns]
        st.dataframe(format_metrics(comp.sort_values(["feature_set", metric], ascending=[True, False])[compact_cols]), width="stretch")

with tabs[2]:
    st.subheader("LOSO drilldown")
    loso_folds = folds[folds["protocol"].astype(str).eq("loso")] if "protocol" in folds.columns else pd.DataFrame()
    if loso_folds.empty:
        st.info("No LOSO fold metrics found.")
    else:
        col_a, col_b, col_c = st.columns(3)
        fs = col_a.selectbox("Feature set", options=sorted_unique(loso_folds, "feature_set"))
        fam = col_b.selectbox("Family", options=sorted_unique(loso_folds[loso_folds["feature_set"].astype(str) == fs], "family"))
        model_pool = loso_folds[(loso_folds["feature_set"].astype(str) == fs) & (loso_folds["family"].astype(str) == fam)]
        model = col_c.selectbox("Model", options=sorted_unique(model_pool, "model"))
        detail = model_pool[model_pool["model"].astype(str) == model].copy()

        c1, c2, c3 = st.columns(3)
        c1.metric("Folds", detail["fold"].nunique() if "fold" in detail else len(detail))
        c2.metric("Mean " + metric, f"{detail[metric].mean():.4f}" if metric in detail else "n/a")
        c3.metric("Std " + metric, f"{detail[metric].std(ddof=1):.4f}" if metric in detail and len(detail) > 1 else "0.0000")

        subj_col = "test_subject" if "test_subject" in detail.columns else "fold_subject"
        line_chart(detail, subj_col, metric, title=f"LOSO {metric} by held-out subject")
        st.subheader("Hardest subjects")
        fold_cols = ["feature_set", "family", "model", "fold", "test_subject", "validation_subject", "accuracy", "balanced_accuracy", "macro_f1"]
        fold_cols = [c for c in fold_cols if c in detail.columns]
        st.dataframe(format_metrics(detail.sort_values(metric)[fold_cols]), width="stretch")

        st.subheader("Subject comparison across models")
        compare_models = st.multiselect("Compare models on same feature set", options=sorted_unique(loso_folds[loso_folds["feature_set"].astype(str) == fs], "model"), default=[model])
        subj_comp = loso_folds[(loso_folds["feature_set"].astype(str) == fs) & (loso_folds["model"].astype(str).isin(compare_models))]
        line_chart(subj_comp, subj_col, metric, group="model", title=f"{fs}: LOSO subject-wise {metric}")

with tabs[3]:
    st.subheader("Random holdout and leakage gap")
    holdout = summary[summary["protocol"].astype(str).eq("random_holdout")] if "protocol" in summary.columns else pd.DataFrame()
    loso = summary[summary["protocol"].astype(str).eq("loso")] if "protocol" in summary.columns else pd.DataFrame()
    if holdout.empty:
        st.info("No random holdout summaries found.")
    else:
        st.write("Random holdout mixes windows across subjects, so it is useful as an optimism/leakage comparison, not as the main generalization claim.")
        best_holdout = best_by_group(holdout, metric, ["feature_set", "family"])
        bar_chart(best_holdout, "feature_set", metric, color="family", title=f"Random holdout best models by {metric}")
        st.dataframe(format_metrics(holdout.sort_values(["feature_set", metric], ascending=[True, False])), width="stretch")

    if not holdout.empty and not loso.empty:
        key_cols = ["result_set", "dataset", "task", "sessions", "variant_name", "feature_set", "window_type", "family", "model", "eval_unit"]
        key_cols = [c for c in key_cols if c in summary.columns]
        h = holdout[key_cols + [metric]].rename(columns={metric: "random_holdout"})
        l = loso[key_cols + [metric]].rename(columns={metric: "loso"})
        gap = h.merge(l, on=key_cols, how="inner")
        gap["holdout_minus_loso"] = gap["random_holdout"] - gap["loso"]
        st.subheader(f"Leakage/optimism gap: random holdout - LOSO ({metric})")
        bar_chart(gap.sort_values("holdout_minus_loso", ascending=False).head(25), "model", "holdout_minus_loso", color="feature_set")
        st.dataframe(format_metrics(gap.sort_values("holdout_minus_loso", ascending=False)), width="stretch")

with tabs[4]:
    st.subheader("Model detail")
    selected_model = st.selectbox("Model", options=sorted_unique(summary, "model"))
    detail = summary[summary["model"].astype(str) == selected_model].copy()
    if detail.empty:
        st.info("No rows for selected model.")
    else:
        bar_chart(detail.sort_values(["protocol", "feature_set"]), "feature_set", metric, color="protocol", title=f"{selected_model}: {metric}")
        cols = ["result_set", "dataset", "task", "sessions", "variant_name", "feature_set", "protocol", "family", "eval_unit", "total_params", "accuracy", "balanced_accuracy", "macro_f1"]
        cols = [c for c in cols if c in detail.columns]
        st.dataframe(format_metrics(detail.sort_values(["feature_set", "protocol"])[cols]), width="stretch")

        st.subheader("Parameter sizes")
        if "total_params" in detail.columns:
            param_cols = ["feature_set", "protocol", "eval_unit", "total_params", "parameter_size_mb_float32", "n_nodes", "node_feat_dim"]
            param_cols = [c for c in param_cols if c in detail.columns]
            st.dataframe(format_metrics(detail[param_cols].drop_duplicates()), width="stretch")

with tabs[5]:
    st.subheader("Artifact browser")
    select_cols = ["result_set", "variant_name", "feature_set", "protocol", "family", "model", "eval_unit", metric]
    select_cols = [c for c in select_cols if c in summary.columns]
    artifact_table = summary.sort_values(metric, ascending=False).reset_index(drop=True)
    labels = [
        " | ".join(str(row.get(c, "")) for c in select_cols)
        for _, row in artifact_table.iterrows()
    ]
    choice = st.selectbox("Result artifact", options=list(range(len(labels))), format_func=lambda i: labels[i])
    row = artifact_table.iloc[int(choice)]
    artifact_dir = Path(str(row.get("artifact_dir", "")))
    st.code(rel(artifact_dir, repo_root), language="text")

    c1, c2, c3, c4 = st.columns(4)
    for col, target in [
        (c1, artifact_for(row, "confusion_csv")),
        (c2, artifact_for(row, "confusion_png")),
        (c3, artifact_for(row, "predictions")),
        (c4, artifact_for(row, "classification_report")),
    ]:
        if target:
            col.caption(target.name)
            col.download_button(
                "Download",
                data=target.read_bytes(),
                file_name=target.name,
                mime="application/octet-stream",
                key=f"download_{target}",
            )
        else:
            col.caption("missing")

    cm_csv = artifact_for(row, "confusion_csv")
    if cm_csv:
        st.subheader("Confusion matrix")
        normalize = st.checkbox("Normalize rows", value=True)
        render_matrix_csv(cm_csv, normalize=normalize)

    report = artifact_for(row, "classification_report")
    if report:
        st.subheader("Classification report")
        st.dataframe(read_csv(report), width="stretch")

    pred = artifact_for(row, "predictions")
    if pred:
        st.subheader("Predictions preview")
        pred_df = read_csv(pred)
        if not pred_df.empty:
            st.dataframe(pred_df.head(2000), width="stretch")
            st.caption(f"Showing 2,000 of {len(pred_df):,} rows.")

    st.subheader("Files in artifact directory")
    if artifact_dir.exists():
        files = [
            {"name": p.name, "size_kb": round(p.stat().st_size / 1024, 2), "path": rel(p, repo_root)}
            for p in sorted(artifact_dir.iterdir())
            if p.is_file()
        ]
        st.dataframe(pd.DataFrame(files), width="stretch")
