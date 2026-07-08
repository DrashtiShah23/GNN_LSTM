#!/usr/bin/env python
"""Build PAMAP2 DOCX-standard derived tables, figures, and report.

This is a post-hoc report builder for the canonical protocol12 result layout.
It does not train models. It enriches the already generated seven-experiment
tables with metrics/figures that are derivable from saved fold rows and
prediction CSVs, and it explicitly records run-required gaps.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_canonical_protocol12_seven_experiments import (  # noqa: E402
    BASELINE_MODELS,
    LABEL_GROUPS,
    V3_MODELS,
    load_predictions,
    load_tables,
)


MODEL_LABELS = {
    "dummy_most_frequent": "Dummy",
    "gaussian_nb": "Gaussian NB",
    "knn_k5": "kNN-5",
    "linear_svm": "Linear SVM",
    "rbf_svm": "RBF SVM",
    "decision_tree_entropy": "Decision Tree",
    "bagged_tree_entropy": "Bagged Trees",
    "random_forest": "Random Forest",
    "adaboost_tree": "AdaBoost",
    "xgboost_hist": "XGBoost",
    "improved_gnn_lstm_res": "Improved GNN-LSTM Res",
    "improved_gnn_lstm_attn_adj_resbn": "Improved AttnAdj ResBN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PAMAP2 DOCX-standard report artifacts.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--out-root", default="results/canonical_protocol12_seven_experiments_top4")
    parser.add_argument("--feature-sets", default="acc16_hr,acc16_gyro,acc16_gyro_hr")
    parser.add_argument("--baseline-models", default="random_forest,knn_k5")
    parser.add_argument("--v3-models", default="improved_gnn_lstm_attn_adj_resbn,improved_gnn_lstm_res")
    parser.add_argument("--top-n-figures", type=int, default=6)
    return parser.parse_args()


def parse_csv(value: str) -> set[str]:
    return {x.strip() for x in str(value).replace(";", ",").split(",") if x.strip()}


def read_table(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size:
        return pd.read_csv(path)
    return pd.DataFrame()


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def pretty_model(model: str) -> str:
    return MODEL_LABELS.get(str(model), str(model))


def proba_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if str(c).startswith("proba_")]
    return sorted(cols, key=lambda c: int(str(c).split("_", 1)[1]) if str(c).split("_", 1)[1].isdigit() else 999)


def valid_probability_frame(part: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray] | None:
    cols = proba_cols(part)
    if not cols:
        return None
    probs = part[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(probs).all(axis=1)
    if not valid.any():
        return None
    probs = probs[valid]
    row_sum = probs.sum(axis=1)
    valid_sum = np.isfinite(row_sum) & (row_sum > 0)
    if not valid_sum.any():
        return None
    probs = probs[valid_sum] / row_sum[valid_sum, None]
    clean = part.loc[valid].iloc[np.where(valid_sum)[0]].copy()
    return clean, probs


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    out = np.zeros_like(probs)
    good = (y >= 0) & (y < probs.shape[1])
    out[np.arange(len(y))[good], y[good]] = 1.0
    return float(np.mean(np.sum((probs - out) ** 2, axis=1)))


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf < hi if i < bins - 1 else conf <= hi)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - conf[mask].mean()))
    return ece


def coverage_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, coverage: float) -> dict[str, float]:
    conf = probs.max(axis=1)
    n_keep = max(1, int(round(len(conf) * coverage)))
    keep = np.argsort(-conf)[:n_keep]
    yt = np.asarray(y_true)[keep]
    yp = np.asarray(y_pred)[keep]
    return {
        f"accuracy_at_{int(coverage * 100)}_coverage": float(accuracy_score(yt, yp)),
        f"macro_f1_at_{int(coverage * 100)}_coverage": float(f1_score(yt, yp, average="macro", zero_division=0)),
        f"n_at_{int(coverage * 100)}_coverage": int(n_keep),
    }


def ci95(values: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return math.nan, math.nan
    mean = vals.mean()
    sem = stats.sem(vals)
    radius = stats.t.ppf(0.975, len(vals) - 1) * sem
    return float(mean - radius), float(mean + radius)


def rank_biserial_from_wilcoxon(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff) & (diff != 0)]
    if len(diff) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diff))
    pos = ranks[diff > 0].sum()
    neg = ranks[diff < 0].sum()
    denom = len(diff) * (len(diff) + 1) / 2.0
    return float((pos - neg) / denom) if denom else 0.0


def enhanced_exp2(folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    loso = folds[folds["protocol"].astype(str).eq("loso")].copy()
    rows = []
    pair_rows = []
    rank_rows = []
    for (rs, fs, wt), part in loso.groupby(["result_set", "feature_set", "window_type"], dropna=False):
        rank_frame = part[["test_subject", "model", "macro_f1"]].dropna().copy()
        rank_frame["fold_rank"] = rank_frame.groupby("test_subject")["macro_f1"].rank(ascending=False, method="average")
        for model, sub_rank in rank_frame.groupby("model"):
            rank_rows.append({
                "result_set": rs,
                "feature_set": fs,
                "window_type": wt,
                "model": model,
                "mean_rank": float(sub_rank["fold_rank"].mean()),
                "rank_std": float(sub_rank["fold_rank"].std(ddof=1)) if len(sub_rank) > 1 else math.nan,
                "best_fold_count": int((sub_rank["fold_rank"] == 1).sum()),
            })

        best_model = (
            part.groupby("model")["macro_f1"].mean().sort_values(ascending=False).index[0]
            if not part.empty else None
        )
        best = part[part["model"].eq(best_model)].set_index("test_subject")["macro_f1"] if best_model else pd.Series(dtype=float)
        for keys, model_part in part.groupby(["family", "model"], dropna=False):
            vals = model_part["macro_f1"].to_numpy(dtype=float)
            lo, hi = ci95(vals)
            subjects = model_part["test_subject"].astype(str).tolist()
            cmp = model_part.set_index("test_subject")["macro_f1"]
            common = sorted(set(best.index).intersection(cmp.index), key=str)
            if best_model and keys[1] != best_model and common:
                diff = cmp.loc[common].to_numpy(dtype=float) - best.loc[common].to_numpy(dtype=float)
                try:
                    pval = float(stats.wilcoxon(diff, zero_method="wilcox").pvalue) if np.any(diff != 0) else 1.0
                except ValueError:
                    pval = math.nan
                effect = rank_biserial_from_wilcoxon(diff)
            else:
                pval = math.nan
                effect = math.nan
            rows.append({
                "result_set": rs,
                "feature_set": fs,
                "window_type": wt,
                "family": keys[0],
                "model": keys[1],
                "folds": int(model_part["fold"].nunique()),
                "test_subjects": ",".join(subjects),
                "accuracy_mean": float(model_part["accuracy"].mean()),
                "accuracy_std": float(model_part["accuracy"].std(ddof=1)),
                "macro_f1_mean": float(model_part["macro_f1"].mean()),
                "macro_f1_std": float(model_part["macro_f1"].std(ddof=1)),
                "macro_f1_ci95_low": lo,
                "macro_f1_ci95_high": hi,
                "balanced_accuracy_mean": float(model_part["balanced_accuracy"].mean()),
                "balanced_accuracy_std": float(model_part["balanced_accuracy"].std(ddof=1)),
                "comparison_model": best_model,
                "wilcoxon_p_vs_best": pval,
                "rank_biserial_effect_vs_best": effect,
            })

        models = sorted(part["model"].dropna().unique())
        for i, a in enumerate(models):
            aa = part[part["model"].eq(a)].set_index("test_subject")["macro_f1"]
            for b in models[i + 1:]:
                bb = part[part["model"].eq(b)].set_index("test_subject")["macro_f1"]
                common = sorted(set(aa.index).intersection(bb.index), key=str)
                if not common:
                    continue
                diff = aa.loc[common].to_numpy(dtype=float) - bb.loc[common].to_numpy(dtype=float)
                lo, hi = ci95(diff)
                try:
                    pval = float(stats.wilcoxon(diff, zero_method="wilcox").pvalue) if np.any(diff != 0) else 1.0
                except ValueError:
                    pval = math.nan
                pair_rows.append({
                    "result_set": rs,
                    "feature_set": fs,
                    "window_type": wt,
                    "model_a": a,
                    "model_b": b,
                    "macro_f1_mean_diff_a_minus_b": float(np.mean(diff)),
                    "macro_f1_diff_ci95_low": lo,
                    "macro_f1_diff_ci95_high": hi,
                    "wilcoxon_p": pval,
                    "rank_biserial_effect": rank_biserial_from_wilcoxon(diff),
                    "paired_folds": int(len(common)),
                })
    return pd.DataFrame(rows), pd.DataFrame(pair_rows), pd.DataFrame(rank_rows)


def enhanced_exp4(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, part in preds.groupby(["result_set", "feature_set", "window_type", "protocol", "family", "model"], dropna=False):
        parsed = valid_probability_frame(part)
        base = {
            "result_set": keys[0],
            "feature_set": keys[1],
            "window_type": keys[2],
            "protocol": keys[3],
            "family": keys[4],
            "model": keys[5],
        }
        if parsed is None:
            rows.append({**base, "status": "probabilities_not_available"})
            continue
        clean, probs = parsed
        y_true = clean["y_true_id"].astype(int).to_numpy()
        y_pred = clean["y_pred_id"].astype(int).to_numpy()
        labels = list(range(probs.shape[1]))
        out = {
            **base,
            "ece": expected_calibration_error(y_true, y_pred, probs),
            "brier_score": multiclass_brier(y_true, probs),
            "nll": float(log_loss(y_true, probs, labels=labels)),
            "mean_confidence": float(probs.max(axis=1).mean()),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "status": "ok",
        }
        out.update(coverage_metrics(y_true, y_pred, probs, 0.90))
        out.update(coverage_metrics(y_true, y_pred, probs, 0.80))
        out.update(coverage_metrics(y_true, y_pred, probs, 0.70))
        rows.append(out)
    return pd.DataFrame(rows)


def subject_failure_from_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    loso = preds[preds["protocol"].astype(str).eq("loso")].copy()
    for keys, part in loso.groupby(["result_set", "feature_set", "window_type", "family", "model", "test_subject"], dropna=False):
        if part.empty:
            continue
        recalls = []
        for label, label_part in part.groupby("y_true_label", dropna=False):
            recall = float((label_part["y_pred_label"].astype(str) == str(label)).mean())
            recalls.append((str(label), recall, int(len(label_part))))
        worst = sorted(recalls, key=lambda x: (x[1], -x[2], x[0]))[0] if recalls else ("", math.nan, 0)
        wrong = part[part["y_true_label"].astype(str) != part["y_pred_label"].astype(str)]
        if wrong.empty:
            dom = "none"
            dom_count = 0
        else:
            combo = wrong.assign(pair=wrong["y_true_label"].astype(str) + " -> " + wrong["y_pred_label"].astype(str))
            vc = combo["pair"].value_counts()
            dom = str(vc.index[0])
            dom_count = int(vc.iloc[0])
        rows.append({
            "result_set": keys[0],
            "feature_set": keys[1],
            "window_type": keys[2],
            "family": keys[3],
            "model": keys[4],
            "test_subject": keys[5],
            "accuracy": float(accuracy_score(part["y_true_id"], part["y_pred_id"])),
            "macro_f1": float(f1_score(part["y_true_id"], part["y_pred_id"], average="macro", zero_division=0)),
            "worst_activity": worst[0],
            "worst_activity_recall": worst[1],
            "worst_activity_support": worst[2],
            "dominant_confusion": dom,
            "dominant_confusion_count": dom_count,
        })
    return pd.DataFrame(rows)


def enhanced_exp7(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    mapping_rows = []
    for label, group in sorted(LABEL_GROUPS.items()):
        mapping_rows.append({"activity_group": group, "included_activity": label})
    for keys, part in preds.groupby(["result_set", "feature_set", "window_type", "protocol", "family", "model"], dropna=False):
        yt = part["y_true_label"].astype(str).map(LABEL_GROUPS)
        yp = part["y_pred_label"].astype(str).map(LABEL_GROUPS)
        mask = yt.notna() & yp.notna()
        if not mask.any():
            continue
        y_true = yt[mask].to_numpy()
        y_pred = yp[mask].to_numpy()
        labels = sorted(set(LABEL_GROUPS.values()))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        total = cm.sum()
        for i, group in enumerate(labels):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = total - tp - fn - fp
            wrong_targets = cm[i, :].copy()
            wrong_targets[i] = 0
            if wrong_targets.sum() == 0:
                main_conf = "none"
                main_conf_count = 0
            else:
                j = int(np.argmax(wrong_targets))
                main_conf = f"{group} -> {labels[j]}"
                main_conf_count = int(wrong_targets[j])
            rows.append({
                "result_set": keys[0],
                "feature_set": keys[1],
                "window_type": keys[2],
                "protocol": keys[3],
                "family": keys[4],
                "model": keys[5],
                "activity_group": group,
                "support": int(cm[i, :].sum()),
                "sensitivity": float(tp / (tp + fn)) if tp + fn else math.nan,
                "specificity": float(tn / (tn + fp)) if tn + fp else math.nan,
                "main_confusion": main_conf,
                "main_confusion_count": main_conf_count,
            })
    return pd.DataFrame(rows), pd.DataFrame(mapping_rows)


def pick_top_loso(summary: pd.DataFrame, n: int) -> pd.DataFrame:
    loso = summary[summary["protocol"].astype(str).eq("loso")].copy()
    return loso.sort_values("macro_f1", ascending=False).head(n)


def plot_exp1(table: pd.DataFrame, fig_dir: Path) -> None:
    df = table.copy()
    if "window_type" not in df.columns:
        df["window_type"] = "overlapping"
    deep = df[df["model"].isin(V3_MODELS)].copy()
    focus = deep if not deep.empty else df.sort_values("macro_f1", ascending=False).head(18)
    focus["model_label"] = focus["model"].map(pretty_model)
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=focus, x="feature_set", y="macro_f1", hue="protocol", ax=ax)
    ax.set_title("Experiment 1: Overlapping Window Protocol Comparison")
    ax.set_ylabel("Macro F1")
    ax.set_xlabel("Feature set")
    ax.text(0.01, -0.22, "Non-overlapping PAMAP2 runs are not present yet; this figure shows current overlapping results.", transform=ax.transAxes, fontsize=9)
    save_fig(fig, fig_dir / "fig_exp1_overlapping_protocol_comparison.png")


def plot_exp2_ranks(rank: pd.DataFrame, fig_dir: Path) -> None:
    if rank.empty:
        return
    df = rank.copy()
    df["model_label"] = df["model"].map(pretty_model)
    fig, ax = plt.subplots(figsize=(11, max(5, len(df["model"].unique()) * 0.28)))
    sns.scatterplot(data=df, x="mean_rank", y="model_label", hue="feature_set", size="best_fold_count", sizes=(30, 180), ax=ax)
    ax.invert_xaxis()
    ax.set_title("Experiment 2: LOSO Ranking Stability")
    ax.set_xlabel("Mean fold rank (lower is better)")
    ax.set_ylabel("Model")
    save_fig(fig, fig_dir / "fig_exp2_model_ranking_stability.png")


def plot_exp3(table: pd.DataFrame, fig_dir: Path) -> None:
    if table.empty:
        return
    ok = table[table["status"].astype(str).eq("ok")].copy()
    ok = ok[ok["test_subject"].astype(str).isin(["aggregate"]) | ok["fold"].astype(str).ne("aggregate")]
    sev_order = {"low": 0, "medium": 1, "high": 2}
    ok["severity_order"] = ok["severity"].map(sev_order)
    agg = ok.groupby(["feature_set", "perturbation", "severity", "model"], dropna=False)["macro_f1_drop"].mean().reset_index()
    top = agg.groupby("model")["macro_f1_drop"].mean().sort_values().head(6).index
    focus = agg[agg["model"].isin(top)].copy()
    focus["model_label"] = focus["model"].map(pretty_model)
    g = sns.relplot(
        data=focus,
        x="severity",
        y="macro_f1_drop",
        hue="model_label",
        col="perturbation",
        row="feature_set",
        kind="line",
        marker="o",
        facet_kws={"sharey": False},
        height=3.1,
        aspect=1.25,
    )
    g.set_axis_labels("Severity", "Macro-F1 drop")
    g.fig.suptitle("Experiment 3: Robustness Degradation Curves", y=1.02)
    out = fig_dir / "fig_exp3_robustness_degradation_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(out, dpi=170, bbox_inches="tight")
    g.fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(g.fig)


def plot_reliability_and_coverage(preds: pd.DataFrame, summary: pd.DataFrame, fig_dir: Path, top_n: int) -> None:
    top = pick_top_loso(summary, top_n)
    wanted = set(zip(top["result_set"], top["feature_set"], top["window_type"], top["protocol"], top["family"], top["model"]))
    for keys, part in preds.groupby(["result_set", "feature_set", "window_type", "protocol", "family", "model"], dropna=False):
        if keys not in wanted:
            continue
        parsed = valid_probability_frame(part)
        if parsed is None:
            continue
        clean, probs = parsed
        y_true = clean["y_true_id"].astype(int).to_numpy()
        y_pred = clean["y_pred_id"].astype(int).to_numpy()
        conf = probs.max(axis=1)
        correct = y_true == y_pred
        bins = np.linspace(0, 1, 11)
        bin_conf, bin_acc = [], []
        for i in range(len(bins) - 1):
            mask = (conf >= bins[i]) & (conf < bins[i + 1] if i < len(bins) - 2 else conf <= bins[i + 1])
            if mask.any():
                bin_conf.append(float(conf[mask].mean()))
                bin_acc.append(float(correct[mask].mean()))
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
        ax.plot(bin_conf, bin_acc, marker="o", label=pretty_model(keys[5]))
        ax.set_title(f"Reliability: {keys[1]} / {keys[2]} / {pretty_model(keys[5])}")
        ax.set_xlabel("Mean confidence")
        ax.set_ylabel("Empirical accuracy")
        ax.legend()
        save_fig(fig, fig_dir / f"fig_exp4_reliability_{keys[1]}_{keys[2]}_{keys[5]}.png")

        coverages = np.linspace(1.0, 0.5, 11)
        rows = []
        for cov in coverages:
            m = coverage_metrics(y_true, y_pred, probs, float(cov))
            rows.append({"coverage": cov, "accuracy": m[f"accuracy_at_{int(cov * 100)}_coverage"], "macro_f1": m[f"macro_f1_at_{int(cov * 100)}_coverage"]})
        cdf = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(cdf["coverage"], cdf["accuracy"], marker="o", label="Accuracy")
        ax.plot(cdf["coverage"], cdf["macro_f1"], marker="o", label="Macro F1")
        ax.set_title(f"Selective Prediction: {keys[1]} / {keys[2]} / {pretty_model(keys[5])}")
        ax.set_xlabel("Coverage kept")
        ax.set_ylabel("Score")
        ax.invert_xaxis()
        ax.legend()
        save_fig(fig, fig_dir / f"fig_exp4_coverage_{keys[1]}_{keys[2]}_{keys[5]}.png")


def plot_exp5_heatmaps(preds: pd.DataFrame, summary: pd.DataFrame, fig_dir: Path, top_n: int) -> None:
    top = pick_top_loso(summary, top_n)
    wanted = set(zip(top["result_set"], top["feature_set"], top["window_type"], top["protocol"], top["family"], top["model"]))
    for keys, part in preds.groupby(["result_set", "feature_set", "window_type", "protocol", "family", "model"], dropna=False):
        if keys not in wanted:
            continue
        labels = sorted(part["y_true_label"].dropna().astype(str).unique())
        subjects = sorted(part["test_subject"].dropna().astype(str).unique())
        mat = np.full((len(subjects), len(labels)), np.nan)
        for i, subj in enumerate(subjects):
            sub = part[part["test_subject"].astype(str).eq(subj)]
            for j, label in enumerate(labels):
                lab = sub[sub["y_true_label"].astype(str).eq(label)]
                if not lab.empty:
                    mat[i, j] = (lab["y_pred_label"].astype(str) == label).mean()
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), max(5, len(subjects) * 0.45)))
        sns.heatmap(mat, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, xticklabels=labels, yticklabels=subjects, ax=ax)
        ax.set_title(f"Experiment 5: Subject x Activity Recall - {keys[1]} / {keys[2]} / {pretty_model(keys[5])}")
        ax.set_xlabel("Activity")
        ax.set_ylabel("Held-out subject")
        save_fig(fig, fig_dir / f"fig_exp5_subject_activity_heatmap_{keys[1]}_{keys[2]}_{keys[5]}.png")


def plot_exp6(table: pd.DataFrame, fig_dir: Path) -> None:
    if table.empty:
        return
    ok = table[table["status"].astype(str).eq("ok")].copy()
    if ok.empty:
        return
    agg = ok.groupby(["feature_set", "model", "fine_tuning_strategy", "calibration_percentage"], dropna=False)["macro_f1_improvement"].mean().reset_index()
    top = agg.groupby("model")["macro_f1_improvement"].mean().sort_values(ascending=False).head(6).index
    focus = agg[agg["model"].isin(top)].copy()
    focus["model_label"] = focus["model"].map(pretty_model)
    g = sns.relplot(
        data=focus,
        x="calibration_percentage",
        y="macro_f1_improvement",
        hue="model_label",
        style="fine_tuning_strategy",
        col="feature_set",
        kind="line",
        marker="o",
        height=3.6,
        aspect=1.15,
    )
    g.set_axis_labels("Calibration fraction", "Macro-F1 improvement")
    g.fig.suptitle("Experiment 6: Few-Shot Calibration Efficiency", y=1.04)
    out = fig_dir / "fig_exp6_calibration_efficiency.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(out, dpi=170, bbox_inches="tight")
    g.fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(g.fig)


def plot_exp7_confusion(preds: pd.DataFrame, summary: pd.DataFrame, fig_dir: Path, top_n: int) -> None:
    top = pick_top_loso(summary, top_n)
    wanted = set(zip(top["result_set"], top["feature_set"], top["window_type"], top["protocol"], top["family"], top["model"]))
    labels = sorted(set(LABEL_GROUPS.values()))
    for keys, part in preds.groupby(["result_set", "feature_set", "window_type", "protocol", "family", "model"], dropna=False):
        if keys not in wanted:
            continue
        yt = part["y_true_label"].astype(str).map(LABEL_GROUPS)
        yp = part["y_pred_label"].astype(str).map(LABEL_GROUPS)
        mask = yt.notna() & yp.notna()
        if not mask.any():
            continue
        cm = confusion_matrix(yt[mask], yp[mask], labels=labels, normalize="true")
        fig, ax = plt.subplots(figsize=(7.5, 6))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f"Experiment 7: Health Group Confusion - {keys[1]} / {keys[2]} / {pretty_model(keys[5])}")
        ax.set_xlabel("Predicted group")
        ax.set_ylabel("True group")
        save_fig(fig, fig_dir / f"fig_exp7_health_group_confusion_{keys[1]}_{keys[2]}_{keys[5]}.png")


def top_rows_text(df: pd.DataFrame, metric: str, n: int = 5, ascending: bool = False) -> list[str]:
    if df.empty or metric not in df.columns:
        return ["- Not available."]
    rows = []
    for _, row in df.sort_values(metric, ascending=ascending).head(n).iterrows():
        rows.append(f"- `{row.get('feature_set', '')}` / `{row.get('model', '')}`: `{metric}={row[metric]:.4f}`")
    return rows


def build_report(
    out_root: Path,
    summary: pd.DataFrame,
    enhanced_dir: Path,
    fig_dir: Path,
    missing: list[str],
) -> None:
    tables = out_root / "manuscript_tables"
    exp1 = read_table(tables / "table_exp1_leakage_control.csv")
    exp2 = read_table(enhanced_dir / "table_exp2_statistical_reliability_enhanced.csv")
    exp3 = read_table(tables / "table_exp3_robustness.csv")
    exp4 = read_table(enhanced_dir / "table_exp4_calibration_selective_prediction.csv")
    exp5 = read_table(enhanced_dir / "table_exp5_subject_failure_enhanced.csv")
    exp6 = read_table(tables / "table_exp6_few_shot_calibration.csv")
    exp7 = read_table(enhanced_dir / "table_exp7_health_groups_enhanced.csv")
    lines = [
        "# PAMAP2 Protocol12 Seven-Experiment Report",
        "",
        "This report is generated from the canonical PAMAP2 protocol-only/protocol12 artifacts.",
        "",
        "## Current Scope",
        "",
        "- Dataset: `pamap2`",
        "- Task: `protocol12`",
        "- Feature sets: `acc16_hr`, `acc16_gyro`, `acc16_gyro_hr`",
        "- Core protocols currently available: `overlapping/loso` and `overlapping/random_holdout`",
        "- Model families: canonical baselines plus v3 improved GNN-LSTM variants",
        "",
        "## Missing Before Full DOCX Completion",
        "",
    ]
    lines.extend([f"- {item}" for item in missing] or ["- No run-required gaps detected."])
    lines.extend([
        "",
        "## Experiment 1: Leakage-Control Evaluation",
        "",
        "Current evidence compares random holdout against LOSO under overlapping windows. This is useful for subject-leakage inflation, but the DOCX-standard overlapping-vs-non-overlapping factorial design is not complete until stride-128 runs exist.",
        "",
    ])
    if not exp1.empty:
        gap = (
            exp1.dropna(subset=["holdout_minus_loso_accuracy"])
            .drop_duplicates(subset=["feature_set", "window_type", "family", "model", "holdout_minus_loso_accuracy"])
            .sort_values("holdout_minus_loso_accuracy", ascending=False)
        )
        lines.append("Largest holdout-minus-LOSO accuracy gaps:")
        lines.extend(top_rows_text(gap, "holdout_minus_loso_accuracy", n=6))
    lines.extend([
        "",
        "## Experiment 2: Statistical Reliability",
        "",
        "Enhanced table adds 95% confidence intervals, Wilcoxon p-values versus the best model in each result-set/feature-set group, rank-biserial effect sizes, and ranking stability.",
        "",
    ])
    if not exp2.empty:
        top = exp2.sort_values("macro_f1_mean", ascending=False).head(6)
        lines.append("Top LOSO mean macro-F1 rows:")
        lines.extend(top_rows_text(top, "macro_f1_mean", n=6))
    lines.extend([
        "",
        "## Experiment 3: Robustness",
        "",
        "Real robustness tables are present for Gaussian noise, random channel dropout, and heart-rate zeroing. The runner now also supports `sensor_node_zero` and `random_window_dropout`, but those new perturbations still need to be executed if they are required in the final manuscript.",
        "",
    ])
    if not exp3.empty:
        ok = exp3[exp3["status"].astype(str).eq("ok")]
        drop = ok.groupby(["feature_set", "model"], dropna=False)["macro_f1_drop"].mean().reset_index()
        useful_drop = drop[~drop["model"].astype(str).eq("dummy_most_frequent")]
        if not useful_drop.empty:
            drop = useful_drop
        lines.append("Most robust rows by lowest average macro-F1 drop:")
        lines.extend(top_rows_text(drop, "macro_f1_drop", n=6, ascending=True))
    lines.extend([
        "",
        "## Experiment 4: Calibration, Uncertainty, and Selective Prediction",
        "",
        "Enhanced table adds Brier score, negative log-likelihood, and selective prediction metrics at 90%, 80%, and 70% coverage for rows with probability columns. Most classical baseline rows still lack saved probabilities.",
        "",
    ])
    if not exp4.empty:
        ok = exp4[exp4["status"].astype(str).eq("ok")]
        lines.append("Lowest ECE among probability-available rows:")
        lines.extend(top_rows_text(ok, "ece", n=6, ascending=True))
    lines.extend([
        "",
        "## Experiment 5: Subject-Level Failure Analysis",
        "",
        "Enhanced table adds worst activity and dominant confusion per held-out subject/model/feature-set.",
        "",
    ])
    if not exp5.empty:
        useful_exp5 = exp5[~exp5["model"].astype(str).eq("dummy_most_frequent")]
        if useful_exp5.empty:
            useful_exp5 = exp5
        hard = useful_exp5.sort_values("macro_f1", ascending=True).head(8)
        lines.append("Hardest subject/model rows by macro-F1:")
        for _, row in hard.iterrows():
            lines.append(
                f"- subject `{row['test_subject']}` / `{row['feature_set']}` / `{row['model']}`: "
                f"macro_f1={row['macro_f1']:.4f}, worst_activity=`{row['worst_activity']}`, "
                f"dominant_confusion=`{row['dominant_confusion']}`"
            )
    lines.extend([
        "",
        "## Experiment 6: Few-Shot Subject Calibration",
        "",
        "Real few-shot calibration rows are present for 0%, 1%, 5%, and 10% calibration with classifier-head and full-model strategies where applicable.",
        "",
    ])
    if not exp6.empty:
        ok = exp6[exp6["status"].astype(str).eq("ok")]
        ten = ok[ok["calibration_percentage"].astype(float).eq(0.10)]
        gain = ten.groupby(["feature_set", "model", "fine_tuning_strategy"], dropna=False)["macro_f1_improvement"].mean().reset_index()
        lines.append("Best 10% calibration gains:")
        lines.extend(top_rows_text(gain, "macro_f1_improvement", n=8))
    lines.extend([
        "",
        "## Experiment 7: Health-Relevant Activity Groups",
        "",
        "Enhanced table adds sensitivity, specificity, and main group confusion. Activity-group mapping is exported separately.",
        "",
    ])
    if not exp7.empty:
        useful_exp7 = exp7[~exp7["model"].astype(str).eq("dummy_most_frequent")]
        if useful_exp7.empty:
            useful_exp7 = exp7
        low = useful_exp7.sort_values("sensitivity", ascending=True).head(8)
        lines.append("Lowest group sensitivity rows:")
        for _, row in low.iterrows():
            lines.append(
                f"- `{row['feature_set']}` / `{row['model']}` / `{row['activity_group']}`: "
                f"sensitivity={row['sensitivity']:.4f}, specificity={row['specificity']:.4f}, "
                f"main_confusion=`{row['main_confusion']}`"
            )
    figures = sorted(p.name for p in fig_dir.glob("*.png"))
    lines.extend([
        "",
        "## Generated Figure Files",
        "",
    ])
    lines.extend([f"- `{name}`" for name in figures] or ["- No figures generated."])
    lines.extend([
        "",
        "## Generated Enhanced Tables",
        "",
    ])
    lines.extend([f"- `{p.name}`" for p in sorted(enhanced_dir.glob("*.csv"))])
    lines.append("")
    (out_root / "PAMAP2_SEVEN_EXPERIMENTS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root)
    tables = out_root / "manuscript_tables"
    enhanced = out_root / "manuscript_tables_enhanced"
    figures = out_root / "manuscript_figures"
    feature_sets = parse_csv(args.feature_sets)
    baseline_models = parse_csv(args.baseline_models)
    v3_models = parse_csv(args.v3_models)

    summary, folds = load_tables(
        Path(args.results_root),
        feature_sets,
        include_baselines=True,
        include_v3=True,
        baseline_models=baseline_models,
        v3_models=v3_models,
    )
    if summary.empty:
        raise SystemExit("No canonical PAMAP2 summary rows found.")
    preds = load_predictions(summary)
    if preds.empty:
        raise SystemExit("No prediction rows found; cannot build derived report.")

    exp2_summary, exp2_pairs, exp2_ranks = enhanced_exp2(folds)
    exp4 = enhanced_exp4(preds)
    exp5 = subject_failure_from_predictions(preds)
    exp7, exp7_map = enhanced_exp7(preds)

    write_table(exp2_summary, enhanced / "table_exp2_statistical_reliability_enhanced.csv")
    write_table(exp2_pairs, enhanced / "table_exp2_pairwise_comparisons_enhanced.csv")
    write_table(exp2_ranks, enhanced / "table_exp2_ranking_stability.csv")
    write_table(exp4, enhanced / "table_exp4_calibration_selective_prediction.csv")
    write_table(exp5, enhanced / "table_exp5_subject_failure_enhanced.csv")
    write_table(exp7, enhanced / "table_exp7_health_groups_enhanced.csv")
    write_table(exp7_map, enhanced / "table_exp7_activity_group_mapping.csv")

    exp1 = read_table(tables / "table_exp1_leakage_control.csv")
    exp3 = read_table(tables / "table_exp3_robustness.csv")
    exp6 = read_table(tables / "table_exp6_few_shot_calibration.csv")
    figures.mkdir(parents=True, exist_ok=True)
    for old in list(figures.glob("fig_exp*.png")) + list(figures.glob("fig_exp*.pdf")):
        old.unlink()
    plot_exp1(exp1, figures)
    plot_exp2_ranks(exp2_ranks, figures)
    plot_exp3(exp3, figures)
    plot_reliability_and_coverage(preds, summary, figures, args.top_n_figures)
    plot_exp5_heatmaps(preds, summary, figures, args.top_n_figures)
    plot_exp6(exp6, figures)
    plot_exp7_confusion(preds, summary, figures, args.top_n_figures)

    missing = []
    if "window_type" not in summary.columns or not summary["window_type"].astype(str).str.contains("non_overlapping").any():
        missing.append("Experiment 1 still needs non-overlapping PAMAP2 protocol12 runs (`window=128`, `stride=128`) for all selected models/protocols.")
    exp3_perturbs = set(exp3.get("perturbation", pd.Series(dtype=str)).dropna().astype(str))
    for required in ["sensor_node_zero", "random_window_dropout"]:
        if required not in exp3_perturbs:
            missing.append(f"Experiment 3 still needs real `{required}` robustness rows if the final report must cover the full DOCX perturbation list.")
    if "xgboost_hist" in set(exp3.get("model", pd.Series(dtype=str)).dropna().astype(str)) and "xgboost_hist" not in set(summary["model"].dropna().astype(str)):
        missing.append("XGBoost is present in Exp3/Exp6 but absent from canonical core Exp1/2/4/5/7 summaries; either rerun core XGBoost or exclude it consistently.")
    if exp4["status"].astype(str).eq("probabilities_not_available").any():
        missing.append("Classical baseline calibration remains partial because most baseline prediction artifacts do not include probability columns.")

    build_report(out_root, summary, enhanced, figures, missing)
    manifest = {
        "out_root": str(out_root),
        "enhanced_tables": str(enhanced),
        "figures": str(figures),
        "report": str(out_root / "PAMAP2_SEVEN_EXPERIMENTS_REPORT.md"),
        "summary_rows": int(len(summary)),
        "fold_rows": int(len(folds)),
        "prediction_rows": int(len(preds)),
        "missing": missing,
    }
    (out_root / "pamap2_docx_standard_report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
