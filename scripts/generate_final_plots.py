"""
Generate all final result plots and SHAP interpretability analysis.
Run after all LOSO experiments have completed.
"""
import numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from src.config import (
    METRICS_DIR, PLOTS_DIR, PAMAP2_ACTIVITIES,
    BATCH_SIZE, SEED, WINDOW_SIZE,
)
from src.evaluation import compute_metrics, plot_confusion_matrix

Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
Path(METRICS_DIR).mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.1)

# ── Load data & label encoder ─────────────────────────────────────────────────
X_p    = np.load("data/processed/pamap2_X.npy")
y_raw_p = np.load("data/processed/pamap2_y.npy")
subj_p  = np.load("data/processed/pamap2_subjects.npy")
le_p = LabelEncoder(); y_p = le_p.fit_transform(y_raw_p)
act_names_p = [PAMAP2_ACTIVITIES[k] for k in sorted(le_p.classes_)]

X_h    = np.load("data/processed/hhar_X.npy")
y_raw_h = np.load("data/processed/hhar_y.npy")
subj_h  = np.load("data/processed/hhar_subjects.npy")
le_h = LabelEncoder(); y_h = le_h.fit_transform(y_raw_h)
act_names_h = list(le_h.classes_)

print(f"PAMAP2 activities: {act_names_p}")
print(f"HHAR activities:   {act_names_h}")

# ════════════════════════════════════════════════════════════════════════════
# 1. SHAP for RF and XGBoost on PAMAP2
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SHAP Feature Importance — PAMAP2 Baselines")
print("=" * 60)
try:
    import shap
    from src.baselines import extract_features, run_baselines_loso
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X_feat = extract_features(X_p)
    # Use all data to fit RF for SHAP (representative)
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(X_feat, y_p)
    n_ch = X_p.shape[2]
    feat_stats = ["mean", "std", "min", "max", "rms", "fft_energy"]
    feat_names = [f"{stat}_ch{ch}" for stat in feat_stats for ch in range(n_ch)]

    explainer_rf = shap.TreeExplainer(rf)
    # Use 500 random samples for speed
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_feat), size=min(500, len(X_feat)), replace=False)
    shap_values = explainer_rf.shap_values(X_feat[idx])

    # Mean absolute SHAP across classes
    if isinstance(shap_values, list):
        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)

    top_k = 20
    top_idx = np.argsort(mean_shap)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feat_names[i] for i in top_idx[::-1]], mean_shap[top_idx[::-1]], color="steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top-{top_k} Features — Random Forest (PAMAP2)")
    plt.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/shap_rf_pamap2.png", dpi=150)
    print(f"  Saved SHAP RF plot → {PLOTS_DIR}/shap_rf_pamap2.png")
    plt.close(fig)
except Exception as e:
    print(f"  SHAP skipped: {e}")

# ════════════════════════════════════════════════════════════════════════════
# 2. Comprehensive Model Comparison — PAMAP2
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Model Comparison Charts — PAMAP2")
print("=" * 60)

# Load baseline results
with open(f"{METRICS_DIR}/pamap2_baselines.json") as f:
    baselines_p = json.load(f)

# Load deep model results
deep_p = {}
deep_path = Path(f"{METRICS_DIR}/pamap2_deep_models.json")
if deep_path.exists():
    with open(deep_path) as f:
        deep_p = json.load(f)
else:
    # Fall back to per-fold npy files if available
    for name in ["lstm_only_pamap2", "gnn_only_pamap2", "gnn_lstm_pamap2"]:
        tp = Path(f"{METRICS_DIR}/{name}_y_true.npy")
        pp = Path(f"{METRICS_DIR}/{name}_y_pred.npy")
        if tp.exists() and pp.exists():
            yt = np.load(tp); yp = np.load(pp)
            m = compute_metrics(yt, yp)
            deep_p[name] = {"mean_accuracy": m["accuracy"], "std_accuracy": 0.0, "macro_f1": m["macro_f1"]}

# Combined data
model_names, means, stds, f1s = [], [], [], []
order = [("SVM", baselines_p), ("RandomForest", baselines_p), ("XGBoost", baselines_p),
         ("lstm_only_pamap2", deep_p), ("gnn_only_pamap2", deep_p), ("gnn_lstm_pamap2", deep_p)]
display_names = ["SVM", "Random Forest", "XGBoost", "LSTM-only", "GNN-only", "GNN+LSTM"]
colors = ["#4C72B0", "#4C72B0", "#4C72B0", "#DD8452", "#DD8452", "#55A868"]

for (key, d), dname in zip(order, display_names):
    if key in d:
        v = d[key]
        model_names.append(dname)
        means.append(v.get("mean_accuracy", v.get("accuracy", 0)))
        stds.append(v.get("std_accuracy", 0))
        f1s.append(v.get("mean_macro_f1", v.get("macro_f1", 0)))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy
bars = axes[0].barh(model_names, means, xerr=stds, color=colors, capsize=5, edgecolor="white")
axes[0].set_xlim(0, 1.05)
axes[0].set_xlabel("LOSO Accuracy", fontsize=12)
axes[0].set_title("Model Accuracy — PAMAP2 LOSO", fontsize=13)
for bar, val in zip(bars, means):
    axes[0].text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=10)

# F1
bars2 = axes[1].barh(model_names, f1s, color=colors, edgecolor="white")
axes[1].set_xlim(0, 1.05)
axes[1].set_xlabel("Macro F1-Score", fontsize=12)
axes[1].set_title("Model Macro F1 — PAMAP2 LOSO", fontsize=13)
for bar, val in zip(bars2, f1s):
    axes[1].text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=10)

from matplotlib.patches import Patch
legend_els = [Patch(facecolor="#4C72B0", label="Classical ML"),
              Patch(facecolor="#DD8452", label="Deep Learning"),
              Patch(facecolor="#55A868", label="Proposed (GNN+LSTM)")]
fig.legend(handles=legend_els, loc="lower center", ncol=3, fontsize=11, frameon=True)
plt.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(f"{PLOTS_DIR}/model_comparison_pamap2.png", dpi=150)
print(f"  Saved → {PLOTS_DIR}/model_comparison_pamap2.png")
plt.close(fig)

# ── Confusion matrices (PAMAP2 best models) ──────────────────────────────────
for name, act_names in [
    ("gnn_lstm_pamap2", act_names_p),
    ("lstm_only_pamap2", act_names_p),
    ("gnn_only_pamap2", act_names_p),
]:
    tp = Path(f"{METRICS_DIR}/{name}_y_true.npy")
    pp = Path(f"{METRICS_DIR}/{name}_y_pred.npy")
    if tp.exists() and pp.exists():
        yt = np.load(tp); yp = np.load(pp)
        cm = confusion_matrix(yt, yp)
        # Normalise row-wise
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=act_names, yticklabels=act_names, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {name} (normalised)")
        plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
        plt.tight_layout()
        fig.savefig(f"{PLOTS_DIR}/cm_{name}.png", dpi=150)
        print(f"  Saved → {PLOTS_DIR}/cm_{name}.png")
        plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# 3. HHAR comparison chart (if results exist)
# ════════════════════════════════════════════════════════════════════════════
hhar_baseline_path = Path(f"{METRICS_DIR}/hhar_baselines.json")
hhar_deep_path     = Path(f"{METRICS_DIR}/hhar_deep_models.json")
if hhar_baseline_path.exists() and hhar_deep_path.exists():
    print("\n" + "=" * 60)
    print("Model Comparison Charts — HHAR")
    print("=" * 60)
    with open(hhar_baseline_path) as f: baselines_h = json.load(f)
    with open(hhar_deep_path) as f: deep_h = json.load(f)

    model_names_h, means_h, stds_h, f1s_h = [], [], [], []
    order_h = [("SVM", baselines_h), ("RandomForest", baselines_h), ("XGBoost", baselines_h),
               ("lstm_only_hhar", deep_h), ("gnn_only_hhar", deep_h), ("gnn_lstm_hhar", deep_h)]
    display_h = ["SVM", "Random Forest", "XGBoost", "LSTM-only", "GNN-only", "GNN+LSTM"]
    for (key, d), dname in zip(order_h, display_h):
        if key in d:
            v = d[key]
            model_names_h.append(dname)
            means_h.append(v.get("mean_accuracy", v.get("accuracy", 0)))
            stds_h.append(v.get("std_accuracy", 0))
            f1s_h.append(v.get("mean_macro_f1", v.get("macro_f1", 0)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(model_names_h, means_h, xerr=stds_h, color=colors[:len(means_h)], capsize=5)
    axes[0].set_xlim(0, 1.05); axes[0].set_xlabel("LOSO Accuracy")
    axes[0].set_title("Model Accuracy — HHAR LOSO")
    axes[1].barh(model_names_h, f1s_h, color=colors[:len(f1s_h)])
    axes[1].set_xlim(0, 1.05); axes[1].set_xlabel("Macro F1")
    axes[1].set_title("Model Macro F1 — HHAR LOSO")
    plt.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/model_comparison_hhar.png", dpi=150)
    print(f"  Saved → {PLOTS_DIR}/model_comparison_hhar.png")
    plt.close(fig)

    # HHAR confusion matrices
    for name in ["gnn_lstm_hhar", "lstm_only_hhar", "gnn_only_hhar"]:
        tp = Path(f"{METRICS_DIR}/{name}_y_true.npy")
        pp = Path(f"{METRICS_DIR}/{name}_y_pred.npy")
        if tp.exists() and pp.exists():
            yt = np.load(tp); yp = np.load(pp)
            cm = confusion_matrix(yt, yp)
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
            fig, ax = plt.subplots(figsize=(9, 7))
            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                        xticklabels=act_names_h, yticklabels=act_names_h, ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title(f"Confusion Matrix — {name} (normalised)")
            plt.tight_layout()
            fig.savefig(f"{PLOTS_DIR}/cm_{name}.png", dpi=150)
            print(f"  Saved → {PLOTS_DIR}/cm_{name}.png")
            plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# 4. Cross-dataset comparison (side by side) if both done
# ════════════════════════════════════════════════════════════════════════════
if hhar_baseline_path.exists() and hhar_deep_path.exists() and deep_path.exists():
    print("\nGenerating cross-dataset comparison...")
    labels = ["SVM", "Random Forest", "XGBoost", "LSTM-only", "GNN-only", "GNN+LSTM"]
    pamap2_accs = [
        baselines_p.get("SVM", {}).get("mean_accuracy", 0),
        baselines_p.get("RandomForest", {}).get("mean_accuracy", 0),
        baselines_p.get("XGBoost", {}).get("mean_accuracy", 0),
        deep_p.get("lstm_only_pamap2", {}).get("mean_accuracy", 0),
        deep_p.get("gnn_only_pamap2",  {}).get("mean_accuracy", 0),
        deep_p.get("gnn_lstm_pamap2",  {}).get("mean_accuracy", 0),
    ]
    hhar_accs = [
        baselines_h.get("SVM", {}).get("mean_accuracy", 0),
        baselines_h.get("RandomForest", {}).get("mean_accuracy", 0),
        baselines_h.get("XGBoost", {}).get("mean_accuracy", 0),
        deep_h.get("lstm_only_hhar", {}).get("mean_accuracy", 0),
        deep_h.get("gnn_only_hhar",  {}).get("mean_accuracy", 0),
        deep_h.get("gnn_lstm_hhar",  {}).get("mean_accuracy", 0),
    ]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - w/2, pamap2_accs, w, label="PAMAP2", color="#4C72B0")
    ax.bar(x + w/2, hhar_accs,   w, label="HHAR",   color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("LOSO Accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Cross-Dataset Model Comparison (LOSO Accuracy)")
    ax.legend(); plt.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/cross_dataset_comparison.png", dpi=150)
    print(f"  Saved → {PLOTS_DIR}/cross_dataset_comparison.png")
    plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# 5. Model parameter count & latency profiling
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Model Profiling — Parameter Count & Latency")
print("=" * 60)
import time, torch
from src.models import GNNLSTMModel, LSTMOnlyModel, GNNOnlyModel
from src.train import get_device
from src.config import PAMAP2_NODE_FEAT_DIM

device = get_device()
n_cls = len(act_names_p)
configs = [
    ("LSTM-only",  LSTMOnlyModel(WINDOW_SIZE * 18, n_cls),         False),
    ("GNN-only",   GNNOnlyModel(PAMAP2_NODE_FEAT_DIM, 3, n_cls),   True),
    ("GNN+LSTM",   GNNLSTMModel(PAMAP2_NODE_FEAT_DIM, 3, n_cls),   True),
]
profile_rows = []
n_reps = 100
for name, model, use_adj in configs:
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    # Dummy forward pass for latency
    if use_adj:
        if "LSTM" in name and "GNN" in name:
            dummy_x = torch.randn(1, 10, 3, PAMAP2_NODE_FEAT_DIM).to(device)
        else:
            dummy_x = torch.randn(1, 3, PAMAP2_NODE_FEAT_DIM).to(device)
        dummy_adj = torch.eye(3).to(device)
        with torch.no_grad():
            for _ in range(10): model(dummy_x, dummy_adj)  # warm-up
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_reps): model(dummy_x, dummy_adj)
    else:
        dummy_x = torch.randn(1, WINDOW_SIZE * 18).to(device)
        with torch.no_grad():
            for _ in range(10): model(dummy_x)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_reps): model(dummy_x)
    latency_ms = (time.perf_counter() - t0) / n_reps * 1000
    profile_rows.append({"Model": name, "Params": n_params, "Latency_ms": latency_ms})
    print(f"  {name:12s}: {n_params:>10,} params  |  {latency_ms:.3f} ms/sample")

# Save profiling table
with open(f"{METRICS_DIR}/model_profiling.json", "w") as f:
    json.dump(profile_rows, f, indent=2)

# Bar chart of param counts
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
names_ = [r["Model"] for r in profile_rows]
params_ = [r["Params"] / 1e6 for r in profile_rows]
lats_   = [r["Latency_ms"] for r in profile_rows]
axes[0].bar(names_, params_, color=["#4C72B0", "#DD8452", "#55A868"])
axes[0].set_ylabel("Parameters (M)"); axes[0].set_title("Model Size")
axes[1].bar(names_, lats_,   color=["#4C72B0", "#DD8452", "#55A868"])
axes[1].set_ylabel("Latency (ms/sample)"); axes[1].set_title("Inference Latency")
plt.tight_layout()
fig.savefig(f"{PLOTS_DIR}/model_profiling.png", dpi=150)
print(f"  Saved → {PLOTS_DIR}/model_profiling.png")
plt.close(fig)

print("\n✅ All final plots & profiling complete!")
