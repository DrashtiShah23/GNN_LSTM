"""
Generate SHAP feature-importance outputs for a Random Forest on PAMAP2.

Outputs:
  - results/plots/shap_rf_pamap2.png
  - results/metrics/shap_rf_pamap2_top20.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier

from src.baselines import extract_features
from src.config import METRICS_DIR, PLOTS_DIR, SEED


def main() -> None:
    x_path = Path("data/processed/pamap2_X.npy")
    y_path = Path("data/processed/pamap2_y.npy")
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "Missing processed PAMAP2 arrays. Expected data/processed/pamap2_X.npy and pamap2_y.npy"
        )

    x = np.load(x_path)
    y = np.load(y_path)
    x_feat = extract_features(x)

    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(x_feat, y)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(x_feat), size=min(500, len(x_feat)), replace=False)
    x_sample = x_feat[idx]

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(x_sample)

    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        sv = np.asarray(shap_values)
        # Handle different SHAP output shapes across versions:
        # (n_samples, n_features), (n_samples, n_features, n_classes), etc.
        if sv.ndim == 3:
            mean_abs = np.abs(sv).mean(axis=(0, 2))
        elif sv.ndim == 2:
            mean_abs = np.abs(sv).mean(axis=0)
        else:
            mean_abs = np.abs(sv).reshape(-1)

    n_ch = x.shape[2]
    stat_names = ["mean", "std", "min", "max", "rms", "fft_energy"]
    feat_names = [f"{s}_ch{c}" for s in stat_names for c in range(n_ch)]

    top_k = 20
    top_idx = np.argsort(mean_abs)[::-1][:top_k]
    top_names = [feat_names[i] for i in top_idx]
    top_vals = mean_abs[top_idx].tolist()

    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(METRICS_DIR).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_names[::-1], np.array(top_vals)[::-1], color="steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top-20 SHAP Features — Random Forest (PAMAP2)")
    plt.tight_layout()
    out_plot = Path(PLOTS_DIR) / "shap_rf_pamap2.png"
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)

    out_json = Path(METRICS_DIR) / "shap_rf_pamap2_top20.json"
    out_json.write_text(
        json.dumps({"top_features": top_names, "top_mean_abs_shap": top_vals}, indent=2),
        encoding="utf-8",
    )

    print(f"Saved plot: {out_plot}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
