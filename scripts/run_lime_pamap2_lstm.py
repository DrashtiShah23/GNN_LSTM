"""
LIME tabular explanations for LSTM on PAMAP2 (flattened window = 128*C features).

Trains a small LSTM on a stratified subsample (fast), then explains held-out
instances with LIME. Saves JSON + a bar chart of mean absolute LIME weights
aggregated over explained samples.

Usage:
  python scripts/run_lime_pamap2_lstm.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lime import lime_tabular

from src.config import PROCESSED_DIR, METRICS_DIR, PLOTS_DIR, SEED, BATCH_SIZE
from src.models import LSTMOnlyModel
from src.train import get_device, set_seed

Path(METRICS_DIR).mkdir(parents=True, exist_ok=True)
Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)


def main():
    set_seed(SEED)
    device = get_device()
    X = np.load(Path(PROCESSED_DIR) / "pamap2_X.npy")
    y_raw = np.load(Path(PROCESSED_DIR) / "pamap2_y.npy")
    u = np.unique(y_raw)
    mp = {int(o): i for i, o in enumerate(u)}
    y = np.vectorize(mp.__getitem__)(y_raw)
    T, C = X.shape[1], X.shape[2]
    flat_dim = T * C

    rng = np.random.default_rng(SEED)
    n_sub = min(4000, len(X))
    idx = rng.choice(len(X), n_sub, replace=False)
    Xs, ys = X[idx], y[idx]

    X_tr, X_te, y_tr, y_te = train_test_split(
        Xs, ys, test_size=0.25, random_state=SEED, stratify=ys
    )
    X_tr_f = X_tr.reshape(len(X_tr), flat_dim)
    X_te_f = X_te.reshape(len(X_te), flat_dim)

    tr_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    n_cls = len(np.unique(y))
    model = LSTMOnlyModel(C, n_cls).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(1, 26):
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()

    model.eval()

    def predict_proba(arr: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.tensor(arr, dtype=torch.float32).view(-1, T, C).to(device)
            return torch.softmax(model(xt), dim=1).cpu().numpy()

    feat_names = [f"t{t}_c{c}" for t in range(T) for c in range(C)]
    explainer = lime_tabular.LimeTabularExplainer(
        X_tr_f,
        mode="classification",
        training_labels=y_tr,
        feature_names=feat_names,
        random_state=SEED,
    )

    n_explain = min(12, len(X_te_f))
    explain_idx = rng.choice(len(X_te_f), n_explain, replace=False)
    per_sample = []
    weight_acc = np.zeros(flat_dim)
    for i in explain_idx:
        row = X_te_f[i]
        exp = explainer.explain_instance(
            row,
            predict_proba,
            num_features=30,
            top_labels=1,
        )
        lab = exp.top_labels[0]
        wdict = dict(exp.as_list(label=lab))
        vec = np.zeros(flat_dim)
        for j, fn in enumerate(feat_names):
            vec[j] = abs(wdict.get(fn, 0.0))
        weight_acc += vec
        per_sample.append({"index": int(i), "top_label": int(lab), "weights_top": exp.as_list(label=lab)[:15]})
    weight_acc /= max(n_explain, 1)
    top_k = 25
    top_ix = np.argsort(weight_acc)[::-1][:top_k]

    out_json = {
        "model": "LSTMOnlyModel",
        "dataset": "pamap2",
        "n_train_subsample": len(X_tr),
        "lime_samples": n_explain,
        "top_features": [{"name": feat_names[j], "mean_abs_weight": float(weight_acc[j])} for j in top_ix],
        "per_sample": per_sample,
    }
    with open(Path(METRICS_DIR) / "lime_lstm_pamap2.json", "w") as f:
        json.dump(out_json, f, indent=2)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(np.arange(top_k), weight_acc[top_ix][::-1])
    ax.set_yticks(np.arange(top_k))
    ax.set_yticklabels([feat_names[j] for j in top_ix[::-1]], fontsize=7)
    ax.set_xlabel("Mean |LIME weight|")
    ax.set_title("LIME — LSTM on PAMAP2 (flattened window features, top-25)")
    plt.tight_layout()
    fig.savefig(Path(PLOTS_DIR) / "lime_lstm_pamap2.png", dpi=150)
    plt.close()
    print(f"Saved {METRICS_DIR}/lime_lstm_pamap2.json and {PLOTS_DIR}/lime_lstm_pamap2.png")


if __name__ == "__main__":
    main()
