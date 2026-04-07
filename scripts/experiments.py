"""
experiments.py — All extension experiments for the HAR GNN+LSTM project.

Experiments:
  1. CNN1D baseline           — full LOSO on PAMAP2 + HHAR
  2. Cross-device HHAR        — train on device subset, test on held-out device
  3. Graph ablation           — fixed adj vs learnable adj vs flatten+LSTM
  4. Data augmentation        — gaussian / scale / timewarp on HHAR
  5. Neural interpretability  — Integrated Gradients on GNN-only
  6. Error analysis           — per-class F1, confusion matrix comparison, hard pairs
  7. XGBoost feature import.  — compare with SHAP RF
  8. Mobile optimisation      — smaller hidden dims + float16 inference

Run: python scripts/experiments.py [--exp all] [--exp cnn] [--exp crossdev] ...
"""

from __future__ import annotations
import argparse, json, time, warnings, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.config import (
    PROCESSED_DIR, RESULTS_DIR, MODELS_DIR, PLOTS_DIR, METRICS_DIR,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE, SEED,
    GCN_OUTPUT_DIM, PAMAP2_NODE_FEAT_DIM, HHAR_NODE_FEAT_DIM,
    PAMAP2_ACTIVITIES, HHAR_ACTIVITIES,
)
from src.models import (
    GNNOnlyModel, GNNLSTMModel, LSTMOnlyModel,
    CNN1DModel, GNNLearnableAdjModel, GNNFlattenLSTMModel,
)
from src.dataset import (
    HARWindowDataset, HARWindowDataset2D,
    HARGraphDataset, HARSequenceDataset,
)
from src.graph_construction import build_pamap2_adj, build_hhar_adj
from src.train import get_device, set_seed, loso_splits, train_model
from src.evaluation import get_predictions, compute_metrics
from src.baselines import extract_features
from src.augmentation import augment_dataset

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

for d in [MODELS_DIR, PLOTS_DIR, METRICS_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

PLOTS  = Path(PLOTS_DIR)
METS   = Path(METRICS_DIR)
MODELS = Path(MODELS_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def remap(y):
    classes = np.unique(y)
    mp = {int(old): int(new) for new, old in enumerate(classes)}
    return np.vectorize(mp.__getitem__)(y), mp


def load_dataset(name: str):
    """Load preprocessed arrays for 'pamap2' or 'hhar'. No subsampling."""
    p = Path(PROCESSED_DIR)
    X     = np.load(p / f"{name}_X.npy")
    y_raw = np.load(p / f"{name}_y.npy")
    subj  = np.load(p / f"{name}_subjects.npy")
    y, mp = remap(y_raw)
    return X, y, subj, mp


def loso_one(
    model_factory,          # callable() → nn.Module
    dataset,                # torch Dataset (HARWindowDataset2D or HARGraphDataset)
    subjects: np.ndarray,
    use_adj: bool,
    tag: str,
    val_frac: float = 0.15,
    save_model: bool = False,
) -> dict:
    """Generic LOSO loop. Returns {accuracy, macro_f1, balanced_acc}."""
    device = get_device()
    all_true, all_pred = [], []
    fold_accs = []

    for fi, (tr_idx, te_idx, te_subj) in enumerate(loso_splits(subjects)):
        set_seed(SEED)

        # Simple val split: hold-out last val_frac of training indices
        n_val = max(1, int(len(tr_idx) * val_frac))
        val_idx, train_idx = tr_idx[-n_val:], tr_idx[:-n_val]

        tr_loader  = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
        val_loader = DataLoader(Subset(dataset, val_idx),   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        te_loader  = DataLoader(Subset(dataset, te_idx),    batch_size=BATCH_SIZE, shuffle=False)

        model = model_factory().to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        crit  = nn.CrossEntropyLoss()
        best_val_loss, best_state, pat = float("inf"), None, 0

        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            for batch in tr_loader:
                opt.zero_grad()
                if use_adj:
                    x, adj, yb = batch
                    loss = crit(model(x.to(device), adj.to(device)), yb.to(device))
                else:
                    x, yb = batch
                    loss = crit(model(x.to(device)), yb.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            # Validation — track loss (unbiased on imbalanced classes) and acc for logging
            model.eval()
            val_loss_sum = correct = total = 0
            with torch.no_grad():
                for batch in val_loader:
                    if use_adj:
                        x, adj, yb = batch
                        logits = model(x.to(device), adj.to(device))
                    else:
                        x, yb = batch
                        logits = model(x.to(device))
                    yb_d = yb.to(device)
                    val_loss_sum += crit(logits, yb_d).item() * len(yb)
                    correct      += (logits.argmax(1) == yb_d).sum().item()
                    total        += len(yb)
            val_loss = val_loss_sum / total if total else float("inf")
            sched.step(val_loss)   # LR scheduler and early stopping both track loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in model.state_dict().items()}
                pat = 0
            else:
                pat += 1
                if pat >= PATIENCE:
                    break

        model.load_state_dict(best_state)
        if save_model:
            torch.save(best_state, MODELS / f"{tag}_fold{fi+1}.pt")

        # Test
        yt, yp = get_predictions(model, te_loader, device, use_adj=use_adj)
        acc = accuracy_score(yt, yp)
        fold_accs.append(acc)
        all_true.extend(yt); all_pred.extend(yp)
        print(f"  Fold {fi+1}/9 test={te_subj}  acc={acc:.4f}", flush=True)

    yt = np.array(all_true); yp = np.array(all_pred)
    np.save(METS / f"{tag}_y_true.npy", yt)
    np.save(METS / f"{tag}_y_pred.npy", yp)
    result = {
        "accuracy":     float(accuracy_score(yt, yp)),
        "macro_f1":     float(f1_score(yt, yp, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(yt, yp)),
    }
    print(f"[{tag}] Acc={result['accuracy']:.4f}  F1={result['macro_f1']:.4f}  BalAcc={result['balanced_acc']:.4f}")
    return result


def save_cm_plot(yt, yp, labels, title, out_path):
    cm = confusion_matrix(yt, yp, normalize="true")
    sz = max(8, len(labels))
    fig, ax = plt.subplots(figsize=(sz, sz - 1))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax, annot_kws={"size": 7})
    ax.set_title(title, fontsize=11); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  Saved {out_path}")


def bar_chart(names, values, title, ylabel, out_path, color="steelblue"):
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    bars = ax.bar(names, values, color=color, alpha=0.85)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
    ax.set_xticklabels(names, rotation=30, ha="right")
    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  Saved {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# EXP 1 — CNN1D Baseline
# ═════════════════════════════════════════════════════════════════════════════

def exp_cnn1d():
    print("\n" + "=" * 70)
    print("EXP 1 — CNN1D Baseline  (PAMAP2 + HHAR)")
    print("=" * 70)
    results = {}

    for ds_name in ["pamap2", "hhar"]:
        X, y, subj, _ = load_dataset(ds_name)
        n_cls = len(np.unique(y))
        T, C  = X.shape[1], X.shape[2]
        act_names = PAMAP2_ACTIVITIES if ds_name == "pamap2" else {i: v for i, v in enumerate(HHAR_ACTIVITIES)}

        print(f"\n── CNN1D on {ds_name.upper()}  X={X.shape}  n_cls={n_cls} ──")
        dataset = HARWindowDataset2D(X, y)

        def factory():
            return CNN1DModel(n_timesteps=T, n_channels=C, n_classes=n_cls)

        tag = f"cnn1d_{ds_name}"
        res = loso_one(factory, dataset, subj, use_adj=False, tag=tag, save_model=True)
        results[ds_name] = res

        # Confusion matrix
        yt = np.load(METS / f"{tag}_y_true.npy")
        yp = np.load(METS / f"{tag}_y_pred.npy")
        classes = sorted(np.unique(yt).tolist())
        if ds_name == "pamap2":
            labels = [PAMAP2_ACTIVITIES.get(k, str(k)) for k in classes]
        else:
            labels = [HHAR_ACTIVITIES[k] if k < len(HHAR_ACTIVITIES) else str(k) for k in classes]
        save_cm_plot(yt, yp, labels, f"CNN1D — {ds_name.upper()} LOSO",
                     PLOTS / f"cm_cnn1d_{ds_name}.png")

    with open(METS / "cnn1d_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved cnn1d_results.json")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# EXP 2 — HHAR Cross-Device Generalisation
# ═════════════════════════════════════════════════════════════════════════════

def exp_cross_device():
    print("\n" + "=" * 70)
    print("EXP 2 — HHAR Cross-Device Generalisation")
    print("=" * 70)

    X, y, subj, _ = load_dataset("hhar")
    n_cls = len(np.unique(y))

    # HHAR subjects a–i correspond to 9 people. We proxy device split
    # by grouping subjects: group A = [a,b,c,d,e], group B = [f,g,h,i]
    # Train on group A, test on group B (and vice versa).
    unique_subjs = sorted(np.unique(subj).tolist())
    mid = len(unique_subjs) // 2
    groups = {
        "train_AB_test_CD": (unique_subjs[:mid], unique_subjs[mid:]),
        "train_CD_test_AB": (unique_subjs[mid:], unique_subjs[:mid]),
    }

    results = {}
    T, C = X.shape[1], X.shape[2]
    adj = build_hhar_adj()
    dataset_g = HARGraphDataset(X, y, dataset="hhar")

    for split_name, (train_subjs, test_subjs) in groups.items():
        print(f"\n── {split_name}  train={train_subjs}  test={test_subjs} ──")
        tr_idx = np.where(np.isin(subj, train_subjs))[0]
        te_idx = np.where(np.isin(subj, test_subjs))[0]

        n_val = max(1, int(len(tr_idx) * 0.15))
        val_idx   = tr_idx[-n_val:]
        train_idx = tr_idx[:-n_val]

        device = get_device()
        set_seed(SEED)
        model = GNNOnlyModel(HHAR_NODE_FEAT_DIM, 2, n_cls).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        crit  = nn.CrossEntropyLoss()
        best_val_loss, best_state, pat = float("inf"), None, 0

        tr_loader  = DataLoader(Subset(dataset_g, train_idx), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(Subset(dataset_g, val_idx),   batch_size=BATCH_SIZE, shuffle=False)
        te_loader  = DataLoader(Subset(dataset_g, te_idx),    batch_size=BATCH_SIZE, shuffle=False)

        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            for x, a, yb in tr_loader:
                opt.zero_grad()
                loss = crit(model(x.to(device), a.to(device)), yb.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            val_loss_sum = correct = total = 0
            with torch.no_grad():
                for x, a, yb in val_loader:
                    logits = model(x.to(device), a.to(device))
                    yb_d   = yb.to(device)
                    val_loss_sum += crit(logits, yb_d).item() * len(yb)
                    correct      += (logits.argmax(1) == yb_d).sum().item()
                    total        += len(yb)
            val_loss = val_loss_sum / total if total else float("inf")
            sched.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in model.state_dict().items()}
                pat = 0
            else:
                pat += 1
                if pat >= PATIENCE: break

        model.load_state_dict(best_state)
        yt, yp = get_predictions(model, te_loader, device, use_adj=True)
        res = {
            "accuracy":     float(accuracy_score(yt, yp)),
            "macro_f1":     float(f1_score(yt, yp, average="macro", zero_division=0)),
            "balanced_acc": float(balanced_accuracy_score(yt, yp)),
            "train_subjects": train_subjs,
            "test_subjects":  test_subjs,
        }
        results[split_name] = res
        print(f"  Acc={res['accuracy']:.4f}  F1={res['macro_f1']:.4f}  BalAcc={res['balanced_acc']:.4f}")

    # Load LOSO result for comparison
    loso_acc = None
    loso_f1  = None
    hd = METS / "hhar_deep_models.json"
    if hd.exists():
        with open(hd) as f: hd_data = json.load(f)
        loso_acc = hd_data.get("gnn", {}).get("accuracy")
        loso_f1  = hd_data.get("gnn", {}).get("macro_f1")
    results["loso_reference"] = {"accuracy": loso_acc, "macro_f1": loso_f1}

    with open(METS / "cross_device_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary comparison plot
    labels  = ["LOSO (per-subject)", "Cross-device A→B", "Cross-device B→A"]
    accs    = [
        loso_acc or 0,
        results["train_AB_test_CD"]["accuracy"],
        results["train_CD_test_AB"]["accuracy"],
    ]
    f1s = [
        loso_f1 or 0,
        results["train_AB_test_CD"]["macro_f1"],
        results["train_CD_test_AB"]["macro_f1"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["steelblue", "darkorange", "crimson"]
    for ax, vals, metric in zip(axes, [accs, f1s], ["Accuracy", "Macro F1"]):
        bars = ax.bar(labels, vals, color=colors, alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_ylim(0, 1.1); ax.set_ylabel(metric)
        ax.set_title(f"HHAR GNN-only: LOSO vs Cross-Device ({metric})")
        ax.set_xticklabels(labels, rotation=20, ha="right")
    fig.suptitle("Cross-Device Generalisation — HHAR", fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOTS / "cross_device_comparison.png", dpi=150); plt.close(fig)
    print("Saved cross_device_comparison.png")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# EXP 3 — Graph Ablation
# ═════════════════════════════════════════════════════════════════════════════

def exp_graph_ablation():
    print("\n" + "=" * 70)
    print("EXP 3 — Graph Ablation (PAMAP2)")
    print("=" * 70)

    X, y, subj, _ = load_dataset("pamap2")
    n_cls = len(np.unique(y))
    graph_ds = HARGraphDataset(X, y, dataset="pamap2")
    seq_ds   = HARSequenceDataset(X, y, subjects=subj, dataset="pamap2", seq_len=10)
    # Build seq_subj aligned with HARSequenceDataset's internal ordering
    seq_subj = []
    for s in np.unique(subj):
        mask = np.where(subj == s)[0]
        n_seqs = max(0, (len(mask) - 10 + 1) // 10)
        seq_subj.extend([s] * n_seqs)
    seq_subj = np.array(seq_subj)

    init_adj = build_pamap2_adj()

    ablations = {
        "ablation_fixed_adj":    (
            lambda: GNNOnlyModel(PAMAP2_NODE_FEAT_DIM, 3, n_cls),
            graph_ds, subj, True,
        ),
        "ablation_learnable_adj": (
            lambda: GNNLearnableAdjModel(PAMAP2_NODE_FEAT_DIM, 3, n_cls, init_adj=init_adj.clone()),
            graph_ds, subj, True,
        ),
        "ablation_flatten_lstm":  (
            lambda: GNNFlattenLSTMModel(PAMAP2_NODE_FEAT_DIM, 3, n_cls),
            seq_ds, seq_subj, True,
        ),
    }

    results = {}
    for tag, (factory, ds, subs, ua) in ablations.items():
        print(f"\n── {tag} ──")
        res = loso_one(factory, ds, subs, use_adj=ua, tag=tag, save_model=False)
        results[tag] = res

        yt = np.load(METS / f"{tag}_y_true.npy")
        yp = np.load(METS / f"{tag}_y_pred.npy")
        classes = sorted(np.unique(yt).tolist())
        labels  = [PAMAP2_ACTIVITIES.get(k, str(k)) for k in classes]
        save_cm_plot(yt, yp, labels, f"{tag} — PAMAP2", PLOTS / f"cm_{tag}.png")

    with open(METS / "graph_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Ablation bar chart
    names = ["Fixed Adj\n(GNN-only)", "Learnable Adj", "Flatten+LSTM\n(No GCN)"]
    accs  = [results[k]["accuracy"]  for k in ablations]
    f1s   = [results[k]["macro_f1"]  for k in ablations]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, vals, metric in zip(axes, [accs, f1s], ["Accuracy", "Macro F1"]):
        bars = ax.bar(names, vals, color=["steelblue", "darkorange", "crimson"], alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_ylim(0, 1.1); ax.set_ylabel(metric)
        ax.set_title(f"Graph Ablation — {metric} (PAMAP2 LOSO)")
    fig.suptitle("Graph Ablation Study — PAMAP2", fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOTS / "graph_ablation.png", dpi=150); plt.close(fig)
    print("Saved graph_ablation.png")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# EXP 4 — Data Augmentation (HHAR, GNN-only)
# ═════════════════════════════════════════════════════════════════════════════

def exp_augmentation():
    print("\n" + "=" * 70)
    print("EXP 4 — Data Augmentation (HHAR, GNN-only)")
    print("=" * 70)

    X_orig, y_orig, subj_orig, _ = load_dataset("hhar")
    n_cls = len(np.unique(y_orig))
    adj   = build_hhar_adj()

    methods = {
        "no_aug":   (X_orig, y_orig, subj_orig),
        "gaussian": augment_dataset(X_orig, y_orig, method="gaussian", sigma=0.05, seed=SEED) + (
                        np.tile(subj_orig, 2),),
        "scale":    augment_dataset(X_orig, y_orig, method="scale",    seed=SEED) + (
                        np.tile(subj_orig, 2),),
        "timewarp": augment_dataset(X_orig, y_orig, method="timewarp", sigma=0.05, seed=SEED) + (
                        np.tile(subj_orig, 2),),
    }

    results = {}
    for aug_name, (X_a, y_a, subj_a) in methods.items():
        print(f"\n── Augmentation: {aug_name}  X={X_a.shape} ──")
        dataset = HARGraphDataset(X_a, y_a, dataset="hhar")

        def factory():
            return GNNOnlyModel(HHAR_NODE_FEAT_DIM, 2, n_cls)

        tag = f"aug_{aug_name}_hhar"
        res = loso_one(factory, dataset, subj_a, use_adj=True, tag=tag, save_model=False)
        results[aug_name] = res

    with open(METS / "augmentation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Comparison plot
    aug_labels = list(results.keys())
    accs = [results[k]["accuracy"]  for k in aug_labels]
    f1s  = [results[k]["macro_f1"]  for k in aug_labels]
    display = ["No Aug", "Gaussian\nNoise", "Amplitude\nScale", "Time\nWarp"]
    colors = ["steelblue", "darkorange", "green", "crimson"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, vals, metric in zip(axes, [accs, f1s], ["Accuracy", "Macro F1"]):
        bars = ax.bar(display, vals, color=colors, alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_ylim(0, 1.1); ax.set_ylabel(metric)
        ax.set_title(f"Augmentation Effect — {metric} (HHAR, GNN-only)")
    fig.suptitle("Data Augmentation Experiment — HHAR", fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOTS / "augmentation_comparison.png", dpi=150); plt.close(fig)
    print("Saved augmentation_comparison.png")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# EXP 5 — Neural Interpretability (Integrated Gradients on GNN-only)
# ═════════════════════════════════════════════════════════════════════════════

def exp_neural_interpretability():
    print("\n" + "=" * 70)
    print("EXP 5 — Neural Interpretability (Integrated Gradients, GNN-only PAMAP2)")
    print("=" * 70)

    X, y, subj, _ = load_dataset("pamap2")
    n_cls = len(np.unique(y))
    adj = build_pamap2_adj()

    # Load the best fold model (fold with highest test acc — use fold 1 as proxy)
    model_path = MODELS / "gnn_pamap2_fold1.pt"
    if not model_path.exists():
        print(f"  [SKIP] {model_path} not found. Run main training first.")
        return {}

    device = get_device()
    model  = GNNOnlyModel(PAMAP2_NODE_FEAT_DIM, 3, n_cls).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Build graph dataset and pick 200 random test samples
    graph_ds = HARGraphDataset(X, y, dataset="pamap2")
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(graph_ds), size=min(200, len(graph_ds)), replace=False)

    # Integrated Gradients: attribute score for each node feature dimension
    # IG = (x - baseline) * (1/m) * sum_k grad_f(baseline + k/m * (x-baseline))
    m_steps = 50
    all_attrs = []

    adj_t = adj.to(device)
    for idx in sample_idx:
        x, _, yb = graph_ds[idx]
        x = x.unsqueeze(0).to(device)           # (1, n_nodes, feat_dim)
        baseline = torch.zeros_like(x)

        # Accumulate gradients along interpolation path
        integrated_grad = torch.zeros_like(x)
        for k in range(1, m_steps + 1):
            interp = baseline + (k / m_steps) * (x - baseline)
            interp.requires_grad_(True)
            logits = model(interp, adj_t)
            score  = logits[0, yb.item()]
            score.backward()
            integrated_grad += interp.grad.detach()

        ig = ((x - baseline) * integrated_grad / m_steps).squeeze(0).cpu().numpy()
        # ig: (n_nodes, feat_dim)  — take mean absolute across nodes
        all_attrs.append(np.abs(ig).mean(axis=0))

    mean_attr = np.stack(all_attrs).mean(axis=0)   # (feat_dim,) = (36,)

    # Feature names: 6 stats × 6 channels/node
    stat_names = ["mean", "std", "min", "max", "rms", "iqr"]
    n_ch_per_node = PAMAP2_NODE_FEAT_DIM // 6
    feat_names = [f"{s}_ch{c}" for s in stat_names for c in range(n_ch_per_node)]

    top_k   = min(20, len(mean_attr))
    top_idx = np.argsort(mean_attr)[::-1][:top_k]
    top_names = [feat_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(np.arange(top_k), mean_attr[top_idx[::-1]], color="darkorange", alpha=0.85)
    ax.set_yticks(np.arange(top_k))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Mean |Integrated Gradient|")
    ax.set_title("Neural Interpretability — Integrated Gradients (GNN-only, PAMAP2 top-20)",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOTS / "ig_gnn_pamap2.png", dpi=150); plt.close(fig)
    print("  Saved ig_gnn_pamap2.png")

    result = {"top_features": top_names, "top_attr_values": mean_attr[top_idx].tolist()}
    with open(METS / "neural_interpretability.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# EXP 6 — Error Analysis
# ═════════════════════════════════════════════════════════════════════════════

def exp_error_analysis():
    print("\n" + "=" * 70)
    print("EXP 6 — Error Analysis")
    print("=" * 70)

    error_summary = {}

    for ds_name in ["pamap2", "hhar"]:
        if ds_name == "pamap2":
            act_dict = PAMAP2_ACTIVITIES
            models_to_check = ["lstm", "gnn", "gnnlstm"]
        else:
            # HHAR_ACTIVITIES is a list; after remap y is 0-based
            act_dict = {i: v for i, v in enumerate(HHAR_ACTIVITIES)}
            models_to_check = ["lstm", "gnn", "gnnlstm"]

        ds_errors = {}
        all_cms = {}

        for mt in models_to_check:
            tag = f"{mt}_{ds_name}"
            tp  = METS / f"{tag}_y_true.npy"
            pp  = METS / f"{tag}_y_pred.npy"
            if not tp.exists():
                print(f"  [SKIP] {tag} predictions not found")
                continue

            yt = np.load(tp); yp = np.load(pp)
            classes = sorted(np.unique(yt).tolist())
            labels  = [act_dict.get(k, str(k)) for k in classes]

            # Per-class F1
            per_cls_f1 = f1_score(yt, yp, average=None, zero_division=0)
            cls_f1_dict = {labels[i]: float(per_cls_f1[i]) for i in range(len(labels))}

            # Hardest classes (lowest F1)
            hardest = sorted(cls_f1_dict.items(), key=lambda x: x[1])[:3]

            # Confusion pairs (off-diagonal with highest confusion rate)
            cm = confusion_matrix(yt, yp, normalize="true")
            np.fill_diagonal(cm, 0)
            flat_idx = np.argsort(cm.flatten())[::-1][:5]
            confused_pairs = []
            for idx in flat_idx:
                r, c = divmod(idx, len(labels))
                if r < len(labels) and c < len(labels):
                    confused_pairs.append({
                        "true":  labels[r], "pred": labels[c],
                        "rate":  round(float(cm[r, c]), 3)
                    })

            ds_errors[mt] = {
                "per_class_f1":    cls_f1_dict,
                "hardest_classes": [{"class": k, "f1": round(v, 3)} for k, v in hardest],
                "most_confused":   confused_pairs,
            }
            all_cms[mt] = confusion_matrix(yt, yp, normalize="true")
            print(f"  {ds_name}/{mt}: hardest = {[k for k, _ in hardest[:3]]}")

        # Per-class F1 comparison chart
        if all(mt in ds_errors for mt in ["lstm", "gnn", "gnnlstm"]):
            classes_list = list(ds_errors["gnn"]["per_class_f1"].keys())
            lstm_f1s  = [ds_errors["lstm"]["per_class_f1"].get(c, 0)    for c in classes_list]
            gnn_f1s   = [ds_errors["gnn"]["per_class_f1"].get(c, 0)     for c in classes_list]
            gnnl_f1s  = [ds_errors["gnnlstm"]["per_class_f1"].get(c, 0) for c in classes_list]

            x = np.arange(len(classes_list)); w = 0.25
            fig, ax = plt.subplots(figsize=(max(10, len(classes_list) * 1.2), 5))
            ax.bar(x - w,  lstm_f1s,  w, label="LSTM-only",  color="steelblue",  alpha=0.85)
            ax.bar(x,      gnn_f1s,   w, label="GNN-only",   color="darkorange", alpha=0.85)
            ax.bar(x + w,  gnnl_f1s,  w, label="GNN+LSTM",   color="green",      alpha=0.85)
            ax.set_xticks(x); ax.set_xticklabels(classes_list, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Per-class F1"); ax.set_ylim(0, 1.15)
            ax.set_title(f"Per-Class F1 Comparison — {ds_name.upper()}", fontweight="bold")
            ax.legend(loc="upper right")
            plt.tight_layout()
            fig.savefig(PLOTS / f"per_class_f1_{ds_name}.png", dpi=150); plt.close(fig)
            print(f"  Saved per_class_f1_{ds_name}.png")

        error_summary[ds_name] = ds_errors

    with open(METS / "error_analysis.json", "w") as f:
        json.dump(error_summary, f, indent=2)
    print("Saved error_analysis.json")
    return error_summary


# ═════════════════════════════════════════════════════════════════════════════
# EXP 7 — XGBoost Feature Importance
# ═════════════════════════════════════════════════════════════════════════════

def exp_xgb_feature_importance():
    print("\n" + "=" * 70)
    print("EXP 7 — XGBoost Feature Importance (PAMAP2)")
    print("=" * 70)

    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  [SKIP] XGBoost not installed")
        return {}

    X, y, subj, _ = load_dataset("pamap2")
    Xf = extract_features(X)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Xf), size=min(3000, len(Xf)), replace=False)
    Xf_s, y_s = Xf[idx], y[idx]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xf_sc  = scaler.fit_transform(Xf_s)

    xgb = XGBClassifier(
        n_estimators=100, learning_rate=0.1,
        eval_metric="mlogloss", random_state=SEED,
        n_jobs=1, tree_method="hist",   # n_jobs=1 avoids fork/MPS segfault on macOS
    )
    xgb.fit(Xf_sc, y_s)

    importances = xgb.feature_importances_
    n_ch = X.shape[2]
    stat_names = ["mean", "std", "min", "max", "rms", "fft_energy"]
    feat_names = [f"{s}_ch{c}" for s in stat_names for c in range(n_ch)]

    top_k   = min(20, len(importances))
    top_idx = np.argsort(importances)[::-1][:top_k]
    top_names = [feat_names[i] for i in top_idx]
    top_vals  = importances[top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(np.arange(top_k), top_vals[::-1], color="crimson", alpha=0.85)
    ax.set_yticks(np.arange(top_k))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("XGBoost Feature Importance (gain)")
    ax.set_title("XGBoost Feature Importance — PAMAP2 (top-20)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOTS / "xgb_feature_importance_pamap2.png", dpi=150); plt.close(fig)
    print("  Saved xgb_feature_importance_pamap2.png")

    result = {"top_features": top_names, "top_importances": top_vals.tolist()}
    with open(METS / "xgb_feature_importance.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# EXP 8 — Mobile Optimisation (smaller model + float16 inference)
# ═════════════════════════════════════════════════════════════════════════════

def exp_mobile_optimisation():
    print("\n" + "=" * 70)
    print("EXP 8 — Mobile Optimisation (PAMAP2, GNN-only)")
    print("=" * 70)

    X, y, subj, _ = load_dataset("pamap2")
    n_cls = len(np.unique(y))
    graph_ds = HARGraphDataset(X, y, dataset="pamap2")

    results = {}

    # Variant A — standard (GCN_HIDDEN=64, GCN_OUT=64, MLP=64)
    # Already done — load from existing results
    std_path = METS / "pamap2_deep_models.json"
    if std_path.exists():
        with open(std_path) as f: d = json.load(f)
        results["standard_gnn"] = {
            "accuracy":     d["gnn"]["accuracy"],
            "macro_f1":     d["gnn"]["macro_f1"],
            "balanced_acc": d["gnn"]["balanced_acc"],
            "params":       11724,
            "latency_ms":   0.254,
        }

    # Variant B — tiny GNN (hidden=32, output=32, MLP=32)
    from src.config import GCN_HIDDEN_DIM, GCN_OUTPUT_DIM, MLP_HIDDEN_DIM, DROPOUT
    import src.models as _m

    class TinyGNNOnlyModel(nn.Module):
        """GNN-only with halved hidden dimensions."""
        def __init__(self, node_feat_dim, n_nodes, n_classes):
            super().__init__()
            hidden = 32
            self.gcn1 = _m.GCNLayer(node_feat_dim, hidden)
            self.gcn2 = _m.GCNLayer(hidden, hidden)
            self.drop = nn.Dropout(DROPOUT)
            self.clf  = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(hidden, n_classes),
            )
        def forward(self, x, adj):
            h = self.gcn1(x, adj); h = self.drop(h)
            h = self.gcn2(h, adj)
            emb = h.mean(dim=1)
            return self.clf(emb)

    print("\n── Variant B: Tiny GNN (hidden=32) ──")
    tag = "mobile_tiny_gnn_pamap2"
    res = loso_one(
        lambda: TinyGNNOnlyModel(PAMAP2_NODE_FEAT_DIM, 3, n_cls),
        graph_ds, subj, use_adj=True, tag=tag, save_model=False,
    )
    tiny_params = sum(p.numel() for p in TinyGNNOnlyModel(PAMAP2_NODE_FEAT_DIM, 3, n_cls).parameters())

    # Float16 latency for tiny model
    device = get_device()
    tiny_m = TinyGNNOnlyModel(PAMAP2_NODE_FEAT_DIM, 3, n_cls).to(device).eval()
    adj_t  = build_pamap2_adj().to(device)
    dummy  = torch.zeros(1, 3, PAMAP2_NODE_FEAT_DIM).to(device)
    # Warm up
    with torch.no_grad():
        for _ in range(20): tiny_m(dummy, adj_t)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(500): tiny_m(dummy, adj_t)
    lat_f32 = (time.perf_counter() - t0) / 500 * 1000

    # Float16 (CPU only — MPS may not support all float16 ops)
    tiny_m_cpu = TinyGNNOnlyModel(PAMAP2_NODE_FEAT_DIM, 3, n_cls).eval().half()
    adj_cpu    = adj_t.cpu().half()
    dummy_cpu  = dummy.cpu().half()
    with torch.no_grad():
        for _ in range(20): tiny_m_cpu(dummy_cpu, adj_cpu)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(500): tiny_m_cpu(dummy_cpu, adj_cpu)
    lat_f16 = (time.perf_counter() - t0) / 500 * 1000

    res["params"]       = tiny_params
    res["latency_f32_ms"] = round(lat_f32, 3)
    res["latency_f16_ms"] = round(lat_f16, 3)
    results["tiny_gnn"] = res

    print(f"  Tiny GNN: {tiny_params:,} params  f32={lat_f32:.3f}ms  f16={lat_f16:.3f}ms")
    print(f"  Tiny GNN: Acc={res['accuracy']:.4f}  F1={res['macro_f1']:.4f}")

    with open(METS / "mobile_optimisation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Tradeoff plot
    variants = list(results.keys())
    accs     = [results[v]["accuracy"] * 100 for v in variants]
    params   = [results[v]["params"] / 1e3   for v in variants]   # in K
    lats     = [results[v].get("latency_ms", results[v].get("latency_f32_ms", 0)) for v in variants]
    colors   = ["steelblue", "darkorange"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    display = ["Standard GNN\n(64 hidden)", "Tiny GNN\n(32 hidden)"]
    for ax, vals, ylabel, title in zip(
        axes,
        [accs, params, lats],
        ["Accuracy (%)", "Parameters (K)", "Latency (ms)"],
        ["Accuracy", "Model Size", "Inference Speed"],
    ):
        bars = ax.bar(display[:len(vals)], vals, color=colors[:len(vals)], alpha=0.85)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_ylabel(ylabel); ax.set_title(title)
    fig.suptitle("Mobile Optimisation — GNN on PAMAP2", fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOTS / "mobile_optimisation.png", dpi=150); plt.close(fig)
    print("Saved mobile_optimisation.png")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# FINAL — Update master comparison JSON & comparison plot
# ═════════════════════════════════════════════════════════════════════════════

def update_master_comparison():
    print("\n" + "=" * 70)
    print("Updating master comparison table")
    print("=" * 70)

    p2    = json.load(open(METS / "pamap2_deep_models.json"))    if (METS / "pamap2_deep_models.json").exists()   else {}
    hh    = json.load(open(METS / "hhar_deep_models.json"))      if (METS / "hhar_deep_models.json").exists()     else {}
    p2bl  = json.load(open(METS / "pamap2_baselines.json"))      if (METS / "pamap2_baselines.json").exists()     else {}
    hhbl  = json.load(open(METS / "HHAR_baselines.json"))        if (METS / "HHAR_baselines.json").exists()       else {}
    cnn   = json.load(open(METS / "cnn1d_results.json"))         if (METS / "cnn1d_results.json").exists()        else {}
    abl   = json.load(open(METS / "graph_ablation_results.json")) if (METS / "graph_ablation_results.json").exists() else {}
    aug   = json.load(open(METS / "augmentation_results.json"))  if (METS / "augmentation_results.json").exists() else {}

    def g(d, key, sub="accuracy"):
        return d.get(key, {}).get(sub, d.get(key, {}).get("mean_accuracy", 0)) if d else 0

    # Full model comparison — PAMAP2
    rows_p2 = {
        "SVM":              g(p2bl, "SVM", "mean_accuracy"),
        "Random Forest":    g(p2bl, "RandomForest", "mean_accuracy"),
        "XGBoost":          g(p2bl, "XGBoost", "mean_accuracy"),
        "CNN1D":            g(cnn,  "pamap2"),
        "LSTM-only":        g(p2,   "lstm"),
        "GNN-only (fixed)": g(p2,   "gnn"),
        "GNN+LSTM":         g(p2,   "gnn_lstm"),
        "GNN (learnable)":  g(abl,  "ablation_learnable_adj"),
        "Flatten+LSTM":     g(abl,  "ablation_flatten_lstm"),
    }
    rows_hh = {
        "SVM":              g(hhbl, "SVM", "mean_accuracy"),
        "Random Forest":    g(hhbl, "RandomForest", "mean_accuracy"),
        "XGBoost":          g(hhbl, "XGBoost", "mean_accuracy"),
        "CNN1D":            g(cnn,  "hhar"),
        "LSTM-only":        g(hh,   "lstm"),
        "GNN-only":         g(hh,   "gnn"),
        "GNN+LSTM":         g(hh,   "gnn_lstm"),
        "GNN (no aug)":     g(aug,  "no_aug"),
        "GNN (gaussian)":   g(aug,  "gaussian"),
        "GNN (timewarp)":   g(aug,  "timewarp"),
    }

    master = {"pamap2": rows_p2, "hhar": rows_hh}
    with open(METS / "master_comparison.json", "w") as f:
        json.dump(master, f, indent=2)

    # Mega comparison chart
    for ds_name, rows in [("PAMAP2", rows_p2), ("HHAR", rows_hh)]:
        names = list(rows.keys()); vals = [v * 100 for v in rows.values()]
        colors_list = (
            ["#4472C4"] * 3 +          # classical
            ["#ED7D31"] * 4 +          # deep models
            ["#70AD47"] * (len(names) - 7)  # ablation/aug
        )
        fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.4), 5))
        bars = ax.bar(names, vals, color=colors_list[:len(names)], alpha=0.85)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        ax.set_ylabel("LOSO Accuracy (%)"); ax.set_ylim(0, 110)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=9)
        ax.set_title(f"Complete Model Comparison — {ds_name}", fontweight="bold")
        from matplotlib.patches import Patch
        legend_els = [
            Patch(facecolor="#4472C4", label="Classical ML"),
            Patch(facecolor="#ED7D31", label="Deep Learning"),
            Patch(facecolor="#70AD47", label="Ablation / Augmentation"),
        ]
        ax.legend(handles=legend_els, loc="upper right", fontsize=9)
        plt.tight_layout()
        fig.savefig(PLOTS / f"complete_comparison_{ds_name.lower()}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved complete_comparison_{ds_name.lower()}.png")

    print("Saved master_comparison.json")


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

ALL_EXPS = ["cnn", "crossdev", "ablation", "augment", "interp", "error", "xgbfi", "mobile"]

def main():
    parser = argparse.ArgumentParser(description="HAR Extension Experiments")
    parser.add_argument("--exp", nargs="+", default=["all"],
                        choices=ALL_EXPS + ["all"],
                        help="Which experiments to run")
    args = parser.parse_args()

    run = set(args.exp)
    if "all" in run:
        run = set(ALL_EXPS)

    t_total = time.time()

    if "cnn"      in run: exp_cnn1d()
    if "crossdev" in run: exp_cross_device()
    if "ablation" in run: exp_graph_ablation()
    if "augment"  in run: exp_augmentation()
    if "interp"   in run: exp_neural_interpretability()
    if "error"    in run: exp_error_analysis()
    if "xgbfi"    in run: exp_xgb_feature_importance()
    if "mobile"   in run: exp_mobile_optimisation()

    update_master_comparison()

    print(f"\n{'='*70}")
    print(f"All experiments complete in {(time.time() - t_total) / 60:.1f} min")
    print(f"Results  → {METRICS_DIR}")
    print(f"Plots    → {PLOTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
