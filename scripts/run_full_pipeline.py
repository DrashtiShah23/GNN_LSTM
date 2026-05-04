"""
Full end-to-end pipeline — DATA 245 HAR project
================================================
Runs full LOSO evaluation for:
  1. PAMAP2 — LSTM-only, GNN-only, GNN+LSTM
  2. HHAR   — LSTM-only, GNN-only, GNN+LSTM  (if data present)

Then generates all final plots:
  - Confusion matrices (normalised)
  - Model comparison (Acc + F1)
  - Cross-dataset comparison
  - SHAP feature importance
  - Model profiling (params + latency)

Usage:
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --models gnn_lstm --gnn-lstm-tuned
    # Re-run only GNN+LSTM with tuned hyperparameters; merges into pamap2_deep_models.json
    # / hhar_deep_models.json so LSTM/GNN rows and plots stay intact.
"""

from __future__ import annotations

import argparse
import sys
import json
import time
import copy
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (
    PROCESSED_DIR, RESULTS_DIR, MODELS_DIR, PLOTS_DIR, METRICS_DIR,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE, SEED,
    GCN_OUTPUT_DIM, PAMAP2_NODE_FEAT_DIM, HHAR_NODE_FEAT_DIM,
    PAMAP2_ACTIVITIES,
)
from src.models import GNNLSTMModel, LSTMOnlyModel, GNNOnlyModel
from src.dataset import HARWindowDataset, HARGraphDataset, HARSequenceDataset
from src.graph_construction import build_pamap2_adj, build_hhar_adj
from src.train import get_device, loso_splits, set_seed

warnings.filterwarnings("ignore")

# ── directories ───────────────────────────────────────────────────────────────
for d in [MODELS_DIR, PLOTS_DIR, METRICS_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

# =============================================================================
# Helpers
# =============================================================================

def remap_labels(y: np.ndarray) -> tuple[np.ndarray, dict]:
    """Remap arbitrary label set to 0..N-1 and return (remapped_y, mapping)."""
    classes = np.unique(y)
    mapping = {int(old): int(new) for new, old in enumerate(classes)}
    return np.vectorize(mapping.__getitem__)(y), mapping


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    use_adj: bool,
    device: torch.device,
    verbose: bool = True,
    *,
    lr: float | None = None,
    weight_decay: float | None = None,
    patience: int | None = None,
) -> nn.Module:
    """Train with early stopping; return best-weight model."""
    lr_ = LEARNING_RATE if lr is None else lr
    wd_ = WEIGHT_DECAY if weight_decay is None else weight_decay
    pat_max = PATIENCE if patience is None else patience
    opt = torch.optim.Adam(model.parameters(), lr=lr_, weight_decay=wd_)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit = nn.CrossEntropyLoss()
    best_acc, best_state, patience_cnt = 0.0, None, 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        t0 = time.time()
        for batch in train_loader:
            opt.zero_grad()
            if use_adj:
                x, adj, y = batch
                x, adj, y = x.to(device), adj.to(device), y.to(device)
                logits = model(x, adj)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if use_adj:
                    x, adj, y = batch
                    x, adj, y = x.to(device), adj.to(device), y.to(device)
                    logits = model(x, adj)
                else:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                val_loss += crit(logits, y).item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                total += len(y)
        val_acc = correct / total
        val_loss /= total
        sched.step(val_loss)

        if verbose:
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | Val Acc: {val_acc:.4f} | "
                  f"Time: {time.time()-t0:.1f}s", flush=True)

        if val_acc > best_acc:
            best_acc, best_state, patience_cnt = val_acc, copy.deepcopy(model.state_dict()), 0
        else:
            patience_cnt += 1
            if patience_cnt >= pat_max:
                if verbose:
                    print(f"  Early stop @ epoch {epoch}. Best val acc: {best_acc:.4f}", flush=True)
                break

    if best_state:
        model.load_state_dict(best_state)
    if verbose:
        print(f"  Best val acc: {best_acc:.4f}", flush=True)
    return model


def loso_deep(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    model_type: str,      # "lstm", "gnn", "gnn_lstm"
    dataset: str,         # "pamap2" | "hhar"
    n_classes: int,
    node_feat_dim: int,
    n_nodes: int,
    adj_builder,
    tag: str,
    *,
    gnn_lstm_kwargs: dict | None = None,
    gnn_lstm_lr: float | None = None,
    gnn_lstm_weight_decay: float | None = None,
    gnn_lstm_patience: int | None = None,
) -> dict:
    """
    Full LOSO evaluation for one deep model on one dataset.
    Returns dict with accuracy, macro_f1, balanced_acc, y_true, y_pred.
    """
    device = get_device()
    set_seed(SEED)
    all_true, all_pred = [], []
    fold_rows: list[dict] = []
    unique_subjs = np.unique(subjects)
    n_folds = len(unique_subjs)
    adj_fixed = adj_builder().to(device)

    for fold_i, test_subj in enumerate(unique_subjs, 1):
        print(f"\n── Fold {fold_i}/{n_folds}: test subject={test_subj} ──", flush=True)
        train_mask = subjects != test_subj
        test_mask  = subjects == test_subj
        X_tr, y_tr, s_tr = X[train_mask], y[train_mask], subjects[train_mask]
        X_te, y_te        = X[test_mask],  y[test_mask]

        # ── subject-based val split: hold out one training subject as val ─────
        # Pick the training subject with fewest windows as val (least data loss)
        train_subjs = np.unique(s_tr)
        subj_counts = {s: np.sum(s_tr == s) for s in train_subjs}
        val_subj    = min(subj_counts, key=subj_counts.__getitem__)
        val_mask_loc   = s_tr == val_subj
        train_mask_loc = s_tr != val_subj
        X_val, y_val, s_val = X_tr[val_mask_loc],   y_tr[val_mask_loc],   s_tr[val_mask_loc]
        X_tr2, y_tr2, s_tr2 = X_tr[train_mask_loc], y_tr[train_mask_loc], s_tr[train_mask_loc]

        use_adj = model_type in ("gnn", "gnn_lstm")

        if model_type == "lstm":
            tr_loader  = DataLoader(HARWindowDataset(X_tr2, y_tr2), BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(HARWindowDataset(X_val, y_val), BATCH_SIZE, shuffle=False)
            te_loader  = DataLoader(HARWindowDataset(X_te,  y_te),  BATCH_SIZE, shuffle=False)
        elif model_type == "gnn":
            tr_loader  = DataLoader(HARGraphDataset(X_tr2, y_tr2, dataset=dataset), BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(HARGraphDataset(X_val, y_val, dataset=dataset), BATCH_SIZE, shuffle=False)
            te_loader  = DataLoader(HARGraphDataset(X_te,  y_te,  dataset=dataset), BATCH_SIZE, shuffle=False)
        else:  # gnn_lstm
            ds_tr_s  = HARSequenceDataset(X_tr2, y_tr2, subjects=s_tr2, dataset=dataset)
            ds_val_s = HARSequenceDataset(X_val, y_val, subjects=s_val,  dataset=dataset)
            ds_te_s  = HARSequenceDataset(X_te,  y_te,  dataset=dataset)
            if len(ds_tr_s) == 0:
                ds_tr_s = HARSequenceDataset(X_tr, y_tr, subjects=s_tr, dataset=dataset)
            if len(ds_val_s) == 0:
                ds_val_s = ds_tr_s
            tr_loader  = DataLoader(ds_tr_s,  BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(ds_val_s, BATCH_SIZE, shuffle=False)
            te_loader  = DataLoader(ds_te_s,  BATCH_SIZE, shuffle=False)

        # ── build model ───────────────────────────────────────────────────────
        if model_type == "lstm":
            model = LSTMOnlyModel(
                input_dim = X_tr.shape[1] * X_tr.shape[2],
                n_classes = n_classes,
            ).to(device)
        elif model_type == "gnn":
            model = GNNOnlyModel(
                node_feat_dim = node_feat_dim,
                n_nodes       = n_nodes,
                n_classes     = n_classes,
            ).to(device)
        else:
            gl_kw = dict(gnn_lstm_kwargs or {})
            model = GNNLSTMModel(
                node_feat_dim=node_feat_dim,
                n_nodes=n_nodes,
                n_classes=n_classes,
                **gl_kw,
            ).to(device)

        # ── train ─────────────────────────────────────────────────────────────
        if model_type == "gnn_lstm":
            model = train_one_fold(
                model, tr_loader, val_loader, use_adj, device, verbose=True,
                lr=gnn_lstm_lr,
                weight_decay=gnn_lstm_weight_decay,
                patience=gnn_lstm_patience,
            )
        else:
            model = train_one_fold(model, tr_loader, val_loader, use_adj, device, verbose=True)

        # Save fold model
        fold_path = Path(MODELS_DIR) / f"{tag}_fold{fold_i}.pt"
        torch.save(model.state_dict(), fold_path)

        # ── test ──────────────────────────────────────────────────────────────
        model.eval()
        fold_true, fold_pred = [], []
        with torch.no_grad():
            for batch in te_loader:
                if use_adj:
                    x, adj, yb = batch
                    x, adj, yb = x.to(device), adj.to(device), yb.to(device)
                    logits = model(x, adj)
                else:
                    x, yb = batch
                    x, yb = x.to(device), yb.to(device)
                    logits = model(x)
                fold_pred.extend(logits.argmax(1).cpu().numpy().tolist())
                fold_true.extend(yb.cpu().numpy().tolist())

        fold_acc = accuracy_score(fold_true, fold_pred)
        fold_f1 = f1_score(fold_true, fold_pred, average="macro", zero_division=0)
        print(f"  Fold test acc: {fold_acc:.4f}", flush=True)
        all_true.extend(fold_true)
        all_pred.extend(fold_pred)
        fold_rows.append(
            {
                "fold": fold_i,
                "test_subject": int(test_subj) if np.issubdtype(type(test_subj), np.integer) else str(test_subj),
                "accuracy": float(fold_acc),
                "macro_f1": float(fold_f1),
            }
        )

    acc   = accuracy_score(all_true, all_pred)
    f1    = f1_score(all_true, all_pred, average="macro", zero_division=0)
    bal   = balanced_accuracy_score(all_true, all_pred)
    fold_accs = [r["accuracy"] for r in fold_rows]
    fold_f1s = [r["macro_f1"] for r in fold_rows]
    acc_std = float(np.std(fold_accs)) if fold_accs else 0.0
    f1_std = float(np.std(fold_f1s)) if fold_f1s else 0.0
    print(f"\n[{tag}] LOSO Acc={acc:.4f} ± {acc_std:.4f}  F1={f1:.4f} ± {f1_std:.4f}  BalAcc={bal:.4f}", flush=True)

    # Save predictions
    np.save(Path(METRICS_DIR) / f"{tag}_y_true.npy", np.array(all_true))
    np.save(Path(METRICS_DIR) / f"{tag}_y_pred.npy", np.array(all_pred))

    out = {
        "accuracy": acc,
        "macro_f1": f1,
        "balanced_acc": bal,
        "accuracy_std": acc_std,
        "macro_f1_std": f1_std,
        "folds": fold_rows,
    }
    if model_type == "gnn_lstm" and gnn_lstm_kwargs is not None:
        out["gnn_lstm_hparams"] = dict(gnn_lstm_kwargs)
    if model_type == "gnn_lstm" and (
        gnn_lstm_lr is not None
        or gnn_lstm_weight_decay is not None
        or gnn_lstm_patience is not None
    ):
        out["gnn_lstm_train_hparams"] = {
            "lr": gnn_lstm_lr,
            "weight_decay": gnn_lstm_weight_decay,
            "patience": gnn_lstm_patience,
        }
    return out


# =============================================================================
# Dataset loader
# =============================================================================

def load_dataset(name: str):
    """Load preprocessed npy files. Returns X, y, subjects (label-remapped)."""
    base = Path(PROCESSED_DIR)
    X = np.load(base / f"{name}_X.npy")
    y = np.load(base / f"{name}_y.npy")
    subjects = np.load(base / f"{name}_subjects.npy")
    y, mapping = remap_labels(y)
    print(f"[{name}] X={X.shape}  classes={len(np.unique(y))}  subjects={np.unique(subjects).tolist()}")
    return X, y, subjects, mapping


# =============================================================================
# Main
# =============================================================================

# Optional GNN+LSTM preset (slightly wider GCN/LSTM/MLP, milder dropout, gentler Adam).
TUNED_GNN_LSTM_MODEL_KWARGS: dict = {
    "gcn_hidden": 96,
    "gcn_output": 96,
    "num_gcn_layers": 2,
    "lstm_hidden": 192,
    "lstm_layers": 2,
    "dropout": 0.2,
    "mlp_hidden": 96,
}
TUNED_GNN_LSTM_LR = 5e-4
TUNED_GNN_LSTM_WEIGHT_DECAY = 5e-5
TUNED_GNN_LSTM_PATIENCE = 22

# Heavier preset for difficult LOSO folds (more capacity; still dropout).
TUNED_STRONG_GNN_LSTM_MODEL_KWARGS: dict = {
    "gcn_hidden": 128,
    "gcn_output": 128,
    "num_gcn_layers": 2,
    "lstm_hidden": 256,
    "lstm_layers": 2,
    "dropout": 0.25,
    "mlp_hidden": 128,
}
TUNED_STRONG_GNN_LSTM_LR = 8e-4
TUNED_STRONG_GNN_LSTM_WEIGHT_DECAY = 4e-5
TUNED_STRONG_GNN_LSTM_PATIENCE = 25


def run_dataset(
    name: str,
    max_windows_per_subject: int | None = None,
    model_types: list[str] | None = None,
    *,
    gnn_lstm_model_kwargs: dict | None = None,
    gnn_lstm_lr: float | None = None,
    gnn_lstm_weight_decay: float | None = None,
    gnn_lstm_patience: int | None = None,
) -> dict:
    print(f"\n{'='*70}")
    print(f"  Dataset: {name.upper()}")
    print(f"{'='*70}\n")

    X, y, subjects, mapping = load_dataset(name)

    # Optionally cap windows per subject (useful for large datasets like HHAR)
    if max_windows_per_subject is not None:
        rng = np.random.default_rng(SEED)
        keep = []
        for s in np.unique(subjects):
            idx = np.where(subjects == s)[0]
            if len(idx) > max_windows_per_subject:
                idx = rng.choice(idx, max_windows_per_subject, replace=False)
                idx = np.sort(idx)
            keep.append(idx)
        keep = np.concatenate(keep)
        X, y, subjects = X[keep], y[keep], subjects[keep]
        print(f"  [Subsampled to {max_windows_per_subject} windows/subject → {len(X)} total]")

    n_classes = len(np.unique(y))

    if name == "pamap2":
        node_feat_dim = PAMAP2_NODE_FEAT_DIM  # 36
        n_nodes       = 3
        adj_builder   = build_pamap2_adj
    else:
        node_feat_dim = HHAR_NODE_FEAT_DIM    # 18
        n_nodes       = 2
        adj_builder   = build_hhar_adj

    mtypes = model_types or ["lstm", "gnn", "gnn_lstm"]
    results: dict = {}

    for model_type in mtypes:
        tag = f"{model_type.replace('_','')}_{name}"  # e.g. gnnlstm_pamap2
        print(f"\n{'─'*60}")
        print(f"  Model: {model_type.upper()}  —  {name.upper()} LOSO")
        print(f"{'─'*60}")
        use_gl_kw = gnn_lstm_model_kwargs if model_type == "gnn_lstm" else None
        use_gl_lr = gnn_lstm_lr if model_type == "gnn_lstm" else None
        use_gl_wd = gnn_lstm_weight_decay if model_type == "gnn_lstm" else None
        use_gl_pt = gnn_lstm_patience if model_type == "gnn_lstm" else None
        res = loso_deep(
            X=X, y=y, subjects=subjects,
            model_type=model_type,
            dataset=name,
            n_classes=n_classes,
            node_feat_dim=node_feat_dim,
            n_nodes=n_nodes,
            adj_builder=adj_builder,
            tag=tag,
            gnn_lstm_kwargs=use_gl_kw,
            gnn_lstm_lr=use_gl_lr,
            gnn_lstm_weight_decay=use_gl_wd,
            gnn_lstm_patience=use_gl_pt,
        )
        results[model_type] = res

    # Merge into existing JSON so partial runs (e.g. only GNN+LSTM) keep other models.
    out_path = Path(METRICS_DIR) / f"{name}_deep_models.json"
    merged: dict = {}
    if out_path.exists():
        with open(out_path) as f:
            merged = json.load(f)
    merged.update(results)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nSaved deep model results → {out_path}  (merged keys: {list(results.keys())})")
    return merged


def main():
    parser = argparse.ArgumentParser(description="HAR full LOSO pipeline + plots")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lstm", "gnn", "gnn_lstm"],
        default=None,
        help="Which deep models to run (default: all three). Example: --models gnn_lstm",
    )
    parser.add_argument(
        "--gnn-lstm-tuned",
        action="store_true",
        help="Use a slightly wider GNN+LSTM + milder dropout + lower LR/WD and longer "
        "patience (only affects GNN+LSTM folds). Combine with --models gnn_lstm.",
    )
    parser.add_argument(
        "--gnn-lstm-tuned-strong",
        action="store_true",
        help="Larger GNN+LSTM (128-d GCN, 256-d LSTM) + lr=8e-4. Mutually exclusive with "
        "--gnn-lstm-tuned in practice: if both set, strong wins.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generate_all_plots (metrics and npy still written).",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=["pamap2", "hhar"],
        default=["pamap2", "hhar"],
        help="Which dataset(s) to run LOSO on (default: both).",
    )
    args = parser.parse_args()

    set_seed(SEED)
    run_p2 = "pamap2" in args.dataset
    run_hh = "hhar" in args.dataset

    model_types = args.models or ["lstm", "gnn", "gnn_lstm"]
    gl_model_kw: dict | None = None
    gl_lr = gl_wd = None
    gl_pat: int | None = None
    if args.gnn_lstm_tuned_strong:
        if "gnn_lstm" not in model_types:
            parser.error("--gnn-lstm-tuned-strong requires --models to include gnn_lstm")
        gl_model_kw = dict(TUNED_STRONG_GNN_LSTM_MODEL_KWARGS)
        gl_lr = TUNED_STRONG_GNN_LSTM_LR
        gl_wd = TUNED_STRONG_GNN_LSTM_WEIGHT_DECAY
        gl_pat = TUNED_STRONG_GNN_LSTM_PATIENCE
        print(
            "\n[GNN+LSTM tuned-strong] model kwargs:", gl_model_kw,
            "| lr=", gl_lr, "wd=", gl_wd, "patience=", gl_pat,
            flush=True,
        )
    elif args.gnn_lstm_tuned:
        if "gnn_lstm" not in model_types:
            parser.error("--gnn-lstm-tuned requires --models to include gnn_lstm")
        gl_model_kw = dict(TUNED_GNN_LSTM_MODEL_KWARGS)
        gl_lr = TUNED_GNN_LSTM_LR
        gl_wd = TUNED_GNN_LSTM_WEIGHT_DECAY
        gl_pat = TUNED_GNN_LSTM_PATIENCE
        print(
            "\n[GNN+LSTM tuned] model kwargs:", gl_model_kw,
            "| lr=", gl_lr, "wd=", gl_wd, "patience=", gl_pat,
            flush=True,
        )

    pamap2_path = Path(METRICS_DIR) / "pamap2_deep_models.json"
    hhar_path = Path(METRICS_DIR) / "hhar_deep_models.json"

    # ── PAMAP2 ────────────────────────────────────────────────────────────────
    if run_p2:
        pamap2_deep = run_dataset(
            "pamap2",
            model_types=model_types,
            gnn_lstm_model_kwargs=gl_model_kw,
            gnn_lstm_lr=gl_lr,
            gnn_lstm_weight_decay=gl_wd,
            gnn_lstm_patience=gl_pat,
        )
    else:
        pamap2_deep = json.load(open(pamap2_path)) if pamap2_path.exists() else {}
        print("\n[PAMAP2] Skipped (--dataset); using existing metrics for plots if present.")

    # ── HHAR ─────────────────────────────────────────────────────────────────
    hhar_X_path = Path(PROCESSED_DIR) / "hhar_X.npy"
    hhar_deep: dict = {}
    if run_hh and hhar_X_path.exists():
        # Cap to 5000 windows/subject so training remains tractable (~45k total).
        # CNN1D in cnn1d_results.json is trained on full HHAR unless you re-run CNN with the same cap.
        hhar_deep = run_dataset(
            "hhar",
            max_windows_per_subject=5000,
            model_types=model_types,
            gnn_lstm_model_kwargs=gl_model_kw,
            gnn_lstm_lr=gl_lr,
            gnn_lstm_weight_decay=gl_wd,
            gnn_lstm_patience=gl_pat,
        )
    elif run_hh:
        print("\n[HHAR] Processed data not found — skipping.")
    else:
        hhar_deep = json.load(open(hhar_path)) if hhar_path.exists() else {}
        print("\n[HHAR] Skipped (--dataset); using existing metrics for plots if present.")

    # ── Load baselines ────────────────────────────────────────────────────────
    pamap2_bl_path = Path(METRICS_DIR) / "pamap2_baselines.json"
    pamap2_baselines = {}
    if pamap2_bl_path.exists():
        with open(pamap2_bl_path) as f:
            pamap2_baselines = json.load(f)

    hhar_bl_path = Path(METRICS_DIR) / "hhar_baselines.json"
    hhar_baselines = {}
    if hhar_bl_path.exists():
        with open(hhar_bl_path) as f:
            hhar_baselines = json.load(f)

    # ── Generate all final plots ──────────────────────────────────────────────
    if args.skip_plots:
        print("\n[skip-plots] Skipping generate_all_plots.")
    else:
        print("\n\nGenerating final plots …")
        try:
            generate_all_plots(pamap2_deep, pamap2_baselines, hhar_deep, hhar_baselines)
        except Exception as e:
            print(f"[WARN] Plot generation failed: {e}")

    print("\n✅  Full pipeline complete.")


# =============================================================================
# Plot generation
# =============================================================================

def generate_all_plots(pamap2_deep, pamap2_baselines, hhar_deep, hhar_baselines):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    import seaborn as sns

    plots_dir = Path(PLOTS_DIR)

    # ── 1. Confusion matrices ─────────────────────────────────────────────────
    def plot_cm(dataset_name, model_tag, display_name):
        true_path = Path(METRICS_DIR) / f"{model_tag}_y_true.npy"
        pred_path = Path(METRICS_DIR) / f"{model_tag}_y_pred.npy"
        if not true_path.exists():
            return
        y_true = np.load(true_path)
        y_pred = np.load(pred_path)

        # Build label names
        if dataset_name == "pamap2":
            X  = np.load(Path(PROCESSED_DIR) / "pamap2_X.npy")
            y_raw = np.load(Path(PROCESSED_DIR) / "pamap2_y.npy")
            raw_classes = np.unique(y_raw)
            _, mapping  = remap_labels(y_raw)
            inv_map = {v: PAMAP2_ACTIVITIES.get(int(k), str(k)) for k, v in mapping.items()}
            class_names = [inv_map.get(i, str(i)) for i in range(len(np.unique(y_true)))]
        else:
            from src.config import HHAR_ACTIVITIES
            class_names = HHAR_ACTIVITIES[:len(np.unique(y_true))]

        cm = confusion_matrix(y_true, y_pred, normalize="true")
        fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(7, len(class_names)-1)))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f"Confusion Matrix — {display_name} ({dataset_name.upper()})")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        plt.tight_layout()
        out = plots_dir / f"cm_{model_tag}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved {out}")

    for ds in ["pamap2", "hhar"]:
        for mt in ["lstm", "gnn", "gnnlstm"]:
            tag = f"{mt}_{ds}"
            names = {"lstm": "LSTM-only", "gnn": "GNN-only", "gnnlstm": "GNN+LSTM"}
            plot_cm(ds, tag, names[mt])

    # ── 2. Model comparison chart (PAMAP2) ────────────────────────────────────
    def comparison_chart(deep_results, baselines, dataset_name):
        models, accs, f1s = [], [], []
        # Baselines first
        for k, v in baselines.items():
            models.append(k)
            accs.append(v.get("accuracy", 0) * 100)
            f1s.append(v.get("macro_f1", 0) * 100)
        # Deep models
        name_map = {"lstm": "LSTM-only", "gnn": "GNN-only", "gnn_lstm": "GNN+LSTM"}
        for k, v in deep_results.items():
            models.append(name_map.get(k, k))
            accs.append(v.get("accuracy", 0) * 100)
            f1s.append(v.get("macro_f1", 0) * 100)

        if not models:
            return

        x = np.arange(len(models))
        w = 0.35
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, vals, metric in zip(axes, [accs, f1s], ["Accuracy (%)", "Macro F1 (%)"]):
            bars = ax.bar(x - w/2, vals, w, label=metric, color="steelblue", alpha=0.85)
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
            ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right")
            ax.set_ylabel(metric); ax.set_title(f"{metric} — {dataset_name.upper()}")
            ax.set_ylim(0, 105)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
        fig.suptitle(f"Model Comparison — {dataset_name.upper()} (LOSO)", fontweight="bold")
        plt.tight_layout()
        out = plots_dir / f"model_comparison_{dataset_name}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved {out}")

    comparison_chart(pamap2_deep, pamap2_baselines, "pamap2")
    if hhar_deep:
        comparison_chart(hhar_deep, hhar_baselines, "hhar")

    # ── 3. Cross-dataset comparison ───────────────────────────────────────────
    if pamap2_deep and hhar_deep:
        name_map = {"lstm": "LSTM-only", "gnn": "GNN-only", "gnn_lstm": "GNN+LSTM"}
        models_common = [k for k in pamap2_deep if k in hhar_deep]
        labels = [name_map.get(m, m) for m in models_common]
        p_accs = [pamap2_deep[m]["accuracy"] * 100 for m in models_common]
        h_accs = [hhar_deep[m]["accuracy"]   * 100 for m in models_common]
        x = np.arange(len(labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(9, 5))
        b1 = ax.bar(x - w/2, p_accs, w, label="PAMAP2", color="steelblue", alpha=0.85)
        b2 = ax.bar(x + w/2, h_accs, w, label="HHAR",   color="darkorange", alpha=0.85)
        ax.bar_label(b1, fmt="%.1f", padding=3, fontsize=8)
        ax.bar_label(b2, fmt="%.1f", padding=3, fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 110)
        ax.set_title("Cross-Dataset Accuracy Comparison (LOSO)", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        out = plots_dir / "cross_dataset_comparison.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved {out}")

    # ── 4. SHAP feature importance ────────────────────────────────────────────
    try:
        import shap
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X_raw = np.load(Path(PROCESSED_DIR) / "pamap2_X.npy")
        y_raw = np.load(Path(PROCESSED_DIR) / "pamap2_y.npy")
        y_rem, _ = remap_labels(y_raw)
        X_flat = X_raw.reshape(len(X_raw), -1)

        print("  Training RF for SHAP …", flush=True)
        rf = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED))])
        rf.fit(X_flat, y_rem)

        bg  = shap.sample(X_flat, 200, random_state=SEED)
        exp = shap.TreeExplainer(rf.named_steps["rf"])
        sv  = exp.shap_values(bg)
        # sv: list of arrays (n_samples, n_features) for each class, or 3-D array
        if isinstance(sv, list):
            mean_abs = np.mean([np.abs(s) for s in sv], axis=0).mean(axis=0)
        else:
            mean_abs = np.abs(sv).mean(axis=(0, -1)) if sv.ndim == 3 else np.abs(sv).mean(axis=0)

        top_k = min(20, len(mean_abs))
        top_idx = np.argsort(mean_abs)[::-1][:top_k]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(np.arange(top_k), mean_abs[top_idx][::-1], color="steelblue", alpha=0.85)
        ax.set_yticks(np.arange(top_k))
        ax.set_yticklabels([f"feat_{i}" for i in top_idx[::-1]], fontsize=8)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("SHAP Feature Importance — RF on PAMAP2 (top-20)", fontweight="bold")
        plt.tight_layout()
        out = plots_dir / "shap_rf_pamap2.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved {out}")
    except Exception as e:
        print(f"  [WARN] SHAP skipped: {e}")

    # ── 5. Model profiling ────────────────────────────────────────────────────
    try:
        device = get_device()

        def count_params(m): return sum(p.numel() for p in m.parameters())

        def latency_ms(m, dummy_input, use_adj=False, dummy_adj=None, n=50):
            m.eval(); m.to(device)
            if use_adj:
                dummy_input = dummy_input.to(device)
                dummy_adj   = dummy_adj.to(device)
                _ = m(dummy_input, dummy_adj)  # warm up
                t0 = time.perf_counter()
                for _ in range(n):
                    with torch.no_grad(): m(dummy_input, dummy_adj)
            else:
                dummy_input = dummy_input.to(device)
                _ = m(dummy_input)
                t0 = time.perf_counter()
                for _ in range(n):
                    with torch.no_grad(): m(dummy_input)
            return (time.perf_counter() - t0) / n * 1000  # ms

        X_sample = np.load(Path(PROCESSED_DIR) / "pamap2_X.npy")
        y_sample  = np.load(Path(PROCESSED_DIR) / "pamap2_y.npy")
        y_rem, _  = remap_labels(y_sample)
        n_cls     = len(np.unique(y_rem))

        from src.graph_construction import build_pamap2_adj
        adj_t = build_pamap2_adj()

        models_profile = {
            "LSTM-only":  (LSTMOnlyModel(X_sample.shape[1]*X_sample.shape[2], n_cls), False, None),
            "GNN-only":   (GNNOnlyModel(PAMAP2_NODE_FEAT_DIM, 3, n_cls), True, adj_t),
            "GNN+LSTM":   (GNNLSTMModel(PAMAP2_NODE_FEAT_DIM, 3, n_cls), True, adj_t),
        }

        # Dummy inputs
        from src.dataset import HARWindowDataset, HARGraphDataset, HARSequenceDataset
        dummy_flat  = torch.zeros(1, X_sample.shape[1]*X_sample.shape[2])
        dummy_graph = torch.zeros(1, 3, PAMAP2_NODE_FEAT_DIM)
        dummy_seq   = torch.zeros(1, 10, 3, PAMAP2_NODE_FEAT_DIM)

        profile_data = {}
        for mname, (mdl, ua, adj_inp) in models_profile.items():
            params = count_params(mdl)
            if mname == "LSTM-only":
                lat = latency_ms(mdl, dummy_flat, use_adj=False)
            elif mname == "GNN-only":
                lat = latency_ms(mdl, dummy_graph, use_adj=True, dummy_adj=adj_inp)
            else:
                lat = latency_ms(mdl, dummy_seq, use_adj=True, dummy_adj=adj_inp)
            profile_data[mname] = {"params": params, "latency_ms": lat}
            print(f"  {mname}: {params:,} params | {lat:.2f} ms/sample")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        names = list(profile_data.keys())
        p_vals = [profile_data[n]["params"] / 1e6 for n in names]
        l_vals = [profile_data[n]["latency_ms"] for n in names]
        ax1.bar(names, p_vals, color=["steelblue","darkorange","green"], alpha=0.85)
        ax1.set_ylabel("Parameters (M)"); ax1.set_title("Model Size")
        for i, v in enumerate(p_vals): ax1.text(i, v + 0.01, f"{v:.2f}M", ha="center", fontsize=9)
        ax2.bar(names, l_vals, color=["steelblue","darkorange","green"], alpha=0.85)
        ax2.set_ylabel("Latency (ms/sample)"); ax2.set_title("Inference Latency")
        for i, v in enumerate(l_vals): ax2.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
        fig.suptitle("Model Profiling — PAMAP2", fontweight="bold")
        plt.tight_layout()
        out = plots_dir / "model_profiling.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved {out}")

        with open(Path(METRICS_DIR) / "model_profiling.json", "w") as f:
            json.dump(profile_data, f, indent=2)
    except Exception as e:
        print(f"  [WARN] Profiling skipped: {e}")

    print("\n✅  All plots saved to results/plots/")


if __name__ == "__main__":
    main()
