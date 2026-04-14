"""
train_holdout.py — 80/20 random hold-out training & evaluation
==============================================================
Trains and evaluates any model on any dataset using a simple
80% train / 20% test split (no LOSO).  Useful for quick checks
and hyperparameter tuning without the 9-fold overhead.

Usage examples
--------------
# single model + dataset
python scripts/train_holdout.py --dataset pamap2 --model gnn

# multiple models at once
python scripts/train_holdout.py --dataset pamap2 --model lstm gnn gnn_lstm

# HHAR, CNN1D only, cap windows to keep it fast
python scripts/train_holdout.py --dataset hhar --model cnn1d --max-windows 5000

# full run — all models, both datasets
python scripts/train_holdout.py --dataset pamap2 hhar --model lstm gnn gnn_lstm cnn1d

# override epochs / batch size / patience for a quick smoke-test
python scripts/train_holdout.py --dataset pamap2 --model gnn --epochs 5 --batch-size 128 --patience 3

Parameters
----------
--dataset       one or more of: pamap2  hhar                      (default: pamap2)
--model         one or more of: lstm  gnn  gnn_lstm  cnn1d        (default: gnn_lstm)
--test-size     fraction of data held out for test                 (default: 0.20)
--max-windows   cap windows per subject before splitting           (default: no cap)
--epochs        override NUM_EPOCHS from config                    (default: config)
--batch-size    override BATCH_SIZE from config                    (default: config)
--patience      early-stopping patience                            (default: config)
--seed          random seed                                        (default: config)
--tag-suffix    appended to saved file names to avoid overwriting  (default: holdout)
--no-save       skip saving model weights                          (flag)
--verbose       print per-epoch progress                           (flag)
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.config import (
    BATCH_SIZE,
    GCN_OUTPUT_DIM,
    HHAR_NODE_FEAT_DIM,
    LEARNING_RATE,
    METRICS_DIR,
    MODELS_DIR,
    NUM_EPOCHS,
    PAMAP2_NODE_FEAT_DIM,
    PATIENCE,
    PROCESSED_DIR,
    SEED,
    WEIGHT_DECAY,
)
from src.dataset import HARGraphDataset, HARSequenceDataset, HARWindowDataset, HARWindowDataset2D
from src.graph_construction import build_hhar_adj, build_pamap2_adj
from src.models import CNN1DModel, GNNLSTMModel, GNNOnlyModel, LSTMOnlyModel
from src.train import get_device, set_seed
from src.baselines import extract_features

for d in [MODELS_DIR, METRICS_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def remap_labels(y: np.ndarray) -> tuple[np.ndarray, dict]:
    classes = np.unique(y)
    mapping = {int(old): int(new) for new, old in enumerate(classes)}
    return np.vectorize(mapping.__getitem__)(y), mapping


# ─────────────────────────────────────────────────────────────────────────────
# Baseline (SVM / RF / XGBoost) holdout runner
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline(
    dataset: str,
    model_type: str,      # "svm", "rf", "xgboost"
    test_size: float,
    max_windows: int | None,
    seed: int,
    tag_suffix: str,
    save_model: bool,
) -> dict:
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.base import clone

    print(f"\n{'═'*65}")
    print(f"  {model_type.upper()}  on  {dataset.upper()}  "
          f"[{int((1-test_size)*100)}/{int(test_size*100)} split]")
    print(f"{'═'*65}")

    X, y, subjects, n_classes, mapping = load_dataset(dataset, max_windows, seed)

    # Stratified split
    indices = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=y
    )
    X_tr, y_tr = X[idx_train], y[idx_train]
    X_te, y_te = X[idx_test],  y[idx_test]
    print(f"  Split  →  train={len(X_tr)}  test={len(X_te)}")

    # Extract hand-crafted features
    print("  Extracting features …", flush=True)
    t0 = time.time()
    X_tr_feat = extract_features(X_tr)
    X_te_feat = extract_features(X_te)
    print(f"  Feature shape: {X_tr_feat.shape[1]} features per window")

    # Build classifier
    if model_type == "svm":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, random_state=seed)),
        ])
    elif model_type == "rf":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)),
        ])
    elif model_type == "xgboost":
        # XGBoost + PyTorch MPS segfaults on Apple Silicon when both are loaded.
        # Workaround: run XGBoost in a clean subprocess (no torch in memory).
        import subprocess
        worker = Path(__file__).parent / "_xgboost_worker.py"
        cmd = [
            sys.executable, str(worker),
            "--dataset",     dataset,
            "--test-size",   str(test_size),
            "--seed",        str(seed),
            "--tag-suffix",  tag_suffix,
            "--save-model",  "1" if save_model else "0",
        ]
        if max_windows is not None:
            cmd += ["--max-windows", str(max_windows)]

        proc = subprocess.run(cmd, capture_output=False, text=True)
        if proc.returncode != 0:
            print(f"  [ERROR] XGBoost worker exited with code {proc.returncode}")
            return {}
        # The worker already printed results and saved files.
        # Re-load y_true/y_pred to compute the result dict for the summary.
        tag = f"xgboost_{dataset}_{tag_suffix}"
        y_te_saved   = np.load(Path(METRICS_DIR) / f"{tag}_y_true.npy")
        y_pred_saved = np.load(Path(METRICS_DIR) / f"{tag}_y_pred.npy")
        acc = accuracy_score(y_te_saved, y_pred_saved)
        f1  = f1_score(y_te_saved, y_pred_saved, average="macro", zero_division=0)
        bal = balanced_accuracy_score(y_te_saved, y_pred_saved)
        return {
            "dataset":      dataset,
            "model":        "xgboost",
            "accuracy":     acc,
            "macro_f1":     f1,
            "balanced_acc": bal,
            "train_time_s": 0,   # not tracked here
            "params":       None,
        }

    # Train
    print(f"  Training {model_type.upper()} …", flush=True)
    clf.fit(X_tr_feat, y_tr)
    elapsed = time.time() - t0

    # Evaluate
    y_pred = clf.predict(X_te_feat)
    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average="macro", zero_division=0)
    bal = balanced_accuracy_score(y_te, y_pred)

    print(f"\n  ── Results ──────────────────────────────────────")
    print(f"  Accuracy     : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro F1     : {f1:.4f}  ({f1*100:.2f}%)")
    print(f"  Balanced Acc : {bal:.4f}")
    print(f"  Train time   : {elapsed:.1f}s")
    print(classification_report(y_te, y_pred, zero_division=0))

    result = {
        "dataset":      dataset,
        "model":        model_type,
        "split":        f"{int((1-test_size)*100)}/{int(test_size*100)}",
        "n_train":      len(X_tr),
        "n_test":       len(X_te),
        "accuracy":     acc,
        "macro_f1":     f1,
        "balanced_acc": bal,
        "train_time_s": round(elapsed, 1),
    }

    # Save predictions
    tag = f"{model_type}_{dataset}_{tag_suffix}"
    np.save(Path(METRICS_DIR) / f"{tag}_y_true.npy", np.array(y_te))
    np.save(Path(METRICS_DIR) / f"{tag}_y_pred.npy", np.array(y_pred))
    print(f"  Saved predictions → results/metrics/{tag}_y_{{true,pred}}.npy")

    # Save model (pickle)
    if save_model:
        import pickle
        mp = Path(MODELS_DIR) / f"{tag}.pkl"
        with open(mp, "wb") as f_out:
            pickle.dump(clf, f_out)
        print(f"  Saved model      → {mp}")

    # Free memory
    del X_tr_feat, X_te_feat, clf
    import gc; gc.collect()

    return result


def load_dataset(name: str, max_windows: int | None, seed: int):
    base = Path(PROCESSED_DIR)
    X = np.load(base / f"{name}_X.npy")
    y = np.load(base / f"{name}_y.npy")
    subjects = np.load(base / f"{name}_subjects.npy")
    y, mapping = remap_labels(y)

    if max_windows is not None:
        rng = np.random.default_rng(seed)
        keep = []
        for s in np.unique(subjects):
            idx = np.where(subjects == s)[0]
            if len(idx) > max_windows:
                idx = rng.choice(idx, max_windows, replace=False)
                idx = np.sort(idx)
            keep.append(idx)
        keep = np.concatenate(keep)
        X, y, subjects = X[keep], y[keep], subjects[keep]

    n_classes = len(np.unique(y))
    print(f"  [{name}] X={X.shape}  classes={n_classes}  "
          f"subjects={np.unique(subjects).tolist()}  total={len(X)}")
    return X, y, subjects, n_classes, mapping


def build_loaders(
    X_tr, y_tr, s_tr,
    X_val, y_val, s_val,
    X_te, y_te,
    model_type: str,
    dataset: str,
    batch_size: int,
):
    """Return (train_loader, val_loader, test_loader, use_adj)."""

    if model_type == "lstm":
        use_adj = False
        tr  = DataLoader(HARWindowDataset(X_tr,  y_tr),  batch_size, shuffle=True,  num_workers=0)
        val = DataLoader(HARWindowDataset(X_val, y_val), batch_size, shuffle=False, num_workers=0)
        te  = DataLoader(HARWindowDataset(X_te,  y_te),  batch_size, shuffle=False, num_workers=0)

    elif model_type == "gnn":
        use_adj = True
        tr  = DataLoader(HARGraphDataset(X_tr,  y_tr,  dataset=dataset), batch_size, shuffle=True,  num_workers=0)
        val = DataLoader(HARGraphDataset(X_val, y_val, dataset=dataset), batch_size, shuffle=False, num_workers=0)
        te  = DataLoader(HARGraphDataset(X_te,  y_te,  dataset=dataset), batch_size, shuffle=False, num_workers=0)

    elif model_type == "gnn_lstm":
        use_adj = True
        ds_tr  = HARSequenceDataset(X_tr,  y_tr,  subjects=s_tr,  dataset=dataset)
        ds_val = HARSequenceDataset(X_val, y_val, subjects=s_val, dataset=dataset)
        ds_te  = HARSequenceDataset(X_te,  y_te,  dataset=dataset)
        if len(ds_tr)  == 0: ds_tr  = ds_val
        if len(ds_val) == 0: ds_val = ds_tr
        tr  = DataLoader(ds_tr,  batch_size, shuffle=True,  num_workers=0)
        val = DataLoader(ds_val, batch_size, shuffle=False, num_workers=0)
        te  = DataLoader(ds_te,  batch_size, shuffle=False, num_workers=0)

    elif model_type == "cnn1d":
        use_adj = False
        tr  = DataLoader(HARWindowDataset2D(X_tr,  y_tr),  batch_size, shuffle=True,  num_workers=0)
        val = DataLoader(HARWindowDataset2D(X_val, y_val), batch_size, shuffle=False, num_workers=0)
        te  = DataLoader(HARWindowDataset2D(X_te,  y_te),  batch_size, shuffle=False, num_workers=0)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return tr, val, te, use_adj


def build_model(
    model_type: str,
    dataset: str,
    n_classes: int,
    input_flat_dim: int,
) -> nn.Module:
    if dataset == "pamap2":
        node_feat_dim, n_nodes = PAMAP2_NODE_FEAT_DIM, 3
    else:
        node_feat_dim, n_nodes = HHAR_NODE_FEAT_DIM, 2

    if model_type == "lstm":
        return LSTMOnlyModel(input_flat_dim, n_classes)
    elif model_type == "gnn":
        return GNNOnlyModel(node_feat_dim, n_nodes, n_classes)
    elif model_type == "gnn_lstm":
        return GNNLSTMModel(node_feat_dim, n_nodes, n_classes)
    elif model_type == "cnn1d":
        # CNN1DModel(n_timesteps, n_channels, n_classes)
        # input_flat_dim here is n_channels (set in run_one for cnn1d)
        n_timesteps = 128  # WINDOW_SIZE from config
        n_channels  = input_flat_dim
        return CNN1DModel(n_timesteps, n_channels, n_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    tr_loader: DataLoader,
    val_loader: DataLoader,
    use_adj: bool,
    device: torch.device,
    epochs: int,
    patience: int,
    verbose: bool,
) -> tuple[nn.Module, list[float], list[float]]:
    """Train with early stopping on val_loss. Returns (best_model, train_losses, val_accs)."""
    opt   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit  = nn.CrossEntropyLoss()

    best_val_loss   = float("inf")
    best_state      = None
    patience_cnt    = 0
    train_losses: list[float] = []
    val_accs:     list[float] = []

    for epoch in range(1, epochs + 1):
        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_batches  = 0
        for batch in tr_loader:
            opt.zero_grad()
            if use_adj:
                x, adj, yb = batch
                x, adj, yb = x.to(device), adj.to(device), yb.to(device)
                logits = model(x, adj)
            else:
                x, yb = batch
                x, yb = x.to(device), yb.to(device)
                logits = model(x)
            loss = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches  += 1
        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if use_adj:
                    x, adj, yb = batch
                    x, adj, yb = x.to(device), adj.to(device), yb.to(device)
                    logits = model(x, adj)
                else:
                    x, yb = batch
                    x, yb = x.to(device), yb.to(device)
                    logits = model(x)
                val_loss += crit(logits, yb).item() * len(yb)
                correct  += (logits.argmax(1) == yb).sum().item()
                total    += len(yb)
        avg_val_loss = val_loss / max(total, 1)
        val_acc      = correct  / max(total, 1)
        val_accs.append(val_acc)
        sched.step(avg_val_loss)

        if verbose:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={avg_train_loss:.4f}  "
                  f"val_loss={avg_val_loss:.4f}  "
                  f"val_acc={val_acc:.4f}  "
                  f"({time.time()-t0:.1f}s)", flush=True)

        # ── early stopping on val_loss ─────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state    = copy.deepcopy(model.state_dict())
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stop at epoch {epoch}  "
                      f"(best val_loss={best_val_loss:.4f})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_accs


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    te_loader: DataLoader,
    use_adj: bool,
    device: torch.device,
) -> tuple[list, list]:
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in te_loader:
            if use_adj:
                x, adj, yb = batch
                x, adj, yb = x.to(device), adj.to(device), yb.to(device)
                preds = model(x, adj).argmax(1)
            else:
                x, yb = batch
                x, yb = x.to(device), yb.to(device)
                preds = model(x).argmax(1)
            all_true.extend(yb.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
    return all_true, all_pred


# ─────────────────────────────────────────────────────────────────────────────
# Single run: one model on one dataset
# ─────────────────────────────────────────────────────────────────────────────

def run_one(
    dataset: str,
    model_type: str,
    test_size: float,
    max_windows: int | None,
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
    tag_suffix: str,
    save_model: bool,
    verbose: bool,
) -> dict:
    print(f"\n{'═'*65}")
    print(f"  {model_type.upper()}  on  {dataset.upper()}  "
          f"[80/{int((1-test_size)*100)} → {int(test_size*100)}% split]")
    print(f"{'═'*65}")

    device = get_device()
    set_seed(seed)

    # ── load ──────────────────────────────────────────────────────────────────
    X, y, subjects, n_classes, mapping = load_dataset(dataset, max_windows, seed)

    # ── split: stratify by label to maintain class balance ───────────────────
    indices = np.arange(len(X))
    idx_trainval, idx_test = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=y
    )
    # further split train into 80% train / 20% val (of the trainval portion)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.20, random_state=seed, stratify=y[idx_trainval]
    )

    X_tr,  y_tr,  s_tr  = X[idx_train], y[idx_train], subjects[idx_train]
    X_val, y_val, s_val = X[idx_val],   y[idx_val],   subjects[idx_val]
    X_te,  y_te         = X[idx_test],  y[idx_test]

    print(f"  Split  →  train={len(X_tr)}  val={len(X_val)}  test={len(X_te)}")

    # ── data loaders ─────────────────────────────────────────────────────────
    tr_loader, val_loader, te_loader, use_adj = build_loaders(
        X_tr, y_tr, s_tr, X_val, y_val, s_val, X_te, y_te,
        model_type, dataset, batch_size,
    )
    print(f"  Batches  →  train={len(tr_loader)}  val={len(val_loader)}  test={len(te_loader)}")

    # ── model ─────────────────────────────────────────────────────────────────
    # for CNN1D input_flat_dim = n_channels (not flattened)
    input_flat_dim = X.shape[2] if model_type == "cnn1d" else X.shape[1] * X.shape[2]
    model = build_model(model_type, dataset, n_classes, input_flat_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model  →  {model.__class__.__name__}  ({n_params:,} params)  device={device}")

    # ── train ─────────────────────────────────────────────────────────────────
    t_start = time.time()
    model, train_losses, val_accs = train_model(
        model, tr_loader, val_loader, use_adj,
        device, epochs, patience, verbose,
    )
    elapsed = time.time() - t_start
    print(f"  Training done in {elapsed/60:.1f} min", flush=True)

    # ── test ──────────────────────────────────────────────────────────────────
    all_true, all_pred = evaluate(model, te_loader, use_adj, device)

    acc = accuracy_score(all_true, all_pred)
    f1  = f1_score(all_true, all_pred, average="macro", zero_division=0)
    bal = balanced_accuracy_score(all_true, all_pred)

    print(f"\n  ── Results ──────────────────────────────────────")
    print(f"  Accuracy     : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro F1     : {f1:.4f}  ({f1*100:.2f}%)")
    print(f"  Balanced Acc : {bal:.4f}")
    print(f"  ─────────────────────────────────────────────────")
    print(classification_report(all_true, all_pred, zero_division=0))

    result = {
        "dataset":      dataset,
        "model":        model_type,
        "split":        f"{int((1-test_size)*100)}/{int(test_size*100)}",
        "n_train":      len(X_tr),
        "n_val":        len(X_val),
        "n_test":       len(X_te),
        "accuracy":     acc,
        "macro_f1":     f1,
        "balanced_acc": bal,
        "params":       n_params,
        "train_time_s": round(elapsed, 1),
    }

    # ── save ──────────────────────────────────────────────────────────────────
    tag = f"{model_type}_{dataset}_{tag_suffix}"

    np.save(Path(METRICS_DIR) / f"{tag}_y_true.npy", np.array(all_true))
    np.save(Path(METRICS_DIR) / f"{tag}_y_pred.npy", np.array(all_pred))
    print(f"  Saved predictions → results/metrics/{tag}_y_{{true,pred}}.npy")

    if save_model:
        mp = Path(MODELS_DIR) / f"{tag}.pt"
        torch.save(model.state_dict(), mp)
        print(f"  Saved model      → {mp}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="80/20 hold-out training for HAR models",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--dataset", nargs="+", default=["pamap2"],
        choices=["pamap2", "hhar"],
        help="Dataset(s) to run  (default: pamap2)",
    )
    p.add_argument(
        "--model", nargs="+", default=["gnn_lstm"],
        choices=["lstm", "gnn", "gnn_lstm", "cnn1d", "svm", "rf", "xgboost"],
        help="Model(s) to train  (default: gnn_lstm)",
    )
    p.add_argument(
        "--test-size", type=float, default=0.20,
        help="Fraction of data used for test  (default: 0.20)",
    )
    p.add_argument(
        "--max-windows", type=int, default=None,
        help="Cap windows per subject before splitting  (default: no cap)",
    )
    p.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS,
        help=f"Max training epochs  (default: {NUM_EPOCHS} from config)",
    )
    p.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Batch size  (default: {BATCH_SIZE} from config)",
    )
    p.add_argument(
        "--patience", type=int, default=PATIENCE,
        help=f"Early-stopping patience  (default: {PATIENCE} from config)",
    )
    p.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed  (default: {SEED} from config)",
    )
    p.add_argument(
        "--tag-suffix", type=str, default="holdout",
        help="Suffix appended to saved file names  (default: holdout)",
    )
    p.add_argument(
        "--no-save", action="store_true",
        help="Skip saving model weights",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print per-epoch progress",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "█"*65)
    print("  HAR Hold-out Training Script")
    print("█"*65)
    print(f"  Datasets   : {args.dataset}")
    print(f"  Models     : {args.model}")
    print(f"  Split      : {int((1-args.test_size)*100)} / {int(args.test_size*100)}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Patience   : {args.patience}")
    print(f"  Seed       : {args.seed}")
    print(f"  Max windows: {args.max_windows or 'no cap'}")
    print(f"  Save model : {not args.no_save}")
    print("█"*65 + "\n")

    BASELINE_MODELS = {"svm", "rf", "xgboost"}

    all_results = {}

    for dataset in args.dataset:
        all_results[dataset] = {}
        for model_type in args.model:
            if model_type in BASELINE_MODELS:
                result = run_baseline(
                    dataset      = dataset,
                    model_type   = model_type,
                    test_size    = args.test_size,
                    max_windows  = args.max_windows,
                    seed         = args.seed,
                    tag_suffix   = args.tag_suffix,
                    save_model   = not args.no_save,
                )
            else:
                result = run_one(
                    dataset      = dataset,
                    model_type   = model_type,
                    test_size    = args.test_size,
                    max_windows  = args.max_windows,
                    epochs       = args.epochs,
                    batch_size   = args.batch_size,
                    patience     = args.patience,
                    seed         = args.seed,
                    tag_suffix   = args.tag_suffix,
                    save_model   = not args.no_save,
                    verbose      = args.verbose,
                )
            all_results[dataset][model_type] = result

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n" + "═"*65)
    print("  SUMMARY")
    print("═"*65)
    print(f"  {'Dataset':<8} {'Model':<10} {'Acc':>7} {'F1':>7} {'BalAcc':>8}  {'Params':>12}  {'Time':>8}")
    print("  " + "─"*62)
    for ds, models in all_results.items():
        for mt, r in models.items():
            params_str = f"{r['params']:>12,}" if r.get("params") is not None else f"{'N/A':>12}"
            print(f"  {ds:<8} {mt:<10} "
                  f"{r['accuracy']:>7.4f} {r['macro_f1']:>7.4f} {r['balanced_acc']:>8.4f}  "
                  f"{params_str}  {r['train_time_s']:>6.0f}s")

    # Save summary JSON
    out_path = Path(METRICS_DIR) / f"holdout_results_{args.tag_suffix}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved summary → {out_path}")
    print("═"*65)


if __name__ == "__main__":
    main()
