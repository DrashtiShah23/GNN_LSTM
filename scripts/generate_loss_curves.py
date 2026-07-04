"""
Generate train/val loss curves for the project's best GNN+LSTM runs only.

  PAMAP2 — ImprovedGNNLSTMAttnAdj (LOSO acc ≈ 0.851), best test fold
  HHAR   — ImprovedGNNLSTM (LOSO acc ≈ 0.532), best test fold, 5000 win/subj

Matches scripts/run_attention_adj.py and scripts/run_improved_gnnlstm.py.

  python scripts/generate_loss_curves.py
  python scripts/generate_loss_curves.py --datasets pamap2 --all-folds
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    BATCH_SIZE,
    DROPOUT,
    GCN_HIDDEN_DIM,
    GCN_OUTPUT_DIM,
    HHAR_NODE_FEAT_DIM,
    LEARNING_RATE,
    LSTM_HIDDEN_DIM,
    LSTM_NUM_LAYERS,
    METRICS_DIR,
    MLP_HIDDEN_DIM,
    NUM_EPOCHS,
    OVERLAP,
    PAMAP2_EDGES,
    PAMAP2_NODE_FEAT_DIM,
    PATIENCE,
    PLOTS_DIR,
    PROCESSED_DIR,
    SEED,
    TARGET_SAMPLING_RATE,
    WEIGHT_DECAY,
    WINDOW_SIZE,
)
from src.dataset import HARSequenceDataset  # noqa: E402
from src.evaluation import plot_loso_folds_grid, plot_training_history  # noqa: E402
from src.graph_construction import build_hhar_adj, build_pamap2_adj  # noqa: E402
from src.models import ImprovedGNNLSTMModel, ImprovedGNNLSTMAttnAdj  # noqa: E402
from src.train import get_device, loso_splits, set_seed  # noqa: E402

# Best test-accuracy fold (0-based) from saved LOSO metrics
BEST_FOLD = {
    ("pamap2", "attn_adj"): 5,   # acc 0.9326
    ("hhar", "improved"): 1,     # acc 0.608
}


def remap(y: np.ndarray) -> tuple[np.ndarray, int]:
    classes = np.unique(y)
    mp = {int(old): int(new) for new, old in enumerate(classes)}
    return np.vectorize(mp.__getitem__)(y), len(classes)


def load(name: str, max_per_subject: int | None = None):
    p = Path(PROCESSED_DIR)
    X = np.load(p / f"{name}_X.npy")
    y, n_classes = remap(np.load(p / f"{name}_y.npy"))
    subj = np.load(p / f"{name}_subjects.npy")
    if max_per_subject:
        rng = np.random.default_rng(SEED)
        keep = []
        for s in np.unique(subj):
            idx = np.where(subj == s)[0]
            if len(idx) > max_per_subject:
                idx = np.sort(rng.choice(idx, max_per_subject, replace=False))
            keep.append(idx)
        idx = np.concatenate(keep)
        X, y, subj = X[idx], y[idx], subj[idx]
    return X, y, subj, n_classes


def seq_subject_labels(subj: np.ndarray, seq_len: int = 10) -> np.ndarray:
    out = []
    for s in np.unique(subj):
        mask = np.where(subj == s)[0]
        n_seqs = len(range(0, len(mask) - seq_len + 1, seq_len))
        out.extend([s] * n_seqs)
    return np.array(out)


def train_fold_with_history(
    model: nn.Module,
    tr_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """Same loop as run_improved_gnnlstm.train_fold, with per-epoch history."""
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit = nn.CrossEntropyLoss()
    best_loss, best_state, pat = float("inf"), None, 0
    best_epoch = 1
    early_stopped = False
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for x, adj, y in tr_loader:
            x, adj, y = x.to(device), adj.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x, adj), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss_sum += loss.item() * len(y)
            train_n += len(y)
        train_loss = train_loss_sum / max(train_n, 1)

        model.eval()
        val_loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, adj, y in val_loader:
                x, adj, y = x.to(device), adj.to(device), y.to(device)
                logits = model(x, adj)
                val_loss_sum += crit(logits, y).item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                total += len(y)
        val_loss = val_loss_sum / max(total, 1)
        val_acc = correct / max(total, 1)
        sched.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_loss:
            best_loss, best_state, pat = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            best_epoch = epoch
        else:
            pat += 1
            if pat >= PATIENCE:
                early_stopped = True
                break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["stop_epoch"] = len(history["train_loss"])
    history["early_stopped"] = early_stopped
    history["best_val_loss"] = float(best_loss)
    return model, history


def build_pamap2_best_hyperparameters(
    history: dict,
    *,
    n_classes: int,
    fold: int,
    test_subject: str,
    seq_len: int = 10,
) -> dict:
    """Full hyperparameter record for best PAMAP2 GNN+LSTM (ImprovedGNNLSTM+AttnAdj)."""
    import json

    loso_metrics = {}
    attn_path = Path(METRICS_DIR) / "attn_adj_results.json"
    if attn_path.exists():
        with open(attn_path) as f:
            loso_metrics = json.load(f).get("pamap2", {})

    return {
        "model": "ImprovedGNNLSTM+AttnAdj",
        "dataset": "pamap2",
        "evaluation": {
            "protocol": "leave-one-subject-out (LOSO)",
            "val_split": "15% of training sequences (tail of train indices)",
            "loss_curve_fold": fold,
            "loss_curve_test_subject": test_subject,
        },
        "reported_loso_metrics": {
            "accuracy_mean": loso_metrics.get("accuracy"),
            "accuracy_std": loso_metrics.get("accuracy_std"),
            "macro_f1_mean": loso_metrics.get("macro_f1"),
            "macro_f1_std": loso_metrics.get("macro_f1_std"),
            "balanced_acc_mean": loso_metrics.get("balanced_acc"),
            "balanced_acc_std": loso_metrics.get("balanced_acc_std"),
            "per_fold_accuracy": loso_metrics.get("per_fold", {}).get("accuracy"),
        },
        "preprocessing": {
            "window_size_samples": WINDOW_SIZE,
            "window_overlap": OVERLAP,
            "sampling_rate_hz": TARGET_SAMPLING_RATE,
            "n_graph_nodes": 3,
            "node_positions": ["wrist", "chest", "ankle"],
            "node_feat_dim": PAMAP2_NODE_FEAT_DIM,
            "fixed_edges": PAMAP2_EDGES,
            "adjacency": "anatomical skeleton × learned attention gate (sigmoid)",
        },
        "sequence": {
            "seq_len_windows": seq_len,
            "sequence_stride": seq_len,
            "label": "majority vote over windows in sequence",
        },
        "architecture": {
            "gcn_hidden": GCN_HIDDEN_DIM,
            "gcn_output": GCN_OUTPUT_DIM,
            "gcn_layers": 2,
            "gcn_norm": "LayerNorm",
            "pooling": "concat per-node (not mean)",
            "skip_connection": True,
            "proj_dim": 128,
            "lstm_hidden": LSTM_HIDDEN_DIM,
            "lstm_layers": LSTM_NUM_LAYERS,
            "lstm_bidirectional": True,
            "temporal_pooling": "soft attention over LSTM steps",
            "mlp_hidden": MLP_HIDDEN_DIM,
            "n_classes": n_classes,
            "dropout": DROPOUT,
        },
        "optimizer": {
            "name": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip_max_norm": 1.0,
            "lr_scheduler": "ReduceLROnPlateau",
            "scheduler_patience": 5,
            "scheduler_factor": 0.5,
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "max_epochs": NUM_EPOCHS,
            "loss": "cross_entropy",
            "early_stopping_monitor": "validation_loss",
            "early_stopping_patience": PATIENCE,
            "seed": SEED,
        },
        "loss_curve_run": {
            "epochs_trained": history.get("stop_epoch"),
            "best_epoch": history.get("best_epoch"),
            "early_stopped": history.get("early_stopped"),
            "best_val_loss": history.get("best_val_loss"),
        },
    }


def pamap2_plot_footer(history: dict) -> str:
    """Footer text for PAMAP2 loss-curve figure."""
    best_e = history.get("best_epoch", "?")
    stop_e = history.get("stop_epoch", "?")
    stopped = history.get("early_stopped", False)
    best_vl = history.get("best_val_loss")
    vl_str = f"{best_vl:.4f}" if best_vl is not None else "—"
    stop_note = f"yes (epoch {stop_e})" if stopped else f"no (ran {stop_e} epochs, max {NUM_EPOCHS})"
    return (
        "Architecture:  GCN {gcn_h}→{gcn_o} (LayerNorm, concat pool, skip)  |  "
        "proj {proj}  |  BiLSTM {lstm_h}×{lstm_l} + temporal attn  |  MLP {mlp}  |  "
        "AttnAdj on edges {edges}  |  seq_len={seq}, batch={bs}, lr={lr}, Adam wd={wd:g}\n"
        "Regularization:  dropout p={dropout}  |  grad clip 1.0  |  "
        "ReduceLROnPlateau (patience=5, factor=0.5)\n"
        "Early stopping:  val loss, patience={pat}  |  triggered={stop_note}  |  "
        "best epoch {best_e} (val loss {vl_str})  →  weights restored"
    ).format(
        gcn_h=GCN_HIDDEN_DIM,
        gcn_o=GCN_OUTPUT_DIM,
        proj=128,
        lstm_h=LSTM_HIDDEN_DIM,
        lstm_l=LSTM_NUM_LAYERS,
        mlp=MLP_HIDDEN_DIM,
        edges=PAMAP2_EDGES,
        seq=10,
        bs=BATCH_SIZE,
        lr=LEARNING_RATE,
        wd=WEIGHT_DECAY,
        dropout=DROPOUT,
        pat=PATIENCE,
        stop_note=stop_note,
        best_e=best_e,
        vl_str=vl_str,
    )


def _train_one_loso_fold(
    fold_idx: int,
    folds: list,
    torch_ds: HARSequenceDataset,
    *,
    dataset: str,
    variant: str,
    n_classes: int,
    node_feat: int,
    device: torch.device,
) -> tuple[dict, str, int]:
    """Train one LOSO fold; return (history, test_subject, fold_number_1based)."""
    tr_idx, te_idx, te_subj = folds[fold_idx]
    n_val = max(1, int(len(tr_idx) * 0.15))
    val_idx = tr_idx[-n_val:]
    tr_idx = tr_idx[:-n_val]

    tr_loader = DataLoader(Subset(torch_ds, tr_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(torch_ds, val_idx), batch_size=BATCH_SIZE, shuffle=False)

    set_seed(SEED)
    if variant == "attn_adj":
        init_adj = build_pamap2_adj() if dataset == "pamap2" else build_hhar_adj()
        model = ImprovedGNNLSTMAttnAdj(
            node_feat_dim=node_feat, n_nodes=3, n_classes=n_classes, init_adj=init_adj,
        ).to(device)
    else:
        model = ImprovedGNNLSTMModel(
            node_feat_dim=node_feat, n_nodes=3, n_classes=n_classes,
        ).to(device)

    _, history = train_fold_with_history(model, tr_loader, val_loader, device)
    return history, str(te_subj), fold_idx + 1


def run_pamap2_all_folds(seq_len: int = 10) -> list[Path]:
    """Train and plot loss curves for every PAMAP2 LOSO fold."""
    import json

    label = "ImprovedGNNLSTM+AttnAdj"
    X, y, subj, n_classes = load("pamap2", max_per_subject=None)
    device = get_device()
    torch_ds = HARSequenceDataset(X, y, subjects=subj, dataset="pamap2", seq_len=seq_len, cache=True)
    seq_subjects = seq_subject_labels(subj, seq_len)
    folds = list(loso_splits(seq_subjects))

    out_dir = Path(PLOTS_DIR) / "gnn_lstm_best_pamap2_folds"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    histories: list[dict] = []
    titles: list[str] = []
    per_fold_meta: list[dict] = []

    attn_metrics = {}
    attn_path = Path(METRICS_DIR) / "attn_adj_results.json"
    if attn_path.exists():
        with open(attn_path) as f:
            attn_metrics = json.load(f).get("pamap2", {}).get("per_fold", {})

    print(f"\n{'='*68}\n  {label} — PAMAP2 — all {len(folds)} LOSO folds\n{'='*68}", flush=True)

    for fold_i in range(len(folds)):
        history, te_subj, fold_num = _train_one_loso_fold(
            fold_i, folds, torch_ds,
            dataset="pamap2", variant="attn_adj",
            n_classes=n_classes, node_feat=PAMAP2_NODE_FEAT_DIM, device=device,
        )
        histories.append(history)
        test_acc = None
        if attn_metrics.get("accuracy") and fold_i < len(attn_metrics["accuracy"]):
            test_acc = attn_metrics["accuracy"][fold_i]
        acc_note = f", test acc {test_acc:.3f}" if test_acc is not None else ""
        titles.append(f"Fold {fold_num} (subj {te_subj}{acc_note})")

        print(
            f"  Fold {fold_num}/{len(folds)} subj {te_subj} | "
            f"best epoch {history['best_epoch']} | stop {history['stop_epoch']} | "
            f"val loss {history['best_val_loss']:.4f}",
            flush=True,
        )

        run_name = f"fold{fold_num:02d}_subj{te_subj}"
        dest = out_dir / f"{run_name}_loss_curves.png"
        plot_training_history(
            history,
            run_name=run_name,
            best_epoch=history["best_epoch"],
            stop_epoch=history["stop_epoch"],
            early_stopped=history["early_stopped"],
            reg_summary=pamap2_plot_footer(history),
            suptitle=f"PAMAP2 — {label}  |  LOSO fold {fold_num}  |  test subject {te_subj}",
            output_path=dest,
        )
        saved.append(dest)

        per_fold_meta.append({
            "fold": fold_num,
            "test_subject": te_subj,
            "reported_test_accuracy": test_acc,
            "loss_curve_run": {
                "epochs_trained": history["stop_epoch"],
                "best_epoch": history["best_epoch"],
                "early_stopped": history["early_stopped"],
                "best_val_loss": history["best_val_loss"],
            },
        })

    grid_path = Path(PLOTS_DIR) / "gnn_lstm_best_pamap2_all_folds_grid.png"
    plot_loso_folds_grid(
        histories,
        titles,
        grid_path,
        suptitle=f"PAMAP2 — {label} — train/val loss (all LOSO folds)",
    )
    saved.append(grid_path)

    base_hparams = build_pamap2_best_hyperparameters(
        histories[0],
        n_classes=n_classes,
        fold=1,
        test_subject=str(folds[0][2]),
    )
    base_hparams["evaluation"]["note"] = "Per-fold loss curves in results/plots/gnn_lstm_best_pamap2_folds/"
    base_hparams["per_fold_loss_runs"] = per_fold_meta

    hparam_path = Path(METRICS_DIR) / "gnn_lstm_best_pamap2_hyperparameters.json"
    with open(hparam_path, "w") as f:
        json.dump(base_hparams, f, indent=2)
    print(f"  Saved hyperparameters → {hparam_path}")

    # Keep single-file alias for fold 6 (best test acc)
    best_i = BEST_FOLD[("pamap2", "attn_adj")]
    alias = Path(PLOTS_DIR) / "gnn_lstm_best_pamap2_loss_curves.png"
    best_fold_png = out_dir / f"fold{best_i + 1:02d}_subj{folds[best_i][2]}_loss_curves.png"
    if best_fold_png.exists():
        alias.write_bytes(best_fold_png.read_bytes())
        print(f"  Copied best fold plot → {alias}")

    return saved


def run_best_fold(
    dataset: str,
    variant: str,
    seq_len: int = 10,
    max_per_subject: int | None = None,
) -> Path:
    fold_idx = BEST_FOLD[(dataset, variant)]
    X, y, subj, n_classes = load(dataset, max_per_subject)
    node_feat = PAMAP2_NODE_FEAT_DIM if dataset == "pamap2" else HHAR_NODE_FEAT_DIM
    device = get_device()

    torch_ds = HARSequenceDataset(X, y, subjects=subj, dataset=dataset, seq_len=seq_len, cache=True)
    seq_subjects = seq_subject_labels(subj, seq_len)

    folds = list(loso_splits(seq_subjects))
    label = "ImprovedGNNLSTM+AttnAdj" if variant == "attn_adj" else "ImprovedGNNLSTM"
    history, te_subj, fold_num = _train_one_loso_fold(
        fold_idx, folds, torch_ds,
        dataset=dataset, variant=variant,
        n_classes=n_classes, node_feat=node_feat, device=device,
    )
    print(f"\n=== {label} | {dataset.upper()} | fold {fold_num} | test subject {te_subj} ===", flush=True)

    run_name = f"gnn_lstm_best_{dataset}"
    plot_kw: dict = {
        "best_epoch": history["best_epoch"],
        "stop_epoch": history["stop_epoch"],
        "early_stopped": history["early_stopped"],
    }
    import json

    if dataset == "pamap2":
        plot_kw["reg_summary"] = pamap2_plot_footer(history)
        plot_kw["suptitle"] = (
            f"PAMAP2 — {label}  |  LOSO fold {fold_num}  |  test subject {te_subj}"
        )
        hparams = build_pamap2_best_hyperparameters(
            history,
            n_classes=n_classes,
            fold=fold_num,
            test_subject=str(te_subj),
        )
        hparam_path = Path(METRICS_DIR) / "gnn_lstm_best_pamap2_hyperparameters.json"
        hparam_path.parent.mkdir(parents=True, exist_ok=True)
        with open(hparam_path, "w") as f:
            json.dump(hparams, f, indent=2)
        print(f"  Saved hyperparameters → {hparam_path}")

    out = plot_training_history(history, run_name=run_name, **plot_kw)

    meta_path = Path(METRICS_DIR) / f"{run_name}_loss_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset == "pamap2":
        with open(meta_path, "w") as f:
            json.dump(hparams, f, indent=2)
    else:
        with open(meta_path, "w") as f:
            json.dump({
                "model": label,
                "dataset": dataset,
                "fold": fold_num,
                "test_subject": str(te_subj),
                "loss_curve_run": {
                    "epochs_trained": history["stop_epoch"],
                    "best_epoch": history["best_epoch"],
                    "early_stopped": history["early_stopped"],
                    "best_val_loss": history["best_val_loss"],
                },
            }, f, indent=2)

    return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["pamap2", "hhar"],
                        choices=["pamap2", "hhar"])
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="PAMAP2: generate loss curves for every LOSO fold (9 plots + grid)",
    )
    args = parser.parse_args()

    saved = []
    if "pamap2" in args.datasets:
        if args.all_folds:
            saved.extend(run_pamap2_all_folds())
        else:
            saved.append(run_best_fold("pamap2", "attn_adj", max_per_subject=None))
    if "hhar" in args.datasets and not args.all_folds:
        saved.append(run_best_fold("hhar", "improved", max_per_subject=5000))
    print("\nBest GNN+LSTM loss curves:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
