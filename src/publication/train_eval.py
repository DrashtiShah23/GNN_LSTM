"""Unified training and evaluation for publication experiments."""

from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.dataset import HARSequenceDataset, HARWindowDataset2D
from src.train import get_device, set_seed
from src.publication.models_registry import build_adj
from src.publication.splits import (
    assert_loso_no_leakage,
    loso_fold_splits,
    random_holdout_split,
    subject_val_split,
)
from src.publication.metrics import compute_full_metrics


def build_sequence_subjects(subjects: np.ndarray, seq_len: int = 10) -> np.ndarray:
    out = []
    for s in np.unique(subjects):
        mask = np.where(subjects == s)[0]
        n_seqs = len(range(0, len(mask) - seq_len + 1, seq_len))
        out.extend([s] * n_seqs)
    return np.array(out)


def _make_dataset(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    dataset: str,
    seq_len: int,
):
    if model_type == "window":
        return HARWindowDataset2D(X, y), subjects
    ds = HARSequenceDataset(X, y, subjects=subjects, dataset=dataset, seq_len=seq_len)
    seq_subj = build_sequence_subjects(subjects, seq_len)
    return ds, seq_subj


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    use_adj: bool,
    cfg_train: dict,
    device: torch.device,
) -> nn.Module:
    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg_train["learning_rate"],
        weight_decay=cfg_train["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit = nn.CrossEntropyLoss()
    best_loss, best_state, pat = float("inf"), None, 0

    for _ in range(cfg_train["num_epochs"]):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            if use_adj:
                x, adj, yb = batch
                loss = crit(model(x.to(device), adj.to(device)), yb.to(device))
            else:
                x, yb = batch
                loss = crit(model(x.to(device)), yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg_train.get("grad_clip", 1.0))
            opt.step()

        model.eval()
        vl, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                if use_adj:
                    x, adj, yb = batch
                    logits = model(x.to(device), adj.to(device))
                else:
                    x, yb = batch
                    logits = model(x.to(device))
                yb = yb.to(device)
                vl += crit(logits, yb).item() * len(yb)
                n += len(yb)
        vl /= max(n, 1)
        sched.step(vl)
        if vl < best_loss:
            best_loss, best_state, pat = vl, copy.deepcopy(model.state_dict()), 0
        else:
            pat += 1
            if pat >= cfg_train["patience"]:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_adj: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    yt, yp, probs = [], [], []
    for batch in loader:
        if use_adj:
            x, adj, y = batch
            logits = model(x.to(device), adj.to(device))
        else:
            x, y = batch
            logits = model(x.to(device))
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
        yp.append(logits.argmax(1).cpu().numpy())
        yt.append(y.numpy())
    return (
        np.concatenate(yt),
        np.concatenate(yp),
        np.concatenate(probs),
    )


def finetune_model(
    model: nn.Module,
    cal_loader: DataLoader,
    *,
    use_adj: bool,
    strategy: str,
    cfg_train: dict,
    device: torch.device,
) -> nn.Module:
    if strategy == "classifier_head_only":
        for name, param in model.named_parameters():
            param.requires_grad = "classifier" in name
    else:
        for param in model.parameters():
            param.requires_grad = True

    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg_train["learning_rate"] * 0.1,
        weight_decay=cfg_train["weight_decay"],
    )
    crit = nn.CrossEntropyLoss()
    for _ in range(min(20, cfg_train["num_epochs"])):
        model.train()
        for batch in cal_loader:
            opt.zero_grad()
            if use_adj:
                x, adj, yb = batch
                loss = crit(model(x.to(device), adj.to(device)), yb.to(device))
            else:
                x, yb = batch
                loss = crit(model(x.to(device)), yb.to(device))
            loss.backward()
            opt.step()
    return model


def run_evaluation(
    *,
    model_factory: Callable[[], nn.Module],
    use_adj: bool,
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    dataset: str,
    protocol: str,
    cfg: dict,
    seed: int = 42,
    max_folds: int | None = None,
    pretrain_model: nn.Module | None = None,
    cal_indices: np.ndarray | None = None,
    test_indices_override: np.ndarray | None = None,
) -> dict[str, Any]:
    device = get_device()
    cfg_train = cfg["training"]
    seq_len = cfg["sequence"]["seq_len"]
    batch_size = cfg_train["batch_size"]
    val_frac = cfg_train["val_fraction"]

    torch_ds, eval_subjects = _make_dataset(model_type, X, y, subjects, dataset, seq_len)

    all_true, all_pred, all_probs, all_subj = [], [], [], []
    fold_metrics = []

    if protocol == "random_holdout":
        set_seed(seed)
        tr_idx, te_idx = random_holdout_split(eval_subjects, seed=seed)
        if test_indices_override is not None:
            te_idx = test_indices_override
        tr_idx, val_idx = subject_val_split(tr_idx, eval_subjects, val_frac)
        loaders = _loaders(torch_ds, tr_idx, val_idx, te_idx, batch_size)
        model = (copy.deepcopy(pretrain_model) if pretrain_model else model_factory()).to(device)
        if cal_indices is not None and len(cal_indices) > 0:
            cal_loader = DataLoader(Subset(torch_ds, cal_indices), batch_size=batch_size, shuffle=True)
            model = finetune_model(model, cal_loader, use_adj=use_adj, strategy="classifier_head_only", cfg_train=cfg_train, device=device)
        model = train_model(model, loaders[0], loaders[1], use_adj=use_adj, cfg_train=cfg_train, device=device)
        yt, yp, pr = predict(model, loaders[2], device, use_adj)
        all_true, all_pred = yt, yp
        all_probs = [pr]
        all_subj = eval_subjects[te_idx]
        fold_metrics.append(compute_full_metrics(yt, yp, n_classes=int(y.max()) + 1))
    else:
        for tr_idx, te_idx, test_subj, fi in loso_fold_splits(eval_subjects, max_folds):
            assert_loso_no_leakage(eval_subjects, tr_idx, te_idx, test_subj)
            set_seed(seed + fi)
            if test_indices_override is not None:
                te_idx = test_indices_override
            tr, val_idx = subject_val_split(tr_idx, eval_subjects, val_frac)
            loaders = _loaders(torch_ds, tr, val_idx, te_idx, batch_size)
            model = (copy.deepcopy(pretrain_model) if pretrain_model else model_factory()).to(device)
            if cal_indices is not None and len(cal_indices) > 0:
                cal_loader = DataLoader(Subset(torch_ds, cal_indices), batch_size=batch_size, shuffle=True)
                model = finetune_model(model, cal_loader, use_adj=use_adj, strategy="classifier_head_only", cfg_train=cfg_train, device=device)
            model = train_model(model, loaders[0], loaders[1], use_adj=use_adj, cfg_train=cfg_train, device=device)
            yt, yp, pr = predict(model, loaders[2], device, use_adj)
            fold_metrics.append({
                **compute_full_metrics(yt, yp, n_classes=int(y.max()) + 1),
                "fold": fi,
                "test_subject": str(test_subj),
            })
            all_true.extend(yt)
            all_pred.extend(yp)
            all_probs.append(pr)
            all_subj.extend([test_subj] * len(yt))

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    probs = np.vstack(all_probs) if len(all_probs) else np.empty((0, int(y.max()) + 1))
    subj_arr = np.array(all_subj) if not isinstance(all_subj, np.ndarray) else all_subj

    agg = compute_full_metrics(y_true, y_pred, n_classes=int(y.max()) + 1)
    return {
        "aggregate": agg,
        "fold_metrics": fold_metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
        "subjects": subj_arr,
    }


def _loaders(ds, tr_idx, val_idx, te_idx, batch_size):
    return (
        DataLoader(Subset(ds, tr_idx), batch_size=batch_size, shuffle=True),
        DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False),
        DataLoader(Subset(ds, te_idx), batch_size=batch_size, shuffle=False),
    )
