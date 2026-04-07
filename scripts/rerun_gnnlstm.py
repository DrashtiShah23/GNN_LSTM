"""
Re-runs only:
  1. PAMAP2 GNN+LSTM  (fix subject-based val split)
  2. HHAR all 3 models (5000 windows/subject cap)

Then merges with existing PAMAP2 lstm/gnn results and generates all final plots.
"""

from __future__ import annotations
import sys, json, time, copy, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.config import (
    PROCESSED_DIR, RESULTS_DIR, MODELS_DIR, PLOTS_DIR, METRICS_DIR,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE, SEED,
    GCN_OUTPUT_DIM, PAMAP2_NODE_FEAT_DIM, HHAR_NODE_FEAT_DIM,
    PAMAP2_ACTIVITIES,
)
from src.models import GNNLSTMModel, LSTMOnlyModel, GNNOnlyModel
from src.dataset import HARWindowDataset, HARGraphDataset, HARSequenceDataset
from src.graph_construction import build_pamap2_adj, build_hhar_adj
from src.train import get_device, set_seed

for d in [MODELS_DIR, PLOTS_DIR, METRICS_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)


def remap_labels(y):
    classes = np.unique(y)
    mapping = {int(old): int(new) for new, old in enumerate(classes)}
    return np.vectorize(mapping.__getitem__)(y), mapping


def train_one_fold(model, tr_loader, val_loader, use_adj, device, tag=""):
    opt   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit  = nn.CrossEntropyLoss()
    best_acc, best_state, patience_cnt = 0.0, None, 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        t0 = time.time()
        for batch in tr_loader:
            opt.zero_grad()
            if use_adj:
                x, adj, y = batch; x, adj, y = x.to(device), adj.to(device), y.to(device)
                loss = crit(model(x, adj), y)
            else:
                x, y = batch; x, y = x.to(device), y.to(device)
                loss = crit(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if use_adj:
                    x, adj, yb = batch; x, adj, yb = x.to(device), adj.to(device), yb.to(device)
                    logits = model(x, adj)
                else:
                    x, yb = batch; x, yb = x.to(device), yb.to(device)
                    logits = model(x)
                val_loss += crit(logits, yb).item() * len(yb)
                correct += (logits.argmax(1) == yb).sum().item()
                total   += len(yb)
        if total == 0:
            continue
        val_acc = correct / total
        sched.step(val_loss / total)
        print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | Val Acc: {val_acc:.4f} | Time: {time.time()-t0:.1f}s", flush=True)

        if val_acc > best_acc:
            best_acc, best_state, patience_cnt = val_acc, copy.deepcopy(model.state_dict()), 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stop @ epoch {epoch}. Best val acc: {best_acc:.4f}", flush=True)
                break

    if best_state: model.load_state_dict(best_state)
    print(f"  Best val acc: {best_acc:.4f}", flush=True)
    return model


def loso_one_model(X, y, subjects, model_type, dataset_name,
                   n_classes, node_feat_dim, n_nodes, adj_builder, tag):
    device = get_device()
    set_seed(SEED)
    all_true, all_pred = [], []
    unique_subjs = np.unique(subjects)

    for fold_i, test_subj in enumerate(unique_subjs, 1):
        print(f"\n── Fold {fold_i}/{len(unique_subjs)}: test={test_subj} ──", flush=True)
        train_mask = subjects != test_subj
        test_mask  = ~train_mask
        X_tr, y_tr, s_tr = X[train_mask], y[train_mask], subjects[train_mask]
        X_te, y_te        = X[test_mask],  y[test_mask]

        use_adj = model_type in ("gnn", "gnn_lstm")

        if model_type == "lstm":
            # Subject-based val: hold out smallest subject (still hundreds of windows)
            train_subjs = np.unique(s_tr)
            counts = {s: int(np.sum(s_tr == s)) for s in train_subjs}
            val_subj = min(counts, key=counts.__getitem__)
            vm = s_tr == val_subj
            X_val, y_val = X_tr[vm],  y_tr[vm]
            X_tr2, y_tr2 = X_tr[~vm], y_tr[~vm]
            tr_l = DataLoader(HARWindowDataset(X_tr2, y_tr2), BATCH_SIZE, shuffle=True,  num_workers=0)
            va_l = DataLoader(HARWindowDataset(X_val, y_val), BATCH_SIZE, shuffle=False, num_workers=0)
            te_l = DataLoader(HARWindowDataset(X_te,  y_te),  BATCH_SIZE, shuffle=False, num_workers=0)

        elif model_type == "gnn":
            # Same subject-based hold-out
            train_subjs = np.unique(s_tr)
            counts = {s: int(np.sum(s_tr == s)) for s in train_subjs}
            val_subj = min(counts, key=counts.__getitem__)
            vm = s_tr == val_subj
            X_val, y_val = X_tr[vm],  y_tr[vm]
            X_tr2, y_tr2 = X_tr[~vm], y_tr[~vm]
            tr_l = DataLoader(HARGraphDataset(X_tr2, y_tr2, dataset=dataset_name), BATCH_SIZE, shuffle=True,  num_workers=0)
            va_l = DataLoader(HARGraphDataset(X_val, y_val, dataset=dataset_name), BATCH_SIZE, shuffle=False, num_workers=0)
            te_l = DataLoader(HARGraphDataset(X_te,  y_te,  dataset=dataset_name), BATCH_SIZE, shuffle=False, num_workers=0)

        else:  # gnn_lstm — use last 20% of each training subject's windows as val
            val_idx_list, tr_idx_list = [], []
            for subj in np.unique(s_tr):
                idx = np.where(s_tr == subj)[0]  # already in order
                if len(idx) < 20:   # too few windows: all go to train (skip val for this subj)
                    tr_idx_list.append(idx)
                    continue
                split = max(10, int(0.8 * len(idx)))
                tr_idx_list.append(idx[:split])
                val_idx_list.append(idx[split:])
            tr_all  = np.concatenate(tr_idx_list)  if tr_idx_list  else np.array([], dtype=int)
            val_all = np.concatenate(val_idx_list) if val_idx_list else tr_all
            X_tr2, y_tr2, s_tr2 = X_tr[tr_all],  y_tr[tr_all],  s_tr[tr_all]
            X_val, y_val, s_val = X_tr[val_all], y_tr[val_all], s_tr[val_all]
            ds_tr  = HARSequenceDataset(X_tr2, y_tr2, subjects=s_tr2, dataset=dataset_name)
            ds_val = HARSequenceDataset(X_val, y_val, subjects=s_val,  dataset=dataset_name)
            ds_te  = HARSequenceDataset(X_te,  y_te,  dataset=dataset_name)
            if len(ds_tr)  == 0: ds_tr  = HARSequenceDataset(X_tr, y_tr, subjects=s_tr, dataset=dataset_name)
            if len(ds_val) == 0: ds_val = ds_tr
            print(f"  GNN+LSTM: {len(ds_tr)} train seqs | {len(ds_val)} val seqs | {len(ds_te)} test seqs")
            tr_l = DataLoader(ds_tr,  BATCH_SIZE, shuffle=True,  num_workers=0)
            va_l = DataLoader(ds_val, BATCH_SIZE, shuffle=False, num_workers=0)
            te_l = DataLoader(ds_te,  BATCH_SIZE, shuffle=False, num_workers=0)

        if model_type == "lstm":
            m = LSTMOnlyModel(X_tr.shape[1] * X_tr.shape[2], n_classes).to(device)
        elif model_type == "gnn":
            m = GNNOnlyModel(node_feat_dim, n_nodes, n_classes).to(device)
        else:
            m = GNNLSTMModel(node_feat_dim, n_nodes, n_classes).to(device)

        m = train_one_fold(m, tr_l, va_l, use_adj, device)
        torch.save(m.state_dict(), Path(MODELS_DIR) / f"{tag}_fold{fold_i}.pt")

        m.eval()
        ft, fp = [], []
        with torch.no_grad():
            for batch in te_l:
                if use_adj:
                    x, adj, yb = batch; x,adj,yb = x.to(device),adj.to(device),yb.to(device)
                    fp.extend(m(x,adj).argmax(1).cpu().tolist())
                else:
                    x, yb = batch; x,yb = x.to(device),yb.to(device)
                    fp.extend(m(x).argmax(1).cpu().tolist())
                ft.extend(yb.cpu().tolist())
        print(f"  Fold test acc: {accuracy_score(ft,fp):.4f}", flush=True)
        all_true.extend(ft); all_pred.extend(fp)

    acc = accuracy_score(all_true, all_pred)
    f1  = f1_score(all_true, all_pred, average="macro", zero_division=0)
    bal = balanced_accuracy_score(all_true, all_pred)
    print(f"\n[{tag}] LOSO Acc={acc:.4f}  F1={f1:.4f}  BalAcc={bal:.4f}", flush=True)
    np.save(Path(METRICS_DIR) / f"{tag}_y_true.npy", np.array(all_true))
    np.save(Path(METRICS_DIR) / f"{tag}_y_pred.npy", np.array(all_pred))
    return {"accuracy": acc, "macro_f1": f1, "balanced_acc": bal}


def load_and_cap(name, max_per_subj=None):
    base = Path(PROCESSED_DIR)
    X = np.load(base / f"{name}_X.npy")
    y = np.load(base / f"{name}_y.npy")
    s = np.load(base / f"{name}_subjects.npy")
    y, mapping = remap_labels(y)
    if max_per_subj:
        rng = np.random.default_rng(SEED)
        keep = []
        for subj in np.unique(s):
            idx = np.where(s == subj)[0]
            if len(idx) > max_per_subj:
                idx = np.sort(rng.choice(idx, max_per_subj, replace=False))
            keep.append(idx)
        keep = np.concatenate(keep)
        X, y, s = X[keep], y[keep], s[keep]
    print(f"[{name}] X={X.shape}  classes={len(np.unique(y))}  subjs={np.unique(s).tolist()}")
    return X, y, s, mapping


def main():
    set_seed(SEED)

    # ── 1. Re-run PAMAP2 GNN+LSTM only ────────────────────────────────────────
    print("\n" + "="*60)
    print("  PAMAP2 — GNN+LSTM (fixed val split)")
    print("="*60)
    Xp, yp, sp, _ = load_and_cap("pamap2")
    gnnlstm_p = loso_one_model(
        Xp, yp, sp, "gnn_lstm", "pamap2",
        n_classes=len(np.unique(yp)),
        node_feat_dim=PAMAP2_NODE_FEAT_DIM,
        n_nodes=3, adj_builder=build_pamap2_adj,
        tag="gnnlstm_pamap2",
    )

    # Load existing PAMAP2 lstm & gnn results
    p2_path = Path(METRICS_DIR) / "pamap2_deep_models.json"
    with open(p2_path) as f:
        pamap2_deep = json.load(f)
    pamap2_deep["gnn_lstm"] = gnnlstm_p
    with open(p2_path, "w") as f:
        json.dump(pamap2_deep, f, indent=2)
    print(f"\nUpdated pamap2_deep_models.json")

    # ── 2. HHAR — all 3 models (capped) ───────────────────────────────────────
    print("\n" + "="*60)
    print("  HHAR — full LOSO (5000 windows/subject)")
    print("="*60)
    Xh, yh, sh, _ = load_and_cap("hhar", max_per_subj=5000)
    n_cls_h = len(np.unique(yh))
    hhar_deep = {}
    for mt in ["lstm", "gnn", "gnn_lstm"]:
        tag = f"{mt.replace('_','')}_{{}}" .format("hhar")
        print(f"\n{'─'*50}")
        print(f"  {mt.upper()} — HHAR LOSO")
        print(f"{'─'*50}")
        res = loso_one_model(
            Xh, yh, sh, mt, "hhar",
            n_classes=n_cls_h,
            node_feat_dim=HHAR_NODE_FEAT_DIM,
            n_nodes=2, adj_builder=build_hhar_adj,
            tag=tag,
        )
        hhar_deep[mt] = res

    with open(Path(METRICS_DIR) / "hhar_deep_models.json", "w") as f:
        json.dump(hhar_deep, f, indent=2)
    print("\nSaved hhar_deep_models.json")

    # ── 3. Load baselines ──────────────────────────────────────────────────────
    def load_bl(name):
        for fname in [f"{name}_baselines.json", f"{name.upper()}_baselines.json"]:
            p = Path(METRICS_DIR) / fname
            if p.exists():
                with open(p) as f: return json.load(f)
        return {}

    pamap2_bl = load_bl("pamap2")
    hhar_bl   = load_bl("hhar")

    # ── 4. Final plots ─────────────────────────────────────────────────────────
    print("\n\nGenerating final plots …")
    generate_plots(pamap2_deep, pamap2_bl, hhar_deep, hhar_bl)
    print("\n✅  All done!")


# ============================================================================
# Plot generation
# ============================================================================

def generate_plots(pamap2_deep, pamap2_bl, hhar_deep, hhar_bl):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    plots = Path(PLOTS_DIR)

    from src.config import HHAR_ACTIVITIES
    _PA = PAMAP2_ACTIVITIES  # capture in local scope to avoid closure issues

    # ── confusion matrices ────────────────────────────────────────────────────
    def activity_names(dataset_name, n_cls):
        if dataset_name == "pamap2":
            y_raw = np.load(Path(PROCESSED_DIR) / "pamap2_y.npy")
            _, mp  = remap_labels(y_raw)
            inv    = {v: _PA.get(int(k), str(k)) for k, v in mp.items()}
            return [inv.get(i, str(i)) for i in range(n_cls)]
        else:
            return HHAR_ACTIVITIES[:n_cls]

    for ds in ["pamap2", "hhar"]:
        for mt in ["lstm", "gnn", "gnnlstm"]:
            tag = f"{mt}_{ds}"
            tp  = Path(METRICS_DIR) / f"{tag}_y_true.npy"
            pp  = Path(METRICS_DIR) / f"{tag}_y_pred.npy"
            if not tp.exists(): continue
            yt = np.load(tp); yp = np.load(pp)
            names = activity_names(ds, len(np.unique(yt)))
            cm = confusion_matrix(yt, yp, normalize="true")
            sz = max(8, len(names))
            fig, ax = plt.subplots(figsize=(sz, sz-1))
            sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                        xticklabels=names, yticklabels=names, ax=ax, annot_kws={"size": 7})
            label_map = {"lstm": "LSTM-only", "gnn": "GNN-only", "gnnlstm": "GNN+LSTM"}
            ax.set_title(f"{label_map.get(mt,mt)} — {ds.upper()} (LOSO)", fontsize=12)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            plt.tight_layout()
            out = plots / f"cm_{tag}.png"
            fig.savefig(out, dpi=150); plt.close(fig)
            print(f"  Saved {out}")

    # ── comparison charts ──────────────────────────────────────────────────────
    def comp_chart(deep, bl, ds_name):
        name_map_bl  = {"SVM": "SVM", "RandomForest": "Random Forest", "XGBoost": "XGBoost"}
        name_map_dl  = {"lstm": "LSTM-only", "gnn": "GNN-only", "gnn_lstm": "GNN+LSTM"}
        models, accs, f1s = [], [], []
        for k, v in bl.items():
            models.append(name_map_bl.get(k, k))
            accs.append(v.get("mean_accuracy", v.get("accuracy", 0)) * 100)
            f1s.append(v.get("mean_macro_f1",  v.get("macro_f1",  0)) * 100)
        for k, v in deep.items():
            models.append(name_map_dl.get(k, k))
            accs.append(v["accuracy"] * 100)
            f1s.append(v["macro_f1"]  * 100)
        if not models: return
        x = np.arange(len(models)); w = 0.35
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = ["steelblue"] * len(bl) + ["darkorange", "green", "crimson"]
        for ax, vals, metric in zip(axes, [accs, f1s], ["Accuracy (%)", "Macro F1 (%)"]):
            bars = ax.bar(x, vals, 0.6, color=colors[:len(models)], alpha=0.85)
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
            ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right")
            ax.set_ylabel(metric); ax.set_ylim(0, 110)
            ax.set_title(f"{metric} — {ds_name.upper()}")
        fig.suptitle(f"Model Comparison — {ds_name.upper()} LOSO", fontweight="bold")
        plt.tight_layout()
        out = plots / f"model_comparison_{ds_name}.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"  Saved {out}")

    comp_chart(pamap2_deep, pamap2_bl, "pamap2")
    if hhar_deep: comp_chart(hhar_deep, hhar_bl, "hhar")

    # ── cross-dataset comparison ───────────────────────────────────────────────
    if pamap2_deep and hhar_deep:
        name_map = {"lstm": "LSTM-only", "gnn": "GNN-only", "gnn_lstm": "GNN+LSTM"}
        common   = [k for k in pamap2_deep if k in hhar_deep]
        labels   = [name_map.get(k, k) for k in common]
        p_acc    = [pamap2_deep[k]["accuracy"] * 100 for k in common]
        h_acc    = [hhar_deep[k]["accuracy"]   * 100 for k in common]
        x = np.arange(len(labels)); w = 0.35
        fig, ax = plt.subplots(figsize=(9, 5))
        b1 = ax.bar(x-w/2, p_acc, w, label="PAMAP2", color="steelblue",  alpha=0.85)
        b2 = ax.bar(x+w/2, h_acc, w, label="HHAR",   color="darkorange", alpha=0.85)
        ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=9)
        ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 110)
        ax.set_title("Cross-Dataset Accuracy Comparison (LOSO)", fontweight="bold")
        ax.legend(); plt.tight_layout()
        out = plots / "cross_dataset_comparison.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"  Saved {out}")

    # ── SHAP ──────────────────────────────────────────────────────────────────
    try:
        import shap
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from src.baselines import extract_features

        X_raw = np.load(Path(PROCESSED_DIR) / "pamap2_X.npy")
        y_raw = np.load(Path(PROCESSED_DIR) / "pamap2_y.npy")
        y_rem, _ = remap_labels(y_raw)
        Xf = extract_features(X_raw)
        print("  Training RF for SHAP …", flush=True)
        pipe = Pipeline([("sc", StandardScaler()),
                         ("rf", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED))])
        pipe.fit(Xf, y_rem)
        bg = shap.sample(Xf, 300, random_state=SEED)
        exp = shap.TreeExplainer(pipe.named_steps["rf"])
        sv  = exp.shap_values(bg)
        if isinstance(sv, list):
            mean_abs = np.stack([np.abs(s) for s in sv]).mean(axis=(0, 1))
        else:
            mean_abs = np.abs(sv).mean(axis=(0, -1)) if sv.ndim == 3 else np.abs(sv).mean(0)
        top_k   = min(20, len(mean_abs))
        top_idx = np.argsort(mean_abs)[::-1][:top_k]

        # Build readable feature names
        n_ch = X_raw.shape[2]
        stat_names = ["mean", "std", "min", "max", "rms", "fft"]
        feat_names = [f"{stat}_ch{ch}" for stat in stat_names for ch in range(n_ch)]
        top_names  = [feat_names[i] if i < len(feat_names) else f"feat_{i}" for i in top_idx]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(np.arange(top_k), mean_abs[top_idx][::-1], color="steelblue", alpha=0.85)
        ax.set_yticks(np.arange(top_k))
        ax.set_yticklabels(top_names[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("SHAP Feature Importance — RF on PAMAP2 (top-20)", fontweight="bold")
        plt.tight_layout()
        out = plots / "shap_rf_pamap2.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"  Saved {out}")
    except Exception as e:
        print(f"  [WARN] SHAP skipped: {e}")

    # ── model profiling ────────────────────────────────────────────────────────
    try:
        device = get_device()
        Xp = np.load(Path(PROCESSED_DIR) / "pamap2_X.npy")
        yp = np.load(Path(PROCESSED_DIR) / "pamap2_y.npy")
        yr, _ = remap_labels(yp)
        nc = len(np.unique(yr))
        from src.graph_construction import build_pamap2_adj

        models_p = {
            "LSTM-only": (LSTMOnlyModel(Xp.shape[1]*Xp.shape[2], nc), False,
                          torch.zeros(1, Xp.shape[1]*Xp.shape[2]), None),
            "GNN-only":  (GNNOnlyModel(PAMAP2_NODE_FEAT_DIM, 3, nc), True,
                          torch.zeros(1, 3, PAMAP2_NODE_FEAT_DIM), build_pamap2_adj()),
            "GNN+LSTM":  (GNNLSTMModel(PAMAP2_NODE_FEAT_DIM, 3, nc), True,
                          torch.zeros(1, 10, 3, PAMAP2_NODE_FEAT_DIM), build_pamap2_adj()),
        }
        profile = {}
        for mname, (mdl, ua, di, da) in models_p.items():
            mdl.to(device); mdl.eval()
            di = di.to(device)
            params = sum(p.numel() for p in mdl.parameters())
            if ua:
                da = da.to(device)
                with torch.no_grad(): mdl(di, da)  # warm up
                t0 = time.perf_counter()
                for _ in range(100):
                    with torch.no_grad(): mdl(di, da)
            else:
                with torch.no_grad(): mdl(di)
                t0 = time.perf_counter()
                for _ in range(100):
                    with torch.no_grad(): mdl(di)
            lat = (time.perf_counter() - t0) / 100 * 1000
            profile[mname] = {"params": params, "latency_ms": round(lat, 3)}
            print(f"  {mname}: {params:,} params | {lat:.3f} ms/sample")

        with open(Path(METRICS_DIR) / "model_profiling.json", "w") as f:
            json.dump(profile, f, indent=2)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        names  = list(profile.keys())
        colors = ["steelblue", "darkorange", "green"]
        p_vals = [profile[n]["params"] / 1e6 for n in names]
        l_vals = [profile[n]["latency_ms"] for n in names]
        ax1.bar(names, p_vals, color=colors, alpha=0.85)
        ax1.set_ylabel("Parameters (M)"); ax1.set_title("Model Size")
        for i, v in enumerate(p_vals): ax1.text(i, v+0.01, f"{v:.2f}M", ha="center", fontsize=9)
        ax2.bar(names, l_vals, color=colors, alpha=0.85)
        ax2.set_ylabel("Latency (ms/sample)"); ax2.set_title("Inference Latency")
        for i, v in enumerate(l_vals): ax2.text(i, v+0.0001, f"{v:.3f}", ha="center", fontsize=9)
        fig.suptitle("Model Profiling — PAMAP2", fontweight="bold")
        plt.tight_layout()
        out = plots / "model_profiling.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"  Saved {out}")
    except Exception as e:
        print(f"  [WARN] Profiling skipped: {e}")


if __name__ == "__main__":
    main()
