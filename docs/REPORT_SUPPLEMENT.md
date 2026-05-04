# Final report supplement (DATA 245)

Use this section verbatim or adapted in the LaTeX/PDF report. It covers rubric gaps from the midway checklist: explicit I/O, dataset provenance, pipeline diagram, honest GNN+LSTM discussion, what worked vs failed, and impact.

---

## 1. Input / output specification

**Input (all deep sequence / graph models on raw windows)**  
- Tensor shape: `(batch, 128, C)` per window before batching, where `128` is `WINDOW_SIZE` at 50 Hz (~2.56 s).  
- **HHAR:** `C = 3` (smartphone/watch accelerometer `x, y, z` after preprocessing).  
- **PAMAP2:** `C = 18` (wrist + chest + ankle, each with 3-axis accelerometer + 3-axis gyroscope from protocol columns).  
- Each channel is **z-score normalised per window** (mean 0, std 1 along time) during preprocessing, so the model receives standardised IMU dynamics, not raw sensor units.

**Graph models (GNN / GNN+LSTM)**  
- Each window is mapped to **node features** `(n_nodes, F)`: PAMAP2 has `n_nodes = 3`, `F = 36` (6 statistics × 6 channels per body site). HHAR has `n_nodes = 2` (phone, watch), `F = 18`.  
- **GNN+LSTM** consumes sequences of `seq_len = 10` consecutive graph snapshots per training item, built **within subject** only (`HARSequenceDataset`).

**Output**  
- **HHAR:** 6 discrete activity classes (`bike`, `sit`, `stand`, `walk`, `stairsup`, `stairsdown`).  
- **PAMAP2:** 12 classes after dropping transient labels (contiguous indices map to the activity ids listed in `src.config.PAMAP2_ACTIVITIES`).  
- Model head: **multi-class logits** → **softmax** → **argmax** for the predicted class; training minimises **cross-entropy**.

**Classical baselines (SVM / RF / XGBoost)**  
- Input is **not** raw windows: `extract_features` builds a **108-D** vector per window on PAMAP2 (`6 stats × 18 channels`) and analogous size on HHAR — see `src/baselines.py`.

---

## 2. Input → model → output (ASCII diagram)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ RAW IMU streams (per dataset)                                            │
│   resample → 50 Hz → sliding window 128 @ 50% overlap → z-score / ch   │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │  X: (N, 128, C)
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
   ┌──────────────┐         ┌──────────────┐         ┌──────────────────┐
   │ flatten +    │         │ window→node │         │ window→node +    │
   │ stats (ML)   │         │ features    │         │ sequence stack   │
   └──────┬───────┘         └──────┬───────┘         │ (len=10)         │
          │                        │                 └────────┬─────────┘
          ▼                        ▼                          ▼
   ┌──────────────┐         ┌──────────────┐         ┌──────────────────┐
   │ SVM / RF /   │         │ GCN layers   │         │ per step: GCN   │
   │ XGBoost      │         │ + mean pool  │         │ → vec; LSTM     │
   └──────┬───────┘         └──────┬───────┘         │ → MLP logits    │
          │                        │                 └────────┬─────────┘
          ▼                        ▼                          ▼
        logits                 logits                    logits
          │                        │                          │
          └────────────────────────┼──────────────────────────┘
                                   ▼
                          softmax → argmax → ŷ  (class index)
```

---

## 3. Dataset description (citations and targets)

| Dataset | Source | Inputs used | Target |
|--------|--------|-------------|--------|
| **PAMAP2** | [UCI PAMAP2 Physical Activity Monitoring](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) — Reiss & Stricker | IMU at wrist, chest, ankle; 18 channels (acc+gyro) per window | Activity label (12 classes in our pipeline after filtering transients) |
| **HHAR** | [UCI Heterogeneity Human Activity Recognition](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition) — Stisen et al. | 3-axis accelerometer from phone/watch streams | 6 activity labels |

**Preprocessing (enumerate)**  
1. Resample sensor streams to **50 Hz**.  
2. **Sliding windows:** length `128`, stride `64` (50% overlap).  
3. **Per-window z-score** normalisation along time for each channel.  
4. Drop invalid / transient segments per dataset rules (HHAR: restrict to known `gt` activities; PAMAP2: drop `activity_id` not in label dictionary; transient handling in loader).  
5. **HHAR subsampling (optional):** cap windows per subject (e.g. 5000) for tractable LOSO training — documented where applied.

---

## 4. What is “new” in GNN+LSTM — and why CNN1D / Flatten+LSTM can still win

**Claim:** The hybrid stacks a **spatial encoder** (GCN over sensor graph) with a **temporal encoder** (LSTM over sequences of graph embeddings), so it should outperform a **purely convolutional** temporal model (CNN1D) when cross-sensor structure and temporal evolution both matter.

**What the numbers showed (PAMAP2 LOSO, representative):** CNN1D and **Flatten+LSTM** (ablation C in `experiments.py`) can exceed **GNN+LSTM**. GNN-only can even exceed GNN+LSTM.

**Honest interpretation (not just citing accuracy):**  
1. **Temporal aggregation:** Flatten+LSTM feeds **rich per-window node statistics** straight into an LSTM without the GCN’s bottleneck. That preserves channel-specific dynamics the GCN may **blur** via mean pooling and message passing.  
2. **GNN+LSTM interface:** The hybrid runs **one GCN forward per time step**, then mean-pools nodes → a **short** embedding before the LSTM. If the GCN loses discriminative detail, the LSTM only sees a **noisy low-dimensional trajectory**, which is harder to optimise than raw or flattened structured features.  
3. **Optimisation / capacity:** GNN+LSTM has more moving parts (depth, dropout, sequence length). Under LOSO, data per fold are limited; a **mis-tuned** hybrid can underfit the temporal branch while the GNN branch **adds variance** — consistent with “GNN-only > GNN+LSTM” on some splits.  
4. **CNN1D** sees the full `(128, C)` tensor with locality bias; it is a **strong inductive bias** for IMU spectra and beats an under-tuned graph hybrid.

**Takeaway for the report:** The architectural *hypothesis* (spatial + temporal) remains plausible, but this implementation shows **integration and tuning** dominate: the GCN stage must **preserve** information the LSTM needs, or the hybrid will trail simpler pipelines.

---

## 5. What worked vs what failed

| Worked | Failed or incomplete (midway promises) |
|--------|----------------------------------------|
| LOSO protocol; subject-only sequences for GNN+LSTM | **Pure** GNN+LSTM beating CNN1D on PAMAP2 (not achieved with default hparams) |
| Leakage story (holdout vs LOSO) | Cross-dataset transfer was previously only a **side-by-side bar** of in-dataset LOSO — now addressed by `scripts/cross_dataset_transfer.py` |
| Classical baselines on hand-crafted stats | Full GNN+LSTM hyperparameter grid is expensive; use `scripts/gnn_lstm_hparam_sweep.py` |
| GNN-only as a small, strong wearable-friendly model | LIME on neural models was stub-only; use `scripts/run_lime_pamap2_lstm.py` |
| Graph ablations (learnable adj, flatten+LSTM) | Attention adjacency was future work; **implemented** as `GNNAttentionAdjModel` + `experiments.py --exp attention` |

---

## 6. Confusion pairs (qualitative)

Use `results/metrics/confusion_pairs_pamap2.json` from `scripts/loso_significance_and_pairs.py`. **Sitting vs lying** and **walking vs Nordic walking** are structurally similar in acceleration; expect elevated off-diagonal mass. Tie numbers to **per-class F1** from `error_analysis.json` where available.

---

## 7. Statistical significance

When `pamap2_deep_models.json` contains a `"folds"` list per model (after re-running `scripts/run_full_pipeline.py`), `scripts/loso_significance_and_pairs.py` writes **paired t-tests** on matched held-out subjects. If `folds` is absent, re-run the pipeline once to populate fold-wise accuracy.

---

## 8. Real-world impact (conclusion boilerplate)

Human activity recognition from wearables supports **context-aware health** (sedentary behaviour, gait, adherence to rehab), **assistive systems**, and **low-power on-device inference**. LOSO evaluation approximates **new-user deployment**; cross-dataset transfer (`cross_dataset_transfer.json`) stress-tests **sensor and protocol shift**. The strongest *practical* story in this codebase is often **small GNN-only** or **classical models on statistics** under LOSO, with **Flatten+LSTM** as an empirically strong deep baseline — useful guidance for engineers choosing models under compute and privacy constraints.

---

## 9. Script index (post–midway report)

| Script | Purpose |
|--------|---------|
| `scripts/cross_dataset_transfer.py` | Train HHAR→PAMAP2 and PAMAP2→HHAR (3-ch wrist alignment + coarse labels) |
| `scripts/run_lime_pamap2_lstm.py` | LIME explanations + `lime_lstm_pamap2.json` / `.png` |
| `scripts/loso_significance_and_pairs.py` | Paired t-tests + confusion-pair JSON |
| `scripts/gnn_lstm_hparam_sweep.py` | GNN+LSTM grid (`--quick` for smoke) |
| `scripts/experiments.py --exp attention` | Attention-gated adjacency GNN ablation |
| `scripts/experiments.py --exp transfer` | Runs cross-dataset transfer module |
| `scripts/refresh_all_results.sh` | Full sequence: LOSO pipeline → export → significance → transfer → LIME (set `HAR_FORCE_DEVICE=cpu` on macOS if MPS dies mid-run) |
| `scripts/cross_dataset_transfer.py --lstm` | Optional slow LSTM transfer (default is RF-only after HHAR subsampling) |

**macOS note:** If `run_full_pipeline.py` stops around LOSO fold 2 on Apple Silicon, export `HAR_FORCE_DEVICE=cpu` before running (see `get_device()` in `src/train.py`).
