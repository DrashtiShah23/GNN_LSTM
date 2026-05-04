# 10-minute presentation — 12 slides (DATA 245 rubric map)

Use **`results/plots/presentation_all_models_summary.png`** as your main results slide (all models, accuracy & macro F1 with ± fold std on PAMAP2 and HHAR). **`results/metrics/presentation_summary_table.csv`** for exact numbers in the deck.

Regenerate after new runs:

```bash
.venv/bin/python scripts/generate_presentation_summary_figure.py
```

---

## Slide 1 — Title + team
Project title, course, names, one line: “HAR under realistic subject-generalisation (LOSO).”

## Slide 2 — Motivation & real-world relevance
Why HAR matters (health, ergonomics, adaptive UI). **Problem:** models look strong on random splits but fail on new users → **LOSO** as the honest test.

## Slide 3 — Problem statement (clear)
Predict **activity class** from **wearable sensor windows**; focus on **cross-user generalisation** and **dataset / sensor heterogeneity**.

## Slide 4 — Dataset description
- **Sources:** UCI PAMAP2, UCI HHAR (cite in talk).
- **Inputs:** 50 Hz, sliding windows **128 × C** (C=18 PAMAP2, C=3 HHAR), **z-score per window**.
- **Outputs:** Multi-class labels (12 / 6). **Preprocessing:** resample, window, normalise, drop transients (brief).

## Slide 5 — **INPUT / OUTPUT (rubric — required)**
- **Input:** Time-series tensor **(128 timesteps × C channels)** per example; graph models add **node features + fixed adjacency**; GNN+LSTM uses **sequences of 10 graph snapshots** per subject.
- **Output:** **Class logits** → **softmax** → **argmax** → discrete activity (multi-class).
- **One diagram:** use ASCII from `docs/REPORT_SUPPLEMENT.md` or copy this:

```
IMU windows (N × 128 × C)  →  [ Classical: hand stats ]
                          →  [ Deep: LSTM / CNN1D on raw ]
                          →  [ GNN / GNN+LSTM on graph seq. ]
                                      ↓
                            softmax → predicted class
```

## Slide 6 — Proposed solution / what is new
- **Baselines:** SVM, RF, XGBoost on engineered features.
- **Deep:** CNN1D, LSTM, GNN-only, **GNN+LSTM (hybrid)** — spatial GCN + temporal LSTM.
- **What’s new:** explicit **sensor graph** + **temporal** modelling in one stack; **ablations** (learnable adj, flatten+LSTM) on PAMAP2.

## Slide 7 — Experimental design
- **Validation:** **LOSO** (9 folds), same for all models.
- **Compared:** classical + CNN1D + LSTM + GNN + GNN+LSTM (+ ablations on PAMAP2).
- **Hyperparameters:** Adam, lr, batch, early stopping (cite `src/config.py`); GNN+LSTM sweep optional.
- **Metrics:** **Accuracy**, **macro F1**, **balanced accuracy**; **± std over folds** where saved.

## Slide 8 — **Main results (the one big figure)**
Show **`presentation_all_models_summary.png`**. Say in one sentence: “Classical models and CNN1D are strong on PAMAP2; GNN+LSTM is our proposed hybrid — **highlight green bar** — compare to GNN-only and Flatten+LSTM on PAMAP2.”

## Slide 9 — Interpretation (NOT just numbers)
- **PAMAP2:** XGBoost / CNN1D / **Flatten+LSTM** often beat **GNN+LSTM** under LOSO → hybrid needs **better integration / tuning**, not only architecture.
- **HHAR:** fewer channels → **GNN-only** competitive; **error bars** show fold variance (hard subjects).
- Tie **std** to “deployment risk on new users.”

## Slide 10 — What worked vs failed
- **Worked:** LOSO protocol, classical baselines, GNN-only efficiency, Flatten+LSTM as strong temporal baseline.
- **Hard:** naive **cross-dataset transfer** (near chance), **GNN+LSTM** under default training vs best baselines.

## Slide 11 — Conclusion & impact
Takeaway: **evaluation design** matters as much as architecture; for wearables, **small GNN** or **strong classical** models can beat an under-tuned hybrid. Real-world: need **subject-stable** models before deployment.

## Slide 12 — Future work
Hyperparameter search for GNN+LSTM, domain adaptation for transfer, attention on edges, more data / full HHAR, on-device latency (see `model_profiling`).

---

### Timing tip (~10 min)
Slides 1–4 ≈ 2.5 min · 5–7 ≈ 3 min · **8** ≈ 2 min · 9–12 ≈ 2.5 min. Rehearse slide 5 (I/O) and slide 8 (figure) most.
