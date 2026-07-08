# PAMAP2 Protocol12 Complete Experiment Report

This report summarizes **all PAMAP2 protocol12 experiments we conducted**, not only the final top4 deliverable subset.

The final top4 folder remains the clean DOCX deliverable package, but it is not the full history of experiments. The full evidence includes the original canonical comparison, v2 improved-model ablation, v3 residual models, the all-model seven-experiment audit, and the final top4 DOCX completion.

## Result Roots Used

| Result root | Purpose | Status |
|---|---|---|
| `results/canonical_protocol_only/core_comparison` | Original canonical protocol12 comparison: baselines plus repo deep models | Completed |
| `results/canonical_protocol_only_v2/core_comparison` | v2 improved-model ablation on `acc16_gyro_hr` LOSO | Completed |
| `results/canonical_protocol_only_v3/core_comparison` | v3 residual GNN-LSTM models across all feature sets/protocols | Completed |
| `results/canonical_protocol12_seven_experiments` | All-model seven-experiment audit | Completed |
| `results/canonical_protocol12_seven_experiments_top4` | Final DOCX-standard top4 deliverable, including non-overlapping windows | Completed, `missing=[]` |

## Model Scope Actually Run

### Original v1 Core Comparison

The original canonical comparison included these deep models:

- `cnn`
- `lstm`
- `gnn`
- `gnn_learnable_adj`
- `gnn_attention_adj`
- `gnn_lstm`
- `gnn_flatten_lstm`
- `improved_gnn_lstm`
- `improved_gnn_lstm_attn_adj`

It also included canonical baselines. The strongest baselines used later were `random_forest` and `knn_k5`, but the all-model audit also preserved additional baseline evidence.

### V2 Improved-Model Ablation

The v2 ablation tested:

- `improved_gnn_lstm`
- `improved_gnn_lstm_attn_adj`

Scope: `acc16_gyro_hr`, LOSO, variant `v2_norm_balanced_ls005`.

### V3 Residual Models

The v3 residual architecture tested:

- `improved_gnn_lstm_res`
- `improved_gnn_lstm_attn_adj_resbn`

Scope: all three feature sets, LOSO and random holdout, overlapping and later non-overlapping windows.

### All-Model Seven-Experiment Audit

The all-model seven-experiment audit at `results/canonical_protocol12_seven_experiments` has:

- `summary_rows=66`
- `fold_rows=330`
- `prediction_rows=1195305`
- Exp3 real robustness table present
- Exp6 real few-shot calibration table present

Models in Exp1:

- Baselines: `adaboost_tree`, `bagged_tree_entropy`, `decision_tree_entropy`, `dummy_most_frequent`, `gaussian_nb`, `knn_k5`, `linear_svm`, `random_forest`, `rbf_svm`
- Deep v3: `improved_gnn_lstm_res`, `improved_gnn_lstm_attn_adj_resbn`

Models in real Exp3/Exp6:

- Same as above, plus `xgboost_hist`

### Final Top4 DOCX Deliverable

The final top4 deliverable at `results/canonical_protocol12_seven_experiments_top4` has:

- Baselines: `random_forest`, `knn_k5`
- Deep: `improved_gnn_lstm_res`, `improved_gnn_lstm_attn_adj_resbn`
- `summary_rows=48`
- `fold_rows=240`
- `prediction_rows=651576`
- non-overlapping windows included
- Exp4 probability/calibration fixed
- report manifest: `missing=[]`

This is the clean folder to submit as deliverables, but the scientific interpretation below also includes the broader runs.

## Direct Answers Across All Conducted Experiments

### 1. Which model wins under LOSO?

Across the broader conducted runs, the strongest LOSO row is still `random_forest` on `acc16_gyro`, but the original v1 `improved_gnn_lstm_attn_adj` is almost tied.

| Rank | Track | Feature set | Window | Family | Model | Accuracy | Macro-F1 |
|---:|---|---|---|---|---|---:|---:|
| 1 | v1/core | `acc16_gyro` | overlapping | baseline | `random_forest` | 0.8823 | 0.8863 |
| 2 | v1/core | `acc16_gyro_hr` | overlapping | deep | `improved_gnn_lstm_attn_adj` | 0.8892 | 0.8855 |
| 3 | top4 | `acc16_gyro` | non_overlapping | baseline | `random_forest` | 0.8800 | 0.8835 |
| 4 | top4/v1 | `acc16_gyro_hr` | overlapping | baseline | `random_forest` | 0.8801 | 0.8815 |
| 5 | v3/top4 | `acc16_gyro` | overlapping | deep | `improved_gnn_lstm_attn_adj_resbn` | 0.8779 | 0.8810 |

Important distinction:

- If we evaluate the **full conducted work**, the original `improved_gnn_lstm_attn_adj` is the best deep LOSO model.
- If we evaluate the **final top4 DOCX package only**, `random_forest` wins LOSO and v3 deep models are competitive but lower.

Blunt conclusion: the GNN family came very close to the best baseline in v1, but the final v3 residual models did not surpass `random_forest` on strict LOSO.

### 2. Which model wins under random holdout?

Across all conducted holdout runs, deep models dominate random holdout.

Top random-holdout rows:

| Track | Feature set | Window | Family | Model | Accuracy | Macro-F1 |
|---|---|---|---|---|---:|---:|
| v3/top4 | `acc16_hr` | overlapping | deep | `improved_gnn_lstm_attn_adj_resbn` | 0.9909 | 0.9906 |
| v3/top4 | `acc16_hr` | non_overlapping | deep | `improved_gnn_lstm_attn_adj_resbn` | 0.9897 | 0.9897 |
| v3/top4 | `acc16_hr` | overlapping | deep | `improved_gnn_lstm_res` | 0.9897 | 0.9896 |
| v3/top4 | `acc16_gyro_hr` | overlapping | deep | `improved_gnn_lstm_res` | 0.9892 | 0.9890 |
| v1/core | `acc16_gyro` | overlapping | deep | `gnn_flatten_lstm` | 0.9876 | 0.9872 |

Interpretation: random holdout rewards deep temporal models strongly, but it is leakage-sensitive because train and test can share subjects. It should not be the headline generalization claim.

### 3. Does non-overlapping windowing change the conclusion?

No. Non-overlapping windows do not materially change the conclusion.

The final top4 non-overlapping runs show:

| Protocol | Window | Best feature | Family | Model | Macro-F1 |
|---|---|---|---|---|---:|
| LOSO | overlapping | `acc16_gyro` | baseline | `random_forest` | 0.8863 |
| LOSO | non_overlapping | `acc16_gyro` | baseline | `random_forest` | 0.8835 |
| random_holdout | overlapping | `acc16_hr` | deep | `improved_gnn_lstm_attn_adj_resbn` | 0.9906 |
| random_holdout | non_overlapping | `acc16_hr` | deep | `improved_gnn_lstm_attn_adj_resbn` | 0.9897 |

The mean LOSO non-overlapping minus overlapping macro-F1 difference across matched top4 rows is approximately `-0.0063`. That is small and does not alter the ranking.

### 4. Are deep models better than baselines?

The answer depends on the protocol and model version.

Under strict LOSO:

- The best overall row is `random_forest` with macro-F1 `0.8863`.
- The original v1 `improved_gnn_lstm_attn_adj` is almost tied at macro-F1 `0.8855`.
- The best v3 deep row is `improved_gnn_lstm_attn_adj_resbn` at macro-F1 `0.8810`.

Under random holdout:

- Deep models clearly beat baselines.
- v3 `improved_gnn_lstm_attn_adj_resbn` reaches macro-F1 `0.9906`.

Under health-group analysis:

- Deep models are strongest.
- Best group sensitivity rows are v3 deep models, especially on `acc16_gyro_hr`.

Under few-shot calibration:

- Deep models benefit strongly from full-model fine-tuning.
- Some baseline models also improve, but the deployment case for deep models is stronger when subject calibration is allowed.

Conclusion: deep models are not a clean zero-shot LOSO winner, but they are scientifically useful because they perform strongly in random holdout, health-group recognition, and subject-calibrated settings.

### 5. Which feature set is best?

For original v1 LOSO across all core models:

| Feature set | Mean Macro-F1 | Best Macro-F1 | Rows |
|---|---:|---:|---:|
| `acc16_gyro` | 0.8001 | 0.8863 | 13 |
| `acc16_gyro_hr` | 0.7989 | 0.8855 | 13 |
| `acc16_hr` | 0.7711 | 0.8528 | 13 |

For the final top4 set:

- Best individual LOSO row: `acc16_gyro` with `random_forest`, macro-F1 `0.8863`.
- Best average LOSO feature set across top4 rows: `acc16_gyro_hr`, mean macro-F1 `0.8592`.
- `acc16_hr` alone is consistently weaker under LOSO.

Practical conclusion: use `acc16_gyro` or `acc16_gyro_hr` for serious LOSO claims. Do not rely on heart-rate-only performance.

### 6. Which subjects and classes are failing?

Subject `109` is the dominant failure case across both the all-model audit and final top4 results.

All-model audit mean subject performance:

| Subject | Mean Macro-F1 | Mean Accuracy |
|---:|---:|---:|
| 109 | 0.1234 | 0.7306 |
| 103 | 0.5291 | 0.7881 |
| 104 | 0.6068 | 0.7314 |
| 108 | 0.6260 | 0.6705 |
| 102 | 0.6799 | 0.7045 |

Final top4 mean subject performance:

| Subject | Mean Macro-F1 | Mean Accuracy |
|---:|---:|---:|
| 109 | 0.3679 | 0.7155 |
| 103 | 0.6826 | 0.8819 |
| 108 | 0.7137 | 0.7512 |
| 104 | 0.7517 | 0.8563 |
| 102 | 0.8029 | 0.8158 |

The repeated class-level failures are:

- `rope_jumping -> running`
- `standing -> sitting`
- `sitting -> standing`
- `vacuum_cleaning -> ironing`
- `nordic_walking -> walking`
- stair labels confused with locomotion

Interpretation: the main problem is subject-specific movement style and biomechanically similar classes, not just model capacity.

## Version-by-Version Findings

### V1 Original Canonical Core

Best v1 LOSO rows:

| Rank | Feature set | Family | Model | Macro-F1 |
|---:|---|---|---|---:|
| 1 | `acc16_gyro` | baseline | `random_forest` | 0.8863 |
| 2 | `acc16_gyro_hr` | deep | `improved_gnn_lstm_attn_adj` | 0.8855 |
| 3 | `acc16_gyro_hr` | baseline | `random_forest` | 0.8815 |
| 4 | `acc16_gyro_hr` | deep | `improved_gnn_lstm` | 0.8690 |
| 5 | `acc16_gyro` | deep | `improved_gnn_lstm` | 0.8553 |

Best v1 holdout rows:

| Rank | Feature set | Family | Model | Macro-F1 |
|---:|---|---|---|---:|
| 1 | `acc16_gyro` | deep | `gnn_flatten_lstm` | 0.9872 |
| 2 | `acc16_gyro` | deep | `improved_gnn_lstm_attn_adj` | 0.9855 |
| 3 | `acc16_gyro` | deep | `improved_gnn_lstm` | 0.9830 |
| 4 | `acc16_gyro_hr` | deep | `improved_gnn_lstm_attn_adj` | 0.9760 |

V1 conclusion: original improved GNN-LSTM attention adjacency was the closest deep model to the best LOSO baseline. It is important and should not be erased from the narrative.

### V2 Improved-Model Ablation

V2 tested normalization, class balancing, label smoothing, and AdamW on `acc16_gyro_hr`.

| Model | Variant | Accuracy | Macro-F1 |
|---|---|---:|---:|
| `improved_gnn_lstm` | `v2_norm_balanced_ls005` | 0.8845 | 0.8707 |
| `improved_gnn_lstm_attn_adj` | `v2_norm_balanced_ls005` | 0.8503 | 0.8458 |

V2 conclusion:

- `improved_gnn_lstm` improved slightly versus its v1 `acc16_gyro_hr` LOSO macro-F1 of `0.8690`.
- `improved_gnn_lstm_attn_adj` degraded versus its v1 macro-F1 of `0.8855`.
- The combined v2 recipe should not be treated as a universal improvement.

### V3 Residual GNN-LSTM Models

Best v3 LOSO rows:

| Rank | Feature set | Window | Model | Macro-F1 |
|---:|---|---|---|---:|
| 1 | `acc16_gyro` | overlapping | `improved_gnn_lstm_attn_adj_resbn` | 0.8810 |
| 2 | `acc16_gyro_hr` | overlapping | `improved_gnn_lstm_res` | 0.8757 |
| 3 | `acc16_gyro_hr` | non_overlapping | `improved_gnn_lstm_attn_adj_resbn` | 0.8685 |
| 4 | `acc16_gyro_hr` | overlapping | `improved_gnn_lstm_attn_adj_resbn` | 0.8647 |
| 5 | `acc16_gyro` | non_overlapping | `improved_gnn_lstm_attn_adj_resbn` | 0.8638 |

V3 conclusion:

- v3 produces very strong random-holdout results.
- v3 is competitive under LOSO.
- v3 does not beat the original v1 `improved_gnn_lstm_attn_adj` LOSO result or the best `random_forest` LOSO result.
- v3 helps more clearly in health-group and calibration-related analysis than in zero-shot fine-grained LOSO.

## Seven Experiments Across Conducted Work

### Experiment 1: Leakage-Control Evaluation

The all-model audit confirms strong leakage inflation:

- Best LOSO: `random_forest`, `acc16_gyro`, macro-F1 `0.8863`.
- Best random holdout: v3 deep model, macro-F1 `0.9906`.
- Deep models show the largest holdout-minus-LOSO gaps.

Largest final top4 leakage gaps:

| Feature set | Window | Model | Accuracy gap |
|---|---|---|---:|
| `acc16_hr` | overlapping | `improved_gnn_lstm_res` | 0.1753 |
| `acc16_hr` | overlapping | `improved_gnn_lstm_attn_adj_resbn` | 0.1562 |
| `acc16_hr` | non_overlapping | `improved_gnn_lstm_res` | 0.1531 |
| `acc16_hr` | non_overlapping | `improved_gnn_lstm_attn_adj_resbn` | 0.1496 |

Claim: random holdout is inflated and must be secondary to LOSO.

### Experiment 2: Statistical Reliability

All-model fold-level ranking is led by `random_forest`.

Top all-model fold-mean LOSO macro-F1 rows:

| Feature set | Family | Model | Mean Macro-F1 | Std |
|---|---|---|---:|---:|
| `acc16_gyro` | baseline | `random_forest` | 0.7682 | 0.2337 |
| `acc16_gyro_hr` | baseline | `random_forest` | 0.7445 | 0.2131 |
| `acc16_hr` | baseline | `random_forest` | 0.7194 | 0.2186 |
| `acc16_gyro` | deep | `improved_gnn_lstm_attn_adj_resbn` | 0.7185 | 0.2535 |
| `acc16_gyro_hr` | deep | `improved_gnn_lstm_res` | 0.7158 | 0.2520 |

The standard deviations are large. Subject variability is a central finding.

### Experiment 3: Robustness

All-model robustness ranking by average macro-F1 drop:

| Family | Model | Mean Macro-F1 drop |
|---|---|---:|
| baseline | `dummy_most_frequent` | 0.0000 |
| baseline | `random_forest` | 0.0261 |
| baseline | `bagged_tree_entropy` | 0.0689 |
| baseline | `xgboost_hist` | 0.0827 |
| baseline | `decision_tree_entropy` | 0.0843 |
| deep | `improved_gnn_lstm_res` | 0.1110 |
| baseline | `knn_k5` | 0.2405 |
| deep | `improved_gnn_lstm_attn_adj_resbn` | 0.2456 |

Note: `dummy_most_frequent` is trivially robust because it barely reacts to input. It is not a useful model. Among meaningful models, `random_forest` is the most robust in the all-model audit.

The final top4 extended robustness run adds `sensor_node_zero` and `random_window_dropout`. In that stricter top4 run:

- `sensor_node_zero` is the most damaging perturbation.
- `random_forest` remains the most robust useful model.
- `improved_gnn_lstm_res` is more robust than `improved_gnn_lstm_attn_adj_resbn`.

### Experiment 4: Calibration, Uncertainty, and Selective Prediction

There are two calibration facts:

1. In the all-model audit, many older baseline rows did not have saved probability columns, so only `12/66` rows had full calibration status.
2. In the final top4 deliverable, we fixed this. All `48/48` top4 rows have probability columns and calibration status `ok`.

Best final top4 LOSO ECE rows:

| Feature set | Window | Family | Model | ECE | Macro-F1 |
|---|---|---|---|---:|---:|
| `acc16_gyro_hr` | non_overlapping | deep | `improved_gnn_lstm_attn_adj_resbn` | 0.0423 | 0.8685 |
| `acc16_gyro` | non_overlapping | deep | `improved_gnn_lstm_attn_adj_resbn` | 0.0445 | 0.8638 |
| `acc16_hr` | non_overlapping | baseline | `knn_k5` | 0.0570 | 0.8192 |
| `acc16_gyro_hr` | non_overlapping | baseline | `knn_k5` | 0.0580 | 0.8271 |

Conclusion: the final top4 calibration deliverable is complete. The all-model calibration audit should be interpreted with the probability-column caveat.

### Experiment 5: Subject-Level Failure Analysis

The all-model and top4 analyses agree that subject `109` is the worst subject.

Dominant failure families:

- `rope_jumping -> running`
- `standing -> sitting`
- `sitting -> standing`
- `vacuum_cleaning -> ironing`
- stair/locomotion confusion

This supports a subject-adaptation direction rather than only larger models.

### Experiment 6: Few-Shot Subject Calibration

All-model Exp6 has `1458` rows and all rows are `ok`.

Best all-model 10% calibration gains:

| Family | Model | Mean Macro-F1 gain | Calibrated Macro-F1 |
|---|---|---:|---:|
| baseline | `decision_tree_entropy` | 0.1082 | 0.6977 |
| baseline | `bagged_tree_entropy` | 0.0932 | 0.7538 |
| baseline | `xgboost_hist` | 0.0744 | 0.7916 |
| deep | `improved_gnn_lstm_attn_adj_resbn` | 0.0730 | 0.7766 |
| deep | `improved_gnn_lstm_res` | 0.0597 | 0.7536 |

Final top4 10% calibration gains:

| Family | Model | Mean 10% Macro-F1 gain |
|---|---|---:|
| deep | `improved_gnn_lstm_res` | 0.0879 |
| deep | `improved_gnn_lstm_attn_adj_resbn` | 0.0792 |
| baseline | `knn_k5` | 0.0482 |
| baseline | `random_forest` | 0.0448 |

Conclusion: few-shot calibration is one of the most useful directions. In the final top4 scope, deep models benefit more than the two chosen baselines.

### Experiment 7: Health-Relevant Activity Groups

Deep v3 models are strongest for grouped health-relevant activity recognition.

Top all-model group-sensitivity rows:

| Feature set | Family | Model | Mean group sensitivity |
|---|---|---|---:|
| `acc16_gyro_hr` | deep | `improved_gnn_lstm_res` | 0.9563 |
| `acc16_gyro_hr` | deep | `improved_gnn_lstm_attn_adj_resbn` | 0.9535 |
| `acc16_gyro` | deep | `improved_gnn_lstm_attn_adj_resbn` | 0.9518 |
| `acc16_hr` | deep | `improved_gnn_lstm_attn_adj_resbn` | 0.9461 |
| `acc16_gyro` | deep | `improved_gnn_lstm_res` | 0.9441 |

Hardest health groups across the all-model audit:

| Group | Average sensitivity |
|---|---:|
| `jump` | 0.7472 |
| `stairs` | 0.7827 |
| `posture` | 0.8387 |
| `locomotion` | 0.8650 |
| `household` | 0.8955 |

Conclusion: even though `random_forest` is the best fine-grained zero-shot LOSO row, the deep models have the strongest case for health-group analysis.

## Final Defensible Claims

1. The best strict LOSO row across conducted experiments is `random_forest` on `acc16_gyro`, macro-F1 `0.8863`.
2. The original v1 `improved_gnn_lstm_attn_adj` is the best deep LOSO model, macro-F1 `0.8855`, nearly tied with random forest.
3. V2 did not uniformly improve the original improved models; it slightly helped `improved_gnn_lstm` but hurt `improved_gnn_lstm_attn_adj`.
4. V3 residual models are strong but do not beat the best v1 LOSO deep result or the best `random_forest` LOSO result.
5. Deep models dominate random holdout, but random holdout is leakage-sensitive and should not be the main generalization claim.
6. Non-overlapping windows do not materially change the final top4 conclusion.
7. Subject `109` and the `rope_jumping -> running` failure are central weaknesses.
8. Robustness analysis favors `random_forest` among meaningful models; v3 residual GNN-LSTM is more robust than the v3 attention-adjacency residual model.
9. Few-shot subject calibration is valuable and is one of the best next directions.
10. Health-group analysis is the strongest practical use case for the v3 deep models.

## What Not To Claim

- Do not claim v3 is the overall best LOSO model. It is not.
- Do not claim deep models clearly beat baselines under zero-shot LOSO. The best deep model is nearly tied in v1, but not clearly ahead.
- Do not use random-holdout performance as the headline result.
- Do not ignore the original v1 `improved_gnn_lstm_attn_adj`; it is the best deep LOSO result we observed.
- Do not present the all-model Exp4 calibration table as fully complete for baselines; the final top4 calibration table is complete, but the all-model calibration audit has probability-column gaps.

## Recommended Next Work

The next work should not be another blind size increase.

Recommended direction:

1. Keep `random_forest` as the strongest zero-shot LOSO baseline.
2. Keep original `improved_gnn_lstm_attn_adj` as the strongest deep LOSO result.
3. Use v3 residual models for health-group and few-shot narratives.
4. Build a v4 model focused on subject adaptation and hard confusions:
   - rare-class/failure-class sampling
   - explicit coarse health-group auxiliary head
   - subject calibration workflow
   - temporal representation before graph aggregation
5. Treat subject `109`, `rope_jumping`, stair labels, and posture/household confusion as required diagnostic targets.

## Deliverable Folder

The clean deliverable folder is:

`deliverables/`

It contains:

- `tables/`: main CSV tables for Experiments 1-7 from the final top4 DOCX-standard package
- `enhanced_tables/`: enhanced CSV tables
- `detail_tables/`: detailed real Exp3 and Exp6 CSVs
- `figures/`: PNG and PDF figures
- `manifests/`: completion manifests
- `report.md`: this complete interpretation report
- `PAMAP2_SEVEN_EXPERIMENTS_REPORT.md`: generated artifact report

