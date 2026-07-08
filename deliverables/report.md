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

Best LOSO row per feature set and family, across conducted core/v3 runs:

| Feature set | Family | Track | Window | Model | Accuracy | Macro-F1 |
|---|---|---|---|---|---:|---:|
| `acc16_gyro` | baseline | v1/top4 | overlapping | `random_forest` | 0.8823 | 0.8863 |
| `acc16_gyro` | deep | v3/top4 | overlapping | `improved_gnn_lstm_attn_adj_resbn` | 0.8779 | 0.8810 |
| `acc16_gyro_hr` | baseline | v1/top4 | overlapping | `random_forest` | 0.8801 | 0.8815 |
| `acc16_gyro_hr` | deep | v1/core | overlapping | `improved_gnn_lstm_attn_adj` | 0.8892 | 0.8855 |
| `acc16_hr` | baseline | top4 | non_overlapping | `random_forest` | 0.8558 | 0.8528 |
| `acc16_hr` | deep | v1/core | overlapping | `gnn_flatten_lstm` | 0.8549 | 0.8467 |

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

For original v1 overlapping LOSO across all core models, each feature set has `11` model evaluations:

- `2` baselines: `random_forest`, `knn_k5`
- `9` deep models: `cnn`, `lstm`, `gnn`, `gnn_learnable_adj`, `gnn_attention_adj`, `gnn_lstm`, `gnn_flatten_lstm`, `improved_gnn_lstm`, `improved_gnn_lstm_attn_adj`

| Feature set | Model evaluations | Mean Macro-F1 | Best model | Best Macro-F1 |
|---|---:|---:|---|---:|
| `acc16_gyro` | 11 | 0.7909 | `random_forest` | 0.8863 |
| `acc16_gyro_hr` | 11 | 0.7890 | `improved_gnn_lstm_attn_adj` | 0.8855 |
| `acc16_hr` | 11 | 0.7593 | `random_forest` | 0.8506 |

For the final top4 set:

- Best individual LOSO row: `acc16_gyro` with `random_forest`, macro-F1 `0.8863`.
- Best average LOSO feature set across top4 rows: `acc16_gyro_hr`, mean macro-F1 `0.8592`.
- `acc16_hr` alone is consistently weaker under LOSO.

Practical conclusion: use `acc16_gyro` or `acc16_gyro_hr` for serious LOSO claims. Do not rely on heart-rate-only performance.

### 6. Which subjects and classes are failing?

Subject `109` is the dominant failure case across both the all-model audit and final top4 results.

Important caveat: the all-model audit includes weak diagnostic baselines such as `dummy_most_frequent`, so its subject averages are useful for broad failure detection but are harsher than the final top4 model set. The final top4 table is the cleaner subject-failure evidence.

All-model audit mean subject performance, all 9 subjects:

| Subject | Mean Macro-F1 | Mean Accuracy |
|---:|---:|---:|
| 109 | 0.2412 | 0.7306 |
| 103 | 0.5457 | 0.7881 |
| 104 | 0.6137 | 0.7314 |
| 108 | 0.6260 | 0.6705 |
| 102 | 0.6799 | 0.7045 |
| 106 | 0.6955 | 0.7669 |
| 101 | 0.7004 | 0.7149 |
| 107 | 0.7253 | 0.8030 |
| 105 | 0.7674 | 0.7818 |

Final top4 mean subject performance, all 9 subjects:

| Subject | Mean Macro-F1 | Mean Accuracy |
|---:|---:|---:|
| 109 | 0.3679 | 0.7155 |
| 103 | 0.6826 | 0.8819 |
| 108 | 0.7137 | 0.7512 |
| 104 | 0.7517 | 0.8563 |
| 102 | 0.8029 | 0.8158 |
| 106 | 0.8111 | 0.8818 |
| 101 | 0.8373 | 0.8406 |
| 107 | 0.8743 | 0.9232 |
| 105 | 0.8824 | 0.8853 |

Worst final top4 row per subject:

| Subject | Feature set | Window | Family | Model | Accuracy | Macro-F1 | Worst activity | Worst activity recall | Dominant confusion |
|---:|---|---|---|---|---:|---:|---|---:|---|
| 101 | `acc16_hr` | overlapping | baseline | `random_forest` | 0.7747 | 0.7568 | `descending_stairs` | 0.4105 | `vacuum_cleaning -> cycling` |
| 102 | `acc16_hr` | non_overlapping | deep | `improved_gnn_lstm_res` | 0.7047 | 0.7012 | `nordic_walking` | 0.0043 | `walking -> ascending_stairs` |
| 103 | `acc16_gyro` | overlapping | baseline | `knn_k5` | 0.7999 | 0.5360 | `sitting` | 0.6696 | `sitting -> standing` |
| 104 | `acc16_hr` | overlapping | baseline | `random_forest` | 0.7597 | 0.6275 | `ironing` | 0.2596 | `ironing -> vacuum_cleaning` |
| 105 | `acc16_hr` | non_overlapping | baseline | `knn_k5` | 0.8241 | 0.8130 | `descending_stairs` | 0.4490 | `vacuum_cleaning -> ironing` |
| 106 | `acc16_gyro` | non_overlapping | baseline | `knn_k5` | 0.8332 | 0.7642 | `rope_jumping` | 0.0000 | `standing -> sitting` |
| 107 | `acc16_hr` | non_overlapping | baseline | `knn_k5` | 0.8549 | 0.7670 | `ascending_stairs` | 0.5620 | `ascending_stairs -> walking` |
| 108 | `acc16_hr` | overlapping | deep | `improved_gnn_lstm_res` | 0.4431 | 0.3794 | `nordic_walking` | 0.0000 | `nordic_walking -> descending_stairs` |
| 109 | `acc16_gyro` | non_overlapping | deep | `improved_gnn_lstm_res` | 0.0000 | 0.0000 | `rope_jumping` | 0.0000 | `rope_jumping -> running` |

Most frequent final top4 dominant confusions:

| Dominant confusion | Count |
|---|---:|
| `standing -> sitting` | 40 |
| `sitting -> standing` | 34 |
| `vacuum_cleaning -> ironing` | 30 |
| `nordic_walking -> walking` | 14 |
| `rope_jumping -> running` | 14 |
| `sitting -> ironing` | 10 |
| `vacuum_cleaning -> cycling` | 7 |
| `ironing -> standing` | 7 |
| `standing -> ironing` | 6 |
| `ironing -> vacuum_cleaning` | 6 |

Most frequent final top4 worst activities:

| Worst activity | Count |
|---|---:|
| `rope_jumping` | 51 |
| `descending_stairs` | 41 |
| `sitting` | 30 |
| `standing` | 21 |
| `vacuum_cleaning` | 18 |
| `nordic_walking` | 17 |
| `running` | 14 |
| `ironing` | 9 |
| `ascending_stairs` | 6 |
| `cycling` | 5 |
| `walking` | 4 |

Subject-level interpretation:

- `109` is the major outlier. Its dominant failure is `rope_jumping -> running`, and some final top4 rows collapse completely on this subject. This points to subject-specific high-intensity motion style rather than a global class-label issue.
- `103` has high mean accuracy but much lower macro-F1, which indicates class imbalance or missing/weak per-class recall. The dominant failure is posture confusion, especially `sitting -> standing`.
- `108` is difficult because `nordic_walking` is repeatedly confused with walking/stairs. The worst top4 row has `nordic_walking` recall of `0.0000`.
- `104` is driven by household-motion confusion, especially `ironing -> vacuum_cleaning`.
- `101`, `105`, and `107` are not globally poor, but their worst rows show stair and household confusions that should still be mentioned.

Interpretation: the main problem is subject-specific movement style and biomechanically similar classes, not just model capacity.

## Version-by-Version Findings

### V1 Original Canonical Core

Best v1 LOSO row per feature set and family:

| Feature set | Family | Model | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---|---:|---:|---:|
| `acc16_gyro` | baseline | `random_forest` | 0.8823 | 0.8742 | 0.8863 |
| `acc16_gyro` | deep | `improved_gnn_lstm` | 0.8570 | 0.8593 | 0.8553 |
| `acc16_gyro_hr` | baseline | `random_forest` | 0.8801 | 0.8785 | 0.8815 |
| `acc16_gyro_hr` | deep | `improved_gnn_lstm_attn_adj` | 0.8892 | 0.8842 | 0.8855 |
| `acc16_hr` | baseline | `random_forest` | 0.8529 | 0.8491 | 0.8506 |
| `acc16_hr` | deep | `gnn_flatten_lstm` | 0.8549 | 0.8477 | 0.8467 |

Best v1 random-holdout row per feature set and family:

| Feature set | Family | Model | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---|---:|---:|---:|
| `acc16_gyro` | baseline | `random_forest` | 0.9434 | 0.9339 | 0.9422 |
| `acc16_gyro` | deep | `gnn_flatten_lstm` | 0.9876 | 0.9874 | 0.9872 |
| `acc16_gyro_hr` | baseline | `random_forest` | 0.9616 | 0.9598 | 0.9616 |
| `acc16_gyro_hr` | deep | `improved_gnn_lstm_attn_adj` | 0.9776 | 0.9758 | 0.9760 |
| `acc16_hr` | baseline | `random_forest` | 0.9674 | 0.9652 | 0.9671 |
| `acc16_hr` | deep | `improved_gnn_lstm` | 0.9759 | 0.9733 | 0.9740 |

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

All v3 LOSO rows are shown below so each feature set is visible, including `acc16_hr`.

| Feature set | Window | Model | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---|---:|---:|---:|
| `acc16_gyro` | overlapping | `improved_gnn_lstm_attn_adj_resbn` | 0.8779 | 0.8766 | 0.8810 |
| `acc16_gyro` | overlapping | `improved_gnn_lstm_res` | 0.8689 | 0.8623 | 0.8570 |
| `acc16_gyro` | non_overlapping | `improved_gnn_lstm_attn_adj_resbn` | 0.8773 | 0.8745 | 0.8638 |
| `acc16_gyro` | non_overlapping | `improved_gnn_lstm_res` | 0.8529 | 0.8456 | 0.8341 |
| `acc16_gyro_hr` | overlapping | `improved_gnn_lstm_res` | 0.8758 | 0.8788 | 0.8757 |
| `acc16_gyro_hr` | overlapping | `improved_gnn_lstm_attn_adj_resbn` | 0.8755 | 0.8716 | 0.8647 |
| `acc16_gyro_hr` | non_overlapping | `improved_gnn_lstm_attn_adj_resbn` | 0.8819 | 0.8735 | 0.8685 |
| `acc16_gyro_hr` | non_overlapping | `improved_gnn_lstm_res` | 0.8585 | 0.8523 | 0.8428 |
| `acc16_hr` | overlapping | `improved_gnn_lstm_attn_adj_resbn` | 0.8347 | 0.8412 | 0.8371 |
| `acc16_hr` | overlapping | `improved_gnn_lstm_res` | 0.8144 | 0.8043 | 0.8044 |
| `acc16_hr` | non_overlapping | `improved_gnn_lstm_attn_adj_resbn` | 0.8401 | 0.8456 | 0.8294 |
| `acc16_hr` | non_overlapping | `improved_gnn_lstm_res` | 0.8333 | 0.8364 | 0.8236 |

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

Best all-model fold-mean LOSO row per feature set and family:

| Feature set | Family | Model | Mean Macro-F1 | Std |
|---|---|---|---:|---:|
| `acc16_gyro` | baseline | `random_forest` | 0.7682 | 0.2337 |
| `acc16_gyro` | deep | `improved_gnn_lstm_attn_adj_resbn` | 0.7185 | 0.2535 |
| `acc16_gyro_hr` | baseline | `random_forest` | 0.7445 | 0.2131 |
| `acc16_gyro_hr` | deep | `improved_gnn_lstm_res` | 0.7158 | 0.2520 |
| `acc16_hr` | baseline | `random_forest` | 0.7194 | 0.2186 |
| `acc16_hr` | deep | `improved_gnn_lstm_attn_adj_resbn` | 0.6825 | 0.2444 |

This is the corrected reliability table. The earlier ranked view hid `acc16_hr` deep because it was below the top five rows, not because the experiment was missing.

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
