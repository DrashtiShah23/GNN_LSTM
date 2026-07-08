# PAMAP2 Protocol12 Seven-Experiment Report

This report is generated from the canonical PAMAP2 protocol-only/protocol12 artifacts.

## Current Scope

- Dataset: `pamap2`
- Task: `protocol12`
- Feature sets: `acc16_hr`, `acc16_gyro`, `acc16_gyro_hr`
- Core protocols currently available: `overlapping/loso` and `overlapping/random_holdout`
- Model families: canonical baselines plus v3 improved GNN-LSTM variants

## Missing Before Full DOCX Completion

- No run-required gaps detected.

## Experiment 1: Leakage-Control Evaluation

Current evidence compares random holdout against LOSO under overlapping windows. This is useful for subject-leakage inflation, but the DOCX-standard overlapping-vs-non-overlapping factorial design is not complete until stride-128 runs exist.

Largest holdout-minus-LOSO accuracy gaps:
- `acc16_hr` / `improved_gnn_lstm_res`: `holdout_minus_loso_accuracy=0.1753`
- `acc16_hr` / `improved_gnn_lstm_attn_adj_resbn`: `holdout_minus_loso_accuracy=0.1562`
- `acc16_hr` / `improved_gnn_lstm_res`: `holdout_minus_loso_accuracy=0.1531`
- `acc16_hr` / `improved_gnn_lstm_attn_adj_resbn`: `holdout_minus_loso_accuracy=0.1496`
- `acc16_gyro` / `improved_gnn_lstm_res`: `holdout_minus_loso_accuracy=0.1308`
- `acc16_gyro_hr` / `improved_gnn_lstm_res`: `holdout_minus_loso_accuracy=0.1298`

## Experiment 2: Statistical Reliability

Enhanced table adds 95% confidence intervals, Wilcoxon p-values versus the best model in each result-set/feature-set group, rank-biserial effect sizes, and ranking stability.

Top LOSO mean macro-F1 rows:
- `acc16_gyro` / `random_forest`: `macro_f1_mean=0.7795`
- `acc16_gyro` / `random_forest`: `macro_f1_mean=0.7682`
- `acc16_gyro_hr` / `random_forest`: `macro_f1_mean=0.7461`
- `acc16_gyro_hr` / `random_forest`: `macro_f1_mean=0.7445`
- `acc16_hr` / `random_forest`: `macro_f1_mean=0.7194`
- `acc16_gyro` / `improved_gnn_lstm_attn_adj_resbn`: `macro_f1_mean=0.7185`

## Experiment 3: Robustness

Real robustness tables are present for Gaussian noise, random channel dropout, and heart-rate zeroing. The runner now also supports `sensor_node_zero` and `random_window_dropout`, but those new perturbations still need to be executed if they are required in the final manuscript.

Most robust rows by lowest average macro-F1 drop:
- `acc16_gyro_hr` / `random_forest`: `macro_f1_drop=0.1232`
- `acc16_hr` / `random_forest`: `macro_f1_drop=0.1319`
- `acc16_hr` / `improved_gnn_lstm_res`: `macro_f1_drop=0.1719`
- `acc16_gyro_hr` / `improved_gnn_lstm_res`: `macro_f1_drop=0.1763`
- `acc16_gyro` / `random_forest`: `macro_f1_drop=0.1799`
- `acc16_gyro` / `improved_gnn_lstm_res`: `macro_f1_drop=0.1860`

## Experiment 4: Calibration, Uncertainty, and Selective Prediction

Enhanced table adds Brier score, negative log-likelihood, and selective prediction metrics at 90%, 80%, and 70% coverage for rows with probability columns. Most classical baseline rows still lack saved probabilities.

Lowest ECE among probability-available rows:
- `acc16_hr` / `improved_gnn_lstm_attn_adj_resbn`: `ece=0.0054`
- `acc16_gyro_hr` / `improved_gnn_lstm_attn_adj_resbn`: `ece=0.0072`
- `acc16_gyro_hr` / `improved_gnn_lstm_res`: `ece=0.0073`
- `acc16_hr` / `improved_gnn_lstm_res`: `ece=0.0074`
- `acc16_gyro` / `improved_gnn_lstm_attn_adj_resbn`: `ece=0.0081`
- `acc16_gyro` / `improved_gnn_lstm_res`: `ece=0.0096`

## Experiment 5: Subject-Level Failure Analysis

Enhanced table adds worst activity and dominant confusion per held-out subject/model/feature-set.

Hardest subject/model rows by macro-F1:
- subject `109` / `acc16_gyro` / `improved_gnn_lstm_res`: macro_f1=0.0000, worst_activity=`rope_jumping`, dominant_confusion=`rope_jumping -> running`
- subject `109` / `acc16_gyro_hr` / `improved_gnn_lstm_res`: macro_f1=0.0000, worst_activity=`rope_jumping`, dominant_confusion=`rope_jumping -> running`
- subject `109` / `acc16_gyro_hr` / `knn_k5`: macro_f1=0.0710, worst_activity=`rope_jumping`, dominant_confusion=`rope_jumping -> running`
- subject `109` / `acc16_gyro` / `knn_k5`: macro_f1=0.0711, worst_activity=`rope_jumping`, dominant_confusion=`rope_jumping -> running`
- subject `109` / `acc16_gyro` / `knn_k5`: macro_f1=0.0770, worst_activity=`rope_jumping`, dominant_confusion=`rope_jumping -> running`
- subject `109` / `acc16_gyro_hr` / `knn_k5`: macro_f1=0.0867, worst_activity=`rope_jumping`, dominant_confusion=`rope_jumping -> running`
- subject `109` / `acc16_hr` / `knn_k5`: macro_f1=0.1139, worst_activity=`rope_jumping`, dominant_confusion=`rope_jumping -> running`
- subject `109` / `acc16_hr` / `knn_k5`: macro_f1=0.1365, worst_activity=`rope_jumping`, dominant_confusion=`rope_jumping -> running`

## Experiment 6: Few-Shot Subject Calibration

Real few-shot calibration rows are present for 0%, 1%, 5%, and 10% calibration with classifier-head and full-model strategies where applicable.

Best 10% calibration gains:
- `acc16_hr` / `improved_gnn_lstm_res`: `macro_f1_improvement=0.1353`
- `acc16_hr` / `improved_gnn_lstm_attn_adj_resbn`: `macro_f1_improvement=0.1168`
- `acc16_hr` / `improved_gnn_lstm_res`: `macro_f1_improvement=0.1059`
- `acc16_gyro_hr` / `improved_gnn_lstm_attn_adj_resbn`: `macro_f1_improvement=0.0927`
- `acc16_gyro` / `improved_gnn_lstm_res`: `macro_f1_improvement=0.0894`
- `acc16_hr` / `improved_gnn_lstm_attn_adj_resbn`: `macro_f1_improvement=0.0850`
- `acc16_gyro_hr` / `improved_gnn_lstm_res`: `macro_f1_improvement=0.0799`
- `acc16_gyro` / `improved_gnn_lstm_attn_adj_resbn`: `macro_f1_improvement=0.0747`

## Experiment 7: Health-Relevant Activity Groups

Enhanced table adds sensitivity, specificity, and main group confusion. Activity-group mapping is exported separately.

Lowest group sensitivity rows:
- `acc16_hr` / `improved_gnn_lstm_res` / `jump`: sensitivity=0.6507, specificity=0.9973, main_confusion=`jump -> locomotion`
- `acc16_hr` / `knn_k5` / `stairs`: sensitivity=0.7295, specificity=0.9858, main_confusion=`stairs -> locomotion`
- `acc16_gyro` / `knn_k5` / `jump`: sensitivity=0.7435, specificity=0.9990, main_confusion=`jump -> locomotion`
- `acc16_gyro` / `knn_k5` / `jump`: sensitivity=0.7454, specificity=0.9989, main_confusion=`jump -> locomotion`
- `acc16_gyro_hr` / `knn_k5` / `jump`: sensitivity=0.7487, specificity=0.9995, main_confusion=`jump -> locomotion`
- `acc16_hr` / `knn_k5` / `jump`: sensitivity=0.7520, specificity=0.9988, main_confusion=`jump -> stairs`
- `acc16_hr` / `knn_k5` / `jump`: sensitivity=0.7592, specificity=0.9986, main_confusion=`jump -> stairs`
- `acc16_gyro_hr` / `knn_k5` / `jump`: sensitivity=0.7625, specificity=0.9993, main_confusion=`jump -> locomotion`

## Generated Figure Files

- `fig_exp1_overlapping_protocol_comparison.png`
- `fig_exp2_model_ranking_stability.png`
- `fig_exp3_robustness_degradation_curves.png`
- `fig_exp4_coverage_acc16_gyro_hr_non_overlapping_random_forest.png`
- `fig_exp4_coverage_acc16_gyro_hr_overlapping_improved_gnn_lstm_res.png`
- `fig_exp4_coverage_acc16_gyro_hr_overlapping_random_forest.png`
- `fig_exp4_coverage_acc16_gyro_non_overlapping_random_forest.png`
- `fig_exp4_coverage_acc16_gyro_overlapping_improved_gnn_lstm_attn_adj_resbn.png`
- `fig_exp4_coverage_acc16_gyro_overlapping_random_forest.png`
- `fig_exp4_reliability_acc16_gyro_hr_non_overlapping_random_forest.png`
- `fig_exp4_reliability_acc16_gyro_hr_overlapping_improved_gnn_lstm_res.png`
- `fig_exp4_reliability_acc16_gyro_hr_overlapping_random_forest.png`
- `fig_exp4_reliability_acc16_gyro_non_overlapping_random_forest.png`
- `fig_exp4_reliability_acc16_gyro_overlapping_improved_gnn_lstm_attn_adj_resbn.png`
- `fig_exp4_reliability_acc16_gyro_overlapping_random_forest.png`
- `fig_exp5_subject_activity_heatmap_acc16_gyro_hr_non_overlapping_random_forest.png`
- `fig_exp5_subject_activity_heatmap_acc16_gyro_hr_overlapping_improved_gnn_lstm_res.png`
- `fig_exp5_subject_activity_heatmap_acc16_gyro_hr_overlapping_random_forest.png`
- `fig_exp5_subject_activity_heatmap_acc16_gyro_non_overlapping_random_forest.png`
- `fig_exp5_subject_activity_heatmap_acc16_gyro_overlapping_improved_gnn_lstm_attn_adj_resbn.png`
- `fig_exp5_subject_activity_heatmap_acc16_gyro_overlapping_random_forest.png`
- `fig_exp6_calibration_efficiency.png`
- `fig_exp7_health_group_confusion_acc16_gyro_hr_non_overlapping_random_forest.png`
- `fig_exp7_health_group_confusion_acc16_gyro_hr_overlapping_improved_gnn_lstm_res.png`
- `fig_exp7_health_group_confusion_acc16_gyro_hr_overlapping_random_forest.png`
- `fig_exp7_health_group_confusion_acc16_gyro_non_overlapping_random_forest.png`
- `fig_exp7_health_group_confusion_acc16_gyro_overlapping_improved_gnn_lstm_attn_adj_resbn.png`
- `fig_exp7_health_group_confusion_acc16_gyro_overlapping_random_forest.png`

## Generated Enhanced Tables

- `table_exp2_pairwise_comparisons_enhanced.csv`
- `table_exp2_ranking_stability.csv`
- `table_exp2_statistical_reliability_enhanced.csv`
- `table_exp4_calibration_selective_prediction.csv`
- `table_exp5_subject_failure_enhanced.csv`
- `table_exp7_activity_group_mapping.csv`
- `table_exp7_health_groups_enhanced.csv`
