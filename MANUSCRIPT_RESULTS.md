# Manuscript Results

## Results

### Experiment 1: Leakage Control

The investigators evaluated whether overlapping sliding windows inflate performance estimates relative to subject-independent LOSO testing. Table 1 summarizes accuracy, macro F1, and balanced accuracy under overlapping and non-overlapping windowing for random holdout and LOSO protocols on PAMAP2 and HHAR.

**Table 1.** Leakage control results across window types and evaluation protocols. Leakage gap is defined as random-holdout accuracy minus LOSO accuracy.

| Dataset | Window_Type | Evaluation_Protocol | Model | Accuracy | Macro_F1 | Balanced_Accuracy | Leakage_Gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pamap2 | non_overlapping | loso | CNN1D | 0.5520 | 0.5093 | 0.5300 | nan |
| pamap2 | non_overlapping | loso | Flatten_LSTM | 0.8102 | 0.7457 | 0.7485 | nan |
| pamap2 | non_overlapping | loso | Improved_GNN_LSTM | 0.8222 | 0.7382 | 0.7483 | nan |
| pamap2 | non_overlapping | random_holdout | CNN1D | 0.9694 | 0.9725 | 0.9707 | 0.4174 |
| pamap2 | non_overlapping | random_holdout | Flatten_LSTM | 0.8859 | 0.7900 | 0.8123 | 0.0757 |
| pamap2 | non_overlapping | random_holdout | Improved_GNN_LSTM | 0.9396 | 0.9200 | 0.9176 | 0.1174 |
| pamap2 | overlapping | loso | CNN1D | 0.7396 | 0.7469 | 0.7395 | nan |
| pamap2 | overlapping | loso | Flatten_LSTM | 0.7916 | 0.7590 | 0.7526 | nan |
| pamap2 | overlapping | loso | Improved_GNN_LSTM | 0.8162 | 0.7420 | 0.7562 | nan |
| pamap2 | overlapping | random_holdout | CNN1D | 0.9937 | 0.9942 | 0.9935 | 0.2541 |
| pamap2 | overlapping | random_holdout | Flatten_LSTM | 0.9267 | 0.9167 | 0.9167 | 0.1351 |
| pamap2 | overlapping | random_holdout | Improved_GNN_LSTM | 0.9400 | 0.9353 | 0.9369 | 0.1238 |
| hhar | non_overlapping | loso | CNN1D | 0.6185 | 0.6143 | 0.6117 | nan |
| hhar | non_overlapping | loso | Flatten_LSTM | 0.5402 | 0.5232 | 0.5420 | nan |
| hhar | non_overlapping | loso | Improved_GNN_LSTM | 0.5180 | 0.4845 | 0.5035 | nan |
| hhar | non_overlapping | random_holdout | CNN1D | 0.9049 | 0.9026 | 0.9032 | 0.2864 |
| hhar | non_overlapping | random_holdout | Flatten_LSTM | 0.9011 | 0.9014 | 0.9015 | 0.3609 |
| hhar | non_overlapping | random_holdout | Improved_GNN_LSTM | 0.9367 | 0.9365 | 0.9367 | 0.4187 |
| hhar | overlapping | loso | CNN1D | 0.6131 | 0.6067 | 0.6074 | nan |
| hhar | overlapping | loso | Flatten_LSTM | 0.5284 | 0.5002 | 0.5286 | nan |
| hhar | overlapping | loso | Improved_GNN_LSTM | 0.5340 | 0.5089 | 0.5214 | nan |
| hhar | overlapping | random_holdout | CNN1D | 0.9072 | 0.9037 | 0.9036 | 0.2941 |
| hhar | overlapping | random_holdout | Flatten_LSTM | 0.8678 | 0.8686 | 0.8675 | 0.3393 |
| hhar | overlapping | random_holdout | Improved_GNN_LSTM | 0.9144 | 0.9133 | 0.9143 | 0.3804 |

On PAMAP2 with overlapping windows, random holdout accuracy reached 99.4% (CNN1D) versus 74.0% under LOSO, yielding a leakage gap of 0.254. Flatten_LSTM and Improved_GNN_LSTM showed smaller but substantial gaps (0.135 and 0.124, respectively). Under non-overlapping windows, CNN1D exhibited the largest gap (0.417), whereas sequence models retained higher LOSO accuracy (Flatten_LSTM LOSO 0.810; Improved_GNN_LSTM LOSO 0.822). HHAR displayed a similar pattern: overlapping random holdout exceeded 0.90 for all models while LOSO accuracy remained near 0.52–0.61, with Improved_GNN_LSTM showing the largest overlapping leakage gap (0.380). These findings confirm that subject-level dependence in windowed sensor data materially inflates apparent performance when subjects appear in both training and test partitions (Figure 1).

### Experiment 2: Statistical Reliability

Table 2 reports LOSO accuracy and macro F1 means with 95% confidence intervals across nine folds. Table 3 presents pairwise model comparisons using bootstrap mean-difference intervals, Cohen's d, and Wilcoxon signed-rank tests.

**Table 2.** LOSO statistical reliability summary.

| Dataset | Model | Accuracy_Mean | Accuracy_SD | Accuracy_CI95_Lower | Accuracy_CI95_Upper | Macro_F1_Mean | Macro_F1_SD | Macro_F1_CI95_Lower | Macro_F1_CI95_Upper | Effect_Size | Wilcoxon_P_Value | Rank_Stability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pamap2 | CNN1D | 0.7450 | 0.1635 | 0.6382 | 0.8518 | 0.6537 | 0.2517 | 0.4893 | 0.8182 | 4.0482 | 0.0039 | rank1:9/9 |
| pamap2 | Flatten_LSTM | 0.7597 | 0.1013 | 0.6935 | 0.8259 | 0.6810 | 0.1473 | 0.5848 | 0.7772 | 6.6762 | 0.0039 | rank1:4/9; rank2:5/9 |
| pamap2 | Improved_GNN_LSTM | 0.7291 | 0.2742 | 0.5499 | 0.9082 | 0.6756 | 0.2579 | 0.5071 | 0.8441 | 2.3552 | 0.0078 | rank1:4/9; rank2:4/9; rank3:1/9 |
| hhar | CNN1D | 0.6131 | 0.0750 | 0.5641 | 0.6621 | 0.5902 | 0.0821 | 0.5365 | 0.6438 | 5.9502 | 0.0039 | rank1:9/9 |
| hhar | Flatten_LSTM | 0.5284 | 0.0724 | 0.4812 | 0.5757 | 0.4806 | 0.0771 | 0.4303 | 0.5310 | 4.9988 | 0.0039 | rank1:1/9; rank2:8/9 |
| hhar | Improved_GNN_LSTM | 0.5340 | 0.0776 | 0.4833 | 0.5847 | 0.4568 | 0.0925 | 0.3964 | 0.5172 | 4.7348 | 0.0039 | rank1:1/9; rank2:5/9; rank3:3/9 |

**Table 3.** Pairwise LOSO accuracy comparisons.

| Dataset | Metric | Model_A | Model_B | Mean_Difference | Bootstrap_CI95_Lower | Bootstrap_CI95_Upper | Effect_Size | Wilcoxon_P_Value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pamap2 | accuracy | CNN1D | Flatten_LSTM | -0.0147 | -0.1412 | 0.0888 | -0.0797 | 0.9102 |
| pamap2 | accuracy | CNN1D | Improved_GNN_LSTM | 0.0160 | -0.1668 | 0.2329 | 0.0505 | 0.6523 |
| pamap2 | accuracy | Flatten_LSTM | Improved_GNN_LSTM | 0.0307 | -0.0463 | 0.1540 | 0.1703 | 0.4961 |
| hhar | accuracy | CNN1D | Flatten_LSTM | 0.0847 | 0.0359 | 0.1333 | 1.0860 | 0.0195 |
| hhar | accuracy | CNN1D | Improved_GNN_LSTM | 0.0791 | 0.0405 | 0.1176 | 1.2329 | 0.0117 |
| hhar | accuracy | Flatten_LSTM | Improved_GNN_LSTM | -0.0056 | -0.0402 | 0.0396 | -0.0862 | 0.4258 |

On PAMAP2, no pairwise LOSO accuracy difference was statistically significant at α=0.05 (all Wilcoxon p > 0.49). On HHAR, CNN1D significantly outperformed both Flatten_LSTM (Δ=0.085, p=0.0195) and Improved_GNN_LSTM (Δ=0.079, p=0.0117), with large effect sizes (Cohen's d > 1.0). CNN1D achieved rank-1 LOSO accuracy on all nine HHAR folds, whereas deep sequence models showed unstable fold-level ranking.

### Experiment 3: Robustness

Table 4 summarizes accuracy degradation under sensor perturbations on both datasets. Gaussian noise produced graded performance drops (Figure 2). Random window removal caused severe degradation because perturbed evaluation altered effective test set size.

**Table 4.** Robustness summary (perturbation-induced accuracy drop).

| Dataset | Perturbation | Severity | Model | Clean_Accuracy | Perturbed_Accuracy | Accuracy_Drop | Clean_Macro_F1 | Perturbed_Macro_F1 | Macro_F1_Drop | Most_Affected_Class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pamap2 | remove_one_sensor_node | low | CNN1D | 0.7396 | 0.5323941790152169 | 0.2071898465014287 | 0.7469 | 0.5363449453121911 | 0.2105890962377847 | house_cleaning |
| pamap2 | remove_one_sensor_node | medium | CNN1D | 0.7396 | 0.5323941790152169 | 0.2071898465014287 | 0.7469 | 0.5363449453121911 | 0.2105890962377847 | house_cleaning |
| pamap2 | remove_one_sensor_node | high | CNN1D | 0.7396 | 0.5323941790152169 | 0.2071898465014287 | 0.7469 | 0.5363449453121911 | 0.2105890962377847 | house_cleaning |
| pamap2 | remove_heart_rate_channel | low | CNN1D | 0.7396 | not_applicable | not_applicable | 0.7469 | not_applicable | not_applicable | not_applicable |
| pamap2 | remove_heart_rate_channel | medium | CNN1D | 0.7396 | not_applicable | not_applicable | 0.7469 | not_applicable | not_applicable | not_applicable |
| pamap2 | remove_heart_rate_channel | high | CNN1D | 0.7396 | not_applicable | not_applicable | 0.7469 | not_applicable | not_applicable | not_applicable |
| pamap2 | gaussian_noise | low | CNN1D | 0.7396 | 0.7395840255166456 | 0.0 | 0.7469 | 0.7468597746189022 | 7.426693107370763e-05 | computer_work |
| pamap2 | gaussian_noise | medium | CNN1D | 0.7396 | 0.733337763306532 | 0.0062462622101135 | 0.7469 | 0.7411687590668145 | 0.0057652824831614 | lying |
| pamap2 | gaussian_noise | high | CNN1D | 0.7396 | 0.7066914745165792 | 0.0328925510000663 | 0.7469 | 0.7186393809063203 | 0.0282946606436556 | lying |
| pamap2 | mask_random_channels | low | CNN1D | 0.7396 | 0.7001794139145459 | 0.0394046116020997 | 0.7469 | 0.7007882820064867 | 0.0461457595434892 | walking |
| pamap2 | mask_random_channels | medium | CNN1D | 0.7396 | 0.4995016280151505 | 0.240082397501495 | 0.7469 | 0.455024120407612 | 0.2919099211423639 | walking |
| pamap2 | mask_random_channels | high | CNN1D | 0.7396 | 0.2856003721177487 | 0.4539836533988969 | 0.7469 | 0.2606819444274525 | 0.4862520971225233 | walking |
| pamap2 | remove_random_windows | low | CNN1D | 0.7396 | 0.7386956841408719 | 0.0008883413757736 | 0.7469 | 0.7460350840820764 | 0.0008989574678994 | sitting |
| pamap2 | remove_random_windows | medium | CNN1D | 0.7396 | 0.7353500761035008 | 0.0042339494131448 | 0.7469 | 0.7420308238350368 | 0.004903217714939 | sitting |
| pamap2 | remove_random_windows | high | CNN1D | 0.7396 | 0.7390438247011952 | 0.0005402008154503 | 0.7469 | 0.7460930234310336 | 0.0008410181189423 | sitting |
| pamap2 | missing_heart_rate_signals | low | CNN1D | 0.7396 | not_applicable | not_applicable | 0.7469 | not_applicable | not_applicable | not_applicable |
| pamap2 | missing_heart_rate_signals | medium | CNN1D | 0.7396 | not_applicable | not_applicable | 0.7469 | not_applicable | not_applicable | not_applicable |
| pamap2 | missing_heart_rate_signals | high | CNN1D | 0.7396 | not_applicable | not_applicable | 0.7469 | not_applicable | not_applicable | not_applicable |
| pamap2 | remove_one_sensor_node | low | Flatten_LSTM | 0.7916 | 0.4021304926764314 | 0.3894806924101198 | 0.7590 | 0.3050970789360325 | 0.4539506158376156 | cycling |
| pamap2 | remove_one_sensor_node | medium | Flatten_LSTM | 0.7916 | 0.4021304926764314 | 0.3894806924101198 | 0.7590 | 0.3050970789360325 | 0.4539506158376156 | cycling |
| pamap2 | remove_one_sensor_node | high | Flatten_LSTM | 0.7916 | 0.4021304926764314 | 0.3894806924101198 | 0.7590 | 0.3050970789360325 | 0.4539506158376156 | cycling |
| pamap2 | remove_heart_rate_channel | low | Flatten_LSTM | 0.7916 | not_applicable | not_applicable | 0.7590 | not_applicable | not_applicable | not_applicable |
| pamap2 | remove_heart_rate_channel | medium | Flatten_LSTM | 0.7916 | not_applicable | not_applicable | 0.7590 | not_applicable | not_applicable | not_applicable |
| pamap2 | remove_heart_rate_channel | high | Flatten_LSTM | 0.7916 | not_applicable | not_applicable | 0.7590 | not_applicable | not_applicable | not_applicable |
| pamap2 | gaussian_noise | low | Flatten_LSTM | 0.7916 | 0.7736351531291611 | 0.0179760319573901 | 0.7590 | 0.7446019950305315 | 0.0144456997431166 | transient |
| pamap2 | gaussian_noise | medium | Flatten_LSTM | 0.7916 | 0.6864181091877497 | 0.1051930758988015 | 0.7590 | 0.6528983714180457 | 0.1061493233556023 | transient |
| pamap2 | gaussian_noise | high | Flatten_LSTM | 0.7916 | 0.6011984021304927 | 0.1904127829560585 | 0.7590 | 0.5523438366887968 | 0.2067038580848512 | transient |
| pamap2 | mask_random_channels | low | Flatten_LSTM | 0.7916 | 0.6784287616511319 | 0.1131824234354194 | 0.7590 | 0.6435903419820604 | 0.1154573527915876 | cycling |
| pamap2 | mask_random_channels | medium | Flatten_LSTM | 0.7916 | 0.5292942743009321 | 0.2623169107856191 | 0.7590 | 0.4969500840685055 | 0.2620976107051425 | cycling |
| pamap2 | mask_random_channels | high | Flatten_LSTM | 0.7916 | 0.2509986684420772 | 0.540612516644474 | 0.7590 | 0.1659147326608772 | 0.5931329621127709 | sitting |

*(Table truncated; see `results/manuscript_tables/combined/table_exp3_robustness.csv` for full results.)*

### Experiment 4: Calibration and Uncertainty

**Table 5.** Expected calibration error (ECE), Brier score, negative log-likelihood, and selective-prediction accuracy at 90%, 80%, and 70% confidence coverage.

| Dataset | Model | ECE | Brier_Score | NLL | Accuracy_At_90_Coverage | Accuracy_At_80_Coverage | Accuracy_At_70_Coverage | Macro_F1_At_90_Coverage | Macro_F1_At_80_Coverage | Macro_F1_At_70_Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pamap2 | CNN1D | 0.1040 | 0.3980 | 1.1900 | 0.7808 | 0.8148 | 0.8487 | 0.7874 | 0.8187 | 0.8490 |
| pamap2 | Flatten_LSTM | 0.0984 | 0.3308 | 0.7629 | 0.8298 | 0.8585 | 0.8773 | 0.8000 | 0.8324 | 0.8327 |
| pamap2 | Improved_GNN_LSTM | 0.0657 | 0.2852 | 0.6609 | 0.8594 | 0.8876 | 0.9029 | 0.7820 | 0.8213 | 0.8317 |
| hhar | CNN1D | 0.1419 | 0.5335 | 1.3097 | 0.6509 | 0.6858 | 0.7199 | 0.6354 | 0.6627 | 0.6919 |
| hhar | Flatten_LSTM | 0.1088 | 0.5824 | 1.1297 | 0.5528 | 0.5775 | 0.6146 | 0.5025 | 0.5061 | 0.5052 |
| hhar | Improved_GNN_LSTM | 0.1059 | 0.5944 | 1.1752 | 0.5580 | 0.5828 | 0.6067 | 0.5186 | 0.5059 | 0.4751 |

On PAMAP2, Improved_GNN_LSTM achieved the lowest ECE (0.066) and supported the highest accuracy at 90% coverage (0.859). On HHAR, Flatten_LSTM and Improved_GNN_LSTM showed lower ECE than CNN1D, but selective prediction accuracy remained modest (best 90%-coverage accuracy 0.651), indicating limited confidence reliability for deployment (Figure 3).

### Experiment 5: Subject-Level Failure Analysis

**Table 6.** Per-subject LOSO metrics and dominant confusion patterns (excerpt).

| Subject | Dataset | Model | Accuracy | Macro_F1 | Balanced_Accuracy | Worst_Activity | Dominant_Confusion | Missingness | Activity_Imbalance | Sensor_Variability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | pamap2 | CNN1D | 0.7686 | 0.7804 | 0.7750 | sitting | lying→ironing | 0.0000 | 0.1090 | 1.0018 |
| 102 | pamap2 | CNN1D | 0.7270 | 0.7548 | 0.7461 | sitting | cycling→standing | 0.0000 | 0.1240 | 1.0022 |
| 103 | pamap2 | CNN1D | 0.7948 | 0.6385 | 0.7988 | walking | lying→sitting | 0.0000 | 0.1667 | 0.9974 |
| 104 | pamap2 | CNN1D | 0.7582 | 0.7561 | 0.7602 | walking | lying→sitting | 0.0000 | 0.1382 | 0.9988 |
| 105 | pamap2 | CNN1D | 0.8399 | 0.8457 | 0.8407 | lying | lying→ironing | 0.0000 | 0.1217 | 1.0003 |
| 106 | pamap2 | CNN1D | 0.8596 | 0.7859 | 0.8039 | house_cleaning | ironing→sitting | 0.0000 | 0.1513 | 1.0010 |
| 107 | pamap2 | CNN1D | 0.8757 | 0.8675 | 0.8658 | house_cleaning | sitting→lying | 0.0000 | 0.1454 | 0.9994 |
| 108 | pamap2 | CNN1D | 0.3315 | 0.2833 | 0.3178 | walking | standing→car_driving | 0.0000 | 0.1261 | 0.9994 |
| 109 | pamap2 | CNN1D | 0.7500 | 0.1714 | 0.7500 | transient | house_cleaning→cycling | 0.0000 | 1.0000 | 1.0307 |
| 101 | pamap2 | Flatten_LSTM | 0.8549 | 0.8538 | 0.8542 | transient | transient→lying | 0.0000 | 0.1090 | 1.0018 |
| 102 | pamap2 | Flatten_LSTM | 0.7647 | 0.6789 | 0.7226 | car_driving | car_driving→computer_work | 0.0000 | 0.1240 | 1.0022 |
| 103 | pamap2 | Flatten_LSTM | 0.7704 | 0.6174 | 0.7950 | walking | ironing→vacuum_cleaning | 0.0000 | 0.1667 | 0.9974 |

*(See combined CSV for all subjects.)*

Subject-level heatmaps (Figure 4) revealed heterogeneous failure modes. On HHAR, stair transitions (stairsup, stairsdown) and sitting were frequent worst-activity classes, with dominant confusions between sedentary and stair classes. Sensor variability and activity imbalance covaried with subject accuracy but did not fully explain fold-level variance.

### Experiment 6: Few-Shot Subject Calibration

With 10% subject-specific calibration data and full-model fine-tuning on HHAR, mean accuracy improved by 0.138 and macro F1 by 0.142 across subjects (Figure 5).
With 10% subject-specific calibration data and full-model fine-tuning on PAMAP2, mean accuracy improved by 0.069 and macro F1 by 0.089 across subjects (Figure 5).

**Table 7.** Few-shot calibration results (10% calibration fraction; excerpt).

| Dataset | Subject | Calibration_Percentage | Fine_Tuning_Strategy | Model | Uncalibrated_Accuracy | Calibrated_Accuracy | Accuracy_Improvement | Uncalibrated_Macro_F1 | Calibrated_Macro_F1 | Macro_F1_Improvement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pamap2 | 101 | 0.0000 | none | CNN1D | 0.7686 | 0.7686 | 0.0000 | 0.7804 | 0.7804 | 0.0000 |
| pamap2 | 101 | 0.1000 | full_model | CNN1D | 0.7686 | 0.8824 | 0.1138 | 0.7804 | 0.8846 | 0.1042 |
| pamap2 | 102 | 0.0000 | none | CNN1D | 0.7270 | 0.7270 | 0.0000 | 0.7548 | 0.7548 | 0.0000 |
| pamap2 | 102 | 0.1000 | full_model | CNN1D | 0.7270 | 0.8578 | 0.1309 | 0.7548 | 0.8630 | 0.1083 |
| pamap2 | 103 | 0.0000 | none | CNN1D | 0.7948 | 0.7948 | 0.0000 | 0.6385 | 0.6385 | 0.0000 |
| pamap2 | 103 | 0.1000 | full_model | CNN1D | 0.7948 | 0.9350 | 0.1402 | 0.6385 | 0.8363 | 0.1978 |
| pamap2 | 104 | 0.0000 | none | CNN1D | 0.7582 | 0.7582 | 0.0000 | 0.7561 | 0.7561 | 0.0000 |
| pamap2 | 104 | 0.1000 | full_model | CNN1D | 0.7582 | 0.8911 | 0.1329 | 0.7561 | 0.8913 | 0.1352 |
| pamap2 | 105 | 0.0000 | none | CNN1D | 0.8399 | 0.8399 | 0.0000 | 0.8457 | 0.8457 | 0.0000 |
| pamap2 | 105 | 0.1000 | full_model | CNN1D | 0.8399 | 0.9353 | 0.0954 | 0.8457 | 0.9383 | 0.0926 |
| pamap2 | 106 | 0.0000 | none | CNN1D | 0.8596 | 0.8596 | 0.0000 | 0.7859 | 0.7859 | 0.0000 |
| pamap2 | 106 | 0.1000 | full_model | CNN1D | 0.8596 | 0.9404 | 0.0808 | 0.7859 | 0.9403 | 0.1544 |
| pamap2 | 107 | 0.0000 | none | CNN1D | 0.8757 | 0.8757 | 0.0000 | 0.8675 | 0.8675 | 0.0000 |
| pamap2 | 107 | 0.1000 | full_model | CNN1D | 0.8757 | 0.9377 | 0.0620 | 0.8675 | 0.9370 | 0.0695 |
| pamap2 | 108 | 0.0000 | none | CNN1D | 0.3315 | 0.3315 | 0.0000 | 0.2833 | 0.2833 | 0.0000 |
| pamap2 | 108 | 0.1000 | full_model | CNN1D | 0.3315 | 0.8183 | 0.4868 | 0.2833 | 0.7698 | 0.4865 |
| pamap2 | 109 | 0.0000 | none | CNN1D | 0.7500 | 0.7500 | 0.0000 | 0.1714 | 0.1714 | 0.0000 |
| pamap2 | 109 | 0.1000 | full_model | CNN1D | 0.7500 | 0.6818 | -0.0682 | 0.1714 | 0.1622 | -0.0093 |
| pamap2 | 101 | 0.0000 | none | Flatten_LSTM | 0.8549 | 0.8549 | 0.0000 | 0.8538 | 0.8538 | 0.0000 |
| pamap2 | 101 | 0.1000 | full_model | Flatten_LSTM | 0.8549 | 0.8851 | 0.0301 | 0.8538 | 0.8688 | 0.0150 |

### Experiment 7: Health-Relevant Activity Groups

**Table 8.** Fine-grained versus group-level LOSO accuracy.

| Dataset | Model | Fine_Grained_Accuracy | Group_Level_Accuracy | Fine_Grained_Macro_F1 | Group_Level_Macro_F1 |
| --- | --- | --- | --- | --- | --- |
| pamap2 | CNN1D | 0.7396 | 0.7941 | 0.7469 | 0.7945 |
| pamap2 | Flatten_LSTM | 0.7916 | 0.8322 | 0.7590 | 0.8407 |
| pamap2 | Improved_GNN_LSTM | 0.8162 | 0.8622 | 0.7420 | 0.8683 |
| hhar | CNN1D | 0.6131 | 0.6586 | 0.6067 | 0.6585 |
| hhar | Flatten_LSTM | 0.5284 | 0.5651 | 0.5002 | 0.5681 |
| hhar | Improved_GNN_LSTM | 0.5340 | 0.5869 | 0.5089 | 0.5853 |

**Table 9.** Group-level sensitivity, specificity, and macro F1.

| Dataset | Clinical_Activity_Group | Included_Activities | Sensitivity | Specificity | Macro_F1 | Main_Confusion |
| --- | --- | --- | --- | --- | --- | --- |
| pamap2 | Ambulatory_Movement | standing; walking | 0.7730 | 0.9787 | 0.8248 | Ambulatory_Movement→Sedentary_or_Rest |
| pamap2 | Postural_Activities | ironing | 0.7086 | 0.9445 | 0.6737 | Postural_Activities→Sedentary_or_Rest |
| pamap2 | Sedentary_or_Rest | lying; sitting; computer_work; car_driving | 0.8279 | 0.8809 | 0.7897 | Sedentary_or_Rest→Postural_Activities |
| pamap2 | Stair_or_High_Intensity_Movement | running | 0.7653 | 0.9909 | 0.8212 | Stair_or_High_Intensity_Movement→Sedentary_or_Rest |
| pamap2 | Unmapped_or_Other | transient | 0.8755 | 0.9762 | 0.8374 | Unmapped_or_Other→Sedentary_or_Rest |
| pamap2 | Vigorous_Activity | cycling; vacuum_cleaning; house_cleaning | 0.7855 | 0.9652 | 0.8205 | Vigorous_Activity→Postural_Activities |
| pamap2 | Ambulatory_Movement | standing; walking | 0.8023 | 0.9782 | 0.8423 | Ambulatory_Movement→Sedentary_or_Rest |
| pamap2 | Postural_Activities | ironing | 0.7514 | 0.9643 | 0.7493 | Postural_Activities→Vigorous_Activity |
| pamap2 | Sedentary_or_Rest | lying; sitting; computer_work; car_driving | 0.8528 | 0.9077 | 0.8277 | Sedentary_or_Rest→Postural_Activities |
| pamap2 | Stair_or_High_Intensity_Movement | running | 0.8968 | 0.9920 | 0.9040 | Stair_or_High_Intensity_Movement→Vigorous_Activity |
| pamap2 | Unmapped_or_Other | transient | 0.9060 | 0.9889 | 0.9030 | Unmapped_or_Other→Sedentary_or_Rest |
| pamap2 | Vigorous_Activity | cycling; vacuum_cleaning; house_cleaning | 0.8139 | 0.9527 | 0.8177 | Vigorous_Activity→Sedentary_or_Rest |
| pamap2 | Ambulatory_Movement | standing; walking | 0.9316 | 0.9564 | 0.8719 | Ambulatory_Movement→Sedentary_or_Rest |
| pamap2 | Postural_Activities | ironing | 0.8811 | 0.9711 | 0.8446 | Postural_Activities→Vigorous_Activity |
| pamap2 | Sedentary_or_Rest | lying; sitting; computer_work; car_driving | 0.8442 | 0.9567 | 0.8696 | Sedentary_or_Rest→Ambulatory_Movement |
| pamap2 | Stair_or_High_Intensity_Movement | running | 0.9286 | 0.9862 | 0.8931 | Stair_or_High_Intensity_Movement→Vigorous_Activity |
| pamap2 | Unmapped_or_Other | transient | 0.9329 | 0.9897 | 0.9205 | Unmapped_or_Other→Sedentary_or_Rest |
| pamap2 | Vigorous_Activity | cycling; vacuum_cleaning; house_cleaning | 0.7603 | 0.9688 | 0.8101 | Vigorous_Activity→Ambulatory_Movement |
| hhar | Ambulatory_Movement | bike; walk | 0.6994 | 0.8289 | 0.6816 | Ambulatory_Movement→Stair_or_High_Intensity_Movement |
| hhar | Sedentary_or_Rest | sit; stand | 0.6243 | 0.8580 | 0.6471 | Sedentary_or_Rest→Stair_or_High_Intensity_Movement |
| hhar | Stair_or_High_Intensity_Movement | stairsup; stairsdown | 0.6517 | 0.7993 | 0.6468 | Stair_or_High_Intensity_Movement→Ambulatory_Movement |
| hhar | Ambulatory_Movement | bike; walk | 0.5221 | 0.9185 | 0.6179 | Ambulatory_Movement→Sedentary_or_Rest |
| hhar | Sedentary_or_Rest | sit; stand | 0.6907 | 0.6675 | 0.5743 | Sedentary_or_Rest→Stair_or_High_Intensity_Movement |
| hhar | Stair_or_High_Intensity_Movement | stairsup; stairsdown | 0.4925 | 0.7624 | 0.5122 | Stair_or_High_Intensity_Movement→Sedentary_or_Rest |
| hhar | Ambulatory_Movement | bike; walk | 0.6574 | 0.7923 | 0.6306 | Ambulatory_Movement→Stair_or_High_Intensity_Movement |
| hhar | Sedentary_or_Rest | sit; stand | 0.4605 | 0.9072 | 0.5548 | Sedentary_or_Rest→Stair_or_High_Intensity_Movement |
| hhar | Stair_or_High_Intensity_Movement | stairsup; stairsdown | 0.6350 | 0.6741 | 0.5706 | Stair_or_High_Intensity_Movement→Ambulatory_Movement |

Collapsing fine-grained labels into clinical activity groups improved group-level accuracy for CNN1D on HHAR (0.613 to 0.659) and for all models on PAMAP2 where group structure aligns with ambulatory versus sedentary behavior (Figure 6). 
All PAMAP2 activities in the evaluation set mapped to configured clinical groups; HHAR had no unmapped activities. 
Several configured PAMAP2 group labels (e.g., ascending_stairs, watching_TV) were absent from processed data and therefore did not contribute to group metrics.

## Discussion

This study demonstrates that window construction and evaluation protocol dominate reported HAR performance more than nominal architectural complexity. Large leakage gaps on both datasets show that overlapping windows with random splitting create optimistic bias that would mislead wearable-AI benchmarks and clinical feasibility claims. LOSO evaluation reduced accuracy by 20–40 percentage points relative to random holdout for several models, consistent with prior HAR literature emphasizing subject-independent validation.

Statistical testing revealed dataset-dependent model rankings. CNN1D significantly outperformed graph-sequence models on HHAR under LOSO, whereas PAMAP2 differences among deep models were not significant, suggesting that multichannel laboratory data favor convolutional temporal features while phone-sensor HHAR benefits from simpler pipelines given class imbalance and limited windows per subject.

Robustness experiments showed that missing windows and sensor masking produce severe degradation, highlighting deployment risk for intermittent connectivity. Calibration analysis indicated that low ECE on PAMAP2 does not transfer to HHAR, where confidence scores poorly rank correct predictions. Few-shot subject calibration partially mitigated subject shift, especially with full-model fine-tuning at 5–10% subject data, supporting personalization strategies for longitudinal wearable monitoring.

Subject-level failure analysis and health-group aggregation further showed that errors concentrate in biomechanically similar transitions (sit–stand–stairs on HHAR; lying–sitting on PAMAP2). Group-level metrics improved interpretability for digital-health endpoints but should be reported alongside fine-grained confusion to avoid masking clinically relevant misclassifications.

## Limitations

Non-overlapping evaluation used a subsampling approximation (every second window per subject) rather than a strict non-overlapping stride across the full recording, which may leave residual temporal dependence. All deep models were trained on CPU for reproducibility, increasing wall-clock time and precluding large-scale hyperparameter search. Heart-rate perturbations were not applicable to processed feature tensors. Activity-group analysis excluded labels absent from processed arrays and listed several YAML-configured activities that never appeared in training data. HHAR window caps (5,000 per subject) subsampled dense phone data and may underrepresent rare classes. Finally, LOSO folds with missing classes in validation triggered sklearn warnings and unstable per-class metrics for rare activities.

## Figure Captions

**fig_exp1_leakage_grouped_bar.** Figure 1. Overlapping-window classification accuracy under random holdout versus leave-one-subject-out (LOSO) evaluation for PAMAP2 and HHAR. (`results/manuscript_figures/fig_exp1_leakage_grouped_bar.png`)
**fig_exp3_robustness_hhar_CNN1D.** Figure 2. Accuracy degradation under increasing Gaussian sensor noise (low, medium, high) for each model and dataset. (`results/manuscript_figures/fig_exp3_robustness_hhar_CNN1D.png`)
**fig_exp3_robustness_hhar_Flatten_LSTM.** Figure 2. Accuracy degradation under increasing Gaussian sensor noise (low, medium, high) for each model and dataset. (`results/manuscript_figures/fig_exp3_robustness_hhar_Flatten_LSTM.png`)
**fig_exp3_robustness_hhar_Improved_GNN_LSTM.** Figure 2. Accuracy degradation under increasing Gaussian sensor noise (low, medium, high) for each model and dataset. (`results/manuscript_figures/fig_exp3_robustness_hhar_Improved_GNN_LSTM.png`)
**fig_exp3_robustness_pamap2_CNN1D.** Figure 2. Accuracy degradation under increasing Gaussian sensor noise (low, medium, high) for each model and dataset. (`results/manuscript_figures/fig_exp3_robustness_pamap2_CNN1D.png`)
**fig_exp3_robustness_pamap2_Flatten_LSTM.** Figure 2. Accuracy degradation under increasing Gaussian sensor noise (low, medium, high) for each model and dataset. (`results/manuscript_figures/fig_exp3_robustness_pamap2_Flatten_LSTM.png`)
**fig_exp3_robustness_pamap2_Improved_GNN_LSTM.** Figure 2. Accuracy degradation under increasing Gaussian sensor noise (low, medium, high) for each model and dataset. (`results/manuscript_figures/fig_exp3_robustness_pamap2_Improved_GNN_LSTM.png`)
**fig_exp4_reliability_hhar_CNN1D.** Figure 3. Reliability diagrams comparing predicted confidence to empirical accuracy under LOSO evaluation. (`results/manuscript_figures/fig_exp4_reliability_hhar_CNN1D.png`)
**fig_exp4_reliability_hhar_Flatten_LSTM.** Figure 3. Reliability diagrams comparing predicted confidence to empirical accuracy under LOSO evaluation. (`results/manuscript_figures/fig_exp4_reliability_hhar_Flatten_LSTM.png`)
**fig_exp4_reliability_hhar_Improved_GNN_LSTM.** Figure 3. Reliability diagrams comparing predicted confidence to empirical accuracy under LOSO evaluation. (`results/manuscript_figures/fig_exp4_reliability_hhar_Improved_GNN_LSTM.png`)
**fig_exp4_reliability_pamap2_CNN1D.** Figure 3. Reliability diagrams comparing predicted confidence to empirical accuracy under LOSO evaluation. (`results/manuscript_figures/fig_exp4_reliability_pamap2_CNN1D.png`)
**fig_exp4_reliability_pamap2_Flatten_LSTM.** Figure 3. Reliability diagrams comparing predicted confidence to empirical accuracy under LOSO evaluation. (`results/manuscript_figures/fig_exp4_reliability_pamap2_Flatten_LSTM.png`)
**fig_exp4_reliability_pamap2_Improved_GNN_LSTM.** Figure 3. Reliability diagrams comparing predicted confidence to empirical accuracy under LOSO evaluation. (`results/manuscript_figures/fig_exp4_reliability_pamap2_Improved_GNN_LSTM.png`)
**fig_exp5_cm_hhar_CNN1D.** Supplementary figure: fig_exp5_cm_hhar_CNN1D. (`results/manuscript_figures/fig_exp5_cm_hhar_CNN1D.png`)
**fig_exp5_cm_hhar_Flatten_LSTM.** Supplementary figure: fig_exp5_cm_hhar_Flatten_LSTM. (`results/manuscript_figures/fig_exp5_cm_hhar_Flatten_LSTM.png`)
**fig_exp5_cm_hhar_Improved_GNN_LSTM.** Supplementary figure: fig_exp5_cm_hhar_Improved_GNN_LSTM. (`results/manuscript_figures/fig_exp5_cm_hhar_Improved_GNN_LSTM.png`)
**fig_exp5_cm_pamap2_CNN1D.** Supplementary figure: fig_exp5_cm_pamap2_CNN1D. (`results/manuscript_figures/fig_exp5_cm_pamap2_CNN1D.png`)
**fig_exp5_cm_pamap2_Flatten_LSTM.** Supplementary figure: fig_exp5_cm_pamap2_Flatten_LSTM. (`results/manuscript_figures/fig_exp5_cm_pamap2_Flatten_LSTM.png`)
**fig_exp5_cm_pamap2_Improved_GNN_LSTM.** Supplementary figure: fig_exp5_cm_pamap2_Improved_GNN_LSTM. (`results/manuscript_figures/fig_exp5_cm_pamap2_Improved_GNN_LSTM.png`)
**fig_exp5_heatmap_hhar_CNN1D.** Figure 4. Per-subject per-class recall heatmaps identifying subject-level failure modes. (`results/manuscript_figures/fig_exp5_heatmap_hhar_CNN1D.png`)
**fig_exp5_heatmap_hhar_Flatten_LSTM.** Figure 4. Per-subject per-class recall heatmaps identifying subject-level failure modes. (`results/manuscript_figures/fig_exp5_heatmap_hhar_Flatten_LSTM.png`)
**fig_exp5_heatmap_hhar_Improved_GNN_LSTM.** Figure 4. Per-subject per-class recall heatmaps identifying subject-level failure modes. (`results/manuscript_figures/fig_exp5_heatmap_hhar_Improved_GNN_LSTM.png`)
**fig_exp5_heatmap_pamap2_CNN1D.** Figure 4. Per-subject per-class recall heatmaps identifying subject-level failure modes. (`results/manuscript_figures/fig_exp5_heatmap_pamap2_CNN1D.png`)
**fig_exp5_heatmap_pamap2_Flatten_LSTM.** Figure 4. Per-subject per-class recall heatmaps identifying subject-level failure modes. (`results/manuscript_figures/fig_exp5_heatmap_pamap2_Flatten_LSTM.png`)
**fig_exp5_heatmap_pamap2_Improved_GNN_LSTM.** Figure 4. Per-subject per-class recall heatmaps identifying subject-level failure modes. (`results/manuscript_figures/fig_exp5_heatmap_pamap2_Improved_GNN_LSTM.png`)
**fig_exp6_cal_hhar_CNN1D_classifier_head_only.** Figure 5. Few-shot calibration curves showing accuracy improvement versus fraction of held-out subject data used for fine-tuning. (`results/manuscript_figures/fig_exp6_cal_hhar_CNN1D_classifier_head_only.png`)
**fig_exp6_cal_hhar_CNN1D_full_model.** Figure 5. Few-shot calibration curves showing accuracy improvement versus fraction of held-out subject data used for fine-tuning. (`results/manuscript_figures/fig_exp6_cal_hhar_CNN1D_full_model.png`)
**fig_exp6_cal_hhar_Flatten_LSTM_classifier_head_only.** Figure 5. Few-shot calibration curves showing accuracy improvement versus fraction of held-out subject data used for fine-tuning. (`results/manuscript_figures/fig_exp6_cal_hhar_Flatten_LSTM_classifier_head_only.png`)
**fig_exp6_cal_hhar_Flatten_LSTM_full_model.** Figure 5. Few-shot calibration curves showing accuracy improvement versus fraction of held-out subject data used for fine-tuning. (`results/manuscript_figures/fig_exp6_cal_hhar_Flatten_LSTM_full_model.png`)
**fig_exp6_cal_hhar_Improved_GNN_LSTM_classifier_head_only.** Figure 5. Few-shot calibration curves showing accuracy improvement versus fraction of held-out subject data used for fine-tuning. (`results/manuscript_figures/fig_exp6_cal_hhar_Improved_GNN_LSTM_classifier_head_only.png`)
**fig_exp6_cal_hhar_Improved_GNN_LSTM_full_model.** Figure 5. Few-shot calibration curves showing accuracy improvement versus fraction of held-out subject data used for fine-tuning. (`results/manuscript_figures/fig_exp6_cal_hhar_Improved_GNN_LSTM_full_model.png`)
**fig_exp6_cal_pamap2_CNN1D_classifier_head_only.** Figure 5. Few-shot calibration curves showing accuracy improvement versus fraction of held-out subject data used for fine-tuning. (`results/manuscript_figures/fig_exp6_cal_pamap2_CNN1D_classifier_head_only.png`)
**fig_exp6_cal_pamap2_CNN1D_full_model.** Figure 5. Few-shot calibration curves showing accuracy improvement versus fraction of held-out subject data used for fine-tuning. (`results/manuscript_figures/fig_exp6_cal_pamap2_CNN1D_full_model.png`)
**fig_exp6_cal_pamap2_Flatten_LSTM_classifier_head_only.** Figure 5. Few-shot calibration curves showing accuracy improvement versus fraction of held-out subject data used for fine-tuning. (`results/manuscript_figures/fig_exp6_cal_pamap2_Flatten_LSTM_classifier_head_only.png`)
**fig_exp6_cal_pamap2_Flatten_LSTM_full_model.** Figure 5. Few-shot calibration curves showing accuracy improvement versus fraction of held-out subject data used for fine-tuning. (`results/manuscript_figures/fig_exp6_cal_pamap2_Flatten_LSTM_full_model.png`)
**fig_exp6_cal_pamap2_Improved_GNN_LSTM_classifier_head_only.** Figure 5. Few-shot calibration curves showing accuracy improvement versus fraction of held-out subject data used for fine-tuning. (`results/manuscript_figures/fig_exp6_cal_pamap2_Improved_GNN_LSTM_classifier_head_only.png`)
**fig_exp6_cal_pamap2_Improved_GNN_LSTM_full_model.** Figure 5. Few-shot calibration curves showing accuracy improvement versus fraction of held-out subject data used for fine-tuning. (`results/manuscript_figures/fig_exp6_cal_pamap2_Improved_GNN_LSTM_full_model.png`)
**fig_exp7_health_cm_hhar_CNN1D.** Figure 6. Confusion matrices at the health-relevant activity-group level. (`results/manuscript_figures/fig_exp7_health_cm_hhar_CNN1D.png`)
**fig_exp7_health_cm_hhar_Flatten_LSTM.** Figure 6. Confusion matrices at the health-relevant activity-group level. (`results/manuscript_figures/fig_exp7_health_cm_hhar_Flatten_LSTM.png`)
**fig_exp7_health_cm_hhar_Improved_GNN_LSTM.** Figure 6. Confusion matrices at the health-relevant activity-group level. (`results/manuscript_figures/fig_exp7_health_cm_hhar_Improved_GNN_LSTM.png`)
**fig_exp7_health_cm_pamap2_CNN1D.** Figure 6. Confusion matrices at the health-relevant activity-group level. (`results/manuscript_figures/fig_exp7_health_cm_pamap2_CNN1D.png`)
**fig_exp7_health_cm_pamap2_Flatten_LSTM.** Figure 6. Confusion matrices at the health-relevant activity-group level. (`results/manuscript_figures/fig_exp7_health_cm_pamap2_Flatten_LSTM.png`)
**fig_exp7_health_cm_pamap2_Improved_GNN_LSTM.** Figure 6. Confusion matrices at the health-relevant activity-group level. (`results/manuscript_figures/fig_exp7_health_cm_pamap2_Improved_GNN_LSTM.png`)

## Reviewer Risk Flags

1. **Leakage magnitude**: Random-holdout accuracies above 0.99 on PAMAP2 may appear implausible to reviewers; emphasize protocol difference and provide non-overlapping results.
2. **HHAR deep-model underperformance**: Improved_GNN_LSTM did not outperform CNN1D under LOSO; reviewers may question architectural contribution without transfer or ablation studies.
3. **Non-significant PAMAP2 pairwise tests**: Wide confidence intervals and high fold variance limit claims of model superiority on PAMAP2.
4. **Robustness window-drop artifact**: Extreme accuracy drops under random window removal partly reflect evaluation design rather than pure sensor failure.
5. **HHAR calibration**: High ECE and weak selective-prediction performance undermine claims about uncertainty-aware deployment on phone data.
6. **Few-shot gains**: Improvements may reflect test-set adaptation rather than true prospective calibration; prospective locked-subject protocols would strengthen claims.
7. **Activity-group mapping**: PAMAP2 YAML includes labels not present in processed data; reviewers may request justification of group definitions and unmapped activities.
8. **CPU-only training**: Computational constraints may leave performance on the table relative to GPU-tuned baselines in the literature.
9. **Class-absent folds**: LOSO warnings for rare classes affect macro-F1 stability and should be disclosed in supplementary per-fold tables.
10. **Dataset-specific CSV overwrite**: Publication runs per dataset overwrote single-table artifacts; combined tables rely on snapshot reconstruction and should be version-controlled.
