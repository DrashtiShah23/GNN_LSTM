| Dataset | Model | Experiment Type | Split | Accuracy | Macro-F1 | Balanced Acc |
|---|---|---|---|---:|---:|---:|
| HHAR | XGBoost | baseline | LOSO | 0.5900 | 0.5779 |  |
| HHAR | CNN1D | deep | LOSO | 0.6627 | 0.6585 | 0.6602 |
| HHAR | RF | holdout | 80/20 | 0.9068 | 0.9031 | 0.9024 |
| PAMAP2 | XGBoost | baseline | LOSO | 0.8076 | 0.7314 |  |
| PAMAP2 | GNN | deep | LOSO | 0.7206 | 0.7151 | 0.7080 |
| PAMAP2 | Flatten+LSTM | ablation | LOSO | 0.8453 | 0.7935 | 0.7886 |
| PAMAP2 | CNN1D | holdout | 80/20 | 0.9934 | 0.9938 | 0.9941 |
