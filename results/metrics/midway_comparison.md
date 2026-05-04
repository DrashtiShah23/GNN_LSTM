| Dataset | Model | Experiment Type | Split | Acc | Acc σ | Macro-F1 | F1 σ | Balanced Acc |
|---|---|---|---|---:|---:|---:|---:|---:|
| HHAR | RandomForest | baseline | LOSO | 0.5630 |  | 0.5481 |  |  |
| HHAR | SVM | baseline | LOSO | 0.5812 |  | 0.5668 |  |  |
| HHAR | XGBoost | baseline | LOSO | 0.5900 |  | 0.5779 |  |  |
| HHAR | GNN | deep | LOSO | 0.5932 | 0.1447 | 0.5957 | 0.1412 | 0.5903 |
| HHAR | GNN+LSTM | deep | LOSO | 0.5751 | 0.1114 | 0.5493 | 0.1263 | 0.5583 |
| HHAR | LSTM | deep | LOSO | 0.5200 | 0.1067 | 0.5223 | 0.1124 | 0.5201 |
| HHAR | CNN1D | holdout | 80/20 | 0.8943 |  | 0.8903 |  | 0.8908 |
| HHAR | GNN | holdout | 80/20 | 0.8573 |  | 0.8512 |  | 0.8507 |
| HHAR | LSTM | holdout | 80/20 | 0.7832 |  | 0.7766 |  | 0.7772 |
| HHAR | RF | holdout | 80/20 | 0.9068 |  | 0.9031 |  | 0.9024 |
| HHAR | SVM | holdout | 80/20 | 0.7980 |  | 0.7885 |  | 0.7888 |
| HHAR | XGBOOST | holdout | 80/20 | 0.8860 |  | 0.8818 |  | 0.8811 |
| PAMAP2 | RandomForest | baseline | LOSO | 0.7749 |  | 0.7121 |  |  |
| PAMAP2 | SVM | baseline | LOSO | 0.7918 |  | 0.7244 |  |  |
| PAMAP2 | XGBoost | baseline | LOSO | 0.8076 |  | 0.7314 |  |  |
| PAMAP2 | GNN | deep | LOSO | 0.6824 | 0.1697 | 0.6676 | 0.1986 | 0.6646 |
| PAMAP2 | GNN+LSTM | deep | LOSO | 0.6385 | 0.2159 | 0.5845 | 0.1997 | 0.5950 |
| PAMAP2 | LSTM | deep | LOSO | 0.6077 | 0.1044 | 0.6046 | 0.1698 | 0.5996 |
| PAMAP2 | Flatten+LSTM | ablation | LOSO | 0.8453 |  | 0.7935 |  | 0.7886 |
| PAMAP2 | GNN (fixed adj) | ablation | LOSO | 0.6911 |  | 0.6783 |  | 0.6717 |
| PAMAP2 | GNN (learnable adj) | ablation | LOSO | 0.6940 |  | 0.6890 |  | 0.6805 |
| PAMAP2 | CNN1D | holdout | 80/20 | 0.9934 |  | 0.9938 |  | 0.9941 |
| PAMAP2 | GNN | holdout | 80/20 | 0.8847 |  | 0.8874 |  | 0.8844 |
| PAMAP2 | LSTM | holdout | 80/20 | 0.7458 |  | 0.7432 |  | 0.7390 |
| PAMAP2 | RF | holdout | 80/20 | 0.9605 |  | 0.9641 |  | 0.9600 |
| PAMAP2 | SVM | holdout | 80/20 | 0.9651 |  | 0.9666 |  | 0.9646 |
| PAMAP2 | XGBOOST | holdout | 80/20 | 0.9834 |  | 0.9847 |  | 0.9838 |
