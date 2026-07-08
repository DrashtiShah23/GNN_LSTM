# PAMAP2 Protocol12 Deliverables

This folder collects the final top4 PAMAP2 deliverables requested by `Final_Seven_HAR_Experiments_Formatted_Standard.docx`.

Source artifact:

`results/canonical_protocol12_seven_experiments_top4`

Final model scope:

- Baselines: `random_forest`, `knn_k5`
- Deep models: `improved_gnn_lstm_res`, `improved_gnn_lstm_attn_adj_resbn`

## Contents

| Folder/File | Contents |
|---|---|
| `tables/` | Main CSV tables for Experiments 1-7 |
| `enhanced_tables/` | Enhanced analysis CSVs for statistical reliability, calibration, subject failures, and health groups |
| `detail_tables/` | Detailed real Exp3 robustness and Exp6 few-shot calibration CSVs |
| `figures/` | Final PNG and PDF figures generated for the manuscript deliverables |
| `manifests/` | Run and report manifests proving completion status |
| `report.md` | Human-written final interpretation and direct answers |
| `PAMAP2_SEVEN_EXPERIMENTS_REPORT.md` | Auto-generated manuscript artifact summary |

## Completion Status

- `run_manifest.json`: `summary_rows=48`, `fold_rows=240`, `prediction_rows=651576`
- `pamap2_docx_standard_report_manifest.json`: `missing=[]`
- Exp3 robustness: real table present
- Exp6 few-shot calibration: real table present
- Exp4 probability/calibration rows: complete for all final top4 rows

