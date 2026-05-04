#!/usr/bin/env bash
# Run full LOSO pipeline (CPU recommended on macOS to avoid MPS OOM mid-LOSO),
# then exports, significance, cross-dataset transfer, and LIME.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
export HAR_FORCE_DEVICE="${HAR_FORCE_DEVICE:-cpu}"
mkdir -p results/metrics results/plots
echo "=== $(date -u) run_full_pipeline (HAR_FORCE_DEVICE=$HAR_FORCE_DEVICE) ===" | tee -a results/run_full_pipeline.log
"$PY" scripts/run_full_pipeline.py 2>&1 | tee -a results/run_full_pipeline.log
echo "=== $(date -u) export_midway_comparison ===" | tee -a results/run_full_pipeline.log
"$PY" scripts/export_midway_comparison.py
echo "=== $(date -u) loso_significance_and_pairs ===" | tee -a results/run_full_pipeline.log
"$PY" scripts/loso_significance_and_pairs.py
echo "=== $(date -u) cross_dataset_transfer ===" | tee -a results/run_full_pipeline.log
"$PY" scripts/cross_dataset_transfer.py
# Add --lstm for slow neural transfer: "$PY" scripts/cross_dataset_transfer.py --lstm
echo "=== $(date -u) run_lime_pamap2_lstm ===" | tee -a results/run_full_pipeline.log
"$PY" scripts/run_lime_pamap2_lstm.py
echo "=== $(date -u) regenerate_all_plots ===" | tee -a results/run_full_pipeline.log
"$PY" scripts/regenerate_all_plots.py
echo "=== $(date -u) ALL DONE ===" | tee -a results/run_full_pipeline.log
