#!/usr/bin/env bash
# Usage: bash scripts/eval_udm10.sh
# Runs PSNR/SSIM/LPIPS on baseline + side-channel UDM10 outputs and saves CSVs.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/nas-files/yuhanchen/miniconda3/envs/lora/bin/python}
GT=${GT:-/nas-files/yuhanchen/CV/data/test_sets/UDM10/GT}
BASELINE_DIR=${BASELINE_DIR:-/nas-files/yuhanchen/tmp/udm10_out}
SC_DIR=${SC_DIR:-/nas-files/yuhanchen/tmp/udm10_sc_out}
mkdir -p runs

echo "==== baseline (DLoRAL) ===="
"$PY" tools/eval_metrics.py --pred_dir "$BASELINE_DIR" --gt_dir "$GT" --out_csv runs/metrics_baseline.csv
echo
echo "==== + side-channel (W3) ===="
"$PY" tools/eval_metrics.py --pred_dir "$SC_DIR"       --gt_dir "$GT" --out_csv runs/metrics_sidechannel.csv
