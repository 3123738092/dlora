#!/usr/bin/env bash
# Usage: bash scripts/eval_ocr.sh
# Compares PaddleOCR detections on baseline vs side-channel outputs.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/nas-files/yuhanchen/miniconda3/envs/lora/bin/python}
export LD_LIBRARY_PATH=/nas-files/yuhanchen/miniconda3/envs/lora/lib:${LD_LIBRARY_PATH:-}
export PYTHONDONTWRITEBYTECODE=1
BASELINE_DIR=${BASELINE_DIR:-/nas-files/yuhanchen/tmp/videolq_out}
SC_DIR=${SC_DIR:-/nas-files/yuhanchen/tmp/videolq_sc_out}
GPU_FLAG=${GPU_FLAG:---use_gpu}
mkdir -p runs
echo "==== baseline OCR ===="
"$PY" tools/eval_ocr.py --pred_dir "$BASELINE_DIR" $GPU_FLAG --out_csv runs/ocr_baseline.csv
echo
echo "==== + side-channel OCR ===="
"$PY" tools/eval_ocr.py --pred_dir "$SC_DIR"       $GPU_FLAG --out_csv runs/ocr_sidechannel.csv
