#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Default: 2 GPUs (0,1). Override with GPU_IDS env.
GPU_IDS=${GPU_IDS:-0,1}
NPROC=$(echo "${GPU_IDS}" | awk -F',' '{print NF}')
CFG=${CFG:-configs/side_channel.yaml}

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
accelerate launch \
  --num_processes="${NPROC}" \
  --num_machines=1 \
  --mixed_precision=no \
  src/train_side_channel.py --config "${CFG}"
