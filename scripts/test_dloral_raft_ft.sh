#!/usr/bin/env bash
# W4-fix: CFR-RAFT fine-tuned evaluation on UDM10
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/nas-files/yuhanchen/miniconda3/envs/lora/bin/python
CKPT=runs/cfr_raft_finetune/checkpoints/model_52001.pkl
INPUT=/nas-files/yuhanchen/tmp/udm10_in
GT=/nas-files/yuhanchen/CV/data/test_sets/UDM10/GT
OUT=/nas-files/yuhanchen/tmp/udm10_cfr_raft_ft_out
GPU=${GPU:-0}
LOG=runs/cfr_raft_finetune/udm10_eval.log

echo "==== CFR-RAFT Fine-Tuned UDM10 Eval ====" | tee "$LOG"
echo "CKPT: $CKPT" | tee -a "$LOG"
echo "GPU: $GPU" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

export CUDA_VISIBLE_DEVICES=$GPU

mkdir -p "$OUT"
for clip in "$INPUT"/*.mp4; do
    name=$(basename "$clip" .mp4)
    echo "[$(date '+%H:%M:%S')] Processing $name ..." | tee -a "$LOG"
    "$PY" src/test_DLoRAL.py         --input_image "$clip"         --output_dir "$OUT"         --pretrained_path "$CKPT"         --load_cfr         --flow_estimator raft 2>&1 | tail -5 | tee -a "$LOG"
done

echo | tee -a "$LOG"
echo "==== Metrics ====" | tee -a "$LOG"
"$PY" tools/eval_metrics.py     --pred_dir "$OUT"     --gt_dir "$GT"     --out_csv runs/cfr_raft_finetune/metrics_udm10_ft.csv 2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "==== Done: $(date) ====" | tee -a "$LOG"
cat runs/cfr_raft_finetune/metrics_udm10_ft.csv | tee -a "$LOG"
