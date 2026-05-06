#!/usr/bin/env bash
# 2-GPU parallel: CFR-RAFT FT + SideChannel UDM10 evaluation
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/nas-files/yuhanchen/miniconda3/envs/lora/bin/python
CKPT=runs/cfr_raft_finetune/checkpoints/model_52001.pkl
SC_CKPT=runs/sidechannel_v2_lastframe/sidechannel_step005000.pt
INPUT=/nas-files/yuhanchen/tmp/udm10_in
GT=/nas-files/yuhanchen/CV/data/test_sets/UDM10/GT
OUT=/nas-files/yuhanchen/tmp/udm10_cfr_raft_ft_sc_out
LOG=runs/cfr_raft_finetune/udm10_eval_raft_ft_sc.log

echo "==== CFR-RAFT FT + SideChannel UDM10 Eval (2-GPU) ====" | tee "$LOG"
echo "CKPT: $CKPT" | tee -a "$LOG"
echo "SC_CKPT: $SC_CKPT" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

mkdir -p "$OUT"

CLIPS=($(ls "$INPUT"/*.mp4))
TOTAL=${#CLIPS[@]}
HALF=$(( (TOTAL + 1) / 2 ))

run_gpu() {
    local gpu=$1; shift
    local log="/tmp/udm10_raft_sc_gpu${gpu}.log"
    export CUDA_VISIBLE_DEVICES=$gpu
    for clip in "$@"; do
        name=$(basename "$clip" .mp4)
        echo "[GPU$gpu $(date +%H:%M:%S)] Processing $name ..." | tee -a "$LOG" "$log"
        "$PY" src/test_DLoRAL.py             --input_image "$clip"             --output_dir "$OUT"             --pretrained_path "$CKPT"             --load_cfr             --flow_estimator raft             --sidechannel_ckpt "$SC_CKPT"             --mixed_precision no 2>&1 | tail -5 | tee -a "$log"
        echo "[GPU$gpu $(date +%H:%M:%S)] Done $name" | tee -a "$LOG" "$log"
    done
}

echo "Total clips: $TOTAL, per GPU: $HALF" | tee -a "$LOG"
echo "GPU 0 clips: ${CLIPS[@]:0:$HALF}" | tee -a "$LOG"
echo "GPU 1 clips: ${CLIPS[@]:$HALF}" | tee -a "$LOG"

run_gpu 0 "${CLIPS[@]:0:$HALF}" &
PID0=$!
run_gpu 1 "${CLIPS[@]:$HALF}" &
PID1=$!

echo "Waiting for GPU0 PID=$PID0 GPU1 PID=$PID1 ..." | tee -a "$LOG"
wait $PID0 $PID1

echo | tee -a "$LOG"
echo "==== Metrics ====" | tee -a "$LOG"
"$PY" tools/eval_metrics.py     --pred_dir "$OUT"     --gt_dir "$GT"     --out_csv runs/cfr_raft_finetune/metrics_udm10_raft_ft_sc.csv 2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "==== Done: $(date) ====" | tee -a "$LOG"
cat runs/cfr_raft_finetune/metrics_udm10_raft_ft_sc.csv | tee -a "$LOG"
