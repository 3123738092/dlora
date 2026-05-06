#!/usr/bin/env bash
# Generic 2-GPU evaluation with configurable dataset and model config
# Usage: eval_2gpu.sh <config> <dataset>
#   config: w1 | w3 | w4 | w4sc
#   dataset: spmcs | videolq
set -euo pipefail

CONFIG="${1:?usage: eval_2gpu.sh <config> <dataset>}"
DATASET="${2:?usage: eval_2gpu.sh <config> <dataset>}"
cd "$(dirname "$0")/.."

PY=/nas-files/yuhanchen/miniconda3/envs/lora/bin/python

case "$CONFIG" in
    w1)
        CKPT=preset/models/checkpoints/model.pkl
        FLOW_EST=spynet
        LOAD_CFR=""
        SC_CKPT=""
        MIXED_PREC=""
        LABEL="W1_Baseline"
        ;;
    w3)
        CKPT=preset/models/checkpoints/model.pkl
        FLOW_EST=spynet
        LOAD_CFR=""
        SC_CKPT="--sidechannel_ckpt runs/sidechannel_v2_lastframe/sidechannel_step005000.pt"
        MIXED_PREC=""
        LABEL="W3_SideChannel"
        ;;
    w4)
        CKPT=runs/cfr_raft_finetune/checkpoints/model_52001.pkl
        FLOW_EST=raft
        LOAD_CFR="--load_cfr"
        SC_CKPT=""
        MIXED_PREC="--mixed_precision no"
        LABEL="W4_CFR-RAFT-FT"
        ;;
    w4sc)
        CKPT=runs/cfr_raft_finetune/checkpoints/model_52001.pkl
        FLOW_EST=raft
        LOAD_CFR="--load_cfr"
        SC_CKPT="--sidechannel_ckpt runs/sidechannel_v2_lastframe/sidechannel_step005000.pt"
        MIXED_PREC="--mixed_precision no"
        LABEL="W4SC_RAFT-FT+SC"
        ;;
    *)
        echo "Unknown config: $CONFIG"
        exit 1
        ;;
esac

case "$DATASET" in
    spmcs)
        INPUT=/nas-files/yuhanchen/tmp/spmcs_in
        GT=/nas-files/yuhanchen/CV/data/test_sets/SPMCS/GT
        OUT=/nas-files/yuhanchen/tmp/spmcs_${CONFIG}_out
        METRIC_CSV=runs/spmcs_metrics_${CONFIG}.csv
        HAS_GT=true
        ;;
    videolq)
        INPUT=/nas-files/yuhanchen/tmp/videolq_in
        GT=""
        OUT=/nas-files/yuhanchen/tmp/videolq_${CONFIG}_out
        HAS_GT=false
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        exit 1
        ;;
esac

LOG=runs/eval_${DATASET}_${CONFIG}.log

echo "==== $LABEL on $DATASET (2-GPU) ====" | tee "$LOG"
echo "CKPT: $CKPT" | tee -a "$LOG"
echo "Config: $CONFIG | Dataset: $DATASET" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

mkdir -p "$OUT"

CLIPS=($(ls "$INPUT"/*.mp4 2>/dev/null))
TOTAL=${#CLIPS[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo "ERROR: No mp4 files found in $INPUT" | tee -a "$LOG"
    exit 1
fi

HALF=$(( (TOTAL + 1) / 2 ))

run_gpu() {
    local gpu=$1; shift
    local log="/tmp/eval_${DATASET}_${CONFIG}_gpu${gpu}.log"
    export CUDA_VISIBLE_DEVICES=$gpu
    for clip in "$@"; do
        name=$(basename "$clip" .mp4)
        echo "[GPU$gpu $(date +%H:%M:%S)] Processing $name ..." | tee -a "$LOG" "$log"
        $PY src/test_DLoRAL.py             --input_image "$clip"             --output_dir "$OUT"             --pretrained_path "$CKPT"             $LOAD_CFR             --flow_estimator $FLOW_EST             $SC_CKPT             $MIXED_PREC 2>&1 | tail -5 | tee -a "$log"
        echo "[GPU$gpu $(date +%H:%M:%S)] Done $name" | tee -a "$LOG" "$log"
    done
}

echo "Total clips: $TOTAL, per GPU: $HALF" | tee -a "$LOG"

run_gpu 0 "${CLIPS[@]:0:$HALF}" &
PID0=$!
run_gpu 1 "${CLIPS[@]:$HALF}" &
PID1=$!

echo "Waiting for GPU0 PID=$PID0 GPU1 PID=$PID1 ..." | tee -a "$LOG"
wait $PID0 $PID1

if [ "$HAS_GT" = "true" ] && [ -d "$GT" ]; then
    echo "" | tee -a "$LOG"
    echo "==== Metrics ====" | tee -a "$LOG"
    $PY tools/eval_metrics.py         --pred_dir "$OUT"         --gt_dir "$GT"         --out_csv "$METRIC_CSV" 2>&1 | tee -a "$LOG"
    echo "" | tee -a "$LOG"
    cat "$METRIC_CSV" | tee -a "$LOG"
else
    echo "No GT — skipping metrics" | tee -a "$LOG"
fi

echo "" | tee -a "$LOG"
echo "==== Done: $(date) ====" | tee -a "$LOG"
