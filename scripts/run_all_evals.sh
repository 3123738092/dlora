#!/usr/bin/env bash
# Master runner: SPMCS (W1, W3, W4, W4SC) then VideoLQ (W1, W3, W4, W4SC)
# Each config runs 2-GPU parallel internally, configs run sequentially.
set -euo pipefail
cd "$(dirname "$0")/.."

MASTER_LOG=runs/master_eval_$(date +%Y%m%d_%H%M).log

run_one() {
    local config=$1 dataset=$2
    echo "" | tee -a "$MASTER_LOG"
    echo "========== $dataset $config START $(date) ==========" | tee -a "$MASTER_LOG"
    bash scripts/eval_2gpu.sh "$config" "$dataset" 2>&1 | tee -a "$MASTER_LOG"
    echo "========== $dataset $config DONE $(date) ==========" | tee -a "$MASTER_LOG"
}

echo "==== Master Eval Start: $(date) ====" | tee "$MASTER_LOG"

for dataset in spmcs videolq; do
    for config in w1 w3 w4 w4sc; do
        # Check if input data exists
        case $dataset in
            spmcs) INDIR=/nas-files/yuhanchen/tmp/spmcs_in ;;
            videolq) INDIR=/nas-files/yuhanchen/tmp/videolq_in ;;
        esac
        if [ ! -d "$INDIR" ] || [ $(ls "$INDIR"/*.mp4 2>/dev/null | wc -l) -eq 0 ]; then
            echo "SKIP $dataset $config: no mp4 files in $INDIR" | tee -a "$MASTER_LOG"
            continue
        fi
        run_one "$config" "$dataset"
    done
done

echo "" | tee -a "$MASTER_LOG"
echo "==== Master Eval Done: $(date) ====" | tee -a "$MASTER_LOG"

# Summary of all metrics
echo "" | tee -a "$MASTER_LOG"
echo "==== Metrics Summary ====" | tee -a "$MASTER_LOG"
for csv in runs/spmcs_metrics_*.csv runs/videolq_metrics_*.csv 2>/dev/null; do
    [ -f "$csv" ] || continue
    echo "--- $csv ---" | tee -a "$MASTER_LOG"
    tail -2 "$csv" | tee -a "$MASTER_LOG"
done
