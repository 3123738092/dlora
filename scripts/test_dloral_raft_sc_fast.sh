#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=1
PY=${PY:-/nas-files/yuhanchen/miniconda3/envs/lora/bin/python}
CKPT=${CKPT:-preset/models/checkpoints/model.pkl}
SC_CKPT=${SC_CKPT:-runs/sidechannel_v2_lastframe/sidechannel_step005000.pt}
INPUT=${INPUT:-/nas-files/yuhanchen/tmp/udm10_in}
GT=${GT:-/nas-files/yuhanchen/CV/data/test_sets/UDM10/GT}
OUT=${OUT:-/nas-files/yuhanchen/tmp/udm10_raft_sc_out}

echo "==== RAFT+SC on GPU 1 ===="
mkdir -p "$OUT"
for clip in "$INPUT"/*.mp4; do
    name=$(basename "$clip" .mp4)
    echo "--- $name ---"
    "$PY" src/test_DLoRAL.py \
        --input_image "$clip" \
        --output_dir "$OUT" \
        --pretrained_path "$CKPT" \
        --load_cfr \
        --flow_estimator raft \
        --sidechannel_ckpt "$SC_CKPT" \
        --vae_encoder_tiled_size 2048 \
        --vae_decoder_tiled_size 448
done
echo "==== Metrics ===="
"$PY" tools/eval_metrics.py --pred_dir "$OUT" --gt_dir "$GT" --out_csv runs/metrics_udm10_raft_sc.csv
echo "==== Done ===="
cat runs/metrics_udm10_raft_sc.csv
