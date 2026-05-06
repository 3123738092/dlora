#!/usr/bin/env bash
set -euo pipefail
PY=/nas-files/yuhanchen/miniconda3/envs/lora/bin/python
CONV=/tmp/convert_png_to_mp4.py

SPMCS_SRC=/nas-files/yuhanchen/CV/data/test_sets/SPMCS/test_set
SPMCS_IN=/nas-files/yuhanchen/tmp/spmcs_in
SPMCS_GT=/nas-files/yuhanchen/CV/data/test_sets/SPMCS/GT
VIDEOLQ_SRC=/nas-files/yuhanchen/CV/data/test_sets/videolq/VideoLQ/VideoLQ
VIDEOLQ_IN=/nas-files/yuhanchen/tmp/videolq_in

echo "==== SPMCS ===="
mkdir -p "$SPMCS_IN" "$SPMCS_GT"
for clip_dir in "$SPMCS_SRC"/*/; do
    name=$(basename "$clip_dir")
    lr_dir="$clip_dir/input2"
    gt_dir="$clip_dir/truth"
    [ -d "$lr_dir" ] && $PY $CONV "$lr_dir" "$SPMCS_IN/${name}.mp4"
    if [ -d "$gt_dir" ]; then
        mkdir -p "$SPMCS_GT/$name"
        cp "$gt_dir"/*.png "$SPMCS_GT/$name/"
    fi
done
echo "SPMCS: $(ls $SPMCS_IN/*.mp4 2>/dev/null | wc -l) mp4s, $(ls $SPMCS_GT/ 2>/dev/null | wc -l) GTs"

echo ""
echo "==== VideoLQ ===="
mkdir -p "$VIDEOLQ_IN"
for clip_dir in "$VIDEOLQ_SRC"/*/; do
    name=$(basename "$clip_dir")
    $PY $CONV "$clip_dir" "$VIDEOLQ_IN/${name}.mp4"
done
echo "VideoLQ: $(ls $VIDEOLQ_IN/*.mp4 2>/dev/null | wc -l) mp4s"
echo "==== Done: $(date) ===="
