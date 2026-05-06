#!/usr/bin/env bash
# Convert VideoLQ PNG-dir clips -> .mp4 for test_DLoRAL.py.
# Usage: bash scripts/videolq_to_mp4.sh
set -euo pipefail
SRC=${SRC:-/nas-files/yuhanchen/CV/data/test_sets/videolq/VideoLQ/VideoLQ}
DST=${DST:-/nas-files/yuhanchen/tmp/videolq_in}
FPS=${FPS:-10}
mkdir -p "$DST"
for clip in "$SRC"/*/; do
    name=$(basename "$clip")
    out="$DST/${name}.mp4"
    [ -f "$out" ] && { echo "skip $name"; continue; }
    ffmpeg -y -loglevel error -framerate "$FPS" -i "$clip/%08d.png" \
        -c:v libx264 -pix_fmt yuv420p -crf 12 "$out"
    echo "$name -> $out"
done
