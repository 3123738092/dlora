#!/usr/bin/env bash
# Convert UDM10 PNG sequences -> .mp4 for test_DLoRAL.py
set -euo pipefail
SRC=${SRC:-/nas-files/yuhanchen/CV/data/test_sets/UDM10/BDx4}
DST=${DST:-/nas-files/yuhanchen/tmp/udm10_mp4}
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
echo "done: $DST"
