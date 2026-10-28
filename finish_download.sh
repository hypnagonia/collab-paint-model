#!/bin/bash
# Finish the partial weights file with parallel range requests, verify sha256, then run generation.
set -e
F=~/.cache/hy3dgen/tencent/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv/model.fp16.safetensors
URL="https://huggingface.co/tencent/Hunyuan3D-2mv/resolve/main/hunyuan3d-dit-v2-mv/model.fp16.safetensors"
TOTAL=4928151562
SHA=d36f5881bcdc56726b73e517cd444c13c60732431622da7268145355c8d38e9c
N=8
PARTS=$(dirname "$F")/parts
mkdir -p "$PARTS"

START=$(stat -f %z "$F")
CHUNK=$(( (TOTAL - START + N - 1) / N ))
for i in $(seq 0 $((N - 1))); do
  a=$(( START + i * CHUNK ))
  b=$(( a + CHUNK - 1 )); (( b >= TOTAL )) && b=$(( TOTAL - 1 ))
  (until curl -sSL --retry 10 -r "$a-$b" -o "$PARTS/$i" "$URL" && [ "$(stat -f %z "$PARTS/$i")" -eq $(( b - a + 1 )) ]; do sleep 2; done) &
done
wait

for i in $(seq 0 $((N - 1))); do cat "$PARTS/$i" >> "$F"; done
rm -rf "$PARTS"
echo "size $(stat -f %z "$F") / $TOTAL"
[ "$(shasum -a 256 "$F" | cut -d' ' -f1)" = "$SHA" ] && echo "sha256 ok" || { echo "sha256 MISMATCH"; exit 1; }

cd ~/IdeaProjects/hunyuan3d
PYTHONPATH=repo PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python gen_ranger.py 50 380
