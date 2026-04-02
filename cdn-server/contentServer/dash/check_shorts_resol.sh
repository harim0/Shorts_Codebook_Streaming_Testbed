#!/bin/bash
ROOT="/home/harim/harim_SRSHORTS/cdn-server/contentServer/dash/data"

echo "=== Checking video resolutions under: $ROOT ==="
echo

for dir in "$ROOT"/*; do
  [ -d "$dir" ] || continue
  echo "📁 Directory: $(basename "$dir")"
  
  for resdir in "$dir"/*p; do
    [ -d "$resdir" ] || continue
    resname=$(basename "$resdir")
    expected_width=$(echo "$resname" | grep -o '[0-9]\+')
    
    mp4=$(find "$resdir" -maxdepth 1 -type f -name "output_*.mp4"  | head -n 1)
    [ -z "$mp4" ] && continue

    IFS="x" read width height <<< $(ffprobe -v error -select_streams v:0 \
      -show_entries stream=width,height -of csv=p=0:s=x "$mp4")

    if [ "$width" = "$expected_width" ]; then
      echo "  ✅ $resname → ${width}x${height} (OK)"
    else
      echo "  ⚠️  $resname → ${width}x${height} (MISMATCH!) [$mp4]"
    fi
  done
  echo
done
