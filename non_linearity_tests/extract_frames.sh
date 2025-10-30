#!/bin/bash

# Usage: ./extract_frames_interval.sh input.mp4 N INTERVAL
# Example: ./extract_frames_interval.sh input.mp4 5 10
#   → extracts 5 frames, each 10 frames apart (frame 0, 10, 20, 30, 40)

INPUT="$1"
NUM_FRAMES="$2"
INTERVAL="$3"

OUTDIR="frames"
mkdir -p "$OUTDIR"

ffmpeg -i "$INPUT" \
  -vf "select='not(mod(n\,$INTERVAL))',setpts=N/FRAME_RATE/TB" \
  -vsync vfr "$OUTDIR/frame_%03d.png"

# Keep only the first N frames
ls "$OUTDIR"/*.png | sort | head -n "$NUM_FRAMES" | while read f; do
    echo "Keeping $f"
done
ls "$OUTDIR"/*.png | sort | tail -n +$((NUM_FRAMES+1)) | xargs rm -f
