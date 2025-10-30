#!/bin/bash

# Usage: ./video_to_gif.sh input.mp4 output.gif

INPUT="$1"
OUTPUT="$2"
TEMP_DIR="./frames"

if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
  echo "Usage: $0 input.mp4 output.gif"
  exit 1
fi

# Clean up old frames
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

echo "Extracting first 10 seconds of frames..."
ffmpeg -y -t 10 -i "$INPUT" -vf "fps=15,scale=640:-1:flags=lanczos" "$TEMP_DIR/frame_%03d.png"

echo "Creating high-quality palette..."
ffmpeg -y -i "$INPUT" -t 10 -vf "fps=15,scale=640:-1:flags=lanczos,palettegen" "$TEMP_DIR/palette.png"

echo "Generating GIF..."
ffmpeg -y -t 10 -i "$INPUT" -i "$TEMP_DIR/palette.png" \
  -filter_complex "fps=15,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse" "$OUTPUT"

echo "Optimizing GIF (optional)..."
convert "$OUTPUT" -layers Optimize "$OUTPUT"

echo "✅ Done! Saved as $OUTPUT"

