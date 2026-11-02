#!/usr/bin/env bash
# Extract the frames used by IPN HandS (640x480, 30 fps, 1-based indices) from the IPN Hand videos.
# Usage: bash extract_frames.sh <videos_dir> <frames_dir>
set -e
for f in "$1"/*.avi; do
  v=$(basename "$f" .avi)
  mkdir -p "$2/$v"
  ffmpeg -v error -i "$f" -vf fps=30 -q:v 2 "$2/$v/${v}_%06d.jpg"
done
