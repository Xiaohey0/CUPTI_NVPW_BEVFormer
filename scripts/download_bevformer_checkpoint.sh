#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p checkpoints

URL="${URL:-https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth}"
OUT="${OUT:-checkpoints/bevformer_tiny_epoch_24.pth}"

if [ -s "$OUT" ]; then
  echo "Checkpoint already exists: $OUT"
  exit 0
fi

if command -v wget >/dev/null 2>&1; then
  wget -O "$OUT" "$URL"
elif command -v curl >/dev/null 2>&1; then
  curl -L "$URL" -o "$OUT"
else
  echo "Neither wget nor curl is available." >&2
  exit 1
fi

test -s "$OUT"
echo "Downloaded $OUT"
