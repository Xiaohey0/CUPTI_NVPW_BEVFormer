#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SAMPLES="${SAMPLES:-1}"
CONFIG="${CONFIG:-third_party/BEVFormer/projects/configs/bevformer/bevformer_tiny.py}"
CHECKPOINT="${CHECKPOINT:-checkpoints/bevformer_tiny_epoch_24.pth}"
DATA_ROOT="${DATA_ROOT:-data/nuscenes}"

python3 integrations/run_instrumented_bevformer.py \
  --samples "$SAMPLES" \
  --workers-per-gpu 0 \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --data-root "$DATA_ROOT" \
  --overwrite-capture
