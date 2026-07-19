#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 benchmarks/MSDA/benchmark_msda_latency.py \
  --input benchmarks/MSDA/captured_tensors/MSDA_sample.pt \
  --warmup "${WARMUP:-10}" \
  --repeats "${REPEATS:-50}" \
  --sweep-real-shapes
