#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

cmake -S profilers/activity_profiler -B profilers/activity_profiler/build
cmake --build profilers/activity_profiler/build -j

export BEV_ACTIVITY_CSV="${BEV_ACTIVITY_CSV:-$ROOT/reports/activity_timeline.csv}"
export BEV_ACTIVITY_JSON="${BEV_ACTIVITY_JSON:-$ROOT/reports/activity_timeline.json}"
export BEV_ACTIVITY_LIB="$ROOT/profilers/activity_profiler/build/libbevformer_activity_profiler.so"
export LD_PRELOAD="$BEV_ACTIVITY_LIB${LD_PRELOAD:+:$LD_PRELOAD}"

python3 integrations/run_instrumented_bevformer.py \
  --samples "${SAMPLES:-1}" \
  --workers-per-gpu 0 \
  --skip-msda-capture
