#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CAPTURE="benchmarks/MSDA/captured_tensors/MSDA_sample.pt"
test -s "$CAPTURE"
test -s reports/activity_timeline.csv
test -s reports/range_metrics_msda_nvpw.csv

mkdir -p reports/official_tools

if command -v nsys >/dev/null 2>&1; then
  nsys profile \
    -o reports/official_tools/bevformer_nsys \
    --force-overwrite=true \
    python3 integrations/run_instrumented_bevformer.py \
      --samples "${SAMPLES:-1}" \
      --workers-per-gpu 0 \
      --skip-msda-capture \
    > reports/official_tools/bevformer_nsys.stdout \
    2> reports/official_tools/bevformer_nsys.stderr
else
  echo "nsys is not installed; skipping Nsight Systems comparison." >&2
fi

if command -v ncu >/dev/null 2>&1; then
  ncu \
    --target-processes all \
    --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed.avg.per_cycle_active,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --csv \
    --log-file reports/official_tools/msda_ncu.csv \
    python3 benchmarks/MSDA/benchmark_msda_latency.py \
      --input "$CAPTURE" \
      --warmup 1 \
      --repeats 1 \
    > reports/official_tools/msda_ncu.stdout \
    2> reports/official_tools/msda_ncu.stderr
else
  echo "ncu is not installed; skipping Nsight Compute comparison." >&2
fi

{
  echo "# Official Tool Comparison"
  echo
  echo "- nsys: \`$(command -v nsys || true)\`"
  echo "- ncu: \`$(command -v ncu || true)\`"
  echo
  echo "The custom profiler outputs are the implementation under study."
  echo "Nsight outputs are independent comparison evidence."
  echo
  find reports/official_tools -maxdepth 1 -type f -printf "- %f (%s bytes)\n" | sort
} > reports/official_tools/summary.txt
