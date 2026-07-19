#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [[ ! -d "$CUDA_HOME/extras/CUPTI" ]]; then
  echo "CUDA_HOME does not contain CUPTI: $CUDA_HOME" >&2
  echo "Set CUDA_HOME to a CUDA Toolkit installation with CUPTI and NVPW." >&2
  exit 1
fi
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

python3 benchmarks/MSDA/profile_msda_nvpw.py \
  --input benchmarks/MSDA/captured_tensors/MSDA_sample.pt \
  --max-passes "${MAX_PASSES:-16}" \
  --build
