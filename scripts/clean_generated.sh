#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

for marker in README.md .gitignore benchmarks integrations profilers scripts; do
  if [[ ! -e "$ROOT/$marker" ]]; then
    echo "Refusing cleanup: project marker is missing: $ROOT/$marker" >&2
    exit 1
  fi
done

clean_generated_directory() {
  local directory="$1"
  local expected="$2"
  local resolved

  resolved="$(realpath "$directory")"
  if [[ "$resolved" != "$expected" ]]; then
    echo "Refusing unexpected generated directory: $resolved" >&2
    exit 1
  fi
  find "$resolved" -mindepth 1 -maxdepth 1 \
    ! -name .gitkeep -exec rm -rf -- {} +
}

clean_generated_directory \
  "$ROOT/reports" \
  "$ROOT/reports"
clean_generated_directory \
  "$ROOT/benchmarks/MSDA/captured_tensors" \
  "$ROOT/benchmarks/MSDA/captured_tensors"

for build_dir in \
  "$ROOT/profilers/activity_profiler/build" \
  "$ROOT/profilers/msda_nvpw_replay/build"; do
  if [[ -d "$build_dir" ]]; then
    resolved="$(realpath "$build_dir")"
    case "$resolved" in
      "$ROOT"/profilers/*/build) rm -rf -- "$resolved" ;;
      *) echo "Refusing unexpected build path: $resolved" >&2; exit 1 ;;
    esac
  fi
done

find "$ROOT" -type d -name __pycache__ -prune -exec rm -rf -- {} +
find "$ROOT" -type f \( -name '*.pyc' -o -name '*.tmp' \) -delete

echo "Generated artifacts removed; inputs and source code were preserved."
