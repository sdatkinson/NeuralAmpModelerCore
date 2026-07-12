#!/bin/bash
# Run NeuralAmpModelerCore benchmarks and save a timestamped report.
# Usage: ./benchmark_reports/run_benchmarks.sh
# Prerequisite: build bench tools first:
#   cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . --target benchmodel benchmodel_bufsize bench_a2_fast -j$(sysctl -n hw.ncpu)

set -euo pipefail
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:${PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

BENCH=./build/tools/benchmodel
BENCH_BUF=./build/tools/benchmodel_bufsize
BENCH_A2=./build/tools/bench_a2_fast
NUM_RUNS=10
REPORT="$SCRIPT_DIR/benchmark_report_$(date +%Y%m%d_%H%M%S).txt"

for exe in "$BENCH" "$BENCH_BUF" "$BENCH_A2"; do
  if [[ ! -x "$exe" ]]; then
    echo "Missing $exe — build first (see script header)." >&2
    exit 1
  fi
done

# Generate A2-shaped test models if missing
if [[ ! -f "$SCRIPT_DIR/tmp_models/a2_nano.nam" ]]; then
  python3 "$SCRIPT_DIR/generate_a2_models.py"
fi

calc_stats() {
  local f="$1" n min max mean median
  n=$(wc -l < "$f" | tr -d ' ')
  min=$(sort -n "$f" | head -1)
  max=$(sort -n "$f" | tail -1)
  mean=$(awk '{s+=$1} END{printf "%.3f", s/NR}' "$f")
  if [ $((n % 2)) -eq 0 ]; then
    local mid1=$((n / 2)) mid2=$((n / 2 + 1))
    local v1 v2
    v1=$(sort -n "$f" | sed -n "${mid1}p")
    v2=$(sort -n "$f" | sed -n "${mid2}p")
    median=$(echo "scale=3; ($v1 + $v2) / 2" | bc)
  else
    local mid=$((n / 2 + 1))
    median=$(sort -n "$f" | sed -n "${mid}p")
  fi
  echo "$median $min $max $mean"
}

extract_ms() {
  echo "$1" | grep -E "^[0-9]+\.[0-9]+ms$" | head -1 | sed 's/ms$//'
}

{
echo "================================================================"
echo "NeuralAmpModelerCore Benchmark Report"
echo "================================================================"
echo ""
echo "Date:       $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Machine:    $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
echo "Arch:       $(uname -m)"
echo "OS:         $(sw_vers -productName 2>/dev/null) $(sw_vers -productVersion 2>/dev/null)"
echo "CPU cores:  $(sysctl -n hw.ncpu 2>/dev/null)"
echo "RAM:        $(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.1f GB", $1/1073741824}')"
echo ""
echo "Build: Release, A2 fast ON"
echo "Git:   $(git rev-parse --short HEAD) ($(git describe --tags --always 2>/dev/null))"
echo ""

echo "================================================================"
echo "1. benchmodel — example models ($NUM_RUNS runs)"
echo "================================================================"
printf "%-35s %10s %10s %10s %10s %8s\n" "Model" "Median" "Min" "Max" "Mean" "RTF"
echo "--------------------------------------------------------------------------------"
for model in example_models/*.nam; do
  name=$(basename "$model")
  tmp=$(mktemp)
  for _ in $(seq 1 "$NUM_RUNS"); do
    out=$("$BENCH" "$model" 2>&1)
    extract_ms "$out" >> "$tmp"
  done
  read -r med min max mean <<< "$(calc_stats "$tmp")"
  rtf=$(echo "scale=1; 2000 / $med" | bc)
  printf "%-35s %9.3fms %9.3fms %9.3fms %9.3fms %7sx\n" "$name" "$med" "$min" "$max" "$mean" "$rtf"
  rm -f "$tmp"
done

echo ""
echo "================================================================"
echo "2. bench_a2_fast — A2 fast vs generic"
echo "================================================================"
"$BENCH_A2" "$SCRIPT_DIR/tmp_models/a2_nano.nam" "$SCRIPT_DIR/tmp_models/a2_standard.nam" 2>&1

echo ""
echo "================================================================"
echo "3. benchmodel_bufsize — buffer size sweep"
echo "================================================================"
for model_label in \
  "wavenet_a1_standard.nam:example_models/wavenet_a1_standard.nam" \
  "wavenet.nam:example_models/wavenet.nam" \
  "lstm.nam:example_models/lstm.nam" \
  "a2_standard.nam:$SCRIPT_DIR/tmp_models/a2_standard.nam"; do
  label="${model_label%%:*}"
  path="${model_label#*:}"
  echo ""
  echo "--- $label ---"
  printf "%-12s %15s %12s\n" "Buffer" "Avg Time (us)" "RTF"
  echo "----------------------------------------------"
  for bufsize in 16 32 64 128 256 512 1024; do
    result=$("$BENCH_BUF" "$path" "$bufsize" 5 2>&1)
    us=$(echo "$result" | grep -E "^[0-9]+,[0-9.]+$" | cut -d, -f2)
    rtf=$(echo "scale=1; 2000000 / $us" | bc)
    printf "%-12s %15.1f %11sx\n" "$bufsize" "$us" "$rtf"
  done
done

echo ""
echo "================================================================"
echo "Report complete."
echo "================================================================"
} > "$REPORT"

echo "Report saved to: $REPORT"
