#!/usr/bin/env bash
# N=20 repeat wrapper around run_sweep_v2.sh for bimodal statistics.
#
# Runs the same MIG preset + AI workload N times, saving each result with
# an index suffix. Designed for SENSITIVITY_EXPERIMENTS §A1/§A2/§D1.
#
# Usage:
#   N=20 PRESET=split-60-40 AI=qwen7b TAG=bimodal_qwen ./run_n20.sh
#
# Env vars:
#   N        — number of repetitions (default 20)
#   PRESET   — MIG preset (split-50-50, split-60-40, etc.) — REQUIRED
#   AI       — AI workload name (qwen7b, qwen7b_prefill, qwen7b_decode, hbm, ...) — REQUIRED
#   TAG      — identifier for this run-set (default = PRESET_AI)
#   CELLS    — L1 cell count (default 20)
#   DURATION — per-run AI duration in sec (default 30)
#   DMON     — if "1", spawn nvidia-smi dmon during each run (default 0)
#
# Output:
#   results/n20_<TAG>/run_<idx>.json
#   results/n20_<TAG>/summary.txt

set -uo pipefail
cd "$HOME/cloudlab_aerial"

N="${N:-20}"
PRESET="${PRESET:?PRESET required (e.g. split-60-40)}"
AI="${AI:?AI required (e.g. qwen7b)}"
CELLS="${CELLS:-20}"
DURATION="${DURATION:-30}"
TAG="${TAG:-${PRESET}_${AI}}"
DMON="${DMON:-0}"
DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
export DATE_DIR   # propagate into run_sweep_v2.sh so child runs also land under same date

OUT_DIR="$HOME/cloudlab_aerial/results/$DATE_DIR/n${N}_$TAG"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo "N=$N reps of PRESET=$PRESET AI=$AI"
echo "CELLS=$CELLS DURATION=${DURATION}s"
echo "DATE_DIR=$DATE_DIR"
echo "Output: $OUT_DIR"
echo "DMON: $DMON"
echo "=========================================="

if [[ "$DMON" == "1" ]]; then
  ./dmon_sync.sh start "$OUT_DIR"
fi

for i in $(seq 1 "$N"); do
  echo ""
  echo "------ run $i/$N : $(date +%H:%M:%S) ------"
  if [[ "$DMON" == "1" ]]; then
    ./dmon_sync.sh mark "$OUT_DIR" "run_${i}_start"
  fi

  # Invoke run_sweep_v2 with single preset. DATE_DIR is exported so child writes
  # into the SAME date directory (results/$DATE_DIR/<timestamp>/...).
  env AI="$AI" L1_NUM_TX_ANT=8 L1_NUM_RX_ANT=8 CELLS="$CELLS" \
      PRESETS="$PRESET" DURATION="$DURATION" \
      bash ./run_sweep_v2.sh > "$OUT_DIR/run_${i}.log" 2>&1

  # Find the most recent JSON produced by this run (under results/$DATE_DIR/<ts>/<preset>/results/*.json)
  latest=$(find "results/$DATE_DIR" -name "*.json" -newer "$OUT_DIR/run_${i}.log" -print 2>/dev/null | head -1)
  if [[ -z "$latest" ]]; then
    latest=$(find "results/$DATE_DIR" -name "*.json" -print 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1)
  fi
  if [[ -n "$latest" && -f "$latest" ]]; then
    cp "$latest" "$OUT_DIR/run_${i}.json"
    mean=$(python3 -c "import json; d=json.load(open('$latest')); print(f\"{d.get('mean_ms', 0):.3f}\")")
    p99=$(python3 -c "import json; d=json.load(open('$latest')); print(f\"{d.get('p99_ms', 0):.3f}\")")
    printf "  run %2d  mean=%s ms  p99=%s ms\n" "$i" "$mean" "$p99" | tee -a "$OUT_DIR/summary.txt"
  else
    echo "  run $i FAILED (no JSON output)" | tee -a "$OUT_DIR/summary.txt"
  fi

  if [[ "$DMON" == "1" ]]; then
    ./dmon_sync.sh mark "$OUT_DIR" "run_${i}_end"
  fi

  sleep 3   # let GPU settle between runs
done

if [[ "$DMON" == "1" ]]; then
  ./dmon_sync.sh stop "$OUT_DIR"
fi

echo ""
echo "=========================================="
echo "DONE — N=$N reps"
echo "Summary: $OUT_DIR/summary.txt"
echo "Run bimodal_detect.py for cluster analysis:"
echo "  python3 ../cloudlab_results/bimodal_detect.py $OUT_DIR"
echo "=========================================="
