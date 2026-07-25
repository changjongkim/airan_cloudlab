#!/usr/bin/env bash
# L1 alone on GPU 1 (full GPU, no MIG) — true "A baseline" for partition cap analysis.
# Runs in parallel with Phase 1 on GPU 0.
#
# Each iteration: one docker run of real_l1.py on GPU 1.
# Output: results/$DATE_DIR/n20_baseline_gpu1_fullGPU/run_*.json

set -uo pipefail
cd "$HOME/cloudlab_aerial"

N="${N:-20}"
CELLS="${CELLS:-20}"
ITERS="${ITERS:-50}"
GPU="${GPU:-1}"
DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
TAG="${TAG:-baseline_gpu${GPU}_fullGPU}"
IMAGE="${IMAGE:-airan:25-3-final}"
AERIAL_SDK="${AERIAL_SDK:-/mydata/aerial-cuda-accelerated-ran}"
SCRIPT_DIR="$HOME/cloudlab_aerial"
REPO_DIR="$HOME/AIRAN_Changjong"

OUT_DIR="$HOME/cloudlab_aerial/results/$DATE_DIR/n${N}_$TAG"
mkdir -p "$OUT_DIR/results"

echo "=========================================="
echo "L1 alone on GPU 1 (full GPU, no MIG)"
echo "N=$N CELLS=$CELLS ITERS=$ITERS"
echo "DATE_DIR=$DATE_DIR  Output: $OUT_DIR"
echo "=========================================="

ts() { date '+%H:%M:%S'; }

for i in $(seq 1 "$N"); do
  echo "------ run $i/$N : $(ts) ------"
  RES_FILE_DIR="$OUT_DIR/run_${i}_tmp"
  mkdir -p "$RES_FILE_DIR" && chmod 777 "$RES_FILE_DIR"
  docker run --rm --gpus "\"device=$GPU\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
    -v "$SCRIPT_DIR:/scripts" \
    -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
    -v "$RES_FILE_DIR:/results_out" \
    -e RESULTS_DIR=/results_out \
    "$IMAGE" \
    python3 /scripts/real_l1.py "baseline_run_$i" "$CELLS" "$ITERS" \
    > "$OUT_DIR/run_${i}.log" 2>&1 || echo "  run $i FAILED"

  latest=$(ls -t "$RES_FILE_DIR"/*.json 2>/dev/null | head -1)
  if [[ -f "$latest" ]]; then
    cp "$latest" "$OUT_DIR/run_${i}.json"
    rm -rf "$RES_FILE_DIR"
    mean=$(python3 -c "import json; d=json.load(open('$latest')); print(f\"{d.get('mean_ms',0):.3f}\")")
    p99=$(python3 -c "import json; d=json.load(open('$latest')); print(f\"{d.get('p99_ms',0):.3f}\")")
    printf "  run %2d  mean=%s ms  p99=%s ms\n" "$i" "$mean" "$p99" | tee -a "$OUT_DIR/summary.txt"
  else
    echo "  run $i NO JSON" | tee -a "$OUT_DIR/summary.txt"
  fi
  sleep 2
done

echo ""
echo "=========================================="
echo "DONE — N=$N reps on GPU 1 (full, no MIG)"
echo "Summary: $OUT_DIR/summary.txt"
echo "Run python3 ~/analyze_run.py $OUT_DIR for verdict"
