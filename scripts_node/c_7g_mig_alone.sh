#!/usr/bin/env bash
# C: 7g.40gb MIG single instance, L1 alone. Measures MIG mode overhead vs no-MIG.
# Uses GPU 1 with MIG enabled + single 7g.40gb GI/CI.

set -uo pipefail
cd "$HOME/cloudlab_aerial"

N="${N:-10}"
CELLS="${CELLS:-20}"
ITERS="${ITERS:-50}"
MIG_UUID="${MIG_UUID:-MIG-8c5da10b-df9f-5dd8-99f6-6bc1c2241b4d}"
DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
TAG="${TAG:-L1_alone_7g40gb_MIG}"
IMAGE="${IMAGE:-airan:25-3-final}"
AERIAL_SDK="/mydata/aerial-cuda-accelerated-ran"

OUT_DIR="$HOME/cloudlab_aerial/results/$DATE_DIR/n${N}_$TAG"
mkdir -p "$OUT_DIR/results"
chmod 777 "$OUT_DIR" "$OUT_DIR/results"

echo "=========================================="
echo "C: 7g.40gb MIG single L1 alone (no AI)"
echo "MIG UUID: $MIG_UUID"
echo "N=$N CELLS=$CELLS ITERS=$ITERS"
echo "Output: $OUT_DIR"
echo "=========================================="

for i in $(seq 1 "$N"); do
  echo "------ run $i/$N : $(date +%H:%M:%S) ------"
  RES=$OUT_DIR/run_${i}_tmp
  mkdir -p "$RES" && chmod 777 "$RES"
  docker run --rm --gpus "\"device=$MIG_UUID\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
    -v "$HOME/cloudlab_aerial:/scripts" \
    -v "$RES:/results_out" \
    -e RESULTS_DIR=/results_out \
    "$IMAGE" python3 /scripts/real_l1.py "7g_mig_$i" "$CELLS" "$ITERS" \
    > "$OUT_DIR/run_${i}.log" 2>&1 || echo "  run $i FAILED"
  latest=$(ls -t "$RES"/*.json 2>/dev/null | head -1)
  if [[ -f "$latest" ]]; then
    cp "$latest" "$OUT_DIR/run_${i}.json"
    rm -rf "$RES"
    mean=$(python3 -c "import json; d=json.load(open('$latest')); print(f\"{d.get('mean_ms',0):.3f}\")")
    p99=$(python3 -c "import json; d=json.load(open('$latest')); print(f\"{d.get('p99_ms',0):.3f}\")")
    printf "  run %2d  mean=%s ms  p99=%s ms\n" "$i" "$mean" "$p99" | tee -a "$OUT_DIR/summary.txt"
  else
    echo "  run $i NO JSON" | tee -a "$OUT_DIR/summary.txt"
  fi
  sleep 2
done

echo ""
echo "DONE — N=$N on 7g.40gb MIG"
