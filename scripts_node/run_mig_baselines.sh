#!/usr/bin/env bash
# Sequentially measure L1 latency baseline on each MIG partition size.
# For each: reconfigure MIG layout, pick the target UUID, run N reps of real_l1.py
# in isolation (no AI co-tenant).
#
# Output: results/$DATE_DIR/n${N}_baseline_<SIZE>_alone/run_<i>.log
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="airan:25-3-final"
AERIAL_SDK="/mydata/aerial-cuda-accelerated-ran"
N="${N:-20}"
CELLS="${CELLS:-20}"
ITERS="${ITERS:-30}"
GPU="${GPU:-0}"
DATE_DIR="$(date +%Y%m%d)"
HOST_UID=$(id -u)
HOST_GID=$(id -g)

# (size, cgi, target profile, tag)
BASELINES=(
  "7g 0 7g.40gb baseline_7g_single"
  "4g 5,14,19 4g.20gb baseline_4g_alone"
  "3g 9,9 3g.20gb baseline_3g_alone"
  "2g 9,14,14 2g.10gb baseline_2g_alone"
)

reconfigure_mig() {
  local cgi="$1"
  sudo nvidia-smi mig -i "$GPU" -dci >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i "$GPU" -dgi >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i "$GPU" -cgi "$cgi" -C >/dev/null 2>&1
  sleep 2
}

get_uuid_for_profile() {
  local target="$1"
  nvidia-smi -L | grep -E "MIG[[:space:]]+${target}[[:space:]]" \
    | grep -oE "MIG-[a-f0-9-]{36}" | head -1
}

run_one_baseline() {
  local size="$1" cgi="$2" profile="$3" tag="$4"
  echo "============================================================"
  echo "BASELINE: $tag (size=$size cgi=$cgi profile=$profile)"
  echo "============================================================"
  reconfigure_mig "$cgi"
  local uuid; uuid=$(get_uuid_for_profile "$profile")
  if [[ -z "$uuid" ]]; then
    echo "ERROR: $profile UUID not found after cgi=$cgi"
    nvidia-smi -L | head -20
    return 1
  fi
  echo "L1 device UUID: $uuid"
  local outdir="$SCRIPT_DIR/results/$DATE_DIR/n${N}_${tag}"
  mkdir -p "$outdir" && chmod 777 "$outdir"
  for i in $(seq 1 $N); do
    echo "------ $tag run $i/$N : $(date +%H:%M:%S) ------"
    docker run --rm \
      --user "$HOST_UID:$HOST_GID" \
      --gpus "\"device=$uuid\"" \
      -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
      -v "$SCRIPT_DIR:/scripts" \
      -v "$outdir:/results_out" \
      -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
      -e HOME=/tmp \
      -e CUPY_CACHE_DIR=/tmp/cupy_cache \
      -w /scripts \
      "$IMAGE" \
      bash -c "mkdir -p /scripts/results && python3 /scripts/real_l1.py ${tag}_run${i} $CELLS $ITERS" 2>&1 | \
      tee "$outdir/run_${i}.log" | grep -E "mean=|p99=|cells=" | tail -3
  done
  echo "DONE: $tag → $outdir"
}

for line in "${BASELINES[@]}"; do
  read -r size cgi profile tag <<< "$line"
  run_one_baseline "$size" "$cgi" "$profile" "$tag" \
    2>&1 | tee -a /tmp/mig_baselines.log
done

echo "[run_mig_baselines] ALL DONE"
