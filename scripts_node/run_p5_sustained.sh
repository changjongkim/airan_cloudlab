#!/usr/bin/env bash
# P5: Sustained 5-min L1+AI co-execution.
# Goal: show steady-state effect (no thermal/transient artifact).
# Method: L1 runs ITERS=10000 (~5min) while AI runs in background.
# Also dmon GPU clock/temp during the window.

set -uo pipefail
cd "$HOME/cloudlab_aerial"

GPU="${GPU:-0}"
N="${N:-2}"
ITERS_L1="${ITERS_L1:-7500}"           # ~5 minutes at ~40ms/iter
CELLS_L1="${CELLS_L1:-20}"
AI_DURATION="${AI_DURATION:-330}"       # AI runs 5.5 minutes
DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
IMAGE="airan:25-3-final"
AERIAL_SDK="/mydata/aerial-cuda-accelerated-ran"
REPO_DIR="$HOME/AIRAN_Changjong"
SCRIPT_DIR="$HOME/cloudlab_aerial"
HF_CACHE="/mydata/hf_cache"
HOST_UID=$(id -u); HOST_GID=$(id -g)

OUT_ROOT="$HOME/cloudlab_aerial/results/$DATE_DIR/p5_sustained"
mkdir -p "$OUT_ROOT" && chmod 777 "$OUT_ROOT"

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

reconfigure_mig() {
  sudo nvidia-smi mig -i "$GPU" -dci >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i "$GPU" -dgi >/dev/null 2>&1 || true
  sudo nvidia-smi mig -i "$GPU" -cgi "$1" -C >/dev/null 2>&1
  sleep 2
}
uuid_for_profile() {
  nvidia-smi -L | grep -E "MIG[[:space:]]+$1[[:space:]]" | grep -oE "MIG-[a-f0-9-]{36}" | head -1
}
uuid_for_profile_nth() {
  nvidia-smi -L | grep -E "MIG[[:space:]]+$1[[:space:]]" | grep -oE "MIG-[a-f0-9-]{36}" | sed -n "${2}p"
}

start_ai_bg() {
  local ai_uuid=$1 wl=$2
  local script_args
  case "$wl" in
    qwen_small)  script_args="/workspace/AIRAN_Changjong/experiments/run_qwen_small_stress.py 0 $AI_DURATION" ;;
    sat_compute) script_args="/workspace/AIRAN_Changjong/experiments/run_partition_saturated.py 0 $AI_DURATION 8.5 8192" ;;
    chanpred)    script_args="/workspace/AIRAN_Changjong/experiments/run_channel_prediction.py 0 $AI_DURATION" ;;
    neuralrx)    script_args="/workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py 0 $AI_DURATION" ;;
    xapp)        script_args="/workspace/AIRAN_Changjong/experiments/run_xapp_anomaly.py 0 $AI_DURATION" ;;
    resnet)      script_args="/workspace/AIRAN_Changjong/experiments/run_resnet_stress.py 0 $AI_DURATION 64 fp16" ;;
    forecaster)  script_args="/workspace/AIRAN_Changjong/experiments/run_traffic_forecaster.py 0 $AI_DURATION 64 384" ;;
    sat_hbm)     script_args="/workspace/AIRAN_Changjong/experiments/run_hbm_saturated.py 0 $AI_DURATION 8.5" ;;
  esac
  docker run -d --rm --gpus "\"device=$ai_uuid\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
    -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
    -v "$SCRIPT_DIR:/scripts" \
    -v "$HF_CACHE:/mnt/dockerdata/hf_cache" \
    -e HF_HOME="/mnt/dockerdata/hf_cache" \
    "$IMAGE" bash -c "python3 $script_args > /tmp/ai.log 2>&1"
}

run_l1_long() {
  local l1_uuid=$1 outfile=$2 tag=$3
  docker run --rm --user "$HOST_UID:$HOST_GID" --gpus "\"device=$l1_uuid\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" -v "$SCRIPT_DIR:/scripts" \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
    -w /scripts "$IMAGE" python3 real_l1.py "$tag" "$CELLS_L1" "$ITERS_L1" > "$outfile" 2>&1
}

start_dmon() {
  local out=$1
  nvidia-smi dmon -s pucvm -i "$GPU" -o T -d 2 > "$out" 2>&1 &
  echo $!
}

reconfigure_mig "9,9"
l1_uuid=$(uuid_for_profile "3g.20gb")
ai_uuid=$(uuid_for_profile_nth "3g.20gb" 2)
log "L1=$l1_uuid  AI=$ai_uuid (3g+3g layout)"

for scenario in alone qwen_small sat_compute chanpred neuralrx xapp resnet forecaster sat_hbm; do
  log "===== Scenario: $scenario (sustained 5 min) ====="
  outdir="$OUT_ROOT/$scenario"
  mkdir -p "$outdir" && chmod 777 "$outdir"
  for i in $(seq 1 $N); do
    log "  run $i/$N (~5 min)"
    dmon_pid=$(start_dmon "$outdir/run_${i}_dmon.csv")
    if [[ "$scenario" == "alone" ]]; then
      run_l1_long "$l1_uuid" "$outdir/run_${i}_l1.log" "P5_${scenario}_run${i}"
    else
      cid=$(start_ai_bg "$ai_uuid" "$scenario")
      sleep 10
      run_l1_long "$l1_uuid" "$outdir/run_${i}_l1.log" "P5_${scenario}_run${i}"
      docker kill "$cid" >/dev/null 2>&1 || true
      docker rm -f "$cid" >/dev/null 2>&1 || true
    fi
    kill "$dmon_pid" 2>/dev/null || true
    grep "mean=" "$outdir/run_${i}_l1.log" | tail -1
    sleep 5
  done
done

log "p5 sustained DONE"
