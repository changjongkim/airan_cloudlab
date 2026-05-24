#!/usr/bin/env bash
# AI workload throughput measurement — alone vs co-located with L1.
#
# Outputs:
#   results/$DATE_DIR/ai_throughput/<ai_name>_<setup>/run_<i>.log
#   results/$DATE_DIR/ai_throughput/<ai_name>_<setup>/throughput.csv
#
# For each AI workload:
#   - Run alone on 2g.10gb partition (no L1)
#   - Run with L1 on neighbor 3g.20gb (split-60-40)
#   - Extract throughput from stdout (regex "X iters" or "Y inf/s")

set -uo pipefail
cd "$HOME/cloudlab_aerial"

GPU="${GPU:-0}"
N="${N:-5}"
DURATION="${DURATION:-30}"
DATE_DIR="${DATE_DIR:-$(date +%Y%m%d)}"
IMAGE="airan:25-3-final"
AERIAL_SDK="/mydata/aerial-cuda-accelerated-ran"
REPO_DIR="$HOME/AIRAN_Changjong"
SCRIPT_DIR="$HOME/cloudlab_aerial"
HF_CACHE="/mydata/hf_cache"

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

OUT_ROOT="$HOME/cloudlab_aerial/results/$DATE_DIR/ai_throughput"
mkdir -p "$OUT_ROOT" && chmod 777 "$OUT_ROOT"

# AI workloads to test (script + args + name in container output)
declare -A AI_SCRIPT
AI_SCRIPT[qwen_small]="/workspace/AIRAN_Changjong/experiments/run_qwen_small_stress.py 0 $DURATION"
AI_SCRIPT[chanpred]="/workspace/AIRAN_Changjong/experiments/run_channel_prediction.py 0 $DURATION"
AI_SCRIPT[xapp]="/workspace/AIRAN_Changjong/experiments/run_xapp_anomaly.py 0 $DURATION"
AI_SCRIPT[neuralrx]="/workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py 0 $DURATION"

# ----------------------------------------------------------------------------
# Setup MIG: split-60-40 on GPU $GPU (3g + 2g)
# ----------------------------------------------------------------------------
log "Reconfigure GPU $GPU to split-60-40 (3g + 2g)"
sudo nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
sudo nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
sudo nvidia-smi mig -i $GPU -cgi 9,14 -C >/dev/null 2>&1
sleep 2

# Get MIG UUIDs for GPU $GPU only (parse output of nvidia-smi -L)
mapfile -t MIG_UUIDS < <(nvidia-smi -L | awk -v g=$GPU 'BEGIN{f=0} /^GPU /{ if(match($0, "^GPU "g":")) f=1; else f=0 } f && /MIG/ { match($0, /MIG-[0-9a-f-]+/); print substr($0, RSTART, RLENGTH) }')
mapfile -t MIG_SIZES < <(nvidia-smi -L | awk -v g=$GPU 'BEGIN{f=0} /^GPU /{ if(match($0, "^GPU "g":")) f=1; else f=0 } f && /MIG/ { print $2 }')

L1_UUID=""
AI_UUID=""
for i in "${!MIG_SIZES[@]}"; do
  if [[ "${MIG_SIZES[$i]}" == "3g.20gb" ]] && [[ -z "$L1_UUID" ]]; then
    L1_UUID="${MIG_UUIDS[$i]}"
  fi
  if [[ "${MIG_SIZES[$i]}" == "2g.10gb" ]] && [[ -z "$AI_UUID" ]]; then
    AI_UUID="${MIG_UUIDS[$i]}"
  fi
done
log "L1 (3g.20gb): $L1_UUID"
log "AI (2g.10gb): $AI_UUID"

# ----------------------------------------------------------------------------
# Function: run AI workload, capture stdout
# ----------------------------------------------------------------------------
run_ai() {
  local ai_name=$1
  local setup=$2  # "alone" or "with_l1"
  local i=$3

  local dir="$OUT_ROOT/${ai_name}_${setup}"
  mkdir -p "$dir" && chmod 777 "$dir"
  local out="$dir/run_${i}.log"

  local script="${AI_SCRIPT[$ai_name]}"

  docker run --rm --gpus "\"device=$AI_UUID\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
    -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
    -v "$SCRIPT_DIR:/scripts" \
    -v "$HF_CACHE:/mnt/dockerdata/hf_cache" \
    -e HF_HOME="/mnt/dockerdata/hf_cache" \
    "$IMAGE" python3 $script > "$out" 2>&1
}

run_l1() {
  # Run L1 in background on $L1_UUID, capture pid
  local dir=$1
  local i=$2
  docker run -d --gpus "\"device=$L1_UUID\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
    -v "$SCRIPT_DIR:/scripts" \
    -v "$dir:/results_out" \
    -e RESULTS_DIR=/results_out \
    --name "l1_bg_${i}_$$" \
    "$IMAGE" python3 /scripts/real_l1.py "l1_bg" 20 50 > "$dir/l1_${i}.log" 2>&1
  echo "l1_bg_${i}_$$"
}

# ----------------------------------------------------------------------------
# Run each AI workload: alone, then with_l1
# ----------------------------------------------------------------------------
for ai_name in qwen_small chanpred xapp neuralrx; do
  log "=== $ai_name ALONE on 2g.10gb (N=$N) ==="
  for i in $(seq 1 $N); do
    run_ai "$ai_name" "alone" $i
    # Extract throughput line
    tail -2 "$OUT_ROOT/${ai_name}_alone/run_${i}.log" | grep "done:" | tee -a "$OUT_ROOT/${ai_name}_alone/throughput.txt"
  done

  log "=== $ai_name WITH L1 (3g L1 + AI on 2g) N=$N ==="
  for i in $(seq 1 $N); do
    # Start L1 in background
    dir_l1="$OUT_ROOT/${ai_name}_with_l1"
    mkdir -p "$dir_l1" && chmod 777 "$dir_l1"
    cid=$(run_l1 "$dir_l1" $i)
    sleep 2  # let L1 warm up

    # Run AI
    run_ai "$ai_name" "with_l1" $i
    tail -2 "$OUT_ROOT/${ai_name}_with_l1/run_${i}.log" | grep "done:" | tee -a "$OUT_ROOT/${ai_name}_with_l1/throughput.txt"

    # Stop L1 background
    docker kill "$cid" >/dev/null 2>&1 || true
    docker rm -f "$cid" >/dev/null 2>&1 || true
    sleep 2
  done
done

log "DONE — AI throughput measurement"
log "Results: $OUT_ROOT"
ls -la "$OUT_ROOT"
