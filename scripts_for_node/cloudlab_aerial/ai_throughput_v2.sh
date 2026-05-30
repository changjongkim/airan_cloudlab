#!/usr/bin/env bash
# AI throughput v2 — VALID measurement with persistent L1 background.
#
# 5/24 issue: L1 background ran ~10s while AI ran 30s, so "with_l1" was
# essentially "alone". This version runs L1 in continuous loop via
# real_l1_loop.sh until AI workload finishes.
#
# Outputs:
#   results/$DATE_DIR/ai_throughput_v2/<ai_name>_<setup>/run_<i>.log
#   results/$DATE_DIR/ai_throughput_v2/<ai_name>_<setup>/throughput.txt

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

OUT_ROOT="$HOME/cloudlab_aerial/results/$DATE_DIR/ai_throughput_v2"
mkdir -p "$OUT_ROOT" && chmod 777 "$OUT_ROOT"

# AI workloads (small enough for 2g.10gb partition)
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

# Get MIG UUIDs
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

if [[ -z "$L1_UUID" ]] || [[ -z "$AI_UUID" ]]; then
  log "ERROR: MIG setup failed"
  exit 1
fi

# ----------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------
start_persistent_l1() {
  # Returns container ID. Caller responsible for docker kill.
  local cid=$(docker run -d --rm --gpus "\"device=$L1_UUID\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
    -v "$SCRIPT_DIR:/scripts" \
    --name "l1_bg_$$_$(date +%s)" \
    "$IMAGE" bash /scripts/real_l1_loop.sh bg_l1 20 50)
  echo "$cid"
}

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

# ----------------------------------------------------------------------------
# Run each AI workload: alone, then with_l1 (persistent L1)
# ----------------------------------------------------------------------------
for ai_name in qwen_small chanpred xapp neuralrx; do
  log "=== $ai_name ALONE on 2g.10gb (N=$N, persistent L1=none) ==="
  for i in $(seq 1 $N); do
    run_ai "$ai_name" "alone" $i
    grep -E "done:|it/s|inf/s|pred/s|inferences in" "$OUT_ROOT/${ai_name}_alone/run_${i}.log" | tail -3 \
      | tee -a "$OUT_ROOT/${ai_name}_alone/throughput.txt"
  done

  log "=== $ai_name WITH PERSISTENT L1 (3g L1 loop + AI on 2g) N=$N ==="
  for i in $(seq 1 $N); do
    # Start L1 in persistent loop
    L1_CID=$(start_persistent_l1)
    log "  L1 loop started: $L1_CID"
    sleep 8  # let L1 reach steady state

    # Run AI for full duration (L1 keeps looping in background)
    run_ai "$ai_name" "with_l1" $i
    grep -E "done:|it/s|inf/s|pred/s|inferences in" "$OUT_ROOT/${ai_name}_with_l1/run_${i}.log" | tail -3 \
      | tee -a "$OUT_ROOT/${ai_name}_with_l1/throughput.txt"

    # Kill L1 loop
    docker kill "$L1_CID" >/dev/null 2>&1 || true
    docker rm -f "$L1_CID" >/dev/null 2>&1 || true
    sleep 3
  done
done

log "DONE — AI throughput v2 measurement"
log "Results: $OUT_ROOT"
ls -la "$OUT_ROOT"
