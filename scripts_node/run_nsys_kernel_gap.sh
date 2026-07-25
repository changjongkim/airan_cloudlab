#!/usr/bin/env bash
# Capture L1 kernel timeline via nsys: alone vs with AI co-tenant.
# Goal: prove "inter-kernel gap delay" hypothesis.
# Uses GPU 1 (idle, MIG enabled) to not conflict with running chain on GPU 0.
#
# Outputs:
#   /tmp/nsys_alone.nsys-rep + .sqlite
#   /tmp/nsys_with_qwen.nsys-rep + .sqlite
#   /tmp/nsys_with_neuralrx.nsys-rep + .sqlite

set -uo pipefail
GPU=1
N_CELLS=10
ITERS=15

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

IMAGE="airan:25-3-final"
AERIAL_SDK="/mydata/aerial-cuda-accelerated-ran"
REPO_DIR="$HOME/AIRAN_Changjong"
SCRIPT_DIR="$HOME/cloudlab_aerial"
HF_CACHE="/mydata/hf_cache"
HOST_UID=$(id -u); HOST_GID=$(id -g)

log "Setting up MIG on GPU $GPU (split-50-50: 3g + 3g)"
sudo nvidia-smi mig -i $GPU -dci >/dev/null 2>&1 || true
sudo nvidia-smi mig -i $GPU -dgi >/dev/null 2>&1 || true
sudo nvidia-smi mig -i $GPU -cgi 9,9 -C >/dev/null 2>&1
sleep 2

mapfile -t UUIDS < <(nvidia-smi -L | awk -v g=$GPU 'BEGIN{f=0} /^GPU /{ if(match($0, "^GPU "g":")) f=1; else f=0 } f && /MIG/ { match($0, /MIG-[0-9a-f-]+/); print substr($0, RSTART, RLENGTH) }')
L1_UUID="${UUIDS[0]}"
AI_UUID="${UUIDS[1]}"
log "L1=$L1_UUID  AI=$AI_UUID"

run_l1_nsys() {
  local tag=$1
  log "Running L1 with nsys profile: tag=$tag"
  docker run --rm --user "$HOST_UID:$HOST_GID" --gpus "\"device=$L1_UUID\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" -v "$SCRIPT_DIR:/scripts" -v /tmp:/out \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
    -w /scripts "$IMAGE" \
    bash -c "nsys profile --trace=cuda --output=/out/nsys_$tag --force-overwrite=true --stats=false \
      python3 real_l1.py nsys_$tag $N_CELLS $ITERS" 2>&1 | tail -5
}

start_ai_bg() {
  local script=$1
  docker run -d --rm --gpus "\"device=$AI_UUID\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
    -v "$SCRIPT_DIR:/scripts" -v "$HF_CACHE:/mnt/dockerdata/hf_cache" \
    -e HF_HOME="/mnt/dockerdata/hf_cache" \
    --name "nsys_ai_$(date +%s)_$RANDOM" \
    "$IMAGE" python3 "$script" 0 60
}

# Phase 1: L1 alone
log "===== Phase 1: L1 alone ====="
run_l1_nsys "alone"

# Phase 2: L1 + Qwen
log "===== Phase 2: L1 + Qwen ====="
CID=$(start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_qwen_small_stress.py")
sleep 10
run_l1_nsys "with_qwen"
docker kill "$CID" >/dev/null 2>&1 || true

# Phase 3: L1 + NeuralRx
log "===== Phase 3: L1 + NeuralRx ====="
CID=$(start_ai_bg "/workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py")
sleep 10
run_l1_nsys "with_neuralrx"
docker kill "$CID" >/dev/null 2>&1 || true

# Phase 4: L1 + sat_compute
log "===== Phase 4: L1 + sat_compute ====="
docker run -d --rm --gpus "\"device=$AI_UUID\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" -v "$REPO_DIR:/workspace/AIRAN_Changjong" \
    -v "$SCRIPT_DIR:/scripts" \
    --name "nsys_sat_$(date +%s)" \
    "$IMAGE" python3 "/workspace/AIRAN_Changjong/experiments/run_partition_saturated.py" 0 60 17 8192 >/dev/null 2>&1
SATCID=$(docker ps --filter "name=nsys_sat" --format "{{.ID}}" | head -1)
sleep 10
run_l1_nsys "with_sat"
docker kill "$SATCID" >/dev/null 2>&1 || true

log "DONE — outputs in /tmp/nsys_*.nsys-rep + .sqlite"
ls -la /tmp/nsys_*.sqlite
