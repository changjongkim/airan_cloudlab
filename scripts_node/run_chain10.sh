#!/usr/bin/env bash
# Chain 10: non-CUDA-API approaches
#
# Approaches:
#   A. baseL1      — real_l1.py (control reference)
#   B. graphL1     — real_l1_graph.py (CUDA Graph capture; eager fallback if pyaerial D2H prevents capture)
#   C. mpsP{PCT}   — real_l1.py under MPS with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={PCT}
#                    Soft SM partition inside MPS ("green-ctx-like" without dedicated Green Context API)
#
# Modes matrix:
#   baseL1  × {TS, MPS}      × {alone, +NRx} × 3
#   graphL1 × {TS, MPS}      × {alone, +NRx} × 3
#   mpsP30  × MPS100 baseline vs {30, 50, 70}% × {alone, +NRx} × 3

set -uo pipefail

GPU=0
CELLS=${CELLS:-20}
ITERS=${ITERS:-100}
N_TRIALS=${N_TRIALS:-3}
DATE_DIR=${DATE_DIR:-$(date +%Y%m%d)}
IMAGE=airan:25-3-final
SDK=/mydata/aerial-cuda-accelerated-ran
REPO=/users/sgkim/AIRAN_Changjong
SCRIPT=/users/sgkim/cloudlab_aerial
OUT=/mydata/results/$DATE_DIR/chain10
mkdir -p "$OUT" && chmod 777 "$OUT"

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$OUT/run.log"; }

TARGET_UUID=""
MPS_ENVS=""

set_ts(){
  docker rm -f mps_srv 2>/dev/null || true
  sudo -n nvidia-smi -i $GPU -c DEFAULT >/dev/null 2>&1
  TARGET_UUID=$(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader)
  MPS_ENVS=""
}

set_mps(){ local thread_pct=${1:-100}
  sudo -n nvidia-smi -i $GPU -c EXCLUSIVE_PROCESS >/dev/null 2>&1
  sudo rm -rf /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
  sudo mkdir -p /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
  sudo chmod 777 /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
  docker rm -f mps_srv 2>/dev/null || true
  docker run -d --gpus "\"device=$GPU\"" --ipc=host --pid=host --user 0:0 \
    -v /tmp:/tmp \
    -e CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$GPU \
    -e CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$GPU \
    -e CUDA_VISIBLE_DEVICES=0 \
    --name mps_srv "$IMAGE" \
    bash -c "nvidia-cuda-mps-control -d && sleep infinity"
  sleep 5
  TARGET_UUID=$(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader)
  MPS_ENVS="-e CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$GPU -e CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$GPU -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$thread_pct"
}

cleanup_mode(){
  docker ps --filter 'name=ai_' -q | xargs -r docker stop -t 3 2>/dev/null
  docker rm -f mps_srv 2>/dev/null || true
  sudo -n nvidia-smi -i $GPU -c DEFAULT >/dev/null 2>&1
  sleep 2
}

run_nrx_bg(){ local tag=$1
  docker run -d --rm --init --user 0:0 --gpus "\"device=$TARGET_UUID\"" \
    --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
    -v "$SDK:/opt/nvidia/cuBB" \
    -v "$REPO:/workspace/AIRAN_Changjong" \
    -v "$OUT:/aiout" \
    -e cuBB_SDK=/opt/nvidia/cuBB \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
    --name "ai_${tag}_$(date +%s)" \
    "$IMAGE" \
    bash -c "python3 /workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py 0 120 > /aiout/${tag}_ai.log 2>&1" >/dev/null
}
kill_nrx(){ docker ps --filter 'name=ai_' -q | xargs -r docker stop -t 3 2>/dev/null; sleep 1; }

profile_l1(){ local label=$1 script=$2
  docker run --rm --user 0:0 --gpus "\"device=$TARGET_UUID\"" \
    --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
    -v "$SDK:/opt/nvidia/cuBB" -v "$SCRIPT:/scripts" -v "$OUT:/out" \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
    -e RESULTS_DIR=/out \
    -w /scripts "$IMAGE" \
    bash -c "nsys profile --trace=cuda --duration=30 --output=/out/${label} --force-overwrite=true --stats=false python3 $script ${label} $CELLS $ITERS" >/dev/null 2>&1
}

# =============================================================================
# Approach A: baseline L1 × {TS, MPS100}
# =============================================================================
run_baseL1(){
  for MODE_NAME in TS MPS; do
    if [[ "$MODE_NAME" == "TS" ]]; then set_ts; fi
    if [[ "$MODE_NAME" == "MPS" ]]; then set_mps 100; fi

    log "--- baseL1_${MODE_NAME}_alone ---"
    for t in $(seq 1 $N_TRIALS); do
      profile_l1 "baseL1_${MODE_NAME}_alone_t${t}" real_l1.py
    done

    log "--- baseL1_${MODE_NAME}_nrx ---"
    for t in $(seq 1 $N_TRIALS); do
      run_nrx_bg "baseL1_${MODE_NAME}_nrx_t${t}"
      sleep 15
      profile_l1 "baseL1_${MODE_NAME}_nrx_t${t}" real_l1.py
      kill_nrx
    done
    cleanup_mode
  done
}

# =============================================================================
# Approach B: CUDA Graph L1 × {TS, MPS100}
# =============================================================================
run_graphL1(){
  for MODE_NAME in TS MPS; do
    if [[ "$MODE_NAME" == "TS" ]]; then set_ts; fi
    if [[ "$MODE_NAME" == "MPS" ]]; then set_mps 100; fi

    log "--- graphL1_${MODE_NAME}_alone ---"
    for t in $(seq 1 $N_TRIALS); do
      profile_l1 "graphL1_${MODE_NAME}_alone_t${t}" real_l1_graph.py
    done

    log "--- graphL1_${MODE_NAME}_nrx ---"
    for t in $(seq 1 $N_TRIALS); do
      run_nrx_bg "graphL1_${MODE_NAME}_nrx_t${t}"
      sleep 15
      profile_l1 "graphL1_${MODE_NAME}_nrx_t${t}" real_l1_graph.py
      kill_nrx
    done
    cleanup_mode
  done
}

# =============================================================================
# Approach C: MPS thread percentage sweep (30, 50, 70)
# =============================================================================
run_mps_pct(){
  for PCT in 30 50 70; do
    set_mps $PCT

    log "--- mpsP${PCT}_alone ---"
    for t in $(seq 1 $N_TRIALS); do
      profile_l1 "mpsP${PCT}_alone_t${t}" real_l1.py
    done

    log "--- mpsP${PCT}_nrx ---"
    for t in $(seq 1 $N_TRIALS); do
      run_nrx_bg "mpsP${PCT}_nrx_t${t}"
      sleep 15
      profile_l1 "mpsP${PCT}_nrx_t${t}" real_l1.py
      kill_nrx
    done
    cleanup_mode
  done
}

# =============================================================================
# Main
# =============================================================================
log "=== START chain10 ==="
log "Approach A: baseL1 × TS/MPS × alone/nrx × 3"
run_baseL1
log "Approach B: graphL1 × TS/MPS × alone/nrx × 3"
run_graphL1
log "Approach C: MPS thread% sweep (30, 50, 70) × alone/nrx × 3"
run_mps_pct

log "=== chain10 DONE ==="
ls "$OUT" | wc -l
