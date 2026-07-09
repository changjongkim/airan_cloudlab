#!/usr/bin/env bash
# Simplified MPS vs Time-slicing comparison.
# Assumes: MIG is DISABLED, GPU 0 is the target.
#
# 4 conditions × 3 trials each:
#   TS_alone    time-slicing, L1 only
#   TS_coloc    time-slicing, L1 + NRx concurrent
#   MPS_alone   MPS mode,     L1 only
#   MPS_coloc   MPS mode,     L1 + NRx concurrent
#
# Each L1 run: 30s NSYS window, real_l1.py with 20 cells × 100 iters (matches Chain 6).

set -uo pipefail
cd "$HOME/cloudlab_aerial"

GPU=${GPU:-0}
CELLS=${CELLS:-20}
ITERS=${ITERS:-100}
N_TRIALS=${N_TRIALS:-3}
DATE_DIR=${DATE_DIR:-$(date +%Y%m%d)}
IMAGE=${IMAGE:-airan:25-3-final}
SDK=/mydata/aerial-cuda-accelerated-ran
REPO=/users/sgkim/AIRAN_Changjong
SCRIPT=/users/sgkim/cloudlab_aerial
OUT=/mydata/results/$DATE_DIR/mps_ts_v2
mkdir -p "$OUT" && chmod 777 "$OUT"

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$OUT/run.log"; }

GPU_UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
log "GPU UUID: $GPU_UUID"

# -----------------------------------------------------------------------------
# MPS helpers
# -----------------------------------------------------------------------------
start_mps(){
  log "starting MPS on GPU $GPU"
  sudo nvidia-smi -i "$GPU" -c EXCLUSIVE_PROCESS >/dev/null 2>&1 || true
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$GPU
  export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$GPU
  sudo mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  sudo chmod 777 "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  CUDA_VISIBLE_DEVICES=$GPU nvidia-cuda-mps-control -d
  sleep 3
}
stop_mps(){
  log "stopping MPS"
  echo quit | nvidia-cuda-mps-control 2>/dev/null || true
  sudo nvidia-smi -i "$GPU" -c DEFAULT >/dev/null 2>&1 || true
  sudo rm -rf /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
  sleep 2
}

# -----------------------------------------------------------------------------
# Workload launchers
# -----------------------------------------------------------------------------
start_nrx_bg(){ local tag=$1 mps_flag=$2
  local mps_vols=""
  local mps_envs=""
  if [[ "$mps_flag" == "mps" ]]; then
    mps_vols="-v /tmp/mps_pipe_$GPU:/tmp/nvidia-mps -v /tmp/mps_log_$GPU:/tmp/nvidia-log"
    mps_envs="-e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps -e CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log"
  fi
  docker run -d --rm --init --gpus "\"device=$GPU_UUID\"" \
    --ipc=host $mps_vols $mps_envs \
    -v "$SDK:/opt/nvidia/cuBB" \
    -v "$REPO:/workspace/AIRAN_Changjong" \
    -v "$OUT:/aiout" \
    -e cuBB_SDK=/opt/nvidia/cuBB \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
    --name "nrx_${tag}_$(date +%s)" \
    "$IMAGE" \
    bash -c "python3 /workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py 0 120 > /aiout/${tag}_nrx.log 2>&1" >/dev/null
}
kill_nrx(){ docker ps --filter 'name=nrx_' -q | xargs -r docker stop -t 3 2>/dev/null; sleep 1; }

profile_l1(){ local label=$1 mps_flag=$2
  local mps_vols=""
  local mps_envs=""
  if [[ "$mps_flag" == "mps" ]]; then
    mps_vols="-v /tmp/mps_pipe_$GPU:/tmp/nvidia-mps -v /tmp/mps_log_$GPU:/tmp/nvidia-log"
    mps_envs="-e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps -e CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log"
  fi
  docker run --rm --gpus "\"device=$GPU_UUID\"" \
    --ipc=host $mps_vols $mps_envs \
    -v "$SDK:/opt/nvidia/cuBB" -v "$SCRIPT:/scripts" -v "$OUT:/out" \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
    -e RESULTS_DIR=/out \
    -w /scripts "$IMAGE" \
    bash -c "nsys profile --trace=cuda --duration=30 --output=/out/${label} --force-overwrite=true --stats=false python3 real_l1.py ${label} $CELLS $ITERS" 2>&1 | tail -3
}

# -----------------------------------------------------------------------------
# Run one condition (with retries)
# -----------------------------------------------------------------------------
run_condition(){ local cond=$1 mps_flag=$2 with_coloc=$3
  log "=== $cond (mps=$mps_flag coloc=$with_coloc) ==="
  for trial in $(seq 1 $N_TRIALS); do
    log "--- trial $trial/$N_TRIALS ---"
    if [[ "$with_coloc" == "yes" ]]; then
      start_nrx_bg "${cond}_t${trial}" "$mps_flag"
      sleep 15  # NRx warmup
    fi
    profile_l1 "${cond}_t${trial}" "$mps_flag"
    if [[ "$with_coloc" == "yes" ]]; then
      kill_nrx
    fi
  done
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
log "=== START mps_ts_v2 ==="
log "GPU=$GPU CELLS=$CELLS ITERS=$ITERS N_TRIALS=$N_TRIALS"

# Ensure MIG is off (should be already)
mode=$(nvidia-smi -i "$GPU" --query-gpu=mig.mode.current --format=csv,noheader)
log "MIG mode: $mode"

# Ensure DEFAULT compute mode
sudo nvidia-smi -i "$GPU" -c DEFAULT >/dev/null 2>&1 || true

# ============ Time-slicing conditions ============
log "=== TIME-SLICING mode ==="
run_condition TS_alone no_mps no
run_condition TS_coloc no_mps yes

# ============ MPS conditions ============
log "=== MPS mode ==="
start_mps
run_condition MPS_alone mps no
run_condition MPS_coloc mps yes
stop_mps

log "=== DONE. Results in $OUT ==="
ls -la "$OUT"
