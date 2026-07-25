#!/usr/bin/env bash
# MPS vs Time-slicing comparison for AI-RAN cudaFree contention study.
# Mirrors MIG 4g L1 + NRx coloc condition (baseline = ~18,000ms cudaFree per 30s).
#
# Conditions measured:
#   A. MIG 4g L1 + NRx coloc (sanity re-baseline vs prior)
#   B. MPS mode  L1 alone
#   C. MPS mode  L1 + NRx coloc
#   D. Time-slicing L1 alone
#   E. Time-slicing L1 + NRx coloc
#
# Each condition: single 30s NSYS window on the L1 process, AI process co-runs
# for the whole duration. Host CUDA time breakdown → what fraction is cudaFree.

set -uo pipefail
cd "$HOME/cloudlab_aerial"

GPU=${GPU:-0}
CELLS=${CELLS:-20}
ITERS=${ITERS:-100}
DATE_DIR=${DATE_DIR:-$(date +%Y%m%d)}
IMAGE=airan:25-3
SDK=/mydata/aerial-cuda-accelerated-ran
REPO=$HOME/AIRAN_Changjong
SCRIPT=$HOME/cloudlab_aerial
UID_=$(id -u); GID_=$(id -g)
OUT=$HOME/results/$DATE_DIR/mps_ts_compare
mkdir -p "$OUT" && chmod 777 "$OUT"

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*"; }

# -----------------------------------------------------------------------------
# MPS helpers
# -----------------------------------------------------------------------------
start_mps(){
  log "starting MPS server on GPU $GPU"
  sudo nvidia-smi -i "$GPU" -c EXCLUSIVE_PROCESS >/dev/null 2>&1 || true
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$GPU
  export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$GPU
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  CUDA_VISIBLE_DEVICES=$GPU nvidia-cuda-mps-control -d
  sleep 3
}
stop_mps(){
  log "stopping MPS server"
  echo quit | nvidia-cuda-mps-control 2>/dev/null || true
  sudo nvidia-smi -i "$GPU" -c DEFAULT >/dev/null 2>&1 || true
  rm -rf /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
  sleep 2
}

# -----------------------------------------------------------------------------
# Workload launchers
# -----------------------------------------------------------------------------
UUID=""
resolve_uuid(){
  # For MIG mode, need MIG instance UUID. For MPS/time-slicing, use GPU UUID.
  local mode="$1"
  if [[ "$mode" == "mig" ]]; then
    UUID=$(nvidia-smi -L | awk '/MIG/ {gsub(/[()]/,""); print $NF}' | head -1)
    if [[ -z "$UUID" ]]; then
      echo "ERROR: no MIG instance found"; exit 1
    fi
  else
    UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
  fi
  log "using UUID=$UUID"
}

start_ai_bg(){ local tag=$1 script=$2 mps_flag=$3
  local mps_vols=""
  local mps_envs=""
  if [[ "$mps_flag" == "mps" ]]; then
    mps_vols="-v /tmp/mps_pipe_$GPU:/tmp/nvidia-mps -v /tmp/mps_log_$GPU:/tmp/nvidia-log"
    mps_envs="-e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps -e CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log"
  fi
  docker run -d --rm --init --gpus "\"device=$UUID\"" \
    --ipc=host $mps_vols $mps_envs \
    -v "$SDK:/opt/nvidia/cuBB" \
    -v "$REPO:/workspace/AIRAN_Changjong" \
    -v "$SCRIPT:/scripts" \
    -v "$OUT:/aiout" \
    --name "ai_${tag}_$(date +%s%N)" \
    "$IMAGE" \
    bash -c "python3 $script > /aiout/${tag}_ai.log 2>&1" >/dev/null
}
kill_ai(){ docker ps --filter 'name=ai_' -q | xargs -r docker stop -t 3 2>/dev/null; sleep 1; }

profile_l1(){ local label=$1 mps_flag=$2
  local mps_vols=""
  local mps_envs=""
  if [[ "$mps_flag" == "mps" ]]; then
    mps_vols="-v /tmp/mps_pipe_$GPU:/tmp/nvidia-mps -v /tmp/mps_log_$GPU:/tmp/nvidia-log"
    mps_envs="-e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps -e CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log"
  fi
  # 30s NSYS window with --duration=30
  docker run --rm --gpus "\"device=$UUID\"" --user "$UID_:$GID_" \
    --ipc=host $mps_vols $mps_envs \
    -v "$SDK:/opt/nvidia/cuBB" -v "$SCRIPT:/scripts" -v "$OUT:/out" \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
    -e RESULTS_DIR=/out \
    -w /scripts "$IMAGE" \
    bash -c "nsys profile --trace=cuda --duration=30 --output=/out/${label} --force-overwrite=true --stats=false python3 real_l1.py ${label} $CELLS $ITERS" 2>&1 | tail -5
}

# -----------------------------------------------------------------------------
# CONDITION A: MIG 4g L1+NRx coloc (sanity re-baseline)
# -----------------------------------------------------------------------------
condition_A(){
  log '=== A. MIG 4g L1 + NRx same partition (sanity) ==='
  sudo ~/02_mig.sh config split-60-40 || { echo "MIG setup failed"; return 1; }
  resolve_uuid mig
  # 4g is the larger one — use the first UUID
  local L1_UUID=$UUID
  # Launch NRx in same partition
  UUID=$L1_UUID
  start_ai_bg "A_mig_nrx" "/workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py" "no_mps"
  sleep 15  # let NRx warm up
  profile_l1 "A_mig_L1_nrx_coloc" "no_mps"
  kill_ai
  sudo ~/02_mig.sh hard-reset || true
}

# -----------------------------------------------------------------------------
# CONDITION B/C: MPS mode
# -----------------------------------------------------------------------------
condition_BC(){
  log '=== B/C. MPS mode ==='
  # Ensure MIG off
  sudo ~/02_mig.sh disable 2>/dev/null || true
  # Start MPS
  start_mps
  resolve_uuid mps

  log '--- B. MPS L1 alone ---'
  profile_l1 "B_mps_L1_alone" "mps"

  log '--- C. MPS L1 + NRx coloc ---'
  start_ai_bg "C_mps_nrx" "/workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py" "mps"
  sleep 15  # let NRx warm up
  profile_l1 "C_mps_L1_nrx_coloc" "mps"
  kill_ai

  stop_mps
}

# -----------------------------------------------------------------------------
# CONDITION D/E: Time-slicing (default GPU sharing)
# -----------------------------------------------------------------------------
condition_DE(){
  log '=== D/E. Time-slicing mode ==='
  # Ensure MIG off + MPS off + compute mode default
  sudo ~/02_mig.sh disable 2>/dev/null || true
  sudo nvidia-smi -i "$GPU" -c DEFAULT >/dev/null 2>&1 || true
  resolve_uuid time_sliced

  log '--- D. Time-slicing L1 alone ---'
  profile_l1 "D_ts_L1_alone" "no_mps"

  log '--- E. Time-slicing L1 + NRx coloc ---'
  start_ai_bg "E_ts_nrx" "/workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py" "no_mps"
  sleep 15
  profile_l1 "E_ts_L1_nrx_coloc" "no_mps"
  kill_ai
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
log "=== Starting MPS vs TS comparison, output → $OUT ==="

# Skip A if already have MIG baseline (comment in if needed)
# condition_A

condition_BC
condition_DE

log "=== DONE. Results in $OUT ==="
ls -la "$OUT"
