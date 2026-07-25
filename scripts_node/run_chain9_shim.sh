#!/usr/bin/env bash
# Chain 9: Hypothesis shim matrix on Chain 6/7-equivalent conditions.
#
# Shims (5): baseline, cudaFreeAsync (Option A), cudaMemPool (Option B),
#            defer (proof_56), arena (Hypothesis A persistent pool)
# Modes (3): TS, MPS, MIG 4g same-partition
# Cells (default 20; overridable): from CELLS_LIST env var
# Coloc:     alone, +NRx
# Trials:    3
#
# Each L1 profile: 30 s NSYS window, real_l1.py <label> <cells> <iters>
# Each coloc: NRx container background for 120 s, kill after L1 profile.
#
# Layout:
#   /mydata/results/${DATE_DIR}/chain9/
#     <SHIM>_<MODE>_c<CELLS>_<COND>_t<N>.nsys-rep
#
# Note: MIG 4g mode requires nvidia-smi mig setup — done inline.

set -uo pipefail

GPU=0
CELLS_LIST=${CELLS_LIST:-"20"}
N_TRIALS=${N_TRIALS:-3}
SHIMS=${SHIMS:-"baseline cudaFreeAsync cudaMemPool defer arena"}
MODES=${MODES:-"TS MPS MIG4g"}
DATE_DIR=${DATE_DIR:-$(date +%Y%m%d)}
IMAGE=airan:25-3-final
SDK=/mydata/aerial-cuda-accelerated-ran
REPO=/users/sgkim/AIRAN_Changjong
SCRIPT=/users/sgkim/cloudlab_aerial
SHIMS_DIR=/users/sgkim/shims
OUT=/mydata/results/$DATE_DIR/chain9
mkdir -p "$OUT" && chmod 777 "$OUT"

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$OUT/run.log"; }

# =============================================================================
# GPU / MIG / MPS state helpers
# =============================================================================
set_mode(){ local m=$1
  case "$m" in
    TS)   set_ts ;;
    MPS)  set_mps ;;
    MIG4g) set_mig4g ;;
    *) echo "unknown mode: $m"; return 1 ;;
  esac
}

set_ts(){
  # Ensure MIG disabled, MPS off, compute mode default
  docker rm -f mps_srv 2>/dev/null || true
  sudo -n nvidia-smi mig -i $GPU -dci 2>/dev/null || true
  sudo -n nvidia-smi mig -i $GPU -dgi 2>/dev/null || true
  sudo -n nvidia-smi -i $GPU -mig 0 >/dev/null 2>&1 || true
  sleep 2
  sudo -n nvidia-smi -i $GPU -c DEFAULT >/dev/null 2>&1
  TARGET_UUID=$(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader)
  MPS_ENVS=""
}

set_mps(){
  # Ensure MIG disabled, MPS on with container
  sudo -n nvidia-smi mig -i $GPU -dci 2>/dev/null || true
  sudo -n nvidia-smi mig -i $GPU -dgi 2>/dev/null || true
  sudo -n nvidia-smi -i $GPU -mig 0 >/dev/null 2>&1 || true
  sleep 2
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
  MPS_ENVS="-e CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$GPU -e CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$GPU"
}

set_mig4g(){
  docker rm -f mps_srv 2>/dev/null || true
  sudo -n nvidia-smi -i $GPU -c DEFAULT >/dev/null 2>&1
  # Reset any existing MIG state
  sudo -n nvidia-smi mig -i $GPU -dci 2>/dev/null || true
  sudo -n nvidia-smi mig -i $GPU -dgi 2>/dev/null || true
  # Enable MIG
  sudo -n nvidia-smi -i $GPU -mig 1 >/dev/null 2>&1
  sleep 2
  sudo -n nvidia-smi mig -i $GPU -cgi 5 -C >/dev/null 2>&1
  sleep 2
  TARGET_UUID=$(nvidia-smi -L | grep -oP 'MIG-[a-f0-9-]+' | head -1)
  MPS_ENVS=""
  if [[ -z "$TARGET_UUID" ]]; then
    log "ERROR: MIG UUID not created"
    return 1
  fi
}

cleanup_mode(){
  docker ps --filter 'name=ai_' -q | xargs -r docker stop -t 3 2>/dev/null
  docker rm -f mps_srv 2>/dev/null || true
  sleep 2
  sudo -n nvidia-smi mig -i $GPU -dci 2>/dev/null || true
  sudo -n nvidia-smi mig -i $GPU -dgi 2>/dev/null || true
  sudo -n nvidia-smi -i $GPU -mig 0 >/dev/null 2>&1 || true
  sudo -n nvidia-smi -i $GPU -c DEFAULT >/dev/null 2>&1
  sleep 2
}

# =============================================================================
# Shim → LD_PRELOAD env
# =============================================================================
shim_env(){ local s=$1
  case "$s" in
    baseline)      echo "" ;;
    cudaFreeAsync) echo "-e LD_PRELOAD=/shims/cudaFreeAsync.so" ;;
    cudaMemPool)   echo "-e LD_PRELOAD=/shims/cudaMemPool.so" ;;
    defer)         echo "-e LD_PRELOAD=/shims/defer.so" ;;
    arena)         echo "-e LD_PRELOAD=/shims/arena.so" ;;
    *) echo "" ;;
  esac
}

# =============================================================================
# Workload launchers
# =============================================================================
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

profile_l1(){ local label=$1 cells=$2 iters=$3 shim=$4
  local shim_e=$(shim_env "$shim")
  docker run --rm --user 0:0 --gpus "\"device=$TARGET_UUID\"" \
    --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS $shim_e \
    -v "$SDK:/opt/nvidia/cuBB" -v "$SCRIPT:/scripts" \
    -v "$SHIMS_DIR:/shims" -v "$OUT:/out" \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
    -e RESULTS_DIR=/out \
    -w /scripts "$IMAGE" \
    bash -c "nsys profile --trace=cuda --duration=30 --output=/out/${label} --force-overwrite=true --stats=false python3 real_l1.py ${label} $cells $iters" >/dev/null 2>&1
}

iters_for_cells(){
  case $1 in
    4)  echo 400 ;;
    10) echo 250 ;;
    20) echo 100 ;;
    40) echo 60  ;;
    60) echo 40  ;;
    *)  echo 100 ;;
  esac
}

# =============================================================================
# Main
# =============================================================================
log "=== START chain9 shim matrix ==="
log "SHIMS=[$SHIMS] MODES=[$MODES] CELLS=[$CELLS_LIST] TRIALS=$N_TRIALS"

for MODE in $MODES; do
  log "===== MODE=$MODE ====="
  set_mode "$MODE" || { log "skip $MODE (setup failed)"; continue; }
  log "  TARGET_UUID=$TARGET_UUID"

  for CELLS in $CELLS_LIST; do
    local_iters=$(iters_for_cells $CELLS)

    for SHIM in $SHIMS; do
      # Alone
      log "--- ${SHIM}_${MODE}_c${CELLS}_alone (${N_TRIALS} trials) ---"
      for t in $(seq 1 $N_TRIALS); do
        profile_l1 "${SHIM}_${MODE}_c${CELLS}_alone_t${t}" $CELLS $local_iters "$SHIM"
      done

      # NRx coloc
      log "--- ${SHIM}_${MODE}_c${CELLS}_nrx (${N_TRIALS} trials) ---"
      for t in $(seq 1 $N_TRIALS); do
        run_nrx_bg "${SHIM}_${MODE}_c${CELLS}_nrx_t${t}"
        sleep 15
        profile_l1 "${SHIM}_${MODE}_c${CELLS}_nrx_t${t}" $CELLS $local_iters "$SHIM"
        kill_nrx
      done
    done
  done

  cleanup_mode
done

log "=== chain9 DONE ==="
ls "$OUT" | wc -l
