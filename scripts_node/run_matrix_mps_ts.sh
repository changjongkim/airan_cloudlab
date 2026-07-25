#!/usr/bin/env bash
# Comprehensive MPS + Time-slicing experiment matrix.
# Mirrors the previous MIG chain 6/7 experiments so results are directly comparable.
#
# Matrix:
#   Cell scaling  : cells = 4, 10, 20, 40  × (alone, +NRx) × (TS, MPS) × 3 trials
#   Workload sweep: cells = 20              × (alone, +NRx, +chanpred, +HBM stress, +ResNet) × (TS, MPS) × 3 trials
#
# Assumes:
#   - GPU 0 in EXCLUSIVE_PROCESS mode
#   - MPS control daemon running in container 'mps_srv' (already up)
#   - MIG disabled

set -uo pipefail

GPU=0
N_TRIALS=${N_TRIALS:-3}
DATE_DIR=${DATE_DIR:-$(date +%Y%m%d)}
IMAGE=airan:25-3-final
SDK=/mydata/aerial-cuda-accelerated-ran
REPO=/users/sgkim/AIRAN_Changjong
SCRIPT=/users/sgkim/cloudlab_aerial
OUT=/mydata/results/$DATE_DIR/matrix_mps_ts
mkdir -p "$OUT" && chmod 777 "$OUT"

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$OUT/run.log"; }

GPU_UUID=$(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader)
log "GPU UUID: $GPU_UUID"
log "MPS status: $(nvidia-smi --query-gpu=compute_mode --format=csv,noheader -i $GPU)"

# =============================================================================
# Common docker run helpers  (mps=on|off toggle)
# =============================================================================
run_ai_bg(){ local tag=$1 workload=$2 mps_flag=$3
  local mps_envs=""
  if [[ "$mps_flag" == "on" ]]; then
    mps_envs="-e CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_0 -e CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_0"
  fi
  local extra_env=""
  case "$workload" in
    hbm)      extra_env="-e PYTHONPATH=/home/aerial/.local/lib/python3.10/site-packages"
              script="/workspace/AIRAN_Changjong/experiments/run_hbm_stress.py 0 120 8.0" ;;
    nrx)      script="/workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py 0 120" ;;
    chanpred) script="/workspace/AIRAN_Changjong/experiments/run_channel_prediction.py" ;;
    resnet)   extra_env="-e PYTHONPATH=/home/aerial/.local/lib/python3.10/site-packages"
              script="/workspace/AIRAN_Changjong/experiments/run_resnet_stress.py" ;;
    *) echo "unknown workload: $workload"; return 1 ;;
  esac

  docker run -d --rm --init --user 0:0 --gpus "\"device=$GPU_UUID\"" \
    --ipc=host --pid=host -v /tmp:/tmp $mps_envs \
    -v "$SDK:/opt/nvidia/cuBB" \
    -v "$REPO:/workspace/AIRAN_Changjong" \
    -v "$OUT:/aiout" \
    -e cuBB_SDK=/opt/nvidia/cuBB \
    -e HOME=/tmp \
    $extra_env \
    --name "ai_${tag}_$(date +%s)" \
    "$IMAGE" \
    bash -c "python3 $script > /aiout/${tag}_ai.log 2>&1" >/dev/null
}
kill_ai(){ docker ps --filter 'name=ai_' -q | xargs -r docker stop -t 3 2>/dev/null; sleep 1; }

profile_l1(){ local label=$1 cells=$2 iters=$3 mps_flag=$4
  local mps_envs=""
  if [[ "$mps_flag" == "on" ]]; then
    mps_envs="-e CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_0 -e CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_0"
  fi
  docker run --rm --user 0:0 --gpus "\"device=$GPU_UUID\"" \
    --ipc=host --pid=host -v /tmp:/tmp $mps_envs \
    -v "$SDK:/opt/nvidia/cuBB" -v "$SCRIPT:/scripts" -v "$OUT:/out" \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
    -e RESULTS_DIR=/out \
    -w /scripts "$IMAGE" \
    bash -c "nsys profile --trace=cuda --duration=30 --output=/out/${label} --force-overwrite=true --stats=false python3 real_l1.py ${label} $cells $iters" 2>&1 | tail -1
}

# =============================================================================
# Cell scaling sweep
# =============================================================================
cell_sweep(){ local mode=$1 mps_flag=$2
  local cells_list="4 10 20 40"
  for cells in $cells_list; do
    local iters
    case $cells in
      4)  iters=400 ;;
      10) iters=200 ;;
      20) iters=100 ;;
      40) iters=50  ;;
    esac

    log "=== ${mode}_c${cells}_alone ==="
    for t in $(seq 1 $N_TRIALS); do
      log "--- trial $t/$N_TRIALS ---"
      profile_l1 "${mode}_c${cells}_alone_t${t}" $cells $iters "$mps_flag"
    done

    log "=== ${mode}_c${cells}_nrx ==="
    for t in $(seq 1 $N_TRIALS); do
      log "--- trial $t/$N_TRIALS ---"
      run_ai_bg "${mode}_c${cells}_nrx_t${t}" nrx "$mps_flag"
      sleep 15
      profile_l1 "${mode}_c${cells}_nrx_t${t}" $cells $iters "$mps_flag"
      kill_ai
    done
  done
}

# =============================================================================
# Workload sweep (at cells=20)
# =============================================================================
workload_sweep(){ local mode=$1 mps_flag=$2
  local cells=20 iters=100
  for wl in chanpred hbm resnet; do
    log "=== ${mode}_c${cells}_${wl} ==="
    for t in $(seq 1 $N_TRIALS); do
      log "--- trial $t/$N_TRIALS ---"
      run_ai_bg "${mode}_c${cells}_${wl}_t${t}" "$wl" "$mps_flag"
      sleep 15
      profile_l1 "${mode}_c${cells}_${wl}_t${t}" $cells $iters "$mps_flag"
      kill_ai
    done
  done
}

# =============================================================================
# Main
# =============================================================================
log "=== START matrix_mps_ts ==="

log "===== TIME-SLICING mode ====="
log "cell sweep..."
cell_sweep "TS" off
log "workload sweep..."
workload_sweep "TS" off

log "===== MPS mode ====="
log "cell sweep..."
cell_sweep "MPS" on
log "workload sweep..."
workload_sweep "MPS" on

log "=== DONE. Results in $OUT ==="
ls -la "$OUT" | head -50
