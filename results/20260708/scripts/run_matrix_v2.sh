#!/usr/bin/env bash
# Full MPS + Time-slicing matrix, corrected version.
# Fixes:
#   - GPU compute mode transition between TS (DEFAULT) and MPS (EXCLUSIVE_PROCESS)
#   - AI process gets PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src (NRx needs aerial)
#   - Full cell × workload matrix per mode

set -uo pipefail

GPU=0
N_TRIALS=${N_TRIALS:-3}
CELLS_LIST=${CELLS_LIST:-"4 10 20 40 60"}
WORKLOADS=${WORKLOADS:-"nrx chanpred hbm resnet"}
DATE_DIR=${DATE_DIR:-$(date +%Y%m%d)}
IMAGE=airan:25-3-final
SDK=/mydata/aerial-cuda-accelerated-ran
REPO=/users/sgkim/AIRAN_Changjong
SCRIPT=/users/sgkim/cloudlab_aerial
OUT=/mydata/results/$DATE_DIR/matrix_v2
mkdir -p "$OUT" && chmod 777 "$OUT"

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$OUT/run.log"; }

GPU_UUID=$(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader)
log "GPU UUID: $GPU_UUID"

set_compute_mode(){ local mode=$1
  sudo -n nvidia-smi -i $GPU -c $mode >/dev/null 2>&1
}

start_mps_server(){
  log "Starting MPS server (in container) for GPU $GPU"
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
}

stop_mps_server(){
  log "Stopping MPS server"
  docker exec mps_srv bash -c "echo quit | nvidia-cuda-mps-control" 2>/dev/null || true
  sleep 2
  docker rm -f mps_srv 2>/dev/null || true
  sudo rm -rf /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
}

# -----------------------------------------------------------------------------
# Docker helpers  (mps_flag: "on"|"off")
# -----------------------------------------------------------------------------
run_ai_bg(){ local tag=$1 workload=$2 mps_flag=$3
  local mps_envs=""
  if [[ "$mps_flag" == "on" ]]; then
    mps_envs="-e CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$GPU -e CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$GPU"
  fi
  local extra_env=""
  local script=""
  case "$workload" in
    hbm)      extra_env="-e PYTHONPATH=/home/aerial/.local/lib/python3.10/site-packages"
              script="/workspace/AIRAN_Changjong/experiments/run_hbm_stress.py 0 180 8.0" ;;
    nrx)      extra_env="-e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src"
              script="/workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py 0 180" ;;
    chanpred) extra_env="-e PYTHONPATH=/home/aerial/.local/lib/python3.10/site-packages"
              script="/workspace/AIRAN_Changjong/experiments/run_channel_prediction.py 0 180" ;;
    resnet)   extra_env="-e PYTHONPATH=/home/aerial/.local/lib/python3.10/site-packages"
              script="/workspace/AIRAN_Changjong/experiments/run_resnet_stress.py 0 180" ;;
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
    mps_envs="-e CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$GPU -e CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$GPU"
  fi
  docker run --rm --user 0:0 --gpus "\"device=$GPU_UUID\"" \
    --ipc=host --pid=host -v /tmp:/tmp $mps_envs \
    -v "$SDK:/opt/nvidia/cuBB" -v "$SCRIPT:/scripts" -v "$OUT:/out" \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
    -e RESULTS_DIR=/out \
    -w /scripts "$IMAGE" \
    bash -c "nsys profile --trace=cuda --duration=30 --output=/out/${label} --force-overwrite=true --stats=false python3 real_l1.py ${label} $cells $iters" >/dev/null 2>&1
}

iters_for_cells(){ local c=$1
  case $c in
    4)  echo 400 ;;
    10) echo 250 ;;
    20) echo 150 ;;
    40) echo 100 ;;
    60) echo 80  ;;
    *)  echo 100 ;;
  esac
}

# -----------------------------------------------------------------------------
# Run one mode's full matrix
# -----------------------------------------------------------------------------
run_mode(){ local mode=$1 mps_flag=$2
  log "===== ${mode} mode ====="

  for cells in $CELLS_LIST; do
    local iters=$(iters_for_cells $cells)

    # alone baseline
    log "--- ${mode}_c${cells}_alone ($N_TRIALS trials) ---"
    for t in $(seq 1 $N_TRIALS); do
      profile_l1 "${mode}_c${cells}_alone_t${t}" $cells $iters "$mps_flag"
    done

    # each AI workload
    for wl in $WORKLOADS; do
      log "--- ${mode}_c${cells}_${wl} ($N_TRIALS trials) ---"
      for t in $(seq 1 $N_TRIALS); do
        run_ai_bg "${mode}_c${cells}_${wl}_t${t}" "$wl" "$mps_flag"
        sleep 15
        profile_l1 "${mode}_c${cells}_${wl}_t${t}" $cells $iters "$mps_flag"
        kill_ai
      done
    done
  done
}

# =============================================================================
# Main
# =============================================================================
log "=== START matrix_v2 (cells=[$CELLS_LIST] workloads=[$WORKLOADS] trials=$N_TRIALS) ==="

# Ensure clean start
docker rm -f mps_srv 2>/dev/null || true

# ==== Time-slicing first (compute DEFAULT, no MPS) ====
set_compute_mode DEFAULT
log "Compute mode: $(nvidia-smi -i $GPU --query-gpu=compute_mode --format=csv,noheader)"
run_mode "TS" off

# ==== MPS next (compute EXCLUSIVE_PROCESS + MPS server) ====
set_compute_mode EXCLUSIVE_PROCESS
log "Compute mode: $(nvidia-smi -i $GPU --query-gpu=compute_mode --format=csv,noheader)"
start_mps_server
run_mode "MPS" on
stop_mps_server

# Restore
set_compute_mode DEFAULT

log "=== DONE. Files in $OUT ==="
ls -la "$OUT" | wc -l
