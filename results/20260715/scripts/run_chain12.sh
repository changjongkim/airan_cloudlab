#!/usr/bin/env bash
# Chain 12 — full mecha-verification matrix combining approaches A/B/C

# Workloads (5):
#   baseL1              real cuPHY L1 (reference)
#   baseL1_arena        real cuPHY L1 + arena_shim LD_PRELOAD  ← Approach C (real cuPHY, buffer-reuse)
#   multiBase           synthetic 6-stage per-iter alloc/launch/free
#   multiMega           synthetic 6-stage inline megakernel   ← Approach B
#   persistMega         (elementwise reference from chain 11 pattern)
#
# (Approach A — PuschRxPipelineFactory — confirmed segfault, omitted here.)
#
# Modes (2): TS, MPS100
# Conditions: alone, +NRx
# Trials: 3

set -uo pipefail

GPU=0
CELLS=${CELLS:-20}
L1_ITERS=${L1_ITERS:-100}
P_ITERS=${P_ITERS:-1000}
N_TRIALS=${N_TRIALS:-3}
DATE_DIR=${DATE_DIR:-$(date +%Y%m%d)}
IMAGE=airan:25-3-final
SDK=/mydata/aerial-cuda-accelerated-ran
REPO=/users/sgkim/AIRAN_Changjong
SCRIPT=/users/sgkim/cloudlab_aerial
SHIMS_DIR=/users/sgkim/shims
OUT=/mydata/results/$DATE_DIR/chain12
mkdir -p "$OUT" && chmod 777 "$OUT"

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$OUT/run.log"; }

TARGET_UUID=""; MPS_ENVS=""

stop_mps(){
  docker rm -f mps_srv 2>/dev/null || true
  sudo rm -rf /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
}

set_ts(){
  stop_mps
  sudo -n nvidia-smi -i $GPU -c DEFAULT >/dev/null 2>&1
  TARGET_UUID=$(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader)
  MPS_ENVS=""
}

set_mps(){
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

cleanup_mode(){
  docker ps --filter 'name=ai_' -q | xargs -r docker stop -t 3 2>/dev/null
  stop_mps
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
    bash -c "python3 /workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py 0 180 > /aiout/${tag}_ai.log 2>&1" >/dev/null
}
kill_nrx(){ docker ps --filter 'name=ai_' -q | xargs -r docker stop -t 3 2>/dev/null; sleep 1; }

# Workload → (script, args, extra_docker_env)
profile_wl(){ local label=$1 wl=$2
  local cmd extra=""
  case "$wl" in
    baseL1)       cmd="python3 real_l1.py            $label $CELLS $L1_ITERS" ;;
    baseL1_arena) cmd="python3 real_l1.py            $label $CELLS $L1_ITERS"
                  extra="-e LD_PRELOAD=/shims/arena.so" ;;
    multiBase)    cmd="python3 multi_stage_kernel.py $label baseline   $P_ITERS" ;;
    multiMega)    cmd="python3 multi_stage_kernel.py $label megakernel $P_ITERS" ;;
    persistMega)  cmd="python3 persistent_kernel.py  $label megakernel $P_ITERS" ;;
    *) echo "unknown workload $wl"; return 1;;
  esac
  docker run --rm --user 0:0 --gpus "\"device=$TARGET_UUID\"" \
    --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS $extra \
    -v "$SDK:/opt/nvidia/cuBB" -v "$SCRIPT:/scripts" \
    -v "$SHIMS_DIR:/shims" -v "$OUT:/out" \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
    -e RESULTS_DIR=/out \
    -w /scripts "$IMAGE" \
    bash -c "nsys profile --trace=cuda --duration=30 --output=/out/${label} --force-overwrite=true --stats=false $cmd" >/dev/null 2>&1
}

run_matrix(){ local mode_name=$1 setup_fn=$2
  log "===== MODE=$mode_name ====="
  eval $setup_fn || { log "$mode_name setup failed"; return 1; }
  log "  TARGET_UUID=$TARGET_UUID"
  for WL in baseL1 baseL1_arena multiBase multiMega persistMega; do
    log "--- ${WL}_${mode_name}_alone ---"
    for t in $(seq 1 $N_TRIALS); do
      profile_wl "${WL}_${mode_name}_alone_t${t}" $WL
    done
    log "--- ${WL}_${mode_name}_nrx ---"
    for t in $(seq 1 $N_TRIALS); do
      run_nrx_bg "${WL}_${mode_name}_nrx_t${t}"
      sleep 15
      profile_wl "${WL}_${mode_name}_nrx_t${t}" $WL
      kill_nrx
    done
  done
  cleanup_mode
}

log "=== chain12 START ==="
run_matrix "TS"     "set_ts"
run_matrix "MPS100" "set_mps"
log "=== chain12 DONE ==="
ls "$OUT"/*.nsys-rep 2>/dev/null | wc -l
