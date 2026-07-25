#!/usr/bin/env bash
# Chain 11 — Megakernel validation across ALL isolation modes.
#
# Workloads (3):
#   baseL1        real_l1.py (reference)
#   persistBase   per-iter alloc/launch/free synthetic
#   persistMega   ONE launch, N iters inside (megakernel candidate)
#
# Modes (5):
#   TS              full GPU, default compute mode (temporal, no MPS, no MIG)
#   MPS100          full GPU, MPS server, no thread cap  (spatial, full budget)
#   MPS30           full GPU, MPS server, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=30
#   MIG_samepart    4g partition, L1 + NRx in SAME 4g slice (temporal, HW-partition)
#   MIG_crosspart   4g partition for L1, 3g partition for NRx (cross-partition isolation)
#
# Conditions: alone, +NRx
# Trials: 3
#
# For persistBase/persistMega the "workload" is 16 MB elementwise transform × 1000 iters.
# For baseL1 the workload is real_l1.py 20 cells × 100 iters.

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
OUT=/mydata/results/$DATE_DIR/chain11
mkdir -p "$OUT" && chmod 777 "$OUT"

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$OUT/run.log"; }

TARGET_UUID=""         # for L1
AI_UUID=""             # for AI coloc (same as TARGET for non-cross-part; different for cross-part)
MPS_ENVS=""

# =============================================================================
# Mode helpers
# =============================================================================
disable_mig(){
  sudo -n nvidia-smi mig -i $GPU -dci 2>/dev/null || true
  sudo -n nvidia-smi mig -i $GPU -dgi 2>/dev/null || true
  sudo -n nvidia-smi -i $GPU -mig 0 >/dev/null 2>&1 || true
}

stop_mps(){
  docker rm -f mps_srv 2>/dev/null || true
  sudo rm -rf /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
}

set_ts(){
  stop_mps; disable_mig; sleep 2
  sudo -n nvidia-smi -i $GPU -c DEFAULT >/dev/null 2>&1
  TARGET_UUID=$(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader)
  AI_UUID=$TARGET_UUID
  MPS_ENVS=""
}

start_mps(){ local thread_pct=${1:-100}
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
  AI_UUID=$TARGET_UUID
  MPS_ENVS="-e CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$GPU -e CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$GPU -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$thread_pct"
}

set_mps100(){ stop_mps; disable_mig; sleep 2; start_mps 100; }
set_mps30(){  stop_mps; disable_mig; sleep 2; start_mps 30; }

set_mig_samepart(){
  # Both L1 and NRx run in the SAME 4g partition
  stop_mps; disable_mig; sleep 2
  sudo -n nvidia-smi -i $GPU -c DEFAULT >/dev/null 2>&1
  sudo -n nvidia-smi -i $GPU -mig 1 >/dev/null 2>&1
  sleep 2
  # Create a 4g partition (profile 5)
  sudo -n nvidia-smi mig -i $GPU -cgi 5 -C >/dev/null 2>&1
  sleep 2
  local uuid=$(nvidia-smi -L | grep -oP 'MIG-[a-f0-9-]+' | head -1)
  if [[ -z "$uuid" ]]; then log "ERROR MIG samepart UUID"; return 1; fi
  TARGET_UUID=$uuid
  AI_UUID=$uuid
  MPS_ENVS=""
}

set_mig_crosspart(){
  # L1 in 4g, NRx in 3g — different partitions
  stop_mps; disable_mig; sleep 2
  sudo -n nvidia-smi -i $GPU -c DEFAULT >/dev/null 2>&1
  sudo -n nvidia-smi -i $GPU -mig 1 >/dev/null 2>&1
  sleep 2
  # Create 4g + 3g (profiles 5 + 9)
  sudo -n nvidia-smi mig -i $GPU -cgi 5,9 -C >/dev/null 2>&1
  sleep 2
  local uuids=($(nvidia-smi -L | grep -oP 'MIG-[a-f0-9-]+'))
  if [[ ${#uuids[@]} -lt 2 ]]; then log "ERROR MIG crosspart needs 2 UUIDs"; return 1; fi
  TARGET_UUID=${uuids[0]}   # 4g (created first)
  AI_UUID=${uuids[1]}       # 3g
  MPS_ENVS=""
}

cleanup_mode(){
  docker ps --filter 'name=ai_' -q | xargs -r docker stop -t 3 2>/dev/null
  stop_mps
  disable_mig
  sudo -n nvidia-smi -i $GPU -c DEFAULT >/dev/null 2>&1
  sleep 3
}

# =============================================================================
# Workload runners
# =============================================================================
run_nrx_bg(){ local tag=$1 uuid=$2
  # Note: uses AI_UUID (may differ from TARGET_UUID for cross-partition)
  docker run -d --rm --init --user 0:0 --gpus "\"device=$uuid\"" \
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

profile_wl(){ local label=$1 workload=$2 uuid=$3   # uuid = TARGET_UUID (L1 side)
  local cmd
  case "$workload" in
    baseL1)      cmd="python3 real_l1.py $label $CELLS $L1_ITERS" ;;
    persistBase) cmd="python3 persistent_kernel.py $label baseline   $P_ITERS" ;;
    persistMega) cmd="python3 persistent_kernel.py $label megakernel $P_ITERS" ;;
    *) echo "unknown workload"; return 1;;
  esac
  docker run --rm --user 0:0 --gpus "\"device=$uuid\"" \
    --ipc=host --pid=host -v /tmp:/tmp $MPS_ENVS \
    -v "$SDK:/opt/nvidia/cuBB" -v "$SCRIPT:/scripts" -v "$OUT:/out" \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
    -e RESULTS_DIR=/out \
    -w /scripts "$IMAGE" \
    bash -c "nsys profile --trace=cuda --duration=30 --output=/out/${label} --force-overwrite=true --stats=false $cmd" >/dev/null 2>&1
}

# =============================================================================
# Matrix
# =============================================================================
run_matrix_one_mode(){ local mode_name=$1 setup_fn=$2
  log "=========== MODE=$mode_name ==========="
  eval "$setup_fn" || { log "$mode_name setup failed, skipping"; return 1; }
  log "  TARGET_UUID=$TARGET_UUID  AI_UUID=$AI_UUID"

  for WL in baseL1 persistBase persistMega; do
    log "--- ${WL}_${mode_name}_alone (${N_TRIALS}) ---"
    for t in $(seq 1 $N_TRIALS); do
      profile_wl "${WL}_${mode_name}_alone_t${t}" $WL "$TARGET_UUID"
    done

    log "--- ${WL}_${mode_name}_nrx (${N_TRIALS}) ---"
    for t in $(seq 1 $N_TRIALS); do
      run_nrx_bg "${WL}_${mode_name}_nrx_t${t}" "$AI_UUID"
      sleep 15
      profile_wl "${WL}_${mode_name}_nrx_t${t}" $WL "$TARGET_UUID"
      kill_nrx
    done
  done

  cleanup_mode
}

# =============================================================================
# Main
# =============================================================================
log "=== CHAIN 11 START ==="

run_matrix_one_mode "TS"           "set_ts"
run_matrix_one_mode "MPS100"       "set_mps100"
run_matrix_one_mode "MPS30"        "set_mps30"
run_matrix_one_mode "MIGsamepart"  "set_mig_samepart"
run_matrix_one_mode "MIGcrosspart" "set_mig_crosspart"

log "=== CHAIN 11 DONE ==="
ls "$OUT"/*.nsys-rep 2>/dev/null | wc -l
