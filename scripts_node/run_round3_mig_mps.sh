#!/usr/bin/env bash
# Round 3: MIG 4g + MPS fair comparison.
# Question: Is MPS's low cudaFree from mechanism (spatial) or from resource abundance (2.6× more SMs)?
# Approach: Constrain MPS to same 42 SMs as MIG 4g. If penalty comes back → resource matters most.
#
# Matrix (all cells=20, 3 trials each):
#   MIG4g_alone_noMPS     L1 alone in 4g partition, no MPS       — baseline temporal
#   MIG4g_nrx_noMPS       L1 + NRx in same 4g, no MPS            — temporal on 42 SMs
#   MIG4g_alone_MPS       L1 alone in 4g partition, MPS on       — spatial alone
#   MIG4g_nrx_MPS         L1 + NRx in same 4g, MPS on            — spatial on 42 SMs  ★ key
#
# When done, auto-kicks off matrix_v2 continuation in background.

set -uo pipefail

GPU=0
N_TRIALS=${N_TRIALS:-3}
CELLS=${CELLS:-20}
ITERS=${ITERS:-100}
DATE_DIR=${DATE_DIR:-$(date +%Y%m%d)}
IMAGE=airan:25-3-final
SDK=/mydata/aerial-cuda-accelerated-ran
REPO=/users/sgkim/AIRAN_Changjong
SCRIPT=/users/sgkim/cloudlab_aerial
OUT=/mydata/results/$DATE_DIR/round3_mig_mps
mkdir -p "$OUT" && chmod 777 "$OUT"

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$OUT/run.log"; }

# =============================================================================
# MIG setup helpers
# =============================================================================
enable_mig(){
  log "Enabling MIG on GPU $GPU"
  sudo -n nvidia-smi -i $GPU -mig 1 2>&1 | tail -2
  sleep 3
  log "Creating 4g.20gb partition (profile 5)"
  sudo -n nvidia-smi mig -i $GPU -cgi 5 -C 2>&1 | tail -3
  sleep 2
}

disable_mig(){
  log "Destroying MIG instances and disabling MIG mode"
  sudo -n nvidia-smi mig -i $GPU -dci 2>/dev/null || true
  sudo -n nvidia-smi mig -i $GPU -dgi 2>/dev/null || true
  sudo -n nvidia-smi -i $GPU -mig 0 2>&1 | tail -1
  sleep 2
}

get_mig_uuid(){
  nvidia-smi -L | grep -oP 'MIG-[a-f0-9-]+' | head -1
}

# =============================================================================
# MPS helpers (inside container, targeting MIG UUID)
# =============================================================================
start_mps_container(){ local target_uuid=$1
  log "Starting MPS server container (target: $target_uuid)"
  sudo rm -rf /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
  sudo mkdir -p /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
  sudo chmod 777 /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
  docker rm -f mps_srv 2>/dev/null || true
  docker run -d --gpus "\"device=$target_uuid\"" --ipc=host --pid=host --user 0:0 \
    -v /tmp:/tmp \
    -e CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$GPU \
    -e CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$GPU \
    --name mps_srv "$IMAGE" \
    bash -c "nvidia-cuda-mps-control -d && sleep infinity"
  sleep 5
}

stop_mps_container(){
  log "Stopping MPS server"
  docker exec mps_srv bash -c "echo quit | nvidia-cuda-mps-control" 2>/dev/null || true
  sleep 2
  docker rm -f mps_srv 2>/dev/null || true
  sudo rm -rf /tmp/mps_pipe_$GPU /tmp/mps_log_$GPU
}

# =============================================================================
# Workload launchers
# =============================================================================
run_nrx_bg(){ local tag=$1 target_uuid=$2 mps_flag=$3
  local mps_envs=""
  if [[ "$mps_flag" == "on" ]]; then
    mps_envs="-e CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$GPU -e CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$GPU"
  fi
  docker run -d --rm --init --user 0:0 --gpus "\"device=$target_uuid\"" \
    --ipc=host --pid=host -v /tmp:/tmp $mps_envs \
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
kill_ai(){ docker ps --filter 'name=ai_' -q | xargs -r docker stop -t 3 2>/dev/null; sleep 1; }

profile_l1(){ local label=$1 target_uuid=$2 mps_flag=$3
  local mps_envs=""
  if [[ "$mps_flag" == "on" ]]; then
    mps_envs="-e CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$GPU -e CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$GPU"
  fi
  docker run --rm --user 0:0 --gpus "\"device=$target_uuid\"" \
    --ipc=host --pid=host -v /tmp:/tmp $mps_envs \
    -v "$SDK:/opt/nvidia/cuBB" -v "$SCRIPT:/scripts" -v "$OUT:/out" \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
    -e RESULTS_DIR=/out \
    -w /scripts "$IMAGE" \
    bash -c "nsys profile --trace=cuda --duration=30 --output=/out/${label} --force-overwrite=true --stats=false python3 real_l1.py ${label} $CELLS $ITERS" >/dev/null 2>&1
}

# =============================================================================
# Main
# =============================================================================
log "=== ROUND 3: MIG 4g + MPS fair comparison ==="

# Step 1: Enable MIG and create 4g partition
enable_mig
MIG_UUID=$(get_mig_uuid)
log "MIG UUID: $MIG_UUID"
if [[ -z "$MIG_UUID" ]]; then
  log "ERROR: MIG UUID empty"
  exit 1
fi

# Step 2: MIG 4g alone (no MPS) — baseline temporal
log ""
log "===== A. MIG 4g without MPS (temporal, 42 SMs) ====="
sudo -n nvidia-smi -i $GPU -c DEFAULT >/dev/null 2>&1
for t in $(seq 1 $N_TRIALS); do
  log "--- MIG4g_alone_noMPS trial $t ---"
  profile_l1 "MIG4g_alone_noMPS_t${t}" "$MIG_UUID" off
done
for t in $(seq 1 $N_TRIALS); do
  log "--- MIG4g_nrx_noMPS trial $t ---"
  run_nrx_bg "MIG4g_nrx_noMPS_t${t}" "$MIG_UUID" off
  sleep 15
  profile_l1 "MIG4g_nrx_noMPS_t${t}" "$MIG_UUID" off
  kill_ai
done

# Step 3: MIG 4g with MPS — spatial within partition
log ""
log "===== B. MIG 4g with MPS (spatial, 42 SMs) ====="
sudo -n nvidia-smi -i $GPU -c EXCLUSIVE_PROCESS >/dev/null 2>&1
start_mps_container "$MIG_UUID"
for t in $(seq 1 $N_TRIALS); do
  log "--- MIG4g_alone_MPS trial $t ---"
  profile_l1 "MIG4g_alone_MPS_t${t}" "$MIG_UUID" on
done
for t in $(seq 1 $N_TRIALS); do
  log "--- MIG4g_nrx_MPS trial $t (★ key measurement) ---"
  run_nrx_bg "MIG4g_nrx_MPS_t${t}" "$MIG_UUID" on
  sleep 15
  profile_l1 "MIG4g_nrx_MPS_t${t}" "$MIG_UUID" on
  kill_ai
done
stop_mps_container

# Step 4: Cleanup
disable_mig
sudo -n nvidia-smi -i $GPU -c DEFAULT >/dev/null 2>&1

log "=== ROUND 3 DONE ==="
log "Files in $OUT:"
ls -la "$OUT" | grep nsys-rep

# =============================================================================
# Auto-continue with Matrix v2 (full workload sweep for TS + MPS)
# =============================================================================
log ""
log "=== AUTO-CONTINUE: kicking off Matrix v2 in background ==="
nohup ~/cloudlab_aerial/run_matrix_v2.sh > /mydata/results/matrix_v2_stdout.log 2>&1 &
sleep 3
log "Matrix v2 PID: $(pgrep -f 'run_matrix_v2' | head -1)"
log "Log: /mydata/results/matrix_v2_stdout.log"
log "Output: /mydata/results/$DATE_DIR/matrix_v2/"
