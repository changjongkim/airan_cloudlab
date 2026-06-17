#!/usr/bin/env bash
# MPS + NSYS on no-MIG — capture kernel overlap/gap under CUDA MPS (concurrent exec).
# Matches the same conditions as the default-time-slice campaigns so MPS can be
# compared 1:1. nsys (tracing) preserves MPS concurrency; ncu (replay) would not.
set -uo pipefail
IMG="${AERIAL_IMAGE:-nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb}"
AERIAL_REPO="${AERIAL_REPO:-/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran}"
REPO="${REPO:-/pscratch/sd/s/sgkim/kcj/airan_cloudlab}"
HANDOFF="${HANDOFF:-$REPO/results/perlmutter_handoff}"
VENV="${VENV:-$HANDOFF/airan_venv}"; HF_HOME="${HF_HOME:-$HANDOFF/hf_cache}"
SCRIPTS_DIR="$REPO/scripts_for_node/cloudlab_aerial"; AI_DIR="$REPO/scripts_for_node/experiments"
RESULTS_DIR="${RESULTS_DIR:-$HANDOFF/perlmutter_nomig/MPS_nsys}"
CELLS="${CELLS:-20}"; ITERS="${ITERS:-50}"; AI_DUR="${AI_DUR:-1800}"; NEURALRX_WAIT="${NEURALRX_WAIT:-75}"
mkdir -p "$RESULTS_DIR"; LOG="$RESULTS_DIR/mps_nsys.log"
ts(){ date '+%H:%M:%S'; }; log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
PIDS="$RESULTS_DIR/ai_pids"; : > "$PIDS"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-$HANDOFF/mps_pipe_${SLURM_JOB_ID:-$$}}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-$HANDOFF/mps_log_${SLURM_JOB_ID:-$$}}"
start_mps(){ log "MPS start"; mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"; nvidia-cuda-mps-control -d; sleep 3; }
stop_mps(){ log "MPS stop"; echo quit | nvidia-cuda-mps-control 2>/dev/null||true; sleep 2; rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"; }

profile_l1_nsys(){
  local label=$1
  log "  NSYS [$label]"
  shifter --image="$IMG" --volume="$AERIAL_REPO:/opt/nvidia/cuBB" --volume="$SCRIPTS_DIR:/scripts" \
    --volume="$RESULTS_DIR:/out" --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src --env=RESULTS_DIR=/out \
    --env=CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" --workdir=/scripts \
    bash -c "nsys profile --trace=cuda --output=/out/${label} --force-overwrite=true --stats=false \
             python3 real_l1.py ${label} $CELLS $ITERS" 2>&1 | tail -2 | tee -a "$LOG"
}
ai_bg_venv(){ local tag=$1 script=$2; shift 2
  ( shifter --image="$IMG" --volume="$AI_DIR:/experiments" --env=HF_HOME="$HF_HOME" \
      --env=CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" \
      "$VENV/bin/python" "/experiments/$script" 0 "$AI_DUR" "$@" ) > "$RESULTS_DIR/${tag}_ai.log" 2>&1 &
  echo $! >> "$PIDS"; log "  AI(venv) [$tag] pid=$!"; }
ai_bg_base(){ local tag=$1 script=$2; shift 2
  ( shifter --image="$IMG" --volume="$AERIAL_REPO:/opt/nvidia/cuBB" --volume="$AI_DIR:/experiments" \
      --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src --env=cuBB_SDK=/opt/nvidia/cuBB \
      --env=CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" \
      python3 "/experiments/$script" 0 "$AI_DUR" "$@" ) > "$RESULTS_DIR/${tag}_ai.log" 2>&1 &
  echo $! >> "$PIDS"; log "  AI(base) [$tag] pid=$!"; }
kill_all_ai(){ while read -r p; do [[ -n "$p" ]] && kill -TERM "$p" 2>/dev/null; done < "$PIDS"; : > "$PIDS"; sleep 6; }

source "$HANDOFF/nsight_conditions.sh"
CONDS="${CONDS:-$CONDS_FULL}"
trap 'stop_mps; kill_all_ai' EXIT INT TERM
start_mps
log "===== MPS+NSYS no-MIG (CELLS=$CELLS ITERS=$ITERS) conds='$CONDS' ====="
for c in $CONDS; do
  log "=== $c ==="
  launch_condition "$c"
  s=$(settle_for "$c"); [ "$s" -gt 0 ] && sleep "$s"
  profile_l1_nsys "MPSnsys_$c"
  kill_all_ai
done
stop_mps; trap - EXIT INT TERM
log "===== DONE: $(ls "$RESULTS_DIR"/MPSnsys_*.nsys-rep 2>/dev/null|wc -l) nsys-rep ====="
