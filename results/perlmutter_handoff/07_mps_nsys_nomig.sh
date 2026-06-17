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

trap 'stop_mps; kill_all_ai' EXIT INT TERM
start_mps
log "===== MPS+NSYS no-MIG (CELLS=$CELLS ITERS=$ITERS) ====="

log "=== alone ==="; profile_l1_nsys "MPSnsys_alone"
log "=== chanpred ==="; ai_bg_venv chanpred run_channel_prediction.py; sleep 10; profile_l1_nsys "MPSnsys_chanpred"; kill_all_ai
log "=== resnet ==="; ai_bg_venv resnet run_resnet_stress.py 64 fp16; sleep 10; profile_l1_nsys "MPSnsys_resnet"; kill_all_ai
log "=== neuralrx ==="; ai_bg_base neuralrx run_neural_rx_stress.py; sleep "$NEURALRX_WAIT"; profile_l1_nsys "MPSnsys_neuralrx"; kill_all_ai
log "=== qwen ==="; ai_bg_venv qwen run_qwen_small_stress.py; sleep 15; profile_l1_nsys "MPSnsys_qwen"; kill_all_ai
log "=== sat_hbm ==="; ai_bg_venv sat_hbm run_hbm_stress.py 16; sleep 10; profile_l1_nsys "MPSnsys_sat_hbm"; kill_all_ai
log "=== forecaster ==="; ai_bg_venv forecaster run_traffic_forecaster.py 64 384; sleep 10; profile_l1_nsys "MPSnsys_forecaster"; kill_all_ai

stop_mps; trap - EXIT INT TERM
log "===== DONE: $(ls "$RESULTS_DIR"/MPSnsys_*.nsys-rep 2>/dev/null|wc -l) nsys-rep ====="
