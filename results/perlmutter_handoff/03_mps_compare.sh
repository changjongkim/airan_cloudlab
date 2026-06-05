#!/usr/bin/env bash
# MPS comparison on no-MIG — same F conditions measured with CUDA MPS enabled.
# Priority 2.
#
# MPS = CUDA Multi-Process Service: processes share the GPU via concurrent
# execution rather than time-slicing. Comparing default time-slice (02) vs MPS
# (this) quantifies how much MPS recovers — and where MIG's isolation still wins.
#
# Adapted for the validated Perlmutter environment (see 02_F_saturation_nomig.sh):
#   - MPS daemon runs on the node (host); pipe/log dirs on $SCRATCH are
#     auto-mounted into shifter at the same absolute path so both the base
#     (L1/neuralrx) and venv (torch) containers connect to the same server.
#   - CUDA_MPS_PIPE_DIRECTORY propagated into every CUDA process.

set -uo pipefail

IMG="${AERIAL_IMAGE:-nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb}"
AERIAL_REPO="${AERIAL_REPO:-/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran}"
REPO="${REPO:-/pscratch/sd/s/sgkim/kcj/airan_cloudlab}"
HANDOFF="${HANDOFF:-$REPO/results/perlmutter_handoff}"
VENV="${VENV:-$HANDOFF/airan_venv}"
HF_HOME="${HF_HOME:-$HANDOFF/hf_cache}"
SCRIPTS_DIR="$REPO/scripts_for_node/cloudlab_aerial"
AI_DIR="$REPO/scripts_for_node/experiments"
RESULTS_DIR="${RESULTS_DIR:-$HANDOFF/perlmutter_nomig/F_nomig_mps}"

CELLS="${CELLS:-20}"
ITERS="${ITERS:-100}"
N="${N:-5}"
AI_DUR="${AI_DUR:-1800}"
NEURALRX_WAIT="${NEURALRX_WAIT:-75}"

mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/F_nomig_mps.log"
ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
PIDS="$RESULTS_DIR/ai_pids"; : > "$PIDS"

# MPS server dirs (under $SCRATCH -> auto-mounted into shifter at same path).
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-$HANDOFF/mps_pipe_${SLURM_JOB_ID:-$$}}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-$HANDOFF/mps_log_${SLURM_JOB_ID:-$$}}"

start_mps() {
    log "Starting MPS server (pipe=$CUDA_MPS_PIPE_DIRECTORY)..."
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d
    sleep 3
    echo get_default_active_thread_percentage | nvidia-cuda-mps-control 2>&1 | head -2 | tee -a "$LOG"
}
stop_mps() {
    log "Stopping MPS server..."
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    sleep 2
    rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
}

GPU_NAME=$(nvidia-smi --query-gpu=name,mig.mode.current --format=csv,noheader 2>/dev/null | head -1)

# L1 in base container, MPS env propagated.
profile_l1() {
    local label=$1 reps=${2:-$N}
    for i in $(seq 1 "$reps"); do
        log "    MPS L1 [$label] run $i/$reps"
        shifter --image="$IMG" \
            --volume="$AERIAL_REPO:/opt/nvidia/cuBB" --volume="$SCRIPTS_DIR:/scripts" \
            --volume="$RESULTS_DIR:/out" \
            --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src --env=RESULTS_DIR=/out \
            --env=CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" \
            --workdir=/scripts \
            python3 real_l1.py "MPS_${label}_run${i}" "$CELLS" "$ITERS" 2>&1 | tail -1 | tee -a "$LOG"
    done
}

ai_bg_venv() {
    local tag=$1 envs=$2 script=$3; shift 3
    ( shifter --image="$IMG" --volume="$AI_DIR:/experiments" \
        --env=HF_HOME="$HF_HOME" --env=CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" $envs \
        "$VENV/bin/python" "/experiments/$script" 0 "$AI_DUR" "$@" ) \
        > "$RESULTS_DIR/${tag}_ai.log" 2>&1 &
    echo $! >> "$PIDS"; log "  AI(venv) [$tag] pid=$!"
}
ai_bg_base() {
    local tag=$1 script=$2; shift 2
    ( shifter --image="$IMG" \
        --volume="$AERIAL_REPO:/opt/nvidia/cuBB" --volume="$AI_DIR:/experiments" \
        --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src --env=cuBB_SDK=/opt/nvidia/cuBB \
        --env=CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" \
        python3 "/experiments/$script" 0 "$AI_DUR" "$@" ) \
        > "$RESULTS_DIR/${tag}_ai.log" 2>&1 &
    echo $! >> "$PIDS"; log "  AI(base) [$tag] pid=$!"
}
kill_all_ai() {
    while read -r pid; do [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null; done < "$PIDS"
    : > "$PIDS"; sleep 6   # let MPS release the dead client before next condition
}

trap 'stop_mps; kill_all_ai' EXIT INT TERM
start_mps

log "===== MPS no-MIG F-subset ====="
log "GPU: $GPU_NAME | CELLS=$CELLS ITERS=$ITERS N=$N AI_DUR=$AI_DUR"
log "Results: $RESULTS_DIR"

log "=== MPS F_0 alone baseline (n=10) ==="
profile_l1 "F_0_alone" 10

log "=== MPS F_E chanpred ==="
ai_bg_venv chanpred "" run_channel_prediction.py
sleep 10; profile_l1 "F_E_chanpred_b64"; kill_all_ai

log "=== MPS F_E resnet ==="
ai_bg_venv resnet "" run_resnet_stress.py 64 fp16
sleep 10; profile_l1 "F_E_resnet_b64"; kill_all_ai

log "=== MPS F_E NeuralRx (critical) ==="
ai_bg_base neuralrx run_neural_rx_stress.py
sleep "$NEURALRX_WAIT"; profile_l1 "F_E_neuralrx"; kill_all_ai

log "=== MPS F_E qwen_small ==="
ai_bg_venv qwen "" run_qwen_small_stress.py
sleep 15; profile_l1 "F_E_qwen_small"; kill_all_ai

log "=== MPS F_E sat_hbm ==="
ai_bg_venv sat_hbm "" run_hbm_stress.py 16
sleep 10; profile_l1 "F_E_sat_hbm"; kill_all_ai

log "=== MPS F_E forecaster ==="
ai_bg_venv forecaster "" run_traffic_forecaster.py 64 384
sleep 10; profile_l1 "F_E_forecaster_d384"; kill_all_ai

stop_mps
trap - EXIT INT TERM
log "===== MPS no-MIG comparison DONE ====="
log "JSON captures: $(ls "$RESULTS_DIR"/realL1_MPS_*.json 2>/dev/null | wc -l)"
