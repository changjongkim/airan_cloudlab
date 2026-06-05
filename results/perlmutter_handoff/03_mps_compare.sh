#!/usr/bin/env bash
# MPS comparison on no-MIG — 같은 F 조건을 MPS 켜고 측정
# 우선순위 2
#
# MPS는 CUDA Multi-Process Service: 두 프로세스가 GPU를 시간 분할이 아닌 동시 실행
# Default time-slice (02 script) vs MPS (이 script) 비교로 MIG의 격리 가치 정량화

set -uo pipefail

AERIAL_IMAGE="${AERIAL_IMAGE:-nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb}"
AERIAL_REPO="${AERIAL_REPO:-$SCRATCH/aerial-cuda-accelerated-ran}"
REPO="${REPO:-$SCRATCH/airan_cloudlab}"
RESULTS_DIR="${RESULTS_DIR:-$REPO/results/perlmutter_nomig/F_nomig_mps}"
AI_WORKLOAD_DIR="$REPO/cloudlab_aerial/experiments"
SCRIPTS_DIR="$REPO/cloudlab_aerial"

CELLS=20
ITERS=100
N=5
AI_DUR=60

mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/F_nomig_mps.log"
ts(){ date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
log "Full GPU UUID: $GPU_UUID"

# --- MPS setup ---
start_mps() {
    log "Starting MPS server..."
    # Perlmutter doesn't allow nvidia-smi -c (exclusive). MPS works in DEFAULT mode too.
    export CUDA_MPS_PIPE_DIRECTORY="$SCRATCH/mps_pipe_$$"
    export CUDA_MPS_LOG_DIRECTORY="$SCRATCH/mps_log_$$"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    nvidia-cuda-mps-control -d
    sleep 3
    # Verify
    echo get_default_active_thread_percentage | nvidia-cuda-mps-control 2>&1 | head -3
}

stop_mps() {
    log "Stopping MPS server..."
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
}

# Shifter run with MPS env propagation
shifter_run() {
    shifter --image="$AERIAL_IMAGE" \
        --volume="$AERIAL_REPO:/opt/nvidia/cuBB" \
        --volume="$SCRIPTS_DIR:/scripts" \
        --volume="$AI_WORKLOAD_DIR:/workspace/AIRAN_Changjong/experiments" \
        --volume="$RESULTS_DIR:/out" \
        --volume="$CUDA_MPS_PIPE_DIRECTORY:$CUDA_MPS_PIPE_DIRECTORY" \
        --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
        --env=RESULTS_DIR=/out \
        --env=CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" \
        --workdir=/scripts \
        "$@"
}

start_ai_bg() {
    local tag=$1 script=$2 ai_dur=$3
    shift 3
    local pidfile="$RESULTS_DIR/ai_${tag}.pid"
    local logfile="$RESULTS_DIR/${tag}_ai.log"
    nohup shifter_run python3 "/workspace/AIRAN_Changjong/experiments/$script" 0 "$ai_dur" "$@" \
        > "$logfile" 2>&1 &
    echo $! > "$pidfile"
}
kill_all_ai() {
    for pidfile in "$RESULTS_DIR"/ai_*.pid; do
        [[ -f $pidfile ]] || continue
        kill -TERM $(cat "$pidfile") 2>/dev/null || true
        rm -f "$pidfile"
    done
    sleep 2
}
profile_l1() {
    local label=$1
    for i in $(seq 1 $N); do
        log "    L1 [$label] run $i/$N"
        shifter_run \
            bash -c "nsys profile --trace=cuda \
                       --output=/out/MPS_${label}_run${i} \
                       --force-overwrite=true --stats=false \
                       python3 real_l1.py MPS_${label}_run${i} $CELLS $ITERS" \
            2>&1 | tail -2 | tee -a "$LOG"
    done
}

# ----------------------------------------------------------
# Start MPS, run subset of F conditions
# ----------------------------------------------------------
trap 'stop_mps; kill_all_ai' EXIT INT TERM

start_mps

log "===== MPS no-MIG F-subset ====="

log "=== MPS_F_0 alone baseline ==="
N=10 profile_l1 "F_0_alone"

log "=== MPS_F_E chanpred ==="
start_ai_bg "chanpred" "run_channel_prediction.py" $AI_DUR --env=BATCH_SIZE=64
sleep 8
profile_l1 "F_E_chanpred_b64"
kill_all_ai

log "=== MPS_F_E resnet ==="
start_ai_bg "resnet" "run_resnet_stress.py" $AI_DUR 64 fp16
sleep 8
profile_l1 "F_E_resnet_b64"
kill_all_ai

log "=== MPS_F_E NeuralRx (critical) ==="
start_ai_bg "neuralrx" "run_neural_rx_stress.py" $AI_DUR
sleep 30
profile_l1 "F_E_neuralrx"
kill_all_ai

log "=== MPS_F_E qwen ==="
start_ai_bg "qwen" "run_qwen_small_stress.py" $AI_DUR
sleep 8
profile_l1 "F_E_qwen_small"
kill_all_ai

log "=== MPS_F_E sat_hbm ==="
start_ai_bg "sat_hbm" "run_hbm_stress.py" $AI_DUR
sleep 8
profile_l1 "F_E_sat_hbm"
kill_all_ai

log "=== MPS_F_E forecaster ==="
start_ai_bg "forecaster" "run_traffic_forecaster.py" $AI_DUR 64 384
sleep 8
profile_l1 "F_E_forecaster_d384"
kill_all_ai

stop_mps
trap - EXIT INT TERM

log "===== MPS no-MIG comparison DONE ====="
