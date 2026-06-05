#!/usr/bin/env bash
# F equivalent on no-MIG (Perlmutter) — full GPU L1 + AI background
# 우선순위 1: paper에 필수 비교 데이터
#
# 사용:
#   salloc -N 1 -C gpu -G 1 -t 4:00:00 -q regular -A <your_account>
#   bash 02_F_saturation_nomig.sh

set -uo pipefail

# Config — Setup script 결과로 채워야 함
AERIAL_IMAGE="${AERIAL_IMAGE:-nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb}"
AERIAL_REPO="${AERIAL_REPO:-$SCRATCH/aerial-cuda-accelerated-ran}"
REPO="${REPO:-$SCRATCH/airan_cloudlab}"
RESULTS_DIR="${RESULTS_DIR:-$REPO/results/perlmutter_nomig/F_nomig}"

# AI workload directory (in the cloned airan_cloudlab repo)
AI_WORKLOAD_DIR="$REPO/cloudlab_aerial/experiments"
SCRIPTS_DIR="$REPO/cloudlab_aerial"

CELLS=20
ITERS=100
N=5
AI_DUR=60

mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/F_nomig.log"
ts(){ date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

# Get full GPU UUID (no MIG)
GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
log "Full GPU UUID: $GPU_UUID"

# Common Shifter command builder
shifter_run() {
    shifter --image="$AERIAL_IMAGE" \
        --volume="$AERIAL_REPO:/opt/nvidia/cuBB" \
        --volume="$SCRIPTS_DIR:/scripts" \
        --volume="$AI_WORKLOAD_DIR:/workspace/AIRAN_Changjong/experiments" \
        --volume="$RESULTS_DIR:/out" \
        --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
        --env=RESULTS_DIR=/out \
        --workdir=/scripts \
        "$@"
}

# Start AI workload in background (no Docker -d equivalent, so use & and PID tracking)
start_ai_bg() {
    local tag=$1 script=$2 ai_dur=$3
    shift 3
    local pidfile="$RESULTS_DIR/ai_${tag}.pid"
    local logfile="$RESULTS_DIR/${tag}_ai.log"

    # nohup so it survives this shell, & for background
    nohup shifter_run python3 "/workspace/AIRAN_Changjong/experiments/$script" 0 "$ai_dur" "$@" \
        > "$logfile" 2>&1 &
    echo $! > "$pidfile"
    log "  Started AI [$tag] pid=$(cat $pidfile)"
}

kill_all_ai() {
    for pidfile in "$RESULTS_DIR"/ai_*.pid; do
        [[ -f $pidfile ]] || continue
        pid=$(cat "$pidfile")
        kill -TERM $pid 2>/dev/null || true
        rm -f "$pidfile"
    done
    sleep 2
}

# Profile L1 — runs N iterations
profile_l1() {
    local label=$1
    for i in $(seq 1 $N); do
        log "    L1 [$label] run $i/$N"
        shifter_run \
            bash -c "nsys profile --trace=cuda \
                       --output=/out/${label}_run${i} \
                       --force-overwrite=true --stats=false \
                       python3 real_l1.py ${label}_run${i} $CELLS $ITERS" \
            2>&1 | tail -2 | tee -a "$LOG"
    done
}

log "===== Perlmutter no-MIG F equivalent =====
GPU: $GPU_UUID
CELLS=$CELLS ITERS=$ITERS N=$N
Results: $RESULTS_DIR
"

# ----------------------------------------------------------
# F_0: L1 alone baseline (n=10)
# ----------------------------------------------------------
log "=== F_0 alone baseline (n=10) ==="
N=10 profile_l1 "F_0_alone"

# ----------------------------------------------------------
# F_B_D2D: large D2D memcpy
# ----------------------------------------------------------
log "=== F_B_D2D 256MB streams=4 ==="
start_ai_bg "memcpy_d2d" "run_memcpy_massive.py" $AI_DUR \
    --env=SIZE_MB=256 --env=N_STREAMS=4 --env=KIND=D2D
sleep 5
profile_l1 "F_B_D2D_256MB_str4"
kill_all_ai

# ----------------------------------------------------------
# F_C_H2D: large H2D memcpy
# ----------------------------------------------------------
log "=== F_C_H2D 256MB streams=4 ==="
start_ai_bg "memcpy_h2d" "run_memcpy_massive.py" $AI_DUR \
    --env=SIZE_MB=256 --env=N_STREAMS=4 --env=KIND=H2D
sleep 5
profile_l1 "F_C_H2D_256MB_str4"
kill_all_ai

# ----------------------------------------------------------
# F_D_GEMM
# ----------------------------------------------------------
log "=== F_D_GEMM DIM=4096 ==="
start_ai_bg "gemm" "run_gemm_massive.py" $AI_DUR --env=DIM=4096
sleep 5
profile_l1 "F_D_GEMM_4096"
kill_all_ai

# ----------------------------------------------------------
# F_E intensity sweeps - core AI workloads
# ----------------------------------------------------------
log "=== F_E chanpred BATCH=64 ==="
start_ai_bg "chanpred" "run_channel_prediction.py" $AI_DUR --env=BATCH_SIZE=64
sleep 8
profile_l1 "F_E_chanpred_b64"
kill_all_ai

log "=== F_E resnet BATCH=64 fp16 ==="
start_ai_bg "resnet" "run_resnet_stress.py" $AI_DUR 64 fp16
sleep 8
profile_l1 "F_E_resnet_b64"
kill_all_ai

log "=== F_E NeuralRx (in-line PHY-AI - critical condition) ==="
start_ai_bg "neuralrx" "run_neural_rx_stress.py" $AI_DUR
sleep 30  # TRT engine build time
profile_l1 "F_E_neuralrx"
kill_all_ai

log "=== F_E qwen_small ==="
start_ai_bg "qwen" "run_qwen_small_stress.py" $AI_DUR
sleep 8
profile_l1 "F_E_qwen_small"
kill_all_ai

log "=== F_E forecaster DIM=384 ==="
start_ai_bg "forecaster" "run_traffic_forecaster.py" $AI_DUR 64 384
sleep 8
profile_l1 "F_E_forecaster_d384"
kill_all_ai

log "=== F_E xapp ==="
start_ai_bg "xapp" "run_xapp_anomaly.py" $AI_DUR
sleep 8
profile_l1 "F_E_xapp"
kill_all_ai

log "=== F_E sat_compute ==="
start_ai_bg "sat_compute" "run_realistic_ai_stress.py" $AI_DUR
sleep 8
profile_l1 "F_E_sat_compute"
kill_all_ai

log "=== F_E sat_hbm ==="
start_ai_bg "sat_hbm" "run_hbm_stress.py" $AI_DUR
sleep 8
profile_l1 "F_E_sat_hbm"
kill_all_ai

# ----------------------------------------------------------
# F_F: multi-AI stacking
# ----------------------------------------------------------
log "=== F_F 4 x chanpred stacked ==="
for k in 1 2 3 4; do
    start_ai_bg "chanpred_x4_inst$k" "run_channel_prediction.py" $AI_DUR
done
sleep 12
profile_l1 "F_F_stack_chanpred_x4"
kill_all_ai

log "=== F_F 2 x ResNet stacked ==="
for k in 1 2; do
    start_ai_bg "resnet_x2_inst$k" "run_resnet_stress.py" $AI_DUR 64 fp16
done
sleep 12
profile_l1 "F_F_stack_resnet_x2"
kill_all_ai

# ----------------------------------------------------------
# F_G: kitchen sink
# ----------------------------------------------------------
log "=== F_G kitchen sink (chanpred + memcpy + GEMM) ==="
start_ai_bg "kitchen_chanpred" "run_channel_prediction.py" $AI_DUR
start_ai_bg "kitchen_memcpy"   "run_memcpy_massive.py" $AI_DUR --env=SIZE_MB=128 --env=N_STREAMS=4 --env=KIND=D2D
start_ai_bg "kitchen_gemm"     "run_gemm_massive.py" $AI_DUR --env=DIM=2048
sleep 12
profile_l1 "F_G_kitchen"
kill_all_ai

log "===== F-equivalent on no-MIG DONE ====="
log "Total captures: $(ls $RESULTS_DIR/*.nsys-rep 2>/dev/null | wc -l)"
