#!/usr/bin/env bash
# NCU hardware counter measurement on no-MIG — DRAM throughput / L2 / SM
# 우선순위 3
#
# CloudLab MIG NCU 결과: L1 DRAM throughput 11~12% peak. 우리 paper의 핵심 evidence.
# Perlmutter no-MIG에서 같은 measurement → MIG 격리가 NCU 측 metric에 어떻게 영향 주는지 비교

set -uo pipefail

AERIAL_IMAGE="${AERIAL_IMAGE:-nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb}"
AERIAL_REPO="${AERIAL_REPO:-$SCRATCH/aerial-cuda-accelerated-ran}"
REPO="${REPO:-$SCRATCH/airan_cloudlab}"
RESULTS_DIR="${RESULTS_DIR:-$REPO/results/perlmutter_nomig/NCU_nomig}"
AI_WORKLOAD_DIR="$REPO/cloudlab_aerial/experiments"
SCRIPTS_DIR="$REPO/cloudlab_aerial"

CELLS=20
ITERS=3   # NCU replay-mode은 매우 느림 → iters 작게

mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/NCU.log"
ts(){ date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

NCU_METRICS="dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active,launch__waves_per_multiprocessor"

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

start_ai_bg() {
    local tag=$1 script=$2 ai_dur=$3; shift 3
    nohup shifter_run python3 "/workspace/AIRAN_Changjong/experiments/$script" 0 "$ai_dur" "$@" \
        > "$RESULTS_DIR/${tag}_ai.log" 2>&1 &
    echo $! > "$RESULTS_DIR/ai_${tag}.pid"
}
kill_all_ai() {
    for pidfile in "$RESULTS_DIR"/ai_*.pid; do
        [[ -f $pidfile ]] || continue
        kill -TERM $(cat "$pidfile") 2>/dev/null || true
        rm -f "$pidfile"
    done
    sleep 2
}

profile_l1_ncu() {
    local label=$1
    log "  NCU [$label] (replay-mode, slow)"
    shifter_run \
        bash -c "ncu --target-processes all \
                     --replay-mode kernel \
                     --clock-control none \
                     --metrics $NCU_METRICS \
                     --csv --log-file /out/${label}.csv \
                     python3 real_l1.py ${label} $CELLS $ITERS" \
        2>&1 | tail -3 | tee -a "$LOG"
}

log "===== Perlmutter no-MIG NCU ====="
log "GPU: $(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)"

log "=== NCU_0 alone (baseline DRAM/L2 metrics) ==="
profile_l1_ncu "NCU_0_alone"

log "=== NCU_NeuralRx (critical comparison) ==="
start_ai_bg "neuralrx" "run_neural_rx_stress.py" 300
sleep 30
profile_l1_ncu "NCU_neuralrx"
kill_all_ai

log "=== NCU_ResNet ==="
start_ai_bg "resnet" "run_resnet_stress.py" 300 64 fp16
sleep 10
profile_l1_ncu "NCU_resnet"
kill_all_ai

log "=== NCU_chanpred (safe AI control) ==="
start_ai_bg "chanpred" "run_channel_prediction.py" 300
sleep 10
profile_l1_ncu "NCU_chanpred"
kill_all_ai

log "=== NCU_sat_hbm ==="
start_ai_bg "sat_hbm" "run_hbm_stress.py" 300
sleep 10
profile_l1_ncu "NCU_sat_hbm"
kill_all_ai

log "=== NCU_sat_compute ==="
start_ai_bg "sat_compute" "run_realistic_ai_stress.py" 300
sleep 10
profile_l1_ncu "NCU_sat_compute"
kill_all_ai

log "===== NCU no-MIG DONE ====="
log "Outputs: $(ls $RESULTS_DIR/*.csv 2>/dev/null | wc -l) CSV files"
