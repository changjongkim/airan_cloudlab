#!/usr/bin/env bash
# 5-min sustained on no-MIG — CloudLab P5와 동등
# 우선순위 4
# Long-running 이라 SLURM batch job으로 제출 권장 (총 ~2시간)

set -uo pipefail

AERIAL_IMAGE="${AERIAL_IMAGE:-nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb}"
AERIAL_REPO="${AERIAL_REPO:-$SCRATCH/aerial-cuda-accelerated-ran}"
REPO="${REPO:-$SCRATCH/airan_cloudlab}"
RESULTS_DIR="${RESULTS_DIR:-$REPO/results/perlmutter_nomig/P5_nomig}"
AI_WORKLOAD_DIR="$REPO/cloudlab_aerial/experiments"
SCRIPTS_DIR="$REPO/cloudlab_aerial"

# 5분 = 7500 iter at 40ms/iter
CELLS=20
ITERS=7500
N=2

mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/P5_nomig.log"
ts(){ date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

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
    local tag=$1 script=$2; shift 2
    nohup shifter_run python3 "/workspace/AIRAN_Changjong/experiments/$script" 0 600 "$@" \
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

profile_l1_sustained() {
    local label=$1
    for i in $(seq 1 $N); do
        log "  Sustained [$label] run $i/$N (~5 min)"
        local outdir="$RESULTS_DIR/$label"
        mkdir -p "$outdir"
        shifter_run \
            bash -c "python3 real_l1.py P5_${label}_run${i} $CELLS $ITERS > /out/${label}/run_${i}_l1.log 2>&1" \
            || true
        cp "$RESULTS_DIR/${label}/run_${i}_l1.log" "$LOG" 2>/dev/null
    done
}

log "===== Perlmutter no-MIG P5 sustained (~2h) ====="

for WORKLOAD in alone qwen_small sat_compute sat_hbm chanpred xapp neuralrx resnet forecaster; do
    log "===== Scenario: $WORKLOAD ====="
    case $WORKLOAD in
        alone)        ;;
        qwen_small)   start_ai_bg qwen "run_qwen_small_stress.py"; sleep 5 ;;
        sat_compute)  start_ai_bg satc "run_realistic_ai_stress.py"; sleep 5 ;;
        sat_hbm)      start_ai_bg sath "run_hbm_stress.py"; sleep 5 ;;
        chanpred)     start_ai_bg cp "run_channel_prediction.py"; sleep 8 ;;
        xapp)         start_ai_bg xa "run_xapp_anomaly.py"; sleep 8 ;;
        neuralrx)     start_ai_bg nrx "run_neural_rx_stress.py"; sleep 30 ;;
        resnet)       start_ai_bg rn "run_resnet_stress.py" 64 fp16; sleep 8 ;;
        forecaster)   start_ai_bg fo "run_traffic_forecaster.py" 64 384; sleep 8 ;;
    esac

    profile_l1_sustained "$WORKLOAD"
    kill_all_ai
done

log "===== P5 sustained no-MIG DONE ====="
