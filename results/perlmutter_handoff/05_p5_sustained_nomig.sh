#!/usr/bin/env bash
# 5-minute sustained on no-MIG — equivalent to CloudLab P5.
# Priority 5. Long-running (~2h); submit as a batch job.
#
# Each scenario: L1 runs a long ITERS sweep (~5 min) x N, with the AI workload
# co-located the whole time, to capture sustained-load behaviour (drift, thermal,
# long-tail) rather than the short bursts of priority 1.
#
# Adapted for the validated environment (see 02_F_saturation_nomig.sh).

set -uo pipefail

IMG="${AERIAL_IMAGE:-nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb}"
AERIAL_REPO="${AERIAL_REPO:-/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran}"
REPO="${REPO:-/pscratch/sd/s/sgkim/kcj/airan_cloudlab}"
HANDOFF="${HANDOFF:-$REPO/results/perlmutter_handoff}"
VENV="${VENV:-$HANDOFF/airan_venv}"
HF_HOME="${HF_HOME:-$HANDOFF/hf_cache}"
SCRIPTS_DIR="$REPO/scripts_for_node/cloudlab_aerial"
AI_DIR="$REPO/scripts_for_node/experiments"
RESULTS_DIR="${RESULTS_DIR:-$HANDOFF/perlmutter_nomig/P5_nomig}"

CELLS="${CELLS:-20}"
ITERS="${ITERS:-7500}"       # high iteration cap; the real bound is MAX_SECONDS below
N="${N:-2}"
# True 5-min sustained: bound each L1 run by wall time, not iteration count, so a
# heavily-contended run still ends in ~5 min instead of ballooning to 30-60 min.
MAX_SECONDS="${MAX_SECONDS:-300}"
AI_DUR="${AI_DUR:-3600}"     # AI runs until killed; sustained runs are long
NEURALRX_WAIT="${NEURALRX_WAIT:-75}"

mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/P5_nomig.log"
ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
PIDS="$RESULTS_DIR/ai_pids"; : > "$PIDS"

profile_l1_sustained() {
    local label=$1
    for i in $(seq 1 "$N"); do
        log "  Sustained [$label] run $i/$N (~5 min)"
        shifter --image="$IMG" \
            --volume="$AERIAL_REPO:/opt/nvidia/cuBB" --volume="$SCRIPTS_DIR:/scripts" \
            --volume="$RESULTS_DIR:/out" \
            --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src --env=RESULTS_DIR=/out \
            --env=MAX_SECONDS="$MAX_SECONDS" \
            --workdir=/scripts \
            python3 real_l1.py "P5_${label}_run${i}" "$CELLS" "$ITERS" 2>&1 | tail -1 | tee -a "$LOG"
    done
}

ai_bg_venv() {
    local tag=$1 envs=$2 script=$3; shift 3
    ( shifter --image="$IMG" --volume="$AI_DIR:/experiments" \
        --env=HF_HOME="$HF_HOME" $envs \
        "$VENV/bin/python" "/experiments/$script" 0 "$AI_DUR" "$@" ) \
        > "$RESULTS_DIR/${tag}_ai.log" 2>&1 &
    echo $! >> "$PIDS"; log "  AI(venv) [$tag] pid=$!"
}
ai_bg_base() {
    local tag=$1 script=$2; shift 2
    ( shifter --image="$IMG" \
        --volume="$AERIAL_REPO:/opt/nvidia/cuBB" --volume="$AI_DIR:/experiments" \
        --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src --env=cuBB_SDK=/opt/nvidia/cuBB \
        python3 "/experiments/$script" 0 "$AI_DUR" "$@" ) \
        > "$RESULTS_DIR/${tag}_ai.log" 2>&1 &
    echo $! >> "$PIDS"; log "  AI(base) [$tag] pid=$!"
}
kill_all_ai() {
    while read -r pid; do [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null; done < "$PIDS"
    : > "$PIDS"; sleep 4
}

log "===== Perlmutter no-MIG P5 sustained (~2h) ====="
log "GPU: $(nvidia-smi --query-gpu=name,mig.mode.current --format=csv,noheader | head -1)"
log "CELLS=$CELLS ITERS=$ITERS N=$N | Results: $RESULTS_DIR"

WORKLOADS="${WORKLOADS:-alone qwen_small sat_compute sat_hbm chanpred xapp neuralrx resnet forecaster}"
for WORKLOAD in $WORKLOADS; do
    log "===== Scenario: $WORKLOAD ====="
    case $WORKLOAD in
        alone)        ;;
        qwen_small)   ai_bg_venv qwen "" run_qwen_small_stress.py;        sleep 15 ;;
        sat_compute)  ai_bg_venv satc "" run_realistic_ai_stress.py matmul 0.8; sleep 10 ;;
        sat_hbm)      ai_bg_venv sath "" run_hbm_stress.py 16;            sleep 10 ;;
        chanpred)     ai_bg_venv cp "" run_channel_prediction.py;         sleep 10 ;;
        xapp)         ai_bg_venv xa "" run_xapp_anomaly.py;               sleep 10 ;;
        neuralrx)     ai_bg_base nrx run_neural_rx_stress.py;             sleep "$NEURALRX_WAIT" ;;
        resnet)       ai_bg_venv rn "" run_resnet_stress.py 64 fp16;      sleep 10 ;;
        forecaster)   ai_bg_venv fo "" run_traffic_forecaster.py 64 384;  sleep 10 ;;
    esac
    profile_l1_sustained "$WORKLOAD"
    kill_all_ai
done

log "===== P5 sustained no-MIG DONE ====="
log "JSON captures: $(ls "$RESULTS_DIR"/realL1_P5_*.json 2>/dev/null | wc -l)"
