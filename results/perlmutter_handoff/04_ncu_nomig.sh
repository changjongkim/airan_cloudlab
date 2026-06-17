#!/usr/bin/env bash
# NCU hardware-counter measurement on no-MIG — DRAM throughput / L2 / SM occupancy.
# Priority 3.
#
# CloudLab MIG NCU result: L1 DRAM throughput ~11-12% of peak (core paper evidence).
# Here we measure the same on Perlmutter no-MIG, alone and under contention, to see
# how the absence of MIG isolation moves those hardware metrics.
#
# Adapted for the validated environment (see 02_F_saturation_nomig.sh).
# ncu ships inside the Aerial container (/usr/local/cuda/bin/ncu); --clock-control
# none because clocks can't be locked on these nodes.

set -uo pipefail

IMG="${AERIAL_IMAGE:-nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb}"
AERIAL_REPO="${AERIAL_REPO:-/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran}"
REPO="${REPO:-/pscratch/sd/s/sgkim/kcj/airan_cloudlab}"
HANDOFF="${HANDOFF:-$REPO/results/perlmutter_handoff}"
VENV="${VENV:-$HANDOFF/airan_venv}"
HF_HOME="${HF_HOME:-$HANDOFF/hf_cache}"
SCRIPTS_DIR="$REPO/scripts_for_node/cloudlab_aerial"
AI_DIR="$REPO/scripts_for_node/experiments"
RESULTS_DIR="${RESULTS_DIR:-$HANDOFF/perlmutter_nomig/NCU_nomig}"

CELLS="${CELLS:-20}"
ITERS="${ITERS:-3}"          # NCU kernel-replay is very slow → keep iters tiny
AI_DUR="${AI_DUR:-1800}"     # AI runs until killed (NCU replay can take minutes)
NEURALRX_WAIT="${NEURALRX_WAIT:-75}"

NCU_METRICS="dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active,launch__waves_per_multiprocessor"

mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/NCU.log"
ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
PIDS="$RESULTS_DIR/ai_pids"; : > "$PIDS"

profile_l1_ncu() {
    local label=$1
    log "  NCU [$label] (replay-mode, slow)"
    shifter --image="$IMG" \
        --volume="$AERIAL_REPO:/opt/nvidia/cuBB" --volume="$SCRIPTS_DIR:/scripts" \
        --volume="$RESULTS_DIR:/out" \
        --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src --env=RESULTS_DIR=/out \
        --workdir=/scripts \
        bash -c "ncu --target-processes all --replay-mode kernel --clock-control none \
                     --metrics $NCU_METRICS \
                     --csv --log-file /out/${label}.csv \
                     python3 real_l1.py ${label} $CELLS $ITERS" 2>&1 | tail -3 | tee -a "$LOG"
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

# Condition subset (default all). Resume runs pass NCU_CONDS to skip done ones.
NCU_CONDS="${NCU_CONDS:-alone neuralrx resnet chanpred sat_hbm sat_compute}"
want(){ [[ " $NCU_CONDS " == *" $1 "* ]]; }

log "===== Perlmutter no-MIG NCU ====="
log "GPU: $(nvidia-smi --query-gpu=name,mig.mode.current --format=csv,noheader | head -1)"
log "CELLS=$CELLS ITERS=$ITERS conds='$NCU_CONDS' | Results: $RESULTS_DIR"

if want alone; then
  log "=== NCU_0_alone (baseline DRAM/L2/SM) ==="
  profile_l1_ncu "NCU_0_alone"
fi
if want neuralrx; then
  log "=== NCU_neuralrx (critical) ==="
  ai_bg_base neuralrx run_neural_rx_stress.py
  sleep "$NEURALRX_WAIT"; profile_l1_ncu "NCU_neuralrx"; kill_all_ai
fi
if want resnet; then
  log "=== NCU_resnet ==="
  ai_bg_venv resnet "" run_resnet_stress.py 64 fp16
  sleep 10; profile_l1_ncu "NCU_resnet"; kill_all_ai
fi
if want chanpred; then
  log "=== NCU_chanpred (safe AI control) ==="
  ai_bg_venv chanpred "" run_channel_prediction.py
  sleep 10; profile_l1_ncu "NCU_chanpred"; kill_all_ai
fi
if want sat_hbm; then
  log "=== NCU_sat_hbm ==="
  ai_bg_venv sat_hbm "" run_hbm_stress.py 16
  sleep 10; profile_l1_ncu "NCU_sat_hbm"; kill_all_ai
fi
if want sat_compute; then
  log "=== NCU_sat_compute ==="
  ai_bg_venv sat_compute "" run_realistic_ai_stress.py matmul 0.8
  sleep 10; profile_l1_ncu "NCU_sat_compute"; kill_all_ai
fi

log "===== NCU no-MIG DONE ====="
log "CSV outputs: $(ls "$RESULTS_DIR"/NCU_*.csv 2>/dev/null | wc -l)"
