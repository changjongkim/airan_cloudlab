#!/usr/bin/env bash
# Per-call NSYS analysis on no-MIG — Priority 4.
#
# Focused nsys capture on the critical conditions (L1 alone, L1+NeuralRx,
# L1+ResNet) so we can break down memcpy direction and per-call kernel/copy
# duration distributions — comparable to CloudLab §16.1 (60KB memcpy 4.2->14.3us).
# Not the full matrix (nsys overhead + huge files); just the conditions that matter.
#
# nsys uses CUDA tracing (not perf counters), so it is NOT blocked by DCGM.
# Convert the .nsys-rep to sqlite afterwards for analysis:
#   nsys stats --report cuda_gpu_mem_time_sum,cuda_gpu_kern_sum <file>.nsys-rep

set -uo pipefail

IMG="${AERIAL_IMAGE:-nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb}"
AERIAL_REPO="${AERIAL_REPO:-/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran}"
REPO="${REPO:-/pscratch/sd/s/sgkim/kcj/airan_cloudlab}"
HANDOFF="${HANDOFF:-$REPO/results/perlmutter_handoff}"
VENV="${VENV:-$HANDOFF/airan_venv}"
SCRIPTS_DIR="$REPO/scripts_for_node/cloudlab_aerial"
AI_DIR="$REPO/scripts_for_node/experiments"
RESULTS_DIR="${RESULTS_DIR:-$HANDOFF/perlmutter_nomig/nsys_nomig}"

CELLS="${CELLS:-20}"
ITERS="${ITERS:-50}"         # enough calls for per-call distributions; nsys adds overhead
AI_DUR="${AI_DUR:-1800}"
NEURALRX_WAIT="${NEURALRX_WAIT:-75}"

mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/nsys.log"
ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
PIDS="$RESULTS_DIR/ai_pids"; : > "$PIDS"

profile_l1_nsys() {
    local label=$1
    log "  NSYS [$label]"
    shifter --image="$IMG" \
        --volume="$AERIAL_REPO:/opt/nvidia/cuBB" --volume="$SCRIPTS_DIR:/scripts" \
        --volume="$RESULTS_DIR:/out" \
        --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src --env=RESULTS_DIR=/out \
        --workdir=/scripts \
        bash -c "nsys profile --trace=cuda --output=/out/${label} \
                     --force-overwrite=true --stats=false \
                     python3 real_l1.py ${label} $CELLS $ITERS" 2>&1 | tail -3 | tee -a "$LOG"
}

ai_bg_venv() {
    local tag=$1 script=$2; shift 2
    ( shifter --image="$IMG" --volume="$AI_DIR:/experiments" \
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

log "===== Perlmutter no-MIG per-call NSYS ====="
log "GPU: $(nvidia-smi --query-gpu=name,mig.mode.current --format=csv,noheader | head -1)"
log "CELLS=$CELLS ITERS=$ITERS | Results: $RESULTS_DIR"

log "=== nsys_alone ==="
profile_l1_nsys "nsys_alone"

log "=== nsys_neuralrx (critical) ==="
ai_bg_base neuralrx run_neural_rx_stress.py
sleep "$NEURALRX_WAIT"; profile_l1_nsys "nsys_neuralrx"; kill_all_ai

log "=== nsys_resnet ==="
ai_bg_venv resnet run_resnet_stress.py 64 fp16
sleep 10; profile_l1_nsys "nsys_resnet"; kill_all_ai

log "===== per-call NSYS no-MIG DONE ====="
log "nsys-rep captures: $(ls "$RESULTS_DIR"/nsys_*.nsys-rep 2>/dev/null | wc -l)"
