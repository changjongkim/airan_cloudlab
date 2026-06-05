#!/usr/bin/env bash
# F-equivalent on no-MIG (Perlmutter) — full GPU L1 + AI background.
# Priority 1: core comparison data for the paper.
#
# Adapted for the validated Perlmutter environment:
#   - paths all under perlmutter_handoff
#   - L1 (real_l1.py) + neuralrx run in the base Aerial container (cupy/aerial/TRT)
#   - torch workloads run in the prebuilt venv ($HANDOFF/airan_venv)
#   - shifter mounts use top-level targets (no nested /workspace/.. mount)
#   - nsys profiling disabled by default (USE_NSYS=1 to enable; nsys is in-container)
#
# Usage:
#   salloc -N 1 -C gpu -G 1 -t 4:00:00 -q regular -A m1248_g
#   bash 02_F_saturation_nomig.sh
# or submit via run_all.sbatch.

set -uo pipefail

# ---- Config (validated paths) -----------------------------------------------
IMG="${AERIAL_IMAGE:-nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb}"
AERIAL_REPO="${AERIAL_REPO:-/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran}"
REPO="${REPO:-/pscratch/sd/s/sgkim/kcj/airan_cloudlab}"
HANDOFF="${HANDOFF:-$REPO/results/perlmutter_handoff}"
VENV="${VENV:-$HANDOFF/airan_venv}"
HF_HOME="${HF_HOME:-$HANDOFF/hf_cache}"
SCRIPTS_DIR="$REPO/scripts_for_node/cloudlab_aerial"   # real_l1.py
AI_DIR="$REPO/scripts_for_node/experiments"            # AI workloads
RESULTS_DIR="${RESULTS_DIR:-$HANDOFF/perlmutter_nomig/F_nomig}"

CELLS="${CELLS:-20}"
ITERS="${ITERS:-100}"
N="${N:-5}"
# AI workloads run "until killed": set a long duration so they outlive the L1
# measurement (heavy conditions can take >800s for N runs); kill_all_ai stops
# them after each condition. Keeping this short would let AI die mid-measurement.
AI_DUR="${AI_DUR:-1800}"
USE_NSYS="${USE_NSYS:-0}"

mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/F_nomig.log"
ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
PIDS="$RESULTS_DIR/ai_pids"; : > "$PIDS"

GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -1)
GPU_NAME=$(nvidia-smi --query-gpu=name,mig.mode.current --format=csv,noheader 2>/dev/null | head -1)

# ---- Runtime helpers --------------------------------------------------------
# L1 in base container (optionally under nsys).
profile_l1() {
    local label=$1 reps=${2:-$N}
    for i in $(seq 1 "$reps"); do
        log "    L1 [$label] run $i/$reps"
        if [[ "$USE_NSYS" == "1" ]]; then
            shifter --image="$IMG" \
                --volume="$AERIAL_REPO:/opt/nvidia/cuBB" --volume="$SCRIPTS_DIR:/scripts" \
                --volume="$RESULTS_DIR:/out" \
                --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src --env=RESULTS_DIR=/out \
                --workdir=/scripts \
                bash -c "nsys profile --trace=cuda --output=/out/${label}_run${i} \
                         --force-overwrite=true --stats=false \
                         python3 real_l1.py ${label}_run${i} $CELLS $ITERS" 2>&1 | tail -2 | tee -a "$LOG"
        else
            shifter --image="$IMG" \
                --volume="$AERIAL_REPO:/opt/nvidia/cuBB" --volume="$SCRIPTS_DIR:/scripts" \
                --volume="$RESULTS_DIR:/out" \
                --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src --env=RESULTS_DIR=/out \
                --workdir=/scripts \
                python3 real_l1.py "${label}_run${i}" "$CELLS" "$ITERS" 2>&1 | tail -1 | tee -a "$LOG"
        fi
    done
}

# Background AI in the venv (torch workloads). $2 = extra "--env=K=V" flags.
ai_bg_venv() {
    local tag=$1 envs=$2 script=$3; shift 3
    ( shifter --image="$IMG" --volume="$AI_DIR:/experiments" \
        --env=HF_HOME="$HF_HOME" $envs \
        "$VENV/bin/python" "/experiments/$script" 0 "$AI_DUR" "$@" ) \
        > "$RESULTS_DIR/${tag}_ai.log" 2>&1 &
    echo $! >> "$PIDS"; log "  AI(venv) [$tag] pid=$!"
}

# Background AI in the base container (neuralrx: cupy/aerial/TRT).
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
    : > "$PIDS"; sleep 3
}

log "===== Perlmutter no-MIG F-equivalent ====="
log "GPU: $GPU_NAME ($GPU_UUID)"
log "CELLS=$CELLS ITERS=$ITERS N=$N AI_DUR=$AI_DUR USE_NSYS=$USE_NSYS"
log "Results: $RESULTS_DIR"

# ---- F_0: L1 alone baseline (n=10) -----------------------------------------
log "=== F_0 alone baseline (n=10) ==="
profile_l1 "F_0_alone" 10

# ---- F_B / F_C: memcpy bandwidth -------------------------------------------
log "=== F_B_D2D 256MB streams=4 ==="
ai_bg_venv memcpy_d2d "--env=SIZE_MB=256 --env=N_STREAMS=4 --env=KIND=D2D" run_memcpy_massive.py
sleep 8; profile_l1 "F_B_D2D_256MB_str4"; kill_all_ai

log "=== F_C_H2D 256MB streams=4 ==="
ai_bg_venv memcpy_h2d "--env=SIZE_MB=256 --env=N_STREAMS=4 --env=KIND=H2D" run_memcpy_massive.py
sleep 8; profile_l1 "F_C_H2D_256MB_str4"; kill_all_ai

# ---- F_D: GEMM tensor cores -------------------------------------------------
log "=== F_D_GEMM DIM=4096 ==="
ai_bg_venv gemm "--env=DIM=4096" run_gemm_massive.py
sleep 8; profile_l1 "F_D_GEMM_4096"; kill_all_ai

# ---- F_E: single AI workloads ----------------------------------------------
log "=== F_E chanpred ==="
ai_bg_venv chanpred "" run_channel_prediction.py
sleep 10; profile_l1 "F_E_chanpred_b64"; kill_all_ai

log "=== F_E resnet b64 fp16 ==="
ai_bg_venv resnet "" run_resnet_stress.py 64 fp16
sleep 10; profile_l1 "F_E_resnet_b64"; kill_all_ai

log "=== F_E NeuralRx (in-line PHY-AI — critical) ==="
ai_bg_base neuralrx run_neural_rx_stress.py
sleep "${NEURALRX_WAIT:-75}"   # ONNX -> TRT engine build window
profile_l1 "F_E_neuralrx"; kill_all_ai

log "=== F_E qwen_small ==="
ai_bg_venv qwen "" run_qwen_small_stress.py
sleep 15; profile_l1 "F_E_qwen_small"; kill_all_ai

log "=== F_E forecaster d384 ==="
ai_bg_venv forecaster "" run_traffic_forecaster.py 64 384
sleep 10; profile_l1 "F_E_forecaster_d384"; kill_all_ai

log "=== F_E xapp ==="
ai_bg_venv xapp "" run_xapp_anomaly.py
sleep 10; profile_l1 "F_E_xapp"; kill_all_ai

log "=== F_E sat_compute (realistic_ai matmul) ==="
ai_bg_venv sat_compute "" run_realistic_ai_stress.py matmul 0.8
sleep 10; profile_l1 "F_E_sat_compute"; kill_all_ai

log "=== F_E sat_hbm ==="
ai_bg_venv sat_hbm "" run_hbm_stress.py 16
sleep 10; profile_l1 "F_E_sat_hbm"; kill_all_ai

# ---- F_F: multi-AI stacking -------------------------------------------------
log "=== F_F 4x chanpred ==="
for k in 1 2 3 4; do ai_bg_venv "chanpred_x4_$k" "" run_channel_prediction.py; done
sleep 12; profile_l1 "F_F_stack_chanpred_x4"; kill_all_ai

log "=== F_F 2x resnet ==="
for k in 1 2; do ai_bg_venv "resnet_x2_$k" "" run_resnet_stress.py 64 fp16; done
sleep 12; profile_l1 "F_F_stack_resnet_x2"; kill_all_ai

# ---- F_G: kitchen sink ------------------------------------------------------
log "=== F_G kitchen (chanpred + memcpy + gemm) ==="
ai_bg_venv kitchen_chanpred "" run_channel_prediction.py
ai_bg_venv kitchen_memcpy "--env=SIZE_MB=128 --env=N_STREAMS=4 --env=KIND=D2D" run_memcpy_massive.py
ai_bg_venv kitchen_gemm "--env=DIM=2048" run_gemm_massive.py
sleep 12; profile_l1 "F_G_kitchen"; kill_all_ai

log "===== F-equivalent on no-MIG DONE ====="
log "JSON captures: $(ls "$RESULTS_DIR"/realL1_*.json 2>/dev/null | wc -l)"
if [[ "$USE_NSYS" == "1" ]]; then log "nsys captures: $(ls "$RESULTS_DIR"/*.nsys-rep 2>/dev/null | wc -l)"; fi
