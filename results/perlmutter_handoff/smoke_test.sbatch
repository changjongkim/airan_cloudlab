#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q debug
#SBATCH -t 0:05:00
#SBATCH -A m1248_g
#SBATCH -J nomig_smoke
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.err

# Smoke test — does the no-MIG L1 pipeline actually execute on Perlmutter?
# - real_l1 alone (small)
# - real_l1 + chanpred coloc (small)
# No nsys (per user request). Validates: shifter + pyaerial + cuPHY + GPU + AI coloc.

set -uo pipefail

AERIAL_IMAGE="nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb"
AERIAL_REPO="/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran"
REPO="/pscratch/sd/s/sgkim/kcj/airan_cloudlab"
SCRIPTS_DIR="$REPO/scripts_for_node/cloudlab_aerial"        # has real_l1.py
AI_WORKLOAD_DIR="$REPO/scripts_for_node/experiments"        # has run_channel_prediction.py etc.
RESULTS_DIR="$REPO/results/perlmutter_nomig/smoke"

CELLS=4
ITERS=20

mkdir -p "$RESULTS_DIR"

echo "==== smoke test ===="; date; hostname
nvidia-smi --query-gpu=index,name,mig.mode.current --format=csv
echo "GPU UUID: $(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)"

shifter_run() {
    shifter --image="$AERIAL_IMAGE" \
        --volume="$AERIAL_REPO:/opt/nvidia/cuBB" \
        --volume="$SCRIPTS_DIR:/scripts" \
        --volume="$AI_WORKLOAD_DIR:/experiments" \
        --volume="$RESULTS_DIR:/out" \
        --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
        --env=RESULTS_DIR=/out \
        --workdir=/scripts \
        "$@"
}

echo; echo "==== [1/2] real_l1 ALONE (cells=$CELLS iters=$ITERS) ===="
shifter_run python3 real_l1.py smoke_alone $CELLS $ITERS 2>&1 | tail -25

echo; echo "==== [2/2] real_l1 + chanpred coloc ===="
echo "-- starting chanpred in background (60s) --"
( shifter --image="$AERIAL_IMAGE" \
    --volume="$AERIAL_REPO:/opt/nvidia/cuBB" \
    --volume="$AI_WORKLOAD_DIR:/experiments" \
    --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    python3 /experiments/run_channel_prediction.py 0 60 ) \
    > "$RESULTS_DIR/chanpred_ai.log" 2>&1 &
AI_PID=$!
echo "chanpred bg pid=$AI_PID"
sleep 10
echo "-- running L1 under contention --"
shifter_run python3 real_l1.py smoke_coloc $CELLS $ITERS 2>&1 | tail -25
kill -TERM $AI_PID 2>/dev/null || true
sleep 2

echo; echo "==== JSON outputs ===="
ls -la "$RESULTS_DIR"/*.json 2>/dev/null
echo "==== smoke test done ===="; date
