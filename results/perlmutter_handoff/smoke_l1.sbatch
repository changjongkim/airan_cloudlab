#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q debug
#SBATCH -t 0:05:00
#SBATCH -A m1248_g
#SBATCH -J nomig_l1
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.err

# L1-only smoke: validate real_l1.py runs on Perlmutter pyaerial (stream API fix).
set -uo pipefail
AERIAL_IMAGE="nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb"
AERIAL_REPO="/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran"
REPO="/pscratch/sd/s/sgkim/kcj/airan_cloudlab"
SCRIPTS_DIR="$REPO/scripts_for_node/cloudlab_aerial"
RESULTS_DIR="$REPO/results/perlmutter_nomig/smoke"
mkdir -p "$RESULTS_DIR"

date; hostname
nvidia-smi --query-gpu=name,mig.mode.current --format=csv,noheader

echo "==== real_l1 ALONE (cells=4 iters=20) ===="
shifter --image="$AERIAL_IMAGE" \
    --volume="$AERIAL_REPO:/opt/nvidia/cuBB" \
    --volume="$SCRIPTS_DIR:/scripts" \
    --volume="$RESULTS_DIR:/out" \
    --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    --env=RESULTS_DIR=/out \
    --workdir=/scripts \
    python3 real_l1.py smoke_l1only 4 20 2>&1 | tail -30

echo "==== JSON ===="
ls -la "$RESULTS_DIR"/realL1_*.json 2>/dev/null && echo "--- contents ---" && cat "$RESULTS_DIR"/realL1_smoke_l1only_*.json 2>/dev/null | head -40
date
