#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -A m1248_g
#SBATCH -J F_nomig
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.err

# Priority 1 — F-equivalent on no-MIG, full parameters.
# 1 node / 1 A100 (no-MIG), default time-slice contention.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff

# Full measurement params (these are the 02 defaults; set explicitly for clarity).
export CELLS=20 ITERS=100 N=5 AI_DUR=1800 NEURALRX_WAIT=75 USE_NSYS=0
export RESULTS_DIR="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/perlmutter_nomig/F_nomig"

date
echo "===== Priority 1: F-equivalent on no-MIG (default time-slice) ====="
bash 02_F_saturation_nomig.sh
date
echo "===== DONE — results in $RESULTS_DIR ====="
ls -1 "$RESULTS_DIR"/realL1_*.json 2>/dev/null | wc -l
