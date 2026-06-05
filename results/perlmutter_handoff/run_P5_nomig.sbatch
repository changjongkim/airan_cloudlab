#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -A m1248_g
#SBATCH -J P5_nomig
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.err

# Priority 5 — 5-minute sustained on no-MIG, all 9 scenarios x N=2.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff

export CELLS=20 ITERS=7500 N=2 AI_DUR=3600 NEURALRX_WAIT=75
export RESULTS_DIR="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/perlmutter_nomig/P5_nomig"

date
echo "===== Priority 5: 5-min sustained on no-MIG ====="
bash 05_p5_sustained_nomig.sh
date
echo "===== DONE — results in $RESULTS_DIR ====="
ls -1 "$RESULTS_DIR"/realL1_P5_*.json 2>/dev/null | wc -l
