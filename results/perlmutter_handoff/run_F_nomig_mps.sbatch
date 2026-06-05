#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -A m1248_g
#SBATCH -J F_nomig_mps
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.err

# Priority 2 — MPS comparison on no-MIG, full parameters.
# Submitted with --dependency=afterok:<priority1 job> so it runs after stage 1.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff

export CELLS=20 ITERS=100 N=5 AI_DUR=1800 NEURALRX_WAIT=75
export RESULTS_DIR="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/perlmutter_nomig/F_nomig_mps"

date
echo "===== Priority 2: MPS comparison on no-MIG ====="
bash 03_mps_compare.sh
date
echo "===== DONE — results in $RESULTS_DIR ====="
ls -1 "$RESULTS_DIR"/realL1_MPS_*.json 2>/dev/null | wc -l
