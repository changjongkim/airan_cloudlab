#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q regular
#SBATCH -t 03:00:00
#SBATCH -A m1248_g
#SBATCH -J NCU_nomig
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.err

# Priority 3 — NCU hardware counters on no-MIG (DRAM/L2/SM), full set.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff

export CELLS=20 ITERS=3 AI_DUR=1800 NEURALRX_WAIT=75
export RESULTS_DIR="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/perlmutter_nomig/NCU_nomig"

date
echo "===== Priority 3: NCU hardware counters on no-MIG ====="
bash 04_ncu_nomig.sh
date
echo "===== DONE — CSVs in $RESULTS_DIR ====="
ls -1 "$RESULTS_DIR"/NCU_*.csv 2>/dev/null | wc -l
