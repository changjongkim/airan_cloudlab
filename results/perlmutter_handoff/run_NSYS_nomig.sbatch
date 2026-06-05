#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q regular
#SBATCH -t 01:30:00
#SBATCH -A m1248_g
#SBATCH -J NSYS_nomig
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.err

# Priority 4 — per-call NSYS capture on no-MIG (alone, neuralrx, resnet).
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff

export CELLS=20 ITERS=50 AI_DUR=1800 NEURALRX_WAIT=75
export RESULTS_DIR="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/perlmutter_nomig/nsys_nomig"

date
echo "===== Priority 4: per-call NSYS on no-MIG ====="
bash 06_nsys_percall_nomig.sh
date
echo "===== DONE — nsys-rep in $RESULTS_DIR ====="
ls -1 "$RESULTS_DIR"/nsys_*.nsys-rep 2>/dev/null
