#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q regular
#SBATCH -t 01:30:00
#SBATCH -A m1248_g
#SBATCH -J MPS_nsys
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.err
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff
export CELLS=20 ITERS=50 AI_DUR=1800 NEURALRX_WAIT=75
export RESULTS_DIR="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/perlmutter_nomig/MPS_nsys"
date
echo "===== MPS + NSYS on no-MIG ====="
bash 07_mps_nsys_nomig.sh
date
ls -1 "$RESULTS_DIR"/MPSnsys_*.nsys-rep 2>/dev/null
