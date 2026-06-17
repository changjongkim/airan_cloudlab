#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -A m1248_g
#SBATCH -J MPS_ncu
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.err

# MPS + NCU on no-MIG. Exclusive node + DCGM pause (ncu needs the counters).
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff
export CELLS=4 ITERS=3 AI_DUR=1800 NEURALRX_WAIT=75
export NCU_CONDS="alone neuralrx resnet chanpred sat_hbm sat_compute"
export RESULTS_DIR="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/perlmutter_nomig/MPS_ncu"

resume_dcgm(){ echo "[ncu] resume DCGM"; dcgmi profile --resume 2>&1|tail -1; }
trap resume_dcgm EXIT INT TERM
date
echo "[ncu] pause DCGM"; dcgmi profile --pause 2>&1|tail -1
bash 08_mps_ncu_nomig.sh
resume_dcgm; trap - EXIT INT TERM
date
ls -1 "$RESULTS_DIR"/MPSncu_*.csv 2>/dev/null
