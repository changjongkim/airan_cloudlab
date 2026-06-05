#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH -q regular
#SBATCH -t 03:00:00
#SBATCH -A m1248_g
#SBATCH -J NCU_nomig
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.err

# Priority 3 — NCU hardware counters on no-MIG (DRAM/L2/SM), full set.
#
# ncu needs the GPU perf counters, which DCGM holds. Per NERSC docs
# (https://docs.nersc.gov/tools/performance/nvidiaproftools/) we pause DCGM
# while profiling and resume after. Exclusive node so the pause only affects
# our own node, not other users. trap guarantees resume even on failure/timeout.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff

export CELLS=20 ITERS=3 AI_DUR=1800 NEURALRX_WAIT=75
export RESULTS_DIR="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/perlmutter_nomig/NCU_nomig"

resume_dcgm() { echo "[ncu] resuming DCGM..."; dcgmi profile --resume 2>&1 | tail -1; }
trap resume_dcgm EXIT INT TERM

date
echo "===== Priority 3: NCU hardware counters on no-MIG ====="
echo "[ncu] pausing DCGM (NERSC-documented ncu workaround)..."
dcgmi profile --pause 2>&1 | tail -1

bash 04_ncu_nomig.sh

resume_dcgm
trap - EXIT INT TERM
date
echo "===== DONE — CSVs in $RESULTS_DIR ====="
ls -1 "$RESULTS_DIR"/NCU_*.csv 2>/dev/null | wc -l
