#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -A m1248_g
#SBATCH -J NCU_resume
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.err

# Resume NCU for the 4 conditions the first run (53991530) timed out before.
# CELLS=4 keeps ncu kernel-replay fast (the CELLS=20 run hit 152s/iter under
# NeuralRx). alone+NeuralRx already captured; skip them. Same NCU_nomig dir.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff

export CELLS=4 ITERS=3 AI_DUR=1800
export NCU_CONDS="resnet chanpred sat_hbm sat_compute"
export RESULTS_DIR="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/perlmutter_nomig/NCU_nomig"

resume_dcgm(){ echo "[ncu] resuming DCGM..."; dcgmi profile --resume 2>&1 | tail -1; }
trap resume_dcgm EXIT INT TERM
date
echo "[ncu] pausing DCGM (NERSC-documented ncu workaround)..."
dcgmi profile --pause 2>&1 | tail -1
bash 04_ncu_nomig.sh
resume_dcgm; trap - EXIT INT TERM
date
echo "===== NCU resume DONE — CSVs in $RESULTS_DIR ====="
ls -1 "$RESULTS_DIR"/NCU_*.csv 2>/dev/null
