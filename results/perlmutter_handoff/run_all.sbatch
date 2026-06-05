#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -t 8:00:00
#SBATCH -q regular
#SBATCH -A m4243        # <-- 사용자 NERSC 계정으로 변경
#SBATCH -J nomig_compare
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.err

# Master SLURM job to run all no-MIG experiments
# 총 ~6-7시간 예상. 8h 한도로 잡음.

set -e

# Config — Setup 결과 적용
export AERIAL_IMAGE="nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb"
export AERIAL_REPO="$SCRATCH/aerial-cuda-accelerated-ran"
export REPO="$SCRATCH/airan_cloudlab"
export HANDOFF="$REPO/results/perlmutter_handoff"

cd $HANDOFF

date
echo "===== Stage 1: F-equivalent on no-MIG (default time-slice) ====="
bash 02_F_saturation_nomig.sh

date
echo "===== Stage 2: MPS comparison ====="
bash 03_mps_compare.sh

date
echo "===== Stage 3: NCU hardware counters ====="
bash 04_ncu_nomig.sh

date
echo "===== Stage 4: 5-min sustained ====="
# 시간 모자라면 cut 가능
timeout 7200 bash 05_p5_sustained_nomig.sh || echo "P5 timed out — partial results saved"

date
echo "===== ALL DONE ====="
echo "Results in: $REPO/results/perlmutter_nomig/"
ls -la $REPO/results/perlmutter_nomig/
