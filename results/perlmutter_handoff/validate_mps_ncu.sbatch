#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH -q debug
#SBATCH -t 0:25:00
#SBATCH -A m1248_g
#SBATCH -J mps_ncu_val
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.err

# Does ncu work under MPS, and does it capture L1 DRAM contention from a
# concurrent HBM hog? sat_hbm is the collapse case. DCGM paused, exclusive node.
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff
IMG="nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb"
AERIAL_REPO="/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran"
SCRIPTS_DIR="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/cloudlab_aerial"
AI_DIR="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/experiments"
VENV="$PWD/airan_venv"
RESULTS_DIR="$PWD/perlmutter_nomig/MPS_ncu_validate"; mkdir -p "$RESULTS_DIR"
METRICS="dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active"
export CUDA_MPS_PIPE_DIRECTORY="$PWD/mps_pipe_val_${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="$PWD/mps_log_val_${SLURM_JOB_ID}"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

cleanup(){ echo quit | nvidia-cuda-mps-control 2>/dev/null||true; dcgmi profile --resume 2>&1|tail -1; rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"; }
trap cleanup EXIT INT TERM
date; nvidia-smi --query-gpu=name,mig.mode.current --format=csv,noheader|head -1
echo "[1] pause DCGM"; dcgmi profile --pause 2>&1|tail -1
echo "[2] start MPS"; nvidia-cuda-mps-control -d; sleep 3

echo "[3] start sat_hbm AI (MPS, concurrent)"
( shifter --image="$IMG" --volume="$AI_DIR:/experiments" \
    --env=CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" \
    "$VENV/bin/python" /experiments/run_hbm_stress.py 0 300 16 ) > "$RESULTS_DIR/sat_hbm_ai.log" 2>&1 &
AIP=$!; sleep 12

echo "[4] ncu on L1 (MPS concurrent with sat_hbm)"
shifter --image="$IMG" --volume="$AERIAL_REPO:/opt/nvidia/cuBB" --volume="$SCRIPTS_DIR:/scripts" \
  --volume="$RESULTS_DIR:/out" --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src --env=RESULTS_DIR=/out \
  --env=CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" --workdir=/scripts \
  bash -c "ncu --target-processes all --replay-mode kernel --clock-control none \
               --metrics $METRICS --csv --log-file /out/MPSncu_sat_hbm.csv \
               python3 real_l1.py MPSncu_sat_hbm 2 1" 2>&1 | tail -18
kill -TERM $AIP 2>/dev/null||true
echo "[5] CSV produced + DRAM rows?"
grep -iE "dram__throughput" "$RESULTS_DIR"/MPSncu_sat_hbm.csv 2>/dev/null | head -3 || echo "  no metric rows"
date
