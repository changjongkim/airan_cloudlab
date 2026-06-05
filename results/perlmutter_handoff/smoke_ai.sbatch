#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q debug
#SBATCH -t 0:12:00
#SBATCH -A m1248_g
#SBATCH -J nomig_ai
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.err

# AI workload smoke: validate venv torch on GPU + L1 coloc + neuralrx.
set -uo pipefail
IMG="nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb"
AERIAL_REPO="/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran"
HANDOFF="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff"
REPO="/pscratch/sd/s/sgkim/kcj/airan_cloudlab"
VENV="$HANDOFF/airan_venv"
SCRIPTS_DIR="$REPO/scripts_for_node/cloudlab_aerial"
AI_DIR="$REPO/scripts_for_node/experiments"
RESULTS_DIR="$HANDOFF/perlmutter_nomig/smoke"
mkdir -p "$RESULTS_DIR"

date; hostname
nvidia-smi --query-gpu=name,mig.mode.current --format=csv,noheader | head -1

# --- helpers ---
l1_run() {  # label cells iters
  shifter --image="$IMG" \
    --volume="$AERIAL_REPO:/opt/nvidia/cuBB" \
    --volume="$SCRIPTS_DIR:/scripts" \
    --volume="$RESULTS_DIR:/out" \
    --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src --env=RESULTS_DIR=/out \
    --workdir=/scripts \
    python3 real_l1.py "$1" "$2" "$3"
}

echo; echo "==== [0] venv torch.cuda check ===="
shifter --image="$IMG" "$VENV/bin/python" -c \
  'import torch; print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(), "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")' 2>&1 | tail -3

echo; echo "==== [A] L1 + chanpred (venv) coloc ===="
( shifter --image="$IMG" --volume="$AI_DIR:/experiments" \
    "$VENV/bin/python" /experiments/run_channel_prediction.py 0 90 ) \
    > "$RESULTS_DIR/A_chanpred.log" 2>&1 &
AI_A=$!
echo "chanpred(venv) pid=$AI_A"; sleep 12
echo "chanpred log head:"; head -5 "$RESULTS_DIR/A_chanpred.log"
l1_run smokeA_l1_chanpred 4 20 2>&1 | tail -6
kill -TERM $AI_A 2>/dev/null || true; sleep 2

echo; echo "==== [B] L1 + neuralrx (base) coloc ===="
( shifter --image="$IMG" \
    --volume="$AERIAL_REPO:/opt/nvidia/cuBB" --volume="$AI_DIR:/experiments" \
    --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src --env=cuBB_SDK=/opt/nvidia/cuBB \
    python3 /experiments/run_neural_rx_stress.py 0 120 ) \
    > "$RESULTS_DIR/B_neuralrx.log" 2>&1 &
AI_B=$!
echo "neuralrx(base) pid=$AI_B — waiting 75s for TRT build"; sleep 75
echo "neuralrx log head:"; head -12 "$RESULTS_DIR/B_neuralrx.log"
l1_run smokeB_l1_neuralrx 4 20 2>&1 | tail -6
kill -TERM $AI_B 2>/dev/null || true; sleep 2

echo; echo "==== JSON outputs ===="
ls -la "$RESULTS_DIR"/realL1_smoke*.json 2>/dev/null
date
