#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -A m1248_g
#SBATCH -J osrt_cap
#SBATCH -o %x_%j.log
#SBATCH -e %x_%j.err

# Syscall / OS-runtime level capture (nsys --trace=cuda,osrt,nvtx) to see WHERE the
# host thread blocks (futex/poll/ioctl) during the cudaFree/sync stalls. Default +
# MPS, key conditions alone/neuralrx/sat_hbm. Small ITERS (osrt trace is heavy).
cd /pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff
IMG="nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb"
AERIAL_REPO="/pscratch/sd/s/sgkim/kcj/AI-RAN/aerial-cuda-accelerated-ran"
SCRIPTS_DIR="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/cloudlab_aerial"
AI_DIR="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/experiments"
VENV="$PWD/airan_venv"
OUT="$PWD/perlmutter_nomig/OSRT"; mkdir -p "$OUT"
CELLS=20; ITERS=30; AI_DUR=600

l1_osrt(){  # label  [mps_pipe]
  local label=$1 mps=$2 envs=""
  [ -n "$mps" ] && envs="--env=CUDA_MPS_PIPE_DIRECTORY=$mps"
  shifter --image="$IMG" --volume="$AERIAL_REPO:/opt/nvidia/cuBB" --volume="$SCRIPTS_DIR:/scripts" \
    --volume="$OUT:/out" --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src --env=RESULTS_DIR=/out $envs --workdir=/scripts \
    bash -c "nsys profile --trace=cuda,osrt,nvtx --output=/out/${label} --force-overwrite=true --stats=false \
             python3 real_l1.py ${label} $CELLS $ITERS" 2>&1 | tail -2
}
start_ai(){ # script  [mps_pipe]  args...
  local script=$1 mps=$2; shift 2; local envs=""
  [ -n "$mps" ] && envs="--env=CUDA_MPS_PIPE_DIRECTORY=$mps"
  if [ "$script" = "run_neural_rx_stress.py" ]; then
    ( shifter --image="$IMG" --volume="$AERIAL_REPO:/opt/nvidia/cuBB" --volume="$AI_DIR:/experiments" \
        --env=PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src --env=cuBB_SDK=/opt/nvidia/cuBB $envs \
        python3 "/experiments/$script" 0 $AI_DUR "$@" ) >"$OUT/ai.log" 2>&1 &
  else
    ( shifter --image="$IMG" --volume="$AI_DIR:/experiments" $envs \
        "$VENV/bin/python" "/experiments/$script" 0 $AI_DUR "$@" ) >"$OUT/ai.log" 2>&1 &
  fi
  echo $!
}
date
echo "########## DEFAULT time-slice ##########"
echo "== alone =="; l1_osrt osrt_alone ""
echo "== neuralrx =="; P=$(start_ai run_neural_rx_stress.py ""); sleep 75; l1_osrt osrt_neuralrx ""; kill $P 2>/dev/null; sleep 3
echo "== sat_hbm =="; P=$(start_ai run_hbm_stress.py "" 16); sleep 10; l1_osrt osrt_sat_hbm ""; kill $P 2>/dev/null; sleep 3

echo "########## MPS ##########"
export CUDA_MPS_PIPE_DIRECTORY="$PWD/mps_pipe_osrt_${SLURM_JOB_ID}"; export CUDA_MPS_LOG_DIRECTORY="$PWD/mps_log_osrt_${SLURM_JOB_ID}"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"; nvidia-cuda-mps-control -d; sleep 3
M="$CUDA_MPS_PIPE_DIRECTORY"
echo "== MPS alone =="; l1_osrt osrt_MPS_alone "$M"
echo "== MPS neuralrx =="; P=$(start_ai run_neural_rx_stress.py "$M"); sleep 75; l1_osrt osrt_MPS_neuralrx "$M"; kill $P 2>/dev/null; sleep 3
echo "== MPS sat_hbm =="; P=$(start_ai run_hbm_stress.py "$M" 16); sleep 10; l1_osrt osrt_MPS_sat_hbm "$M"; kill $P 2>/dev/null; sleep 3
echo quit | nvidia-cuda-mps-control 2>/dev/null; rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
date
ls -1 "$OUT"/*.nsys-rep
