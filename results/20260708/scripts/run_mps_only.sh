#!/usr/bin/env bash
# MPS-only experiment. Assumes:
#   - MPS server ALREADY running in container 'mps_srv'
#   - GPU 0 in EXCLUSIVE_PROCESS mode
#   - MIG disabled
#
# Conditions × 3 trials:
#   MPS_alone   L1 only
#   MPS_coloc   L1 + NRx concurrent
# 30s NSYS window each. Same params as TS runs for direct comparison.

set -uo pipefail
GPU=0
CELLS=20
ITERS=100
N_TRIALS=3
DATE_DIR=${DATE_DIR:-$(date +%Y%m%d)}
IMAGE=airan:25-3-final
SDK=/mydata/aerial-cuda-accelerated-ran
REPO=/users/sgkim/AIRAN_Changjong
SCRIPT=/users/sgkim/cloudlab_aerial
OUT=/mydata/results/$DATE_DIR/mps_only
mkdir -p "$OUT" && chmod 777 "$OUT"

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$OUT/run.log"; }

GPU_UUID=$(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader)
log "GPU UUID: $GPU_UUID"
log "MPS status: $(nvidia-smi --query-gpu=compute_mode --format=csv,noheader -i $GPU)"

start_nrx_bg(){ local tag=$1
  docker run -d --rm --init --user 0:0 --gpus "\"device=$GPU_UUID\"" \
    --ipc=host --pid=host \
    -v /tmp:/tmp \
    -e CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_0 \
    -e CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_0 \
    -v "$SDK:/opt/nvidia/cuBB" \
    -v "$REPO:/workspace/AIRAN_Changjong" \
    -v "$OUT:/aiout" \
    -e cuBB_SDK=/opt/nvidia/cuBB \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
    --name "nrx_${tag}_$(date +%s)" \
    "$IMAGE" \
    bash -c "python3 /workspace/AIRAN_Changjong/experiments/run_neural_rx_stress.py 0 120 > /aiout/${tag}_nrx.log 2>&1" >/dev/null
}
kill_nrx(){ docker ps --filter 'name=nrx_' -q | xargs -r docker stop -t 3 2>/dev/null; sleep 1; }

profile_l1(){ local label=$1
  docker run --rm --user 0:0 --gpus "\"device=$GPU_UUID\"" \
    --ipc=host --pid=host \
    -v /tmp:/tmp \
    -e CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_0 \
    -e CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_0 \
    -v "$SDK:/opt/nvidia/cuBB" -v "$SCRIPT:/scripts" -v "$OUT:/out" \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/cupy_cache \
    -e RESULTS_DIR=/out \
    -w /scripts "$IMAGE" \
    bash -c "nsys profile --trace=cuda --duration=30 --output=/out/${label} --force-overwrite=true --stats=false python3 real_l1.py ${label} $CELLS $ITERS" 2>&1 | tail -3
}

log "=== MPS_alone ==="
for t in $(seq 1 $N_TRIALS); do
  log "--- MPS_alone trial $t ---"
  profile_l1 "MPS_alone_t$t"
done

log "=== MPS_coloc ==="
for t in $(seq 1 $N_TRIALS); do
  log "--- MPS_coloc trial $t ---"
  start_nrx_bg "MPS_coloc_t$t"
  sleep 15
  profile_l1 "MPS_coloc_t$t"
  kill_nrx
done

log "=== DONE. Results in $OUT ==="
ls -la "$OUT"
