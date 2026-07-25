#!/usr/bin/env bash
# Run L1 baseline on a fully isolated GPU (no MIG, no MIG instance contention).
# 5/31 fix: explicit GPU UUID prevents docker --gpus all from being routed to a MIG instance.
# Also uses --user to match host UID so results write succeeds.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="airan:25-3-final"
AERIAL_SDK="/mydata/aerial-cuda-accelerated-ran"
N="${N:-20}"
CELLS="${CELLS:-20}"
ITERS="${ITERS:-30}"
TAG="${TAG:-baseline_fullGPU_v2}"
GPU_IDX="${GPU_IDX:-3}"
DATE_DIR="$(date +%Y%m%d)"
OUTDIR="$SCRIPT_DIR/results/$DATE_DIR/n${N}_${TAG}"
mkdir -p "$OUTDIR" && chmod 777 "$OUTDIR"
HOST_UID=$(id -u)
HOST_GID=$(id -g)

GPU_UUID=$(nvidia-smi -L | awk -v g="$GPU_IDX" '/^GPU /{ if (match($0, "^GPU "g":")) print }' | grep -oE 'GPU-[0-9a-f-]+' | head -1)
if [[ -z "$GPU_UUID" ]]; then echo "ERROR: GPU $GPU_IDX UUID not found"; exit 1; fi
echo "[run_fullgpu] GPU $GPU_IDX UUID=$GPU_UUID"
echo "[run_fullgpu] OUTDIR=$OUTDIR"
echo "[run_fullgpu] N=$N CELLS=$CELLS ITERS=$ITERS TAG=$TAG UID=$HOST_UID GID=$HOST_GID"

for i in $(seq 1 $N); do
  echo "------ run $i/$N : $(date +%H:%M:%S) ------"
  docker run --rm \
    --user "$HOST_UID:$HOST_GID" \
    --gpus "\"device=$GPU_UUID\"" \
    -v "$AERIAL_SDK:/opt/nvidia/cuBB" \
    -v "$SCRIPT_DIR:/scripts" \
    -v "$OUTDIR:/results_out" \
    -e PYTHONPATH=/opt/nvidia/cuBB/pyaerial/src \
    -e HOME=/tmp \
    -e CUPY_CACHE_DIR=/tmp/cupy_cache \
    -w /scripts \
    "$IMAGE" \
    bash -c "mkdir -p /scripts/results && python3 /scripts/real_l1.py ${TAG}_run${i} $CELLS $ITERS" 2>&1 | \
    tee "$OUTDIR/run_${i}.log" | grep -E "mean=|p99=|cells=" | tail -3
done

echo "[run_fullgpu] DONE — results in $OUTDIR"
