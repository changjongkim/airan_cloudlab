#!/usr/bin/env bash
set -euo pipefail

IMG=${IMG:-airan:25-3-final}
GPU_INDEX=${GPU_INDEX:-3}
QWEN_MPS_PCT=${QWEN_MPS_PCT:-50}
NRX_RATE=${NRX_RATE:-900}
SCHEDULE=${SCHEDULE:-off:2,on:2,off:3,on:2,off:3}
RESULTS_ROOT=${RESULTS_ROOT:-/mydata/results/drain_free/reclaim_trial}
REPO=/mydata/aerial-cuda-accelerated-ran
ENGINE=/mydata/results/nrx_deep_profile/engines/neural_rx_fp16_full.trt
MPS_PIPE=/tmp/nvidia-mps-reclaim
MPS_LOG=/tmp/nvidia-mps-reclaim-log
QWEN_NAME=drainfree_reclaim_qwen

cleanup() {
    docker stop -t 20 "$QWEN_NAME" >/dev/null 2>&1 || true
    docker logs "$QWEN_NAME" >"$RESULTS_ROOT/qwen.log" 2>&1 || true
    docker rm -f "$QWEN_NAME" >/dev/null 2>&1 || true
    if [[ -S "$MPS_PIPE/control" ]]; then
        echo quit | sudo env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
            nvidia-cuda-mps-control >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

sudo mkdir -p "$RESULTS_ROOT" "$RESULTS_ROOT/control" "$MPS_PIPE" "$MPS_LOG"
sudo chmod 0777 "$RESULTS_ROOT" "$RESULTS_ROOT/control" "$MPS_PIPE" "$MPS_LOG"
rm -f "$RESULTS_ROOT/control/gate" "$RESULTS_ROOT/control/qwen.ready"
printf '0\n' >"$RESULTS_ROOT/control/gate"
chmod 0666 "$RESULTS_ROOT/control/gate"

GPU_UUID=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader)
sudo env CUDA_VISIBLE_DEVICES="$GPU_UUID" \
    CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$MPS_LOG" \
    nvidia-cuda-mps-control -d </dev/null >"$RESULTS_ROOT/mps_daemon.log" 2>&1 &
for _ in $(seq 1 100); do
    [[ -S "$MPS_PIPE/control" ]] && break
    sleep 0.1
done
[[ -S "$MPS_PIPE/control" ]]

docker run -d --name "$QWEN_NAME" --runtime=nvidia --ipc=host \
    -e NVIDIA_VISIBLE_DEVICES="$GPU_INDEX" \
    -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$QWEN_MPS_PCT" \
    -e CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$MPS_PIPE:$MPS_PIPE" -v "$MPS_LOG:$MPS_LOG" \
    -v "$REPO:/opt/nvidia/cuBB" \
    -v /mydata/hf_cache:/mydata/hf_cache \
    -v "$RESULTS_ROOT:/results" \
    -w /opt/nvidia/cuBB/pyaerial \
    "$IMG" python3 -B qwen7b_gated.py \
        --gate-file /results/control/gate \
        --ready-file /results/control/qwen.ready \
        --output /results/qwen_timeline.json \
        --duration 300 >/dev/null

for _ in $(seq 1 600); do
    [[ -f "$RESULTS_ROOT/control/qwen.ready" ]] && break
    if ! docker inspect -f '{{.State.Running}}' "$QWEN_NAME" 2>/dev/null | grep -qx true; then
        docker logs "$QWEN_NAME" >&2
        exit 1
    fi
    sleep 0.25
done
[[ -f "$RESULTS_ROOT/control/qwen.ready" ]]

docker run --rm --runtime=nvidia --ipc=host \
    -e NVIDIA_VISIBLE_DEVICES="$GPU_INDEX" \
    -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 \
    -e CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$MPS_PIPE:$MPS_PIPE" -v "$MPS_LOG:$MPS_LOG" \
    -v "$REPO:/opt/nvidia/cuBB" \
    -v /mydata/results/nrx_deep_profile:/nrx_results:ro \
    -v "$RESULTS_ROOT:/results" \
    -w /opt/nvidia/cuBB/pyaerial \
    "$IMG" python3 -B nrx_reclaim_timeline.py \
        --engine /nrx_results/engines/neural_rx_fp16_full.trt \
        --gate-file /results/control/gate \
        --rate "$NRX_RATE" \
        --schedule "$SCHEDULE" \
        --output /results/nrx_timeline.json | tee "$RESULTS_ROOT/nrx.log"

cleanup
trap - EXIT INT TERM
echo "[RECLAIM-RUNNER] OK results=$RESULTS_ROOT"
