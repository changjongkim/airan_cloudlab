#!/usr/bin/env bash
set -Eeuo pipefail

SDK_HOST=/mydata/aerial-cuda-accelerated-ran
SDK_MNT=/opt/nvidia/cuBB
ENGINE_HOST=/mydata/results/nrx_deep_profile/engines
RESULTS_ROOT=${RESULTS_ROOT:-/mydata/results/placement_matrix/mps_direct}
IMG=airan:25-3-final
GPU_UUID=${GPU_UUID:-GPU-ee09d6cb-2e38-e9e1-9b76-bbc0f8f79b1a}
MPS_PCTS=${MPS_PCTS:-"30 50 70 100"}
TRIALS=${TRIALS:-3}
ITERS=${ITERS:-1000}
WARMUP=${WARMUP:-100}
QWEN_DUR=${QWEN_DUR:-300}
QWEN_WARMUP=${QWEN_WARMUP:-60}
MPS_PIPE=/tmp/nvidia-mps-direct
MPS_LOG=/tmp/nvidia-mps-direct-log

cleanup() {
    docker rm -f mps_direct_qwen mps_direct_bench >/dev/null 2>&1 || true
    if [[ -S "$MPS_PIPE/control" ]]; then
        echo quit | sudo env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
            nvidia-cuda-mps-control >/dev/null 2>&1 || true
    fi
    sudo pkill -f nvidia-cuda-mps-server >/dev/null 2>&1 || true
    sudo pkill -f nvidia-cuda-mps-control >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

if [[ -e "$RESULTS_ROOT/COMPLETE" ]]; then
    echo "[MPS-DIRECT] results already complete: $RESULTS_ROOT" >&2
    exit 1
fi
mkdir -p "$RESULTS_ROOT"
sudo chmod 0777 "$RESULTS_ROOT"
cleanup
sudo mkdir -p "$MPS_PIPE" "$MPS_LOG"
sudo chmod 0777 "$MPS_PIPE" "$MPS_LOG"
sudo env CUDA_VISIBLE_DEVICES="$GPU_UUID" \
    CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$MPS_LOG" \
    nvidia-cuda-mps-control -d </dev/null >/tmp/mps_direct_daemon.log 2>&1 &
sleep 3
[[ -S "$MPS_PIPE/control" ]] || { echo "[MPS-DIRECT] daemon failed" >&2; exit 1; }

run_bench() {
    local mode=$1 label=$2 outdir=$3
    mkdir -p "$outdir"; chmod 0777 "$outdir"
    docker run --rm --name mps_direct_bench --runtime=nvidia --ipc=host \
        -e NVIDIA_VISIBLE_DEVICES="$GPU_UUID" \
        -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 \
        -e CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
        -e NRX_ENGINE=/engines/neural_rx_fp16_4g.trt \
        -e RESULTS_DIR=/results \
        -v "$MPS_PIPE:$MPS_PIPE" -v "$MPS_LOG:$MPS_LOG" \
        -v "$ENGINE_HOST:/engines:ro" \
        -v "$SDK_HOST:$SDK_MNT" -v "$outdir:/results" \
        -w "$SDK_MNT/pyaerial" "$IMG" \
        python3 p2p_overlap_bench.py "$mode" "$label" \
            --iterations "$ITERS" --warmup "$WARMUP" --ring-depth 2 \
        >"$outdir/bench.log" 2>&1
    grep -q '\[P2P-BENCH\] RESULT' "$outdir/bench.log"
}

for pct in $MPS_PCTS; do
    pct_dir="$RESULTS_ROOT/pct${pct}"
    mkdir -p "$pct_dir"; chmod 0777 "$pct_dir"
    docker run -d --name mps_direct_qwen --runtime=nvidia --ipc=host \
        -e NVIDIA_VISIBLE_DEVICES="$GPU_UUID" \
        -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$pct" \
        -e CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
        -e HF_HOME=/mydata/hf_cache -e TRANSFORMERS_OFFLINE=1 \
        -v "$MPS_PIPE:$MPS_PIPE" -v "$MPS_LOG:$MPS_LOG" \
        -v /mydata/hf_cache:/mydata/hf_cache \
        -v "$SDK_HOST/pyaerial:/work" -w /work "$IMG" \
        python3 qwen7b_stress.py "$QWEN_DUR" >/dev/null
    sleep "$QWEN_WARMUP"
    docker inspect -f '{{.State.Running}}' mps_direct_qwen | grep -qx true
    for trial in $(seq 1 "$TRIALS"); do
        run_bench standalone "mps_p${pct}_l1_t${trial}" \
            "$pct_dir/trial${trial}/l1_only"
        run_bench same "mps_p${pct}_same_t${trial}" \
            "$pct_dir/trial${trial}/same_overlap"
    done
    docker logs mps_direct_qwen >"$pct_dir/qwen.log" 2>&1 || true
    docker rm -f mps_direct_qwen >/dev/null 2>&1 || true
    grep -qE '\[Qwen\] (progress|done): [0-9]+ iters' "$pct_dir/qwen.log"
    echo "[MPS-DIRECT] pct=$pct complete"
done

date -u +%FT%TZ >"$RESULTS_ROOT/COMPLETE"
echo "[MPS-DIRECT] OK"
