#!/usr/bin/env bash
set -Eeuo pipefail

SDK_HOST=/mydata/aerial-cuda-accelerated-ran
SDK_MNT=/opt/nvidia/cuBB
ENGINE_HOST=/mydata/results/nrx_deep_profile/engines
RESULTS_ROOT=${RESULTS_ROOT:-/mydata/results/placement_matrix/mig_mps_direct}
IMG=airan:25-3-final
MIG_PCTS=${MIG_PCTS:-"30 50 70 100"}
TRIALS=${TRIALS:-3}
ITERS=${ITERS:-1000}
WARMUP=${WARMUP:-100}
MPS_PIPE=/tmp/nvidia-mps-mig4g
MPS_LOG=/tmp/nvidia-mps-mig4g-log

cleanup() {
    docker rm -f mig_mps_bench >/dev/null 2>&1 || true
    if [[ -S "$MPS_PIPE/control" ]]; then
        echo quit | sudo env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
            nvidia-cuda-mps-control >/dev/null 2>&1 || true
    fi
    sudo pkill -f nvidia-cuda-mps-server >/dev/null 2>&1 || true
    sudo pkill -f nvidia-cuda-mps-control >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

MIG_4G=$(nvidia-smi -L | awk '/MIG 4g/{print $6}' | tr -d ')' | head -1)
MIG_3G=$(nvidia-smi -L | awk '/MIG 3g/{print $6}' | tr -d ')' | head -1)
[[ -n "$MIG_4G" && -n "$MIG_3G" ]]
sudo mkdir -p "$RESULTS_ROOT" "$MPS_PIPE" "$MPS_LOG"
sudo chmod 0777 "$RESULTS_ROOT" "$MPS_PIPE" "$MPS_LOG"
cleanup
sudo mkdir -p "$MPS_PIPE" "$MPS_LOG"
sudo chmod 0777 "$MPS_PIPE" "$MPS_LOG"
sudo env CUDA_VISIBLE_DEVICES="$MIG_4G" \
    CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$MPS_LOG" \
    nvidia-cuda-mps-control -d </dev/null >/tmp/mig_mps_daemon.log 2>&1 &
sleep 3
[[ -S "$MPS_PIPE/control" ]] || { cat /tmp/mig_mps_daemon.log >&2; exit 1; }

run_bench() {
    local mode=$1 label=$2 pct=$3 outdir=$4
    mkdir -p "$outdir"; chmod 0777 "$outdir"
    docker run --rm --name mig_mps_bench --runtime=nvidia --ipc=host \
        -e NVIDIA_VISIBLE_DEVICES="$MIG_4G" \
        -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$pct" \
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

for pct in $MIG_PCTS; do
    for trial in $(seq 1 "$TRIALS"); do
        run_bench standalone "mig_mps_p${pct}_l1_t${trial}" "$pct" \
            "$RESULTS_ROOT/pct${pct}/trial${trial}/l1_only"
        run_bench same "mig_mps_p${pct}_same_t${trial}" "$pct" \
            "$RESULTS_ROOT/pct${pct}/trial${trial}/same_overlap"
    done
    echo "[MIG-MPS-DIRECT] pct=$pct complete"
done
date -u +%FT%TZ >"$RESULTS_ROOT/COMPLETE"
echo "[MIG-MPS-DIRECT] OK MIG4g=$MIG_4G sibling3g=$MIG_3G"
