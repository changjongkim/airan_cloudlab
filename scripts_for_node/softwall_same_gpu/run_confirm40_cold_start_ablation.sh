#!/usr/bin/env bash

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh

require_allocation
softwall_mps_configure
raw="$SOFTWALL_ROOT/results/softwall_same_gpu/raw"
result="$SOFTWALL_ROOT/results/softwall_same_gpu"
state="$SOFTWALL_ROOT/run_state/softwall_same_gpu/job${SLURM_JOB_ID}"
mkdir -p "$raw" "$state"
gpu_lock_dir="$raw/.lock_gpu0_job${SLURM_JOB_ID}"
mkdir "$gpu_lock_dir" 2>/dev/null || {
    echo "another GPU0 experiment is already running in job $SLURM_JOB_ID" >&2
    exit 4
}
worker_pid=""
cleanup() {
    if [[ -n "$worker_pid" ]]; then kill "$worker_pid" 2>/dev/null || true; fi
    softwall_mps_stop || true
    rmdir "$gpu_lock_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

softwall_mps_stop
sleep 20
softwall_mps_start
softwall_mps_assert

for round in 1 2 3 4; do
    case "$round" in
        1|4) mode=off ;;
        2|3) mode=on ;;
    esac
    tag="cold_${SLURM_JOB_ID}_${round}_${BASHPID}"
    prefix="confirm40_cold_start_r${round}_${mode}_job${SLURM_JOB_ID}"
    controller_output="$raw/${prefix}_controller.json"
    worker_output="$raw/${prefix}_worker.json"
    shifter_gpu env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80 \
        CUDA_MPS_CLIENT_PRIORITY=1 \
        python3 /softwall/same_request_ipc_worker.py \
        --tag "$tag" --ipc-dir "$state" \
        --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
        --output "$worker_output" &
    worker_pid=$!
    shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
        python3 /softwall/same_request_ipc_controller.py \
        --tag "$tag" --ipc-dir "$state" \
        --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
        --output "$controller_output" --iterations 500 --warmup 20 \
        --period-ms 90 --deadline-ms 80 --endpoint-timeout-ms 30 \
        --seed 20321035 --snr-db -8.5 --channel-seed-base 20325000 \
        --policy s2 --conventional-warmup "$mode"
    wait "$worker_pid"
    worker_pid=""
    sleep 20
done

shifter_gpu python3 /softwall/analyze_cold_start_ablation.py \
    --raw "$raw" --job "$SLURM_JOB_ID" \
    --protocol "$result/confirm40_cold_start_ablation_protocol.json" \
    --output "$result/confirm40_cold_start_ablation.md"

cleanup
trap - EXIT INT TERM
echo 'confirm40 cold-start ablation completed'
