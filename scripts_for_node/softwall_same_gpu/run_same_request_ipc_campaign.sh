#!/usr/bin/env bash

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh

require_allocation
softwall_mps_configure
softwall_mps_assert

rounds=${1:-3}
iterations=${2:-1000}
campaign=${3:-confirm32_same_request_ipc}
warmup=${WARMUP:-20}
period_ms=${PERIOD_MS:-60}
deadline_ms=${DEADLINE_MS:-35}
timeout_ms=${ENDPOINT_TIMEOUT_MS:-5}
snr_db=${SNR_DB:--8.5}
seed_base=${SEED_BASE:-20321000}
channel_seed_base=${CHANNEL_SEED_BASE:-20322000}
endpoint_cap=${ENDPOINT_CAP:-80}
fault_delay_ms=${FAULT_DELAY_MS:-30}
raw="$SOFTWALL_ROOT/results/softwall_same_gpu/raw"
ipc_dir="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/ipc_confirm32"
lock_dir="$raw/.lock_${campaign}"
mkdir -p "$raw" "$ipc_dir"
mkdir "$lock_dir" 2>/dev/null || {
    echo "campaign already running: $campaign" >&2
    exit 3
}

children=()
cleanup() {
    local pid
    for pid in "${children[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    rmdir "$lock_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

run_ipc_condition() {
    local round=$1 seed=$2 channel_seed=$3 condition=$4 delay_sequence=$5
    local prefix="${campaign}_r${round}_${condition}_job${SLURM_JOB_ID}"
    local controller_output="$raw/${prefix}_controller.json"
    local worker_output="$raw/${prefix}_worker.json"
    if [[ -s "$controller_output" && -s "$worker_output" ]]; then
        echo "same-request resume: keeping $prefix"
        return
    fi
    local tag="sr_${SLURM_JOB_ID}_${round}_${condition}"
    rm -f "$controller_output" "$worker_output" \
        "$ipc_dir/cuda_ipc_${tag}.info" "$ipc_dir/cuda_ipc_${tag}.ctrl"
    shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
        python3 /softwall/same_request_ipc_controller.py \
        --tag "$tag" --ipc-dir "$ipc_dir" \
        --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
        --output "$controller_output" --iterations "$iterations" \
        --warmup "$warmup" --period-ms "$period_ms" \
        --deadline-ms "$deadline_ms" --endpoint-timeout-ms "$timeout_ms" \
        --seed "$seed" --snr-db "$snr_db" \
        --channel-seed-base "$channel_seed" &
    local controller_pid=$!
    children+=("$controller_pid")
    shifter_gpu env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$endpoint_cap" \
        CUDA_MPS_CLIENT_PRIORITY=1 \
        python3 /softwall/same_request_ipc_worker.py \
        --tag "$tag" --ipc-dir "$ipc_dir" \
        --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
        --output "$worker_output" --delay-sequence "$delay_sequence" \
        --delay-ms "$fault_delay_ms" &
    local worker_pid=$!
    children+=("$worker_pid")
    wait "$controller_pid"
    wait "$worker_pid"
    children=()
}

for ((round=1; round<=rounds; round++)); do
    seed=$((seed_base + round))
    channel_seed=$((channel_seed_base + round * 100000))
    baseline="$raw/${campaign}_r${round}_conventional_job${SLURM_JOB_ID}.json"
    if [[ ! -s "$baseline" ]]; then
        shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
            python3 /softwall/s2_runner.py --policy conventional_only \
            --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
            --output "$baseline" --iterations "$iterations" \
            --warmup "$warmup" --cells 1 --period-ms "$period_ms" \
            --deadline-ms "$deadline_ms" --nrx-bound-ms 20 \
            --conv-bound-ms 20 --commit-guard-ms 2 \
            --inject-failure-every 0 --seed "$seed" \
            --snr-db "$snr_db" --channel-seed-base "$channel_seed"
    fi
    run_ipc_condition "$round" "$seed" "$channel_seed" ipc_s2 0
    noisy_warmup=0
    [[ -n "$snr_db" ]] && noisy_warmup=1
    run_ipc_condition "$round" "$seed" "$channel_seed" ipc_timeout \
        "$((warmup + noisy_warmup + 1))"
done

cleanup
trap - EXIT INT TERM
echo "same-request IPC campaign completed: $campaign"
