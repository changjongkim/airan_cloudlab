#!/usr/bin/env bash

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh

require_allocation
softwall_mps_configure
softwall_mps_assert

rounds=${1:-3}
iterations=${2:-500}
campaign=${3:-confirm16_s2}
[[ "$campaign" =~ ^[a-z0-9_]+$ ]] || {
    echo "campaign tag must contain only lowercase letters, digits, and underscores" >&2
    exit 2
}
cells=${CELLS:-1}
period_ms=${PERIOD_MS:-50}
deadline_ms=${DEADLINE_MS:-40}
nrx_bound_ms=${NRX_BOUND_MS:-14}
conv_bound_ms=${CONV_BOUND_MS:-24}
commit_guard_ms=${COMMIT_GUARD_MS:-2}
failure_every=${FAILURE_EVERY:-10}
snr_db=${SNR_DB:-}
channel_seed_base=${CHANNEL_SEED_BASE:-20270750}
channel_seed_stride=${CHANNEL_SEED_STRIDE:-100000}
ai_budget_ms=${AI_BUDGET_MS:-6}
ai_guard_ms=${AI_GUARD_MS:-1}
ai_timeout_ms=${AI_RPC_TIMEOUT_MS:-7}
quiet_seconds=${QUIET_SECONDS:-0}
background_cap=${BACKGROUND_CAP:-100}
[[ "$background_cap" =~ ^[0-9]+$ ]] && ((background_cap >= 1 && background_cap <= 100)) || {
    echo "invalid background MPS cap: $background_cap" >&2
    exit 2
}
start_delay_every=${START_DELAY_EVERY:-0}
start_delay_ms=${START_DELAY_MS:-0}
seed_base=${SEED_BASE:-20270500}
snr_args=()
[[ -n "$snr_db" ]] && snr_args=(--snr-db "$snr_db")
IFS=',' read -r -a policies <<<"${POLICIES:-conventional_only,nrx_only,eager_dual,s2_reserved}"
for policy in "${policies[@]}"; do
    [[ "$policy" =~ ^(conventional_only|nrx_only|eager_dual|s2_reserved)$ ]] || {
        echo "unsupported S2 policy: $policy" >&2
        exit 2
    }
done

raw="$SOFTWALL_ROOT/results/softwall_same_gpu/raw"
mkdir -p "$raw"
children=()
cleanup() {
    local pid
    for pid in "${children[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

run_policy() {
    local round=$1 seed=$2 policy=$3
    local completed_ran completed_ai
    for completed_ran in "$raw/${campaign}_r${round}_${policy}_job"*_ran.json; do
        [[ -s "$completed_ran" ]] || continue
        completed_ai=${completed_ran%_ran.json}_ai.json
        if [[ -s "$completed_ai" ]]; then
            echo "S2 campaign resume: keeping completed $(basename "$completed_ran")"
            return
        fi
    done
    local prefix="${campaign}_r${round}_${policy}_job${SLURM_JOB_ID}"
    local ran_output="$raw/${prefix}_ran.json"
    local ai_output="$raw/${prefix}_ai.json"
    local socket="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/s2_${round}_${policy:0:4}.sock"
    [[ -s "$ran_output" && -s "$ai_output" ]] && return
    rm -f "$socket" "$ran_output" "$ai_output"
    shifter_gpu env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$background_cap" \
        CUDA_MPS_CLIENT_PRIORITY=1 python3 /softwall/ai_worker.py \
        --kind nrx --mode rpc --socket "$socket" --output "$ai_output" &
    local ai_pid=$!
    children+=("$ai_pid")
    local attempt
    for attempt in {1..1200}; do
        [[ -S "$socket" ]] && break
        kill -0 "$ai_pid" 2>/dev/null || {
            echo "S2 background worker exited before ready: $prefix" >&2
            return 1
        }
        sleep 0.05
    done
    [[ -S "$socket" ]] || {
        echo "S2 background worker readiness timed out: $prefix" >&2
        return 1
    }
    shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
        python3 /softwall/s2_runner.py \
        --policy "$policy" --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
        --output "$ran_output" --iterations "$iterations" --warmup 20 \
        --cells "$cells" --period-ms "$period_ms" --deadline-ms "$deadline_ms" \
        --nrx-bound-ms "$nrx_bound_ms" --conv-bound-ms "$conv_bound_ms" \
        --commit-guard-ms "$commit_guard_ms" \
        --inject-failure-every "$failure_every" --seed "$seed" \
        --socket "$socket" --ai-budget-ms "$ai_budget_ms" \
        --ai-guard-ms "$ai_guard_ms" --ai-rpc-timeout-ms "$ai_timeout_ms" \
        --channel-seed-base "$((channel_seed_base + round * channel_seed_stride))" \
        --inject-start-delay-every "$start_delay_every" \
        --inject-start-delay-ms "$start_delay_ms" \
        "${snr_args[@]}"
    wait "$ai_pid"
    children=()
}

for ((round=1; round<=rounds; round++)); do
    seed=$((seed_base + round))
    if (( round % 2 == 1 )); then
        ordered=("${policies[@]}")
    else
        ordered=()
        for ((index=${#policies[@]}-1; index>=0; index--)); do
            ordered+=("${policies[index]}")
        done
    fi
    for policy in "${ordered[@]}"; do
        run_policy "$round" "$seed" "$policy"
        if [[ "$quiet_seconds" != 0 ]]; then sleep "$quiet_seconds"; fi
    done
done

trap - EXIT INT TERM
echo "S2 campaign completed: tag=$campaign rounds=$rounds iterations=$iterations job=$SLURM_JOB_ID"
