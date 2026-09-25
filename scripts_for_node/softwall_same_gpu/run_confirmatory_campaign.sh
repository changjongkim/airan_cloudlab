#!/usr/bin/env bash

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh

require_allocation
softwall_mps_configure
softwall_mps_assert

rounds=${1:-5}
iterations=${2:-500}
campaign=${3:-confirm}
[[ "$campaign" =~ ^[a-z0-9_]+$ ]] || {
    echo "campaign tag must contain only lowercase letters, digits, and underscores" >&2
    exit 2
}
period_ms=${PERIOD_MS:-50}
deadline_ms=${DEADLINE_MS:-25}
ai_budget_ms=${AI_BUDGET_MS:-5}
guard_ms=${GUARD_MS:-1}
seed_base=${SEED_BASE:-20260920}
cells=${CELLS:-1}
ran_mode=${RAN_MODE:-synthetic}
case "$ran_mode" in
    synthetic) ran_script=/softwall/ran_fullpath.py ;;
    paired) ran_script=/softwall/paired_ran_fullpath.py ;;
    *) echo "unsupported RAN mode: $ran_mode" >&2; exit 2 ;;
esac
IFS=',' read -r -a workloads <<<"${WORKLOADS:-gemm,hbm}"
for kind in "${workloads[@]}"; do
    [[ "$kind" =~ ^(gemm|hbm|nrx|qwen)$ ]] || {
        echo "unsupported workload: $kind" >&2
        exit 2
    }
done
qwen_model=${QWEN_MODEL:-Qwen/Qwen2.5-1.5B}
qwen_phase=${QWEN_PHASE:-prefill}
qwen_context_len=${QWEN_CONTEXT_LEN:-16}
qwen_batch_size=${QWEN_BATCH_SIZE:-1}
raw="$SOFTWALL_ROOT/results/softwall_same_gpu/raw"
state_root="$SOFTWALL_ROOT/run_state/softwall_same_gpu/${campaign}_job$SLURM_JOB_ID"
mkdir -p "$raw" "$state_root"

children=()
cleanup() {
    local pid
    for pid in "${children[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

launch_ai() {
    local kind=$1 mode=$2 output=$3
    shift 3
    if [[ "$kind" == qwen ]]; then
        shifter_qwen python3 /softwall/qwen_worker.py \
            --mode "$mode" --model "$qwen_model" --phase "$qwen_phase" \
            --context-length "$qwen_context_len" --batch-size "$qwen_batch_size" \
            --output "$output" "$@"
    else
        shifter_gpu python3 /softwall/ai_worker.py \
            --kind "$kind" --mode "$mode" --output "$output" "$@"
    fi
}

run_ran_alone() {
    local round=$1 seed=$2 prefix="${campaign}_r${round}_mps_alone_job${SLURM_JOB_ID}"
    [[ -s "$raw/${prefix}.json" ]] && return
    shifter_gpu python3 "$ran_script" \
        --label "${campaign}_r${round}_mps_alone" \
        --output "$raw/${prefix}.json" \
        --cells "$cells" --iterations "$iterations" --warmup 30 --period-ms "$period_ms" \
        --deadline-ms "$deadline_ms" --seed "$seed"
}

run_uncontrolled() {
    local round=$1 seed=$2 kind=$3
    local prefix="${campaign}_r${round}_uncontrolled_${kind}_job${SLURM_JOB_ID}"
    local ran_output="$raw/${prefix}_ran.json"
    local ai_output="$raw/${prefix}_ai.json"
    [[ -s "$ran_output" && -s "$ai_output" ]] && return
    local state="$state_root/${prefix}"
    mkdir -p "$state"
    rm -f "$state/ai.ready" "$state/ran.ready" "$state/start" "$state/stop"
    rm -f "$ran_output" "$ai_output"
    local duration_s
    duration_s=$(python3 -c "print(($iterations * $period_ms) / 1000.0 + 2.0)")
    launch_ai "$kind" continuous "$ai_output" --duration-s "$duration_s" \
        --ready-file "$state/ai.ready" \
        --start-file "$state/start" --stop-file "$state/stop" &
    local ai_pid=$!
    children+=("$ai_pid")
    shifter_gpu python3 "$ran_script" \
        --label "${campaign}_r${round}_uncontrolled_${kind}" \
        --output "$ran_output" --cells "$cells" --iterations "$iterations" --warmup 30 \
        --period-ms "$period_ms" --deadline-ms "$deadline_ms" --seed "$seed" \
        --ready-file "$state/ran.ready" --start-file "$state/start" &
    local ran_pid=$!
    children+=("$ran_pid")
    local attempt
    for attempt in {1..1200}; do
        if [[ -e "$state/ai.ready" && -e "$state/ran.ready" ]]; then
            break
        fi
        kill -0 "$ai_pid" 2>/dev/null && kill -0 "$ran_pid" 2>/dev/null || {
            echo "paired worker exited before ready: $prefix" >&2
            return 1
        }
        sleep 0.05
    done
    [[ -e "$state/ai.ready" && -e "$state/ran.ready" ]] || {
        echo "paired worker readiness timed out: $prefix" >&2
        return 1
    }
    touch "$state/start"
    wait "$ran_pid"
    touch "$state/stop"
    wait "$ai_pid"
    children=()
}

run_protected() {
    local round=$1 seed=$2 kind=$3
    local prefix="${campaign}_r${round}_protected_${kind}_job${SLURM_JOB_ID}"
    local ran_output="$raw/${prefix}_ran.json"
    local ai_output="$raw/${prefix}_ai.json"
    [[ -s "$ran_output" && -s "$ai_output" ]] && return
    local state="$state_root/${prefix}"
    # AF_UNIX paths are limited to roughly 108 bytes on Linux. Keep the socket
    # name short even though result and marker paths remain descriptive.
    local socket="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/r${round}_${kind}.sock"
    mkdir -p "$state"
    rm -f "$socket" "$ran_output" "$ai_output"
    launch_ai "$kind" rpc "$ai_output" --socket "$socket" &
    local ai_pid=$!
    children+=("$ai_pid")
    local attempt
    for attempt in {1..3600}; do
        [[ -S "$socket" ]] && break
        kill -0 "$ai_pid" 2>/dev/null || {
            echo "RPC worker exited before socket creation: $prefix" >&2
            return 1
        }
        sleep 0.05
    done
    [[ -S "$socket" ]] || {
        echo "RPC worker socket timed out: $prefix" >&2
        return 1
    }
    shifter_gpu python3 /softwall/protected_runner.py \
        --label "${campaign}_r${round}_protected_${kind}" --socket "$socket" \
        --output "$ran_output" --cells "$cells" --iterations "$iterations" --warmup 30 \
        --period-ms "$period_ms" --deadline-ms "$deadline_ms" \
        --ai-budget-ms "$ai_budget_ms" --guard-ms "$guard_ms" --seed "$seed" \
        --ran-input "$ran_mode"
    wait "$ai_pid"
    children=()
}

for ((round=1; round<=rounds; round++)); do
    seed=$((seed_base + round))
    run_ran_alone "$round" "$seed"
    if (( round % 2 == 1 )); then
        ordered=("${workloads[@]}")
    else
        ordered=()
        for ((index=${#workloads[@]}-1; index>=0; index--)); do
            ordered+=("${workloads[index]}")
        done
    fi
    for kind in "${ordered[@]}"; do
        run_uncontrolled "$round" "$seed" "$kind"
        run_protected "$round" "$seed" "$kind"
    done
done

trap - EXIT INT TERM
echo "campaign completed: tag=$campaign rounds=$rounds iterations=$iterations job=$SLURM_JOB_ID"
