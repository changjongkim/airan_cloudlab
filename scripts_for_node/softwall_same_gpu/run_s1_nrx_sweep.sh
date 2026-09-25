#!/usr/bin/env bash

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh

require_allocation
softwall_mps_configure
softwall_mps_assert

rounds=${1:-3}
iterations=${2:-500}
campaign=${3:-s1_nrx}
ai_kind=${AI_KIND:-nrx}
[[ "$ai_kind" =~ ^(nrx|qwen)$ ]] || {
    echo "unsupported S1 AI kind: $ai_kind" >&2
    exit 2
}
qwen_model=${QWEN_MODEL:-Qwen/Qwen2.5-1.5B}
qwen_phase=${QWEN_PHASE:-prefill}
qwen_context_len=${QWEN_CONTEXT_LEN:-16}
qwen_batch_size=${QWEN_BATCH_SIZE:-1}
period_ms=${PERIOD_MS:-50}
deadline_ms=${DEADLINE_MS:-35}
seed_base=${SEED_BASE:-20263000}
cells=${CELLS:-1}
ran_mode=${RAN_MODE:-synthetic}
case "$ran_mode" in
    synthetic) ran_script=/softwall/ran_fullpath.py ;;
    paired) ran_script=/softwall/paired_ran_fullpath.py ;;
    *) echo "unsupported RAN mode: $ran_mode" >&2; exit 2 ;;
esac
IFS=',' read -r -a configurations <<<"${S1_CONFIGS:-100:0,100:1,40:1,30:1,20:1,10:1}"
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

run_ran_alone() {
    local round=$1 seed=$2
    local output="$raw/${campaign}_r${round}_alone_job${SLURM_JOB_ID}.json"
    [[ -s "$output" ]] && return
    (
        unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
        export CUDA_MPS_CLIENT_PRIORITY=0
        shifter_gpu python3 "$ran_script" \
            --label "${campaign}_r${round}_alone" --output "$output" \
            --cells "$cells" --iterations "$iterations" --warmup 30 --period-ms "$period_ms" \
            --deadline-ms "$deadline_ms" --seed "$seed"
    )
}

run_overlap() {
    local round=$1 seed=$2 cap=$3 priority=$4
    local condition="cap${cap}_p${priority}"
    local prefix="${campaign}_r${round}_${condition}_job${SLURM_JOB_ID}"
    local ran_output="$raw/${prefix}_ran.json"
    local ai_output="$raw/${prefix}_ai.json"
    [[ -s "$ran_output" && -s "$ai_output" ]] && return
    local state="$state_root/$prefix"
    mkdir -p "$state"
    rm -f "$state/ai.ready" "$state/ran.ready" "$state/start" "$state/stop"
    rm -f "$ran_output" "$ai_output"
    local duration_s
    duration_s=$(python3 -c "print(($iterations * $period_ms) / 1000.0 + 2.0)")
    (
        export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$cap"
        export CUDA_MPS_CLIENT_PRIORITY="$priority"
        if [[ "$ai_kind" == qwen ]]; then
            shifter_qwen python3 /softwall/qwen_worker.py --mode continuous \
                --model "$qwen_model" --phase "$qwen_phase" \
                --context-length "$qwen_context_len" --batch-size "$qwen_batch_size" \
                --duration-s "$duration_s" --output "$ai_output" \
                --ready-file "$state/ai.ready" --start-file "$state/start" \
                --stop-file "$state/stop"
        else
            shifter_gpu python3 /softwall/ai_worker.py --kind nrx --mode continuous \
                --duration-s "$duration_s" --output "$ai_output" \
                --ready-file "$state/ai.ready" --start-file "$state/start" \
                --stop-file "$state/stop"
        fi
    ) &
    local ai_pid=$!
    children+=("$ai_pid")
    (
        unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
        export CUDA_MPS_CLIENT_PRIORITY=0
        shifter_gpu python3 "$ran_script" \
            --label "${campaign}_r${round}_${condition}" --output "$ran_output" \
            --cells "$cells" --iterations "$iterations" --warmup 30 --period-ms "$period_ms" \
            --deadline-ms "$deadline_ms" --seed "$seed" \
            --ready-file "$state/ran.ready" --start-file "$state/start"
    ) &
    local ran_pid=$!
    children+=("$ran_pid")
    local attempt
    for attempt in {1..3600}; do
        [[ -e "$state/ai.ready" && -e "$state/ran.ready" ]] && break
        kill -0 "$ai_pid" 2>/dev/null && kill -0 "$ran_pid" 2>/dev/null || {
            echo "S1 worker exited before ready: $prefix" >&2
            return 1
        }
        sleep 0.05
    done
    [[ -e "$state/ai.ready" && -e "$state/ran.ready" ]] || {
        echo "S1 readiness timed out: $prefix" >&2
        return 1
    }
    touch "$state/start"
    wait "$ran_pid"
    touch "$state/stop"
    wait "$ai_pid"
    children=()
}

for ((round=1; round<=rounds; round++)); do
    seed=$((seed_base + round))
    run_ran_alone "$round" "$seed"
    if (( round % 2 == 1 )); then
        ordered=("${configurations[@]}")
    else
        ordered=()
        for ((index=${#configurations[@]}-1; index>=0; index--)); do
            ordered+=("${configurations[index]}")
        done
    fi
    for configuration in "${ordered[@]}"; do
        IFS=':' read -r cap priority <<<"$configuration"
        [[ "$cap" =~ ^[0-9]+$ && "$cap" -ge 1 && "$cap" -le 100 \
            && "$priority" =~ ^[01]$ ]] || {
            echo "invalid S1 configuration: $configuration" >&2
            exit 2
        }
        run_overlap "$round" "$seed" "$cap" "$priority"
    done
done

trap - EXIT INT TERM
echo "S1 sweep completed: tag=$campaign rounds=$rounds iterations=$iterations job=$SLURM_JOB_ID"
