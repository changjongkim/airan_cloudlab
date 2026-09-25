#!/usr/bin/env bash

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh

require_allocation
softwall_mps_configure
softwall_mps_assert

rounds=${1:-3}
iterations=${2:-500}
campaign=${3:-smoke_overrun}
round_offset=${ROUND_OFFSET:-0}
repeats=${OVERRUN_REPEATS:-40}
fault_every=${FAULT_EVERY:-10}
period_ms=${PERIOD_MS:-60}
deadline_ms=${DEADLINE_MS:-25}
timeout_ms=${TIMEOUT_MS:-5}
drain_guard_ms=${DRAIN_GUARD_MS:-2}
seed_base=${SEED_BASE:-20320000}
retire_after_timeout=${RETIRE_AFTER_TIMEOUT:-0}
retire_before_first_release=${RETIRE_BEFORE_FIRST_RELEASE:-0}
IFS=',' read -r -a caps <<<"${CAPS:-100,20}"
raw="$SOFTWALL_ROOT/results/softwall_same_gpu/raw"
mkdir -p "$raw"
gpu_lock_dir="$raw/.lock_gpu0_job${SLURM_JOB_ID}"
owns_gpu_lock=0
if [[ "${SOFTWALL_GPU_LOCK_HELD:-0}" == 1 ]]; then
    [[ -d "$gpu_lock_dir" ]] || {
        echo "declared GPU0 lock is absent" >&2
        exit 4
    }
else
    if ! mkdir "$gpu_lock_dir" 2>/dev/null; then
        echo "another GPU0 experiment is already running in job $SLURM_JOB_ID" >&2
        exit 4
    fi
    owns_gpu_lock=1
fi
lock_dir="$raw/.lock_${campaign}"
if ! mkdir "$lock_dir" 2>/dev/null; then
    echo "campaign already running: $campaign" >&2
    if ((owns_gpu_lock)); then rmdir "$gpu_lock_dir" 2>/dev/null || true; fi
    exit 3
fi

children=()
cleanup() {
    local pid
    for pid in "${children[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    rmdir "$lock_dir" 2>/dev/null || true
    if ((owns_gpu_lock)); then rmdir "$gpu_lock_dir" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

for ((round=1; round<=rounds; round++)); do
    actual_round=$((round + round_offset))
    for cap in "${caps[@]}"; do
        [[ "$cap" =~ ^[0-9]+$ ]] && ((cap >= 1 && cap <= 100)) || {
            echo "invalid MPS cap: $cap" >&2
            exit 2
        }
        prefix="${campaign}_r${actual_round}_cap${cap}_job${SLURM_JOB_ID}"
        ran_output="$raw/${prefix}_ran.json"
        worker_output="$raw/${prefix}_worker.json"
        socket="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/ov_${BASHPID}_${actual_round}_${cap}.sock"
        rm -f "$socket" "$ran_output" "$worker_output"
        shifter_gpu env \
            CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$cap" \
            CUDA_MPS_CLIENT_PRIORITY=1 \
            python3 /softwall/ai_worker.py \
            --kind nrx --mode rpc --repeats "$repeats" \
            --socket "$socket" --output "$worker_output" &
        worker_pid=$!
        children+=("$worker_pid")
        for _ in {1..1200}; do
            [[ -S "$socket" ]] && break
            kill -0 "$worker_pid" 2>/dev/null || {
                echo "overrun worker exited before ready" >&2
                exit 1
            }
            sleep 0.05
        done
        [[ -S "$socket" ]] || {
            echo "overrun worker readiness timeout" >&2
            exit 1
        }
        controller_args=()
        if [[ "$retire_after_timeout" == "1" ]]; then
            controller_args+=(--retire-after-timeout)
        fi
        if [[ "$retire_before_first_release" == "1" ]]; then
            controller_args+=(--retire-before-first-release)
        fi
        shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
            python3 /softwall/overrun_controller.py \
            --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
            --socket "$socket" --output "$ran_output" \
            --iterations "$iterations" --warmup 20 \
            --period-ms "$period_ms" --deadline-ms "$deadline_ms" \
            --timeout-ms "$timeout_ms" --drain-guard-ms "$drain_guard_ms" \
            --fault-every "$fault_every" --seed "$((seed_base + actual_round))" \
            "${controller_args[@]}"
        wait "$worker_pid"
        children=()
    done
done

cleanup
trap - EXIT INT TERM
echo "physical overrun campaign completed: $campaign"
