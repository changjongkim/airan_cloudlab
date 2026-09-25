#!/usr/bin/env bash

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh

require_allocation
softwall_mps_configure
softwall_mps_assert

rounds=${1:-3}
campaign=${2:-confirm31_maintenance_requalification}
pre_iterations=${PRE_ITERATIONS:-100}
post_iterations=${POST_ITERATIONS:-1000}
quiet_seconds=${QUIET_SECONDS:-20}
repeats=${OVERRUN_REPEATS:-40}
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
    mkdir "$gpu_lock_dir" 2>/dev/null || {
        echo "another GPU0 experiment is already running in job $SLURM_JOB_ID" >&2
        exit 4
    }
    owns_gpu_lock=1
fi
lock_dir="$raw/.lock_${campaign}"
mkdir "$lock_dir" 2>/dev/null || {
    echo "campaign already running: $campaign" >&2
    if ((owns_gpu_lock)); then rmdir "$gpu_lock_dir" 2>/dev/null || true; fi
    exit 3
}

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
    for cap in "${caps[@]}"; do
        prefix="${campaign}_r${round}_cap${cap}_job${SLURM_JOB_ID}"
        pre_output="$raw/${prefix}_pre_ran.json"
        worker_output="$raw/${prefix}_worker.json"
        post_output="$raw/${prefix}_post_ran.json"
        socket="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/mq_${BASHPID}_${round}_${cap}.sock"
        if [[ -s "$pre_output" && -s "$worker_output" && -s "$post_output" ]]; then
            echo "maintenance campaign resume: keeping $prefix"
            continue
        fi
        rm -f "$socket" "$pre_output" "$worker_output" "$post_output"
        shifter_gpu env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$cap" \
            CUDA_MPS_CLIENT_PRIORITY=1 \
            python3 /softwall/ai_worker.py --kind nrx --mode rpc \
            --repeats "$repeats" --socket "$socket" --output "$worker_output" &
        worker_pid=$!
        children+=("$worker_pid")
        for _ in {1..1200}; do
            [[ -S "$socket" ]] && break
            kill -0 "$worker_pid" 2>/dev/null || {
                echo "maintenance worker exited before ready" >&2
                exit 1
            }
            sleep 0.05
        done
        [[ -S "$socket" ]] || {
            echo "maintenance worker readiness timeout" >&2
            exit 1
        }
        shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
            python3 /softwall/overrun_controller.py \
            --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
            --socket "$socket" --output "$pre_output" \
            --iterations "$pre_iterations" --warmup 20 \
            --period-ms 60 --deadline-ms 35 --timeout-ms 5 \
            --drain-guard-ms 2 --fault-every 100000 \
            --seed "$((20360000 + round))"
        wait "$worker_pid"
        children=()

        sleep "$quiet_seconds"
        shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
            python3 /softwall/s2_runner.py --policy conventional_only \
            --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
            --output "$post_output" --iterations "$post_iterations" \
            --warmup 20 --cells 1 --period-ms 60 --deadline-ms 35 \
            --nrx-bound-ms 50 --conv-bound-ms 35 --commit-guard-ms 2 \
            --inject-failure-every 0 --seed "$((20370000 + round))" \
            --channel-seed-base "$((20380000 + round * 100000))"
    done
done

cleanup
trap - EXIT INT TERM
echo "maintenance requalification campaign completed: $campaign"
