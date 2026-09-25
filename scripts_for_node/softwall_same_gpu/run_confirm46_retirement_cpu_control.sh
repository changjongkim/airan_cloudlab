#!/usr/bin/env bash

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh

require_allocation
softwall_mps_configure
raw="$SOFTWALL_ROOT/results/softwall_same_gpu/raw"
result="$SOFTWALL_ROOT/results/softwall_same_gpu"
mkdir -p "$raw"
gpu_lock_dir="$raw/.lock_gpu0_job${SLURM_JOB_ID}"
mkdir "$gpu_lock_dir" 2>/dev/null || {
    echo "another GPU0 experiment is already running in job $SLURM_JOB_ID" >&2
    exit 4
}
worker_pid=""
socket=""
cleanup() {
    if [[ -n "$worker_pid" ]]; then kill "$worker_pid" 2>/dev/null || true; fi
    if [[ -n "$socket" ]]; then rm -f "$socket"; fi
    softwall_mps_stop || true
    rmdir "$gpu_lock_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

softwall_mps_stop
sleep 20
softwall_mps_start
softwall_mps_assert

requal="$raw/confirm46_requalification_job${SLURM_JOB_ID}.json"
shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/s2_runner.py --policy conventional_only \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$requal" --iterations 200 --warmup 20 --cells 1 \
    --period-ms 60 --deadline-ms 35 --nrx-bound-ms 50 \
    --conv-bound-ms 35 --commit-guard-ms 2 --inject-failure-every 0 \
    --seed 20321946 --channel-seed-base 20325946
python3 - "$requal" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
if x['iterations'] != 200 or x['deadline_misses']:
    raise SystemExit('confirm46 operational requalification failed')
print('confirm46 operational requalification passed')
PY

for round in {1..10}; do
    seed=$((20320000 + round))
    if ((round % 2)); then
        order="cpu_sham mps_sham"
    else
        order="mps_sham cpu_sham"
    fi
    for condition in $order; do
        sleep 20
        prefix="confirm46_r${round}_${condition}_job${SLURM_JOB_ID}"
        ran_output="$raw/${prefix}_ran.json"
        worker_output="$raw/${prefix}_worker.json"
        socket="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/c46_${BASHPID}_${round}_${condition}.sock"
        rm -f "$socket" "$ran_output" "$worker_output"
        if [[ "$condition" == cpu_sham ]]; then
            shifter_gpu python3 /softwall/ai_worker_cpu_sham.py \
                --socket "$socket" --output "$worker_output" &
        else
            shifter_gpu env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20 \
                CUDA_MPS_CLIENT_PRIORITY=1 \
                python3 /softwall/ai_worker.py --kind nrx --mode rpc \
                --repeats 40 --socket "$socket" --output "$worker_output" &
        fi
        worker_pid=$!
        for _ in {1..1200}; do
            [[ -S "$socket" ]] && break
            kill -0 "$worker_pid" 2>/dev/null || {
                echo "confirm46 worker exited before ready" >&2
                exit 1
            }
            sleep 0.05
        done
        [[ -S "$socket" ]] || { echo 'confirm46 worker readiness timeout' >&2; exit 1; }

        shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
            python3 /softwall/overrun_controller.py \
            --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
            --socket "$socket" --output "$ran_output" \
            --iterations 1000 --warmup 20 \
            --period-ms 60 --deadline-ms 35 \
            --timeout-ms 5 --drain-guard-ms 2 \
            --fault-every 100000 --seed "$seed" \
            --retire-before-first-release
        wait "$worker_pid"
        worker_pid=""
        socket=""
    done
done

shifter_gpu python3 /softwall/analyze_confirm46_retirement_cpu_control.py \
    --raw "$raw" --job "$SLURM_JOB_ID" \
    --protocol "$result/confirm46_retirement_cpu_control_protocol.json" \
    --output "$result/confirm46_retirement_cpu_control.md"

cleanup
trap - EXIT INT TERM
echo 'confirm46 retirement CPU control completed'
