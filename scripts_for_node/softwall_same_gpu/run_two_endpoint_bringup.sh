#!/usr/bin/env bash

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh

require_allocation
softwall_mps_configure
raw="$SOFTWALL_ROOT/results/softwall_same_gpu/raw"
state="$SOFTWALL_ROOT/run_state/softwall_same_gpu/job${SLURM_JOB_ID}"
mkdir -p "$raw" "$state"
gpu_lock_dir="$raw/.lock_gpu0_job${SLURM_JOB_ID}"
mkdir "$gpu_lock_dir" 2>/dev/null || {
    echo "another GPU0 experiment is already running in job $SLURM_JOB_ID" >&2
    exit 4
}
worker0_pid=""
worker1_pid=""
background_pid=""
tag="two_${SLURM_JOB_ID}_${BASHPID}"
background_socket="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/a.sock"
cleanup() {
    for pid in "$worker0_pid" "$worker1_pid" "$background_pid"; do
        if [[ -n "$pid" ]]; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    rm -f "$background_socket"
    softwall_mps_stop || true
    rmdir "$gpu_lock_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

softwall_mps_stop
sleep 20
softwall_mps_start
softwall_mps_assert

requal="$raw/confirm55_requalification_job${SLURM_JOB_ID}.json"
shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/s2_runner.py --policy conventional_only \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$requal" --iterations 200 --warmup 20 --cells 1 \
    --period-ms 90 --deadline-ms 80 --nrx-bound-ms 50 \
    --conv-bound-ms 25 --commit-guard-ms 2 --inject-failure-every 0 \
    --seed 20321955 --snr-db -8.5 --channel-seed-base 20359900
python3 - "$requal" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
if x['iterations'] != 200 or x['deadline_misses']:
    raise SystemExit('confirm55 operational requalification failed')
print('confirm55 operational requalification passed', flush=True)
PY

sleep 20
for cell in 0 1; do
    output="$raw/confirm55_two_endpoint_job${SLURM_JOB_ID}_worker${cell}.json"
    shifter_gpu env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40 CUDA_MPS_CLIENT_PRIORITY=1 \
        python3 /softwall/same_request_ipc_worker.py \
        --tag "${tag}_c${cell}" --ipc-dir "$state" \
        --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
        --output "$output" &
    if [[ "$cell" == 0 ]]; then worker0_pid=$!; else worker1_pid=$!; fi
done
shifter_gpu env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20 CUDA_MPS_CLIENT_PRIORITY=1 \
    python3 /softwall/ai_worker.py --kind nrx --mode rpc \
    --repeats 1 --socket "$background_socket" \
    --output "$raw/confirm55_two_endpoint_job${SLURM_JOB_ID}_background.json" &
background_pid=$!
for _ in {1..1200}; do
    [[ -S "$background_socket" ]] && break
    kill -0 "$background_pid" 2>/dev/null || {
        echo "background worker exited before ready" >&2
        exit 1
    }
    sleep 0.05
done
[[ -S "$background_socket" ]] || {
    echo 'background worker readiness timeout' >&2
    exit 1
}

shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/two_endpoint_bringup_controller.py \
    --tag-prefix "$tag" --ipc-dir "$state" \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --socket "$background_socket" \
    --output "$raw/confirm55_two_endpoint_job${SLURM_JOB_ID}_controller.json" \
    --iterations 1000 --warmup 20 --period-ms 150 --deadline-ms 130 \
    --nrx-bound-ms 50 --conv-bound-ms 25 --commit-guard-ms 2 \
    --endpoint-timeout-ms 100 --ai-budget-ms 40 --ai-guard-ms 2 \
    --ai-rpc-timeout-ms 35 --seed 20321055 --snr-db -8.5 \
    --channel-seed-base 20355000

wait "$worker0_pid"; worker0_pid=""
wait "$worker1_pid"; worker1_pid=""
wait "$background_pid"; background_pid=""
cleanup
trap - EXIT INT TERM
echo 'confirm55 two-endpoint bring-up completed'
