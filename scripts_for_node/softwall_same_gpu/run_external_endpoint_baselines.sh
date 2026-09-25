#!/usr/bin/env bash

set -euo pipefail

source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/common.sh
source /pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/softwall_same_gpu/mps_runtime.sh

require_allocation
softwall_mps_configure
softwall_mps_assert

campaign=${1:-confirm34_external_endpoint_baselines}
iterations=${ITERATIONS:-10000}
period_ms=${PERIOD_MS:-90}
deadline_ms=${DEADLINE_MS:-80}
snr_db=${SNR_DB:--8.5}
seed=${SEED:-20321033}
channel_seed_base=${CHANNEL_SEED_BASE:-20323000}
quiet_seconds=${QUIET_SECONDS:-20}
raw="$SOFTWALL_ROOT/results/softwall_same_gpu/raw"
mkdir -p "$raw"
gpu_lock_dir="$raw/.lock_gpu0_job${SLURM_JOB_ID}"
owns_gpu_lock=0
if [[ "${SOFTWALL_GPU_LOCK_HELD:-0}" == 1 ]]; then
    [[ -d "$gpu_lock_dir" ]] || {
        echo "external GPU0 lock was declared but is absent" >&2
        exit 4
    }
else
    if ! mkdir "$gpu_lock_dir" 2>/dev/null; then
        echo "another GPU0 experiment is already running in job $SLURM_JOB_ID" >&2
        exit 4
    fi
    owns_gpu_lock=1
fi
cleanup() {
    ((owns_gpu_lock == 0)) || rmdir "$gpu_lock_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for policy in conventional_only eager_dual; do
    output="$raw/${campaign}_${policy}_job${SLURM_JOB_ID}.json"
    if [[ -s "$output" ]]; then
        echo "external baseline resume: keeping $(basename "$output")"
        continue
    fi
    shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
        python3 /softwall/s2_runner.py --policy "$policy" \
        --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
        --output "$output" --iterations "$iterations" --warmup 20 \
        --cells 1 --period-ms "$period_ms" --deadline-ms "$deadline_ms" \
        --nrx-bound-ms 20 --conv-bound-ms 25 --commit-guard-ms 2 \
        --inject-failure-every 0 --seed "$seed" --snr-db "$snr_db" \
        --channel-seed-base "$channel_seed_base"
    sleep "$quiet_seconds"
done

cleanup
trap - EXIT INT TERM
echo "external endpoint baselines completed: $campaign"
