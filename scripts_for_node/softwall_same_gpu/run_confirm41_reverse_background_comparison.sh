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
if [[ "${SOFTWALL_GPU_LOCK_PREHELD:-0}" == 1 ]]; then
    [[ -d "$gpu_lock_dir" ]] || { echo 'pre-held GPU0 lock is absent' >&2; exit 4; }
else
    mkdir "$gpu_lock_dir" 2>/dev/null || {
        echo "another GPU0 experiment is already running in job $SLURM_JOB_ID" >&2
        exit 4
    }
fi
cleanup() {
    softwall_mps_stop || true
    rm -f "$gpu_lock_dir/owner"
    rmdir "$gpu_lock_dir" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

softwall_mps_stop
sleep 20
softwall_mps_start
softwall_mps_assert

requal="$raw/confirm41_requalification_job${SLURM_JOB_ID}.json"
shifter_gpu env CUDA_MPS_CLIENT_PRIORITY=0 \
    python3 /softwall/s2_runner.py --policy conventional_only \
    --engine /softwall_runtime/engines/neural_rx_fp16_full.trt \
    --output "$requal" --iterations 200 --warmup 20 --cells 1 \
    --period-ms 90 --deadline-ms 80 --nrx-bound-ms 20 \
    --conv-bound-ms 25 --commit-guard-ms 2 --inject-failure-every 0 \
    --seed 20321941 --snr-db -8.5 --channel-seed-base 20331900
python3 - "$requal" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
if len(d['records']) != 200 or d['deadline_misses']:
    raise SystemExit('confirm41 requalification failed')
print('confirm41 operational requalification passed')
PY
sleep 20

# Round 1 uses the same payload/channel seeds; eager first for order reversal.
POLICIES=eager_dual,s2_reserved CELLS=1 PERIOD_MS=90 DEADLINE_MS=80 \
NRX_BOUND_MS=20 CONV_BOUND_MS=25 COMMIT_GUARD_MS=2 FAILURE_EVERY=0 \
SNR_DB=-8.5 SEED_BASE=20321040 CHANNEL_SEED_BASE=20331000 \
CHANNEL_SEED_STRIDE=0 BACKGROUND_CAP=20 AI_BUDGET_MS=40 \
AI_GUARD_MS=2 AI_RPC_TIMEOUT_MS=45 QUIET_SECONDS=20 \
    bash "$SOFTWALL_SCRIPTS/run_s2_campaign.sh" \
    1 10000 confirm41_local_background_baselines

SOFTWALL_GPU_LOCK_HELD=1 ENDPOINT_CAP=80 BACKGROUND_KIND=nrx \
BACKGROUND_CAP=20 BACKGROUND_REPEATS=1 AI_BUDGET_MS=40 AI_GUARD_MS=2 \
AI_RPC_TIMEOUT_MS=45 PERIOD_MS=90 DEADLINE_MS=80 \
ENDPOINT_TIMEOUT_MS=30 WARMUP=20 SEED=20321041 SNR_DB=-8.5 \
CHANNEL_SEED_BASE=20331000 \
    bash "$SOFTWALL_SCRIPTS/run_same_request_ipc.sh" \
    confirm41_external_background_main 10000
sleep 20

shifter_gpu python3 /softwall/analyze_external_background_comparison.py \
    --raw "$raw" --job "$SLURM_JOB_ID" \
    --external-campaign confirm41_external_background_main \
    --local-campaign confirm41_local_background_baselines \
    --protocol "$result/confirm41_reverse_background_comparison_protocol.json" \
    --output "$result/confirm41_external_background_comparison.md"

cleanup
trap - EXIT INT TERM
echo 'confirm41 background comparison completed'
